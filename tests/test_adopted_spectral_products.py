"""Conservation, finite-window loss, branching, and legacy profile parity."""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import erf

from quokka2s.adopted_spectral_products import (
    AMU_G, LINE_MASSES_AMU, SPEED_OF_LIGHT_KMS, AdoptedSpectralAccumulator,
    accumulate_velocity_spectra, plot_adopted_spectra,
)

KB = 1.3806488000000003e-16


class AdoptedSpectralProductTests(unittest.TestCase):
    def test_luminosity_conservation_and_streaming_regime_split(self):
        keys = ("halpha", "hi21", "cii", "co10", "co21")
        edges = np.linspace(-100., 100., 301)
        velocity = np.array([-20., -1., 3., 15.])
        temperature = np.full((len(keys), 4), 30.)
        temperature[3:] = 10.
        epsilon = np.arange(20.).reshape(5, 4)
        epsilon[4] = 0.
        volume = np.array([1., 2., 3., 4.])
        cold = np.array([True, False, True, False])
        full = AdoptedSpectralAccumulator(keys, edges, KB, cell_chunk=2)
        full.add(velocity, temperature, epsilon, volume, cold_mask=cold)
        payload, report = full.finalize()
        expected = np.column_stack(((epsilon[:, cold] * volume[cold]).sum(axis=1),
                                    (epsilon[:, ~cold] * volume[~cold]).sum(axis=1)))
        np.testing.assert_allclose(payload["input_luminosity_erg_s"], expected)
        np.testing.assert_allclose(payload["captured_luminosity_erg_s"], expected, rtol=2e-14)
        np.testing.assert_allclose(payload["outside_velocity_fraction"], 0., atol=2e-14)
        np.testing.assert_array_equal(payload["cell_counts_by_regime"], [2, 2])
        self.assertEqual(report["lines"]["co21"]["total"]["outside_velocity_fraction"], 0.)
        stream = AdoptedSpectralAccumulator(keys, edges, KB, cell_chunk=2, workers=2)
        for sl in (slice(0, 1), slice(1, 4)):
            stream.add(velocity[sl], temperature[:, sl], epsilon[:, sl], volume[sl],
                       cold_mask=cold[sl])
        np.testing.assert_allclose(stream.finalize()[0]["dL_dv_erg_s_per_kms"],
                                   payload["dL_dv_erg_s_per_kms"], rtol=2e-14)

    def test_boundary_profile_loses_half_without_renormalization(self):
        accumulator = AdoptedSpectralAccumulator(("hi21",), np.linspace(-50., 0., 101), KB)
        accumulator.add([0.], [[100.]], [[4.]], 3., cold_mask=[True])
        payload, report = accumulator.finalize()
        self.assertAlmostEqual(payload["input_luminosity_erg_s"][0, 0], 12.)
        self.assertAlmostEqual(payload["captured_luminosity_erg_s"][0, 0], 6., places=12)
        self.assertAlmostEqual(report["lines"]["hi21"]["total"]["outside_velocity_fraction"], .5)

    def test_thermal_mass_and_doppler_factor_match_gaussian_probability(self):
        keys = ("halpha", "cii", "co10")
        # A high diagnostic velocity makes omission of the existing 1-v/c
        # factor measurable; this is a numerical convention test, not a
        # proposed relativistic spectrum model for the simulation.
        velocity = 30000.
        temperature = 1.e7
        edges = np.array([velocity - 5., velocity, velocity + 5.])
        accumulator = AdoptedSpectralAccumulator(keys, edges, KB)
        accumulator.add([velocity], np.full((3, 1), temperature), np.ones((3, 1)), 2.)
        payload, _ = accumulator.finalize()
        sigma = np.sqrt(KB * temperature / (np.array([LINE_MASSES_AMU[key] for key in keys]) * AMU_G)) / 1.e5
        sigma *= 1. - velocity / SPEED_OF_LIGHT_KMS
        expected_capture = 2. * erf(5. / (np.sqrt(2.) * sigma))
        np.testing.assert_allclose(payload["captured_luminosity_erg_s"][:, 1], expected_capture,
                                   rtol=3.e-14)
        self.assertGreater(expected_capture[2], expected_capture[0])

    def test_unselected_and_zero_emission_cells_do_not_contribute(self):
        accumulator = AdoptedSpectralAccumulator(("cii",), [-1., 0., 1.], KB)
        accumulator.add([0., 0., 100.], [[0., 0., 0.]], [[5., 7., 0.]], 2.,
                        selected_mask=[False, True, True], cold_mask=[True, True, False])
        payload, _ = accumulator.finalize()
        np.testing.assert_allclose(payload["input_luminosity_erg_s"], [[14., 0.]])
        np.testing.assert_allclose(payload["captured_luminosity_erg_s"], [[14., 0.]])
        np.testing.assert_allclose(payload["dL_dv_erg_s_per_kms"][0, 0], [7., 7.])
        np.testing.assert_array_equal(payload["cell_counts_by_regime"], [1, 1])

    def test_invalid_inputs_fail_before_mutating_accumulator(self):
        bad = (
            dict(velocity_kms=[np.nan]), dict(velocity_kms=[SPEED_OF_LIGHT_KMS]),
            dict(temperatures_K=[[np.nan]]), dict(temperatures_K=[[-1.]]),
            dict(emissivity=[[-1.]]), dict(emissivity=[[np.inf]]),
            dict(cell_volume_cm3=0.), dict(cell_volume_cm3=[1., 2.]),
            dict(selected_mask=[1]), dict(cold_mask=[1]), dict(emissivity=[1.]),
        )
        for change in bad:
            with self.subTest(change=change):
                accumulator = AdoptedSpectralAccumulator(("cii",), [-1., 0., 1.], KB)
                inputs = dict(velocity_kms=[0.], temperatures_K=[[10.]], emissivity=[[1.]], cell_volume_cm3=1.)
                inputs.update(change)
                with self.assertRaises(ValueError):
                    accumulator.add(**inputs)
                np.testing.assert_array_equal(accumulator.finalize()[0]["cell_counts_by_regime"], [0, 0])
        with self.assertRaisesRegex(ValueError, "uniform"):
            AdoptedSpectralAccumulator(("cii",), [-1., 0., 2.], KB)
        with self.assertRaisesRegex(ValueError, "thermal mass"):
            AdoptedSpectralAccumulator(("unknown",), [-1., 0., 1.], KB)

    def test_kernel_matches_existing_script_for_narrow_and_broad_lines(self):
        source = Path(__file__).resolve().parents[1] / "scripts/plot_cloudy_line_physics_ablation_spectra.py"
        node = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "accumulate_velocity_spectra")
        namespace = {"np": np, "scipy_erf": erf, "ThreadPoolExecutor": ThreadPoolExecutor}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        old_kernel = namespace["accumulate_velocity_spectra"]
        velocity = np.array([-51., -5., 0., 2., 51.])
        thermal = np.array([10., 1.e-8, 2., 100., 1.])
        luminosity = np.array([[1., 2.], [3., 4.], [0., 0.], [5., 6.], [7., 8.]])
        edges = np.linspace(-50., 50., 301)
        for workers in (1, 2):
            options = dict(cell_chunk=2, workers=workers)
            expected = old_kernel(velocity, thermal, luminosity, edges, **options)
            actual = accumulate_velocity_spectra(velocity, thermal, luminosity, edges, **options)
            np.testing.assert_array_equal(actual, expected)

    def test_payload_serializes_and_plot_writes_both_formats(self):
        accumulator = AdoptedSpectralAccumulator(("cii", "halpha", "co10"), np.linspace(-20., 20., 101), KB)
        accumulator.add([0.], [[100.], [100.], [10.]], [[1.], [2.], [0.]], 1.)
        payload, _ = accumulator.finalize()
        with tempfile.TemporaryDirectory() as tmp:
            np.savez_compressed(Path(tmp) / "spectra.npz", **payload)
            with np.load(Path(tmp) / "spectra.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["line_keys"], ["cii", "halpha", "co10"])
            paths = plot_adopted_spectra(payload, Path(tmp) / "spectra", projected_area_cm2=10.)
            for path in paths.values():
                self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
