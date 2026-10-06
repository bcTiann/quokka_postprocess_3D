"""Conservation, finite-window loss, branching, and legacy profile parity."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.special import erf

from quokka2s.products.integrated_spectra import (
    LINE_MASSES_AMU,
    IntegratedSpectra,
    accumulate_velocity_spectra,
    check_spectrum_luminosity,
)
from quokka2s.constants import ATOMIC_MASS_UNIT_G, BOLTZMANN_ERG_K, SPEED_OF_LIGHT_KMS
from quokka2s.figures.emission_results import plot_line_spectra
from quokka2s.physics.cell_emission import LineEmission


def make_emission_batch(*, line_keys, velocity, temperature, intrinsic, volume=1., cold=None, attenuated=None):
    """Build named production inputs; optional volume weights belong to the fixture.

    The snapshot API uses one scalar cell volume. Fold any diagnostic weights
    into epsilon so these tests retain their independent luminosity references.
    """
    velocity = np.asarray(velocity, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    intrinsic = np.asarray(intrinsic, dtype=float) * volume
    if attenuated is None:
        attenuated = intrinsic.copy()
    else:
        attenuated = np.asarray(attenuated, dtype=float) * volume
    if cold is None:
        cold = np.zeros(velocity.size, dtype=bool)
    cells = SimpleNamespace(velocity_z_kms=velocity, cell_volume_cm3=1.)
    emission = SimpleNamespace(
        cold_cells=np.asarray(cold, dtype=bool),
        lines={
            key: LineEmission(
                intrinsic_emissivity_erg_s_cm3=intrinsic[row],
                attenuated_emissivity_erg_s_cm3=attenuated[row],
                temperature_K=temperature[row],
            )
            for row, key in enumerate(line_keys)
        },
    )
    return cells, emission


class IntegratedSpectraTests(unittest.TestCase):
    def test_spectrum_conservation_allows_measured_summation_roundoff(self):
        expected = np.array([
            [[1.e36, 0.], [2.e36, 3.e36]],
            [[.5e36, 0.], [1.e36, 1.5e36]],
        ])
        spectrum = expected.copy()
        spectrum[1, 1, 1] *= 1. + 2.e-12
        original_spectrum = spectrum.copy()
        payload = {
            'line_keys': np.array(['cii', 'halpha']),
            'input_luminosity_erg_s': spectrum,
            'captured_luminosity_erg_s': spectrum.copy(),
            'outside_velocity_luminosity_erg_s': np.zeros_like(expected),
            'full_line_luminosity_erg_s': expected.sum(axis=-1),
            'velocity_edges_kms': np.array([0., 1.]),
            'dL_dv_erg_s_per_kms': spectrum[..., None].copy(),
        }
        check_spectrum_luminosity(payload, expected)
        np.testing.assert_array_equal(payload['input_luminosity_erg_s'], original_spectrum)

    def test_spectrum_conservation_rejects_missing_light_with_named_location(self):
        expected = np.ones((2, 2, 2)) * 1.e36
        for field in ('input_luminosity_erg_s', 'captured_luminosity_erg_s'):
            with self.subTest(field=field):
                payload = {
                    'line_keys': np.array(['cii', 'halpha']),
                    'input_luminosity_erg_s': expected.copy(),
                    'captured_luminosity_erg_s': expected.copy(),
                    'outside_velocity_luminosity_erg_s': np.zeros_like(expected),
                    'full_line_luminosity_erg_s': expected.sum(axis=-1),
                    'velocity_edges_kms': np.array([0., 1.]),
                    'dL_dv_erg_s_per_kms': expected[..., None].copy(),
                }
                payload[field][1, 1, 1] *= 1. - 1.e-6
                with self.assertRaisesRegex(
                    ValueError,
                    r'attenuated halpha hot has relative difference 1\.000e-06',
                ):
                    check_spectrum_luminosity(payload, expected)

    def test_spectrum_conservation_requires_expected_zero_to_remain_zero(self):
        expected = np.ones((2, 2, 2)) * 1.e36
        expected[0, 0, 1] = 0.
        for field in ('input_luminosity_erg_s', 'captured_luminosity_erg_s'):
            with self.subTest(field=field):
                payload = {
                    'line_keys': np.array(['cii', 'halpha']),
                    'input_luminosity_erg_s': expected.copy(),
                    'captured_luminosity_erg_s': expected.copy(),
                    'outside_velocity_luminosity_erg_s': np.zeros_like(expected),
                    'full_line_luminosity_erg_s': expected.sum(axis=-1),
                    'velocity_edges_kms': np.array([0., 1.]),
                    'dL_dv_erg_s_per_kms': expected[..., None].copy(),
                }
                payload[field][0, 0, 1] = 1.e-100
                with self.assertRaisesRegex(
                    ValueError,
                    r'intrinsic cii hot has relative difference inf',
                ):
                    check_spectrum_luminosity(payload, expected)

    def test_channel_integral_is_checked_independently_of_saved_totals(self):
        expected = np.full((2, 1, 2), 1.e36)
        profiles = expected[..., None] / 2.
        payload = {
            'line_keys': np.array(['halpha']),
            'input_luminosity_erg_s': expected.copy(),
            'captured_luminosity_erg_s': expected.copy(),
            'outside_velocity_luminosity_erg_s': np.zeros_like(expected),
            'full_line_luminosity_erg_s': expected.sum(axis=-1),
            'velocity_edges_kms': np.array([0., 2.]),
            'dL_dv_erg_s_per_kms': profiles,
        }
        check_spectrum_luminosity(payload, expected)
        profiles[1, 0, 1, 0] *= 0.99
        with self.assertRaisesRegex(
            ValueError,
            'Integrated velocity channels differ from captured luminosity: attenuated halpha hot',
        ):
            check_spectrum_luminosity(payload, expected)

    def test_named_lines_keep_independent_available_cells_and_physical_zeros(self):
        keys = ("halpha", "co10", "ciii_977")
        edges = np.linspace(-100., 100., 401)
        cells = SimpleNamespace(
            velocity_z_kms=np.array([0., 1., 2., -1.]),
            cell_volume_cm3=2.,
        )
        cold_cells = np.array([True, False, False, True])
        lines = {
            "halpha": LineEmission(
                intrinsic_emissivity_erg_s_cm3=np.array([np.nan, 5., np.nan, 2.]),
                attenuated_emissivity_erg_s_cm3=np.array([np.nan, 2.5, np.nan, 1.]),
                temperature_K=np.array([np.nan, 10000., np.nan, 50.]),
            ),
            "co10": LineEmission(
                intrinsic_emissivity_erg_s_cm3=np.array([np.nan, np.nan, 7., 1.]),
                attenuated_emissivity_erg_s_cm3=np.array([np.nan, np.nan, 1.75, .25]),
                temperature_K=np.array([np.nan, np.nan, 100., 50.]),
            ),
            "ciii_977": LineEmission(
                intrinsic_emissivity_erg_s_cm3=np.array([0., 3., np.nan, 0.]),
                attenuated_emissivity_erg_s_cm3=np.array([0., 1.5, np.nan, 0.]),
                temperature_K=np.array([100., 10000., np.nan, 50.]),
            ),
        }
        # Deliberately supply no common valid_cells: each line must select its
        # own finite epsilon, including the prescribed cold CIII zeros.
        emission = SimpleNamespace(lines=lines, cold_cells=cold_cells)
        expected_input = np.array([[4., 10.], [2., 14.], [0., 6.]])
        expected_dust_input = expected_input * np.array([.5, .25, .5])[:, None]
        expected_counts = np.array([[1, 1], [1, 1], [2, 1]])
        original_arrays = {
            key: tuple(values.copy() for values in (
                line.intrinsic_emissivity_erg_s_cm3,
                line.attenuated_emissivity_erg_s_cm3,
                line.temperature_K,
            ))
            for key, line in lines.items()
        }

        for workers in (1, 2):
            with self.subTest(workers=workers):
                spectra = IntegratedSpectra(keys, edges, workers=workers, cell_chunk=2)
                spectra.add_batch(cells, emission)
                payload, _ = spectra.build_output(projected_area_cm2=10.)
                np.testing.assert_array_equal(
                    payload["input_luminosity_erg_s"],
                    np.stack([expected_input, expected_dust_input]),
                )
                np.testing.assert_array_equal(payload["cell_counts_by_regime"], expected_counts)
                np.testing.assert_allclose(
                    payload["captured_luminosity_erg_s"],
                    payload["input_luminosity_erg_s"], rtol=2.e-14,
                )
                # Direct kernel inputs provide a reference for each line's
                # own availability, independently of the batch grouping.
                for line_index, key in enumerate(keys):
                    line = lines[key]
                    available = ~line.emissivity_is_missing
                    for branch_index, branch_cells in enumerate((cold_cells, ~cold_cells)):
                        selected = available & branch_cells
                        velocity = cells.velocity_z_kms[selected]
                        width = np.sqrt(
                            BOLTZMANN_ERG_K * line.temperature_K[selected]
                            / (LINE_MASSES_AMU[key] * ATOMIC_MASS_UNIT_G)
                        ) / 1.e5
                        width *= 1. - velocity / SPEED_OF_LIGHT_KMS
                        luminosity = np.column_stack((
                            line.intrinsic_emissivity_erg_s_cm3[selected] * cells.cell_volume_cm3,
                            line.attenuated_emissivity_erg_s_cm3[selected] * cells.cell_volume_cm3,
                        ))
                        reference = accumulate_velocity_spectra(
                            velocity_kms=velocity,
                            thermal_width_kms=width,
                            luminosity_matrix=luminosity,
                            velocity_edges_kms=edges,
                            workers=workers,
                            cell_chunk=2,
                        )
                        np.testing.assert_array_equal(
                            payload["dL_dv_erg_s_per_kms"][:, line_index, branch_index],
                            reference.T,
                        )
                # Dictionary insertion order must not determine saved line rows.
                reversed_emission = SimpleNamespace(
                    lines=dict(reversed(list(lines.items()))),
                    cold_cells=cold_cells,
                )
                reordered = IntegratedSpectra(keys, edges, workers=workers, cell_chunk=2)
                reordered.add_batch(cells, reversed_emission)
                reordered_payload, _ = reordered.build_output(projected_area_cm2=10.)
                for field in ("line_keys", "dL_dv_erg_s_per_kms", "input_luminosity_erg_s",
                              "cell_counts_by_regime", "line_centroid_window_kms", "line_sigma_window_kms"):
                    np.testing.assert_array_equal(reordered_payload[field], payload[field])

        for key, line in lines.items():
            for actual, original in zip((
                line.intrinsic_emissivity_erg_s_cm3,
                line.attenuated_emissivity_erg_s_cm3,
                line.temperature_K,
            ), original_arrays[key]):
                np.testing.assert_array_equal(actual, original)

    def test_common_availability_matches_direct_gaussian_channel_probabilities(self):
        keys = ("halpha", "hi21", "cii", "co10")
        edges = np.linspace(-40., 40., 161)
        velocity = np.array([-9., -3., 2., 5., 12., -1.])
        volume = np.array([1., 2., 1.5, 3., 2., .75])
        cold = np.array([True, True, False, False, True, False])
        epsilon = np.arange(24., dtype=float).reshape(4, 6)
        epsilon[:, 2] = np.nan
        temperature = np.full_like(epsilon, 100.)
        temperature[3] = 25.
        temperature[:, 2] = np.nan
        dust_epsilon = epsilon * np.array([1., 0., .4, 1., .2, .8])
        cells, emission = make_emission_batch(
            line_keys=keys,
            velocity=velocity,
            temperature=temperature,
            intrinsic=epsilon,
            attenuated=dust_epsilon,
            volume=volume,
            cold=cold,
        )
        spectra = IntegratedSpectra(keys, edges, cell_chunk=2, workers=2)
        spectra.add_batch(cells=cells, emission=emission)
        payload, _ = spectra.build_output(projected_area_cm2=10.)
        for line_index, key in enumerate(keys):
            line = emission.lines[key]
            available = ~line.emissivity_is_missing
            for branch_index, branch_cells in enumerate((cold, ~cold)):
                selected = available & branch_cells
                width = np.sqrt(
                    BOLTZMANN_ERG_K * temperature[line_index, selected]
                    / (LINE_MASSES_AMU[key] * ATOMIC_MASS_UNIT_G)
                ) / 1.e5
                width *= 1. - velocity[selected] / SPEED_OF_LIGHT_KMS
                standardized_edges = (edges[:, None] - velocity[selected]) / (np.sqrt(2.) * width)
                fractions = .5 * np.diff(erf(standardized_edges), axis=0)
                for dust_index, luminosity in enumerate((epsilon, dust_epsilon)):
                    expected = fractions @ (luminosity[line_index, selected] * volume[selected])
                    expected /= np.diff(edges)
                    np.testing.assert_allclose(
                        payload["dL_dv_erg_s_per_kms"][dust_index, line_index, branch_index],
                        expected,
                        rtol=2.e-14,
                        atol=2.e-14,
                    )

    def test_different_availability_groups_validate_before_main_sums_change(self):
        keys = ("halpha", "co10")
        spectra = IntegratedSpectra(keys, np.linspace(-20., 20., 81))
        emission = SimpleNamespace(
            cold_cells=np.array([True, False]),
            lines={
                "halpha": LineEmission(
                    np.array([2., 1.]), np.array([1., .5]), np.array([50., 10000.]),
                ),
                "co10": LineEmission(
                    np.array([np.nan, 3.]), np.array([np.nan, -1.]), np.array([np.nan, 100.]),
                ),
            },
        )
        cells = SimpleNamespace(velocity_z_kms=np.array([0., 1.]), cell_volume_cm3=2.)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            spectra.add_batch(cells, emission)
        np.testing.assert_array_equal(spectra.dL_dv, 0.)
        np.testing.assert_array_equal(spectra.input_luminosity, 0.)
        np.testing.assert_array_equal(spectra.cell_counts, 0)

    def test_fully_missing_line_does_not_discard_an_available_line(self):
        spectra = IntegratedSpectra(("co10", "halpha"), np.linspace(-30., 30., 121))
        cells = SimpleNamespace(velocity_z_kms=np.array([0., 1.]), cell_volume_cm3=2.)
        emission = SimpleNamespace(
            cold_cells=np.array([True, False]),
            lines={
                "co10": LineEmission(
                    np.full(2, np.nan), np.full(2, np.nan), np.full(2, np.nan),
                ),
                "halpha": LineEmission(
                    np.array([2., 3.]), np.array([1., 1.5]), np.array([50., 100.]),
                ),
            },
        )
        spectra.add_batch(cells, emission)
        payload, _ = spectra.build_output(projected_area_cm2=10.)
        np.testing.assert_array_equal(payload["cell_counts_by_regime"], [[0, 0], [1, 1]])
        np.testing.assert_array_equal(payload["input_luminosity_erg_s"][0], [[0., 0.], [4., 6.]])
        np.testing.assert_array_equal(payload["dL_dv_erg_s_per_kms"][:, 0], 0.)
        self.assertTrue(np.isnan(payload["line_sigma_window_kms"][:, 0]).all())

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
        cells, emission = make_emission_batch(
            line_keys=keys, velocity=velocity, temperature=temperature,
            intrinsic=epsilon, volume=volume, cold=cold,
        )
        full = IntegratedSpectra(keys, edges, cell_chunk=2)
        full.add_batch(cells=cells, emission=emission)
        payload, report = full.build_output(projected_area_cm2=10.)
        expected = np.column_stack((
            (epsilon[:, cold] * volume[cold]).sum(axis=1),
            (epsilon[:, ~cold] * volume[~cold]).sum(axis=1),
        ))
        np.testing.assert_allclose(payload["input_luminosity_erg_s"], np.stack((expected, expected)))
        np.testing.assert_allclose(payload["captured_luminosity_erg_s"], np.stack((expected, expected)), rtol=2e-14)
        np.testing.assert_allclose(payload["outside_velocity_luminosity_erg_s"], 0., atol=2e-12)
        np.testing.assert_array_equal(payload["cell_counts_by_regime"], np.tile([2, 2], (len(keys), 1)))
        self.assertEqual(report["intrinsic"]["lines"]["co21"]["total"]["outside_velocity_fraction"], 0.)
        stream = IntegratedSpectra(keys, edges, cell_chunk=2, workers=2)
        for part in (slice(0, 1), slice(1, 4)):
            cells, emission = make_emission_batch(
                line_keys=keys, velocity=velocity[part], temperature=temperature[:, part],
                intrinsic=epsilon[:, part], volume=volume[part], cold=cold[part],
            )
            stream.add_batch(cells=cells, emission=emission)
        streamed, _ = stream.build_output(projected_area_cm2=10.)
        np.testing.assert_allclose(streamed["dL_dv_erg_s_per_kms"], payload["dL_dv_erg_s_per_kms"], rtol=2e-14)

    def test_boundary_profile_loses_half_without_renormalization(self):
        accumulator = IntegratedSpectra(("hi21",), np.linspace(-50., 0., 101))
        cells, emission = make_emission_batch(
            line_keys=("hi21",), velocity=[0.], temperature=[[100.]],
            intrinsic=[[4.]], volume=3., cold=[True],
        )
        accumulator.add_batch(cells=cells, emission=emission)
        payload, report = accumulator.build_output(projected_area_cm2=10.)
        self.assertAlmostEqual(payload["input_luminosity_erg_s"][0, 0, 0], 12.)
        self.assertAlmostEqual(payload["captured_luminosity_erg_s"][0, 0, 0], 6., places=12)
        self.assertAlmostEqual(report["intrinsic"]["lines"]["hi21"]["total"]["outside_velocity_fraction"], .5)

    def test_thermal_mass_and_doppler_factor_match_gaussian_probability(self):
        keys = ("halpha", "cii", "co10")
        # A high diagnostic velocity makes omission of the existing 1-v/c
        # factor measurable; this is a numerical convention test, not a
        # proposed relativistic spectrum model for the simulation.
        velocity = 30000.
        temperature = 1.e7
        edges = np.array([velocity - 5., velocity, velocity + 5.])
        accumulator = IntegratedSpectra(keys, edges)
        cells, emission = make_emission_batch(
            line_keys=keys, velocity=[velocity], temperature=np.full((3, 1), temperature),
            intrinsic=np.ones((3, 1)), volume=2.,
        )
        accumulator.add_batch(cells=cells, emission=emission)
        payload, _ = accumulator.build_output(projected_area_cm2=10.)
        line_mass = np.array([LINE_MASSES_AMU[key] for key in keys]) * const.u
        sigma = np.sqrt(const.k_B * temperature * u.K / line_mass).to(u.km / u.s)
        cell_velocity = velocity * u.km / u.s
        sigma *= 1. - (cell_velocity / const.c).to_value(u.dimensionless_unscaled)
        edge_offset = edges[:, None] * u.km / u.s - cell_velocity
        standardized_edges = (edge_offset / (np.sqrt(2.) * sigma)).to_value(
            u.dimensionless_unscaled)
        channel_fractions = (0.5 * np.diff(erf(standardized_edges), axis=0)).T
        expected_profile = 2. * channel_fractions / np.diff(edges)
        expected_capture = np.sum(expected_profile * np.diff(edges), axis=-1)
        np.testing.assert_allclose(payload["dL_dv_erg_s_per_kms"][0, :, 1], expected_profile,
                                   rtol=3.e-14)
        np.testing.assert_allclose(payload["captured_luminosity_erg_s"][0, :, 1], expected_capture,
                                   rtol=3.e-14)
        self.assertGreater(expected_capture[2], expected_capture[0])

    def test_unselected_and_zero_emission_cells_do_not_contribute(self):
        accumulator = IntegratedSpectra(("cii",), [-1., 0., 1.])
        cells, emission = make_emission_batch(
            line_keys=("cii",), velocity=[0., 0., 100.], temperature=[[np.nan, 0., 0.]],
            intrinsic=[[np.nan, 7., 0.]], volume=2., cold=[True, True, False],
        )
        accumulator.add_batch(cells=cells, emission=emission)
        payload, _ = accumulator.build_output(projected_area_cm2=10.)
        np.testing.assert_allclose(payload["input_luminosity_erg_s"][0], [[14., 0.]])
        np.testing.assert_allclose(payload["captured_luminosity_erg_s"][0], [[14., 0.]])
        np.testing.assert_allclose(payload["dL_dv_erg_s_per_kms"][0, 0, 0], [7., 7.])
        np.testing.assert_array_equal(payload["cell_counts_by_regime"], [[1, 1]])

    def test_paired_dust_states_match_independent_kernel_columns_with_different_zero_support(self):
        keys = ("halpha", "hi21", "cii", "co10")
        edges = np.linspace(-30., 30., 121)
        velocity = np.array([-9., -3., 2., 5., 12., -1., 15., 0.])
        temperature = np.full((len(keys), velocity.size), 100.)
        temperature[3] = 25.
        intrinsic = np.array([
            [2., 0., 1., 3., 0., 4., 2., 0.],
            [1., 2., 0., 1., 4., 0., 1., 2.],
            [0., 3., 2., 0., 1., 2., 0., 3.],
            [1., 0., 3., 2., 0., 1., 4., 0.],
        ])
        attenuated = intrinsic * np.array([1., 0., .4, 1., 0., .2, .8, 0.])
        # One diagnostic column emits only after attenuation to exercise
        # different kernel-column support; this is not a physical dust model.
        attenuated[2, 0] = .5
        volume = np.array([1., 2., 1.5, 3., 2., .75, 4., 2.5])
        cold = np.array([True, True, False, False, True, False, True, False])
        selected = np.array([True, True, True, False, True, True, True, True])
        intrinsic[:, ~selected] = np.nan
        attenuated[:, ~selected] = np.nan
        temperature[:, ~selected] = np.nan
        for workers in (1, 2):
            spectra = IntegratedSpectra(keys, edges, cell_chunk=2, workers=workers)
            reference = np.zeros_like(spectra.dL_dv)
            for part in (slice(0, 4), slice(4, 8)):
                cells, emission = make_emission_batch(
                    line_keys=keys, velocity=velocity[part], temperature=temperature[:, part],
                    intrinsic=intrinsic[:, part], attenuated=attenuated[:, part],
                    volume=volume[part], cold=cold[part],
                )
                spectra.add_batch(cells=cells, emission=emission)
                for line_index, key in enumerate(keys):
                    for branch_index, branch_cells in enumerate((cold, ~cold)):
                        retained = np.flatnonzero(selected[part] & branch_cells[part]) + part.start
                        width = np.sqrt(
                            BOLTZMANN_ERG_K * temperature[line_index, retained]
                            / (LINE_MASSES_AMU[key] * ATOMIC_MASS_UNIT_G)
                        ) / 1.e5
                        width *= 1. - velocity[retained] / SPEED_OF_LIGHT_KMS
                        for dust_index, epsilon in enumerate((intrinsic, attenuated)):
                            profile = accumulate_velocity_spectra(
                                velocity_kms=velocity[retained], thermal_width_kms=width,
                                luminosity_matrix=(epsilon[line_index, retained] * volume[retained])[:, None],
                                velocity_edges_kms=edges, workers=workers, cell_chunk=2,
                            )
                            reference[dust_index, line_index, branch_index] += profile[:, 0]
            np.testing.assert_allclose(spectra.dL_dv, reference, rtol=2e-14, atol=2e-14)

    def test_paired_accumulation_validates_both_inputs_before_mutation(self):
        accumulator = IntegratedSpectra(("cii",), [-1., 0., 1.], cell_chunk=2)
        cells, emission = make_emission_batch(
            line_keys=("cii",), velocity=[0.], temperature=[[10.]],
            intrinsic=[[1.]], attenuated=[[-1.]], cold=[True],
        )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            accumulator.add_batch(cells=cells, emission=emission)
        np.testing.assert_array_equal(accumulator.cell_counts, [[0, 0]])
        np.testing.assert_array_equal(accumulator.dL_dv, 0.)

    def test_paired_accumulation_evaluates_shared_kernel_once_per_group(self):
        keys = ("halpha", "hi21")
        spectra = IntegratedSpectra(keys, np.linspace(-10., 10., 41), cell_chunk=2)
        cells, emission = make_emission_batch(
            line_keys=keys, velocity=[-1., 1.], temperature=[[100., 100.], [100., 100.]],
            intrinsic=[[1., 2.], [3., 4.]], attenuated=[[.5, 1.], [1.5, 2.]],
            cold=[True, False],
        )
        with patch("quokka2s.products.integrated_spectra.accumulate_velocity_spectra",
                   wraps=accumulate_velocity_spectra) as kernel:
            spectra.add_batch(cells=cells, emission=emission)
        self.assertEqual(kernel.call_count, 2)  # One H-mass group in each regime.

    def test_invalid_inputs_fail_before_mutating_accumulator(self):
        bad = (
            dict(velocity=[np.nan]), dict(velocity=[SPEED_OF_LIGHT_KMS]),
            dict(temperature=[[np.nan]]), dict(temperature=[[-1.]]),
            dict(intrinsic=[[-1.]]), dict(attenuated=[[np.inf]]),
        )
        for change in bad:
            accumulator = IntegratedSpectra(("cii",), [-1., 0., 1.])
            inputs = dict(line_keys=("cii",), velocity=[0.], temperature=[[10.]], intrinsic=[[1.]])
            inputs.update(change)
            cells, emission = make_emission_batch(**inputs)
            with self.assertRaises(ValueError):
                accumulator.add_batch(cells=cells, emission=emission)
            np.testing.assert_array_equal(accumulator.cell_counts, [[0, 0]])
        for volume in (0., np.inf):
            accumulator = IntegratedSpectra(("cii",), [-1., 0., 1.])
            cells, emission = make_emission_batch(
                line_keys=("cii",), velocity=[0.], temperature=[[10.]], intrinsic=[[1.]],
            )
            cells.cell_volume_cm3 = volume
            with self.assertRaisesRegex(ValueError, "volume"):
                accumulator.add_batch(cells=cells, emission=emission)
            np.testing.assert_array_equal(accumulator.cell_counts, [[0, 0]])

    def test_kernel_matches_existing_script_for_narrow_and_broad_lines(self):
        # Frozen outputs from the pre-refactor kernel, including its summation order.
        # The fixture also records that source file's SHA256 for reproducibility.
        reference_path = Path(__file__).parent / "data/velocity_kernel_reference.npz"
        with np.load(reference_path, allow_pickle=False) as reference:
            expected_by_workers = {1: reference["serial"], 2: reference["parallel"]}
        velocity = np.array([-51., -5., 0., 2., 51.])
        thermal = np.array([10., 1.e-8, 2., 100., 1.])
        luminosity = np.array([[1., 2.], [3., 4.], [0., 0.], [5., 6.], [7., 8.]])
        edges = np.linspace(-50., 50., 301)
        for workers in (1, 2):
            options = dict(cell_chunk=2, workers=workers)
            expected = expected_by_workers[workers]
            actual = accumulate_velocity_spectra(velocity, thermal, luminosity, edges, **options)
            np.testing.assert_array_equal(actual, expected)

    def test_payload_serializes_and_plot_writes_both_formats(self):
        keys = ("cii", "halpha", "co10")
        accumulator = IntegratedSpectra(keys, np.linspace(-20., 20., 101))
        cells, emission = make_emission_batch(
            line_keys=keys, velocity=[0.], temperature=[[100.], [100.], [10.]],
            intrinsic=[[1.], [2.], [0.]],
        )
        accumulator.add_batch(cells=cells, emission=emission)
        payload, _ = accumulator.build_output(projected_area_cm2=10.)
        with tempfile.TemporaryDirectory() as tmp:
            np.savez_compressed(Path(tmp) / "spectra.npz", **payload)
            with np.load(Path(tmp) / "spectra.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["line_keys"], keys)
            paths = plot_line_spectra(
                keys=keys,
                spectra=payload,
                output=Path(tmp),
                per_projected_area=False,
                diagnostic_suffix="",
                titled=False,
                formats=("png", "pdf"),
            )
            self.assertEqual(len(paths), 2 * len(keys))
            for path in paths:
                self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
