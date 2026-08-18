import unittest
from unittest.mock import Mock
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
from yt.units.yt_array import YTQuantity

from quokka2s.pipeline.prep.physics_fields import (
    build_integrated_spectrum,
    build_spectral_cube,
)
from quokka2s.pipeline.cache import (
    CACHED_FIELDS,
    load_results_dict,
    save_results_dict,
)
from quokka2s.pipeline.spectrum_units import (
    DSIGMA_DV_UNIT,
    SPECTRAL_VELOCITY_UNIT,
    SURFACE_BRIGHTNESS_UNIT,
    convert_dsigma_dnu_to_dsigma_dv,
    dsigma_dv_ylabel,
    unit_latex,
)
from quokka2s.pipeline.tasks.integrated_spectrum import SPECIES_CFG
from quokka2s.pipeline.services.spectrum_service import SpectrumStore


class IntegratedSpectrumTests(unittest.TestCase):
    def test_frequency_to_velocity_jacobian_is_unit_checked(self):
        dsigma_dnu = np.array([1.0, 2.0])
        at_nu = convert_dsigma_dnu_to_dsigma_dv(dsigma_dnu, 1.0e9)
        at_2nu = convert_dsigma_dnu_to_dsigma_dv(dsigma_dnu, 2.0e9)

        self.assertEqual(at_nu.units, YTQuantity(1.0, DSIGMA_DV_UNIT).units)
        np.testing.assert_allclose(at_2nu, 2.0 * at_nu, rtol=1.0e-15)

    def test_spectrum_axis_label_uses_yt_latex_unit(self):
        label = dsigma_dv_ylabel(DSIGMA_DV_UNIT)
        self.assertIn(unit_latex("erg"), label)
        self.assertIn(unit_latex("s"), label)
        self.assertIn(unit_latex("cm"), label)
        self.assertIn(unit_latex(SPECTRAL_VELOCITY_UNIT), label)

    def test_task_cache_restores_spectrum_units(self):
        spectrum = convert_dsigma_dnu_to_dsigma_dv(np.ones(2), 1.0e9)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'spectrum.h5'
            save_results_dict(path, {'spectrum': spectrum}, cache_key='test')
            restored = load_results_dict(path, expected_cache_key='test')

        self.assertIsNotNone(restored)
        self.assertEqual(restored['spectrum'].units, spectrum.units)
        np.testing.assert_allclose(restored['spectrum'], spectrum)

    def test_co_ladder_uses_explicit_result_names_and_legacy_co10_fields(self):
        by_name = {entry['name']: entry for entry in SPECIES_CFG}
        self.assertEqual(by_name['CO10']['lum_field'], 'CO_luminosity')
        self.assertEqual(by_name['CO10']['freq_field'], 'CO_freq')
        self.assertEqual(by_name['CO21']['lum_field'], 'CO21_luminosity')
        self.assertEqual(by_name['CO21']['freq_field'], 'CO21_freq')
        self.assertIn(('gas', 'CO21_luminosity'), CACHED_FIELDS)

    def test_spectrum_store_keeps_legacy_co_alias(self):
        store = SpectrumStore(provider=None)
        expected = (np.array([0.0, 1.0]), np.array([2.0, 3.0]))
        store._build = Mock(return_value=expected)
        actual = store.get_spectrum('CO', 'x')
        store._build.assert_called_once_with('CO10', 'x', 'total')
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_direct_accumulation_matches_spatial_cube_sum(self):
        rng = np.random.default_rng(20260720)
        shape = (4, 3, 5)
        c_cms = 2.99792458e10
        nu_0 = 1.4204057518e9

        velocity = rng.uniform(-2.0e6, 2.0e6, size=shape)
        shifted = nu_0 * (1.0 - velocity / c_cms)
        luminosity = 10.0 ** rng.uniform(25.0, 30.0, size=shape)
        thermal_width = 10.0 ** rng.uniform(4.0, 6.5, size=shape)
        bandwidth = nu_0 * (5.0e6 / c_cms) * 2.0
        edges = np.linspace(
            nu_0 - bandwidth / 2.0,
            nu_0 + bandwidth / 2.0,
            308,
        )

        cube = build_spectral_cube(
            shifted, luminosity, thermal_width, edges, c_cms,
        )
        direct = build_integrated_spectrum(
            shifted,
            luminosity,
            thermal_width,
            edges,
            c_cms,
            cell_chunk=7,
        )

        np.testing.assert_allclose(
            direct,
            cube.sum(axis=(1, 2)),
            rtol=2.0e-14,
            atol=0.0,
        )

    def test_direct_accumulation_rejects_mismatched_shapes(self):
        edges = np.linspace(1.0, 2.0, 5)
        with self.assertRaises(ValueError):
            build_integrated_spectrum(
                np.ones((2, 2, 2)),
                np.ones((2, 2, 1)),
                np.ones((2, 2, 2)),
                edges,
                3.0e10,
            )

    def test_direct_accumulation_discards_exact_zero_cells_only(self):
        shifted = np.array([1.0, 1.1, 1.2, 1.3])
        luminosity = np.array([0.0, 2.0, 0.0, -0.5])
        thermal = np.full(4, 1.0e8)
        edges = np.linspace(0.8, 1.5, 9)
        full = build_integrated_spectrum(
            shifted, luminosity, thermal, edges, 3.0e10,
        )
        emitting = luminosity != 0.0
        compact = build_integrated_spectrum(
            shifted[emitting], luminosity[emitting], thermal[emitting],
            edges, 3.0e10,
        )

        np.testing.assert_array_equal(full, compact)


if __name__ == '__main__':
    unittest.main()
