"""Numerical-product and relocated-input checks for the process command."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from quokka2s.cloudy.lookup import CloudyFailureTouchError
from quokka2s.constants import HYDROGEN_MASS_G
from quokka2s.physics.settings import X_H
from quokka2s.products.gas_phase_velocity import GasPhaseVelocityAccumulator, check_phase_accounting
from quokka2s.cloudy.cell_fields import CloudyCellReader
from quokka2s.products.line_luminosity_images import check_image_luminosity
from quokka2s.products.integrated_spectra import IntegratedSpectra
from quokka2s.physics.cell_emission import LineEmission
from quokka2s.snapshot_reader import CellBatch, slab_windows
from quokka2s.result_files import add_dust_metadata


class EmissionProcessingTests(unittest.TestCase):
    def test_dust_metadata_uses_output_line_names_not_dictionary_insertion_order(self):
        emission_calculator = SimpleNamespace(
            line_keys=('cii', 'halpha'),
            dust_cross_section_cm2_H={'halpha': 3., 'cii': 2.},
        )
        image = {}
        spectrum = {}
        add_dust_metadata(
            emission_calculator=emission_calculator,
            image_payload=image,
            spectrum_payload=spectrum,
        )
        for payload in (image, spectrum):
            np.testing.assert_array_equal(payload['dust_sigma_ext_cm2_H'], [2., 3.])
        self.assertIsNot(image['dust_sigma_ext_cm2_H'], spectrum['dust_sigma_ext_cm2_H'])

    def test_phase_product_uses_retained_cells_mixed_temperature_and_full_velocity_width(self):
        phases = GasPhaseVelocityAccumulator(np.linspace(-200., 200., 401))
        cells = CellBatch(
            temperature_QUOKKA_K=np.array([50., 3500., 2500., 1e6]),
            velocity_z_kms=np.array([0., 50., 10., 500.]),
            density_g_cm3=np.array([1., 2., 3., 4.]),
            shielding_NH_cm2=np.zeros(4),
            foreground_NH_cm2=np.zeros(4),
            velocity_gradient_s=np.zeros(4),
            x_start=0,
            slab_shape=(1, 1, 4),
            batch_start=0,
            cell_volume_cm3=1.,
        )
        emission = SimpleNamespace(
            cold_cells=np.array([True, False, True, False]),
            despotic_temperature_K=np.array([100., np.nan, np.nan, np.nan]))
        phases.add_batch(cells, emission)
        payload, report = phases.build_output()
        check_phase_accounting(report['groups'], payload['histogram_mass_g'],
                               gas_cell_count=3, gas_mass_g=7.)
        self.assertEqual(tuple(payload['phase_keys']),
                         ('CNM', 'UNM', 'WNM', 'WIM', 'HIM', 'total'))
        np.testing.assert_array_equal(payload['group_count'], [1, 0, 1, 0, 1, 3])
        np.testing.assert_array_equal(payload['group_mass_g'], [1., 0., 2., 0., 4., 7.])
        self.assertEqual(payload['histogram_mass_g'].shape, (6, 400))
        self.assertEqual(payload['histogram_mass_g'][-1].sum(), 3.)
        self.assertEqual(report['groups']['HIM']['above_window']['mass_g'], 4.)
        self.assertEqual(payload['phase_temperature_method'].item(), 'mixed')
        self.assertEqual(payload['bundle_source'].item(), 'process')
        self.assertTrue(np.isnan(payload['sigma_internal_kms'][1]))
        self.assertGreater(payload['sigma_internal_kms'][-1], 0.)
        with self.assertRaisesRegex(ValueError, 'Phase cell counts'):
            check_phase_accounting(report['groups'], payload['histogram_mass_g'],
                                   gas_cell_count=4, gas_mass_g=7.)
        with self.assertRaisesRegex(ValueError, 'Phase masses'):
            check_phase_accounting(report['groups'], payload['histogram_mass_g'],
                                   gas_cell_count=3, gas_mass_g=8.)

    def test_image_conservation_allows_roundoff_but_rejects_missing_light(self):
        cells = np.array([[1e36, 0.], [1e36, 0.]])
        image = cells.copy()
        image[1, 0] *= 1 + 2e-12
        report = check_image_luminosity(image, cells, ('cii', 'halpha'))
        self.assertEqual(report['worst_dust_state'], 'attenuated')
        self.assertEqual(report['worst_line'], 'cii')

        image[1, 0] *= 1 - 1e-6
        with self.assertRaisesRegex(ValueError, 'attenuated cii has relative difference'):
            check_image_luminosity(image, cells, ('cii', 'halpha'))

        image = cells.copy()
        image[0, 1] = 1e-100
        with self.assertRaisesRegex(ValueError, 'intrinsic halpha has relative difference'):
            check_image_luminosity(image, cells, ('cii', 'halpha'))

    def test_unexpected_lookup_errors_propagate(self):
        cells = CellBatch(
            density_g_cm3=np.array([HYDROGEN_MASS_G / X_H]),
            temperature_QUOKKA_K=np.array([1e4]),
            shielding_NH_cm2=np.array([1e19]),
            foreground_NH_cm2=np.zeros(1),
            velocity_gradient_s=np.zeros(1),
            velocity_z_kms=np.zeros(1),
            x_start=0,
            slab_shape=(1, 1, 1),
            batch_start=0,
            cell_volume_cm3=1.,
        )
        for error in (ValueError('invalid table domain'), CloudyFailureTouchError('unexpected')):
            lookup = SimpleNamespace()
            with patch.object(lookup, 'interpolate_available', create=True, side_effect=error):
                with self.assertRaises(type(error)):
                    reader = CloudyCellReader(lookup=lookup)
                    reader.read_fields(
                        cells=cells,
                        selected_cells=np.array([True]),
                    )

    def test_two_dust_states_use_saved_channel_luminosity(self):
        spectra = IntegratedSpectra(("cii",), np.array([-2., 0., 2.]))
        cells = SimpleNamespace(velocity_z_kms=np.array([-1., 1.]), cell_volume_cm3=1.)
        emission = SimpleNamespace(
            cold_cells=np.array([True, False]),
            lines={"cii": LineEmission(
                intrinsic_emissivity_erg_s_cm3=np.array([2., 2.]),
                attenuated_emissivity_erg_s_cm3=np.array([1., 1.]),
                temperature_K=np.array([0., 0.]),
            )},
        )
        spectra.add_batch(cells=cells, emission=emission)
        result, _ = spectra.build_output(projected_area_cm2=10.)
        self.assertEqual(result['dL_dv_erg_s_per_kms'].shape, (2, 1, 2, 2))
        np.testing.assert_allclose(result['line_centroid_window_kms'][:, 0], 0.)
        np.testing.assert_allclose(result['line_sigma_window_kms'][:, 0], 1.)
        np.testing.assert_allclose(result['line_sigma_full_kms'][:, 0], 1.)
        np.testing.assert_allclose(result['total_dL_dv_erg_s_per_kms'][:, 0],
                                   [[1., 1.], [.5, .5]])

    def test_slab_windows_cover_native_x_without_overlap(self):
        windows = list(slab_windows(10, 4))
        self.assertEqual(windows, [(0, 4), (4, 8), (8, 10)])


if __name__ == '__main__':
    unittest.main()
