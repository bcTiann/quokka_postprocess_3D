"""Batch Cloudy queries preserve selection, emissivity_per_nH2 and failures."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from quokka2s.cloudy.cell_fields import CloudyCellReader
from quokka2s.cloudy.lookup import CloudyLookup, CloudyFailureTouchError
from quokka2s.cloudy.table_definition import EXPECTED_AXIS_ORDER
from quokka2s.constants import HYDROGEN_MASS_G
from quokka2s.physics.settings import X_H
from quokka2s.snapshot_reader import CellBatch


class CloudyCellFieldsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'table.npz'
        shape = (2, 2, 2, 3)
        coefficients = np.ones(shape) * 1e-30
        coefficients[:, 1] *= 8
        coefficients[1] *= 3
        self.payload = dict(
            axis_order=EXPECTED_AXIS_ORDER, line_keys=['halpha', 'hi21'],
            log_NH_attenuation=[18., 21.], log_nH=[0., 1.],
            log_T=[2., 3., 4.],
            emissivity_per_nH2=coefficients,
            log_emissivity_per_nH2=np.log10(coefficients),
            failure_mask=np.zeros(shape, dtype=bool),
            zero_mask=np.zeros(shape, dtype=bool),
        )
        nH = np.array([1., 2., 3., 1.])
        self.cells = CellBatch(
            density_g_cm3=nH * HYDROGEN_MASS_G / X_H,
            temperature_QUOKKA_K=np.array([100., 3000., 1e4, 1e4]),
            shielding_NH_cm2=np.array([1e17, 1e17, 1e22, 1e22]),
            foreground_NH_cm2=np.zeros(4),
            velocity_gradient_s=np.zeros(4),
            velocity_z_kms=np.zeros(4),
            x_start=0,
            slab_shape=(1, 1, 4),
            batch_start=0,
            cell_volume_cm3=1.,
        )
        # Cold cell 0 and upstream-excluded hot cell 3 are never queried.
        self.selected = np.array([False, True, True, False])

    def lookup(self):
        failed = self.payload['failure_mask']
        self.payload['emissivity_per_nH2'][failed] = 0.
        self.payload['log_emissivity_per_nH2'][failed] = np.nan
        np.savez(self.path, **self.payload)
        return CloudyLookup(self.path)

    def query(self, lookup=None):
        if lookup is None:
            lookup = self.lookup()
        reader = CloudyCellReader(lookup=lookup)
        return reader.read_fields(
            cells=self.cells,
            selected_cells=self.selected,
        )

    def test_reused_reader_keeps_only_lookup_and_does_not_retain_batch_state(self):
        lookup = self.lookup()
        reader = CloudyCellReader(lookup=lookup)
        first_selection = np.array([False, True, True, False])
        first_fields = reader.read_fields(
            cells=self.cells,
            selected_cells=first_selection,
        )
        saved_first_values = first_fields.emissivity_per_nH2['halpha'].copy()
        second_selection = np.array([True, False, False, True])
        second_fields = reader.read_fields(
            cells=self.cells,
            selected_cells=second_selection,
        )
        self.assertEqual(vars(reader), {'lookup': lookup})
        np.testing.assert_array_equal(
            first_fields.emissivity_per_nH2['halpha'],
            saved_first_values,
        )
        self.assertTrue(np.isfinite(second_fields.emissivity_per_nH2['halpha'][second_selection]).all())
        self.assertTrue(np.isnan(second_fields.emissivity_per_nH2['halpha'][first_selection]).all())

    def test_returned_fields_are_named_by_actual_lookup_lines(self):
        lookup = self.lookup()
        fields = self.query(lookup)
        self.assertIsInstance(fields.emissivity_per_nH2, dict)
        self.assertEqual(tuple(fields.emissivity_per_nH2), tuple(lookup.line_keys))
        self.assertEqual(tuple(fields.emissivity_per_nH2), ('halpha', 'hi21'))
        for values in fields.emissivity_per_nH2.values():
            self.assertEqual(values.shape, (4,))

    def test_returned_temperature_preserves_the_queried_state_and_cell_order(self):
        fields = self.query()
        self.assertEqual(fields.temperature_K.shape, self.cells.temperature_QUOKKA_K.shape)
        np.testing.assert_array_equal(
            fields.temperature_K[self.selected],
            self.cells.temperature_QUOKKA_K[self.selected],
        )
        self.assertTrue(np.isnan(fields.temperature_K[~self.selected]).all())

    def test_named_arrays_share_one_restored_matrix_without_copying(self):
        fields = self.query()
        halpha = fields.emissivity_per_nH2['halpha']
        hi21 = fields.emissivity_per_nH2['hi21']
        self.assertIs(halpha.base, hi21.base)
        self.assertEqual(halpha.base.shape, (2, 4))
        self.assertTrue(np.shares_memory(halpha, halpha.base))
        self.assertTrue(np.shares_memory(hi21, hi21.base))

    def test_nonstandard_table_order_keeps_line_names_attached_to_values(self):
        self.payload['line_keys'] = ['hi21', 'halpha']
        for name in (
            'emissivity_per_nH2',
            'log_emissivity_per_nH2',
            'failure_mask',
            'zero_mask',
        ):
            self.payload[name] = self.payload[name][::-1].copy()
        fields = self.query()
        self.assertEqual(tuple(fields.emissivity_per_nH2), ('hi21', 'halpha'))
        np.testing.assert_allclose(
            fields.emissivity_per_nH2['halpha'][self.selected],
            [1e-30, 8e-30],
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            fields.emissivity_per_nH2['hi21'][self.selected],
            [3e-30, 24e-30],
            rtol=1e-14,
        )

    def test_only_selected_hot_cells_queried_and_no_density_multiplication(self):
        self.payload['failure_mask'][:, :, :, 0] = True
        lookup = self.lookup()
        with patch.object(lookup, 'interpolate_available', wraps=lookup.interpolate_available) as query:
            fields = self.query(lookup)
        np.testing.assert_array_equal(query.call_args.kwargs['temperature_K'], [3000., 1e4])
        np.testing.assert_allclose(
            fields.emissivity_per_nH2['halpha'][1:3],
            [1e-30, 8e-30],
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            fields.emissivity_per_nH2['hi21'][1:3],
            [3e-30, 24e-30],
            rtol=1e-14,
        )
        for values in fields.emissivity_per_nH2.values():
            self.assertTrue(np.isnan(values[[0, 3]]).all())
        self.assertFalse(fields.failed_cells.any())
        self.assertEqual(fields.column_clipped_cells, dict(below=1, above=1))

    def test_one_line_failure_excludes_all_lines_in_that_cell_without_retry(self):
        self.payload['failure_mask'][0, 0, :, 1] = True
        lookup = self.lookup()
        original_columns = self.cells.shielding_NH_cm2.copy()
        original_selection = self.selected.copy()
        with patch.object(lookup, 'prepare_interpolation_coordinates', wraps=lookup.prepare_interpolation_coordinates) as prepare:
            with patch.object(lookup, 'inspect_interpolation_nodes', wraps=lookup.inspect_interpolation_nodes) as inspect:
                fields = self.query(lookup)
        prepare.assert_called_once()
        inspect.assert_called_once()
        np.testing.assert_array_equal(fields.failed_cells, [False, True, False, False])
        for values in fields.emissivity_per_nH2.values():
            self.assertTrue(np.isnan(values[1]))
            self.assertTrue(np.isfinite(values[2]))
        self.assertEqual(fields.column_clipped_cells, dict(below=0, above=1))
        np.testing.assert_array_equal(self.cells.shielding_NH_cm2, original_columns)
        np.testing.assert_array_equal(self.selected, original_selection)

    def test_physical_column_clipping_counts_keep_nextafter_endpoint_excursions(self):
        self.cells.shielding_NH_cm2[1] = np.nextafter(1e18, -np.inf)
        self.cells.shielding_NH_cm2[2] = np.nextafter(1e21, np.inf)
        lookup = self.lookup()
        queried = lookup.interpolate_available(
            temperature_K=self.cells.temperature_QUOKKA_K[self.selected],
            hydrogen_density_cm3=self.cells.hydrogen_density_cm3[self.selected],
            shielding_NH_cm2=self.cells.shielding_NH_cm2[self.selected],
        )
        # Logarithms round these excursions to the exact bounds. Reader counts
        # retain the original physical-NH statistic, independently of those flags.
        self.assertFalse(queried.attenuation_column_below_table.any())
        self.assertFalse(queried.attenuation_column_above_table.any())
        fields = self.query(lookup=lookup)
        self.assertEqual(fields.column_clipped_cells, dict(below=1, above=1))

    def test_all_failed_skips_interpolation(self):
        self.payload['failure_mask'][:] = True
        lookup = self.lookup()
        with patch.object(lookup, '_interpolate_emissivity_per_nH2', side_effect=AssertionError('must not interpolate')):
            fields = self.query(lookup)
        np.testing.assert_array_equal(fields.failed_cells, self.selected)
        for values in fields.emissivity_per_nH2.values():
            self.assertTrue(np.isnan(values).all())

    def test_zero_is_not_a_failed_node(self):
        self.payload['emissivity_per_nH2'][:] = 0.
        self.payload['log_emissivity_per_nH2'][:] = np.nan
        self.payload['zero_mask'][:] = True
        fields = self.query()
        for values in fields.emissivity_per_nH2.values():
            np.testing.assert_array_equal(values[self.selected], 0.)
        self.assertFalse(fields.failed_cells.any())

    def test_no_selected_cells_returns_no_failures(self):
        self.selected[:] = False
        self.payload['failure_mask'][:] = True
        fields = self.query()
        self.assertEqual(tuple(fields.emissivity_per_nH2), ('halpha', 'hi21'))
        for values in fields.emissivity_per_nH2.values():
            self.assertTrue(np.isnan(values).all())
        self.assertFalse(fields.failed_cells.any())
        self.assertEqual(fields.column_clipped_cells, dict(below=0, above=0))

    def test_query_passes_only_the_three_physical_coordinates(self):
        self.payload['log_nH'] = [0., 6.]
        lookup = self.lookup()
        with patch.object(lookup, 'interpolate_available', wraps=lookup.interpolate_available) as query:
            fields = self.query(lookup)
        self.assertEqual(set(query.call_args.kwargs),
                         {'temperature_K', 'hydrogen_density_cm3', 'shielding_NH_cm2'})
        for values in fields.emissivity_per_nH2.values():
            self.assertTrue(np.isfinite(values[self.selected]).all())

    def test_non_three_dimensional_axis_order_is_rejected(self):
        self.payload['axis_order'] = EXPECTED_AXIS_ORDER + ',extra_coordinate'
        with self.assertRaisesRegex(ValueError, 'unexpected Cloudy axis order'):
            self.lookup()

    def test_nonconsecutive_results_keep_original_positions(self):
        self.selected = np.array([True, False, True, False])
        fields = self.query()
        np.testing.assert_allclose(
            fields.emissivity_per_nH2['halpha'][[0, 2]],
            [1e-30, 8e-30],
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            fields.emissivity_per_nH2['hi21'][[0, 2]],
            [3e-30, 24e-30],
            rtol=1e-14,
        )
        for values in fields.emissivity_per_nH2.values():
            self.assertTrue(np.isnan(values[[1, 3]]).all())
        self.assertFalse(fields.failed_cells.any())

    def test_outside_physical_table_domain_still_raises(self):
        self.payload['log_nH'] = [-2., -1.]
        with self.assertRaisesRegex(ValueError, 'log_nH is outside'):
            self.query()

    def test_strict_diagnostic_still_raises_on_failure(self):
        self.payload['failure_mask'][:] = True
        with self.assertRaises(CloudyFailureTouchError):
            self.lookup().sample(1e4, 2., 1e19)


if __name__ == '__main__':
    unittest.main()
