"""Hot-branch sampling must not query or silently zero excluded/cold gas."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from quokka2s.cloudy_cell_queries import CloudyCellEmission, prepare_cloudy_cell_queries
from quokka2s.cloudy_hot_emission import sample_cloudy_hot_emission
from quokka2s.cloudy_sixline_lookup import (
    CloudyFailureTouchError, CloudySixLineLookup, DEPTH_AXIS_ORDER, EXPECTED_AXIS_ORDER,
)
from quokka2s.tables.abundances import QUOKKA_MASS_FRACTIONS


CONSTANTS = dict(hydrogen_mass_g=1.6735575e-24, boltzmann_erg_K=1.380649e-16,
                 gravitational_cm3_g_s2=6.67430e-8, parsec_cm=3.0856775814913673e18)
X = QUOKKA_MASS_FRACTIONS['X']


class CloudyHotEmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)/'synthetic.npz'
        shape = (2, 2, 2, 3, 2)
        coefficient = np.ones(shape)*1e-30
        coefficient[:, 1] *= 8
        coefficient[1] *= 3
        self.payload = dict(
            axis_order=DEPTH_AXIS_ORDER, line_keys=['halpha', 'hi21'],
            log_NH_attenuation=[18., 21.], log_nH=[0., 1.],
            log_T=[2., 3., 4.], log_L_model_pc=[-.25, 2.],
            emissivity_per_nH2=coefficient,
            log_emissivity_per_nH2=np.log10(coefficient),
            failure_mask=np.zeros(shape, dtype=bool),
            zero_mask=np.zeros(shape, dtype=bool),
        )

    def lookup(self):
        np.savez(self.path, **self.payload)
        return CloudySixLineLookup(self.path)

    def fail(self, index):
        self.payload['failure_mask'][index] = True
        self.payload['zero_mask'][index] = False
        self.payload['emissivity_per_nH2'][index] = 0
        self.payload['log_emissivity_per_nH2'][index] = np.nan

    def use_legacy_payload(self):
        self.payload['axis_order'] = EXPECTED_AXIS_ORDER
        self.payload.pop('log_L_model_pc')
        for name in ('emissivity_per_nH2', 'log_emissivity_per_nH2', 'failure_mask', 'zero_mask'):
            self.payload[name] = self.payload[name][..., 0]

    def queries(self):
        n_h = np.array([1., 2., 3., 1.])
        rho = n_h*CONSTANTS['hydrogen_mass_g']/X
        tq = np.array([100., 3000., 1e4, 100.])
        energy = rho*CONSTANTS['boltzmann_erg_K']*tq / (
            (5/3-1)*CONSTANTS['hydrogen_mass_g']*.62)
        return prepare_cloudy_cell_queries(
            rho, [1e17, 1e17, 1e22, 1e22], tq, energy,
            [100., np.nan, np.nan, np.nan], [1.3, np.nan, np.nan, np.nan],
            authorized_excluded=[False, False, False, True], **CONSTANTS,
        )

    def test_cold_failure_does_not_block_hot_and_boundary_is_hot(self):
        self.fail((slice(None), slice(None), slice(None), 0, slice(None)))
        queries = self.queries()
        lookup = self.lookup()
        with self.assertRaises(CloudyFailureTouchError):
            queries.sample(lookup)
        result = sample_cloudy_hot_emission(queries, lookup)
        self.assertIsInstance(result, CloudyCellEmission)
        np.testing.assert_array_equal(result.applicable, [False, True, True, False])
        np.testing.assert_array_equal(result.excluded, [False, False, False, True])
        self.assertTrue(np.isnan(result.emissivity_erg_s_cm3[:, [0, 3]]).all())
        self.assertTrue(np.isfinite(result.emissivity_erg_s_cm3[:, [1, 2]]).all())

    def test_original_columns_clipping_flags_and_actual_density_squared(self):
        queries = self.queries()
        columns = queries.column_density_H_cm2.copy()
        result = sample_cloudy_hot_emission(queries, self.lookup())
        np.testing.assert_allclose(result.emissivity_erg_s_cm3[:, [1, 2]],
                                   [[4e-30, 72e-30], [12e-30, 216e-30]], rtol=1e-14)
        np.testing.assert_array_equal(queries.column_density_H_cm2, columns)
        np.testing.assert_array_equal(result.attenuation_column_below_table,
                                      [False, True, False, False])
        np.testing.assert_array_equal(result.attenuation_column_above_table,
                                      [False, False, True, False])
        result.excluded[:] = False
        np.testing.assert_array_equal(queries.excluded, [False, False, False, True])

    def test_hot_failed_support_propagates(self):
        self.fail((0, 0, slice(None), 1, 1))
        with self.assertRaises(CloudyFailureTouchError):
            sample_cloudy_hot_emission(self.queries(), self.lookup())

    def test_hot_depth_out_of_domain_is_not_clipped(self):
        self.payload['log_L_model_pc'] = [-.25, 1.]
        with self.assertRaisesRegex(ValueError, 'model_depth_pc is outside'):
            sample_cloudy_hot_emission(self.queries(), self.lookup())

    def test_true_zero_hot_emission_is_distinct_from_unqueried_cells(self):
        self.payload['emissivity_per_nH2'][:] = 0
        self.payload['log_emissivity_per_nH2'][:] = np.nan
        self.payload['zero_mask'][:] = True
        result = sample_cloudy_hot_emission(self.queries(), self.lookup())
        np.testing.assert_array_equal(result.emissivity_erg_s_cm3[:, [1, 2]], 0.)
        self.assertTrue(np.isnan(result.emissivity_erg_s_cm3[:, [0, 3]]).all())

    def test_scalar_and_broadcast_hot_shapes(self):
        rho = CONSTANTS['hydrogen_mass_g']/X
        tq = 3000.
        energy = rho*CONSTANTS['boltzmann_erg_K']*tq / (
            (5/3-1)*CONSTANTS['hydrogen_mass_g']*.62)
        for column in (1e18, [[1e17, 1e18], [1e21, 1e22]]):
            with self.subTest(column=column):
                queries = prepare_cloudy_cell_queries(
                    rho, column, tq, energy, np.nan, np.nan, **CONSTANTS)
                result = sample_cloudy_hot_emission(queries, self.lookup())
                self.assertEqual(result.emissivity_erg_s_cm3.shape, (2, *np.shape(column)))
                self.assertTrue(result.applicable.all())
                self.assertFalse(result.excluded.any())
                self.assertTrue(np.isfinite(result.emissivity_erg_s_cm3).all())

    def test_all_cold_skips_failed_lookup_but_still_rejects_legacy_table(self):
        rho = CONSTANTS['hydrogen_mass_g']/X
        queries = prepare_cloudy_cell_queries(rho, 1e19, 100., np.nan,
                                             100., 1.3, **CONSTANTS)
        self.fail(...)
        result = sample_cloudy_hot_emission(queries, self.lookup())
        self.assertFalse(result.applicable)
        self.assertFalse(result.excluded)
        self.assertTrue(np.isnan(result.emissivity_erg_s_cm3).all())
        self.payload['axis_order'] = EXPECTED_AXIS_ORDER
        self.payload.pop('log_L_model_pc')
        for name in ('emissivity_per_nH2', 'log_emissivity_per_nH2', 'failure_mask', 'zero_mask'):
            self.payload[name] = self.payload[name][..., 0]
        with self.assertRaisesRegex(ValueError, 'four-dimensional'):
            sample_cloudy_hot_emission(queries, self.lookup())

    def test_legacy_hot_requires_opt_in_and_matches_capped_coefficients(self):
        queries = self.queries()
        expected = sample_cloudy_hot_emission(queries, self.lookup())
        self.use_legacy_payload()
        with self.assertRaisesRegex(ValueError, 'four-dimensional'):
            sample_cloudy_hot_emission(queries, self.lookup())
        result = sample_cloudy_hot_emission(
            queries, self.lookup(), allow_capped_legacy_jeans=True)
        np.testing.assert_array_equal(result.emissivity_erg_s_cm3, expected.emissivity_erg_s_cm3)
        np.testing.assert_array_equal(result.attenuation_column_below_table,
                                      expected.attenuation_column_below_table)
        np.testing.assert_array_equal(result.attenuation_column_above_table,
                                      expected.attenuation_column_above_table)

    def test_legacy_rejects_short_cell_depth_even_with_capped_support(self):
        self.use_legacy_payload()
        queries = self.queries()
        queries.model_depth_pc[1] = 99.
        with self.assertRaisesRegex(ValueError, 'model depths of 100 pc'):
            sample_cloudy_hot_emission(queries, self.lookup(), allow_capped_legacy_jeans=True)

    def test_legacy_rejects_uncapped_corner_even_when_cell_is_capped(self):
        self.use_legacy_payload()
        self.payload['log_nH'] = [0., 6.]
        queries = self.queries()
        np.testing.assert_allclose(queries.model_depth_pc[1:3], 100.)
        with self.assertRaisesRegex(ValueError, 'uncapped table nodes'):
            sample_cloudy_hot_emission(queries, self.lookup(), allow_capped_legacy_jeans=True)

    def test_zero_weight_uncapped_corner_is_not_used(self):
        self.use_legacy_payload()
        self.payload['log_nH'] = [0., 6.]
        queries = prepare_cloudy_cell_queries(
            CONSTANTS['hydrogen_mass_g']/X, 1e18, 3000., np.nan, np.nan, np.nan, **CONSTANTS)
        result = sample_cloudy_hot_emission(queries, self.lookup(), allow_capped_legacy_jeans=True)
        self.assertTrue(np.isfinite(result.emissivity_erg_s_cm3).all())

    def test_legacy_rejects_node_that_becomes_uncapped_after_xh_correction(self):
        self.use_legacy_payload()
        # At nH=94 and T=1000 K, the legacy length exceeds the cap, but the
        # corrected length does not. A capped cell still touches this node.
        self.payload['log_nH'] = [0., np.log10(94.)]
        queries = self.queries()
        np.testing.assert_allclose(queries.model_depth_pc[1:3], 100.)
        with self.assertRaisesRegex(ValueError, 'corrected density conversion'):
            sample_cloudy_hot_emission(queries, self.lookup(), allow_capped_legacy_jeans=True)

    def test_legacy_failed_support_and_domain_exceptions_still_propagate(self):
        self.use_legacy_payload()
        self.fail((0, 0, slice(None), 1))
        with self.assertRaises(CloudyFailureTouchError):
            sample_cloudy_hot_emission(self.queries(), self.lookup(), allow_capped_legacy_jeans=True)
        self.payload['log_nH'] = [-2., -1.]
        with self.assertRaisesRegex(ValueError, 'log_nH is outside'):
            sample_cloudy_hot_emission(self.queries(), self.lookup(), allow_capped_legacy_jeans=True)


if __name__ == '__main__':
    unittest.main()
