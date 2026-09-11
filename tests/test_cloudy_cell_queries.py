"""Cell thermodynamics through real four-dimensional coefficient sampling."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from quokka2s.cloudy_cell_queries import prepare_cloudy_cell_queries
from quokka2s.cloudy_sixline_lookup import CloudyFailureTouchError, CloudySixLineLookup, DEPTH_AXIS_ORDER
from quokka2s.tables.abundances import QUOKKA_MASS_FRACTIONS

CONSTANTS = dict(hydrogen_mass_g=1.6735575e-24, boltzmann_erg_K=1.380649e-16,
                 gravitational_cm3_g_s2=6.67430e-8, parsec_cm=3.0856775814913673e18)
X = QUOKKA_MASS_FRACTIONS['X']


def log_coefficient(column, density, temperature, depth):
    column, density, temperature, depth = map(np.asarray, (column, density, temperature, depth))
    return -30 + .02*column + .03*density + .05*temperature + .07*depth + .004*temperature*depth


class CloudyCellQueryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'synthetic.npz'
        axes = [np.array([18., 21.]), np.array([-1., 6.]), np.array([0., 6.]), np.array([-.25, 2.])]
        coefficient = 10**log_coefficient(*np.meshgrid(*axes, indexing='ij'))[None]
        self.payload = dict(axis_order=DEPTH_AXIS_ORDER, line_keys=['test_line'],
                            log_NH_attenuation=axes[0], log_nH=axes[1], log_T=axes[2],
                            log_L_model_pc=axes[3], emissivity_per_nH2=coefficient,
                            log_emissivity_per_nH2=np.log10(coefficient),
                            zero_mask=np.zeros_like(coefficient, dtype=bool),
                            failure_mask=np.zeros_like(coefficient, dtype=bool))

    def lookup(self):
        np.savez(self.path, **self.payload)
        return CloudySixLineLookup(self.path)

    def prepare(self, *, excluded=None, td=None, mu=None):
        n_h = np.array([1e4, 2e4, 1.])
        rho = n_h * CONSTANTS['hydrogen_mass_g']/X
        tq = np.array([100., 2999., 3000.])
        u = np.array([np.nan, np.nan, rho[-1]*CONSTANTS['boltzmann_erg_K']*tq[-1]
                      / ((5/3-1)*CONSTANTS['hydrogen_mass_g']*.62)])
        return prepare_cloudy_cell_queries(
            rho, [1e17, 1e20, 1e22], tq, u,
            [40., 90., np.nan] if td is None else td,
            [2.3, 1.3, np.nan] if mu is None else mu,
            authorized_excluded=excluded, **CONSTANTS,
        )

    def test_paired_state_and_density_through_real_lookup(self):
        queries = self.prepare()
        np.testing.assert_allclose(queries.n_H_cm3, [1e4, 2e4, 1.])
        np.testing.assert_allclose(queries.state.temperature_K, [40., 90., 3000.])
        np.testing.assert_allclose(queries.state.mean_molecular_weight, [2.3, 1.3, .62])
        self.assertLess(queries.model_depth_pc[0], 100.)
        self.assertEqual(queries.model_depth_pc[-1], 100.)
        result = queries.sample(self.lookup())
        expected = 10**log_coefficient(
            [18., 20., 21.], np.log10(queries.n_H_cm3),
            np.log10([40., 90., 3000.]), np.log10(queries.model_depth_pc),
        ) * queries.n_H_cm3**2
        np.testing.assert_allclose(result.emissivity_erg_s_cm3[0], expected, rtol=3e-14)
        np.testing.assert_array_equal(queries.column_density_H_cm2, [1e17, 1e20, 1e22])
        np.testing.assert_array_equal(result.attenuation_column_below_table, [True, False, False])
        np.testing.assert_array_equal(result.attenuation_column_above_table, [False, False, True])

    def test_only_explicit_invalid_cold_exclusion_is_allowed(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected invalid'):
            self.prepare(td=[np.nan, 90., np.nan])
        queries = self.prepare(excluded=[True, False, False], td=[np.nan, 90., np.nan])
        result = queries.sample(self.lookup())
        self.assertTrue(np.isnan(result.emissivity_erg_s_cm3[:, 0]).all())
        self.assertTrue(np.isfinite(result.emissivity_erg_s_cm3[:, 1:]).all())
        self.assertEqual(queries.n_H_cm3[0], 1e4)
        self.assertEqual(queries.column_density_H_cm2[0], 1e17)
        for mask in ([False, False, True], [True, False, False], [0, 0, 0]):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                self.prepare(excluded=mask)

    def test_true_zero_is_distinct_from_failed_nodes(self):
        self.payload['emissivity_per_nH2'][:] = 0
        self.payload['log_emissivity_per_nH2'][:] = -99
        self.payload['zero_mask'][:] = True
        result = self.prepare().sample(self.lookup())
        self.assertTrue((result.emissivity_erg_s_cm3 == 0).all())
        self.payload['failure_mask'][:] = True
        self.payload['zero_mask'][:] = False
        self.payload['log_emissivity_per_nH2'][:] = np.nan
        with self.assertRaises(CloudyFailureTouchError):
            self.prepare().sample(self.lookup())

    def test_physical_depth_outside_table_is_not_clamped(self):
        self.payload['log_L_model_pc'] = np.array([-.25, 1.])
        with self.assertRaisesRegex(ValueError, 'model_depth_pc is outside'):
            self.prepare().sample(self.lookup())

    def test_legacy_table_is_rejected(self):
        from quokka2s.cloudy_sixline_lookup import EXPECTED_AXIS_ORDER
        self.payload['axis_order'] = EXPECTED_AXIS_ORDER
        self.payload.pop('log_L_model_pc')
        for key in ('emissivity_per_nH2', 'log_emissivity_per_nH2', 'zero_mask', 'failure_mask'):
            self.payload[key] = self.payload[key][..., 0]
        with self.assertRaisesRegex(ValueError, 'four-dimensional'):
            self.prepare().sample(self.lookup())

    def test_scalar_and_broadcast_shapes(self):
        rho = CONSTANTS['hydrogen_mass_g']/X
        for column in (1e19, [[1e17, 1e19], [1e20, 1e22]]):
            queries = prepare_cloudy_cell_queries(rho, column, 100., np.nan, 40., 2.3, **CONSTANTS)
            result = queries.sample(self.lookup())
            self.assertEqual(result.emissivity_erg_s_cm3.shape, (1, *np.shape(column)))
            self.assertTrue(np.isfinite(result.emissivity_erg_s_cm3).all())

    def test_input_column_is_owned_and_not_mutated(self):
        column = np.array([1e17, 1e22])
        queries = prepare_cloudy_cell_queries(1e-20, column, 100., np.nan, 40., 2.3, **CONSTANTS)
        queries.sample(self.lookup())
        np.testing.assert_array_equal(column, [1e17, 1e22])
        column[:] = 1e19
        np.testing.assert_array_equal(queries.column_density_H_cm2, [1e17, 1e22])

    def test_invalid_hot_energy_is_not_replaced_by_despotic(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected invalid'):
            prepare_cloudy_cell_queries(1e-20, 1e20, 1e4, -1., 40., 2.3, **CONSTANTS)


if __name__ == '__main__':
    unittest.main()
