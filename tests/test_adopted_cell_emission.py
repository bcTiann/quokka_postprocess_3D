"""Adopted branches preserve failures, clipping, and explicit exclusions."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from quokka2s.adopted_cell_emission import (
    ATOMIC_LINE_KEYS, COLD_OMITTED_LINES, compute_adopted_cell_emission,
)
from quokka2s.cloudy_cell_queries import prepare_cloudy_cell_queries
from quokka2s.cloudy_sixline_lookup import (
    CloudyFailureTouchError, CloudySixLineLookup, DEPTH_AXIS_ORDER, EXPECTED_AXIS_ORDER,
)
from quokka2s.tables.abundances import QUOKKA_MASS_FRACTIONS


CONSTANTS = dict(hydrogen_mass_g=1.6735575e-24, boltzmann_erg_K=1.380649e-16,
                 gravitational_cm3_g_s2=6.67430e-8, parsec_cm=3.0856775814913673e18)
X = QUOKKA_MASS_FRACTIONS['X']


class StubDespotic:
    def __init__(self):
        self.table = SimpleNamespace(nH_values=np.array([1., 10.]),
            col_density_values=np.array([1e18, 1e21]), dVdr_values=np.array([1e-16, 1e-13]))
        self.calls = []
        self.bad_line = None
        self.bad_density = None
        self.temperature_value = 50.

    def temperature(self, nh, column, dvdr):
        self.calls.append(('temperature', nh.copy()))
        return np.full(nh.shape, self.temperature_value)

    def line_field(self, species, field, nh, column, dvdr):
        self.calls.append((species, nh.copy()))
        if field != 'lumPerH':
            raise AssertionError('unexpected requested field')
        return np.full(nh.shape, np.nan if species == self.bad_line else
                       {'C+': 2e-24, 'CO': 3e-24, 'CO21': 5e-24}[species])

    def number_densities(self, species, nh, column, dvdr):
        self.calls.append(('number_densities', nh.copy()))
        return {name: np.full(nh.shape, np.nan) if name == self.bad_density else
                nh * {'e-': .02, 'H+': .03, 'H': .7}[name] for name in species}


class AdoptedCellEmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)/'synthetic.npz'
        shape = (8, 2, 2, 3, 2)
        coefficients = np.broadcast_to(np.arange(1., 9.)[:, None, None, None, None]*1e-28, shape).copy()
        self.payload = dict(axis_order=DEPTH_AXIS_ORDER, line_keys=ATOMIC_LINE_KEYS,
            log_NH_attenuation=[18., 21.], log_nH=[-1., 2.],
            log_T=[np.log10(50.), np.log10(3000.), 4.], log_L_model_pc=[-.25, 2.],
            emissivity_per_nH2=coefficients, log_emissivity_per_nH2=np.log10(coefficients),
            failure_mask=np.zeros(shape, dtype=bool), zero_mask=np.zeros(shape, dtype=bool))
        self.dsp = StubDespotic()

    def lookup(self):
        np.savez(self.path, **self.payload)
        return CloudySixLineLookup(self.path)

    def fail(self, index):
        self.payload['failure_mask'][index] = True
        self.payload['emissivity_per_nH2'][index] = 0.
        self.payload['log_emissivity_per_nH2'][index] = np.nan

    def queries(self):
        nh = np.array([.5, 2., 20., 3.])
        rho = nh*CONSTANTS['hydrogen_mass_g']/X
        tq = np.array([2999., 3000., 100., 100.])
        u = rho*CONSTANTS['boltzmann_erg_K']*tq / ((5/3-1)*CONSTANTS['hydrogen_mass_g']*.62)
        return prepare_cloudy_cell_queries(rho, [1e17, 1e19, 1e22, 1e20], tq, u,
            [50., np.nan, 50., np.nan], [1.3, np.nan, 2.3, np.nan],
            authorized_excluded=[False, False, False, True], **CONSTANTS)

    def calculate(self):
        return compute_adopted_cell_emission(self.queries(), [1e-17, 1e-15, 1e-12, np.nan],
                                              self.dsp, self.lookup())

    def test_adopted_branches_and_thermal_temperatures(self):
        from quokka2s.pipeline.prep.physics_fields import (
            _HI_emissivity_from_number_density, effective_halpha_recombination_coefficient,
            h, c, lambda_Halpha,
        )
        # A failed cold Cloudy support plane must never be sampled.
        self.fail((slice(None), slice(None), slice(None), 0, slice(None)))
        result = self.calculate()
        self.assertEqual(result.line_keys, ATOMIC_LINE_KEYS+('co10', 'co21'))
        np.testing.assert_allclose(result.emissivity_erg_s_cm3[:8, 1], np.arange(1., 9.)*4e-28, atol=0)
        cold = [0, 2]
        nh_query = np.array([1., 10.])
        np.testing.assert_allclose(result.emissivity_erg_s_cm3[0, cold], nh_query*2e-24, atol=0)
        photon = float(((h*c)/lambda_Halpha).in_cgs().value)
        np.testing.assert_allclose(result.emissivity_erg_s_cm3[1, cold],
            photon*effective_halpha_recombination_coefficient(50.)*(.02*nh_query)*(.03*nh_query), atol=0)
        np.testing.assert_allclose(result.emissivity_erg_s_cm3[2, cold],
                                   _HI_emissivity_from_number_density(.7*nh_query), atol=0)
        for key in COLD_OMITTED_LINES:
            np.testing.assert_array_equal(result.emissivity_erg_s_cm3[result.line_keys.index(key), cold], 0.)
        np.testing.assert_allclose(result.emissivity_erg_s_cm3[8, :3], [3e-24, 6e-24, 30e-24], atol=0)
        np.testing.assert_allclose(result.emissivity_erg_s_cm3[9, :3], [5e-24, 10e-24, 50e-24], atol=0)
        np.testing.assert_array_equal(result.thermal_temperature_K[:8, :3], np.tile([50., 3000., 50.], (8, 1)))
        np.testing.assert_array_equal(result.thermal_temperature_K[8:, :3], 50.)

    def test_exclusion_and_coordinate_clipping_are_explicit_and_owned(self):
        queries = self.queries()
        dvdr = np.array([1e-17, 1e-15, 1e-12, np.nan])
        original = (queries.n_H_cm3.copy(), queries.column_density_H_cm2.copy(), dvdr.copy())
        result = compute_adopted_cell_emission(queries, dvdr, self.dsp, self.lookup())
        np.testing.assert_array_equal(result.valid, [True, True, True, False])
        np.testing.assert_array_equal(result.excluded, [False, False, False, True])
        for flags in result.despotic_clipped.values():
            np.testing.assert_array_equal(flags, [True, False, True, False])
        self.assertTrue(np.isnan(result.emissivity_erg_s_cm3[:, 3]).all())
        self.assertTrue(np.isnan(result.thermal_temperature_K[:, 3]).all())
        for actual, reference in zip((queries.n_H_cm3, queries.column_density_H_cm2, dvdr), original):
            np.testing.assert_array_equal(actual, reference)
        number_calls = [values for name, values in self.dsp.calls if name == 'number_densities']
        np.testing.assert_array_equal(number_calls, [[1., 10.]])
        result.excluded[:] = False
        self.assertTrue(queries.excluded[3])

    def test_missing_despotic_line_or_density_is_not_zero_filled(self):
        for species in ('C+', 'CO', 'CO21'):
            with self.subTest(species=species):
                self.dsp.bad_line = species
                with self.assertRaisesRegex(ValueError, 'lumPerH must be finite'):
                    self.calculate()
        self.dsp.bad_line = None
        self.dsp.bad_density = 'H'
        with self.assertRaisesRegex(ValueError, 'number density must be finite'):
            self.calculate()

    def test_new_despotic_temperature_must_match_accepted_cold_state(self):
        self.dsp.temperature_value = 51.
        with self.assertRaisesRegex(ValueError, 'differs from the accepted paired state'):
            self.calculate()
        self.dsp.temperature_value = np.nan
        with self.assertRaisesRegex(ValueError, 'DESPOTIC temperature must be finite'):
            self.calculate()

    def test_hot_failure_and_invalid_retained_dvdr_raise(self):
        self.fail((0, slice(None), slice(None), 1, slice(None)))
        with self.assertRaises(CloudyFailureTouchError):
            self.calculate()
        with self.assertRaisesRegex(ValueError, 'dVdr must be finite and positive'):
            compute_adopted_cell_emission(self.queries(), [0., 1e-15, 1e-15, np.nan],
                                          self.dsp, CloudySixLineLookup(self.path))

    def test_scalar_hot_cell_uses_despotic_only_for_co(self):
        rho = 2.*CONSTANTS['hydrogen_mass_g']/X
        u = rho*CONSTANTS['boltzmann_erg_K']*3000. / ((5/3-1)*CONSTANTS['hydrogen_mass_g']*.62)
        queries = prepare_cloudy_cell_queries(rho, 1e19, 3000., u, np.nan, np.nan, **CONSTANTS)
        result = compute_adopted_cell_emission(queries, 1e-15, self.dsp, self.lookup())
        self.assertEqual(result.emissivity_erg_s_cm3.shape, (10,))
        self.assertTrue(result.valid)
        self.assertEqual([name for name, values in self.dsp.calls], ['temperature', 'CO', 'CO21'])
        np.testing.assert_array_equal(result.thermal_temperature_K[:8], 3000.)
        np.testing.assert_array_equal(result.thermal_temperature_K[8:], 50.)

    def test_all_excluded_cells_do_not_query_despotic(self):
        rho = CONSTANTS['hydrogen_mass_g']/X
        queries = prepare_cloudy_cell_queries(rho, 1e19, 100., np.nan, np.nan, np.nan,
                                             authorized_excluded=True, **CONSTANTS)
        result = compute_adopted_cell_emission(queries, np.nan, self.dsp, self.lookup())
        self.assertFalse(result.valid)
        self.assertTrue(result.excluded)
        self.assertTrue(np.isnan(result.emissivity_erg_s_cm3).all())
        self.assertEqual(self.dsp.calls, [])

    def test_legacy_opt_in_preserves_branches_and_omits_cold_carbon_ions(self):
        expected = self.calculate()
        self.payload['axis_order'] = EXPECTED_AXIS_ORDER
        self.payload.pop('log_L_model_pc')
        for name in ('emissivity_per_nH2', 'log_emissivity_per_nH2', 'failure_mask', 'zero_mask'):
            self.payload[name] = self.payload[name][..., 0]
        with self.assertRaisesRegex(ValueError, 'four-dimensional'):
            self.calculate()
        result = compute_adopted_cell_emission(
            self.queries(), [1e-17, 1e-15, 1e-12, np.nan], self.dsp, self.lookup(),
            allow_capped_legacy_jeans=True)
        np.testing.assert_array_equal(result.emissivity_erg_s_cm3, expected.emissivity_erg_s_cm3)
        for key in COLD_OMITTED_LINES:
            np.testing.assert_array_equal(result.emissivity_erg_s_cm3[result.line_keys.index(key), [0, 2]], 0.)

    def test_authorized_hot_exclusion_is_not_sampled_or_zero_filled(self):
        rho = np.array([1., 2.])*CONSTANTS['hydrogen_mass_g']/X
        queries = prepare_cloudy_cell_queries(
            rho, 1e19, 3000., np.nan, [50., np.nan], np.nan,
            authorized_excluded=[False, True], allow_hot_missing_despotic_exclusions=True,
            **CONSTANTS)
        result = compute_adopted_cell_emission(queries, [1e-15, np.nan], self.dsp, self.lookup())
        np.testing.assert_array_equal(result.valid, [True, False])
        self.assertTrue(np.isfinite(result.emissivity_erg_s_cm3[:, 0]).all())
        self.assertTrue(np.isnan(result.emissivity_erg_s_cm3[:, 1]).all())
        for _, densities in self.dsp.calls:
            np.testing.assert_array_equal(densities, [1.])


if __name__ == '__main__':
    unittest.main()
