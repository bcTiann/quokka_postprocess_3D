"""Joint DESPOTIC sampling preserves independent scalar field results."""
from dataclasses import replace
import unittest

import numpy as np

from quokka2s.despotic.lookup import DespoticTemperatureCO, DespoticLookup
from quokka2s.despotic.table_data import DespoticTable, SpeciesLineGrid, SpeciesRecord


def synthetic_table():
    axes = (np.array([.1, 1., 10.]), np.array([1e18, 1e20, 1e22]),
            np.array([1e-16, 1e-14, 1e-12]))
    i, j, k = np.indices((3, 3, 3), dtype=float)
    temperature = 20. + 7.*i + 3.*j + 2.*k + i*j*k
    species = {}
    for name, luminosity in (
        ('CO', (1. + i + 2.*j + 3.*k + i*j)*1e-25),
        ('CO21', (2. + 4.*i + j + 2.*k + j*k)*1e-25),
    ):
        abundance = np.full(temperature.shape, 1e-4)
        line = SpeciesLineGrid(freq=np.ones(temperature.shape),
            intIntensity=luminosity.copy(), intTB=luminosity.copy(),
            lumPerH=luminosity, tau=np.ones(temperature.shape),
            tauDust=np.zeros(temperature.shape), abundance=abundance)
        species[name] = SpeciesRecord(name, abundance, line, True)
    return DespoticTable(species_data=species, tg_final=temperature,
        nH_values=axes[0], col_density_values=axes[1], dVdr_values=axes[2],
        mu_values=np.ones(temperature.shape), cv_values=np.ones(temperature.shape),
        Eint_values=np.ones(temperature.shape))


class DespoticLookupTests(unittest.TestCase):
    def assert_matches_scalar(self, lookup, *coordinates):
        sample = lookup.temperature_and_co(*coordinates)
        self.assertIsInstance(sample, DespoticTemperatureCO)
        expected_shape = np.broadcast_arrays(*coordinates)[0].shape
        for actual, expected in (
            (sample.temperature_K, lookup.temperature(*coordinates)),
            (sample.co10_luminosity_per_H, lookup.line_field('CO', 'lumPerH', *coordinates)),
            (sample.co21_luminosity_per_H, lookup.line_field('CO21', 'lumPerH', *coordinates)),
        ):
            self.assertEqual(actual.shape, expected_shape)
            np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0, equal_nan=True)
        return sample

    def test_random_interior_and_boundaries_match_individual_fields(self):
        lookup = DespoticLookup(synthetic_table())
        random = np.random.default_rng(23)
        coordinates = [10.**random.uniform(lo, hi, 37) for lo, hi in ((-1., 1.), (18., 22.), (-16., -12.))]
        self.assert_matches_scalar(lookup, *coordinates)
        for nH in (.1, 1., 10.):
            for column in (1e18, 1e20, 1e22):
                for dvdr in (1e-16, 1e-14, 1e-12):
                    self.assert_matches_scalar(lookup, nH, column, dvdr)

    def test_broadcast_and_empty_shapes_with_multiple_chunks(self):
        lookup = DespoticLookup(synthetic_table())
        lookup._EVAL_CHUNK = 3
        self.assert_matches_scalar(lookup, np.array([.1, 1., 10.])[:, None],
                                   np.array([1e18, 1e19, 1e20, 1e22]), 1e-14)
        self.assert_matches_scalar(lookup, np.empty((0, 1)), np.empty((1, 3)), 1e-14)

    def test_invalid_coordinates_and_per_field_nan_support_are_preserved(self):
        table = synthetic_table()
        table.tg_final[0, 0, 0] = np.nan
        table.species_data['CO'].line.lumPerH[2, 2, 2] = np.nan
        lookup = DespoticLookup(table)
        sample = self.assert_matches_scalar(lookup,
            [.3, 3., np.nan, .01, 100.], [1e19, 1e21, 1e20, 1e20, 1e20],
            [1e-15, 1e-13, 1e-14, 1e-14, 1e-14])
        self.assertTrue(np.isnan(sample.temperature_K[0]))
        self.assertTrue(np.isfinite(sample.co10_luminosity_per_H[0]))
        self.assertTrue(np.isfinite(sample.temperature_K[1]))
        self.assertTrue(np.isnan(sample.co10_luminosity_per_H[1]))
        self.assertTrue(np.isfinite(sample.co21_luminosity_per_H[:2]).all())
        with np.errstate(divide='ignore', invalid='ignore'):
            self.assert_matches_scalar(lookup, [0., -1., np.inf], 1e20, 1e-14)

    def test_missing_line_only_fails_when_combined_sample_is_requested(self):
        table = synthetic_table()
        for species in ({'CO': table.species_data['CO']},
                        {'CO': replace(table.species_data['CO'], line=None),
                         'CO21': table.species_data['CO21']}):
            with self.subTest(species=tuple(species)):
                lookup = DespoticLookup(replace(table, species_data=species))
                self.assertTrue(np.isfinite(lookup.temperature(1., 1e20, 1e-14)))
                with self.assertRaisesRegex(ValueError, 'requires line data for CO'):
                    lookup.temperature_and_co(1., 1e20, 1e-14)


if __name__ == '__main__':
    unittest.main()
