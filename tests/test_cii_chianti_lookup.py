from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from quokka2s.cii_chianti_lookup import CiiUpperFractionLookup


class CiiUpperFractionLookupTests(unittest.TestCase):
    def setUp(self):
        self.lookup = CiiUpperFractionLookup(
            log10_temperature=np.array([4.0, 5.0]),
            log10_hydrogen_density=np.array([-2.0, 0.0]),
            log10_nu=np.array([
                [-6.0, -4.0],
                [-5.0, -3.0],
            ]),
        )

    def test_interpolates_in_log_coordinates(self):
        actual = self.lookup(10**4.5, 10**-1.0)
        self.assertAlmostEqual(float(actual), 10**-4.5)

    def test_broadcasts_inputs(self):
        actual = self.lookup(np.array([1e4, 1e5]), 1.0)
        np.testing.assert_allclose(actual, [1e-4, 1e-3])

    def test_clamps_outside_table(self):
        actual = self.lookup(np.array([1.0, 1e9]), np.array([1e-9, 1e9]))
        np.testing.assert_allclose(actual, [1e-6, 1e-3])

    def test_zero_hydrogen_density_has_zero_upper_population(self):
        self.assertEqual(float(self.lookup(1e5, 0.0)), 0.0)

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            CiiUpperFractionLookup([4.0, 5.0], [-2.0, 0.0], np.zeros((2, 3)))

    def test_versioned_project_table(self):
        table_path = Path(__file__).resolve().parents[1] / 'data' / 'cii_chianti_nu_cie_v3.npz'
        lookup = CiiUpperFractionLookup.from_npz(
            table_path,
            hydrogen_mass_fraction=0.74,
            helium_mass_fraction=0.26,
        )
        self.assertGreater(float(lookup(2.0e4, 1.0)), 0.0)
        with np.load(table_path) as table:
            self.assertTrue(bool(table['include_protons']))
            self.assertTrue(bool(table['explicit_proton_density']))
            self.assertFalse(bool(table['use_two_ion_model']))
            self.assertEqual(str(table['collider_elements']), 'H+He')

    def test_rejects_table_built_for_different_composition(self):
        table_path = Path(__file__).resolve().parents[1] / 'data' / 'cii_chianti_nu_cie_v3.npz'
        with self.assertRaisesRegex(ValueError, 'helium_mass_fraction'):
            CiiUpperFractionLookup.from_npz(
                table_path,
                hydrogen_mass_fraction=0.74,
                helium_mass_fraction=0.25,
            )


if __name__ == '__main__':
    unittest.main()
