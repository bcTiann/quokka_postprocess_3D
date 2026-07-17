from __future__ import annotations

import unittest

import numpy as np

from quokka2s.cie_composition import (
    cie_h_he_charge_per_hydrogen,
    helium_abundance_per_hydrogen,
)


class CieCompositionTests(unittest.TestCase):
    def test_helium_number_abundance(self):
        self.assertAlmostEqual(
            helium_abundance_per_hydrogen(
                hydrogen_mass_fraction=0.74,
                helium_mass_fraction=0.26,
            ),
            0.26 / (4.0 * 0.74),
        )

    def test_neutral_gas_has_no_free_charge(self):
        ne_per_h, np_per_h = cie_h_he_charge_per_hydrogen(
            [[1.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            hydrogen_mass_fraction=0.74,
            helium_mass_fraction=0.26,
        )
        np.testing.assert_array_equal(ne_per_h, [0.0])
        np.testing.assert_array_equal(np_per_h, [0.0])

    def test_fully_ionized_h_he_charge_neutrality(self):
        ne_per_h, np_per_h = cie_h_he_charge_per_hydrogen(
            [[0.0, 1.0]],
            [[0.0, 0.0, 1.0]],
            hydrogen_mass_fraction=0.74,
            helium_mass_fraction=0.26,
        )
        expected_ne = 1.0 + 2.0 * 0.26 / (4.0 * 0.74)
        np.testing.assert_allclose(ne_per_h, [expected_ne])
        np.testing.assert_array_equal(np_per_h, [1.0])


if __name__ == '__main__':
    unittest.main()
