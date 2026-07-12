from __future__ import annotations

import unittest

import numpy as np

from quokka2s.line_regimes import (
    hydrogen_ionization_fraction_from_mean_molecular_weight,
    temperature_regime_masks,
)


class LineRegimeTests(unittest.TestCase):
    def test_boundary_membership(self):
        despotic_cutoff = 3000.0
        cie_cutoff = 1.307e4
        temperature = np.array([
            despotic_cutoff - 1.0,
            despotic_cutoff,
            cie_cutoff - 1.0,
            cie_cutoff,
        ])
        low, intermediate, high = temperature_regime_masks(
            temperature,
            despotic_cutoff,
            cie_cutoff,
        )
        np.testing.assert_array_equal(low, [True, False, False, False])
        np.testing.assert_array_equal(intermediate, [False, True, True, False])
        np.testing.assert_array_equal(high, [False, False, False, True])

    def test_mean_molecular_weight_inversion_recovers_xe(self):
        X, Y = 0.74, 0.26
        gamma = 5.0 / 3.0
        m_h = 1.6735575e-24
        k_b = 1.380649e-16
        rho = np.array([1.0e-24, 2.0e-24, 3.0e-24])
        temperature = np.array([4000.0, 8000.0, 12000.0])
        expected_xe = np.array([0.0, 0.25, 1.0])
        inverse_mu = (1.0 + expected_xe) * X + Y / 4.0
        e_int = inverse_mu * rho * k_b * temperature / ((gamma - 1.0) * m_h)

        actual = hydrogen_ionization_fraction_from_mean_molecular_weight(
            e_int,
            rho,
            temperature,
            hydrogen_mass_g=m_h,
            boltzmann_erg_K=k_b,
        )
        np.testing.assert_allclose(actual, expected_xe, rtol=1e-13, atol=1e-13)

    def test_mean_molecular_weight_inversion_stays_physical(self):
        actual = hydrogen_ionization_fraction_from_mean_molecular_weight(
            np.array([-1.0, 0.0, 1.0e30]),
            np.array([1.0, 0.0, 1.0]),
            np.ones(3),
            hydrogen_mass_g=1.0,
            boltzmann_erg_K=1.0,
        )
        np.testing.assert_array_equal(actual, [0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
