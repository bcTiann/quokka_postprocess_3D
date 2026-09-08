from __future__ import annotations

import unittest

import numpy as np

from quokka2s.tables.model_depth import derive_model_depth


CONSTANTS = {
    "hydrogen_mass_g": 1.6735575e-24,
    "boltzmann_erg_K": 1.380649e-16,
    "gravitational_cm3_g_s2": 6.67430e-8,
    "parsec_cm": 3.0856775814913673e18,
}
GAMMA = 5.0 / 3.0


def internal_energy(rho, temperature, mu):
    return (
        np.asarray(rho) * CONSTANTS["boltzmann_erg_K"] * temperature
        / ((GAMMA - 1.0) * CONSTANTS["hydrogen_mass_g"] * mu)
    )


class ModelDepthTests(unittest.TestCase):
    def test_boundary_uses_paired_despotic_below_3000_and_quokka_at_3000(self):
        rho = 1.0e-20
        T_quokka = np.array([2999.0, 3000.0])
        u = internal_energy(rho, T_quokka, 0.62)
        result = derive_model_depth(
            rho, T_quokka, u, [40.0, 80.0], [2.3, 2.0], **CONSTANTS,
        )
        np.testing.assert_array_equal(result.cold_mask, [True, False])
        np.testing.assert_array_equal(result.valid, [True, True])
        np.testing.assert_allclose(result.temperature_K, [40.0, 3000.0])
        np.testing.assert_allclose(result.mean_molecular_weight, [2.3, 0.62])

    def test_quokka_mu_inverts_energy_relation_without_clipping(self):
        rho = np.array([1e-20, 3e-22, 2e-24])
        T = np.array([3000.0, 2e4, 1e6])
        mu = np.array([0.1, 0.61, 3.0])
        u = internal_energy(rho, T, mu)
        result = derive_model_depth(rho, T, u, np.nan, np.nan, **CONSTANTS)
        np.testing.assert_allclose(result.mean_molecular_weight, mu, rtol=1e-14)
        expected_length = np.pi * np.sqrt(
            GAMMA * (GAMMA - 1.0) * u
            / (CONSTANTS["gravitational_cm3_g_s2"] * rho**2)
        )
        np.testing.assert_allclose(result.jeans_length_cm, expected_length, rtol=1e-14)

    def test_unused_branch_inputs_do_not_spoil_valid_state(self):
        result = derive_model_depth(
            1e-20, [100.0, 3000.0], [np.nan, 1e-9],
            [40.0, np.nan], [2.3, -1.0], **CONSTANTS,
        )
        np.testing.assert_array_equal(result.valid, [True, True])
        self.assertTrue(np.isfinite(result.model_depth_cm).all())

    def test_invalid_relevant_inputs_stay_nan(self):
        result = derive_model_depth(
            [0.0, np.nan, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20, 1e-20],
            [100.0, 3000.0, 100.0, 100.0, 3000.0, 3000.0, 100.0, 100.0],
            [np.nan, 1e-9, np.nan, np.nan, 0.0, np.inf, np.nan, np.nan],
            [40.0, np.nan, 0.0, 40.0, np.nan, np.nan, np.inf, 40.0],
            [2.3, np.nan, 2.3, -1.0, np.nan, np.nan, 2.3, np.nan],
            **CONSTANTS,
        )
        self.assertFalse(result.valid.any())
        for values in (
            result.temperature_K, result.mean_molecular_weight,
            result.jeans_length_cm, result.model_depth_cm,
        ):
            self.assertTrue(np.isnan(values).all())

    def test_invalid_quokka_temperature_never_selects_regime(self):
        result = derive_model_depth(
            1e-20, [np.nan, np.inf, -np.inf, 0.0, -1.0],
            1e-9, 40.0, 2.3, **CONSTANTS,
        )
        self.assertFalse(result.cold_mask.any())
        self.assertFalse(result.valid.any())
        self.assertTrue(np.isnan(result.model_depth_cm).all())

    def test_jeans_density_scaling_and_pi_outside_square_root(self):
        rho = np.array([1e-20, 4e-20])
        result = derive_model_depth(rho, 100.0, np.nan, 40.0, 2.3, **CONSTANTS)
        expected_length = np.pi * np.sqrt(
            GAMMA * CONSTANTS["boltzmann_erg_K"] * 40.0
            / (
                CONSTANTS["gravitational_cm3_g_s2"] * 2.3
                * CONSTANTS["hydrogen_mass_g"] * rho
            )
        )
        np.testing.assert_allclose(result.jeans_length_cm, expected_length, rtol=1e-14)
        self.assertAlmostEqual(result.jeans_length_cm[0] / result.jeans_length_cm[1], 2.0)

    def test_100_pc_cap_preserves_uncapped_jeans_length(self):
        result = derive_model_depth(
            [1e-30, 1e-18], 100.0, np.nan, 40.0, 2.3, **CONSTANTS,
        )
        cap = 100.0 * CONSTANTS["parsec_cm"]
        self.assertGreater(result.jeans_length_cm[0], cap)
        self.assertEqual(result.model_depth_cm[0], cap)
        self.assertLess(result.jeans_length_cm[1], cap)
        self.assertEqual(result.model_depth_cm[1], result.jeans_length_cm[1])

    def test_broadcasting_and_scalar_outputs_do_not_mutate_inputs(self):
        rho = np.array([[1e-20], [2e-20]])
        T = np.array([100.0, 3000.0, np.nan])
        rho_before, T_before = rho.copy(), T.copy()
        result = derive_model_depth(rho, T, 1e-9, 40.0, 2.3, **CONSTANTS)
        self.assertEqual(result.model_depth_cm.shape, (2, 3))
        np.testing.assert_array_equal(result.valid, [[True, True, False]] * 2)
        np.testing.assert_array_equal(rho, rho_before)
        np.testing.assert_array_equal(T, T_before)
        scalar = derive_model_depth(1e-20, 100.0, np.nan, 40.0, 2.3, **CONSTANTS)
        self.assertEqual(scalar.model_depth_cm.shape, ())
        self.assertTrue(scalar.valid)

    def test_unrepresentable_length_is_invalid_even_when_cap_would_be_finite(self):
        result = derive_model_depth(1e-320, 100.0, np.nan, 40.0, 2.3, **CONSTANTS)
        self.assertFalse(result.valid)
        self.assertTrue(np.isnan(result.model_depth_cm))

    def test_invalid_constants_rejected(self):
        for key in CONSTANTS:
            for invalid in [0.0, -1.0, np.nan, np.inf, [1.0]]:
                with self.subTest(key=key, invalid=invalid):
                    constants = dict(CONSTANTS, **{key: invalid})
                    with self.assertRaises(ValueError):
                        derive_model_depth(1e-20, 100.0, np.nan, 40.0, 2.3, **constants)
        with self.assertRaises(ValueError):
            derive_model_depth(1e-20, 100.0, np.nan, 40.0, 2.3, gamma=1.0, **CONSTANTS)


if __name__ == "__main__":
    unittest.main()
