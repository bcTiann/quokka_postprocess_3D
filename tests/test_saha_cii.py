from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from quokka2s.saha_cii import legacy_saha_lte_emissivity


class LegacySahaCIITests(unittest.TestCase):
    @staticmethod
    def _atomic_data() -> dict[str, object]:
        return {
            'temperature': np.array([1.0, 1.0e4]),
            'U_C0': np.ones(2),
            'U_Cp': np.ones(2),
            'U_Cpp': np.ones(2),
            'T_C': 0.0,
            'T_C2': 0.0,
            'A_ul': 1.0,
            'photon_energy_erg': 1.0,
            'T_star': 0.0,
            'g_l': 1.0,
            'g_u': 1.0,
            'saha_prefactor': 1.0,
        }

    @patch('quokka2s.saha_cii._atomic_data')
    def test_all_temperature_option_evaluates_cold_diagnostic(self, atomic):
        atomic.return_value = self._atomic_data()
        temperature = np.array([1000.0, 4000.0])
        default = legacy_saha_lte_emissivity(
            temperature,
            np.ones(2),
            np.ones(2),
            carbon_abundance_per_H=1.0,
        )
        all_temperature = legacy_saha_lte_emissivity(
            temperature,
            np.ones(2),
            np.ones(2),
            carbon_abundance_per_H=1.0,
            minimum_temperature_K=0.0,
        )

        self.assertEqual(default[0], 0.0)
        self.assertGreater(default[1], 0.0)
        self.assertGreater(all_temperature[0], 0.0)
        self.assertEqual(all_temperature[1], default[1])


if __name__ == '__main__':
    unittest.main()
