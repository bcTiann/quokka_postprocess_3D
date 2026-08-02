from __future__ import annotations

import unittest

import numpy as np

from quokka2s.pipeline.tasks.cplus_temperature_spectrum import (
    CPLUS_TEMPERATURE_COMPONENTS,
    combine_cplus_component_spectra,
)


class CplusTemperatureSpectrumTests(unittest.TestCase):
    def test_components_are_exact_tquokka_complements(self):
        by_name = {case['name']: case for case in CPLUS_TEMPERATURE_COMPONENTS}
        cold = by_name['CPLUS_DESPOTIC_TQK_LT3000']
        hot = by_name['CPLUS_CLOUDY_TQK_GE3000']
        self.assertEqual(cold['selection_temperature_field'], 'temperature_quokka')
        self.assertEqual(hot['selection_temperature_field'], 'temperature_quokka')
        self.assertEqual(cold['selection_operator'], 'lt')
        self.assertEqual(hot['selection_operator'], 'ge')
        self.assertEqual(cold['selection_cutoff_K'], 3000.0)
        self.assertEqual(hot['selection_cutoff_K'], 3000.0)
        self.assertEqual(cold['lum_field'], 'C+_luminosity')
        self.assertEqual(hot['lum_field'], 'C+_luminosity')

    def test_total_spectrum_is_cold_plus_hot_channel_by_channel(self):
        cold = np.array([0.0, 2.0, 3.0])
        hot = np.array([5.0, 7.0, 0.0])
        np.testing.assert_array_equal(
            combine_cplus_component_spectra(cold, hot),
            [5.0, 9.0, 3.0],
        )


if __name__ == '__main__':
    unittest.main()
