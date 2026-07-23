from __future__ import annotations

import unittest

import numpy as np
import yt

from quokka2s.pipeline.cache import CACHED_FIELDS
from quokka2s.pipeline.prep.physics_fields import (
    A_HI_21,
    NU_HI_21,
    _Halpha_luminosity,
    _HI_luminosity,
    _HI_luminosity_despotic,
    _HI_luminosity_quokka,
    _HI_luminosity_two_regime,
    _HI_thermal_width_despotic,
    _HI_thermal_width_quokka,
    c,
    h,
    kb,
    lambda_Halpha,
    m_H,
)
from quokka2s.pipeline.tasks.integrated_spectrum import SPECIES_CFG, spectrum_los


class HydrogenLineRegimeTests(unittest.TestCase):
    def setUp(self):
        T_qk = np.array([2999.0, 3000.0, 13069.0, 13070.0, 2.0e4])
        T_use = np.array([100.0, 3000.0, 13069.0, 13070.0, 2.0e4])
        rho = np.full(5, 2.0e-24)
        self.expected_xe = np.array([0.2, 0.4, 0.8, 0.8, 1.1])

        X, Y = 0.74, 0.26
        inverse_mu = X + Y / 4.0 + X * self.expected_xe
        e_int = (
            inverse_mu
            * rho
            * float(kb.to('erg/K').value)
            * T_qk
            / ((5.0 / 3.0 - 1.0) * float(m_H.to('g').value))
        )

        self.data = {
            ('gas', 'temperature_quokka'): yt.YTArray(T_qk, 'K'),
            ('gas', 'temperature_despotic'): yt.YTArray(
                [100.0, 200.0, 300.0, 400.0, 500.0], 'K'
            ),
            ('gas', 'temperature_two_regime'): yt.YTArray(T_use, 'K'),
            ('gas', 'number_density_H'): yt.YTArray(np.full(5, 10.0), 'cm**-3'),
            ('gas', 'e-'): yt.YTArray([0.1, 7.0, 7.0, 7.0, 7.0], 'cm**-3'),
            ('gas', 'H+'): yt.YTArray([1.0, 8.0, 9.0, 9.0, 9.0], 'cm**-3'),
            ('gas', 'HI'): yt.YTArray([7.0, 2.0, 2.0, 2.0, 2.0], 'cm**-3'),
            ('gas', 'internal_energy_density'): yt.YTArray(e_int, 'erg/cm**3'),
            ('gas', 'density'): yt.YTArray(rho, 'g/cm**3'),
        }

    def test_halpha_caps_hydrogen_fraction_at_one_above_3000K(self):
        actual = _Halpha_luminosity(None, self.data).to_value('erg/s/cm**3')

        # Cold cell keeps DESPOTIC values. Every cell at or above 3000 K uses
        # x_H+=min(x_e,1); 13070 K is deliberately not a hydrogen boundary.
        expected_ne = np.array([0.1, 4.0, 8.0, 8.0, 11.0])
        expected_n_Hp = np.array([1.0, 4.0, 8.0, 8.0, 10.0])
        T_use = np.array([100.0, 3000.0, 13069.0, 13070.0, 2.0e4])
        T4 = T_use / 1.0e4
        alpha_B = 2.54e-13 * np.power(T4, -0.8163 - 0.0208 * np.log(T4))
        E_Halpha = ((h * c) / lambda_Halpha).in_cgs().value
        expected = 0.45 * E_Halpha * alpha_B * expected_ne * expected_n_Hp

        np.testing.assert_allclose(actual, expected, rtol=1.0e-13)

    def test_hi_despotic_uses_table_neutral_hydrogen_at_all_temperatures(self):
        actual = _HI_luminosity_despotic(None, self.data).to_value(
            'erg/s/cm**3'
        )
        expected_n_HI = np.array([7.0, 2.0, 2.0, 2.0, 2.0])
        coefficient = 0.75 * A_HI_21 * float(h.in_cgs().value) * NU_HI_21
        np.testing.assert_allclose(actual, coefficient * expected_n_HI)

    def test_hi_quokka_uses_mu_derived_xe_at_all_temperatures(self):
        actual = _HI_luminosity_quokka(None, self.data).to_value(
            'erg/s/cm**3'
        )

        # x_e=1.1 gives x_H+=1 and n_HI=0 instead of a negative density.
        expected_n_HI = np.array([8.0, 6.0, 2.0, 2.0, 0.0])
        coefficient = 0.75 * A_HI_21 * float(h.in_cgs().value) * NU_HI_21
        expected = coefficient * expected_n_HI

        np.testing.assert_allclose(actual, expected, rtol=1.0e-13)

    def test_hi_two_regime_switches_at_3000K(self):
        actual = _HI_luminosity_two_regime(None, self.data).to_value(
            'erg/s/cm**3'
        )
        expected_n_HI = np.array([7.0, 6.0, 2.0, 2.0, 0.0])
        coefficient = 0.75 * A_HI_21 * float(h.in_cgs().value) * NU_HI_21
        np.testing.assert_allclose(actual, coefficient * expected_n_HI)

    def test_legacy_hi_name_selects_two_regime_result(self):
        legacy = _HI_luminosity(None, self.data)
        two_regime = _HI_luminosity_two_regime(None, self.data)
        np.testing.assert_allclose(legacy, two_regime)

    def test_hi_thermal_widths_use_matching_temperatures(self):
        width_dsp = _HI_thermal_width_despotic(None, self.data).to_value('cm/s')
        width_qk = _HI_thermal_width_quokka(None, self.data).to_value('cm/s')
        expected_ratio = np.sqrt(
            self.data[('gas', 'temperature_quokka')].to_value('K')
            / self.data[('gas', 'temperature_despotic')].to_value('K')
        )
        np.testing.assert_allclose(width_qk / width_dsp, expected_ratio)

    def test_pipeline_exposes_two_explicit_hi_results(self):
        by_name = {entry['name']: entry for entry in SPECIES_CFG}
        self.assertEqual(
            by_name['HI_DESPOTIC']['lum_field'],
            'HI_luminosity_despotic',
        )
        self.assertEqual(
            by_name['HI_QUOKKA']['lum_field'],
            'HI_luminosity_quokka',
        )
        self.assertEqual(spectrum_los(by_name['HI_DESPOTIC']), ('x', 'y'))
        self.assertEqual(spectrum_los(by_name['HI_QUOKKA']), ('x', 'y'))
        self.assertIn(('gas', 'HI_luminosity_despotic'), CACHED_FIELDS)
        self.assertIn(('gas', 'HI_luminosity_quokka'), CACHED_FIELDS)
        self.assertIn(('gas', 'HI_luminosity_two_regime'), CACHED_FIELDS)
        self.assertNotIn(('gas', 'HI_luminosity'), CACHED_FIELDS)


if __name__ == '__main__':
    unittest.main()
