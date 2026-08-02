from __future__ import annotations

import unittest

import numpy as np
import yt

from quokka2s.pipeline.services.spectrum_service import (
    SpectrumStore,
    temperature_selection_mask,
)
from quokka2s.pipeline.tasks.hi_temperature_spectrum import (
    HI_TEMPERATURE_COMPLEMENT_CONFIG,
    HI_TEMPERATURE_SPECTRUM_CONFIG,
    HI_TEMPERATURE_SPECTRUM_LOS,
    shared_hi_spectrum_ylim,
)
from quokka2s.pipeline.tasks.integrated_spectrum import SPECIES_CFG


class _FakeProvider:
    def __init__(self, fields):
        self.fields = fields

    def get_slab_z(self, field):
        return self.fields[field], None


class HITemperatureSpectrumTests(unittest.TestCase):
    def test_shared_ylim_uses_los_y_global_max_across_figures(self):
        full = {'spectra': {}}
        for species, peak in [('HI_DESPOTIC', 4.0), ('HI_QUOKKA', 20.0)]:
            full['spectra'][species] = {}
            for los, scale in [('x', 1.0), ('y', 1.2)]:
                full['spectra'][species][los] = {
                    'total': {'dsigma_dv': np.array([0.0, peak * scale])}
                }

        selected = {'spectra': {}}
        for index, case in enumerate(HI_TEMPERATURE_SPECTRUM_CONFIG, start=1):
            selected['spectra'][case['name']] = {}
            for los in ('x', 'y'):
                selected['spectra'][case['name']][los] = {
                    'dsigma_dv': np.array([0.0, float(index)])
                }

        # Global maximum is QUOKKA LOS y = 20 * 1.2 = 24.
        ylim = shared_hi_spectrum_ylim(full, selected)
        self.assertEqual(ylim[0], 0.0)
        self.assertAlmostEqual(ylim[1], 25.2)

    def test_temperature_masks_are_strict_at_3000_K(self):
        temperature = np.array([2999.0, 3000.0, 3001.0])
        np.testing.assert_array_equal(
            temperature_selection_mask(temperature, 'lt', 3000.0),
            [True, False, False],
        )
        np.testing.assert_array_equal(
            temperature_selection_mask(temperature, 'gt', 3000.0),
            [False, False, True],
        )

    def test_complement_masks_are_disjoint_and_cover_every_cell(self):
        temperature = np.array([2999.0, 3000.0, 3001.0])
        for selected_op, complement_op in [
            ('gt', 'le'), ('ge', 'lt'), ('lt', 'ge'),
        ]:
            selected = temperature_selection_mask(
                temperature, selected_op, 3000.0,
            )
            complement = temperature_selection_mask(
                temperature, complement_op, 3000.0,
            )
            np.testing.assert_array_equal(
                selected & complement, np.zeros(3, dtype=bool),
            )
            np.testing.assert_array_equal(
                selected | complement, np.ones(3, dtype=bool),
            )

    def test_complement_cases_match_fields_and_opposite_masks(self):
        selected_by_lum_and_temperature = {
            (case['lum_field'], case['selection_temperature_field']): case
            for case in HI_TEMPERATURE_SPECTRUM_CONFIG
        }
        opposite = {'gt': 'le', 'ge': 'lt', 'lt': 'ge'}
        self.assertEqual(len(HI_TEMPERATURE_COMPLEMENT_CONFIG), 3)

        for complement in HI_TEMPERATURE_COMPLEMENT_CONFIG:
            key = (
                complement['lum_field'],
                complement['selection_temperature_field'],
            )
            selected = selected_by_lum_and_temperature[key]
            self.assertEqual(
                complement['selection_operator'],
                opposite[selected['selection_operator']],
            )
            self.assertEqual(
                complement['width_field'], selected['width_field'],
            )

    def test_three_cases_use_requested_nhi_and_temperature_fields(self):
        by_name = {case['name']: case for case in HI_TEMPERATURE_SPECTRUM_CONFIG}
        self.assertEqual(HI_TEMPERATURE_SPECTRUM_LOS, ('y',))
        self.assertEqual(len(by_name), 3)
        qk_hot = by_name['HI_QUOKKA_TQK_GE3000']
        dsp_tdsp = by_name['HI_DESPOTIC_TDSP_LT3000']
        dsp_tqk = by_name['HI_DESPOTIC_TQK_LT3000']

        self.assertEqual(qk_hot['lum_field'], 'HI_luminosity_quokka')
        self.assertEqual(qk_hot['selection_temperature_field'], 'temperature_quokka')
        self.assertEqual(qk_hot['selection_operator'], 'ge')
        self.assertEqual(qk_hot['width_field'], 'HI_thermal_width_quokka')

        self.assertEqual(dsp_tdsp['lum_field'], 'HI_luminosity_despotic')
        self.assertEqual(dsp_tdsp['selection_temperature_field'], 'temperature_despotic')
        self.assertEqual(dsp_tdsp['selection_operator'], 'lt')
        self.assertEqual(dsp_tdsp['width_field'], 'HI_thermal_width_despotic')

        self.assertEqual(dsp_tqk['lum_field'], 'HI_luminosity_despotic')
        self.assertEqual(dsp_tqk['selection_temperature_field'], 'temperature_quokka')
        self.assertEqual(dsp_tqk['selection_operator'], 'lt')
        self.assertEqual(dsp_tqk['width_field'], 'HI_thermal_width_quokka')

    def test_spectrum_store_applies_selection_before_cell_sum(self):
        shape = (2, 1, 1)
        fields = {
            ('boxlib', 'dx'): yt.YTArray(np.ones(shape), 'cm'),
            ('boxlib', 'dy'): yt.YTArray(np.ones(shape), 'cm'),
            ('boxlib', 'dz'): yt.YTArray(np.ones(shape), 'cm'),
            ('gas', 'Bulk_Doppler_factor_x'): yt.YTArray(np.ones(shape), ''),
            ('gas', 'test_lum'): yt.YTArray(
                np.array([1.0, 2.0]).reshape(shape), 'erg/s/cm**3'
            ),
            ('gas', 'test_width'): yt.YTArray(np.full(shape, 1.0e4), 'cm/s'),
            ('gas', 'test_freq'): yt.YTArray(
                np.full(shape, 1.420405751768e9), 'Hz'
            ),
            ('gas', 'test_temperature'): yt.YTArray(
                np.array([3000.0, 3001.0]).reshape(shape), 'K'
            ),
        }
        config = ({
            'name': 'TEST_HI',
            'lum_field': 'test_lum',
            'width_field': 'test_width',
            'freq_field': 'test_freq',
            'selection_temperature_field': 'test_temperature',
            'selection_operator': 'gt',
            'selection_cutoff_K': 3000.0,
        },)
        store = SpectrumStore(_FakeProvider(fields), species_config=config)
        velocity, spectrum = store.get_spectrum('TEST_HI', 'x', R=float('inf'))

        # Only the second cell (luminosity 2 erg/s) survives.  The spectrum is
        # per projected area, which is 1 cm^2 for this 2x1x1 grid and x LOS.
        # Integrating dSigma_L/dv over velocity must recover 2 erg/s/cm^2.
        dv_kms = abs(float(velocity[1] - velocity[0]))
        spectrum_cgs = spectrum.to('erg/s/cm**2/(km/s)')
        self.assertAlmostEqual(float(spectrum_cgs.sum() * dv_kms), 2.0, places=11)

    def test_canonical_hi_spectrum_equals_green_plus_hot_quokka(self):
        shape = (2, 1, 1)
        temperature = np.array([2999.0, 3001.0]).reshape(shape)
        despotic_lum = np.array([3.0, 30.0]).reshape(shape)
        quokka_lum = np.array([20.0, 2.0]).reshape(shape)
        hybrid_lum = np.where(temperature < 3000.0, despotic_lum, quokka_lum)
        fields = {
            ('boxlib', 'dx'): yt.YTArray(np.ones(shape), 'cm'),
            ('boxlib', 'dy'): yt.YTArray(np.ones(shape), 'cm'),
            ('boxlib', 'dz'): yt.YTArray(np.ones(shape), 'cm'),
            ('gas', 'Bulk_Doppler_factor_y'): yt.YTArray(np.ones(shape), ''),
            ('gas', 'HI_luminosity_despotic'): yt.YTArray(
                despotic_lum, 'erg/s/cm**3'
            ),
            ('gas', 'HI_luminosity_quokka'): yt.YTArray(
                quokka_lum, 'erg/s/cm**3'
            ),
            ('gas', 'HI_luminosity_two_regime'): yt.YTArray(
                hybrid_lum, 'erg/s/cm**3'
            ),
            ('gas', 'HI_thermal_width_quokka'): yt.YTArray(
                np.full(shape, 1.0e4), 'cm/s'
            ),
            ('gas', 'HI_freq'): yt.YTArray(
                np.full(shape, 1.420405751768e9), 'Hz'
            ),
            ('gas', 'temperature_quokka'): yt.YTArray(temperature, 'K'),
        }
        by_name = {case['name']: case for case in HI_TEMPERATURE_SPECTRUM_CONFIG}
        pieces = (
            by_name['HI_DESPOTIC_TQK_LT3000'],
            by_name['HI_QUOKKA_TQK_GE3000'],
        )
        hybrid_config = next(case for case in SPECIES_CFG if case['name'] == 'HI')
        provider = _FakeProvider(fields)

        hybrid_store = SpectrumStore(provider, species_config=(hybrid_config,))
        _, hybrid = hybrid_store.get_spectrum('HI', 'y', R=float('inf'))

        piece_store = SpectrumStore(provider, species_config=pieces)
        _, cold = piece_store.get_spectrum(
            'HI_DESPOTIC_TQK_LT3000', 'y', R=float('inf')
        )
        _, hot = piece_store.get_spectrum(
            'HI_QUOKKA_TQK_GE3000', 'y', R=float('inf')
        )
        np.testing.assert_allclose(hybrid, cold + hot, rtol=1.0e-13)

    def test_unknown_selection_operator_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unknown temperature selection'):
            temperature_selection_mask(np.array([1.0]), 'eq', 3000.0)


if __name__ == '__main__':
    unittest.main()
