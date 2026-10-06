"""The semantic stages preserve branches, masks, units, and lookup reuse."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from quokka2s.constants import HYDROGEN_MASS_G
from quokka2s.physics.settings import X_H
from quokka2s.physics.cell_emission import (
    IntrinsicLineEmission,
    LineEmission,
    CellEmissionCalculator,
)
from quokka2s.physics.line_emissivity import (
    ATOMIC_LINE_KEYS,
    CIII_CIV_LINE_KEYS,
)
from quokka2s.physics.hydrogen_emissivity import (
    HALPHA_PHOTON_ENERGY_ERG,
    effective_halpha_recombination_coefficient,
    hi21_emissivity,
)
from quokka2s.despotic.cell_fields import DespoticCellReader
from quokka2s.despotic.lookup import DespoticTemperatureCO, DespoticLookup
from quokka2s.cloudy.cell_fields import CloudyCellReader
from quokka2s.cloudy.lookup import CloudyLookup
from quokka2s.cloudy.table_definition import EXPECTED_AXIS_ORDER
from quokka2s.snapshot_reader import CellBatch


class StubDespotic:
    clip_coordinates = DespoticLookup.clip_coordinates
    prepare_queries = DespoticLookup.prepare_queries
    _EVAL_CHUNK = DespoticLookup._EVAL_CHUNK

    def __init__(self):
        self.table = SimpleNamespace(
            nH_values=np.array([1., 10.]),
            col_density_values=np.array([1e18, 1e21]),
            dVdr_values=np.array([1e-16, 1e-13]),
        )
        self.calls = []
        self.temperature_K = np.array([50., 50., 50., np.nan])
        self.bad_line = None
        self.bad_density = None

    def temperature_and_co(self, *, queries):
        hydrogen_density_cm3 = queries.hydrogen_density_cm3
        self.calls.append(('temperature_and_co', hydrogen_density_cm3.copy()))
        co10 = np.full(hydrogen_density_cm3.shape, 3e-24)
        co21 = np.full(hydrogen_density_cm3.shape, 5e-24)
        if self.bad_line == 'CO':
            co10[:] = np.nan
        if self.bad_line == 'CO21':
            co21[:] = np.nan
        return DespoticTemperatureCO(self.temperature_K.copy(), co10, co21)

    def line_field(self, *, species, field_name, queries):
        hydrogen_density_cm3 = queries.hydrogen_density_cm3
        self.calls.append((species, hydrogen_density_cm3.copy()))
        assert species == 'C+' and field_name == 'lumPerH'
        return np.full(hydrogen_density_cm3.shape, np.nan if self.bad_line == 'C+' else 2e-24)

    def number_densities(self, *, species, queries):
        hydrogen_density_cm3 = queries.hydrogen_density_cm3
        self.calls.append(('number_densities', hydrogen_density_cm3.copy()))
        factors = {'e-': .02, 'H+': .03, 'H': .7}
        return {
            key: np.full(hydrogen_density_cm3.shape, np.nan) if key == self.bad_density else hydrogen_density_cm3 * factors[key]
            for key in species
        }


class LineEmissivityTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'cloudy.npz'
        shape = (8, 2, 2, 3)
        coefficients = np.broadcast_to(np.arange(1., 9.)[:, None, None, None] * 1e-28, shape).copy()
        self.payload = dict(
            axis_order=EXPECTED_AXIS_ORDER,
            line_keys=ATOMIC_LINE_KEYS,
            log_NH_attenuation=[18., 21.],
            log_nH=[-1., 2.],
            log_T=[np.log10(50.), np.log10(3000.), 4.],
            emissivity_per_nH2=coefficients,
            log_emissivity_per_nH2=np.log10(coefficients),
            failure_mask=np.zeros(shape, dtype=bool),
            zero_mask=np.zeros(shape, dtype=bool),
        )
        nH = np.array([1. - 5e-8, 2., 10. * (1. + 5e-8), 3.])
        self.cells = CellBatch(
            density_g_cm3=nH * HYDROGEN_MASS_G / X_H,
            temperature_QUOKKA_K=np.array([2999., 3000., 100., 100.]),
            shielding_NH_cm2=np.array([1e18 * (1. - 5e-8), 1e19, 1e21 * (1. + 5e-8), 1e20]),
            velocity_gradient_s=np.array([1e-16 * (1. - 5e-8), 1e-15, 1e-13 * (1. + 5e-8), 1e-15]),
            foreground_NH_cm2=np.full(4, 1e20),
            velocity_z_kms=np.zeros(4),
            x_start=0,
            slab_shape=(1, 1, 4),
            batch_start=0,
            cell_volume_cm3=1.,
        )
        self.despotic = StubDespotic()
        self.line_keys = ATOMIC_LINE_KEYS + ('co10', 'co21')
        self.sigma = np.full(10, 1e-21)
        self.sigma[2] = 0.

    def cloudy_lookup(self):
        failed = self.payload['failure_mask']
        self.payload['emissivity_per_nH2'][failed] = 0.
        self.payload['log_emissivity_per_nH2'][failed] = np.nan
        np.savez(self.path, **self.payload)
        return CloudyLookup(self.path)

    def calculator(self, line_keys=None):
        if line_keys is None:
            line_keys = self.line_keys
        cross_sections = {}
        # Dust values stay attached to names even when output order changes.
        for line_key, cross_section in zip(self.line_keys, self.sigma):
            cross_sections[line_key] = float(cross_section)
        return CellEmissionCalculator(
            despotic_reader=DespoticCellReader(lookup=self.despotic),
            cloudy_reader=CloudyCellReader(lookup=self.cloudy_lookup()),
            line_keys=line_keys,
            dust_cross_section_cm2_H=cross_sections,
        )

    def calculate(self):
        return self.calculator().calculate(cells=self.cells)

    def query_fields(self):
        nH = self.cells.hydrogen_density_cm3
        cold = self.cells.temperature_QUOKKA_K < 3000.
        despotic_reader = DespoticCellReader(lookup=self.despotic)
        despotic = despotic_reader.read_fields(
            cells=self.cells,
            cold_cells=cold,
        )
        cloudy_reader = CloudyCellReader(lookup=self.cloudy_lookup())
        cloudy = cloudy_reader.read_fields(
            cells=self.cells,
            selected_cells=~cold,
        )
        return nH, cold, despotic, cloudy

    def test_branches_and_thermal_temperatures(self):
        # Missing cold Cloudy support must never be queried.
        self.payload['failure_mask'][:, :, :, 0] = True
        result = self.calculate()
        for line_index, line_key in enumerate(ATOMIC_LINE_KEYS):
            np.testing.assert_allclose(
                result.lines[line_key].intrinsic_emissivity_erg_s_cm3[1],
                (line_index + 1) * 4e-28,
                atol=0,
            )
        cold = [0, 2]
        query_nH = np.array([1., 10.])
        np.testing.assert_allclose(result.lines['cii'].intrinsic_emissivity_erg_s_cm3[cold], query_nH * 2e-24, atol=0)
        expected_halpha = (HALPHA_PHOTON_ENERGY_ERG * effective_halpha_recombination_coefficient(50.)
                           * (.02 * query_nH) * (.03 * query_nH))
        np.testing.assert_array_equal(result.lines['halpha'].intrinsic_emissivity_erg_s_cm3[cold], expected_halpha)
        np.testing.assert_array_equal(result.lines['hi21'].intrinsic_emissivity_erg_s_cm3[cold], hi21_emissivity(.7 * query_nH))
        for key in CIII_CIV_LINE_KEYS:
            np.testing.assert_array_equal(result.lines[key].intrinsic_emissivity_erg_s_cm3[cold], 0.)
        np.testing.assert_allclose(result.lines['co10'].intrinsic_emissivity_erg_s_cm3[:3], [3e-24, 6e-24, 30e-24], atol=0)
        np.testing.assert_allclose(result.lines['co21'].intrinsic_emissivity_erg_s_cm3[:3], [5e-24, 10e-24, 50e-24], atol=0)
        for line_key in ('cii', 'halpha', 'hi21'):
            np.testing.assert_array_equal(result.lines[line_key].temperature_K[:3], [50., 3000., 50.])
        for line_key in CIII_CIV_LINE_KEYS:
            np.testing.assert_array_equal(
                result.lines[line_key].temperature_K,
                self.cells.temperature_QUOKKA_K,
            )
        for line_key in ('co10', 'co21'):
            np.testing.assert_array_equal(result.lines[line_key].temperature_K[:3], 50.)

    def test_named_fields_have_original_cell_order_and_cold_only_densities(self):
        _, _, fields, _ = self.query_fields()
        for name in ('temperature_K', 'co10_luminosity_per_H', 'co21_luminosity_per_H',
                     'cii_luminosity_per_H', 'electron_density_cm3',
                     'ionized_hydrogen_density_cm3', 'neutral_hydrogen_density_cm3'):
            self.assertEqual(getattr(fields, name).shape, (4,))
        np.testing.assert_array_equal(fields.electron_density_cm3[[0, 2]], [.02, .2])
        self.assertTrue(np.isnan(fields.electron_density_cm3[[1, 3]]).all())
        self.assertEqual([name for name, _ in self.despotic.calls],
                         ['temperature_and_co', 'C+', 'number_densities'])
        np.testing.assert_array_equal(self.despotic.calls[-1][1], [1., 10.])

    def test_calculation_after_querying_does_not_access_tables(self):
        _, cold, despotic, cloudy = self.query_fields()
        before = {name: value.copy() for name, value in vars(despotic).items() if isinstance(value, np.ndarray)}
        calculator = self.calculator()
        with (
            patch.object(self.despotic, 'temperature_and_co', side_effect=AssertionError('repeated lookup')),
            patch.object(self.despotic, 'number_densities', side_effect=AssertionError('repeated lookup')),
            patch.object(self.despotic, 'line_field', side_effect=AssertionError('repeated lookup')),
            patch.object(
                calculator.cloudy_reader.lookup,
                'interpolate_available',
                side_effect=AssertionError('repeated lookup'),
            ),
        ):
            intrinsic_lines = calculator.calculate_intrinsic_lines(
                cells=self.cells,
                cold_cells=cold,
                despotic_fields=despotic,
                cloudy_fields=cloudy,
            )
            lines = calculator.apply_foreground_dust(
                intrinsic_lines=intrinsic_lines,
                foreground_column_cm2=self.cells.foreground_NH_cm2,
            )
        expected = self.calculate()
        for line_key in self.line_keys:
            intrinsic_line = intrinsic_lines[line_key]
            line = lines[line_key]
            np.testing.assert_array_equal(
                intrinsic_line.emissivity_erg_s_cm3,
                expected.lines[line_key].intrinsic_emissivity_erg_s_cm3,
            )
            np.testing.assert_array_equal(
                intrinsic_line.temperature_K,
                expected.lines[line_key].temperature_K,
            )
            np.testing.assert_array_equal(
                line.attenuated_emissivity_erg_s_cm3,
                expected.lines[line_key].attenuated_emissivity_erg_s_cm3,
            )
            self.assertIs(line.temperature_K, intrinsic_line.temperature_K)
        for name, values in before.items():
            np.testing.assert_array_equal(getattr(despotic, name), values)

    def test_line_specific_missing_values_clipping_and_original_cells_are_preserved(self):
        original = {
            name: value.copy()
            for name, value in vars(self.cells).items()
            if isinstance(value, np.ndarray)
        }
        result = self.calculate()
        self.assertFalse(hasattr(result, 'valid_cells'))
        self.assertEqual(result.despotic_coordinate_clipped_cells,
                         {'nH': 2, 'NH': 2, 'dVdr': 2})
        for line_key in ('cii', 'halpha', 'hi21', 'co10', 'co21'):
            line = result.lines[line_key]
            np.testing.assert_array_equal(line.emissivity_is_missing, [False, False, False, True])
            self.assertTrue(np.isnan(line.attenuated_emissivity_erg_s_cm3[3]))
        for line_key in CIII_CIV_LINE_KEYS:
            line = result.lines[line_key]
            self.assertFalse(line.emissivity_is_missing.any())
            self.assertEqual(line.intrinsic_emissivity_erg_s_cm3[3], 0.)
            self.assertEqual(line.attenuated_emissivity_erg_s_cm3[3], 0.)
        for name, values in original.items():
            np.testing.assert_array_equal(getattr(self.cells, name), values)

    def test_invalid_required_line_and_density_values_raise(self):
        for species in ('C+', 'CO', 'CO21'):
            with self.subTest(species=species):
                self.despotic.bad_line = species
                with self.assertRaisesRegex(ValueError, 'lumPerH must be finite'):
                    self.calculate()
        self.despotic.bad_line = None
        for species in ('e-', 'H+', 'H'):
            with self.subTest(species=species):
                self.despotic.bad_density = species
                with self.assertRaisesRegex(ValueError, 'number density must be finite'):
                    self.calculate()

    def test_zero_emission_is_preserved(self):
        _, cold, despotic, cloudy = self.query_fields()
        despotic = replace(despotic, co10_luminosity_per_H=np.zeros(4))
        intrinsic_lines = self.calculator().calculate_intrinsic_lines(
            cells=self.cells,
            cold_cells=cold,
            despotic_fields=despotic,
            cloudy_fields=cloudy,
        )
        epsilon = intrinsic_lines['co10'].emissivity_erg_s_cm3
        np.testing.assert_array_equal(epsilon[:3], 0.)
        self.assertTrue(np.isnan(epsilon[3]))

    def test_missing_despotic_temperature_preserves_hot_atomic_and_cold_prescribed_zeros(self):
        self.despotic.temperature_K = np.array([0., np.nan, -1., np.inf])
        result = self.calculate()
        for line_key in ('cii', 'halpha', 'hi21'):
            line = result.lines[line_key]
            np.testing.assert_array_equal(line.emissivity_is_missing, [True, False, True, True])
            self.assertEqual(line.temperature_K[1], 3000.)
        for line_key in CIII_CIV_LINE_KEYS:
            line = result.lines[line_key]
            self.assertFalse(line.emissivity_is_missing.any())
            np.testing.assert_array_equal(line.intrinsic_emissivity_erg_s_cm3[[0, 2, 3]], 0.)
            np.testing.assert_array_equal(line.temperature_K, self.cells.temperature_QUOKKA_K)
        for line_key in ('co10', 'co21'):
            line = result.lines[line_key]
            self.assertTrue(line.emissivity_is_missing.all())
            self.assertTrue(np.isnan(line.temperature_K).all())
        self.assertEqual([name for name, _ in self.despotic.calls], ['temperature_and_co'])

    def test_hot_cloudy_failure_does_not_remove_despotic_co(self):
        self.payload['failure_mask'][0, :, :, 1] = True
        result = self.calculate()
        for line_key in ATOMIC_LINE_KEYS:
            self.assertTrue(result.lines[line_key].emissivity_is_missing[1])
        for line_key in ('co10', 'co21'):
            line = result.lines[line_key]
            self.assertFalse(line.emissivity_is_missing[1])
            self.assertEqual(line.temperature_K[1], 50.)
        self.assertEqual(result.lines['co10'].intrinsic_emissivity_erg_s_cm3[1], 6e-24)
        np.testing.assert_array_equal(result.despotic_temperature_K, [50., 50., 50., np.nan])

    def test_cloudy_failure_does_not_hide_invalid_required_co_values(self):
        self.payload['failure_mask'][0, :, :, 1] = True
        original = self.despotic.temperature_and_co
        def sample(*args, **kwargs):
            result = original(*args, **kwargs)
            result.co10_luminosity_per_H[1] = np.nan
            return result
        self.despotic.temperature_and_co = sample
        with self.assertRaisesRegex(ValueError, 'CO lumPerH must be finite'):
            self.calculate()

    def test_hot_missing_despotic_fields_do_not_block_cloudy_or_its_dust_attenuation(self):
        self.cells.temperature_QUOKKA_K[3] = 10000.
        result = self.calculate()
        for line_index, line_key in enumerate(ATOMIC_LINE_KEYS):
            line = result.lines[line_key]
            self.assertFalse(line.emissivity_is_missing[3])
            self.assertEqual(line.temperature_K[3], 10000.)
            expected = (line_index + 1) * 1e-28 * self.cells.hydrogen_density_cm3[3] ** 2
            np.testing.assert_allclose(line.intrinsic_emissivity_erg_s_cm3[3], expected, rtol=1e-14)
            self.assertEqual(
                line.attenuated_emissivity_erg_s_cm3[3],
                line.intrinsic_emissivity_erg_s_cm3[3] * np.exp(-self.sigma[line_index] * 1e20),
            )
        for line_key in ('co10', 'co21'):
            self.assertTrue(result.lines[line_key].emissivity_is_missing[3])

    def test_hot_cells_skip_cold_cii_and_particle_queries(self):
        self.cells.temperature_QUOKKA_K[:] = 3000.
        result = self.calculate()
        self.assertEqual([name for name, _ in self.despotic.calls], ['temperature_and_co'])
        for line_key in ATOMIC_LINE_KEYS:
            np.testing.assert_array_equal(result.lines[line_key].temperature_K[:3], 3000.)
        for line_key in ('co10', 'co21'):
            np.testing.assert_array_equal(result.lines[line_key].temperature_K[:3], 50.)

    def test_despotic_temperature_does_not_select_atomic_branch(self):
        self.despotic.temperature_K[0] = 1e4
        result = self.calculate()
        self.assertTrue(result.cold_cells[0])
        self.assertEqual(result.lines['halpha'].temperature_K[0], 1e4)
        self.assertEqual(result.lines['ciii_977'].intrinsic_emissivity_erg_s_cm3[0], 0.)

    def test_dust_uses_foreground_column_and_leaves_hi_unchanged(self):
        result = self.calculate()
        for line_index, line_key in enumerate(self.line_keys):
            line = result.lines[line_key]
            expected = line.intrinsic_emissivity_erg_s_cm3[:3] * np.exp(-self.sigma[line_index] * 1e20)
            np.testing.assert_array_equal(line.attenuated_emissivity_erg_s_cm3[:3], expected)
        hi = result.lines['hi21']
        np.testing.assert_array_equal(hi.attenuated_emissivity_erg_s_cm3, hi.intrinsic_emissivity_erg_s_cm3)

    def test_source_and_output_row_orders_are_matched_by_line_name(self):
        canonical = self.calculate()
        _, cold, despotic, cloudy = self.query_fields()

        # Cloudy source order and output order are deliberately different.
        # CO is interleaved with atomic lines in the output, not placed last.
        source_rows = [6, 2, 0, 7, 4, 1, 5, 3]
        source_line_keys = tuple(cloudy.emissivity_per_nH2)
        reordered_cloudy_emissivity = {}
        for source_row in source_rows:
            line_key = source_line_keys[source_row]
            reordered_cloudy_emissivity[line_key] = cloudy.emissivity_per_nH2[line_key]
        cloudy = replace(
            cloudy,
            emissivity_per_nH2=reordered_cloudy_emissivity,
        )
        output_line_keys = (
            'co21', 'halpha', 'ciii_1907', 'cii', 'co10',
            'civ_1551', 'hi21', 'ciii_977', 'civ_1548', 'ciii_1909',
        )
        calculator = self.calculator(line_keys=output_line_keys)
        intrinsic_lines = calculator.calculate_intrinsic_lines(
            cells=self.cells,
            cold_cells=cold,
            despotic_fields=despotic,
            cloudy_fields=cloudy,
        )
        self.assertEqual(tuple(intrinsic_lines), output_line_keys)
        for line_key in output_line_keys:
            np.testing.assert_array_equal(
                intrinsic_lines[line_key].emissivity_erg_s_cm3,
                canonical.lines[line_key].intrinsic_emissivity_erg_s_cm3,
            )
            np.testing.assert_array_equal(
                intrinsic_lines[line_key].temperature_K,
                canonical.lines[line_key].temperature_K,
            )

    def test_hot_atomic_temperature_stays_paired_with_cloudy_query(self):
        _, cold, despotic, cloudy = self.query_fields()
        # A later cell temperature must not replace the queried gas state's T.
        cells_after_query = replace(
            self.cells,
            temperature_QUOKKA_K=np.array([2999., 4000., 100., 100.]),
        )
        intrinsic_lines = self.calculator().calculate_intrinsic_lines(
            cells=cells_after_query,
            cold_cells=cold,
            despotic_fields=despotic,
            cloudy_fields=cloudy,
        )
        self.assertNotEqual(cells_after_query.temperature_QUOKKA_K[1], cloudy.temperature_K[1])
        for line_key in ATOMIC_LINE_KEYS:
            line = intrinsic_lines[line_key]
            self.assertEqual(line.temperature_K[1], cloudy.temperature_K[1])
            np.testing.assert_array_equal(
                line.emissivity_erg_s_cm3[1],
                cloudy.emissivity_per_nH2[line_key][1]
                * cells_after_query.hydrogen_density_cm3[1] ** 2,
            )
        for line_key in ('co10', 'co21'):
            self.assertEqual(intrinsic_lines[line_key].temperature_K[1], despotic.temperature_K[1])

    def test_hot_co_uses_query_density_and_despotic_temperature(self):
        # This hot cell is just above DESPOTIC's upper nH endpoint. Cloudy
        # covers its physical nH; DESPOTIC clips only its query nH to 10.
        self.cells.density_g_cm3[1] = 10. * (1. + 5e-8) * HYDROGEN_MASS_G / X_H
        result = self.calculate()
        _, _, _, cloudy = self.query_fields()
        physical_nH = self.cells.hydrogen_density_cm3[1]
        self.assertGreater(physical_nH, 10.)

        np.testing.assert_array_equal(
            result.lines['co10'].intrinsic_emissivity_erg_s_cm3[1],
            10. * 3e-24,
        )
        for line_key, emissivity_per_nH2 in cloudy.emissivity_per_nH2.items():
            np.testing.assert_array_equal(
                result.lines[line_key].intrinsic_emissivity_erg_s_cm3[1],
                emissivity_per_nH2[1] * physical_nH ** 2,
            )
            self.assertEqual(result.lines[line_key].temperature_K[1], 3000.)
        for line_key in ('co10', 'co21'):
            self.assertEqual(result.lines[line_key].temperature_K[1], 50.)

    def test_empty_batch_preserves_named_output_shapes(self):
        _, _, despotic, cloudy = self.query_fields()
        despotic_arrays = {}
        for name, value in vars(despotic).items():
            if isinstance(value, np.ndarray):
                despotic_arrays[name] = value[:0]
        despotic = replace(despotic, **despotic_arrays)
        empty_cloudy_emissivity = {}
        for line_key, values in cloudy.emissivity_per_nH2.items():
            empty_cloudy_emissivity[line_key] = values[:0]
        cloudy = replace(
            cloudy,
            emissivity_per_nH2=empty_cloudy_emissivity,
            temperature_K=cloudy.temperature_K[:0],
            failed_cells=cloudy.failed_cells[:0],
        )
        empty_cells = replace(
            self.cells,
            density_g_cm3=self.cells.density_g_cm3[:0],
            temperature_QUOKKA_K=self.cells.temperature_QUOKKA_K[:0],
            foreground_NH_cm2=self.cells.foreground_NH_cm2[:0],
            shielding_NH_cm2=self.cells.shielding_NH_cm2[:0],
            velocity_gradient_s=self.cells.velocity_gradient_s[:0],
            velocity_z_kms=self.cells.velocity_z_kms[:0],
        )
        empty_mask = np.empty(0, dtype=bool)
        calculator = self.calculator()
        intrinsic_lines = calculator.calculate_intrinsic_lines(
            cells=empty_cells,
            cold_cells=empty_mask,
            despotic_fields=despotic,
            cloudy_fields=cloudy,
        )
        self.assertEqual(tuple(intrinsic_lines), self.line_keys)
        for line_key in self.line_keys:
            self.assertEqual(intrinsic_lines[line_key].emissivity_erg_s_cm3.shape, (0,))
            self.assertEqual(intrinsic_lines[line_key].temperature_K.shape, (0,))

    def test_named_line_records_reference_results_without_copying(self):
        _, cold, despotic, cloudy = self.query_fields()
        calculator = self.calculator()
        intrinsic_lines = calculator.calculate_intrinsic_lines(
            cells=self.cells,
            cold_cells=cold,
            despotic_fields=despotic,
            cloudy_fields=cloudy,
        )
        lines = calculator.apply_foreground_dust(
            intrinsic_lines=intrinsic_lines,
            foreground_column_cm2=self.cells.foreground_NH_cm2,
        )
        result = calculator.build_batch_emission(
            lines=lines,
            despotic_fields=despotic,
            cloudy_fields=cloudy,
            cold_cells=cold,
        )
        intrinsic = {}
        attenuated = {}
        temperature = {}
        for line_key, line in result.lines.items():
            intrinsic_line = intrinsic_lines[line_key]
            self.assertIsInstance(intrinsic_line, IntrinsicLineEmission)
            self.assertIsInstance(line, LineEmission)
            self.assertEqual(line.intrinsic_emissivity_erg_s_cm3.shape, (4,))
            self.assertIs(line.intrinsic_emissivity_erg_s_cm3, intrinsic_line.emissivity_erg_s_cm3)
            self.assertIs(line.temperature_K, intrinsic_line.temperature_K)
            intrinsic[line_key] = line.intrinsic_emissivity_erg_s_cm3
            attenuated[line_key] = line.attenuated_emissivity_erg_s_cm3
            temperature[line_key] = line.temperature_K
        rebuilt = calculator.build_batch_emission(
            lines=result.lines,
            despotic_fields=despotic,
            cloudy_fields=cloudy,
            cold_cells=result.cold_cells,
        )
        for line_key, line in rebuilt.lines.items():
            self.assertIs(line, result.lines[line_key])
            self.assertIs(line.intrinsic_emissivity_erg_s_cm3, intrinsic[line_key])
            self.assertIs(line.attenuated_emissivity_erg_s_cm3, attenuated[line_key])
            self.assertIs(line.temperature_K, temperature[line_key])
        self.assertFalse(hasattr(result, 'emissivity_erg_s_cm3'))
        self.assertFalse(hasattr(result, 'temperature_K'))

    def test_reused_calculator_retains_configuration_without_batch_state(self):
        calculator = self.calculator()
        initial_attributes = vars(calculator).copy()
        first_result = calculator.calculate(cells=self.cells)
        first_attenuated = first_result.lines['halpha'].attenuated_emissivity_erg_s_cm3.copy()

        self.cells.foreground_NH_cm2[:] = 2e20
        second_result = calculator.calculate(cells=self.cells)

        self.assertEqual(set(vars(calculator)), set(initial_attributes))
        for name, value in initial_attributes.items():
            self.assertIs(getattr(calculator, name), value)
        np.testing.assert_array_equal(
            first_result.lines['halpha'].attenuated_emissivity_erg_s_cm3,
            first_attenuated,
        )
        self.assertTrue(np.all(
            second_result.lines['halpha'].attenuated_emissivity_erg_s_cm3[:3]
            < first_attenuated[:3]
        ))


if __name__ == '__main__':
    unittest.main()
