"""Physical table checks remain independent of source-file identities."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from quokka2s.physics.cell_emission import CellEmissionCalculator
from quokka2s.physics.line_emissivity import ATOMIC_LINE_KEYS, CO_LINE_KEYS
from quokka2s.despotic.cell_fields import DespoticCellReader
from quokka2s.despotic.snapshot_domain import validate_snapshot_domain
from quokka2s.cloudy.cell_fields import CloudyCellReader
from quokka2s.processing_inputs import (
    load_emission_calculator,
    prepare_line_dust_cross_sections,
)


class EmissionInputTests(unittest.TestCase):
    def test_calculator_keeps_declared_line_order_and_named_dust_cross_sections(self):
        line_keys = ATOMIC_LINE_KEYS + CO_LINE_KEYS
        cloudy = SimpleNamespace(line_keys=list(ATOMIC_LINE_KEYS))
        dust = {name: float(index + 1) for index, name in enumerate(reversed(line_keys))}
        calculator = CellEmissionCalculator(
            despotic_reader=DespoticCellReader(lookup=object()),
            cloudy_reader=CloudyCellReader(lookup=cloudy),
            line_keys=line_keys,
            dust_cross_section_cm2_H=dust,
        )
        self.assertEqual(calculator.line_keys, line_keys)
        self.assertIs(calculator.dust_cross_section_cm2_H, dust)
        self.assertEqual(calculator.dust_cross_section_cm2_H['halpha'], dust['halpha'])
        # The declared output order and named dust values do not follow mutations
        # of the separately loaded Cloudy line list.
        cloudy.line_keys.reverse()
        self.assertEqual(calculator.line_keys, line_keys)
        self.assertEqual(calculator.dust_cross_section_cm2_H['halpha'], dust['halpha'])

    def test_calculator_loading_chooses_the_line_order_once(self):
        config = SimpleNamespace(
            despotic_table='despotic.npz',
            cloudy_table='cloudy.npz',
            dust_opacity_table='draine.dat',
        )
        snapshot = object()
        despotic = object()
        atomic_order = tuple(reversed(ATOMIC_LINE_KEYS))
        cloudy = SimpleNamespace(line_keys=list(atomic_order))
        expected_order = atomic_order + CO_LINE_KEYS
        dust = {name: float(index) for index, name in enumerate(expected_order)}
        with patch('quokka2s.processing_inputs.load_despotic_lookup', return_value=despotic):
            with patch('quokka2s.processing_inputs.CloudyLookup', return_value=cloudy) as load_cloudy:
                with patch('quokka2s.processing_inputs.prepare_line_dust_cross_sections', return_value=dust) as prepare:
                    calculator = load_emission_calculator(config=config, snapshot=snapshot)
        load_cloudy.assert_called_once_with(path=config.cloudy_table)
        self.assertEqual(calculator.line_keys, expected_order)
        prepare.assert_called_once_with(
            table_path=config.dust_opacity_table,
            line_keys=expected_order,
        )
        self.assertIs(calculator.despotic_reader.lookup, despotic)
        self.assertIs(calculator.cloudy_reader.lookup, cloudy)
        self.assertIs(calculator.dust_cross_section_cm2_H, dust)

    def test_prepared_dust_values_are_accessible_by_line_name(self):
        line_keys = ('hi21', 'halpha', 'cii')
        wavelengths = np.array([0.1, 1.])
        table_cross_sections = np.array([1e-21, 1e-22])
        values = np.array([0., 3.8e-22, 2.2e-25])
        with patch('quokka2s.processing_inputs.load_draine_extinction', return_value=(wavelengths, table_cross_sections)):
            with patch('quokka2s.processing_inputs.extinction_cross_sections', return_value=values) as interpolate:
                dust = prepare_line_dust_cross_sections(
                    table_path='draine.dat',
                    line_keys=line_keys,
                )
        self.assertEqual(dust, {'hi21': 0., 'halpha': 3.8e-22, 'cii': 2.2e-25})
        self.assertTrue(all(isinstance(value, float) for value in dust.values()))
        self.assertEqual(interpolate.call_args.kwargs['line_keys'], line_keys)

    def setUp(self):
        self.shape = (4, 2, 8)
        self.config = SimpleNamespace(
            X_H=.71577, COLUMN_DENSITY_MEAN='harmonic',
            COLUMN_DENSITY_DIRECTIONS='z2ray')
        domain = {
            'selection': 'all simulation cells', 'shape': list(self.shape),
            'total_cells': 64, 'X_H': self.config.X_H,
            'column_mean': self.config.COLUMN_DENSITY_MEAN,
            'column_directions': self.config.COLUMN_DENSITY_DIRECTIONS,
            'axes': {
                'nH': {'minimum': 1., 'maximum': 10.},
                'NH': {'minimum': 1e18, 'maximum': 1e21},
                'dVdr': {'minimum': 1e-20, 'maximum': 1e-10},
            },
        }
        self.table = SimpleNamespace(
            build_metadata={'snapshot_domain': domain},
            nH_values=np.array([1., 10.]),
            col_density_values=np.array([1e18, 1e21]),
            dVdr_values=np.array([1e-20, 1e-10]),
        )

    def test_table_does_not_require_a_matching_source_hash(self):
        validate_snapshot_domain(self.table, self.shape, self.config)
        # A saved table may contain an older source identity; numerical settings
        # determine compatibility, even after source comments or paths change.
        self.table.build_metadata['snapshot_domain']['physics_source_sha256'] = 'old source'
        validate_snapshot_domain(self.table, self.shape, self.config)

    def test_inconsistent_grid_and_physical_settings_still_raise(self):
        for name, wrong in (
            ('shape', [4, 2, 9]), ('total_cells', 65), ('X_H', .76),
            ('column_mean', 'arithmetic'), ('column_directions', 'xyz'),
        ):
            with self.subTest(name=name):
                table = deepcopy(self.table)
                table.build_metadata['snapshot_domain'][name] = wrong
                with self.assertRaisesRegex(ValueError, name):
                    validate_snapshot_domain(table, self.shape, self.config)

    def test_inconsistent_axis_bounds_still_raise(self):
        self.table.nH_values[-1] = 100.
        with self.assertRaisesRegex(ValueError, 'nH bounds'):
            validate_snapshot_domain(self.table, self.shape, self.config)


if __name__ == '__main__':
    unittest.main()
