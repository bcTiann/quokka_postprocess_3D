"""Small endpoint shifts may clamp; genuinely uncovered cells still fail."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from quokka2s.physics.cell_emission import CellEmissionCalculator
from quokka2s.physics.line_emissivity import ATOMIC_LINE_KEYS, CO_LINE_KEYS
from quokka2s.despotic.cell_fields import DespoticCellReader
from quokka2s.constants import HYDROGEN_MASS_G
from quokka2s.physics import settings as cfg
from quokka2s.despotic.lookup import DespoticLookup
from quokka2s.snapshot_reader import CellBatch


class LookupReached(Exception):
    """Stop the test after checking the actual coordinates sent to the table."""


class CellEmissionDomainTests(unittest.TestCase):
    def setUp(self):
        self.table = SimpleNamespace(
            nH_values=np.array([1., 10.]),
            col_density_values=np.array([1e18, 1e21]),
            dVdr_values=np.array([1e-16, 1e-13]),
        )
        self.lookup = Mock(side_effect=LookupReached)
        despotic = SimpleNamespace(table=self.table, temperature_and_co=self.lookup)
        despotic.clip_coordinates = lambda **coordinates: DespoticLookup.clip_coordinates(
            despotic, **coordinates,
        )
        self.calculator = CellEmissionCalculator(
            despotic_reader=DespoticCellReader(lookup=despotic),
            cloudy_reader=None,
            line_keys=ATOMIC_LINE_KEYS + CO_LINE_KEYS,
            dust_cross_section_cm2_H={key: 0.0 for key in ATOMIC_LINE_KEYS + CO_LINE_KEYS},
        )

    def cells(self, n_h, column, gradient):
        return CellBatch(
            density_g_cm3=np.array([n_h * HYDROGEN_MASS_G / cfg.X_H]),
            temperature_QUOKKA_K=np.array([100.]),
            shielding_NH_cm2=np.array([column]),
            velocity_gradient_s=np.array([gradient]),
            foreground_NH_cm2=np.zeros(1),
            velocity_z_kms=np.zeros(1),
            x_start=0,
            slab_shape=(1, 1, 1),
            batch_start=0,
            cell_volume_cm3=1.,
        )

    def test_tiny_endpoint_offsets_clamp_only_lookup_coordinates(self):
        for coordinates, expected in (
            ((1.-9e-8, 1e18*(1.-9e-8), 1e-16), (1., 1e18, 1e-16)),
            ((10.*(1.+9e-8), 1e21*(1.+9e-8), 1e-13), (10., 1e21, 1e-13)),
        ):
            with self.subTest(coordinates=coordinates):
                cells = self.cells(*coordinates)
                original = {
                    name: value.copy()
                    for name, value in vars(cells).items()
                    if isinstance(value, np.ndarray)
                }
                with self.assertRaises(LookupReached):
                    self.calculator.calculate(cells=cells)
                for actual, endpoint in zip(self.lookup.call_args.kwargs.values(), expected):
                    np.testing.assert_array_equal(actual, [endpoint])
                for name, values in original.items():
                    np.testing.assert_array_equal(getattr(cells, name), values)

    def test_larger_excursions_are_rejected_before_table_lookup(self):
        for coordinates in (
            (.999, 1e19, 1e-15), (10.01, 1e19, 1e-15),
            (2., .999e18, 1e-15), (2., 1.001e21, 1e-15),
            (2., 1e19, .999e-16), (2., 1e19, 1.001e-13),
        ):
            with self.subTest(coordinates=coordinates):
                with self.assertRaisesRegex(ValueError, 'outside the table domain'):
                    self.calculator.calculate(cells=self.cells(*coordinates))
        self.lookup.assert_not_called()

    def test_nonfinite_or_nonpositive_coordinates_still_fail(self):
        for value in (np.nan, np.inf, 0., -1.):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'outside the table domain'):
                    self.calculator.calculate(
                        cells=self.cells(value, 1e19, 1e-15),
                    )
        self.lookup.assert_not_called()


if __name__ == '__main__':
    unittest.main()
