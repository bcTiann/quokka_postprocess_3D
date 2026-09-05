import unittest

import numpy as np

from quokka2s.tables import ExplicitGrid
from quokka2s.tables.dvdr_domain import (
    LEGACY_DVDR_MAX_S,
    LEGACY_DVDR_MIN_S,
    SIMULATION_DVDR_MAX_S,
    SIMULATION_DVDR_MIN_S,
    added_dvdr_values,
    extended_dvdr_values,
    legacy_dvdr_values,
)


class DvdrDomainTests(unittest.TestCase):
    def test_explicit_grid_accepts_one_checkpoint_node(self):
        self.assertTrue(np.array_equal(ExplicitGrid((1.0e-20,)).sample(), [1.0e-20]))

    def test_extended_axis_preserves_every_legacy_node(self):
        legacy = legacy_dvdr_values()
        extended = extended_dvdr_values()
        positions = np.searchsorted(extended, legacy)
        np.testing.assert_array_equal(extended[positions], legacy)

    def test_extended_axis_covers_measured_snapshot_range(self):
        extended = extended_dvdr_values()
        self.assertEqual(extended[0], SIMULATION_DVDR_MIN_S)
        self.assertEqual(extended[-1], SIMULATION_DVDR_MAX_S)
        self.assertTrue(np.all(np.diff(extended) > 0.0))

    def test_added_axis_excludes_legacy_domain(self):
        added = added_dvdr_values()
        self.assertTrue(np.all(
            (added < LEGACY_DVDR_MIN_S) | (added > LEGACY_DVDR_MAX_S)
        ))
        self.assertEqual(
            added.size,
            extended_dvdr_values().size - legacy_dvdr_values().size,
        )


if __name__ == "__main__":
    unittest.main()
