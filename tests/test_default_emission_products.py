"""The output runner must use exactly the accepted unavailable-cell mask."""
import unittest
import numpy as np
from scripts.build_default_emission_products import check_exclusion_queries


class AcceptedExclusionTests(unittest.TestCase):
    def test_exact_missing_queries_are_excluded(self):
        np.testing.assert_array_equal(check_exclusion_queries(
            np.array([10, 11, 12]), np.array([1, 11, 20]), np.array([50., np.nan, 90.])),
            [False, True, False])

    def test_new_missing_cell_is_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, 'differs'):
            check_exclusion_queries(np.array([10, 11]), np.array([11]), np.array([np.nan, np.nan]))

    def test_available_cell_is_not_silently_excluded(self):
        with self.assertRaisesRegex(ValueError, 'differs'):
            check_exclusion_queries(np.array([10, 11]), np.array([11]), np.array([50., 60.]))


if __name__ == '__main__':
    unittest.main()
