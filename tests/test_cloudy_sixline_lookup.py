import tempfile
import unittest
from pathlib import Path

import numpy as np

from quokka2s.cloudy_sixline_lookup import (
    CloudyFailureTouchError,
    CloudySixLineLookup,
)


class CloudySixLineLookupTests(unittest.TestCase):
    def _write_table(
        self,
        directory: Path,
        *,
        zero_corner: tuple[int, int, int] | None = None,
        failure_corner: tuple[int, int, int] | None = None,
    ) -> Path:
        log_column = np.asarray((18.0, 21.0))
        log_density = np.asarray((-1.0, 1.0))
        log_temperature = np.asarray((3.0, 5.0))
        raw = np.empty((2, 2, 2, 2))
        for line in range(2):
            for i, column in enumerate(log_column):
                for j, density in enumerate(log_density):
                    for k, temperature in enumerate(log_temperature):
                        raw[line, i, j, k] = (
                            line + 0.1 * column + 0.2 * density + 0.3 * temperature
                        )
        failure = np.zeros_like(raw, dtype=bool)
        zero = np.zeros_like(raw, dtype=bool)
        if zero_corner is not None:
            zero[(slice(None), *zero_corner)] = True
            raw[(slice(None), *zero_corner)] = -99.0
        if failure_corner is not None:
            failure[(slice(None), *failure_corner)] = True
            raw[(slice(None), *failure_corner)] = np.nan
        coefficient = np.zeros_like(raw)
        positive = ~failure & ~zero
        coefficient[positive] = np.power(10.0, raw[positive])
        path = directory / "table.npz"
        np.savez_compressed(
            path,
            axis_order=np.asarray("line,log_NH_attenuation,log_nH,log_T"),
            line_keys=np.asarray(("line0", "line1")),
            log_NH_attenuation=log_column,
            log_nH=log_density,
            log_T=log_temperature,
            log_emissivity_per_nH2=raw,
            emissivity_per_nH2=coefficient,
            failure_mask=failure,
            zero_mask=zero,
        )
        return path

    def test_grid_node_and_log_trilinear_interpolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            lookup = CloudySixLineLookup(
                self._write_table(Path(temporary))
            )
            node = lookup.sample(1.0e3, 1.0e-1, 1.0e18)
            expected_node_log = np.asarray((2.5, 3.5))
            np.testing.assert_allclose(
                node.emissivity_per_nH2, np.power(10.0, expected_node_log)
            )

            middle = lookup.sample(1.0e4, 1.0, 10.0**19.5)
            expected_middle_log = np.asarray((3.15, 4.15))
            np.testing.assert_allclose(
                middle.emissivity_per_nH2,
                np.power(10.0, expected_middle_log),
                rtol=1.0e-14,
            )

    def test_column_is_clipped_but_density_and_temperature_are_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            lookup = CloudySixLineLookup(
                self._write_table(Path(temporary))
            )
            low = lookup.sample(1.0e4, 1.0, 1.0e17)
            lower_boundary = lookup.sample(1.0e4, 1.0, 1.0e18)
            high = lookup.sample(1.0e4, 1.0, 1.0e22)
            upper_boundary = lookup.sample(1.0e4, 1.0, 1.0e21)
            np.testing.assert_allclose(
                low.emissivity_per_nH2, lower_boundary.emissivity_per_nH2
            )
            np.testing.assert_allclose(
                high.emissivity_per_nH2, upper_boundary.emissivity_per_nH2
            )
            self.assertTrue(bool(low.attenuation_column_below_table))
            self.assertTrue(bool(high.attenuation_column_above_table))
            with self.assertRaises(ValueError):
                lookup.sample(1.0e4, 1.0e2, 1.0e19)
            with self.assertRaises(ValueError):
                lookup.sample(1.0e6, 1.0, 1.0e19)

    def test_zero_corner_switches_to_linear_interpolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            lookup = CloudySixLineLookup(
                self._write_table(Path(temporary), zero_corner=(0, 0, 0))
            )
            sample = lookup.sample(1.0e4, 1.0, 10.0**19.5)
            expected = np.mean(lookup.emissivity_per_nH2, axis=(1, 2, 3))
            np.testing.assert_allclose(sample.emissivity_per_nH2, expected)

    def test_failed_corner_with_positive_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            lookup = CloudySixLineLookup(
                self._write_table(Path(temporary), failure_corner=(0, 0, 0))
            )
            with self.assertRaises(CloudyFailureTouchError):
                lookup.sample(1.0e4, 1.0, 10.0**19.5)
            diagnostics = lookup.diagnose(1.0e4, 1.0, 10.0**19.5)
            np.testing.assert_array_equal(
                diagnostics.failure_touched, np.asarray((True, True))
            )
            self.assertGreater(diagnostics.maximum_failure_weight, 0.0)
            # The failed corner has exactly zero interpolation weight here.
            lookup.sample(1.0e5, 1.0e1, 1.0e21)


if __name__ == "__main__":
    unittest.main()
