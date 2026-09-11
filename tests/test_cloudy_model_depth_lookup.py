"""Depth interpolation against known functions and unavailable-node support."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from quokka2s.cloudy_sixline_lookup import (
    CloudyFailureTouchError,
    CloudySixLineLookup,
    DEPTH_AXIS_ORDER,
    EXPECTED_AXIS_ORDER,
    TOUCH_EPS,
)


def analytic_log_coefficient(line, column, density, temperature, depth):
    # Cross terms make this multilinear in all four log coordinates, and
    # expose axis transposition as well as omitted or misweighted corners.
    return (
        line + 0.1 * column + 0.2 * density + 0.3 * temperature + 0.4 * depth
        + 0.07 * density * depth + 0.03 * temperature * depth
        - 0.002 * column * density * temperature * depth
    )


class CloudyModelDepthLookupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "depth_table.npz"

    def write_table(self, *, failures=(), zeros=(), depth_axis=(-0.25, 0.5, 2.0), legacy=False):
        axes = [np.array((18.0, 21.0)), np.array((-1.0, 1.0)), np.array((3.0, 5.0))]
        if not legacy:
            axes.append(np.array(depth_axis))
        coordinates = list(np.meshgrid(np.arange(2), *axes, indexing="ij"))
        if legacy:
            coordinates.append(0.0)
        raw = analytic_log_coefficient(*coordinates)
        failure = np.zeros_like(raw, dtype=bool)
        zero = np.zeros_like(raw, dtype=bool)
        for index in failures:
            failure[index] = True
            raw[index] = np.nan
        for index in zeros:
            zero[index] = True
            raw[index] = -99.0
        coefficient = np.zeros_like(raw)
        good = ~failure & ~zero
        coefficient[good] = 10.0**raw[good]
        payload = dict(
            axis_order=np.array(EXPECTED_AXIS_ORDER if legacy else DEPTH_AXIS_ORDER),
            line_keys=np.array(("line0", "line1")),
            log_NH_attenuation=axes[0], log_nH=axes[1], log_T=axes[2],
            log_emissivity_per_nH2=raw, emissivity_per_nH2=coefficient,
            failure_mask=failure, zero_mask=zero,
        )
        if not legacy:
            payload["log_L_model_pc"] = axes[3]
        np.savez_compressed(self.path, **payload)
        return CloudySixLineLookup(self.path)

    def test_every_exact_grid_node(self):
        lookup = self.write_table()
        column, density, temperature, depth = np.meshgrid(
            lookup.log_NH_attenuation, lookup.log_nH, lookup.log_T,
            lookup.log_L_model_pc, indexing="ij",
        )
        result = lookup.sample(10.0**temperature, 10.0**density, 10.0**column,
                               model_depth_pc=10.0**depth)
        np.testing.assert_allclose(result.emissivity_per_nH2,
                                   lookup.emissivity_per_nH2, rtol=2e-14)
        diagnostics = lookup.diagnose(10.0**temperature, 10.0**density, 10.0**column,
                                     model_depth_pc=10.0**depth)
        self.assertFalse(np.any(diagnostics.failure_touched))

    def test_analytic_four_dimensional_interpolation_and_broadcasting(self):
        lookup = self.write_table()
        temperature = np.array(((3.2,), (4.7,)))
        depth = np.array((-0.1, 0.3, 1.2, 1.9))
        result = lookup.sample(10.0**temperature, 10.0**0.2, 10.0**19.2,
                               model_depth_pc=10.0**depth)
        expected = np.array([
            10.0**analytic_log_coefficient(line, 19.2, 0.2, temperature, depth)
            for line in range(2)
        ])
        self.assertEqual(result.emissivity_per_nH2.shape, (2, 2, 4))
        self.assertEqual(result.attenuation_column_below_table.shape, (2, 4))
        np.testing.assert_allclose(result.emissivity_per_nH2, expected, rtol=2e-14)
        diagnostics = lookup.diagnose(10.0**temperature, 10.0**0.2, 10.0**19.2,
                                     model_depth_pc=10.0**depth)
        self.assertEqual(diagnostics.failure_touched.shape, (2, 2, 4))

    def test_both_query_methods_require_valid_in_domain_depth(self):
        lookup = self.write_table()
        for method in (lookup.sample, lookup.diagnose):
            with self.subTest(method=method.__name__, depth="missing"):
                with self.assertRaisesRegex(ValueError, "model_depth_pc is required"):
                    method(1e4, 1.0, 1e19)
            for depth in (0.0, -1.0, np.nan, np.inf, -np.inf, 0.5, 101.0,
                          100.0 * (1.0 + 1e-12), 10.0**-0.25 * (1.0 - 1e-12)):
                with self.subTest(method=method.__name__, depth=depth):
                    with self.assertRaises(ValueError):
                        method(1e4, 1.0, 1e19, model_depth_pc=depth)
            for temperature, density in ((1e6, 1.0), (1e4, 100.0)):
                with self.assertRaises(ValueError):
                    method(temperature, density, 1e19, model_depth_pc=1.0)

    def test_depth_boundaries_allow_only_conversion_roundoff(self):
        lookup = self.write_table()
        self.assertEqual(lookup.model_depth_bounds_pc, (10.0**-0.25, 100.0))
        for bound, direction in ((10.0**-0.25, -np.inf), (100.0, np.inf)):
            exact = lookup.sample(1e4, 1.0, 1e19, model_depth_pc=bound)
            roundoff = lookup.sample(1e4, 1.0, 1e19,
                                     model_depth_pc=np.nextafter(bound, direction))
            np.testing.assert_allclose(roundoff.emissivity_per_nH2,
                                       exact.emissivity_per_nH2, rtol=2e-14)

    def test_attenuation_clips_lookup_coordinate_without_mutating_cell_columns(self):
        lookup = self.write_table()
        column = np.array((1e17, 1e20, 1e22))
        original = column.copy()
        column.setflags(write=False)
        result = lookup.sample(1e4, 1.0, column, model_depth_pc=10.0)
        boundary_result = lookup.sample(1e4, 1.0, np.clip(column, 1e18, 1e21),
                                        model_depth_pc=10.0)
        np.testing.assert_allclose(result.emissivity_per_nH2,
                                   boundary_result.emissivity_per_nH2)
        diagnostics = lookup.diagnose(1e4, 1.0, column, model_depth_pc=10.0)
        for output in (result, diagnostics):
            np.testing.assert_array_equal(output.attenuation_column_below_table,
                                          (True, False, False))
            np.testing.assert_array_equal(output.attenuation_column_above_table,
                                          (False, False, True))
        np.testing.assert_array_equal(column, original)

    def test_true_zero_is_not_failure_and_switches_only_affected_line_to_linear(self):
        lookup = self.write_table(zeros=((0, 0, 0, 0, 0),), depth_axis=(0.0, 2.0))
        exact_zero = lookup.sample(1e3, 0.1, 1e18, model_depth_pc=1.0)
        self.assertEqual(exact_zero.emissivity_per_nH2[0], 0.0)
        middle = lookup.sample(1e4, 1.0, 10.0**19.5, model_depth_pc=10.0)
        self.assertAlmostEqual(middle.emissivity_per_nH2[0],
                               lookup.emissivity_per_nH2[0].mean())
        expected_positive = 10.0**analytic_log_coefficient(1, 19.5, 0.0, 4.0, 1.0)
        self.assertAlmostEqual(middle.emissivity_per_nH2[1], expected_positive)
        diagnostics = lookup.diagnose(1e4, 1.0, 10.0**19.5, model_depth_pc=10.0)
        self.assertFalse(np.any(diagnostics.failure_touched))
        # A zero at the unused lower-depth plane must not change the upper plane
        # to linear interpolation among its positive corners.
        upper = lookup.sample(1e4, 1.0, 10.0**19.5, model_depth_pc=100.0)
        expected_upper = np.array([
            10.0**analytic_log_coefficient(line, 19.5, 0.0, 4.0, 2.0)
            for line in range(2)
        ])
        np.testing.assert_allclose(upper.emissivity_per_nH2, expected_upper, rtol=2e-14)

    def test_failed_corner_support_includes_depth_but_ignores_zero_weight(self):
        lookup = self.write_table(failures=((0, 0, 0, 0, 0),), depth_axis=(0.0, 2.0))
        with self.assertRaises(CloudyFailureTouchError):
            lookup.sample(1e4, 1.0, 10.0**19.5, model_depth_pc=10.0)
        diagnostics = lookup.diagnose(1e4, 1.0, 10.0**19.5, model_depth_pc=10.0)
        np.testing.assert_array_equal(diagnostics.failure_touched, (True, False))
        self.assertEqual(diagnostics.maximum_failure_weight, 1.0 / 16.0)
        # The failed node's other three coordinates remain active here; only
        # its depth weight vanishes. NaN log placeholders must not propagate.
        result = lookup.sample(1e4, 1.0, 10.0**19.5, model_depth_pc=100.0)
        self.assertTrue(np.isfinite(result.emissivity_per_nH2).all())
        diagnostics = lookup.diagnose(1e4, 1.0, 10.0**19.5, model_depth_pc=100.0)
        self.assertFalse(np.any(diagnostics.failure_touched))

    def test_failure_touch_threshold_applies_to_total_weight(self):
        lookup = self.write_table(failures=((0, 0, 0, 0, 0), (0, 1, 0, 0, 0)),
                                  depth_axis=(0.0, 1.0))
        for total_weight, touched in ((0.5 * TOUCH_EPS, False), (1.5 * TOUCH_EPS, True)):
            # Each failed corner's individual weight is below TOUCH_EPS even
            # when their sum must reject the query.
            depth = 10.0**(1.0 - total_weight)
            diagnostics = lookup.diagnose(1e3, 0.1, 10.0**19.5, model_depth_pc=depth)
            self.assertEqual(bool(diagnostics.failure_touched[0]), touched)
            if touched:
                with self.assertRaises(CloudyFailureTouchError):
                    lookup.sample(1e3, 0.1, 10.0**19.5, model_depth_pc=depth)
            else:
                lookup.sample(1e3, 0.1, 10.0**19.5, model_depth_pc=depth)

    def test_legacy_table_remains_readable_but_cannot_claim_explicit_depth(self):
        lookup = self.write_table(legacy=True)
        self.assertIsNone(lookup.model_depth_bounds_pc)
        self.assertIsNone(lookup.log_L_model_pc)
        lookup.sample(1e4, 1.0, 1e19)
        lookup.diagnose(1e4, 1.0, 1e19)
        for method in (lookup.sample, lookup.diagnose):
            with self.assertRaisesRegex(ValueError, "legacy Jeans table"):
                method(1e4, 1.0, 1e19, model_depth_pc=100.0)

    def test_depth_schema_requires_finite_axis_and_correct_tensor_shape(self):
        self.write_table()
        with np.load(self.path) as source:
            payload = {name: source[name] for name in source.files}
        for bad_axis in (None, np.array((0.0, np.nan, 2.0)), np.array((0.0, 1.0)),
                         np.array((-400.0, 0.0, 2.0)), np.array((0.0, 1.0, 400.0))):
            modified = payload.copy()
            if bad_axis is None:
                del modified["log_L_model_pc"]
            else:
                modified["log_L_model_pc"] = bad_axis
            np.savez_compressed(self.path, **modified)
            with self.assertRaises(ValueError):
                CloudySixLineLookup(self.path)


if __name__ == "__main__":
    unittest.main()
