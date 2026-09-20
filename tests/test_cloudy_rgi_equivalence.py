"""Compare the SciPy sampler with the pre-refactor corner arithmetic."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from quokka2s.cloudy_sixline_lookup import (
    CloudyFailureTouchError,
    CloudySixLineLookup,
    DEPTH_AXIS_ORDER,
    EXPECTED_AXIS_ORDER,
    TOUCH_EPS,
)


def legacy_corner_sample(lookup, temperature, density, column, *, depth=None):
    """Frozen value/support arithmetic from the sampler before the RGI change.

    Query validation retains its existing four-value interface. The numerical
    reference deliberately does not invoke either of the SciPy interpolators.
    """
    original_shape, below, above, brackets = lookup._prepare_query(
        temperature, density, column, depth
    )
    shape = (len(lookup.line_keys), below.size)
    linear_sum = np.zeros(shape)
    log_sum = np.zeros(shape)
    zero_support = np.zeros(shape, dtype=bool)
    failure_weight = np.zeros(shape)

    def visit(axis_number, indices, weight):
        if axis_number == len(brackets):
            index = (slice(None), *indices)
            local_weight = weight[None, :]
            local_log = lookup.log_emissivity_per_nH2[index]
            linear_sum[:] += lookup.emissivity_per_nH2[index] * local_weight
            log_sum[:] += np.where(
                np.isfinite(local_log), local_log, 0.0
            ) * local_weight
            zero_support[:] |= lookup.zero_mask[index] & (local_weight > TOUCH_EPS)
            failure_weight[:] += lookup.failure_mask[index] * local_weight
            return
        lower, upper, fraction = brackets[axis_number]
        visit(axis_number + 1, indices + [lower], weight * (1.0 - fraction))
        visit(axis_number + 1, indices + [upper], weight * fraction)

    visit(0, [], np.ones(below.size))
    if np.any(failure_weight > TOUCH_EPS):
        raise CloudyFailureTouchError("legacy reference touched a failed node")
    coefficient = np.where(zero_support, linear_sum, np.power(10.0, log_sum))
    return (
        coefficient.reshape((len(lookup.line_keys), *original_shape)),
        below.reshape(original_shape),
        above.reshape(original_shape),
        zero_support.reshape((len(lookup.line_keys), *original_shape)),
    )


class CloudyRGIEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "table.npz"

    def write_table(self, axes, log_values, *, zeros=(), failures=()):
        log_values = np.array(log_values, dtype=float, copy=True)
        failure = np.zeros_like(log_values, dtype=bool)
        zero = np.zeros_like(log_values, dtype=bool)
        for number, index in enumerate(zeros):
            zero[index] = True
            # Real tables have used both finite sentinels and nonfinite values.
            log_values[index] = (-99.0, -np.inf, np.nan)[number % 3]
        for index in failures:
            failure[index] = True
            log_values[index] = np.nan
        linear_values = np.zeros_like(log_values)
        valid = ~failure & ~zero
        linear_values[valid] = np.power(10.0, log_values[valid])
        payload = dict(
            axis_order=np.array(DEPTH_AXIS_ORDER if len(axes) == 4 else EXPECTED_AXIS_ORDER),
            line_keys=np.array([f"line{i}" for i in range(log_values.shape[0])]),
            log_NH_attenuation=axes[0],
            log_nH=axes[1],
            log_T=axes[2],
            log_emissivity_per_nH2=log_values,
            emissivity_per_nH2=linear_values,
            failure_mask=failure,
            zero_mask=zero,
        )
        if len(axes) == 4:
            payload["log_L_model_pc"] = axes[3]
        np.savez_compressed(self.path, **payload)
        return CloudySixLineLookup(self.path)

    def assert_matches_legacy(self, lookup, temperature, density, column, *, depth=None):
        expected, below, above, zero_support = legacy_corner_sample(
            lookup, temperature, density, column, depth=depth
        )
        actual = lookup.sample(temperature, density, column, model_depth_pc=depth)
        self.assertEqual(actual.emissivity_per_nH2.shape, expected.shape)
        # The same multilinear sum may differ only by floating-point rounding.
        np.testing.assert_allclose(
            actual.emissivity_per_nH2, expected, rtol=5e-14, atol=0.0
        )
        np.testing.assert_array_equal(actual.attenuation_column_below_table, below)
        np.testing.assert_array_equal(actual.attenuation_column_above_table, above)
        return actual, zero_support

    def test_random_irregular_three_and_four_dimensional_tables(self):
        rng = np.random.default_rng(23491)
        for dimension in (3, 4):
            with self.subTest(dimension=dimension):
                axes = [
                    np.array((18.0, 18.35, 19.8, 21.0)),
                    np.array((-4.0, -1.25, 0.1, 4.0)),
                    np.array((2.0, 3.4771, 4.23, 7.0)),
                ]
                if dimension == 4:
                    axes.append(np.array((-2.0, -0.4, 1.1, 2.0)))
                shape = (3, *(axis.size for axis in axes))
                logs = rng.uniform(-55.0, -18.0, size=shape)
                zeros = [
                    (line, *index)
                    for line in (1, 2)
                    for index in np.ndindex(shape[1:])
                    if rng.random() < 0.15
                ]
                lookup = self.write_table(axes, logs, zeros=zeros)
                self.assertIsInstance(lookup._linear_interpolator, RegularGridInterpolator)
                self.assertIsInstance(lookup._log_interpolator, RegularGridInterpolator)
                cache_ids = (id(lookup._linear_interpolator), id(lookup._log_interpolator))
                coordinates = [rng.uniform(axis[0], axis[-1], size=701) for axis in axes]
                coordinates[0][:2] = (axes[0][0] - 1, axes[0][-1] + 1)
                actual, zero_support = self.assert_matches_legacy(
                    lookup, 10.0**coordinates[2], 10.0**coordinates[1],
                    10.0**coordinates[0],
                    depth=None if dimension == 3 else 10.0**coordinates[3],
                )
                self.assertFalse(zero_support[0].any())
                self.assertTrue(zero_support[1:].any())
                self.assertTrue((~zero_support[1:]).any())
                self.assertTrue(np.isfinite(actual.emissivity_per_nH2).all())
                self.assertEqual(
                    cache_ids, (id(lookup._linear_interpolator), id(lookup._log_interpolator))
                )
                # Every outer and internal node; mixed zeros also exercise the
                # exact-zero weights assigned to the other corners.
                nodes = np.meshgrid(*axes, indexing="ij")
                self.assert_matches_legacy(
                    lookup, 10.0**nodes[2], 10.0**nodes[1], 10.0**nodes[0],
                    depth=None if dimension == 3 else 10.0**nodes[3],
                )

    def test_scalar_broadcast_and_empty_query_shapes(self):
        for dimension in (3, 4):
            with self.subTest(dimension=dimension):
                axes = [np.array((0.0, 1.0, 2.0)) for _ in range(dimension)]
                logs = np.arange(2 * 3**dimension).reshape((2,) + (3,) * dimension) / 7 - 30
                lookup = self.write_table(axes, logs)
                depth = None if dimension == 3 else 10.0
                scalar, _ = self.assert_matches_legacy(lookup, 10.0, 10.0, 10.0, depth=depth)
                self.assertEqual(scalar.emissivity_per_nH2.shape, (2,))
                broadcast, _ = self.assert_matches_legacy(
                    lookup, np.array((1.0, 10.0, 100.0))[None, :],
                    np.array((2.0, 30.0))[:, None], 10.0, depth=depth,
                )
                self.assertEqual(broadcast.emissivity_per_nH2.shape, (2, 2, 3))
                empty, _ = self.assert_matches_legacy(
                    lookup, np.empty((2, 0)), 10.0, 10.0, depth=depth,
                )
                self.assertEqual(empty.emissivity_per_nH2.shape, (2, 2, 0))
                self.assertEqual(empty.attenuation_column_below_table.shape, (2, 0))

    def test_zero_threshold_is_per_corner_not_sum_and_is_per_line(self):
        axes = [np.array((0.0, 1.0)) for _ in range(3)]
        logs = np.full((2, 2, 2, 2), -24.0)
        logs[:, :, 1, :] = -20.0
        lookup = self.write_table(axes, logs, zeros=((0, 1, 0, 0), (0, 1, 1, 0)))
        # Two zero corners each have 0.75 EPS, so their total exceeds EPS but
        # neither individually triggers the switch to raw-value interpolation.
        for total_weight, expected_raw in ((0.0, False), (1.5 * TOUCH_EPS, False),
                                           (3.0 * TOUCH_EPS, True)):
            with self.subTest(total_weight=total_weight):
                actual, zero_support = self.assert_matches_legacy(
                    lookup, 1.0, 10.0**0.5, 10.0**total_weight
                )
                self.assertEqual(bool(zero_support[0]), expected_raw)
                self.assertFalse(bool(zero_support[1]))
                self.assertAlmostEqual(actual.emissivity_per_nH2[1] / 1e-22, 1.0, places=12)
                if expected_raw:
                    self.assertGreater(actual.emissivity_per_nH2[0], 4e-21)
                else:
                    self.assertLess(actual.emissivity_per_nH2[0], 2e-22)

    def test_failure_threshold_uses_sum_and_zero_weight_failures_do_not_propagate(self):
        axes = [np.array((0.0, 1.0)) for _ in range(3)]
        logs = np.full((2, 2, 2, 2), -24.0)
        lookup = self.write_table(axes, logs, failures=((0, 1, 0, 0), (0, 1, 1, 0)))
        for total_weight in (0.0, 0.5 * TOUCH_EPS, 1.5 * TOUCH_EPS):
            with self.subTest(total_weight=total_weight):
                query = (1.0, 10.0**0.5, 10.0**total_weight)
                diagnostic = lookup.diagnose(*query)
                if total_weight > TOUCH_EPS:
                    np.testing.assert_array_equal(diagnostic.failure_touched, (True, False))
                    with self.assertRaises(CloudyFailureTouchError):
                        legacy_corner_sample(lookup, *query)
                    with self.assertRaises(CloudyFailureTouchError):
                        lookup.sample(*query)
                else:
                    self.assertFalse(diagnostic.failure_touched.any())
                    actual, _ = self.assert_matches_legacy(lookup, *query)
                    self.assertTrue(np.isfinite(actual.emissivity_per_nH2).all())


if __name__ == "__main__":
    unittest.main()
