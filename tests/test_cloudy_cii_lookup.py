from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import yt

from quokka2s.cloudy_cii_lookup import (
    CloudyCIILookup,
    fill_failures_along_log_temperature,
)
from quokka2s.pipeline.prep.physics_fields import _Cplus_luminosity


class CloudyCIILookupTests(unittest.TestCase):
    def test_failure_fill_is_linear_only_along_log_temperature(self):
        values = np.array([[[-30.0, -99.0, -99.0, -27.0]]])
        filled, mask = fill_failures_along_log_temperature(values)
        np.testing.assert_allclose(filled, [[[-30.0, -29.0, -28.0, -27.0]]])
        np.testing.assert_array_equal(mask, [[[False, True, True, False]]])

    def test_unbracketed_failure_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not bracketed'):
            fill_failures_along_log_temperature(
                np.array([[[-99.0, -30.0, -29.0]]])
            )

    def test_lookup_is_trilinear_in_log_coefficient_and_scales_by_nH2(self):
        log_nH = np.array([0.0, 2.0])
        log_NH = np.array([18.0, 20.0])
        log_T = np.array([3.0, 4.0])
        grid = np.empty((2, 2, 2))
        for i, nH in enumerate(log_nH):
            for j, NH in enumerate(log_NH):
                for k, T in enumerate(log_T):
                    grid[i, j, k] = -35.0 + 0.1 * nH + 0.2 * NH + 0.3 * T

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cloudy.npz'
            np.savez(
                path,
                log_nH=log_nH,
                log_NH=log_NH,
                log_T=log_T,
                log_emissivity_per_nH2=grid,
                failure_mask=np.zeros_like(grid, dtype=bool),
            )
            lookup = CloudyCIILookup(path)
            actual = lookup.emissivity(10.0 ** 3.5, 10.0, 1.0e19)

        expected_log_coefficient = -35.0 + 0.1 * 1.0 + 0.2 * 19.0 + 0.3 * 3.5
        expected = 10.0 ** expected_log_coefficient * 10.0 ** 2
        self.assertAlmostEqual(float(actual), expected, delta=expected * 1.0e-12)

    def test_temperature_above_truncated_table_returns_zero(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cloudy.npz'
            grid = np.full((2, 2, 2), -25.0)
            np.savez(
                path,
                log_nH=np.array([0.0, 1.0]),
                log_NH=np.array([18.0, 19.0]),
                log_T=np.log10(np.array([3000.0, 4.0e4])),
                log_emissivity_per_nH2=grid,
                failure_mask=np.zeros_like(grid, dtype=bool),
            )
            lookup = CloudyCIILookup(path)
            actual = lookup.emissivity(
                np.array([3000.0, 4.0e4, 5.0e4]), 1.0, 1.0e18,
            )

        self.assertGreater(actual[0], 0.0)
        self.assertGreater(actual[1], 0.0)
        self.assertEqual(actual[2], 0.0)

    def test_failure_interpolation_weight_tracks_trilinear_corner_weight(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cloudy.npz'
            grid = np.full((2, 2, 2), -25.0)
            failure_mask = np.zeros_like(grid, dtype=bool)
            failure_mask[1, 1, 1] = True
            np.savez(
                path,
                log_nH=np.array([0.0, 2.0]),
                log_NH=np.array([18.0, 20.0]),
                log_T=np.array([3.0, 4.0]),
                log_emissivity_per_nH2=grid,
                failure_mask=failure_mask,
            )
            lookup = CloudyCIILookup(path)
            actual = lookup.failure_interpolation_weight(
                np.array([10.0 ** 3.5, 1.0e4, 1.0e3, 1.0e5]),
                np.array([10.0, 1.0e2, 1.0, 1.0e2]),
                np.array([1.0e19, 1.0e20, 1.0e18, 1.0e20]),
            )

        np.testing.assert_allclose(actual, [0.125, 1.0, 0.0, 0.0])

    @staticmethod
    def _write_schema2(
        path: Path,
        coefficient: np.ndarray,
        out_of_bounds_policy: str = 'raise',
        failure_mask: np.ndarray | None = None,
    ) -> None:
        if failure_mask is None:
            failure_mask = np.zeros_like(coefficient, dtype=bool)
        log_coefficient = np.full_like(coefficient, -99.0, dtype=float)
        positive = coefficient > 0.0
        log_coefficient[positive] = np.log10(coefficient[positive])
        log_coefficient[failure_mask] = np.nan
        np.savez(
            path,
            schema_version=np.asarray(2),
            log_nH=np.array([0.0, 2.0]),
            log_NH=np.array([18.0, 20.0]),
            log_T=np.array([3.0, 4.0]),
            emissivity_per_nH2=coefficient,
            log_emissivity_per_nH2=log_coefficient,
            zero_mask=(coefficient == 0.0) & ~failure_mask,
            failure_mask=failure_mask,
            out_of_bounds_policy=np.asarray(out_of_bounds_policy),
        )

    def test_schema2_uses_log_interpolation_for_positive_stencil(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cloudy.npz'
            coefficient = np.ones((2, 2, 2))
            coefficient[1, 1, 1] = 1.0e8
            self._write_schema2(path, coefficient)
            lookup = CloudyCIILookup(path)
            actual = lookup.emissivity(10.0 ** 3.5, 10.0, 1.0e19)

        # Cube center gives every corner weight 1/8, hence log coefficient=1.
        self.assertAlmostEqual(float(actual), 10.0 * 10.0 ** 2)

    def test_schema2_mixed_zero_stencil_uses_linear_interpolation(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cloudy.npz'
            coefficient = np.ones((2, 2, 2))
            coefficient[1, 1, 1] = 0.0
            self._write_schema2(path, coefficient)
            lookup = CloudyCIILookup(path)
            actual = lookup.emissivity(10.0 ** 3.5, 10.0, 1.0e19)

        self.assertAlmostEqual(float(actual), (7.0 / 8.0) * 10.0 ** 2)

    def test_schema2_retains_failure_as_unavailable_not_as_zero(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cloudy.npz'
            coefficient = np.ones((2, 2, 2))
            failure_mask = np.zeros_like(coefficient, dtype=bool)
            failure_mask[1, 1, 1] = True
            coefficient[failure_mask] = 0.0
            self._write_schema2(
                path, coefficient, failure_mask=failure_mask,
            )
            lookup = CloudyCIILookup(path)
            unaffected = lookup.emissivity(1.0e3, 1.0, 1.0e18)
            with self.assertRaisesRegex(ValueError, 'unavailable failed node'):
                lookup.emissivity(10.0 ** 3.5, 10.0, 1.0e19)

        self.assertEqual(float(unaffected), 1.0)

    def test_schema2_rejects_out_of_bounds_request(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cloudy.npz'
            self._write_schema2(path, np.ones((2, 2, 2)))
            lookup = CloudyCIILookup(path)
            with self.assertRaisesRegex(ValueError, 'log_T outside table'):
                lookup.emissivity(1.0e5, 10.0, 1.0e19)

    def test_schema2_verified_temperature_above_max_returns_zero(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cloudy.npz'
            self._write_schema2(
                path,
                np.ones((2, 2, 2)),
                'temperature_above_max_zero; other_axes_raise',
            )
            lookup = CloudyCIILookup(path)
            actual = lookup.emissivity(
                np.array([1.0e3, 1.0e4, 1.0e5]),
                10.0,
                1.0e19,
            )

        np.testing.assert_allclose(actual, [100.0, 100.0, 0.0])

    def test_schema2_verified_zero_policy_still_rejects_other_axes(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cloudy.npz'
            self._write_schema2(
                path,
                np.ones((2, 2, 2)),
                'temperature_above_max_zero; other_axes_raise',
            )
            lookup = CloudyCIILookup(path)
            with self.assertRaisesRegex(ValueError, 'log_nH outside table'):
                lookup.emissivity(1.0e5, 1.0e3, 1.0e19)

    @patch('quokka2s.pipeline.prep.physics_fields._table_emissivity')
    @patch('quokka2s.pipeline.prep.physics_fields.ensure_cloudy_cii_lookup')
    @patch('quokka2s.pipeline.prep.physics_fields.ensure_table_lookup')
    def test_hybrid_field_uses_tquokka_at_3000K_boundary(
        self,
        ensure_despotic,
        ensure_cloudy,
        table_emissivity,
    ):
        table_emissivity.return_value = np.array([1.0, 2.0, 3.0, 4.0])
        cloudy = Mock()
        cloudy.emissivity.return_value = np.array([20.0, 30.0, 0.0])
        ensure_cloudy.return_value = cloudy
        data = {
            ('gas', 'temperature_quokka'): yt.YTArray(
                [2999.0, 3000.0, 3001.0, 5.0e4], 'K',
            ),
            ('gas', 'number_density_H'): yt.YTArray(np.ones(4), 'cm**-3'),
            ('gas', 'column_density_H'): yt.YTArray(np.full(4, 1.0e20), 'cm**-2'),
            ('gas', 'dVdr_lvg'): yt.YTArray(np.full(4, 1.0e-15), '1/s'),
        }

        actual = _Cplus_luminosity(None, data).to_value('erg/s/cm**3')

        np.testing.assert_array_equal(actual, [1.0, 20.0, 30.0, 0.0])
        ensure_despotic.assert_called_once()


if __name__ == '__main__':
    unittest.main()
