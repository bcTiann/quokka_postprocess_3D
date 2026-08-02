"""Lookup for the HM2012 shielded Cloudy [C II] 158 micron table.

The stored quantity is ``log10(epsilon_CII / n_H**2)`` on a regular grid in
``(log10 n_H, log10 N_H, log10 T)``.  Runtime sampling is trilinear in those
logarithmic coordinates.  Schema-2 tables distinguish completed calculations,
Cloudy's true zero nodes, and explicitly unavailable failed nodes.  A query
that gives a failed node positive interpolation weight is rejected.  The older
schema-1 coarse table remains readable so existing diagnostics can still be
reproduced.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def fill_failures_along_log_temperature(
    log_emissivity_per_nH2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill non-finite/``-99`` cells linearly along the final (log-T) axis.

    Returns a new filled array and a Boolean mask of the original failures.
    Every failed cell must have a valid temperature neighbour on both sides;
    extrapolation is deliberately rejected.
    """
    values = np.asarray(log_emissivity_per_nH2, dtype=float)
    if values.ndim != 3:
        raise ValueError('Cloudy CII emissivity grid must be three-dimensional')

    failure_mask = (~np.isfinite(values)) | (values <= -90.0)
    filled = values.copy()
    x = np.arange(values.shape[-1], dtype=float)

    for index in np.ndindex(values.shape[:-1]):
        row = values[index]
        missing = failure_mask[index]
        if not missing.any():
            continue
        valid = ~missing
        if valid.sum() < 2:
            raise ValueError(f'not enough valid temperatures to fill row {index}')
        first = int(np.flatnonzero(valid)[0])
        last = int(np.flatnonzero(valid)[-1])
        if missing[:first + 1].any() or missing[last:].any():
            raise ValueError(
                f'Cloudy failures at row {index} are not bracketed in temperature'
            )
        filled_row = filled[index]
        filled_row[missing] = np.interp(x[missing], x[valid], row[valid])

    return filled, failure_mask


class CloudyCIILookup:
    """Memory-light trilinear sampler for the Cloudy [C II] table."""

    _EVAL_CHUNK = 4_000_000

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as table:
            self.schema_version = int(
                np.asarray(table['schema_version']).item()
                if 'schema_version' in table.files else 1
            )
            self.log_nH = np.asarray(table['log_nH'], dtype=float)
            self.log_NH = np.asarray(table['log_NH'], dtype=float)
            self.log_T = np.asarray(table['log_T'], dtype=float)
            self.log_emissivity_per_nH2 = np.asarray(
                table['log_emissivity_per_nH2'], dtype=float,
            )
            self.failure_mask = np.asarray(table['failure_mask'], dtype=bool)
            if self.schema_version >= 2:
                self.emissivity_per_nH2 = np.asarray(
                    table['emissivity_per_nH2'], dtype=float,
                )
                self.zero_mask = np.asarray(table['zero_mask'], dtype=bool)
                self.out_of_bounds_policy = str(
                    np.asarray(table['out_of_bounds_policy']).item()
                )
            else:
                self.emissivity_per_nH2 = np.power(
                    10.0, self.log_emissivity_per_nH2,
                )
                self.zero_mask = np.zeros_like(self.failure_mask)
                self.out_of_bounds_policy = 'legacy_zero_temperature_clamp_axes'

        expected = (self.log_nH.size, self.log_NH.size, self.log_T.size)
        if self.log_emissivity_per_nH2.shape != expected:
            raise ValueError(
                f'Cloudy CII grid has shape {self.log_emissivity_per_nH2.shape}, '
                f'expected {expected}'
            )
        if self.failure_mask.shape != expected:
            raise ValueError('Cloudy CII failure mask does not match the grid')
        if self.emissivity_per_nH2.shape != expected:
            raise ValueError('Cloudy CII linear coefficient grid has wrong shape')
        if self.zero_mask.shape != expected:
            raise ValueError('Cloudy CII zero mask does not match the grid')
        for name, axis in (
            ('log_nH', self.log_nH), ('log_NH', self.log_NH),
            ('log_T', self.log_T),
        ):
            if axis.ndim != 1 or axis.size < 2 or np.any(np.diff(axis) <= 0):
                raise ValueError(f'{name} must be a strictly increasing 1D axis')
        if self.schema_version == 1 and not np.isfinite(
            self.log_emissivity_per_nH2
        ).all():
            raise ValueError('Cloudy CII production grid still contains failures')
        if self.schema_version >= 2:
            allowed_policies = {
                'raise',
                'temperature_above_max_zero; other_axes_raise',
            }
            if self.out_of_bounds_policy not in allowed_policies:
                raise ValueError(
                    'unsupported schema-2 Cloudy CII out-of-bounds policy: '
                    f'{self.out_of_bounds_policy!r}'
                )
            if not np.isfinite(self.emissivity_per_nH2).all():
                raise ValueError('Cloudy CII linear coefficient grid is not finite')
            if np.any(self.emissivity_per_nH2 < 0.0):
                raise ValueError('Cloudy CII coefficients must be non-negative')
            unavailable_is_zero = (
                self.emissivity_per_nH2 == 0.0
            ) & ~self.failure_mask
            if not np.array_equal(self.zero_mask, unavailable_is_zero):
                raise ValueError('Cloudy CII zero mask disagrees with coefficients')
            if np.any(self.emissivity_per_nH2[self.failure_mask] != 0.0):
                raise ValueError('failed Cloudy nodes must have zero placeholder coefficients')

        self._interpolator = RegularGridInterpolator(
            (self.log_nH, self.log_NH, self.log_T),
            self.log_emissivity_per_nH2,
            method='linear',
            bounds_error=False,
            fill_value=np.nan,
        )

    @property
    def temperature_bounds_K(self) -> tuple[float, float]:
        return 10.0 ** self.log_T[0], 10.0 ** self.log_T[-1]

    def emissivity(self, temperature_K, n_H_cm3, column_density_H_cm2) -> np.ndarray:
        """Return volumetric emissivity in ``erg s^-1 cm^-3``.

        Schema-2 tables always reject density/column requests outside the
        grid.  Their metadata chooses whether temperature above the verified
        exact-zero boundary is rejected or returned as zero; temperature below
        the table is always rejected.  In-range values interpolate in log
        coefficient when every contributing corner is positive, and in the
        non-negative linear coefficient when a contributing corner is a
        Cloudy zero.  Schema-1 tables retain their historical behavior:
        density/column clamping and zero outside the truncated T interval.
        """
        if self.schema_version >= 2:
            return self._strict_emissivity(
                temperature_K, n_H_cm3, column_density_H_cm2,
            )

        temperature, n_H, column = np.broadcast_arrays(
            np.asarray(temperature_K, dtype=float),
            np.asarray(n_H_cm3, dtype=float),
            np.asarray(column_density_H_cm2, dtype=float),
        )
        output = np.zeros(temperature.shape, dtype=float)
        t_min, t_max = self.temperature_bounds_K
        # Include values that are exactly the requested physical endpoints;
        # 10**log10(T) can differ from T by a few floating-point ulps.
        t_min *= 1.0 - 1.0e-12
        t_max *= 1.0 + 1.0e-12
        valid = (
            np.isfinite(temperature) & np.isfinite(n_H) & np.isfinite(column)
            & (temperature >= t_min) & (temperature <= t_max)
            & (n_H > 0.0) & (column > 0.0)
        )
        if not valid.any():
            return output

        flat_valid = np.flatnonzero(valid.ravel())
        flat_T = temperature.ravel()
        flat_nH = n_H.ravel()
        flat_column = column.ravel()
        flat_output = output.ravel()

        for start in range(0, flat_valid.size, self._EVAL_CHUNK):
            indices = flat_valid[start:start + self._EVAL_CHUNK]
            log_nH = np.clip(
                np.log10(flat_nH[indices]), self.log_nH[0], self.log_nH[-1],
            )
            log_NH = np.clip(
                np.log10(flat_column[indices]), self.log_NH[0], self.log_NH[-1],
            )
            points = np.column_stack((
                log_nH,
                log_NH,
                np.clip(
                    np.log10(flat_T[indices]), self.log_T[0], self.log_T[-1],
                ),
            ))
            log_coefficient = self._interpolator(points)
            flat_output[indices] = np.power(10.0, log_coefficient) * np.square(
                flat_nH[indices]
            )

        return output

    def _strict_emissivity(
        self,
        temperature_K,
        n_H_cm3,
        column_density_H_cm2,
    ) -> np.ndarray:
        """Evaluate a strict schema-2 table without clamping/extrapolation."""
        temperature, n_H, column = np.broadcast_arrays(
            np.asarray(temperature_K, dtype=float),
            np.asarray(n_H_cm3, dtype=float),
            np.asarray(column_density_H_cm2, dtype=float),
        )
        if not (
            np.isfinite(temperature).all()
            and np.isfinite(n_H).all()
            and np.isfinite(column).all()
        ):
            raise ValueError('Cloudy CII lookup inputs must all be finite')
        if np.any((temperature <= 0.0) | (n_H <= 0.0) | (column <= 0.0)):
            raise ValueError('Cloudy CII lookup inputs must all be positive')

        coordinates = (
            np.log10(n_H), np.log10(column), np.log10(temperature),
        )
        axes = (self.log_nH, self.log_NH, self.log_T)
        for name, axis, coordinate in zip(
            ('log_nH', 'log_NH'), axes[:2], coordinates[:2],
        ):
            tolerance = 1.0e-12 * max(1.0, abs(axis[0]), abs(axis[-1]))
            if np.any(coordinate < axis[0] - tolerance) or np.any(
                coordinate > axis[-1] + tolerance
            ):
                raise ValueError(
                    f'Cloudy CII {name} outside table [{axis[0]}, {axis[-1]}]: '
                    f'input range [{float(np.min(coordinate))}, '
                    f'{float(np.max(coordinate))}]'
                )

        temperature_axis = self.log_T
        log_temperature = coordinates[2]
        temperature_tolerance = 1.0e-12 * max(
            1.0, abs(temperature_axis[0]), abs(temperature_axis[-1]),
        )
        if np.any(log_temperature < temperature_axis[0] - temperature_tolerance):
            raise ValueError(
                f'Cloudy CII log_T outside table '
                f'[{temperature_axis[0]}, {temperature_axis[-1]}]: '
                f'input range [{float(np.min(log_temperature))}, '
                f'{float(np.max(log_temperature))}]'
            )
        above_temperature_boundary = (
            log_temperature > temperature_axis[-1] + temperature_tolerance
        )
        zero_above = (
            self.out_of_bounds_policy
            == 'temperature_above_max_zero; other_axes_raise'
        )
        if above_temperature_boundary.any() and not zero_above:
            raise ValueError(
                f'Cloudy CII log_T outside table '
                f'[{temperature_axis[0]}, {temperature_axis[-1]}]: '
                f'input range [{float(np.min(log_temperature))}, '
                f'{float(np.max(log_temperature))}]'
            )

        flat_coordinates = tuple(value.ravel() for value in coordinates)
        flat_nH = n_H.ravel()
        flat_output = np.zeros(temperature.size, dtype=float)
        evaluate_indices = np.flatnonzero(~above_temperature_boundary.ravel())
        for start in range(0, evaluate_indices.size, self._EVAL_CHUNK):
            indices = evaluate_indices[start:start + self._EVAL_CHUNK]
            brackets = tuple(
                self._bracketing_indices_and_fraction(
                    axis, coordinate[indices],
                )
                for axis, coordinate in zip(axes, flat_coordinates)
            )
            linear_sum = np.zeros(indices.size, dtype=float)
            log_sum = np.zeros(indices.size, dtype=float)
            has_contributing_zero = np.zeros(indices.size, dtype=bool)
            has_contributing_failure = np.zeros(indices.size, dtype=bool)
            for n_corner in (0, 1):
                n_index = brackets[0][n_corner]
                n_weight = brackets[0][2] if n_corner else 1.0 - brackets[0][2]
                for column_corner in (0, 1):
                    column_index = brackets[1][column_corner]
                    column_weight = (
                        brackets[1][2]
                        if column_corner else 1.0 - brackets[1][2]
                    )
                    for temperature_corner in (0, 1):
                        temperature_index = brackets[2][temperature_corner]
                        temperature_weight = (
                            brackets[2][2]
                            if temperature_corner else 1.0 - brackets[2][2]
                        )
                        weight = n_weight * column_weight * temperature_weight
                        coefficient = self.emissivity_per_nH2[
                            n_index, column_index, temperature_index,
                        ]
                        linear_sum += weight * coefficient
                        contributing = weight > 1.0e-15
                        failed = self.failure_mask[
                            n_index, column_index, temperature_index,
                        ]
                        has_contributing_failure |= (weight > 1.0e-12) & failed
                        positive = coefficient > 0.0
                        has_contributing_zero |= contributing & ~positive
                        log_sum[positive] += (
                            weight[positive] * np.log10(coefficient[positive])
                        )
            if np.any(has_contributing_failure):
                raise ValueError(
                    'Cloudy CII lookup query touches an unavailable failed node'
                )
            interpolated = np.where(
                has_contributing_zero,
                linear_sum,
                np.power(10.0, log_sum),
            )
            flat_output[indices] = (
                interpolated * np.square(flat_nH[indices])
            )
        return flat_output.reshape(temperature.shape)

    @staticmethod
    def _bracketing_indices_and_fraction(
        axis: np.ndarray,
        coordinates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return lower/upper grid indices and the linear upper weight."""
        clipped = np.clip(coordinates, axis[0], axis[-1])
        upper = np.searchsorted(axis, clipped, side='right')
        upper = np.clip(upper, 1, axis.size - 1)
        lower = upper - 1
        fraction = (
            (clipped - axis[lower]) / (axis[upper] - axis[lower])
        )
        return lower, upper, fraction

    def failure_interpolation_weight(
        self,
        temperature_K,
        n_H_cm3,
        column_density_H_cm2,
    ) -> np.ndarray:
        """Return how much of each trilinear result comes from failed nodes.

        This diagnostic evaluates the sum of the trilinear corner weights
        whose nodes are marked in ``failure_mask``.  A result of zero means no
        failed node participates; one means the interpolation stencil is
        entirely an unavailable node.  It is primarily used to audit a
        simulation before permitting a table with retained failures.
        """
        temperature, n_H, column = np.broadcast_arrays(
            np.asarray(temperature_K, dtype=float),
            np.asarray(n_H_cm3, dtype=float),
            np.asarray(column_density_H_cm2, dtype=float),
        )
        output = np.zeros(temperature.shape, dtype=float)
        t_min, t_max = self.temperature_bounds_K
        t_min *= 1.0 - 1.0e-12
        t_max *= 1.0 + 1.0e-12
        valid = (
            np.isfinite(temperature) & np.isfinite(n_H) & np.isfinite(column)
            & (temperature >= t_min) & (temperature <= t_max)
            & (n_H > 0.0) & (column > 0.0)
        )
        if not valid.any():
            return output

        flat_valid = np.flatnonzero(valid.ravel())
        flat_T = temperature.ravel()
        flat_nH = n_H.ravel()
        flat_column = column.ravel()
        flat_output = output.ravel()

        for start in range(0, flat_valid.size, self._EVAL_CHUNK):
            indices = flat_valid[start:start + self._EVAL_CHUNK]
            coordinates = (
                np.log10(flat_nH[indices]),
                np.log10(flat_column[indices]),
                np.log10(flat_T[indices]),
            )
            brackets = tuple(
                self._bracketing_indices_and_fraction(axis, coordinate)
                for axis, coordinate in zip(
                    (self.log_nH, self.log_NH, self.log_T), coordinates,
                )
            )
            local_weight = np.zeros(indices.size, dtype=float)
            for n_corner in (0, 1):
                n_index = brackets[0][n_corner]
                n_weight = (
                    brackets[0][2] if n_corner else 1.0 - brackets[0][2]
                )
                for column_corner in (0, 1):
                    column_index = brackets[1][column_corner]
                    column_weight = (
                        brackets[1][2]
                        if column_corner else 1.0 - brackets[1][2]
                    )
                    for temperature_corner in (0, 1):
                        temperature_index = brackets[2][temperature_corner]
                        temperature_weight = (
                            brackets[2][2]
                            if temperature_corner else 1.0 - brackets[2][2]
                        )
                        corner_failed = self.failure_mask[
                            n_index, column_index, temperature_index,
                        ]
                        local_weight += (
                            n_weight * column_weight * temperature_weight
                            * corner_failed
                        )
            flat_output[indices] = local_weight

        return output
