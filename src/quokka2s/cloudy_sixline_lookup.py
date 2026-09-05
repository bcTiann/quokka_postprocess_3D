"""Strict sampler for the seven-field, six-line Cloudy Jeans table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


TOUCH_EPS = 1.0e-12
EXPECTED_AXIS_ORDER = "line,log_NH_attenuation,log_nH,log_T"


class CloudyFailureTouchError(RuntimeError):
    """Raised when a query gives a failed Cloudy node positive weight."""


@dataclass(frozen=True)
class CloudySixLineSample:
    """Interpolated coefficients and the applied attenuation-axis clipping."""

    emissivity_per_nH2: np.ndarray
    attenuation_column_below_table: np.ndarray
    attenuation_column_above_table: np.ndarray


@dataclass(frozen=True)
class CloudySixLineDiagnostics:
    """Failure-touch and attenuation-clipping diagnostics for input cells."""

    failure_touched: np.ndarray
    maximum_failure_weight: float
    attenuation_column_below_table: np.ndarray
    attenuation_column_above_table: np.ndarray


def _validate_axis(name: str, axis: np.ndarray) -> None:
    if axis.ndim != 1 or axis.size < 2 or np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be a strictly increasing 1D axis")


def _brackets(
    axis: np.ndarray,
    coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper = np.searchsorted(axis, coordinate, side="right")
    upper = np.clip(upper, 1, axis.size - 1)
    lower = upper - 1
    fraction = (coordinate - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, fraction


class CloudySixLineLookup:
    """Trilinear lookup in attenuation column, density, and temperature.

    The simulation column is clipped only on the attenuation axis. Density and
    temperature must lie inside the table. Positive corners are interpolated
    in log emissivity coefficient; if an exact-zero Cloudy corner contributes,
    interpolation switches to the non-negative linear coefficient. Failed
    Cloudy nodes are never silently filled.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as source:
            required = {
                "axis_order",
                "line_keys",
                "log_NH_attenuation",
                "log_nH",
                "log_T",
                "log_emissivity_per_nH2",
                "emissivity_per_nH2",
                "failure_mask",
                "zero_mask",
            }
            missing = sorted(required - set(source.files))
            if missing:
                raise ValueError(f"Cloudy table is missing fields: {missing}")
            axis_order = str(np.asarray(source["axis_order"]).item())
            self.line_keys = tuple(
                str(value) for value in np.asarray(source["line_keys"]).tolist()
            )
            self.log_NH_attenuation = np.asarray(
                source["log_NH_attenuation"], dtype=float
            )
            self.log_nH = np.asarray(source["log_nH"], dtype=float)
            self.log_T = np.asarray(source["log_T"], dtype=float)
            self.log_emissivity_per_nH2 = np.asarray(
                source["log_emissivity_per_nH2"], dtype=float
            )
            self.emissivity_per_nH2 = np.asarray(
                source["emissivity_per_nH2"], dtype=float
            )
            self.failure_mask = np.asarray(source["failure_mask"], dtype=bool)
            self.zero_mask = np.asarray(source["zero_mask"], dtype=bool)
            self.metadata = {
                name: np.asarray(source[name])
                for name in source.files
                if name not in required
            }

        if axis_order != EXPECTED_AXIS_ORDER:
            raise ValueError(f"unexpected Cloudy axis order: {axis_order!r}")
        _validate_axis("log_NH_attenuation", self.log_NH_attenuation)
        _validate_axis("log_nH", self.log_nH)
        _validate_axis("log_T", self.log_T)
        expected = (
            len(self.line_keys),
            self.log_NH_attenuation.size,
            self.log_nH.size,
            self.log_T.size,
        )
        for name, array in (
            ("log_emissivity_per_nH2", self.log_emissivity_per_nH2),
            ("emissivity_per_nH2", self.emissivity_per_nH2),
            ("failure_mask", self.failure_mask),
            ("zero_mask", self.zero_mask),
        ):
            if array.shape != expected:
                raise ValueError(f"{name} has shape {array.shape}, expected {expected}")
        if np.any(self.emissivity_per_nH2 < 0.0):
            raise ValueError("Cloudy emissivity coefficients must be non-negative")
        if not np.isfinite(self.emissivity_per_nH2).all():
            raise ValueError("Cloudy linear emissivity coefficients must be finite")
        expected_zero = (self.emissivity_per_nH2 == 0.0) & ~self.failure_mask
        if not np.array_equal(self.zero_mask, expected_zero):
            raise ValueError("zero_mask disagrees with linear coefficients")
        if np.any(self.emissivity_per_nH2[self.failure_mask] != 0.0):
            raise ValueError("failed nodes must have zero placeholder coefficients")

    @property
    def attenuation_column_bounds_cm2(self) -> tuple[float, float]:
        return (
            float(10.0 ** self.log_NH_attenuation[0]),
            float(10.0 ** self.log_NH_attenuation[-1]),
        )

    def sample(
        self,
        temperature_K,
        n_H_cm3,
        column_density_H_cm2,
    ) -> CloudySixLineSample:
        temperature, n_h, column = np.broadcast_arrays(
            np.asarray(temperature_K, dtype=float),
            np.asarray(n_H_cm3, dtype=float),
            np.asarray(column_density_H_cm2, dtype=float),
        )
        if not (
            np.isfinite(temperature).all()
            and np.isfinite(n_h).all()
            and np.isfinite(column).all()
        ):
            raise ValueError("Cloudy lookup inputs must all be finite")
        if np.any((temperature <= 0.0) | (n_h <= 0.0) | (column <= 0.0)):
            raise ValueError("Cloudy lookup inputs must all be positive")

        original_shape = temperature.shape
        log_column = np.log10(column).ravel()
        below = log_column < self.log_NH_attenuation[0]
        above = log_column > self.log_NH_attenuation[-1]
        coordinates = (
            np.clip(
                log_column,
                self.log_NH_attenuation[0],
                self.log_NH_attenuation[-1],
            ),
            np.log10(n_h).ravel(),
            np.log10(temperature).ravel(),
        )
        for name, axis, coordinate in (
            ("log_nH", self.log_nH, coordinates[1]),
            ("log_T", self.log_T, coordinates[2]),
        ):
            tolerance = 1.0e-12 * max(1.0, abs(axis[0]), abs(axis[-1]))
            if np.any(coordinate < axis[0] - tolerance) or np.any(
                coordinate > axis[-1] + tolerance
            ):
                raise ValueError(
                    f"{name} is outside [{axis[0]:.8g}, {axis[-1]:.8g}]"
                )
        coordinates = (
            coordinates[0],
            np.clip(coordinates[1], self.log_nH[0], self.log_nH[-1]),
            np.clip(coordinates[2], self.log_T[0], self.log_T[-1]),
        )
        brackets = tuple(
            _brackets(axis, coordinate)
            for axis, coordinate in zip(
                (self.log_NH_attenuation, self.log_nH, self.log_T), coordinates
            )
        )

        n_points = coordinates[0].size
        linear_sum = np.zeros((len(self.line_keys), n_points))
        log_sum = np.zeros_like(linear_sum)
        zero_support = np.zeros_like(linear_sum, dtype=bool)
        failure_weight = np.zeros_like(linear_sum)

        def visit(
            axis_number: int,
            indices: list[np.ndarray],
            weight: np.ndarray,
        ) -> None:
            if axis_number == len(brackets):
                local_index = (slice(None), *indices)
                local_weight = weight[None, :]
                local_log = self.log_emissivity_per_nH2[local_index]
                linear_sum[:] += self.emissivity_per_nH2[local_index] * local_weight
                log_sum[:] += np.where(
                    np.isfinite(local_log), local_log, 0.0
                ) * local_weight
                zero_support[:] |= self.zero_mask[local_index] & (
                    local_weight > TOUCH_EPS
                )
                failure_weight[:] += self.failure_mask[local_index] * local_weight
                return
            lower, upper, fraction = brackets[axis_number]
            visit(axis_number + 1, indices + [lower], weight * (1.0 - fraction))
            visit(axis_number + 1, indices + [upper], weight * fraction)

        visit(0, [], np.ones(n_points))
        touched = failure_weight > TOUCH_EPS
        if np.any(touched):
            counts = {
                key: int(np.count_nonzero(touched[index]))
                for index, key in enumerate(self.line_keys)
                if np.any(touched[index])
            }
            raise CloudyFailureTouchError(
                "simulation touches unavailable Cloudy nodes: "
                f"counts={counts}, maximum_weight={failure_weight[touched].max():.6g}"
            )
        coefficient = np.where(zero_support, linear_sum, np.power(10.0, log_sum))
        return CloudySixLineSample(
            emissivity_per_nH2=coefficient.reshape(
                (len(self.line_keys), *original_shape)
            ),
            attenuation_column_below_table=below.reshape(original_shape),
            attenuation_column_above_table=above.reshape(original_shape),
        )

    def diagnose(
        self,
        temperature_K,
        n_H_cm3,
        column_density_H_cm2,
    ) -> CloudySixLineDiagnostics:
        """Return all failed-node touches without aborting at the first cell."""
        temperature, n_h, column = np.broadcast_arrays(
            np.asarray(temperature_K, dtype=float),
            np.asarray(n_H_cm3, dtype=float),
            np.asarray(column_density_H_cm2, dtype=float),
        )
        if not (
            np.isfinite(temperature).all()
            and np.isfinite(n_h).all()
            and np.isfinite(column).all()
        ):
            raise ValueError("Cloudy lookup inputs must all be finite")
        if np.any((temperature <= 0.0) | (n_h <= 0.0) | (column <= 0.0)):
            raise ValueError("Cloudy lookup inputs must all be positive")

        original_shape = temperature.shape
        log_column = np.log10(column).ravel()
        below = log_column < self.log_NH_attenuation[0]
        above = log_column > self.log_NH_attenuation[-1]
        coordinates = (
            np.clip(
                log_column,
                self.log_NH_attenuation[0],
                self.log_NH_attenuation[-1],
            ),
            np.log10(n_h).ravel(),
            np.log10(temperature).ravel(),
        )
        for name, axis, coordinate in (
            ("log_nH", self.log_nH, coordinates[1]),
            ("log_T", self.log_T, coordinates[2]),
        ):
            tolerance = 1.0e-12 * max(1.0, abs(axis[0]), abs(axis[-1]))
            if np.any(coordinate < axis[0] - tolerance) or np.any(
                coordinate > axis[-1] + tolerance
            ):
                raise ValueError(
                    f"{name} is outside [{axis[0]:.8g}, {axis[-1]:.8g}]"
                )
        coordinates = (
            coordinates[0],
            np.clip(coordinates[1], self.log_nH[0], self.log_nH[-1]),
            np.clip(coordinates[2], self.log_T[0], self.log_T[-1]),
        )
        brackets = tuple(
            _brackets(axis, coordinate)
            for axis, coordinate in zip(
                (self.log_NH_attenuation, self.log_nH, self.log_T), coordinates
            )
        )
        failure_weight = np.zeros((len(self.line_keys), coordinates[0].size))

        def visit(
            axis_number: int,
            indices: list[np.ndarray],
            weight: np.ndarray,
        ) -> None:
            if axis_number == len(brackets):
                failure_weight[:] += (
                    self.failure_mask[(slice(None), *indices)] * weight[None, :]
                )
                return
            lower, upper, fraction = brackets[axis_number]
            visit(axis_number + 1, indices + [lower], weight * (1.0 - fraction))
            visit(axis_number + 1, indices + [upper], weight * fraction)

        visit(0, [], np.ones(coordinates[0].size))
        touched = failure_weight > TOUCH_EPS
        return CloudySixLineDiagnostics(
            failure_touched=touched.reshape((len(self.line_keys), *original_shape)),
            maximum_failure_weight=(
                float(failure_weight[touched].max()) if np.any(touched) else 0.0
            ),
            attenuation_column_below_table=below.reshape(original_shape),
            attenuation_column_above_table=above.reshape(original_shape),
        )
