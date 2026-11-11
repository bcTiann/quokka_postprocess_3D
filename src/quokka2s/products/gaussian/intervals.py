"""Shared inputs and an independent, tail-stable SciPy integration reference.

Velocity and sigma use the same arbitrary physical unit. Results are probability
mass in each bin, without normalization to the finite velocity window.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import erf, log_ndtr, roots_legendre


_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = roots_legendre(8)


@dataclass(frozen=True)
class VelocityGrid:
    v_min: float = -200.0
    v_max: float = 200.0
    delta_v: float = 1.0

    def __post_init__(self):
        values = np.asarray([self.v_min, self.v_max, self.delta_v])
        if not np.all(np.isfinite(values)):
            raise ValueError("Velocity grid parameters must be finite.")
        if self.delta_v <= 0.0 or self.v_max <= self.v_min:
            raise ValueError("Velocity grid needs positive width and spacing.")
        count = (self.v_max - self.v_min) / self.delta_v
        if not np.isclose(count, round(count), rtol=1e-12, atol=1e-12):
            raise ValueError("Velocity window must contain an integer number of bins.")

    @property
    def K(self):
        return int(round((self.v_max - self.v_min) / self.delta_v))

    @property
    def edges(self):
        return self.v_min + self.delta_v * np.arange(self.K + 1)


def validate_inputs(mu, sigma):
    """Return contiguous float64 vectors; preserve row order and reject NaNs."""
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if mu.ndim != 1 or sigma.ndim != 1 or mu.shape != sigma.shape:
        raise ValueError("mu and sigma must be one-dimensional vectors of equal length.")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        raise ValueError("mu and sigma must be finite.")
    if np.any(sigma <= 0.0):
        raise ValueError("sigma is a standard deviation and must be positive.")
    return np.ascontiguousarray(mu), np.ascontiguousarray(sigma)


def interval_probability(a, b):
    """Stable standard-normal mass in broadcast intervals [a, b].

    For example [8, 9] uses two *negative* log-CDF values, avoiding 1-1.
    Opposite-sign endpoints use erf, whose terms then cannot cancel.
    """
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(a > b):
        raise ValueError("Interval endpoints must satisfy a <= b and cannot be NaN.")
    result = np.zeros(a.shape, dtype=float)
    active = a < b
    # Even log-CDF differences lose digits for extraordinarily narrow intervals.
    # An independent SciPy 8-point Gaussian rule on these smooth tiny intervals
    # avoids endpoint subtraction in table building, direct integration and
    # independent original-cell window accounting.
    with np.errstate(over="ignore", invalid="ignore"):
        width = b - a
        midpoint = 0.5 * a + 0.5 * b
        narrow = active & np.isfinite(midpoint) & (width * (1.0 + np.abs(midpoint)) < 1e-5)
    half_width = 0.5 * width[narrow]
    points = midpoint[narrow, None] + half_width[:, None] * _QUADRATURE_NODES
    density = np.exp(-0.5 * points**2) / np.sqrt(2.0 * np.pi)
    result[narrow] = half_width * np.sum(density * _QUADRATURE_WEIGHTS, axis=-1)
    active = active & ~narrow
    negative = active & (b <= 0.0)
    positive = active & (a >= 0.0)
    crossing = active & ~(negative | positive)
    for selected, large_endpoint, small_endpoint in (
        (negative, b, a),
        (positive, -a, -b),
    ):
        log_large = log_ndtr(large_endpoint[selected])
        log_small = log_ndtr(small_endpoint[selected])
        result[selected] = np.exp(log_large) * -np.expm1(log_small - log_large)
    result[crossing] = 0.5 * (
        erf(b[crossing] / np.sqrt(2.0)) - erf(a[crossing] / np.sqrt(2.0))
    )
    return result
