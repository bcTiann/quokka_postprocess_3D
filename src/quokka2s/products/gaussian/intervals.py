"""Shared inputs and an independent, tail-stable SciPy integration reference.

Velocity and sigma use the same arbitrary physical unit. Results are probability
mass in each bin, without normalization to the finite velocity window.
"""

import numpy as np
from scipy.special import erf, log_ndtr, roots_legendre


_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = roots_legendre(8)


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
    if np.any(narrow):
        half_width = 0.5 * width[narrow]
        points = midpoint[narrow, None] + half_width[:, None] * _QUADRATURE_NODES
        density = np.exp(-0.5 * points**2) / np.sqrt(2.0 * np.pi)
        result[narrow] = half_width * np.sum(density * _QUADRATURE_WEIGHTS, axis=-1)
    active = active & ~narrow
    negative = active & (b <= 0.0)
    positive = active & (a >= 0.0)
    crossing = active & ~(negative | positive)
    if np.any(negative):
        log_large = log_ndtr(b[negative])
        log_small = log_ndtr(a[negative])
        result[negative] = np.exp(log_large) * -np.expm1(log_small - log_large)
    if np.any(positive):
        # Reflect selected positive-tail endpoints, not both full input arrays.
        log_large = log_ndtr(-a[positive])
        log_small = log_ndtr(-b[positive])
        result[positive] = np.exp(log_large) * -np.expm1(log_small - log_large)
    if np.any(crossing):
        result[crossing] = 0.5 * (
            erf(b[crossing] / np.sqrt(2.0)) - erf(a[crossing] / np.sqrt(2.0))
        )
    return result
