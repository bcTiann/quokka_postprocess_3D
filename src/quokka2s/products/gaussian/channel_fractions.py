"""NumPy/SciPy Gaussian bin fractions for bounded channel/cell subbatches."""
from __future__ import annotations

import numpy as np
from scipy.special import erf


def channel_fractions(edges, centers, denominator, *, lookup_values=None):
    """Return fractions[channel,cell] from shared adjacent erf boundaries.

    edges and centers use km/s; denominator is sqrt(2)*max(sigma,1e-5).
    The optional existing 32769-value table samples erf on [0,8] at 1/4096.
    No five-sigma cutoff is introduced into these reference methods.
    """
    arguments = edges[:, None] - centers[None, :]
    if lookup_values is None:
        arguments /= denominator[None, :]
        erf(arguments, out=arguments)
    else:
        arguments *= np.reciprocal(denominator)[None, :]
        absolute = np.abs(arguments)
        coordinate = np.minimum(absolute, 8.0) * 4096.0
        finite = np.isfinite(coordinate)
        coordinate = np.where(finite, coordinate, 0.0)
        lower = np.minimum(coordinate.astype(np.intp), lookup_values.size - 2)
        fraction = coordinate - lower
        values = lookup_values[lower] + fraction * (lookup_values[lower + 1] - lookup_values[lower])
        values[absolute >= 8.0] = 1.0
        values[arguments < 0.0] *= -1.0
        values[np.isnan(arguments)] = np.nan
        arguments = values
    fractions = np.diff(arguments, axis=0)
    fractions *= 0.5
    return fractions
