"""Read Cloudy's text exports without importing snapshot or plotting code."""

from pathlib import Path

import numpy as np


def read_incident_spectrum(
    path: Path,
    *,
    usecols: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Load an incident-continuum export with increasing photon energies.

    Parameters
    ----------
    path : Path
        Cloudy ``save incident continuum`` text file. Lines beginning with
        ``#`` are comments. Column 0 is photon energy [Ryd]; column 1 is the
        incident continuum [erg cm^-2 s^-1] used by the current SED tools.
    usecols : tuple of int or None
        None retains every numeric export column for the SED generator.
        Figure tools pass (0, 1) to read only energy and incident intensity;
        unused columns are then neither parsed nor returned.

    Returns
    -------
    ndarray, shape (N, C)
        Selected columns in their original order and values, with C >= 2.
        The first returned column must increase strictly. A malformed export
        or a single-row export raises ValueError, as in the existing tools.

    Example
    -------
    For rows ``0.1 2e-6 9`` and ``1.0 3e-6 8``, passing usecols=(0, 1)
    returns ``[[0.1, 2e-6], [1.0, 3e-6]]`` with shape (2, 2).
    """
    data = np.loadtxt(
        fname=path,
        comments="#",
        usecols=usecols,
    )
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"unexpected save incident continuum format: {path}")
    if np.any(np.diff(data[:, 0]) <= 0.0):
        raise ValueError(f"non-increasing energy mesh: {path}")
    return data
