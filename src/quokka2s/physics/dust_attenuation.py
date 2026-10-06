"""One-sided foreground dust extinction for the adopted LOS-z spectra.

The opacity is the Draine Milky Way R_V=3.1 total-extinction cross-section
per H nucleus. The observer plane is at the outer -z boundary face
of each (x, y) column, matching the manuscript's no-external-column setup.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quokka2s.constants import SPEED_OF_LIGHT_CM_S


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DRAINE_TABLE = REPO_ROOT / "vendor/draine/kext_albedo_WD_MW_3.1_60_D03.all"

# Cloudy labels for the atomic lines; DESPOTIC's CO transition frequencies.
LINE_WAVELENGTH_MICRON = {
    "cii": 157.636,
    "halpha": 6562.81e-4,
    "hi21": 21.1207e4,
    "ciii_977": 977.020e-4,
    "ciii_1907": 1906.68e-4,
    "ciii_1909": 1908.73e-4,
    "civ_1548": 1548.19e-4,
    "civ_1551": 1550.78e-4,
    "co10": SPEED_OF_LIGHT_CM_S / 115.271e9 * 1e4,
    "co21": SPEED_OF_LIGHT_CM_S / 230.538e9 * 1e4,
}


def load_draine_extinction(path=DEFAULT_DRAINE_TABLE):
    """Read the two Draine columns used by extinction_cross_sections().

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Draine text table; defaults to the bundled Milky Way R_V=3.1 file.

    Returns
    -------
    wavelength, extinction : tuple of numpy.ndarray, each shape (N,)
        Increasing wavelengths [micron] and matching C_ext/H [cm^2/H].
        The bundled file has N=1077 rows.

    Examples
    --------
    >>> wave, sigma = load_draine_extinction()
    >>> wave.shape, sigma.shape
    ((1077,), (1077,))
    """
    with Path(path).open(encoding="ascii") as table:
        # Skip the explanatory header; numeric rows follow the dashed separator.
        for line in table:
            if line.startswith("-----------"):
                break
        else:
            raise ValueError("Missing data separator in Draine opacity table")

        # Read columns 1 and 4 as named fields: wavelength [micron]
        # and extinction cross-section per H nucleus [cm^2/H].
        data = np.loadtxt(
            table, usecols=(0, 3),
            dtype=[("wavelength", float), ("extinction", float)], ndmin=1,
        )

    # The file runs mostly from 10000 down to 0.0001 micron, with two rows
    # out of order. np.interp needs increasing wavelengths. Sort whole rows
    # by wavelength, keeping each extinction value with its wavelength.
    data.sort(order="wavelength")
    return data["wavelength"], data["extinction"]


def extinction_cross_sections(
    line_keys: Sequence[str],
    wavelength_micron: ArrayLike,
    sigma_cm2_H: ArrayLike,
) -> NDArray[np.float64]:
    """Get one dust extinction cross-section per line by log-log interpolation.

    Parameters
    ----------
    line_keys : list or tuple of str
        Requested line names; rest wavelengths come from LINE_WAVELENGTH_MICRON.
    wavelength_micron : array-like, shape (N,)
        Sorted table wavelengths [micron], returned as the first output
        of load_draine_extinction(). N is the number of table rows.
    sigma_cm2_H : array-like, shape (N,)
        Matching table cross-sections [cm^2/H], returned as the second output
        of load_draine_extinction().

    Returns
    -------
    numpy.ndarray of float64, shape (len(line_keys),)
        Cross-sections [cm^2/H], in input line order; HI 21 cm is zero.

    Examples
    --------
    >>> wave, sigma = load_draine_extinction()
    >>> result = extinction_cross_sections(("halpha", "cii", "hi21"), wave, sigma)
    >>> [f"{value:.3e}" for value in result]
    ['3.809e-22', '2.243e-25', '0.000e+00']
    """
    wavelength_micron = np.asarray(wavelength_micron, dtype=float)
    sigma_cm2_H = np.asarray(sigma_cm2_H, dtype=float)
    result = []
    for key in line_keys:
        if key not in LINE_WAVELENGTH_MICRON:
            raise ValueError(f"No adopted rest wavelength for {key}")
        if key == "hi21":
            result.append(0.)
            continue
        wavelength = LINE_WAVELENGTH_MICRON[key]
        if not wavelength_micron[0] <= wavelength <= wavelength_micron[-1]:
            raise ValueError(f"Line {key} lies outside the Draine opacity grid")
        value = np.exp(np.interp(np.log(wavelength), np.log(wavelength_micron),
                                 np.log(sigma_cm2_H)))
        result.append(float(value))
    return np.asarray(result)


def observer_side_hydrogen_column(nH_cm3, dz_cm):
    """Column from the outer -z boundary face to each emitting cell centre.

    Parameters
    ----------
    nH_cm3 : array-like, shape (Nx, Ny, Nz)
        Hydrogen-nuclei density [cm^-3], derived from rho in read_slab().
        The last axis follows increasing z and spans the full column.
    dz_cm : float
        Cell width along z [cm], from snapshot.cell_widths.

    Returns
    -------
    numpy.ndarray, shape (Nx, Ny, Nz)
        Foreground column [cm^-2]. Foreground cells contribute full widths;
        each emitting cell contributes half. Used by attenuate_emissivities().

    Examples
    --------
    >>> observer_side_hydrogen_column([[[1., 2., 4.]]], 3.)
    array([[[ 1.5,  6. , 15. ]]])
    """
    nH = np.asarray(nH_cm3, dtype=float)
    dz = float(dz_cm)
    if (nH.ndim != 3 or not nH.shape[-1] or not np.isfinite(nH).all()
            or np.any(nH < 0) or not np.isfinite(dz) or dz <= 0):
        raise ValueError("Expected finite nonnegative (x,y,z) nH and positive dz")
    integrated = np.cumsum(nH, axis=-1, dtype=float)
    integrated -= .5 * nH
    integrated *= dz
    return integrated


def attenuate_emissivities(emissivity, foreground_NH, sigma_cm2_H):
    """Apply exp(-sigma*N_H) to each line's cell emissivities.

    Parameters
    ----------
    emissivity : array-like, shape (N,) or (L, N)
        Intrinsic emissivity [erg s^-1 cm^-3] from the lines returned by
        CellEmissionCalculator.calculate_intrinsic_lines(), selected for
        available entries. L is the line count; N is the selected cell count.
    foreground_NH : array-like, shape (N,)
        Foreground columns [cm^-2] from observer_side_hydrogen_column(),
        flattened and selected for the same cells.
    sigma_cm2_H : scalar or array-like, shape (L,)
        Cross-section [cm^2/H] for one line, or cross-sections in line order.

    Returns
    -------
    numpy.ndarray, same shape as emissivity
        Attenuated emissivity [erg s^-1 cm^-3], with the same line/cell order.

    Examples
    --------
    >>> attenuate_emissivities([4., 4.], [0., 2.], 0.5).round(3)
    array([4.   , 1.472])
    """
    epsilon = np.asarray(emissivity, dtype=float)
    column = np.asarray(foreground_NH, dtype=float)
    sigma = np.asarray(sigma_cm2_H, dtype=float)
    if (epsilon.ndim not in (1, 2) or column.shape != (epsilon.shape[-1],)
            or sigma.shape != epsilon.shape[:-1] or not np.isfinite(epsilon).all()
            or not np.isfinite(column).all() or not np.isfinite(sigma).all()
            or np.any(epsilon < 0) or np.any(column < 0) or np.any(sigma < 0)):
        raise ValueError("Expected nonnegative finite emissivity, foreground column, and opacity")
    return epsilon * np.exp(-sigma[..., None] * column)
