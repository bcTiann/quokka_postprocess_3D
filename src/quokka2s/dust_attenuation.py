"""One-sided foreground dust extinction for the adopted LOS-z spectra.

The opacity is the Draine Milky Way R_V=3.1 total-extinction cross-section
per H nucleus. The observer plane is at the outer -z boundary face
of each (x, y) column, matching the manuscript's no-external-column setup.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DRAINE_TABLE = REPO_ROOT / "vendor/draine/kext_albedo_WD_MW_3.1_60_D03.all"
DRAINE_TABLE_SHA256 = "b56680cc38b85f051f20c4405303e8c480cc9bec714fd5ba722a257a40ae840c"
SPEED_OF_LIGHT_CM_S = 2.99792458e10

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
    """Return increasing vacuum wavelengths (micron) and C_ext/H (cm²/H)."""
    path = Path(path)
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != DRAINE_TABLE_SHA256:
        raise ValueError(f"Unexpected Draine MW R_V=3.1 opacity table: {path}")
    rows = []
    after_header = False
    for line in raw.decode("ascii").splitlines():
        if line.startswith("-----------"):
            after_header = True
            continue
        if after_header and line.strip():
            fields = line.split()
            if len(fields) < 6:
                raise ValueError("Expected six columns in Draine opacity table")
            values = [float(field) for field in fields[:6]]
            rows.append(values)
    data = np.asarray(rows, dtype=float)
    if (data.ndim != 2 or data.shape[1] != 6 or data.shape[0] < 2
            or not np.isfinite(data).all() or np.any(data[:, 0] <= 0)
            or np.any(data[:, 3] <= 0)):
        raise ValueError("Invalid Draine opacity table grid")
    # The published file has two out-of-order X-ray rows near 0.008 micron.
    # All adopted dust-attenuated lines are at wavelengths >= 0.0977 micron.
    order = np.argsort(data[:, 0])
    wavelength = data[order, 0]
    if not np.all(np.diff(wavelength) > 0):
        raise ValueError("Repeated wavelengths in Draine opacity table")
    return wavelength, data[order, 3].copy()


def extinction_cross_sections(line_keys, wavelength_micron, sigma_cm2_H):
    """Evaluate C_ext/H for the requested lines by log-log interpolation.

    H I 21 cm lies beyond the table's 1 cm maximum and is set to zero by
    the adopted approximation; other lines must be inside the table.
    """
    wavelength_micron = np.asarray(wavelength_micron, dtype=float)
    sigma_cm2_H = np.asarray(sigma_cm2_H, dtype=float)
    if (wavelength_micron.ndim != 1 or sigma_cm2_H.shape != wavelength_micron.shape
            or wavelength_micron.size < 2 or not np.isfinite(wavelength_micron).all()
            or not np.isfinite(sigma_cm2_H).all() or np.any(sigma_cm2_H <= 0)
            or not np.all(np.diff(wavelength_micron) > 0)):
        raise ValueError("Opacity interpolation requires increasing positive finite arrays")
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

    The last dimension is increasing z. Cells in front of the emitter
    contribute their full width; the emitting cell contributes half.
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
    """Apply exp(-sigma*N_H) to line-by-cell emissivities before binning."""
    epsilon = np.asarray(emissivity, dtype=float)
    column = np.asarray(foreground_NH, dtype=float)
    sigma = np.asarray(sigma_cm2_H, dtype=float)
    if (epsilon.ndim != 2 or column.shape != (epsilon.shape[1],)
            or sigma.shape != (epsilon.shape[0],) or not np.isfinite(epsilon).all()
            or not np.isfinite(column).all() or not np.isfinite(sigma).all()
            or np.any(epsilon < 0) or np.any(column < 0) or np.any(sigma < 0)):
        raise ValueError("Expected nonnegative finite emissivity, foreground column, and opacity")
    return epsilon * np.exp(-sigma[:, None] * column[None, :])
