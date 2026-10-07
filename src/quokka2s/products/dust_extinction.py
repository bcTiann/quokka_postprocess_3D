"""Prepare the full Draine extinction grid and the adopted line samples."""
from __future__ import annotations

from pathlib import Path

import numpy as np

DUST_EXTINCTION_STEM = "draine_mw31_full_range_and_lines"


def prepare_dust_extinction_data(table_path=None):
    """Read and sample Draine once, returning arrays and their source report.

    Parameters
    ----------
    table_path : str, Path or None
        The MW R_V=3.1 extinction table, with 1077 wavelength rows.
        None selects the bundled default table.

    Returns
    -------
    payload : dict[str, ndarray]
        wavelength_micron and sigma_ext_cm2_H have shape (1077,).
        line_keys, line_wavelength_micron and line_sigma_ext_cm2_H have
        shape (10,), in LINE_DEFINITIONS order. HI21's cross-section is zero
        under the adopted approximation, although its wavelength is saved.
    report : dict
        Source path, full table range, and named line samples for the JSON.

    Example: the saved line_sigma_ext_cm2_H entry for 'halpha' is already
    interpolated; the plot reads it without opening Draine's text file.
    """
    from quokka2s.line_definitions import LINE_DEFINITIONS
    from quokka2s.physics.dust_attenuation import (
        DEFAULT_DRAINE_TABLE,
        extinction_cross_sections,
        load_draine_extinction,
    )

    if table_path is None:
        table_path = DEFAULT_DRAINE_TABLE
    wavelength, sigma = load_draine_extinction(path=table_path)
    if wavelength[0] != 1.e-4 or wavelength[-1] != 1.e4 or wavelength.size != 1077:
        raise ValueError("Unexpected Draine table wavelength grid")
    line_keys = tuple(LINE_DEFINITIONS)
    line_wavelength = np.asarray([
        LINE_DEFINITIONS[key].rest_wavelength_micron for key in line_keys
    ])
    line_sigma = extinction_cross_sections(
        line_keys=line_keys,
        wavelength_micron=wavelength,
        sigma_cm2_H=sigma,
    )
    payload = {
        "wavelength_micron": wavelength,
        "sigma_ext_cm2_H": sigma,
        "line_keys": np.asarray(line_keys),
        "line_wavelength_micron": line_wavelength,
        "line_sigma_ext_cm2_H": line_sigma,
    }
    report = {
        "source": str(Path(table_path)),
        "table_rows": int(wavelength.size),
        "table_wavelength_micron": [float(wavelength[0]), float(wavelength[-1])],
        "line_wavelength_micron": dict(zip(line_keys, line_wavelength.tolist())),
        "line_sigma_ext_cm2_H": dict(zip(line_keys, line_sigma.tolist())),
        "hi21": "Outside the 1 cm table limit; no dust attenuation applied",
    }
    return payload, report
