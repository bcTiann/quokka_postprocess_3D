"""Display names for the saved emission-line keys."""

from quokka2s.line_definitions import LINE_DEFINITIONS


def format_uv_line_title(
    *,
    line_key: str,
    ion_label: str,
    decimal_places: int,
) -> str:
    """Format an adopted UV wavelength with the figure's existing precision.

    Parameters
    ----------
    line_key : str
        Key in LINE_DEFINITIONS, for example "ciii_977".
    ion_label : str
        Displayed ion name, for example "C III"; owned by the figure module.
    decimal_places : int
        Decimal digits in angstroms: three for C III 977, two for other UV lines.

    Returns
    -------
    str
        Figure title containing the ion name, wavelength [angstrom], and unit.

    Examples
    --------
    >>> format_uv_line_title(line_key="ciii_977", ion_label="C III", decimal_places=3)
    'C III 977.020 $\\\\AA$'
    """
    wavelength_angstrom = LINE_DEFINITIONS[line_key].rest_wavelength_micron * 1e4
    return rf"{ion_label} {wavelength_angstrom:.{decimal_places}f} $\AA$"


LINE_TITLES = {
    "cii": r"C II 158 $\mu$m", "halpha": r"H$\alpha$", "hi21": "H I 21 cm",
    "ciii_977": format_uv_line_title(
        line_key="ciii_977",
        ion_label="C III",
        decimal_places=3,
    ),
    "ciii_1907": format_uv_line_title(
        line_key="ciii_1907",
        ion_label="C III",
        decimal_places=2,
    ),
    "ciii_1909": format_uv_line_title(
        line_key="ciii_1909",
        ion_label="C III",
        decimal_places=2,
    ),
    "civ_1548": format_uv_line_title(
        line_key="civ_1548",
        ion_label="C IV",
        decimal_places=2,
    ),
    "civ_1551": format_uv_line_title(
        line_key="civ_1551",
        ion_label="C IV",
        decimal_places=2,
    ),
    "co10": "CO(1-0)", "co21": "CO(2-1)",
}
