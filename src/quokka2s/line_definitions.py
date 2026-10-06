"""Adopted rest wavelengths and emitter masses shared by the pipeline.

Cloudy labels supply the atomic wavelengths; DESPOTIC transition frequencies
supply the CO wavelengths. Emitter masses retain the existing thermal-width
values, including 1.00794 amu for hydrogen rather than the density conversion's
1.007947 amu. Build tokens, emission recipes, and plot labels live with their
respective consumers. Run arrays continue to follow their supplied line_keys.
"""
from __future__ import annotations

from dataclasses import dataclass

from quokka2s.constants import SPEED_OF_LIGHT_CM_S


@dataclass(frozen=True)
class LineDefinition:
    """The adopted physical description of one saved emission-line key.

    Parameters
    ----------
    key : str
        Stable saved identifier, for example "halpha" or "co10".
    rest_wavelength_micron : float
        Rest wavelength [micron], used to select the dust cross-section.
    emitter_mass_amu : float
        Emitting atom or molecule's mass [amu], used for thermal broadening.

    Returns
    -------
    LineDefinition
        Immutable line metadata, without any cell arrays or plot settings.

    Examples
    --------
    >>> line = LINE_DEFINITIONS["halpha"]
    >>> line.key, line.rest_wavelength_micron, line.emitter_mass_amu
    ('halpha', 0.656281, 1.00794)
    """

    key: str
    rest_wavelength_micron: float
    emitter_mass_amu: float


# Preserve the original dust diagnostic's iteration order. Consumers of saved
# arrays select definitions by their own line_keys instead of using this order.
LINE_DEFINITIONS: dict[str, LineDefinition] = {
    line.key: line
    for line in (
        LineDefinition(
            key="cii",
            rest_wavelength_micron=157.636,
            emitter_mass_amu=12.01,
        ),
        LineDefinition(
            key="halpha",
            rest_wavelength_micron=6562.81e-4,
            emitter_mass_amu=1.00794,
        ),
        LineDefinition(
            key="hi21",
            rest_wavelength_micron=21.1207e4,
            emitter_mass_amu=1.00794,
        ),
        LineDefinition(
            key="ciii_977",
            rest_wavelength_micron=977.020e-4,
            emitter_mass_amu=12.01,
        ),
        LineDefinition(
            key="ciii_1907",
            rest_wavelength_micron=1906.68e-4,
            emitter_mass_amu=12.01,
        ),
        LineDefinition(
            key="ciii_1909",
            rest_wavelength_micron=1908.73e-4,
            emitter_mass_amu=12.01,
        ),
        LineDefinition(
            key="civ_1548",
            rest_wavelength_micron=1548.19e-4,
            emitter_mass_amu=12.01,
        ),
        LineDefinition(
            key="civ_1551",
            rest_wavelength_micron=1550.78e-4,
            emitter_mass_amu=12.01,
        ),
        LineDefinition(
            key="co10",
            rest_wavelength_micron=SPEED_OF_LIGHT_CM_S / 115.271e9 * 1e4,
            emitter_mass_amu=28.009,
        ),
        LineDefinition(
            key="co21",
            rest_wavelength_micron=SPEED_OF_LIGHT_CM_S / 230.538e9 * 1e4,
            emitter_mass_amu=28.009,
        ),
    )
}
