"""Independent default compositions for Cloudy and DESPOTIC line tables.

Cloudy uses unscaled C17.02 default.abn. DESPOTIC uses the pinned GOW
network defaults; its C/O/Si abundances are not taken from Cloudy.
"""
from __future__ import annotations

from types import MappingProxyType


ABUNDANCE_SETUP = "cloudy_c17_02_default_gow_default_v2"
SUPERSEDED_ABUNDANCE_SETUP = "quokka_xyz_cloudy_c17_02_default_v1"
# Legacy simulation density conversion, retained independently of the table
# compositions. Do not infer either code's elemental abundances from this XYZ.
QUOKKA_MASS_FRACTIONS = MappingProxyType({
    "X": 0.7157683773530885,
    "Y": 0.26423162264691147,
    "Z": 0.02,
})
METAL_REFERENCE_SCALE = 1.0
# GOW.py defaults at the DESPOTIC commit pinned in pyproject.toml.
GOW_ELEMENTAL_ABUNDANCES = MappingProxyType({
    "xHe": 0.1,
    "xC": 1.6e-4,
    "xO": 3.2e-4,
    "xSi": 1.7e-6,
})


def abundance_metadata() -> dict[str, object]:
    """Return independent, JSON-compatible provenance for a new table."""
    return {
        "setup": ABUNDANCE_SETUP,
        "normalization": "element nuclei per hydrogen nucleus",
        "metal_reference": "Cloudy 17.02 default.abn, unscaled",
        "metal_reference_scale": METAL_REFERENCE_SCALE,
        "cloudy_helium_abundance": 0.1,
        "despotic_reference": "native GOW default He/C/O/Si abundances",
        "gow_elemental_abundances": dict(GOW_ELEMENTAL_ABUNDANCES),
        "mu_definition": "DESPOTIC native composition.computeDerived",
    }


def reject_superseded_composition(setup: str | None) -> None:
    """Prevent mixing the withdrawn XYZ-corrected tables into new runs."""
    if setup == SUPERSEDED_ABUNDANCE_SETUP:
        raise ValueError(
            "This table uses the superseded XYZ-corrected composition. "
            "Rebuild with Cloudy/GOW defaults, or explicitly allow the old "
            "composition for historical inspection only."
        )
