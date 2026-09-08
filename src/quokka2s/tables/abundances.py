"""Shared elemental composition adopted for the QUOKKA line tables.

The metal pattern is Cloudy 17.02's ``default.abn`` (the C13 reference
pattern), normalized to the adopted total metal mass fraction.  GOW takes
only He, C, O, and Si from that composition; other metals are not
redistributed among the elements represented by GOW.
"""
from __future__ import annotations

from types import MappingProxyType


ABUNDANCE_SETUP = "quokka_xyz_cloudy_c17_02_default_v1"
QUOKKA_MASS_FRACTIONS = MappingProxyType({
    "X": 0.7157683773530885,
    "Y": 0.26423162264691147,
    "Z": 0.02,
})
# Obtained by preserving Zi/ZC = (mi/mC) * (alpha_i_ref/alpha_C_ref)
# and requiring sum(Zi for all metals) = Z, then alpha_i = Zi*mH/(X*mi).
METAL_REFERENCE_SCALE = 1.5242485583008916
GOW_ELEMENTAL_ABUNDANCES = MappingProxyType({
    "xHe": 0.09296180232949455,
    "xC": 0.0003734408967837184,
    "xO": 0.0007468817935674368,
    "xSi": 5.289142497304094e-05,
})


def abundance_metadata() -> dict[str, object]:
    """Return independent, JSON-compatible provenance for a new table."""
    return {
        "setup": ABUNDANCE_SETUP,
        "normalization": "element nuclei per hydrogen nucleus",
        "mass_fractions": dict(QUOKKA_MASS_FRACTIONS),
        "metal_reference": "Cloudy 17.02 default.abn; C13 reference metal pattern",
        "metal_reference_scale": METAL_REFERENCE_SCALE,
        "gow_elemental_abundances": dict(GOW_ELEMENTAL_ABUNDANCES),
        "unrepresented_metals": "omitted from GOW without redistribution",
        "mu_definition": "DESPOTIC native composition.computeDerived",
    }
