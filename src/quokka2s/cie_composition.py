"""Convert CHIANTI CIE ion fractions into simulation number densities."""

from __future__ import annotations

import numpy as np


def helium_abundance_per_hydrogen(*, hydrogen_mass_fraction: float, helium_mass_fraction: float) -> float:
    """Return n_He/n_H from H/He mass fractions using m_He ~= 4 m_H."""
    if hydrogen_mass_fraction <= 0.0:
        raise ValueError('hydrogen_mass_fraction must be positive')
    if helium_mass_fraction < 0.0:
        raise ValueError('helium_mass_fraction must be non-negative')
    return helium_mass_fraction / (4.0 * hydrogen_mass_fraction)


def cie_h_he_charge_per_hydrogen(
    hydrogen_fractions,
    helium_fractions,
    *,
    hydrogen_mass_fraction: float,
    helium_mass_fraction: float,
):
    """Return ``(n_e/n_H, n_p/n_H)`` for an H+He CIE composition.

    Each fractions array has ionization stage on its last axis, ordered as
    neutral, +1, +2, ... exactly as returned by
    ``fiasco.Element(...).equilibrium_ionization``.  ``n_p`` denotes H+ only;
    electrons from helium are included in ``n_e``.
    """
    x_h = np.asarray(hydrogen_fractions, dtype=float)
    x_he = np.asarray(helium_fractions, dtype=float)
    if x_h.shape[:-1] != x_he.shape[:-1]:
        raise ValueError('H and He CIE fraction grids must share leading dimensions')
    if x_h.shape[-1] < 2 or x_he.shape[-1] < 3:
        raise ValueError('CIE fraction arrays do not contain the required ion stages')

    n_he_over_n_h = helium_abundance_per_hydrogen(
        hydrogen_mass_fraction=hydrogen_mass_fraction,
        helium_mass_fraction=helium_mass_fraction,
    )
    charge_h = np.arange(x_h.shape[-1], dtype=float)
    charge_he = np.arange(x_he.shape[-1], dtype=float)

    proton_per_h = x_h[..., 1]
    electron_per_h = (
        np.sum(x_h * charge_h, axis=-1)
        + n_he_over_n_h * np.sum(x_he * charge_he, axis=-1)
    )
    return electron_per_h, proton_per_h
