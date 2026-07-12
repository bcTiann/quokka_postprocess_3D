"""Shared temperature-regime definitions for line-emission physics.

This module deliberately has no yt/fiasco imports so the regime bookkeeping
and mean-molecular-weight inversion can be unit-tested in isolation.
"""

from __future__ import annotations

import numpy as np


HYDROGEN_MASS_FRACTION = 0.74
HELIUM_MASS_FRACTION = 0.26
QUOKKA_ADIABATIC_INDEX = 5.0 / 3.0


def temperature_regime_masks(
    T_quokka_K: np.ndarray,
    despotic_cutoff_K: float,
    cie_cutoff_K: float,
):
    """Return masks for three temperature bins using the supplied boundaries."""
    temperature = np.asarray(T_quokka_K, dtype=np.float64)
    low = temperature < despotic_cutoff_K
    intermediate = ((temperature >= despotic_cutoff_K)
                    & (temperature < cie_cutoff_K))
    high = temperature >= cie_cutoff_K
    return low, intermediate, high


def hydrogen_ionization_fraction_from_mean_molecular_weight(
    internal_energy_density_erg_cm3: np.ndarray,
    density_g_cm3: np.ndarray,
    temperature_K: np.ndarray,
    *,
    hydrogen_mass_g: float,
    boltzmann_erg_K: float,
    gamma: float = QUOKKA_ADIABATIC_INDEX,
    X: float = HYDROGEN_MASS_FRACTION,
    Y: float = HELIUM_MASS_FRACTION,
) -> np.ndarray:
    """Infer ``x_e = n_e/n_H`` from QUOKKA's internal energy and temperature.

    Implements the methodology equations

        1/mu = (gamma-1) m_H e_int / (rho k_B T)
        x_e  = (1/mu - X - Y/4) / X.

    The result is clipped to [0, 1], the physical domain of the assumed
    hydrogen ionization fraction. Non-finite input combinations map to zero.
    """
    e_int = np.asarray(internal_energy_density_erg_cm3, dtype=np.float64)
    rho = np.asarray(density_g_cm3, dtype=np.float64)
    temperature = np.asarray(temperature_K, dtype=np.float64)

    denominator = rho * boltzmann_erg_K * temperature
    inverse_mu = np.divide(
        (gamma - 1.0) * hydrogen_mass_g * e_int,
        denominator,
        out=np.zeros(np.broadcast_shapes(e_int.shape, denominator.shape), dtype=np.float64),
        where=denominator > 0.0,
    )
    x_e = (inverse_mu - X - Y / 4.0) / X
    return np.clip(np.nan_to_num(x_e, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
