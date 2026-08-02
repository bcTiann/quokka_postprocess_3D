"""Shared temperature-regime definitions for line-emission physics.

This module deliberately has no yt/fiasco imports so the regime bookkeeping
and mean-molecular-weight inversion can be unit-tested in isolation.
"""

from __future__ import annotations

import numpy as np


CLOUDY_HELIUM_TO_HYDROGEN_NUMBER_RATIO = 0.1
CLOUDY_HELIUM_ATOMIC_WEIGHT = 3.971
HYDROGEN_MASS_FRACTION = 1.0 / (
    1.0
    + CLOUDY_HELIUM_TO_HYDROGEN_NUMBER_RATIO * CLOUDY_HELIUM_ATOMIC_WEIGHT
)
METAL_MASS_FRACTION = 0.02
HELIUM_MASS_FRACTION = 1.0 - HYDROGEN_MASS_FRACTION - METAL_MASS_FRACTION
MEAN_METAL_ATOMIC_WEIGHT = 16.0
QUOKKA_ADIABATIC_INDEX = 5.0 / 3.0

# Species-specific temperatures for the molecular/ionic line products.  Keep
# this mapping in the lightweight module so field construction and phase-plot
# registration cannot silently choose different temperatures.
EMITTER_TEMPERATURE_FIELDS = {
    'CO': 'temperature_despotic',
    'CO21': 'temperature_despotic',
    'C+': 'temperature_quokka',
    # Retained for archived callers; HCO+ is no longer registered by default.
    'HCO+': 'temperature_despotic',
}


def emitter_temperature_field_name(species: str) -> str:
    """Return the sole per-cell temperature field assigned to an emitter."""
    try:
        return EMITTER_TEMPERATURE_FIELDS[species]
    except KeyError as exc:
        raise ValueError(f'No temperature policy defined for emitter {species!r}') from exc


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


def electron_fraction_from_mean_molecular_weight(
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
    """Infer total ``x_e = n_e/n_H`` from QUOKKA's thermodynamic state.

    Implements the methodology equations

        1/mu = (gamma-1) m_H e_int / (rho k_B T)
        x_e  = (1/mu - X - Y/4) / X.

    The baseline particles in this inversion are H and He nuclei.  The adopted
    X/Y/Z composition remains normalized to one, but the metal-nucleus term is
    deliberately omitted to match QUOKKA's working EOS relation.  Every free
    electron adds one particle regardless of which element supplied it. Return
    the direct algebraic result without clipping ``x_e``.
    """
    e_int = np.asarray(internal_energy_density_erg_cm3, dtype=np.float64)
    rho = np.asarray(density_g_cm3, dtype=np.float64)
    temperature = np.asarray(temperature_K, dtype=np.float64)

    inverse_mu = (
        (gamma - 1.0) * hydrogen_mass_g * e_int
        / (rho * boltzmann_erg_K * temperature)
    )
    if X <= 0.0:
        raise ValueError('X must be positive')
    if Y < 0.0:
        raise ValueError('Y must be non-negative')

    x_e = (inverse_mu - X - Y / 4.0) / X
    return x_e


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
    """Backward-compatible name for the mean-molecular-weight inversion.

    New code should use :func:`electron_fraction_from_mean_molecular_weight`.
    This alias returns the same unclipped algebraic result.
    """
    return electron_fraction_from_mean_molecular_weight(
        internal_energy_density_erg_cm3,
        density_g_cm3,
        temperature_K,
        hydrogen_mass_g=hydrogen_mass_g,
        boltzmann_erg_K=boltzmann_erg_K,
        gamma=gamma,
        X=X,
        Y=Y,
    )
