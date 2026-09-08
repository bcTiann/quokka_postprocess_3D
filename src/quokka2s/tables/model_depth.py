"""Derive the agreed Cloudy model depth from a supplied cell state.

This pure NumPy calculation does not select a table, interpolate DESPOTIC,
alter cell data, or decide whether a candidate table may be adopted.  The
caller supplies paired DESPOTIC temperature and mean molecular weight from
the same lookup, and QUOKKA internal energy density after subtracting kinetic
energy from total energy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelDepthResult:
    """Broadcast-shaped state and lengths, with invalid results left as NaN.

    ``cold_mask`` selects finite, positive QUOKKA temperatures below 3000 K;
    it is a regime label, not a validity check on the remaining cell inputs.
    ``valid`` indicates that every returned physical quantity is finite and
    positive.  Mean molecular weight is dimensionless, in units of the
    supplied hydrogen mass.
    """

    temperature_K: np.ndarray
    mean_molecular_weight: np.ndarray
    jeans_length_cm: np.ndarray
    model_depth_cm: np.ndarray
    valid: np.ndarray
    cold_mask: np.ndarray


def derive_model_depth(
    density_g_cm3,
    temperature_quokka_K,
    internal_energy_density_erg_cm3,
    temperature_despotic_K,
    mean_molecular_weight_despotic,
    *,
    hydrogen_mass_g: float,
    boltzmann_erg_K: float,
    gravitational_cm3_g_s2: float,
    parsec_cm: float,
    gamma: float = 5.0 / 3.0,
) -> ModelDepthResult:
    """Calculate Jeans length and its 100 pc capped Cloudy model depth.

    All five state inputs may be scalars or broadcastable arrays.  Physical
    constants must be finite positive scalars and ``gamma`` must exceed one.
    Supplying constants explicitly lets the caller use the same cgs values
    as the simulation pipeline without importing yt or DESPOTIC here.

    For ``T_QUOKKA < 3000 K``, use the paired DESPOTIC temperature and mu.
    Otherwise use QUOKKA temperature and infer

        mu = rho * k_B * T_QUOKKA / ((gamma - 1) * m_H * u).

    In both regimes the density is the actual supplied cell density, and

        L_J = pi * sqrt(gamma * k_B * T / (G * mu * m_H * rho)),
        L_model = min(L_J, 100 pc).

    The pi factor is outside the square root by the agreed CIAOLoop
    convention.  No missing state is substituted: nonpositive/nonfinite
    relevant input or an unrepresentable result produces NaN physical
    outputs and ``valid=False``.  Unused branch inputs may be invalid.
    An invalid QUOKKA temperature cannot select either branch.
    """
    constants = {
        "hydrogen_mass_g": hydrogen_mass_g,
        "boltzmann_erg_K": boltzmann_erg_K,
        "gravitational_cm3_g_s2": gravitational_cm3_g_s2,
        "parsec_cm": parsec_cm,
        "gamma": gamma,
    }
    for name, value in constants.items():
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive scalar")
    if gamma <= 1.0:
        raise ValueError("gamma must exceed one")
    maximum_depth_cm = 100.0 * parsec_cm
    if not np.isfinite(maximum_depth_cm):
        raise ValueError("100 pc must be representable in cm")

    rho, T_quokka, u, T_despotic, mu_despotic = np.broadcast_arrays(
        *[
            np.asarray(value, dtype=np.float64)
            for value in (
                density_g_cm3,
                temperature_quokka_K,
                internal_energy_density_erg_cm3,
                temperature_despotic_K,
                mean_molecular_weight_despotic,
            )
        ]
    )
    def positive_finite(value):
        return np.isfinite(value) & (value > 0.0)

    quokka_temperature_valid = positive_finite(T_quokka)
    cold = quokka_temperature_valid & (T_quokka < 3000.0)
    hot = quokka_temperature_valid & ~cold
    density_valid = positive_finite(rho)
    cold_valid = (
        cold & density_valid
        & positive_finite(T_despotic) & positive_finite(mu_despotic)
    )
    hot_valid = hot & density_valid & positive_finite(u)

    temperature = np.full(rho.shape, np.nan)
    mu = np.full(rho.shape, np.nan)
    jeans_length = np.full(rho.shape, np.nan)
    model_depth = np.full(rho.shape, np.nan)
    temperature[cold_valid] = T_despotic[cold_valid]
    mu[cold_valid] = mu_despotic[cold_valid]
    temperature[hot_valid] = T_quokka[hot_valid]
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        mu[hot_valid] = (
            rho[hot_valid] * boltzmann_erg_K * T_quokka[hot_valid]
            / ((gamma - 1.0) * hydrogen_mass_g * u[hot_valid])
        )
        state_valid = (cold_valid | hot_valid) & positive_finite(mu)
        jeans_length[state_valid] = np.pi * np.sqrt(
            gamma * boltzmann_erg_K * temperature[state_valid]
            / (
                gravitational_cm3_g_s2 * mu[state_valid]
                * hydrogen_mass_g * rho[state_valid]
            )
        )

    valid = state_valid & positive_finite(jeans_length)
    model_depth[valid] = np.minimum(jeans_length[valid], maximum_depth_cm)
    temperature[~valid] = np.nan
    mu[~valid] = np.nan
    jeans_length[~valid] = np.nan
    return ModelDepthResult(
        temperature, mu, jeans_length, model_depth, valid, cold,
    )
