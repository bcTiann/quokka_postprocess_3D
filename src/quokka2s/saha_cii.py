"""Legacy two-stage carbon Saha + two-level LTE [C II] diagnostic.

This module preserves the pre-Cloudy high-temperature prescription for model
comparison only.  Its CHIANTI constants are loaded lazily, so the production
Cloudy/DESPOTIC pipeline does not require fiasco or its database at import.
"""
from __future__ import annotations

from functools import lru_cache
import warnings

import numpy as np


SAHA_HIGH_TEMPERATURE_K = 1.307e4
SAHA_DIAGNOSTIC_MIN_TEMPERATURE_K = 3000.0


@lru_cache(maxsize=1)
def _atomic_data():
    """Return the same CHIANTI partition functions/constants as the old code."""
    import astropy.units as u
    from astropy import constants as const
    import fiasco

    temperature_quantity = np.logspace(2.5, 7.5, 600) * u.K
    temperature = temperature_quantity.to_value('K')
    cii = fiasco.Ion('C 2', temperature_quantity)
    ci = fiasco.Ion('C 1', temperature_quantity)
    ciii = fiasco.Ion('C 3', temperature_quantity)

    beta = (1.0 / (const.k_B * temperature_quantity)).to_value('1/erg')

    def partition_function(ion):
        energy = ion.levels.energy.to_value('erg')[:, None]
        weight = np.asarray(ion.levels.weight, dtype=float)[:, None]
        return (weight * np.exp(-energy * beta[None, :])).sum(axis=0)

    U_C0 = partition_function(ci)
    U_Cp = partition_function(cii)
    U_Cpp = partition_function(ciii)

    I_C_eV = float(np.atleast_1d(
        fiasco.Ion('C 1', np.array([8000.0]) * u.K)
        .ionization_potential.to_value(u.eV)
    )[0])
    I_C2_eV = float(np.atleast_1d(
        fiasco.Ion('C 2', np.array([8000.0]) * u.K)
        .ionization_potential.to_value(u.eV)
    )[0])

    lower = cii.transitions.lower_level
    upper = cii.transitions.upper_level
    selected = np.where((lower == 1) & (upper == 2))[0]
    if selected.size == 0:
        raise RuntimeError('CHIANTI C II 158 micron transition was not found')
    transition = int(selected[0])
    A_ul = float(cii.transitions.A[transition].to_value('1/s'))
    photon_energy_erg = float(
        cii.transitions.delta_energy[transition].to_value('erg')
    )
    T_star = float(
        (cii.transitions.delta_energy[transition] / const.k_B).to_value('K')
    )
    g_l = float(cii.levels.weight[0])
    g_u = float(cii.levels.weight[1])
    eV_to_K = float((1.0 * u.eV / const.k_B).to_value('K'))
    saha_prefactor = float(
        (2.0 * np.pi * const.m_e * const.k_B / const.h**2)
        .to_value('cm**-2/K')
    )

    return {
        'temperature': temperature,
        'U_C0': U_C0,
        'U_Cp': U_Cp,
        'U_Cpp': U_Cpp,
        'T_C': I_C_eV * eV_to_K,
        'T_C2': I_C2_eV * eV_to_K,
        'A_ul': A_ul,
        'photon_energy_erg': photon_energy_erg,
        'T_star': T_star,
        'g_l': g_l,
        'g_u': g_u,
        'saha_prefactor': saha_prefactor,
    }


def legacy_saha_lte_emissivity(
    temperature_K,
    hydrogen_density_cm3,
    electron_density_cm3,
    *,
    carbon_abundance_per_H: float,
    minimum_temperature_K: float = SAHA_DIAGNOSTIC_MIN_TEMPERATURE_K,
    evaluation_chunk: int = 4_000_000,
) -> np.ndarray:
    """Evaluate the old C/C+/C++ Saha ion fraction and LTE line emissivity.

    By default, values below 3000 K are zero because this diagnostic preserves
    the old high-temperature branch.  A caller may set
    ``minimum_temperature_K=0`` for an explicitly labelled all-temperature
    diagnostic.  No clipping is applied to electron density, ion fractions,
    or finite negative emissivities, preserving the old result.
    """
    temperature, n_H, n_e = np.broadcast_arrays(
        np.asarray(temperature_K, dtype=float),
        np.asarray(hydrogen_density_cm3, dtype=float),
        np.asarray(electron_density_cm3, dtype=float),
    )
    if evaluation_chunk <= 0:
        raise ValueError('evaluation_chunk must be positive')
    if minimum_temperature_K < 0.0:
        raise ValueError('minimum_temperature_K must be non-negative')

    atomic = _atomic_data()
    output = np.zeros(temperature.shape, dtype=float)
    selected = np.flatnonzero(
        (temperature.ravel() >= minimum_temperature_K)
        & np.isfinite(temperature.ravel())
        & np.isfinite(n_H.ravel())
        & np.isfinite(n_e.ravel())
        & (n_H.ravel() > 0.0)
    )
    flat_T = temperature.ravel()
    flat_nH = n_H.ravel()
    flat_ne = n_e.ravel()
    flat_output = output.ravel()

    for start in range(0, selected.size, evaluation_chunk):
        indices = selected[start:start + evaluation_chunk]
        T = flat_T[indices]
        ne = flat_ne[indices]
        U_C0 = np.interp(T, atomic['temperature'], atomic['U_C0'])
        U_Cp = np.interp(T, atomic['temperature'], atomic['U_Cp'])
        U_Cpp = np.interp(T, atomic['temperature'], atomic['U_Cpp'])

        prefactor = 2.0 * np.power(atomic['saha_prefactor'] * T, 1.5)
        S_C1 = prefactor * (U_Cp / U_C0) * np.exp(-atomic['T_C'] / T)
        S_C2 = prefactor * (U_Cpp / U_Cp) * np.exp(-atomic['T_C2'] / T)

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            r1 = S_C1 / ne
            r2 = S_C2 / ne
            x_Cp_three_state = r1 / (1.0 + r1 + r1 * r2)
            x_Cp_hot = ne / (ne + S_C2)
        x_Cp = np.where(T >= SAHA_HIGH_TEMPERATURE_K, x_Cp_hot, x_Cp_three_state)

        upper_ratio = (atomic['g_u'] / atomic['g_l']) * np.exp(
            -atomic['T_star'] / T
        )
        upper_fraction = upper_ratio / (1.0 + upper_ratio)
        emissivity = (
            x_Cp * carbon_abundance_per_H * flat_nH[indices]
            * upper_fraction * atomic['A_ul'] * atomic['photon_energy_erg']
        )
        nonfinite = ~np.isfinite(emissivity)
        if nonfinite.any():
            raise ValueError(
                'legacy Saha diagnostic produced '
                f'{int(nonfinite.sum())} non-finite high-temperature emissivities'
            )
        negative = emissivity < 0.0
        if negative.any():
            warnings.warn(
                'legacy Saha diagnostic retained '
                f'{int(negative.sum())} finite negative emissivities caused by '
                'the original unclipped electron-density inversion',
                RuntimeWarning,
                stacklevel=2,
            )
        flat_output[indices] = emissivity

    return output
