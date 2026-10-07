"""Prepare saved spectrum and gas-phase arrays before any figure is drawn."""

from __future__ import annotations

import numpy as np


def peak_normalize_profiles(profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divide each profile by its peak across all saved velocity channels.

    Parameters
    ----------
    profiles : numpy.ndarray, shape (..., V)
        Nonnegative spectrum or mass values; the last axis contains channels.

    Returns
    -------
    normalized_profiles : numpy.ndarray, shape (..., V)
        Dimensionless profiles. A profile without positive values stays zero.
    has_positive_peak : numpy.ndarray, shape (...,)
        True for profiles with a positive peak in the saved velocity window.

    Examples
    --------
    Rows [0, 2, 1] and [0, 0, 0] become [0, 1, 0.5] and [0, 0, 0].
    """
    peak_values = np.max(profiles, axis=-1)
    has_positive_peak = peak_values > 0.0
    normalized_profiles = np.divide(
        profiles,
        peak_values[..., np.newaxis],
        out=np.zeros_like(profiles, dtype=float),
        where=has_positive_peak[..., np.newaxis],
    )
    return normalized_profiles, has_positive_peak


def add_spectral_display_fields(payload: dict[str, np.ndarray]) -> None:
    """Add surface-luminosity and peak-normalized spectra to a saved payload.

    Parameters
    ----------
    payload : dict of numpy.ndarray
        From IntegratedSpectra.build_output(). Branch dL/dv has shape
        (2, L, 2, V), and cold+hot total dL/dv has shape (2, L, V)
        [erg/s/(km/s)]. projected_area_cm2 is the selected x-y area [cm^2].

    Returns
    -------
    None
        Updates payload without changing its raw arrays. dSigmaL/dv arrays
        keep the input shapes [erg/s/cm^2/(km/s)]; normalized profiles keep
        those shapes without units. Positive-light flags omit the channel axis.

    Examples
    --------
    peak_normalized_profile[1, 0, 1] is the attenuated hot profile of line 0.
    """
    branch_profiles = payload["dL_dv_erg_s_per_kms"]
    total_profiles = payload["total_dL_dv_erg_s_per_kms"]
    projected_area_cm2 = payload["projected_area_cm2"]

    payload["dSigmaL_dv_erg_s_cm2_per_kms"] = branch_profiles / projected_area_cm2
    payload["total_dSigmaL_dv_erg_s_cm2_per_kms"] = total_profiles / projected_area_cm2

    branch_normalized, branch_has_light = peak_normalize_profiles(
        profiles=branch_profiles,
    )
    payload["peak_normalized_profile"] = branch_normalized
    payload["profile_has_light"] = branch_has_light

    total_normalized, total_has_light = peak_normalize_profiles(
        profiles=total_profiles,
    )
    payload["total_peak_normalized_profile"] = total_normalized
    payload["total_profile_has_light"] = total_has_light


def add_gas_phase_display_fields(payload: dict[str, np.ndarray]) -> None:
    """Add velocity centres, normalized gas profiles and comparison sigmas.

    Parameters
    ----------
    payload : dict of numpy.ndarray
        From GasPhaseVelocityAccumulator.build_output(). histogram_mass_g
        has shape (P, V) [g], velocity_edges_kms has shape (V + 1,) [km/s],
        and phase_keys and both full-range sigma arrays have shape (P,).

    Returns
    -------
    None
        Updates payload with velocity_kms (V,), peak_normalized_mass_profile
        (P, V), profile_has_mass (P,), and comparison_sigma_kms (P,).
        Individual phases use sigma about the common gas mean; total uses
        its internal sigma. Existing histogram and sigma arrays are unchanged.

    Examples
    --------
    For the saved order CNM, UNM, WNM, WIM, HIM, total, the last comparison
    sigma is copied from sigma_internal_kms; the first five keep their reference.
    """
    velocity_edges_kms = payload["velocity_edges_kms"]
    payload["velocity_kms"] = 0.5 * (
        velocity_edges_kms[:-1] + velocity_edges_kms[1:]
    )

    normalized_mass, profile_has_mass = peak_normalize_profiles(
        profiles=payload["histogram_mass_g"],
    )
    payload["peak_normalized_mass_profile"] = normalized_mass
    payload["profile_has_mass"] = profile_has_mass

    comparison_sigma_kms = payload["sigma_about_global_mean_kms"].copy()
    total_phase = payload["phase_keys"] == "total"
    comparison_sigma_kms[total_phase] = payload["sigma_internal_kms"][total_phase]
    payload["comparison_sigma_kms"] = comparison_sigma_kms
