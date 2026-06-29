from __future__ import annotations

from ..utils.axes import axis_label
from typing import Iterable
import numpy as np
from yt.units.yt_array import YTArray
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter1d

C_KMS = 299_792.458  # speed of light [km/s]


def apply_spectral_lsf(spec: np.ndarray, dv_per_channel_kms: float, R: float,
                       axis: int = 0) -> np.ndarray:
    """Convolve spectrum with a Gaussian instrumental LSF of resolving power R.

    σ_LSF = (c/R) / 2.355  [km/s]  (FWHM = c/R by definition of R = λ/Δλ)
    sigma_channels = σ_LSF / dv_per_channel_kms   (unit conversion for scipy)
    scipy's gaussian_filter1d normalises the kernel so total flux is conserved.
    """
    sigma_kms      = (C_KMS / R) / 2.355
    sigma_channels = sigma_kms / dv_per_channel_kms
    return gaussian_filter1d(spec, sigma=sigma_channels, axis=axis, mode='nearest')


def apply_spatial_bin(cube: np.ndarray, bin_size: int) -> np.ndarray:
    """SUM-bin on spatial axes (1, 2): b×b sim cells → 1 instrument pixel.

    cube: (n_ch, N1, N2).  Returns (n_ch, N1//b, N2//b).
    b = 1 is a no-op.  Trailing rows/columns are trimmed if N not divisible.
    SUM conserves total flux: out.sum(axis=(1,2)) equals trimmed cube sum.
    """
    if bin_size <= 1:
        return cube
    n_ch, n1, n2 = cube.shape
    n1b, n2b = n1 // bin_size, n2 // bin_size
    trimmed = cube[:, :n1b * bin_size, :n2b * bin_size]
    return trimmed.reshape(n_ch, n1b, bin_size, n2b, bin_size).sum(axis=(2, 4))


# 5-phase ISM classification (TIGRESS-style, Draine 2011 Table 1.3 inspired).
# CNM and molecular gas merged into the coldest bin; WIM extends up to the
# Draine coronal threshold so that transient cooling gas (3e4-3e5 K, no stable
# equilibrium) is bundled with WIM rather than HIM.
T_CNM_MAX = 2.0e2          # K
T_UNM_MAX = 3.0e3          # K
T_WNM_MAX = 1.0e4          # K
T_WIM_MAX = 10 ** 5.5      # K  ≈ 3.16e5  (Draine coronal gas threshold)
PHASE_ORDER = ('CNM', 'UNM', 'WNM', 'WIM', 'HIM')


def classify_temperature_phase(T_K: np.ndarray) -> dict[str, np.ndarray]:
    """Disjoint masks for the 5 ISM phases (T-only, TIGRESS-style)."""
    CNM = T_K <  T_CNM_MAX
    UNM = (T_K >= T_CNM_MAX) & (T_K < T_UNM_MAX)
    WNM = (T_K >= T_UNM_MAX) & (T_K < T_WNM_MAX)
    WIM = (T_K >= T_WNM_MAX) & (T_K < T_WIM_MAX)
    HIM =  T_K >= T_WIM_MAX
    return {'CNM': CNM, 'UNM': UNM, 'WNM': WNM, 'WIM': WIM, 'HIM': HIM}


# Single-line summary of the 5-phase thresholds, suitable for figure suptitles.
# Auto-kept in sync with the T_*_MAX constants above.
PHASE_LABEL_LINE = (
    'Phase cuts:  '
    f'CNM < {int(T_CNM_MAX)} K  |  '
    f'UNM {int(T_CNM_MAX)}–{int(T_UNM_MAX):,} K  |  '
    f'WNM {int(T_UNM_MAX):,}–{int(T_WNM_MAX):,} K  |  '
    f'WIM {int(T_WNM_MAX):,}–10$^{{5.5}}$ K  |  '
    f'HIM ≥ 10$^{{5.5}}$ K'
)


def mass_weighted_sigma(vel_kms: np.ndarray, rho: np.ndarray) -> tuple[float, float]:
    """Mass-weighted mean & σ of a velocity component [km/s]. Returns (nan, nan) if empty."""
    total = rho.sum()
    if total <= 0:
        return float('nan'), float('nan')
    w = rho / total
    v_mean = float(np.sum(vel_kms * w))
    sigma  = float(np.sqrt(np.sum((vel_kms - v_mean) ** 2 * w)))
    return v_mean, sigma


def mass_weighted_sigma_3d(vx_kms: np.ndarray, vy_kms: np.ndarray,
                           vz_kms: np.ndarray, rho: np.ndarray
                           ) -> tuple[tuple[float, float, float], float]:
    """Total 3D mass-weighted velocity dispersion (all gas) — the full velocity
    deviation VECTOR RMS'd, each component centered on its OWN global mean:

        σ_3D² = Σ_i w_i [ (vx_i−⟨vx⟩)² + (vy_i−⟨vy⟩)² + (vz_i−⟨vz⟩)² ]
              = σ_x² + σ_y² + σ_z² ,   w_i = ρ_i / Σρ

    This is NOT a per-cell component average (do NOT collapse a cell's three
    components to one scalar first) — every component keeps its own mean and the
    squared deviations add.  Returns ((⟨vx⟩,⟨vy⟩,⟨vz⟩), σ_3D); (nan,…) if empty.
    """
    total = rho.sum()
    if total <= 0:
        return (float('nan'), float('nan'), float('nan')), float('nan')
    w   = rho / total
    vmx = float(np.sum(vx_kms * w))
    vmy = float(np.sum(vy_kms * w))
    vmz = float(np.sum(vz_kms * w))
    var = (np.sum((vx_kms - vmx) ** 2 * w)
           + np.sum((vy_kms - vmy) ** 2 * w)
           + np.sum((vz_kms - vmz) ** 2 * w))
    return (vmx, vmy, vmz), float(np.sqrt(var))


def spaxel_moments_along_axis(weight: np.ndarray,
                              vel: np.ndarray,
                              los_axis: int
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-spaxel weighted mean V and σ along `los_axis`.

    weight, vel: 3D arrays of identical shape (e.g. (nx, ny, nz)).
    Returns 2D arrays (V_map, sigma_map, W_map) on the perpendicular plane.
    V_map and sigma_map are NaN where W_map == 0.
    Uses the numerically stable form σ² = ⟨v²⟩ - ⟨v⟩² and clamps tiny
    negative ULP drift to zero before sqrt.
    """
    W   = weight.sum(axis=los_axis)
    WV  = (weight * vel).sum(axis=los_axis)
    WV2 = (weight * vel * vel).sum(axis=los_axis)
    with np.errstate(invalid='ignore', divide='ignore'):
        V     = WV / W
        var   = WV2 / W - V * V
        sigma = np.sqrt(np.maximum(var, 0.0))
    return V, sigma, W


def mass_weighted_sigma_by_phase(vel_kms: np.ndarray, rho: np.ndarray,
                                  T_K: np.ndarray) -> dict[str, dict[str, float]]:
    """Per-phase σ_v, each measured about the GLOBAL (all-phase) mass-weighted
    mean velocity — NOT each phase's own mean.

        v_global  = Σ_all ρ_i v_i / Σ_all ρ_i                 (every cell, all phases)
        σ_phase²  = Σ_phase w_i (v_i − v_global)²,  w_i = ρ_i / Σ_phase ρ

    So a phase that streams relative to the bulk gets a larger σ — it now includes
    that bulk offset:  σ_phase² = σ_phase,own² + (v_mean_phase − v_global)².
    (Changed 2026-06-29 from per-phase-COM to this global-COM subtraction.)

    ``v_mean`` is still each phase's OWN bulk velocity (so v_mean − v_global is the
    offset).  Returns {phase: {'v_mean', 'sigma', 'mass_frac', 'cell_frac'}}.
    """
    masks       = classify_temperature_phase(T_K)
    total_mass  = rho.sum()
    total_cells = T_K.size
    # Global mass-weighted mean velocity (all phases together) — the common
    # reference frame every phase's σ is now measured about.
    v_global, _ = mass_weighted_sigma(vel_kms, rho)
    out = {}
    for phase, mask in masks.items():
        m_p = rho[mask]
        tot = float(m_p.sum())
        if tot > 0 and np.isfinite(v_global):
            w      = m_p / tot
            v_mean = float(np.sum(vel_kms[mask] * w))                              # phase's own mean
            sigma  = float(np.sqrt(np.sum((vel_kms[mask] - v_global) ** 2 * w)))   # about GLOBAL mean
        else:
            v_mean = sigma = float('nan')
        out[phase] = {
            'v_mean':    v_mean,
            'sigma':     sigma,
            'mass_frac': float(tot / total_mass) if total_mass > 0 else 0.0,
            'cell_frac': float(mask.sum() / total_cells),
        }
    return out


_PLANE_AXES = {
    "x": ("y", "z"),
    "y": ("x", "z"),
    "z": ("x", "y"),
}


def plane_axes(axis: str) -> tuple[str, str]:
    """Return the two axes that span the plotting plane for the given projection axis."""
    canonical = axis_label(axis)
    return _PLANE_AXES[canonical]


def make_axis_labels(axis: str, units: str) -> tuple[str, str]:
    """Produce human-readable axis labels with units."""
    horiz, vert = plane_axes(axis)
    return f"{horiz.upper()} ({units})", f"{vert.upper()} ({units})"



def weighted_percentile(data: np.ndarray,
                        weights: np.ndarray,
                        pct: float) -> float:
    """Weighted percentile in the range [0, 100].

    Weights are renormalized internally so they need not sum to 1.
    Returns NaN if all weights are zero (e.g. species absent in this regime).
    """
    data    = np.asarray(data, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total   = weights.sum()
    if total <= 0:
        return float('nan')
    weights = weights / total
    idx = np.argsort(data)
    cdf = np.cumsum(weights[idx])
    return float(data[idx[np.searchsorted(cdf, pct / 100.0)]])


def shared_lognorm(*arrays: Iterable[YTArray]) -> LogNorm | None:
    """
    Build a LogNorm spanning the finite positive values of all provided arrays.
    Returns None if no positive values exist.
    """

    merged = np.concatenate(arrays)
    vmin = merged.min()
    vmax = merged.max()

    return LogNorm(vmin=vmin, vmax=vmax)
