"""Spatially integrated spectrum dΣ/dv for CO, C+, H-alpha, HI.

Two projection directions:
  - y-z plane: project along x, use Bulk_Doppler_factor_x, area = dy*dz
  - x-z plane: project along y, use Bulk_Doppler_factor_y, area = dx*dz

Both dΣ/dv curves are plotted on the same axes for each species.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from ..prep import config as _cfg
from ...utils.axes import axis_index
from ..base import AnalysisTask, PipelinePlotContext
from ..utils import make_axis_labels


def _gaussian(v, A, v0, sigma):
    return A * np.exp(-0.5 * ((v - v0) / sigma) ** 2)


def _fit_gaussian(v: np.ndarray, spec: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Gaussian fit on normalized spectrum.

    Returns (v0_kms, sigma_kms, fit_curve). On failure returns (nan, nan, zeros).
    """
    peak = spec.max()
    if peak <= 0:
        return np.nan, np.nan, np.zeros_like(spec)
    spec_norm = spec / peak
    v0_guess  = v[np.argmax(spec_norm)]
    try:
        popt, _ = curve_fit(
            _gaussian, v, spec_norm,
            p0=[1.0, v0_guess, 5.0],
            bounds=([0, -50, 0.1], [2.0, 50, 100]),
            maxfev=10000,
        )
        fit_curve = _gaussian(v, *popt) * peak
        return float(popt[1]), abs(float(popt[2])), fit_curve
    except Exception:
        return np.nan, np.nan, np.zeros_like(spec)


def _moment_sigma(v: np.ndarray, spec: np.ndarray) -> tuple[float, float]:
    """Velocity dispersion via second moment. Returns (v_mean_kms, sigma_kms)."""
    total = spec.sum()
    if total <= 0:
        return np.nan, np.nan
    w      = spec / total
    v_mean = float(np.sum(v * w))
    sigma  = float(np.sqrt(np.sum((v - v_mean) ** 2 * w)))
    return v_mean, sigma


def _mass_weighted_sigma(vel_kms: np.ndarray, rho: np.ndarray) -> float:
    """Mass-weighted velocity dispersion from 3D cell arrays. Returns σ [km/s]."""
    w      = rho / rho.sum()
    v_mean = float(np.sum(vel_kms * w))
    return float(np.sqrt(np.sum((vel_kms - v_mean) ** 2 * w)))


SPECIES_CFG = [
    {'name': 'CO',      'freq_field': 'CO_freq',      'lum_field': 'CO_luminosity',      'width_field': 'CO_thermal_width',      'color': 'royalblue'},
    {'name': 'C+',      'freq_field': 'C+_freq',      'lum_field': 'C+_luminosity',      'width_field': 'C+_thermal_width',      'color': 'forestgreen'},
    {'name': 'H_alpha', 'freq_field': 'H_alpha_freq', 'lum_field': 'H_alpha_luminosity', 'width_field': 'H_alpha_thermal_width', 'color': 'crimson'},
    {'name': 'HI',      'freq_field': 'HI_freq',      'lum_field': 'HI_luminosity',      'width_field': 'HI_thermal_width',      'color': 'goldenrod'},
]

N_CHANNELS  = 300    # 300 channels sufficient: LSF σ≈127ch at R=1e4 smears all fine structure
V_RANGE_KMS = 50.0   # ±50 km/s window

# (proj_axis, doppler_field, cell_area_axes)
PROJECTIONS = [
    {'label': 'y-z (proj x)', 'doppler_field': 'Bulk_Doppler_factor_x', 'area_axes': ('dy', 'dz'), 'ls': '-'},
    {'label': 'x-z (proj y)', 'doppler_field': 'Bulk_Doppler_factor_y', 'area_axes': ('dx', 'dz'), 'ls': '--'},
]


class IntegratedSpectrumTask(AnalysisTask):
    """Spatially integrated spectrum dΣ/dv for CO, C+, H-alpha, HI, two projections."""

    def __init__(self, config, axis: str | None = None, figure_units: str | None = None,
                 R: float | None = None):
        super().__init__(config)
        self.axis         = axis or 'x'
        self.axis_idx     = axis_index(self.axis)
        self.figure_units = figure_units or config.figure_units
        self.xlabel, self.ylabel = make_axis_labels(self.axis, self.figure_units)
        self.R = R if R is not None else _cfg.SPECTRAL_RESOLUTION_R

    def compute(self, context: PipelinePlotContext) -> dict:
        provider = context.provider
        # σ_gas annotation: needs vx, vy, density. Released when compute returns.
        vel_x_u, _ = provider.get_slab_z(('gas', 'velocity_x'))
        vel_y_u, _ = provider.get_slab_z(('gas', 'velocity_y'))
        rho_u,   _ = provider.get_slab_z(('gas', 'density'))
        vel_x = vel_x_u.in_units('km/s').value
        vel_y = vel_y_u.in_units('km/s').value
        rho   = rho_u.value
        del vel_x_u, vel_y_u, rho_u

        sigma_x = _mass_weighted_sigma(vel_x, rho)
        sigma_y = _mass_weighted_sigma(vel_y, rho)
        print(f'  σ_gas: x={sigma_x:.2f} km/s, y={sigma_y:.2f} km/s')
        del vel_x, vel_y, rho   # 3 GB freed before spectrum builds

        # One species at a time: fresh SpectrumStore + fresh yt covering_grid
        # per species.  Dropping the SpectrumStore alone is not enough — yt
        # caches every field on provider._cached_grid, so without the explicit
        # evict the covering_grid would accumulate ~6 GB of species-specific
        # fields (lum, width, freq, ...) per species. That accumulation pushes
        # macOS into compressed memory + swap at down=1.
        import gc
        from ..services import SpectrumStore
        los_for_proj = {'yz': 'x', 'xz': 'y'}
        out = {'yz': {}, 'xz': {}, 'sigma_v_data': {'yz': sigma_x, 'xz': sigma_y}}

        for sp in SPECIES_CFG:
            name = sp['name']
            store = SpectrumStore(provider)
            for proj_key in ('yz', 'xz'):
                los = los_for_proj[proj_key]
                v_axis, dsigma_dv     = store.get_spectrum(name, los, R=float('inf'))
                _,      dsigma_dv_obs = store.get_spectrum(name, los, R=self.R)
                out[proj_key][name] = {
                    'v_axis':         v_axis,
                    'dsigma_dv':      dsigma_dv,
                    'dsigma_dv_obs':  dsigma_dv_obs,
                }
                print(f'  [{name}] {proj_key} peak intrinsic={dsigma_dv.max():.3e}'
                      f'  observed={dsigma_dv_obs.max():.3e} erg/s/Hz/cm²')
            del store
            provider._cached_grid = None
            gc.collect()

        return out

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        self._plot_individual(results)
        self._plot_overlay(results)

    def _plot_one_proj(self, ax, results: dict, proj_key: str, title: str) -> None:
        """Plot all 3 species for one projection direction onto ax."""
        for sp in SPECIES_CFG:
            name = sp['name']
            v    = results[proj_key][name]['v_axis']
            spec = results[proj_key][name]['dsigma_dv']
            ax.plot(v, spec, color=sp['color'], lw=1.5,
                    drawstyle='steps-mid', label=name)
        ax.axvline(0, color='gray', ls=':', lw=0.8, alpha=0.6)
        ax.set_xlabel('Velocity [km/s]', fontsize=12)
        ax.set_ylabel(r'$\mathrm{d}\Sigma/\mathrm{d}v$ [erg s$^{-1}$ Hz$^{-1}$ cm$^{-2}$]',
                      fontsize=11)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10, framealpha=0.8)
        ax.grid(True, alpha=0.25, ls='--', lw=0.5)

    def _annotate_sigma(self, ax, v, spec, color, sigma_data: float | None = None) -> None:
        """Overlay Gaussian fit + moment method + data σ_v, annotate all three."""
        # Gaussian fit
        _, sigma_gauss, fit_curve = _fit_gaussian(v, spec)
        if np.isfinite(sigma_gauss):
            ax.plot(v, fit_curve, color='black', lw=1.2, ls='--',
                    label=f'Gaussian  σ={sigma_gauss:.1f} km/s')

        # Moment method (spectral)
        _, sigma_mom = _moment_sigma(v, spec)
        if np.isfinite(sigma_mom):
            ax.axvspan(-sigma_mom, sigma_mom, alpha=0.08, color=color,
                       label=f'Moment   σ={sigma_mom:.1f} km/s')

        # Mass-weighted σ_v from simulation data
        if sigma_data is not None and np.isfinite(sigma_data):
            ax.axvline( sigma_data, color='darkorange', lw=1.4, ls='-.',
                       label=f'σ_gas={sigma_data:.1f} km/s')
            ax.axvline(-sigma_data, color='darkorange', lw=1.4, ls='-.')

        ax.legend(fontsize=9, framealpha=0.85)

    def _plot_individual(self, results: dict) -> None:
        """One figure per species: left = x line-of-sight (y-z plane), right = y line-of-sight (x-z plane)."""
        for sp in SPECIES_CFG:
            name = sp['name']
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

            for ax, proj_key, title in [
                (axes[0], 'yz', 'Line-of-sight: x  (y-z plane)'),
                (axes[1], 'xz', 'Line-of-sight: y  (x-z plane)'),
            ]:
                v        = results[proj_key][name]['v_axis']
                spec     = results[proj_key][name]['dsigma_dv']
                spec_obs = results[proj_key][name]['dsigma_dv_obs']
                sigma_data = results['sigma_v_data'][proj_key]
                # intrinsic: thin, faded
                ax.plot(v, spec, color=sp['color'], lw=0.8, alpha=0.35,
                        drawstyle='steps-mid', label='intrinsic')
                # observed: full weight
                ax.plot(v, spec_obs, color=sp['color'], lw=1.5,
                        drawstyle='steps-mid', label=f'observed (R={self.R:.0e})')
                self._annotate_sigma(ax, v, spec_obs, sp['color'], sigma_data=sigma_data)
                ax.axvline(0, color='gray', ls=':', lw=0.8, alpha=0.6)
                ax.set_xlabel('Velocity [km/s]', fontsize=12)
                ax.set_ylabel(r'$\mathrm{d}\Sigma/\mathrm{d}v$ [erg s$^{-1}$ Hz$^{-1}$ cm$^{-2}$]',
                              fontsize=11)
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
                ax.set_title(title, fontsize=12)
                ax.grid(True, alpha=0.25, ls='--', lw=0.5)

            fig.suptitle(f'{name}  Spatially Integrated Spectrum  (R={self.R:.0e})', fontsize=13)
            tag = name.replace('+', 'plus')
            out = self.config.output_dir / f'IntegratedSpectrum_{tag}.png'
            plt.tight_layout()
            plt.savefig(str(out), dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved: {out}')

    def _plot_overlay(self, results: dict) -> None:
        """3 species, left = x line-of-sight, right = y line-of-sight, normalized."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, proj_key, title in [
            (axes[0], 'yz', 'Line-of-sight: x  (y-z plane)'),
            (axes[1], 'xz', 'Line-of-sight: y  (x-z plane)'),
        ]:
            for sp in SPECIES_CFG:
                name     = sp['name']
                v        = results[proj_key][name]['v_axis']
                spec_obs = results[proj_key][name]['dsigma_dv_obs']
                peak = spec_obs.max()
                norm = spec_obs / peak if peak > 0 else spec_obs
                ax.plot(v, norm, color=sp['color'], lw=1.5,
                        drawstyle='steps-mid', label=name)
            ax.axvline(0, color='gray', ls=':', lw=0.8, alpha=0.6)
            ax.set_xlabel('Velocity [km/s]', fontsize=12)
            ax.set_ylabel('Normalized Intensity', fontsize=12)
            ax.set_ylim(-0.05, 1.15)
            ax.legend(fontsize=10, framealpha=0.8)
            ax.set_title(title, fontsize=12)
            ax.grid(True, alpha=0.25, ls='--', lw=0.5)

        species_label = ' / '.join(s['name'] for s in SPECIES_CFG)
        fig.suptitle(f'{species_label}  Spatially Integrated Spectra (normalized, observed R={self.R:.0e})', fontsize=13)
        out = self.config.output_dir / 'IntegratedSpectrum_overlay.png'
        plt.tight_layout()
        plt.savefig(str(out), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
