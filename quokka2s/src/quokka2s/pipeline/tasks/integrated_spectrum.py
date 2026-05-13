"""Spatially integrated spectrum dΣ/dv for CO, C+, H-alpha, HI.

Two projection directions:
  - y-z plane: project along x, use Bulk_Doppler_factor_x, area = dy*dz
  - x-z plane: project along y, use Bulk_Doppler_factor_y, area = dx*dz

Both dΣ/dv curves are plotted on the same axes for each species.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from yt.units import m, s, km
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from ..prep.physics_fields import build_spectral_cube
from ..prep import config as _cfg
from ...utils.axes import axis_index
from ..base import AnalysisTask, PipelinePlotContext
from ..utils import make_axis_labels, apply_spectral_lsf


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
        self._doppler_x   = None
        self._doppler_y   = None
        self._volume_3d   = None
        self._cell_area   = {}   # {'yz': float, 'xz': float}
        self._sp_data: dict[str, dict] = {}
        self._vel_x: np.ndarray | None = None
        self._vel_y: np.ndarray | None = None
        self._rho:   np.ndarray | None = None

    def prepare(self, context: PipelinePlotContext) -> None:
        provider = context.provider
        self._doppler_x, _ = provider.get_slab_z(('gas', 'Bulk_Doppler_factor_x'))
        self._doppler_y, _ = provider.get_slab_z(('gas', 'Bulk_Doppler_factor_y'))
        dx, _ = provider.get_slab_z(('boxlib', 'dx'))
        dy, _ = provider.get_slab_z(('boxlib', 'dy'))
        dz, _ = provider.get_slab_z(('boxlib', 'dz'))
        self._volume_3d = dx * dy * dz
        self._cell_area['yz'] = (dy * dz)[0, 0, 0].in_units('cm**2').value
        self._cell_area['xz'] = (dx * dz)[0, 0, 0].in_units('cm**2').value

        vel_x, _ = provider.get_slab_z(('gas', 'velocity_x'))
        vel_y, _ = provider.get_slab_z(('gas', 'velocity_y'))
        rho,   _ = provider.get_slab_z(('gas', 'density'))
        self._vel_x = vel_x.in_units('km/s').value
        self._vel_y = vel_y.in_units('km/s').value
        self._rho   = rho.value

        for sp in SPECIES_CFG:
            name = sp['name']
            freq,  _ = provider.get_slab_z(('gas', sp['freq_field']))
            lum,   _ = provider.get_slab_z(('gas', sp['lum_field']))
            width, _ = provider.get_slab_z(('gas', sp['width_field']))
            self._sp_data[name] = {'freq': freq, 'lum': lum, 'width': width}

    def _build_cube(self, sp_name: str, doppler) -> tuple[np.ndarray, np.ndarray]:
        """Return (spec_cube [n_chan, N1, N2], v_axis [km/s])."""
        c       = 3.0e8 * m / s
        v_range = V_RANGE_KMS * km / s

        freq_3d = self._sp_data[sp_name]['freq'].in_units('Hz')
        nu_0    = freq_3d[0, 0, 0]
        lum_3d  = (self._sp_data[sp_name]['lum'] * self._volume_3d).in_units('erg/s')
        therm   = self._sp_data[sp_name]['width'].in_units('cm/s')
        shifted = (freq_3d * doppler).in_units('Hz')

        bw_hz      = nu_0 * (v_range / c) * 2.0
        freq_edges = np.linspace(nu_0 - bw_hz / 2, nu_0 + bw_hz / 2, N_CHANNELS + 1)
        freq_ctr   = 0.5 * (freq_edges[:-1] + freq_edges[1:])

        cube = build_spectral_cube(
            shifted.in_units('Hz').value,
            lum_3d.in_units('erg/s').value,
            therm.in_units('cm/s').value,
            freq_edges.in_units('Hz').value,
            c.in_units('cm/s').value,
        )

        v_axis = (c * (nu_0 - freq_ctr) / nu_0).in_units('km/s').value
        return cube, v_axis

    def _compute_one(self, sp_name: str, proj_key: str, doppler) -> tuple[str, str, dict]:
        """Build cube for one (species, projection) pair. Returns (sp_name, proj_key, data)."""
        print(f'  [{sp_name}] {proj_key} building cube ...')
        cube, v_axis = self._build_cube(sp_name, doppler)
        _, n1, n2    = cube.shape

        total_lum      = cube.sum(axis=(1, 2))
        total_area_cm2 = n1 * n2 * self._cell_area[proj_key]
        dsigma_dv      = total_lum / total_area_cm2

        dv_per_channel = abs(v_axis[1] - v_axis[0])
        dsigma_dv_obs  = apply_spectral_lsf(dsigma_dv, dv_per_channel, self.R, axis=0)
        print(f'  [{sp_name}] {proj_key} peak intrinsic={dsigma_dv.max():.3e}  observed={dsigma_dv_obs.max():.3e} erg/s/Hz/cm²')
        return sp_name, proj_key, {'v_axis': v_axis, 'dsigma_dv': dsigma_dv, 'dsigma_dv_obs': dsigma_dv_obs}

    def compute(self, context: PipelinePlotContext) -> dict:
        print('IntegratedSpectrumTask: building spectral cubes (6 tasks in parallel) ...')
        sigma_x = _mass_weighted_sigma(self._vel_x, self._rho)
        sigma_y = _mass_weighted_sigma(self._vel_y, self._rho)
        print(f'  σ_gas: x={sigma_x:.2f} km/s, y={sigma_y:.2f} km/s')

        proj_doppler = [('yz', self._doppler_x), ('xz', self._doppler_y)]
        out = {'yz': {}, 'xz': {}, 'sigma_v_data': {'yz': sigma_x, 'xz': sigma_y}}

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [
                pool.submit(self._compute_one, sp['name'], proj_key, doppler)
                for sp in SPECIES_CFG
                for proj_key, doppler in proj_doppler
            ]
            for fut in futures:
                sp_name, proj_key, data = fut.result()
                out[proj_key][sp_name] = data

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
