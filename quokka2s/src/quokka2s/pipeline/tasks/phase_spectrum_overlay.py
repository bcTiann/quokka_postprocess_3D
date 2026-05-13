"""PhaseSpectrumOverlayTask: overlay species line profiles with phase-resolved velocity PDFs.

Compares each species' spatially-integrated line profile (CO, C+, Hα, HI 21cm)
against the density-weighted velocity PDF of the gas, split by the 5 ISM
phases (CNM / UNM / WNM / WIM / HIM) plus a "total" with no phase split.
Observationally only the integrated line profile is accessible — the
phase-resolved PDFs are simulation-only ground truth. The overlay calibrates
which gas phase each species' line profile most closely traces.

Spectrum: emissivity-weighted (∝ species emissivity) with thermal-Doppler +
LSF broadening. PDF: mass-weighted bulk velocity only. Spectrum is therefore
systematically broader than the matching phase PDF — that is physical, not a
bug.

Defaults (`bin_size=1`, `R=inf`) reproduce the maximally ideal observation.
`bin_size` is mathematically a no-op for an integrated spectrum (sum is
associative) and intentionally does not affect the PDFs either (PDFs are not
observables, so an "instrument resolution" on them has no meaning).

Output: one PNG per LOS, n_species × 6 grid (5 phases + total). Both curves
peak-normalised.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from yt.units import m, s, km

from ..base import AnalysisTask, PipelinePlotContext
from ..prep.physics_fields import build_spectral_cube
from ..utils import (
    apply_spatial_bin, apply_spectral_lsf,
    PHASE_ORDER, PHASE_LABEL_LINE,
    classify_temperature_phase,
    mass_weighted_sigma, mass_weighted_sigma_by_phase,
)
from .integrated_spectrum import SPECIES_CFG, N_CHANNELS, V_RANGE_KMS


PHASE_PLOT_ORDER = PHASE_ORDER + ('total',)
PHASE_COLOR = {
    'CNM':   'navy',
    'UNM':   'steelblue',
    'WNM':   'mediumseagreen',
    'WIM':   'goldenrod',
    'HIM':   'crimson',
    'total': 'darkviolet',   # avoid collision with the (black) spectrum line
}
SPECTRUM_COLOR      = 'black'
SPECTRUM_BAND_COLOR = 'dimgray'
BAND_ALPHA          = 0.18


def _moment_sigma(v: np.ndarray, spec: np.ndarray) -> tuple[float, float]:
    """Velocity dispersion via second moment of a 1D spectrum [km/s]."""
    total = spec.sum()
    if total <= 0:
        return float('nan'), float('nan')
    w = spec / total
    v_mean = float(np.sum(v * w))
    sigma  = float(np.sqrt(np.sum((v - v_mean) ** 2 * w)))
    return v_mean, sigma


class PhaseSpectrumOverlayTask(AnalysisTask):
    """Overlay species integrated line profiles with phase-split velocity PDFs."""

    def __init__(self, config, bin_size: int = 1, R: float = np.inf):
        super().__init__(config)
        self.bin_size = int(bin_size)
        self.R = float(R)
        self._vx = self._vy = self._rho = self._T = None
        self._doppler_x = self._doppler_y = None
        self._sp_data: dict[str, dict] = {}
        self._cell_area: dict[str, float] = {}
        self._volume_3d = None

    def prepare(self, context: PipelinePlotContext) -> None:
        p = context.provider
        self._vx,  _ = p.get_slab_z(('gas', 'velocity_x'))
        self._vy,  _ = p.get_slab_z(('gas', 'velocity_y'))
        self._rho, _ = p.get_slab_z(('gas', 'density'))
        self._T,   _ = p.get_slab_z(('gas', 'temperature_despotic'))

        self._doppler_x, _ = p.get_slab_z(('gas', 'Bulk_Doppler_factor_x'))
        self._doppler_y, _ = p.get_slab_z(('gas', 'Bulk_Doppler_factor_y'))

        dx, _ = p.get_slab_z(('boxlib', 'dx'))
        dy, _ = p.get_slab_z(('boxlib', 'dy'))
        dz, _ = p.get_slab_z(('boxlib', 'dz'))
        self._volume_3d = dx * dy * dz
        self._cell_area['yz'] = (dy * dz)[0, 0, 0].in_units('cm**2').value
        self._cell_area['xz'] = (dx * dz)[0, 0, 0].in_units('cm**2').value

        for sp in SPECIES_CFG:
            name = sp['name']
            freq,  _ = p.get_slab_z(('gas', sp['freq_field']))
            lum,   _ = p.get_slab_z(('gas', sp['lum_field']))
            width, _ = p.get_slab_z(('gas', sp['width_field']))
            self._sp_data[name] = {'freq': freq, 'lum': lum, 'width': width}

    def _build_spectrum(self, sp_name: str, doppler, plane_key: str
                        ) -> tuple[np.ndarray, np.ndarray]:
        """Return (dsigma_dv [erg/s/Hz/cm^2], v_axis [km/s])."""
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

        if self.bin_size > 1:
            cube = apply_spatial_bin(cube, self.bin_size)

        _, n1, n2      = cube.shape
        total_lum      = cube.sum(axis=(1, 2))
        total_area_cm2 = n1 * n2 * self._cell_area[plane_key]
        dsigma_dv      = total_lum / total_area_cm2

        v_axis = (c * (nu_0 - freq_ctr) / nu_0).in_units('km/s').value
        dv     = abs(float(v_axis[1] - v_axis[0]))

        if np.isfinite(self.R) and self.R > 0:
            dsigma_dv = apply_spectral_lsf(dsigma_dv, dv, self.R, axis=0)

        return dsigma_dv, v_axis

    def compute(self, context: PipelinePlotContext) -> dict:
        vx  = self._vx.in_units('km/s').value.ravel()
        vy  = self._vy.in_units('km/s').value.ravel()
        rho = self._rho.value.ravel()
        T   = self._T.in_units('K').value.ravel()

        masks   = classify_temperature_phase(T)
        phase_x = mass_weighted_sigma_by_phase(vx, rho, T)
        phase_y = mass_weighted_sigma_by_phase(vy, rho, T)
        v_mean_total_x, sigma_total_x = mass_weighted_sigma(vx, rho)
        v_mean_total_y, sigma_total_y = mass_weighted_sigma(vy, rho)

        def _bundle(v_arr, rho_arr, mask, sigma, v_mean, mass_frac):
            return {'v': v_arr[mask], 'rho': rho_arr[mask],
                    'sigma': sigma, 'v_mean': v_mean, 'mass_frac': mass_frac}

        pdf = {'x': {}, 'y': {}}
        for p in PHASE_ORDER:
            pdf['x'][p] = _bundle(vx, rho, masks[p],
                                  phase_x[p]['sigma'], phase_x[p]['v_mean'],
                                  phase_x[p]['mass_frac'])
            pdf['y'][p] = _bundle(vy, rho, masks[p],
                                  phase_y[p]['sigma'], phase_y[p]['v_mean'],
                                  phase_y[p]['mass_frac'])
        pdf['x']['total'] = {'v': vx, 'rho': rho,
                             'sigma': sigma_total_x, 'v_mean': v_mean_total_x,
                             'mass_frac': 1.0}
        pdf['y']['total'] = {'v': vy, 'rho': rho,
                             'sigma': sigma_total_y, 'v_mean': v_mean_total_y,
                             'mass_frac': 1.0}

        spec = {'x': {}, 'y': {}}
        for sp in SPECIES_CFG:
            name = sp['name']
            print(f'  [{name}] LOS=x: building spectrum '
                  f'(bin_size={self.bin_size}, R={self.R}) ...')
            d_x, v_x = self._build_spectrum(name, self._doppler_x, 'yz')
            _, sigma_obs_x = _moment_sigma(v_x, d_x)
            spec['x'][name] = {'v_axis': v_x, 'spec': d_x,
                               'sigma_obs': sigma_obs_x, 'color': sp['color']}

            print(f'  [{name}] LOS=y: building spectrum ...')
            d_y, v_y = self._build_spectrum(name, self._doppler_y, 'xz')
            _, sigma_obs_y = _moment_sigma(v_y, d_y)
            spec['y'][name] = {'v_axis': v_y, 'spec': d_y,
                               'sigma_obs': sigma_obs_y, 'color': sp['color']}

        return {'pdf': pdf, 'spec': spec}

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        for los in ('x', 'y'):
            self._plot_one_los(results, los)

    def _plot_one_los(self, results: dict, los: str) -> None:
        pdf  = results['pdf']
        spec = results['spec']

        n_rows = len(SPECIES_CFG)
        n_cols = len(PHASE_PLOT_ORDER)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.75 * n_rows),
                                 sharex='all', sharey='all')

        x_window = (-V_RANGE_KMS, V_RANGE_KMS)
        n_bins   = 120

        for r, sp in enumerate(SPECIES_CFG):
            name      = sp['name']
            spec_v    = spec[los][name]['v_axis']
            spec_y    = spec[los][name]['spec']
            sigma_obs = spec[los][name]['sigma_obs']
            spec_peak = spec_y.max()
            spec_norm = spec_y / spec_peak if spec_peak > 0 else spec_y

            for c, phase in enumerate(PHASE_PLOT_ORDER):
                ax = axes[r, c]

                p_v   = pdf[los][phase]['v']
                p_rho = pdf[los][phase]['rho']
                sigma_p = pdf[los][phase]['sigma']

                if np.isfinite(sigma_p) and sigma_p > 0:
                    ax.axvspan(-sigma_p, sigma_p,
                               color=PHASE_COLOR[phase], alpha=BAND_ALPHA, zorder=0)
                if np.isfinite(sigma_obs) and sigma_obs > 0:
                    ax.axvspan(-sigma_obs, sigma_obs,
                               color=SPECTRUM_BAND_COLOR, alpha=BAND_ALPHA, zorder=0)

                if p_v.size > 0 and p_rho.sum() > 0:
                    counts, edges = np.histogram(p_v, bins=n_bins,
                                                 range=x_window,
                                                 weights=p_rho)
                    peak = counts.max()
                    if peak > 0:
                        counts_norm = counts / peak
                        centers = 0.5 * (edges[:-1] + edges[1:])
                        ax.step(centers, counts_norm, where='mid',
                                color=PHASE_COLOR[phase], lw=1.4,
                                label=f'{phase} PDF')

                ax.step(spec_v, spec_norm, where='mid',
                        color=SPECTRUM_COLOR, lw=1.4, alpha=0.95,
                        label=name)

                ax.axvline(0, color='gray', ls=':', lw=0.8, alpha=0.6)
                ax.set_xlim(*x_window)
                ax.set_ylim(-0.05, 1.15)
                ax.grid(True, alpha=0.25, ls='--', lw=0.5)

                sigma_p_str = f'{sigma_p:.1f} km/s' if np.isfinite(sigma_p) else 'nan'
                sigma_o_str = f'{sigma_obs:.1f} km/s' if np.isfinite(sigma_obs) else 'nan'
                mf = pdf[los][phase].get('mass_frac', float('nan'))
                if phase == 'total' or not np.isfinite(mf):
                    mf_str = ''
                else:
                    mf_str = f'  (m_frac={mf*100:.1f}%)'
                ax.set_title(f'{name} vs {phase}{mf_str}\n'
                             f'σ_{phase}={sigma_p_str}, σ_{name}={sigma_o_str}',
                             fontsize=10)

                if r == n_rows - 1:
                    ax.set_xlabel(f'v_{los}  [km/s]', fontsize=10)
                if c == 0:
                    ax.set_ylabel('Normalised\nintensity / PDF', fontsize=10)

        R_tag = 'inf' if not np.isfinite(self.R) else f'{int(self.R)}'
        fig.suptitle(
            f'Phase × species: spectrum (black) overlaid on phase velocity PDF  '
            f'(LOS={los}, R={R_tag})\n' + PHASE_LABEL_LINE,
            fontsize=11,
        )
        plt.tight_layout()

        out = self.config.output_dir / (
            f'PhaseSpectrumOverlay_los{los}_bin{self.bin_size}_R{R_tag}.png'
        )
        plt.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
