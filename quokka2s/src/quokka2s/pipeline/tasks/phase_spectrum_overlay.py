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

from ..base import AnalysisTask, PipelinePlotContext
from ..utils import (
    PHASE_ORDER, PHASE_LABEL_LINE,
    classify_temperature_phase,
    mass_weighted_sigma, mass_weighted_sigma_by_phase,
)
from .integrated_spectrum import SPECIES_CFG, V_RANGE_KMS


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
        if self.bin_size != 1:
            raise NotImplementedError(
                'PhaseSpectrumOverlayTask: bin_size > 1 not supported; pass bin_size=1.'
            )
        self.R = float(R)

    # Histogram binning controls. Pre-bin in compute() so the task intermediate
    # stores ~6×N_BINS doubles per LOS instead of the raw (v[mask], rho[mask])
    # cubes (which are gigabyte-scale when WIM dominates).
    HIST_N_BINS = 120

    def compute(self, context: PipelinePlotContext) -> dict:
        p = context.provider
        vx_u,  _ = p.get_slab_z(('gas', 'velocity_x'))
        vy_u,  _ = p.get_slab_z(('gas', 'velocity_y'))
        rho_u, _ = p.get_slab_z(('gas', 'density'))
        T_u,   _ = p.get_slab_z(('gas', 'temperature_despotic'))
        vx  = vx_u.in_units('km/s').value.ravel()
        vy  = vy_u.in_units('km/s').value.ravel()
        rho = rho_u.value.ravel()
        T   = T_u.in_units('K').value.ravel()
        del vx_u, vy_u, rho_u, T_u

        masks   = classify_temperature_phase(T)
        phase_x = mass_weighted_sigma_by_phase(vx, rho, T)
        phase_y = mass_weighted_sigma_by_phase(vy, rho, T)
        v_mean_total_x, sigma_total_x = mass_weighted_sigma(vx, rho)
        v_mean_total_y, sigma_total_y = mass_weighted_sigma(vy, rho)

        # Pre-binned mass-weighted velocity histograms.  Range matches the
        # plot's x-axis so we don't need to keep the raw v/rho cubes.
        x_window = (-V_RANGE_KMS, V_RANGE_KMS)
        bin_edges = np.linspace(x_window[0], x_window[1], self.HIST_N_BINS + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        def _hist(v_arr, rho_arr, mask):
            if not mask.any() or rho_arr[mask].sum() <= 0:
                return np.zeros_like(bin_centers)
            counts, _ = np.histogram(v_arr[mask], bins=bin_edges,
                                     weights=rho_arr[mask])
            return counts

        pdf = {'x': {'bin_centers': bin_centers},
               'y': {'bin_centers': bin_centers}}
        for los_name, vel_arr, phase_dict, sig_tot, vmean_tot in (
            ('x', vx, phase_x, sigma_total_x, v_mean_total_x),
            ('y', vy, phase_y, sigma_total_y, v_mean_total_y),
        ):
            for ph in PHASE_ORDER:
                pdf[los_name][ph] = {
                    'counts':    _hist(vel_arr, rho, masks[ph]),
                    'sigma':     phase_dict[ph]['sigma'],
                    'v_mean':    phase_dict[ph]['v_mean'],
                    'mass_frac': phase_dict[ph]['mass_frac'],
                }
            # 'total' uses all cells.
            all_mask = np.ones_like(rho, dtype=bool)
            pdf[los_name]['total'] = {
                'counts':    _hist(vel_arr, rho, all_mask),
                'sigma':     sig_tot,
                'v_mean':    vmean_tot,
                'mass_frac': 1.0,
            }

        # Done with raw 3D cubes used for PDFs — release ~4 GB before the
        # spectrum loop starts allocating its own primitives.
        del vx, vy, rho, T, masks, all_mask

        # Fresh SpectrumStore + fresh yt covering_grid per species.  See
        # IntegratedSpectrumTask.compute for the reasoning.
        import gc
        from ..services import SpectrumStore
        spec = {'x': {}, 'y': {}}
        for sp in SPECIES_CFG:
            name = sp['name']
            store = SpectrumStore(p)
            for los in ('x', 'y'):
                v_axis, dsigma_dv = store.get_spectrum(name, los, R=self.R)
                _, sigma_obs = _moment_sigma(v_axis, dsigma_dv)
                spec[los][name] = {'v_axis':    v_axis,
                                   'spec':      dsigma_dv,
                                   'sigma_obs': sigma_obs,
                                   'color':     sp['color']}
            del store
            p._cached_grid = None
            gc.collect()

        return {'pdf': pdf, 'spec': spec}

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        for los in ('x', 'y'):
            self._plot_one_los(results, los)

    def _plot_one_los(self, results: dict, los: str) -> None:
        pdf  = results['pdf']
        spec = results['spec']
        bin_centers = pdf[los]['bin_centers']

        n_rows = len(SPECIES_CFG)
        n_cols = len(PHASE_PLOT_ORDER)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.75 * n_rows),
                                 sharex='all', sharey='all')

        x_window = (-V_RANGE_KMS, V_RANGE_KMS)

        for r, sp in enumerate(SPECIES_CFG):
            name      = sp['name']
            spec_v    = spec[los][name]['v_axis']
            spec_y    = spec[los][name]['spec']
            sigma_obs = spec[los][name]['sigma_obs']
            spec_peak = spec_y.max()
            spec_norm = spec_y / spec_peak if spec_peak > 0 else spec_y

            for c, phase in enumerate(PHASE_PLOT_ORDER):
                ax = axes[r, c]

                phase_pdf = pdf[los][phase]
                counts    = phase_pdf['counts']
                sigma_p   = phase_pdf['sigma']

                if np.isfinite(sigma_p) and sigma_p > 0:
                    ax.axvspan(-sigma_p, sigma_p,
                               color=PHASE_COLOR[phase], alpha=BAND_ALPHA, zorder=0)
                if np.isfinite(sigma_obs) and sigma_obs > 0:
                    ax.axvspan(-sigma_obs, sigma_obs,
                               color=SPECTRUM_BAND_COLOR, alpha=BAND_ALPHA, zorder=0)

                peak = counts.max() if counts.size else 0.0
                if peak > 0:
                    counts_norm = counts / peak
                    ax.step(bin_centers, counts_norm, where='mid',
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
