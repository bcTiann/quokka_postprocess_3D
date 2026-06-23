"""VelocityPhaseTask (Group 1): pure sim-side velocity + phase analysis.

Reads vx, vy, rho, T_DSP, computes:
  - phase masks (CNM/UNM/WNM/WIM/HIM) from T_DSP
  - mass-weighted σ_v per phase per LOS (x, y)
  - velocity PDFs per phase per LOS — TWO formats:
      a)  auto-bin (per-phase percentile-bracketed range, 120 bins) for the
          PhaseSigmaV_hist plot (per-panel auto-scaling)
      b)  fixed-range (±V_RANGE_KMS, 120 bins) for the PhaseSpectrumOverlay
          plot (spectrum overlay needs consistent axis with the species
          spectra)

Stores both in the task intermediate so downstream plot-only tasks
(PhaseSpectrumOverlay) can load them without recomputing.

Outputs (PNG):
  - PhaseSigmaV_bar.png
  - PhaseSigmaV_hist.png

Replaces the old `PhaseSigmaVTask` in this refactor.  No species spectra
computed here — those go in `SpeciesSpectrumTask`.
"""
from __future__ import annotations

import gc

import numpy as np
import matplotlib.pyplot as plt

from ..base import AnalysisTask, PipelinePlotContext
from ..utils import (
    T_CNM_MAX, T_UNM_MAX, T_WNM_MAX, T_WIM_MAX, PHASE_ORDER,
    PHASE_LABEL_LINE,
    classify_temperature_phase,
    mass_weighted_sigma, mass_weighted_sigma_by_phase,
)

# Fixed-range PDF bin window (matches spectrum's V_RANGE_KMS).
V_RANGE_KMS_FIXED = 50.0

PHASE_COLOR = {
    'CNM': 'navy',
    'UNM': 'steelblue',
    'WNM': 'mediumseagreen',
    'WIM': 'goldenrod',
    'HIM': 'crimson',
}
PHASE_LABEL = {
    'CNM': f'CNM (T < {T_CNM_MAX:.0e})',
    'UNM': f'UNM ({T_CNM_MAX:.0e}–{T_UNM_MAX:.0e})',
    'WNM': f'WNM ({T_UNM_MAX:.0e}–{T_WNM_MAX:.0e})',
    'WIM': f'WIM ({T_WNM_MAX:.0e}–{T_WIM_MAX:.1e})',
    'HIM': f'HIM (T ≥ {T_WIM_MAX:.1e})',
}


class VelocityPhaseTask(AnalysisTask):
    """Phase-split velocity dispersion + PDFs (Group 1 of velocity refactor)."""

    HIST_N_BINS_AUTO  = 120
    HIST_PCT_LO       = 0.5    # auto-bin percentile bracket
    HIST_PCT_HI       = 99.5
    HIST_N_BINS_FIXED = 120
    FIXED_RANGE_KMS   = V_RANGE_KMS_FIXED

    def __init__(self, config):
        super().__init__(config)

    # ── compute ──────────────────────────────────────────────────────────
    def compute(self, context: PipelinePlotContext) -> dict:
        p = context.provider
        vx_u,  _ = p.get_slab_z(('gas', 'velocity_x'))
        vy_u,  _ = p.get_slab_z(('gas', 'velocity_y'))
        rho_u, _ = p.get_slab_z(('gas', 'density'))
        # T_two_regime so phase masks match SpectrumStore's (2026-06-18).
        T_u,   _ = p.get_slab_z(('gas', 'temperature_two_regime'))
        vx  = vx_u.in_units('km/s').value.ravel()
        vy  = vy_u.in_units('km/s').value.ravel()
        rho = rho_u.value.ravel()
        T   = T_u.in_units('K').value.ravel()
        del vx_u, vy_u, rho_u, T_u

        # Per-phase σ_v stats (mass-weighted) for both LOS.
        phase_x = mass_weighted_sigma_by_phase(vx, rho, T)
        phase_y = mass_weighted_sigma_by_phase(vy, rho, T)
        # 'total' stats (all cells, mass-weighted) for both LOS — needed by
        # PhaseSpectrumOverlay's "total" column.
        v_mean_total_x, sigma_total_x = mass_weighted_sigma(vx, rho)
        v_mean_total_y, sigma_total_y = mass_weighted_sigma(vy, rho)
        total_x = {'sigma': sigma_total_x, 'v_mean': v_mean_total_x, 'mass_frac': 1.0}
        total_y = {'sigma': sigma_total_y, 'v_mean': v_mean_total_y, 'mass_frac': 1.0}

        print('\n=== Phase-split density-weighted σ_v ===')
        print(f'{"phase":<6} {"cell_frac":>10} {"mass_frac":>10} '
              f'{"σ_v(x) km/s":>14} {"σ_v(y) km/s":>14}')
        for ph in PHASE_ORDER:
            px, py = phase_x[ph], phase_y[ph]
            print(f'{ph:<6} {px["cell_frac"]:>10.3f} {px["mass_frac"]:>10.3f} '
                  f'{px["sigma"]:>14.2f} {py["sigma"]:>14.2f}')
        print(f'{"total":<6} {"1.000":>10} {"1.000":>10} '
              f'{sigma_total_x:>14.2f} {sigma_total_y:>14.2f}')
        print('=====================================\n')

        masks = classify_temperature_phase(T)

        # ── Auto-bin PDFs (per-phase percentile bracket) ─────────────────
        # Used by PhaseSigmaV_hist.png (per-panel adaptive range).
        hist_auto = {}
        for vel_arr, los in ((vx, 'x'), (vy, 'y')):
            hist_auto[los] = {}
            for ph in PHASE_ORDER:
                m = masks[ph]
                if not m.any():
                    hist_auto[los][ph] = {
                        'bin_centers': np.zeros(0),
                        'pdf':         np.zeros(0),
                    }
                    continue
                v = vel_arr[m]
                lo, hi = np.percentile(v, [self.HIST_PCT_LO, self.HIST_PCT_HI])
                bin_edges = np.linspace(lo, hi, self.HIST_N_BINS_AUTO + 1)
                counts, _ = np.histogram(v, bins=bin_edges,
                                         weights=rho[m], density=True)
                hist_auto[los][ph] = {
                    'bin_centers': 0.5 * (bin_edges[:-1] + bin_edges[1:]),
                    'pdf':         counts,
                }

        # ── Fixed-range PDFs (±V_RANGE_KMS, 120 bins) ───────────────────
        # Used by PhaseSpectrumOverlay_*.png (must align with spectrum
        # v-axis).  Stored as unnormalised mass-weighted counts (not
        # density=True) so the plot can peak-normalise per panel.
        fixed_edges = np.linspace(-self.FIXED_RANGE_KMS, self.FIXED_RANGE_KMS,
                                  self.HIST_N_BINS_FIXED + 1)
        fixed_centers = 0.5 * (fixed_edges[:-1] + fixed_edges[1:])

        def _hist_fixed(v_arr, mask):
            if not mask.any() or rho[mask].sum() <= 0:
                return np.zeros_like(fixed_centers)
            counts, _ = np.histogram(v_arr[mask], bins=fixed_edges,
                                     weights=rho[mask])
            return counts

        hist_fixed = {
            'x': {'bin_centers': fixed_centers},
            'y': {'bin_centers': fixed_centers},
        }
        for vel_arr, los, phase_dict, total in (
            (vx, 'x', phase_x, total_x),
            (vy, 'y', phase_y, total_y),
        ):
            for ph in PHASE_ORDER:
                hist_fixed[los][ph] = {
                    'counts':    _hist_fixed(vel_arr, masks[ph]),
                    'sigma':     phase_dict[ph]['sigma'],
                    'v_mean':    phase_dict[ph]['v_mean'],
                    'mass_frac': phase_dict[ph]['mass_frac'],
                }
            all_mask = np.ones_like(rho, dtype=bool)
            hist_fixed[los]['total'] = {
                'counts':    _hist_fixed(vel_arr, all_mask),
                'sigma':     total['sigma'],
                'v_mean':    total['v_mean'],
                'mass_frac': total['mass_frac'],
            }

        result = {
            'phase_x':    phase_x,
            'phase_y':    phase_y,
            'total_x':    total_x,
            'total_y':    total_y,
            'histograms': hist_auto,      # auto-bin → PhaseSigmaV_hist
            'pdf_fixed':  hist_fixed,     # ±V_RANGE_KMS → PhaseSpectrumOverlay
        }
        # Free the ~14 GB of raw cell arrays + the provider's in-RAM covering
        # grid before returning, so the next task (SpeciesSpectrumTask) starts
        # clean on the 16 GB Mac.  The returned dict holds only small histogram
        # summaries.  (SpeciesSpectrumTask used to evict this leak for us — see
        # its compute() preamble; that crutch is no longer required.)
        del vx, vy, rho, T, masks
        p._cached_grid = None
        gc.collect()
        return result

    # ── plot ─────────────────────────────────────────────────────────────
    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        self._plot_bar(results)
        self._plot_hist(results)

    def _plot_bar(self, results: dict) -> None:
        phases = list(PHASE_ORDER)
        x      = np.arange(len(phases))
        width  = 0.38
        sig_x  = [results['phase_x'][p]['sigma'] for p in phases]
        sig_y  = [results['phase_y'][p]['sigma'] for p in phases]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(x - width / 2, sig_x, width, label='LOS = x', color='steelblue')
        ax.bar(x + width / 2, sig_y, width, label='LOS = y', color='darkorange')
        for i in range(len(phases)):
            for off, val in [(-width / 2, sig_x[i]), (width / 2, sig_y[i])]:
                if np.isfinite(val):
                    ax.text(i + off, val, f'{val:.1f}',
                            ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([PHASE_LABEL[p] for p in phases], fontsize=9)
        ax.set_ylabel('Density-weighted σ_v [km/s]', fontsize=11)
        ax.set_title('Density-weighted σ_v by temperature phase', fontsize=12)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3, ls='--')
        plt.tight_layout()
        out = self.config.output_dir / 'PhaseSigmaV_bar.png'
        plt.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')

    def _plot_hist(self, results: dict) -> None:
        histograms = results['histograms']
        n_phases   = len(PHASE_ORDER)
        fig, axes  = plt.subplots(n_phases, 2, figsize=(13, 3.2 * n_phases))
        for col, (los, phase_dict) in enumerate([
            ('x', results['phase_x']),
            ('y', results['phase_y']),
        ]):
            for row, p in enumerate(PHASE_ORDER):
                ax    = axes[row, col]
                hist  = histograms[los][p]
                centers = hist['bin_centers']
                pdf     = hist['pdf']
                if centers.size == 0:
                    ax.text(0.5, 0.5, f'{p}: no cells',
                            transform=ax.transAxes, ha='center', va='center')
                    continue
                v_mean = phase_dict[p]['v_mean']
                sig    = phase_dict[p]['sigma']
                mf     = phase_dict[p]['mass_frac']
                ax.step(centers, pdf, where='mid',
                        lw=1.6, color=PHASE_COLOR[p])
                ax.axvline(v_mean,       color='gray',          ls=':',  lw=0.8, alpha=0.5)
                ax.axvline(v_mean - sig, color=PHASE_COLOR[p],  ls='--', lw=0.9, alpha=0.6)
                ax.axvline(v_mean + sig, color=PHASE_COLOR[p],  ls='--', lw=0.9, alpha=0.6)
                ax.set_title(
                    f'{p}  (LOS={los})   σ={sig:.1f} km/s,  '
                    f'⟨v⟩={v_mean:.1f} km/s,  m_frac={mf:.3g}',
                    fontsize=10, color=PHASE_COLOR[p],
                )
                if row == n_phases - 1:
                    ax.set_xlabel(f'v_{los}  [km/s]', fontsize=11)
                if col == 0:
                    ax.set_ylabel(r'PDF  [(km/s)$^{-1}$]', fontsize=11)
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
                ax.grid(True, alpha=0.25, ls='--')

        fig.suptitle('Density-weighted velocity PDF by phase  '
                     '(each panel normalised independently)\n'
                     + PHASE_LABEL_LINE, fontsize=11)
        plt.tight_layout()
        out = self.config.output_dir / 'PhaseSigmaV_hist.png'
        plt.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
