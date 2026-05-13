"""PhaseSigmaVTask: density-weighted σ_v split by temperature phase.

Splits cells into 5 ISM phases (CNM / UNM / WNM / WIM / HIM) by T alone using
temperature_despotic, then computes density-weighted σ_v for v_x (LOS = x) and
v_y (LOS = y) per phase.

Outputs:
  - Console table: phase × LOS σ_v + mass/cell fractions
  - PhaseSigmaV_bar.png  — grouped bar chart (phase × LOS)
  - PhaseSigmaV_hist.png — density-weighted velocity PDFs, 1 panel per LOS,
                          stepped line per phase, σ annotated in the legend.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..base import AnalysisTask, PipelinePlotContext
from ..utils import (
    T_CNM_MAX, T_UNM_MAX, T_WNM_MAX, T_WIM_MAX, PHASE_ORDER,
    PHASE_LABEL_LINE,
    classify_temperature_phase,
    mass_weighted_sigma_by_phase,
)

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


class PhaseSigmaVTask(AnalysisTask):
    """Density-weighted σ_v decomposed by CNM / UNM / WNM / WIM / HIM phase."""

    def __init__(self, config):
        super().__init__(config)
        self._vx = self._vy = self._rho = self._T = None

    def prepare(self, context: PipelinePlotContext) -> None:
        p = context.provider
        self._vx,  _ = p.get_slab_z(('gas', 'velocity_x'))
        self._vy,  _ = p.get_slab_z(('gas', 'velocity_y'))
        self._rho, _ = p.get_slab_z(('gas', 'density'))
        self._T,   _ = p.get_slab_z(('gas', 'temperature_despotic'))

    def compute(self, context: PipelinePlotContext) -> dict:
        vx  = self._vx.in_units('km/s').value.ravel()
        vy  = self._vy.in_units('km/s').value.ravel()
        rho = self._rho.value.ravel()
        T   = self._T.in_units('K').value.ravel()

        phase_x = mass_weighted_sigma_by_phase(vx, rho, T)
        phase_y = mass_weighted_sigma_by_phase(vy, rho, T)

        print('\n=== Phase-split density-weighted σ_v ===')
        print(f'{"phase":<6} {"cell_frac":>10} {"mass_frac":>10} '
              f'{"σ_v(x) km/s":>14} {"σ_v(y) km/s":>14}')
        for ph in PHASE_ORDER:
            px, py = phase_x[ph], phase_y[ph]
            print(f'{ph:<6} {px["cell_frac"]:>10.3f} {px["mass_frac"]:>10.3f} '
                  f'{px["sigma"]:>14.2f} {py["sigma"]:>14.2f}')
        print('=====================================\n')

        return {'phase_x': phase_x, 'phase_y': phase_y,
                'vx': vx, 'vy': vy, 'rho': rho, 'T': T}

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
        masks = classify_temperature_phase(results['T'])
        rho   = results['rho']

        n_phases = len(PHASE_ORDER)
        fig, axes = plt.subplots(n_phases, 2, figsize=(13, 3.2 * n_phases))
        for col, (vel, los, phase_dict) in enumerate([
            (results['vx'], 'x', results['phase_x']),
            (results['vy'], 'y', results['phase_y']),
        ]):
            for row, p in enumerate(PHASE_ORDER):
                ax = axes[row, col]
                m  = masks[p]
                if m.sum() == 0:
                    ax.text(0.5, 0.5, f'{p}: no cells',
                            transform=ax.transAxes, ha='center', va='center')
                    continue
                v_mean = phase_dict[p]['v_mean']
                v  = vel[m]
                lo, hi = np.percentile(v, [0.5, 99.5])
                bins   = np.linspace(lo, hi, 120)
                ax.hist(v, bins=bins, weights=rho[m], density=True,
                        histtype='step', lw=1.6, color=PHASE_COLOR[p])
                sig = phase_dict[p]['sigma']
                mf  = phase_dict[p]['mass_frac']
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
