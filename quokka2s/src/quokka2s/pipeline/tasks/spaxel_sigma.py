"""SpaxelSigmaTask: per-spaxel σ_LOS, aggregated to a galaxy-level σ.

For each LOS direction (x, y) and each weighting scheme (density + the species
in SPECIES_CFG):
  - Per spaxel (i,j), compute weighted mean V_LOS(i,j) and σ_LOS(i,j) along
    the column.
  - Aggregate across spaxels:
      * density: unweighted mean of σ(i,j) over finite spaxels.
      * species lum: luminosity-weighted quadratic mean
            σ²_galaxy = Σ W·σ² / Σ W,    σ_galaxy = sqrt(σ²_galaxy)
        which is threshold-free (empty/near-zero spaxels contribute ~0 weight).

Each (LOS, weighting) combo is also split by ISM phase
(CNM / UNM / WNM / WIM / HIM) and reported alongside the whole-cube version.

Distinct from PhaseSigmaVTask which collapses to a single global σ over the
whole cube — that includes inter-spaxel bulk shear, while this per-spaxel σ
captures only intra-sightline dispersion.

Outputs:
  - SpaxelSigma_LOS{x,y}_{density,species...}.png
        each: 2 rows × 6 cols (row 0 σ-map, row 1 σ histogram; cols = whole + 5 phases)
  - SpaxelSigma_bar.png  — grouped bar summary
  - Console: per-LOS table of σ_galaxy by (weighting × phase)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ..base import AnalysisTask, PipelinePlotContext
from ..utils import (
    PHASE_ORDER,
    PHASE_LABEL_LINE,
    classify_temperature_phase,
    mass_weighted_sigma,
    mass_weighted_sigma_by_phase,
    spaxel_moments_along_axis,
    plane_axes,
)
from .integrated_spectrum import SPECIES_CFG


PHASE_GROUPS = ('whole',) + PHASE_ORDER
PHASE_COLOR = {
    'whole': 'dimgray',
    'CNM':   'navy',
    'UNM':   'steelblue',
    'WNM':   'mediumseagreen',
    'WIM':   'goldenrod',
    'HIM':   'crimson',
}

LOS_CFG = [
    {'name': 'x', 'axis': 0, 'vel_field': 'velocity_x'},
    {'name': 'y', 'axis': 1, 'vel_field': 'velocity_y'},
]


def _safe_filename(name: str) -> str:
    return name.replace('+', 'plus')


def _aggregate_density(sigma_map: np.ndarray) -> float:
    finite = np.isfinite(sigma_map)
    if not finite.any():
        return float('nan')
    return float(sigma_map[finite].mean())


def _aggregate_lum(sigma_map: np.ndarray, W_map: np.ndarray) -> float:
    finite = np.isfinite(sigma_map) & (W_map > 0)
    if not finite.any():
        return float('nan')
    s2 = sigma_map[finite] ** 2
    w  = W_map[finite]
    total_w = w.sum()
    if total_w <= 0:
        return float('nan')
    return float(np.sqrt(np.sum(w * s2) / total_w))


class SpaxelSigmaTask(AnalysisTask):
    """Per-spaxel σ_LOS aggregated to galaxy σ — density and species-lum weighted."""

    def __init__(self, config):
        super().__init__(config)
        self._vel = {}            # {'x': vx_kms, 'y': vy_kms}
        self._rho = None
        self._T   = None
        self._lum = {}            # {'CO': lum, 'C+': lum, 'HCO+': lum}

    def prepare(self, context: PipelinePlotContext) -> None:
        p = context.provider
        self._vel['x'], _ = p.get_slab_z(('gas', 'velocity_x'))
        self._vel['y'], _ = p.get_slab_z(('gas', 'velocity_y'))
        self._vel['x'] = self._vel['x'].in_units('km/s').value
        self._vel['y'] = self._vel['y'].in_units('km/s').value

        rho, _ = p.get_slab_z(('gas', 'density'))
        self._rho = rho.value

        T, _ = p.get_slab_z(('gas', 'temperature_despotic'))
        self._T = T.in_units('K').value

        for sp in SPECIES_CFG:
            lum, _ = p.get_slab_z(('gas', sp['lum_field']))
            self._lum[sp['name']] = lum.value

    def compute(self, context: PipelinePlotContext) -> dict:
        phase_masks = classify_temperature_phase(self._T)

        # Reference: existing PhaseSigmaV-style global σ (includes inter-spaxel shear)
        rho_flat = self._rho.ravel()
        T_flat   = self._T.ravel()
        global_x = mass_weighted_sigma_by_phase(
            self._vel['x'].ravel(), rho_flat, T_flat
        )
        global_y = mass_weighted_sigma_by_phase(
            self._vel['y'].ravel(), rho_flat, T_flat
        )
        _, sx_whole = mass_weighted_sigma(self._vel['x'].ravel(), rho_flat)
        _, sy_whole = mass_weighted_sigma(self._vel['y'].ravel(), rho_flat)
        global_sigma = {'x': global_x, 'y': global_y}
        global_sigma_whole = {'x': sx_whole, 'y': sy_whole}

        weight_specs = [{'name': 'density', 'kind': 'density', 'data': self._rho,
                         'color': 'black'}]
        for sp in SPECIES_CFG:
            weight_specs.append({
                'name':  sp['name'],
                'kind':  'lum',
                'data':  self._lum[sp['name']],
                'color': sp['color'],
            })

        results = {'global_sigma': global_sigma,
                   'global_sigma_whole': global_sigma_whole,
                   'weight_specs': weight_specs,
                   'by_los': {}}

        for los in LOS_CFG:
            los_name = los['name']
            axis     = los['axis']
            vel      = self._vel[los_name]
            results['by_los'][los_name] = {}

            for ws in weight_specs:
                w = ws['data']
                w_block = {}

                for phase in PHASE_GROUPS:
                    if phase == 'whole':
                        w_used = w
                    else:
                        w_used = w * phase_masks[phase]
                    V_map, sigma_map, W_map = spaxel_moments_along_axis(
                        w_used, vel, axis
                    )
                    if ws['kind'] == 'density':
                        sigma_galaxy = _aggregate_density(sigma_map)
                    else:
                        sigma_galaxy = _aggregate_lum(sigma_map, W_map)
                    w_block[phase] = {
                        'V_map':        V_map,
                        'sigma_map':    sigma_map,
                        'W_map':        W_map,
                        'sigma_galaxy': sigma_galaxy,
                    }

                results['by_los'][los_name][ws['name']] = w_block

        self._print_table(results)
        return results

    def _print_table(self, results: dict) -> None:
        weight_names = [ws['name'] for ws in results['weight_specs']]
        for los_name in ('x', 'y'):
            print(f'\n=== Spaxel-σ aggregate, LOS = {los_name} ===')
            header = f'{"phase":<6}'
            for wn in weight_names:
                header += f' {"σ_"+wn:>14}'
            header += f' {"σ_global(ρ)":>14}'
            print(header)
            for phase in PHASE_GROUPS:
                row = f'{phase:<6}'
                for wn in weight_names:
                    s = results['by_los'][los_name][wn][phase]['sigma_galaxy']
                    row += f' {s:>14.2f}' if np.isfinite(s) else f' {"nan":>14}'
                if phase == 'whole':
                    sg = results['global_sigma_whole'][los_name]
                else:
                    sg = results['global_sigma'][los_name][phase]['sigma']
                row += f' {sg:>14.2f}' if np.isfinite(sg) else f' {"nan":>14}'
                print(row)
        print('=' * 60)
        print('density σ_galaxy ≤ σ_global expected (gap = inter-spaxel bulk shear)')
        print('=' * 60 + '\n')

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        for los in LOS_CFG:
            los_name = los['name']
            for ws in results['weight_specs']:
                self._plot_weighting_los(results, los_name, ws)
        self._plot_summary_bar(results)

    def _plot_weighting_los(self, results: dict, los_name: str, ws: dict) -> None:
        block = results['by_los'][los_name][ws['name']]
        plane = plane_axes(los_name)  # ('y','z') or ('x','z')

        # Combined σ-value range across the four phase panels for shared colorbar.
        all_sigma = np.concatenate([
            block[p]['sigma_map'][np.isfinite(block[p]['sigma_map'])].ravel()
            for p in PHASE_GROUPS
            if np.isfinite(block[p]['sigma_map']).any()
        ]) if any(np.isfinite(block[p]['sigma_map']).any() for p in PHASE_GROUPS) else np.array([])
        if all_sigma.size > 0:
            v_lo = max(np.nanpercentile(all_sigma, 1), 1e-3)
            v_hi = max(np.nanpercentile(all_sigma, 99), v_lo * 1.01)
            map_norm = mcolors.LogNorm(vmin=v_lo, vmax=v_hi)
        else:
            map_norm = mcolors.LogNorm(vmin=1e-3, vmax=1.0)

        n_cols = len(PHASE_GROUPS)
        fig, axes = plt.subplots(2, n_cols, figsize=(4.2 * n_cols, 9))
        for col, phase in enumerate(PHASE_GROUPS):
            sigma_map    = block[phase]['sigma_map']
            sigma_galaxy = block[phase]['sigma_galaxy']

            # Row 0: σ-map
            ax_map = axes[0, col]
            im = ax_map.imshow(sigma_map.T, origin='lower', norm=map_norm,
                               cmap='viridis', aspect='auto')
            ax_map.set_xlabel(plane[0].upper())
            ax_map.set_ylabel(plane[1].upper())
            sg_str = f'{sigma_galaxy:.2f}' if np.isfinite(sigma_galaxy) else 'nan'
            ax_map.set_title(f'{phase}    σ_galaxy = {sg_str} km/s', fontsize=11,
                             color=PHASE_COLOR[phase])
            fig.colorbar(im, ax=ax_map, label='σ(i,j) [km/s]',
                         fraction=0.046, pad=0.04)

            # Row 1: σ histogram across spaxels
            ax_hist = axes[1, col]
            finite_sigma = sigma_map[np.isfinite(sigma_map)]
            if ws['kind'] == 'lum':
                W_map = block[phase]['W_map']
                weights = W_map[np.isfinite(sigma_map)]
            else:
                weights = None
            if finite_sigma.size > 0:
                lo = max(np.nanpercentile(finite_sigma, 0.5), 1e-3)
                hi = max(np.nanpercentile(finite_sigma, 99.5), lo * 1.01)
                bins = np.logspace(np.log10(lo), np.log10(hi), 60)
                ax_hist.hist(finite_sigma, bins=bins, weights=weights,
                             histtype='stepfilled', alpha=0.45,
                             color=PHASE_COLOR[phase], edgecolor=PHASE_COLOR[phase])
                if np.isfinite(sigma_galaxy):
                    ax_hist.axvline(sigma_galaxy, color=PHASE_COLOR[phase],
                                    ls='--', lw=1.2,
                                    label=f'σ_galaxy={sigma_galaxy:.2f}')
                    ax_hist.legend(fontsize=9, framealpha=0.8)
            else:
                ax_hist.text(0.5, 0.5, 'no finite spaxels',
                             transform=ax_hist.transAxes, ha='center', va='center')
            ax_hist.set_xscale('log')
            ax_hist.set_xlabel('σ(i,j) [km/s]')
            ax_hist.set_ylabel('count' if ws['kind'] == 'density' else 'lum-weighted count')
            ax_hist.grid(True, which='both', alpha=0.25, ls='--', lw=0.5)

        weight_label = 'density-weighted' if ws['kind'] == 'density' \
                                          else f'{ws["name"]} luminosity-weighted'
        fig.suptitle(f'Spaxel σ_LOS  —  LOS = {los_name},  {weight_label}\n'
                     + PHASE_LABEL_LINE, fontsize=11)
        plt.tight_layout()
        out = self.config.output_dir / (
            f'SpaxelSigma_LOS{los_name}_{_safe_filename(ws["name"])}.png'
        )
        plt.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')

    def _plot_summary_bar(self, results: dict) -> None:
        weight_names = [ws['name'] for ws in results['weight_specs']]
        weight_colors = {ws['name']: ws['color'] for ws in results['weight_specs']}
        n_w = len(weight_names)
        n_p = len(PHASE_GROUPS)

        fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
        x_pos = np.arange(n_p)
        bar_w = 0.8 / n_w

        for ax, los_name in zip(axes, ('x', 'y')):
            for j, wn in enumerate(weight_names):
                vals = [results['by_los'][los_name][wn][p]['sigma_galaxy']
                        for p in PHASE_GROUPS]
                offset = (j - (n_w - 1) / 2) * bar_w
                bars = ax.bar(x_pos + offset, vals, bar_w,
                              color=weight_colors[wn], label=wn, alpha=0.85)
                for rect, v in zip(bars, vals):
                    if np.isfinite(v):
                        ax.text(rect.get_x() + rect.get_width()/2, v,
                                f'{v:.1f}', ha='center', va='bottom', fontsize=8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(PHASE_GROUPS)
            ax.set_xlabel('Temperature phase')
            ax.set_title(f'LOS = {los_name}', fontsize=12)
            ax.grid(True, axis='y', alpha=0.3, ls='--')
            ax.legend(fontsize=9, framealpha=0.85)
        axes[0].set_ylabel('σ_galaxy [km/s]')
        fig.suptitle('Spaxel σ_LOS aggregated to galaxy (per-spaxel, '
                     'no inter-spaxel shear)\n' + PHASE_LABEL_LINE,
                     fontsize=11)
        plt.tight_layout()
        out = self.config.output_dir / 'SpaxelSigma_bar.png'
        plt.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
