"""EmitterCompareTask — compare CO/C+/HCO+ emission under three temperature methods.

Each luminosity is computed inline via lookup.line_field(), with T explicitly
passed per method. This prevents any cross-contamination between temperature
fields (e.g. using temperature_eint data but temperature_despotic label).
Note: temperature_eint is currently not registered (legacy); only quokka and
despotic are live. The task is preserved as commented-out scaffolding.

Temperature methods compared:
  temperature_quokka   — QUOKKA ideal-gas temperature (boxlib field)
  temperature_eint     — Eint bisection (DESPOTIC table, legacy, not registered)
  temperature_despotic — (n_H, N_H, dV/dr) DESPOTIC table lookup (tg_final)

Outputs:
  emitter_compare_luminosity.png   — 3×4 grid (species × method + ratio)
  emitter_compare_temperature.png  — 1×4 (T projection × method + ratio)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ..base import AnalysisTask, PipelinePlotContext
from ..prep import config as cfg
from ..prep.config import T_CUTOFF, T_CUTOFF_DEFAULT
from ..prep.physics_fields import ensure_table_lookup
from ..utils import make_axis_labels

SPECIES   = ['CO', 'C+', 'HCO+']
T_METHODS = [
    ('quokka',   'temperature_quokka',   'T$_{QUOKKA}$'),
    ('eint',     'temperature_eint',     'T$_{Eint}$'),
    ('mu_gamma', 'temperature_despotic', 'T$_{\\mu\\gamma}$'),
]


class EmitterCompareTask(AnalysisTask):
    """Compare CO/C+/HCO+ emission and density-weighted T across three temperature methods."""

    def __init__(self, config, axis: str | None = None, figure_units: str | None = None):
        super().__init__(config)
        self.axis = axis or 'x'
        self.figure_units = figure_units or config.figure_units
        self.xlabel, self.ylabel = make_axis_labels(self.axis, self.figure_units)

    def prepare(self, context: PipelinePlotContext) -> None:
        provider = context.provider

        # Shared spatial fields (temperature-independent)
        self._nH_3d,     _ = provider.get_slab_z(('gas', 'number_density_H'))
        self._col_3d,    _ = provider.get_slab_z(('gas', 'column_density_H'))
        self._rho_3d,    _ = provider.get_slab_z(('gas', 'density'))
        self._dx_3d, self._extent = provider.get_slab_z(('boxlib', 'dx'))

        # Each temperature field read separately and labelled — no aliasing
        self._T_quokka,   _ = provider.get_slab_z(('gas', 'temperature_quokka'))
        self._T_eint,     _ = provider.get_slab_z(('gas', 'temperature_eint'))
        self._T_mu_gamma, _ = provider.get_slab_z(('gas', 'temperature_despotic'))

    def compute(self, context: PipelinePlotContext) -> dict:
        lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)

        nH_min,  nH_max  = lookup.table.nH_values.min(),          lookup.table.nH_values.max()
        col_min, col_max = lookup.table.col_density_values.min(), lookup.table.col_density_values.max()
        T_min,   T_max   = lookup.table.T_values.min(),           lookup.table.T_values.max()

        # nH and col are shared — clip once
        nH_raw  = self._nH_3d.in_cgs().value
        col_raw = self._col_3d.in_cgs().value
        dx_raw  = self._dx_3d.in_cgs().value
        rho_raw = self._rho_3d.in_cgs().value

        nH_safe  = np.clip(nH_raw,  nH_min,  nH_max)
        col_safe = np.clip(col_raw, col_min, col_max)

        mass_col = np.sum(rho_raw * dx_raw, axis=0)  # g/cm²

        results: dict = {}

        # Map T_name → raw T array (in K, no units)
        T_raw_map = {
            'quokka':   self._T_quokka.in_cgs().value,
            'eint':     self._T_eint.in_cgs().value,
            'mu_gamma': self._T_mu_gamma.in_cgs().value,
        }

        for T_name, yt_field, _ in T_METHODS:
            T_raw  = T_raw_map[T_name]           # (nx, ny, nz) raw K values
            T_safe = np.clip(T_raw, T_min, T_max)  # clipped for table lookup

            # Density-weighted projected temperature — use raw T (not clipped) for physics
            results[f'T_{T_name}'] = (
                np.sum(T_raw * rho_raw * dx_raw, axis=0) / mass_col
            )

            for sp in SPECIES:
                cutoff = T_CUTOFF.get(sp, T_CUTOFF_DEFAULT)

                # lumPerH lookup — T_safe prevents NaN outside table bounds
                lumPerH = lookup.line_field(sp, 'lumPerH', nH_safe, col_safe, T_safe)
                lumPerH = np.nan_to_num(lumPerH, nan=0.0)
                # Zero out cells above temperature cutoff (use raw T for physical criterion)
                lumPerH[T_raw > cutoff] = 0.0

                lum_density = nH_safe * lumPerH             # erg/s/cm³
                results[f'{sp}_{T_name}'] = np.sum(lum_density * dx_raw, axis=0)

        results['extent'] = self._extent[self.axis]
        return results

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        extent = [float(v.to(self.figure_units).value) for v in results['extent']]
        self._plot_luminosity(results, extent)
        self._plot_temperature(results, extent)

    # ── Figure 1: luminosity 3×4 ──────────────────────────────────────────────

    def _plot_luminosity(self, results: dict, extent: list) -> None:
        fig, axes = plt.subplots(len(SPECIES), 4,
                                 figsize=(22, 5 * len(SPECIES)),
                                 squeeze=False)

        for row, sp in enumerate(SPECIES):
            # Shared LogNorm across the three temperature columns for this species
            maps = [results[f'{sp}_{name}'] for name, _, _ in T_METHODS]
            all_pos = np.concatenate([m[m > 0].ravel() for m in maps if np.any(m > 0)])
            if all_pos.size > 0:
                vmin = np.nanpercentile(all_pos, 1)
                vmax = np.nanpercentile(all_pos, 99)
                lum_norm = mcolors.LogNorm(vmin=max(vmin, 1e-40), vmax=vmax)
            else:
                lum_norm = mcolors.LogNorm()

            for col, (T_name, _, T_label) in enumerate(T_METHODS):
                ax  = axes[row, col]
                dat = results[f'{sp}_{T_name}']
                im  = ax.imshow(dat.T, origin='lower', norm=lum_norm,
                                cmap='inferno', aspect='auto',
                                extent=extent)
                fig.colorbar(im, ax=ax, label='erg/s/cm²', fraction=0.046, pad=0.04)
                ax.set_xlabel(self.xlabel, fontsize=9)
                ax.set_ylabel(self.ylabel, fontsize=9)
                ax.set_title(f'{sp}  ({T_label})', fontsize=10)

            # Column 4: ratio mu_gamma / eint
            ax_r = axes[row, 3]
            sb_eint = results[f'{sp}_eint']
            sb_mg   = results[f'{sp}_mu_gamma']
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(sb_eint > 0, sb_mg / sb_eint, np.nan)
            r_vals = ratio[np.isfinite(ratio) & (ratio > 0)]
            if r_vals.size > 0:
                r_log = np.abs(np.log10(r_vals))
                r_max = max(np.nanpercentile(r_log, 99), 0.05)
            else:
                r_max = 0.5
            ratio_norm = mcolors.LogNorm(vmin=10**(-r_max), vmax=10**r_max)
            im_r = ax_r.imshow(ratio.T, origin='lower', norm=ratio_norm,
                               cmap='RdBu_r', aspect='auto', extent=extent)
            fig.colorbar(im_r, ax=ax_r, label='ratio', fraction=0.046, pad=0.04)
            ax_r.set_xlabel(self.xlabel, fontsize=9)
            ax_r.set_ylabel(self.ylabel, fontsize=9)
            ax_r.set_title(f'{sp}  T$_{{\\mu\\gamma}}$ / T$_{{Eint}}$', fontsize=10)

        fig.suptitle('Surface Brightness Comparison — CO / C+ / HCO+  (three temperature methods)',
                     fontsize=13)
        plt.tight_layout()
        out = self.config.output_dir / 'emitter_compare_luminosity.png'
        plt.savefig(str(out), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')

    # ── Figure 2: density-weighted T projection 1×4 ──────────────────────────

    def _plot_temperature(self, results: dict, extent: list) -> None:
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))

        T_maps = [results[f'T_{name}'] for name, _, _ in T_METHODS]
        all_T  = np.concatenate([t[t > 0].ravel() for t in T_maps if np.any(t > 0)])
        if all_T.size > 0:
            t_norm = mcolors.LogNorm(vmin=np.nanpercentile(all_T, 1),
                                     vmax=np.nanpercentile(all_T, 99))
        else:
            t_norm = mcolors.LogNorm()

        for col, (T_name, _, T_label) in enumerate(T_METHODS):
            ax  = axes[col]
            dat = results[f'T_{T_name}']
            im  = ax.imshow(dat.T, origin='lower', norm=t_norm,
                            cmap='inferno', aspect='auto', extent=extent)
            fig.colorbar(im, ax=ax, label='K', fraction=0.046, pad=0.04)
            ax.set_xlabel(self.xlabel, fontsize=9)
            ax.set_ylabel(self.ylabel, fontsize=9)
            ax.set_title(f'Density-weighted T  ({T_label})', fontsize=10)

        # Column 4: ratio T_mu_gamma / T_eint
        ax_r = axes[3]
        T_eint = results['T_eint']
        T_mg   = results['T_mu_gamma']
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(T_eint > 0, T_mg / T_eint, np.nan)
        r_vals = ratio[np.isfinite(ratio) & (ratio > 0)]
        if r_vals.size > 0:
            r_max = max(np.nanpercentile(np.abs(np.log10(r_vals)), 99), 0.05)
        else:
            r_max = 0.5
        im_r = ax_r.imshow(ratio.T, origin='lower',
                           norm=mcolors.LogNorm(vmin=10**(-r_max), vmax=10**r_max),
                           cmap='RdBu_r', aspect='auto', extent=extent)
        fig.colorbar(im_r, ax=ax_r, label='ratio', fraction=0.046, pad=0.04)
        ax_r.set_xlabel(self.xlabel, fontsize=9)
        ax_r.set_ylabel(self.ylabel, fontsize=9)
        ax_r.set_title('T$_{\\mu\\gamma}$ / T$_{Eint}$  (density-weighted proj.)', fontsize=10)

        fig.suptitle('Density-Weighted Temperature Projection — three temperature methods',
                     fontsize=13)
        plt.tight_layout()
        out = self.config.output_dir / 'emitter_compare_temperature.png'
        plt.savefig(str(out), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
