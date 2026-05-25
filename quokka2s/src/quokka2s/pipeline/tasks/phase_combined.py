"""PhaseCombinedTask: 6-column ρ-T phase diagram = PhasePlotTask + colden, with
optional cross-L_ext shared colour bars.

Layout (2 rows × 6 columns):
    Row 0 (Y = log10 T_QUOKKA):    mass | ⟨logN_H⟩ | CO | C+ | Hα | HI
    Row 1 (Y = log10 T_DESPOTIC):  mass | ⟨logN_H⟩ | CO | C+ | Hα | HI

Five columns are *sum* reductions (mass + 4 line luminosities), identical to
PhasePlotTask: colour = log10(Σ weight per bin), spanning `_COLOR_DEX` decades
down from the per-column max.  The second column is the *mean* reduction from
PhaseColdenTask: colour = mass-weighted ⟨log10 N_H⟩_M per bin.

Shared colour bars across L_ext
-------------------------------
When ``share_lext_partners`` is non-empty (e.g. ``(0.0, 9.0)``), ``plot()``
reads the sibling-dir PhaseCombinedTask intermediates for those L_ext values
and pools each column's colour range across all of them, so the
L_ext=0 and L_ext=9 figures use IDENTICAL (vmin, vmax) per column and are
directly comparable.  Workflow: compute both L_ext first, then (re-)plot both
— each plot pass reads the other's intermediate and applies the pooled range.

Output: phase_combined.png  (in the current L_ext output dir)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ..base import AnalysisTask, PipelinePlotContext
from .temperature_lext_diff import (
    _glob_one_taskcache,
    _load_results,
)


# (column_key, reduction, colourbar label)
#   reduction='sum'  → colour = log10(Σ weight),  span _COLOR_DEX dex from max
#   reduction='mean' → colour = mass-weighted ⟨log10 N_H⟩,  p1–p99 range
_COLUMNS = [
    ('mass',   'sum',  r'$\log_{10}\,M_{\rm bin}$ [g]'),
    ('colden', 'mean', r'$\langle\log_{10}\,N_{\rm H}\rangle_M$ [cm$^{-2}$]'),
    ('CO',     'sum',  r'$\log_{10}\,L_{\rm CO}$ [erg s$^{-1}$]'),
    ('Cplus',  'sum',  r'$\log_{10}\,L_{\rm C^+}$ [erg s$^{-1}$]'),
    ('Halpha', 'sum',  r'$\log_{10}\,L_{\rm H\alpha}$ [erg s$^{-1}$]'),
    ('HI',     'sum',  r'$\log_{10}\,L_{\rm HI}$ [erg s$^{-1}$]'),
]

# (row_key, y-axis label)
_ROWS = [
    ('T_QUOKKA',   r'$\log_{10}\,T_{\rm QUOKKA}$ [K]'),
    ('T_DESPOTIC', r'$\log_{10}\,T_{\rm DESPOTIC}$ [K]'),
]


class PhaseCombinedTask(AnalysisTask):
    """6-column ρ–T phase plot (sum-weighted mass+lines + mean-weighted colden),
    with optional shared colour bars across L_ext runs."""

    _PHASE_BOUNDARIES_LOG_T = (np.log10(2.0e4), np.log10(1.0e6))
    _COLOR_DEX = 4.0    # sum columns span this many decades down from max

    def __init__(self, config,
                 bin_dex: float = 0.2,
                 share_lext_partners: tuple[float, ...] = (0.0, 9.0),
                 filename: str = 'phase_combined.png'):
        super().__init__(config)
        self.bin_dex = float(bin_dex)
        # L_ext values to pool colour ranges over (incl. the current run).
        # Empty tuple → each figure auto-scales independently (no sharing).
        self.share_lext_partners = tuple(float(x) for x in share_lext_partners)
        self.filename = filename

    # ── compute ─────────────────────────────────────────────────────────
    def compute(self, context: PipelinePlotContext) -> dict:
        # Memory-frugal at down=1 (each field is a ~1 GB cube): we hold only the
        # axis arrays + dV + mass throughout, and stream the colden / luminosity
        # fields ONE at a time (load → histogram → free).  The naive
        # "load all 11 fields up front" version peaks ~20 GB and OOM-kills a
        # 24 GB Mac at down=1.
        p = context.provider

        # --- base fields: axes (ρ, T_QK, T_DSP) + cell volume ---
        rho_u,   _ = p.get_slab_z(('gas', 'density'))
        T_qk_u,  _ = p.get_slab_z(('gas', 'temperature_quokka'))
        T_dsp_u, _ = p.get_slab_z(('gas', 'temperature_despotic'))
        dx_u,    _ = p.get_slab_z(('boxlib', 'dx'))
        dy_u,    _ = p.get_slab_z(('boxlib', 'dy'))
        dz_u,    _ = p.get_slab_z(('boxlib', 'dz'))

        rho   = np.asarray(rho_u.in_cgs()).ravel()
        T_qk  = np.asarray(T_qk_u.in_cgs()).ravel()
        T_dsp = np.asarray(T_dsp_u.in_cgs()).ravel()
        dV    = (np.asarray(dx_u.in_cgs()) *
                 np.asarray(dy_u.in_cgs()) *
                 np.asarray(dz_u.in_cgs())).ravel()
        del rho_u, T_qk_u, T_dsp_u, dx_u, dy_u, dz_u

        with np.errstate(divide='ignore', invalid='ignore'):
            log_rho   = np.log10(np.where(rho   > 0, rho,   np.nan))
            log_T_qk  = np.log10(np.where(T_qk  > 0, T_qk,  np.nan))
            log_T_dsp = np.log10(np.where(T_dsp > 0, T_dsp, np.nan))
        del T_qk, T_dsp

        rho_lo, rho_hi = float(np.nanmin(log_rho)),   float(np.nanmax(log_rho))
        qk_lo,  qk_hi  = float(np.nanmin(log_T_qk)),  float(np.nanmax(log_T_qk))
        dsp_lo, dsp_hi = float(np.nanmin(log_T_dsp)), float(np.nanmax(log_T_dsp))
        y_lo = min(qk_lo, dsp_lo)
        y_hi = max(qk_hi, dsp_hi)

        def _aligned_edges(lo: float, hi: float, step: float) -> np.ndarray:
            lo_snap = np.floor(lo / step) * step
            hi_snap = (np.ceil(hi / step) + 0.5) * step
            n_bins = int(np.round((hi_snap - lo_snap) / step))
            return np.linspace(lo_snap, hi_snap, n_bins + 1)

        x_edges     = _aligned_edges(rho_lo, rho_hi, self.bin_dex)
        y_qk_edges  = _aligned_edges(y_lo,   y_hi,   self.bin_dex)
        y_dsp_edges = y_qk_edges.copy()

        # Held throughout: mass = ρ·dV (the mass column AND colden weight).
        mass = rho * dV
        del rho

        # Histogram one sum-weight against both T rows (NaN coords auto-drop).
        def _sum_hist(weight, store, name):
            H_qk,  _, _ = np.histogram2d(log_rho, log_T_qk,  bins=[x_edges, y_qk_edges],  weights=weight)
            H_dsp, _, _ = np.histogram2d(log_rho, log_T_dsp, bins=[x_edges, y_dsp_edges], weights=weight)
            store[f'T_QUOKKA_{name}']   = H_qk
            store[f'T_DESPOTIC_{name}'] = H_dsp

        sum_hists: dict[str, np.ndarray] = {}
        _sum_hist(mass, sum_hists, 'mass')              # mass column

        # --- luminosity sum columns, streamed one at a time ---
        for fld_name, store_key in (
            ('CO_luminosity',      'CO'),
            ('C+_luminosity',      'Cplus'),
            ('H_alpha_luminosity', 'Halpha'),
            ('HI_luminosity',      'HI'),
        ):
            lum_u, _ = p.get_slab_z(('gas', fld_name))
            weight = np.asarray(lum_u.in_cgs()).ravel() * dV
            del lum_u
            _sum_hist(weight, sum_hists, store_key)
            del weight

        # --- colden column: mass-weighted ⟨log N_H⟩ per bin (mean reduction) ---
        colden_u, _ = p.get_slab_z(('gas', 'column_density_H'))
        colden = np.asarray(colden_u.in_cgs()).ravel()
        del colden_u
        with np.errstate(divide='ignore', invalid='ignore'):
            log_colden = np.log10(np.where(colden > 0, colden, np.nan))
        del colden
        mass_x_logN = mass * log_colden

        colden_mean: dict[str, np.ndarray] = {}
        for y_data, y_edges_arr, key in (
            (log_T_qk,  y_qk_edges,  'T_QUOKKA'),
            (log_T_dsp, y_dsp_edges, 'T_DESPOTIC'),
        ):
            valid = (np.isfinite(log_rho) & np.isfinite(y_data)
                     & np.isfinite(log_colden) & np.isfinite(mass))
            num, _, _ = np.histogram2d(log_rho[valid], y_data[valid],
                                       bins=[x_edges, y_edges_arr],
                                       weights=mass_x_logN[valid])
            den, _, _ = np.histogram2d(log_rho[valid], y_data[valid],
                                       bins=[x_edges, y_edges_arr],
                                       weights=mass[valid])
            with np.errstate(invalid='ignore', divide='ignore'):
                colden_mean[key] = np.where(den > 0, num / den, np.nan)

        return {
            'sum_hists':   sum_hists,
            'colden_mean': colden_mean,
            'x_edges':     x_edges,
            'y_qk_edges':  y_qk_edges,
            'y_dsp_edges': y_dsp_edges,
        }

    # ── shared-range helpers ────────────────────────────────────────────
    def _sibling_dir(self, l_ext: float) -> Path:
        """Sibling output dir for a given L_ext (mirrors config.OUTPUT_DIR naming)."""
        cur = Path(self.config.output_dir)
        name = cur.name
        idx = name.rfind('_Lext')
        if idx < 0:
            return cur.parent / f'{name}_Lext{l_ext:g}kpc'
        base = name[:idx]
        geom_suffix = '_sphere' if '_sphere' in name[idx:] else ''
        return cur.parent / f'{base}_Lext{l_ext:g}kpc{geom_suffix}'

    def _gather_results(self, own_results: dict) -> list[dict]:
        """Own results + any sibling PhaseCombinedTask intermediates for the
        partner L_ext values.  Falls back to [own] only if no sharing or the
        siblings are missing."""
        if not self.share_lext_partners:
            return [own_results]

        cur_lext = float(getattr(self.config, 'column_extension_lateral_kpc', 0.0))
        gathered = [own_results]
        for l_ext in self.share_lext_partners:
            if abs(l_ext - cur_lext) < 1e-9:
                continue   # the current run is already `own_results`
            sib = self._sibling_dir(l_ext)
            path = _glob_one_taskcache(sib, 'PhaseCombinedTask')
            if path is None:
                print(f'  [share] sibling L_ext={l_ext:g} kpc not found at '
                      f'{sib.name}/task_intermediates — its range is excluded '
                      f'(re-plot after it computes for full sharing).')
                continue
            gathered.append(_load_results(path))
            print(f'  [share] pooled colour range with L_ext={l_ext:g} kpc '
                  f'({sib.name})')
        return gathered

    def _column_range(self, col_key: str, reduction: str,
                      results_list: list[dict]) -> tuple[float, float]:
        """Pool (vmin, vmax) for one column across all gathered results."""
        if reduction == 'sum':
            maxes = []
            for res in results_list:
                sh = res['sum_hists']
                for row_key, *_ in _ROWS:
                    H = np.asarray(sh[f'{row_key}_{col_key}'])
                    pos = H[H > 0]
                    if pos.size:
                        maxes.append(float(np.nanmax(pos)))
            if not maxes:
                return (0.0, 1.0)
            log_vmax = float(np.log10(max(maxes)))
            return (log_vmax - self._COLOR_DEX, log_vmax)

        # reduction == 'mean'  (colden) — pool finite means, p1–p99
        vals = []
        for res in results_list:
            cm = res['colden_mean']
            for row_key, *_ in _ROWS:
                arr = np.asarray(cm[row_key])
                vals.append(arr[np.isfinite(arr)].ravel())
        allv = np.concatenate(vals) if vals else np.array([0.0])
        if allv.size == 0:
            return (0.0, 1.0)
        return (float(np.nanpercentile(allv, 1.0)),
                float(np.nanpercentile(allv, 99.0)))

    # ── plot ────────────────────────────────────────────────────────────
    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        x_edges = np.asarray(results['x_edges'])
        y_edges = {
            'T_QUOKKA':   np.asarray(results['y_qk_edges']),
            'T_DESPOTIC': np.asarray(results['y_dsp_edges']),
        }

        # Pool colour ranges across L_ext partners (incl. self).
        shared = self._gather_results(results)
        col_ranges = {
            col_key: self._column_range(col_key, reduction, shared)
            for col_key, reduction, _label in _COLUMNS
        }

        n_rows = len(_ROWS)
        n_cols = len(_COLUMNS)

        fig = plt.figure(figsize=(2.8 * n_cols, 3.0 * n_rows))
        gs = fig.add_gridspec(
            3, n_cols,
            height_ratios=[0.04, 1.0, 1.0],
            hspace=0.35, wspace=0.30,
            left=0.05, right=0.99, top=0.92, bottom=0.10,
        )
        cax_row = [fig.add_subplot(gs[0, c]) for c in range(n_cols)]
        axes = np.array([
            [fig.add_subplot(gs[1, c]) for c in range(n_cols)],
            [fig.add_subplot(gs[2, c]) for c in range(n_cols)],
        ])
        for c in range(n_cols):
            axes[1, c].sharex(axes[0, c])
            axes[1, c].sharey(axes[0, c])
        for c in range(1, n_cols):
            axes[0, c].sharex(axes[0, 0])
            axes[0, c].sharey(axes[0, 0])

        for c, (col_key, reduction, col_label) in enumerate(_COLUMNS):
            vmin, vmax = col_ranges[col_key]
            im = None
            for r, (row_key, y_axis_label) in enumerate(_ROWS):
                ax = axes[r, c]
                ye = y_edges[row_key]
                if reduction == 'sum':
                    H = np.asarray(results['sum_hists'][f'{row_key}_{col_key}'])
                    with np.errstate(divide='ignore'):
                        data = np.where(H > 0, np.log10(np.where(H > 0, H, 1.0)), np.nan)
                else:  # mean — already in log10 N_H units
                    data = np.asarray(results['colden_mean'][row_key])

                im = ax.imshow(
                    data.T,
                    origin='lower',
                    extent=[x_edges[0], x_edges[-1], ye[0], ye[-1]],
                    aspect='auto',
                    cmap='viridis_r',
                    norm=Normalize(vmin=vmin, vmax=vmax),
                )
                ax.tick_params(axis='both', labelsize=8)
                ax.set_xlim(x_edges[0], x_edges[-1])
                ax.set_ylim(ye[0], ye[-1])
                for log_T_bnd in self._PHASE_BOUNDARIES_LOG_T:
                    if ye[0] <= log_T_bnd <= ye[-1]:
                        ax.axhline(log_T_bnd, color='gray', ls='--', lw=0.6, alpha=0.7)
                if c == 0:
                    ax.set_ylabel(y_axis_label, fontsize=10)
                if r == n_rows - 1:
                    ax.set_xlabel(r'$\log_{10}\,\rho$ [g cm$^{-3}$]', fontsize=9)
                if r == 0:
                    ax.tick_params(labelbottom=False)

            cbar = fig.colorbar(im, cax=cax_row[c], orientation='horizontal')
            cbar.ax.tick_params(labelsize=8, top=True, bottom=False,
                                labeltop=True, labelbottom=False)
            cax_row[c].set_title(col_label, fontsize=9, pad=4)

        # Title notes the L_ext of THIS figure so the two siblings are labelled.
        # Left-aligned in the top-left corner so it does not collide with the
        # centre columns' colour-bar titles.
        cur_lext = float(getattr(self.config, 'column_extension_lateral_kpc', 0.0))
        fig.text(0.005, 0.985, f'$L_{{\\rm ext}}$ = {cur_lext:g} kpc',
                 fontsize=13, fontweight='bold', ha='left', va='top')

        out = context.config.output_dir / self.filename
        fig.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
