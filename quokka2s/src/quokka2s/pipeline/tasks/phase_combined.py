# ═══════════════════════════════════════════════════════════════════════════
# ⚠️  DEPRECATED 2026-06-19  — this module is no longer used.
#
# Replaced by:
#   • tasks/phase_hist.py            — PhaseHistTask  (one 2D histogram each)
#                                      PhaseHistNHRhoTask  (special NH-ρ)
#   • tasks/phase_combined_plot.py   — PhaseCombinedPlotTask  (plot-only,
#                                      reads sibling intermediates)
#
# Why: the old single PhaseCombinedTask computed 15 histograms but only
# displayed 7.  The new split-task design only computes what's actually
# registered, AND each intermediate is independently cacheable — changing
# one luminosity field only invalidates that one task's cache.
#
# Convention: when a task is replaced, the OLD code stays in the file
# (wrapped in a docstring so Python ignores it) as historical reference.
# See memory `[[deprecate-by-wrapping-not-deleting]]`.
# ═══════════════════════════════════════════════════════════════════════════
"""[Original 2026-06-12 docstring — preserved for reference]

PhaseCombinedTask: 6-column ρ-T phase diagram = PhasePlotTask + an N_H–ρ
mass panel, with optional cross-L_ext shared colour bars.

Layout (2 rows × 6 columns):
    Row 0 (Y = log10 T_QUOKKA):    mass |  [NH-ρ spans both rows]  | CO | C+ | Hα | HI
    Row 1 (Y = log10 T_DESPOTIC):  mass |        ↓                  | CO | C+ | Hα | HI

Five "regular" columns are *sum* reductions (mass + 4 line luminosities),
identical to PhasePlotTask: colour = log10(Σ weight per bin), spanning
`_COLOR_DEX` decades down from the per-column max.  The second column is a
single tall panel showing the mass distribution in the (log N_H, log ρ)
plane (replaces the old ⟨logN_H⟩ colden mean column; N_H uses the DESPOTIC
table's 35-pt grid so the panel is directly comparable to
output/table_plots/<TAG>_L<L>_mass/tg_alldvdr_*.png).  As of 2026-06-12 this
panel folds in what phase_rho_3panel_mass.png used to provide, so
phase_rho_3panel_mass is no longer needed.

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

# ═══════════════════════════════════════════════════════════════════════════
# ORIGINAL IMPLEMENTATION BELOW — wrapped in a triple-quoted string so Python
# treats it as a docstring (no execution).  Kept verbatim as historical
# reference for the architecture that produced phase_combined.png from
# 2026-06-12 to 2026-06-19.  Do NOT un-wrap without first reviewing whether
# the new PhaseHistTask / PhaseCombinedPlotTask split should be extended
# instead.
# ═══════════════════════════════════════════════════════════════════════════
_DEPRECATED_2026_06_19_ORIGINAL_CODE = r'''
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


# (col_key, reduction, t_row, label)
#   reduction='sum'        → ρ-T phase plot, t_row picks T_QUOKKA/T_DESPOTIC.
#   reduction='sum_NH_rho' → (N_H, ρ) mass panel; t_row=None.
# Single-row layout: each tuple = one panel.  Cache pooling keyed by col_key
# alone so the two `mass` entries share a vmin/vmax (same field, different y).
_COLUMNS = [
    # 2026-06-19 layout: 2 rows × 4 cols.
    #   Row 1: mass×T_QK | mass×T_DSP | mass×T_two_regime | NH-ρ
    #   Row 2: CO        | C+         | Hα                | HI    (all y=T_two_regime)
    # Row 2 species use T_two_regime so their y-axis matches Row 1 col 3 — the
    # "canonical cell T" used by all hot-branch luminosity formulas (Hα Saha,
    # HI two-regime, C+ LTE).  Row 1 cols 1, 2 keep T_QK / T_DSP as reference.
    ('mass',    'sum',         'T_QUOKKA',     r'$\log_{10}\,M_{\rm bin}$ [g]'),
    ('mass',    'sum',         'T_DESPOTIC',   r'$\log_{10}\,M_{\rm bin}$ [g]'),
    ('mass',    'sum',         'T_TWO_REGIME', r'$\log_{10}\,M_{\rm bin}$ [g]'),
    ('NH_rho',  'sum_NH_rho',  None,           r'$\log_{10}\,M_{\rm bin}$ [g]'),
    ('CO',      'sum',         'T_TWO_REGIME', r'$\log_{10}\,L_{\rm CO}$ [erg s$^{-1}$]'),
    ('Cplus',   'sum',         'T_TWO_REGIME', r'$\log_{10}\,L_{\rm C^+}$ [erg s$^{-1}$]'),
    ('Halpha',  'sum',         'T_TWO_REGIME', r'$\log_{10}\,L_{\rm H\alpha}$ [erg s$^{-1}$]'),
    ('HI',      'sum',         'T_TWO_REGIME', r'$\log_{10}\,L_{\rm HI}$ [erg s$^{-1}$]'),
]

# (row_key, y-axis label)
_ROWS = [
    ('T_QUOKKA',     r'$\log_{10}\,T_{\rm QUOKKA}$ [K]'),
    ('T_DESPOTIC',   r'$\log_{10}\,T_{\rm DESPOTIC}$ [K]'),
    ('T_TWO_REGIME', r'$\log_{10}\,T_{\rm use}$ [K]'),
]


class PhaseCombinedTask(AnalysisTask):
    """6-column ρ–T phase plot (sum-weighted mass+lines + mean-weighted colden),
    with optional shared colour bars across L_ext runs."""

    def __init__(self, config,
                 bin_dex: float = 0.2,
                 share_lext_partners: tuple[float, ...] = (0.0, 9.0),
                 filename: str = 'phase_combined.png',
                 columns: tuple[str, ...] | None = None,
                 cache_version: str = 'v3'):
        # cache_version: bump to bust on-disk task_intermediate caches when
        # compute() output schema changes (2026-06-12: colden_mean → nh_rho_hist;
        # 2026-06-18: T_two_regime panel added + 2×4 layout).
        super().__init__(config)
        self.bin_dex = float(bin_dex)
        self.cache_version = str(cache_version)
        # L_ext values to pool colour ranges over (incl. the current run).
        # Empty tuple → each figure auto-scales independently (no sharing).
        self.share_lext_partners = tuple(float(x) for x in share_lext_partners)
        self.filename = filename
        # Subset of column keys to render (default = all 6 in _COLUMNS).
        # E.g. columns=('mass', 'colden') for a 2×2 phase plot.
        self.columns = tuple(columns) if columns is not None else None

    # ── compute ─────────────────────────────────────────────────────────
    def compute(self, context: PipelinePlotContext) -> dict:
        # Memory-frugal at down=1 (each field is a ~1 GB cube): we hold only the
        # axis arrays + dV + mass throughout, and stream the colden / luminosity
        # fields ONE at a time (load → histogram → free).  The naive
        # "load all 11 fields up front" version peaks ~20 GB and OOM-kills a
        # 24 GB Mac at down=1.
        p = context.provider

        # --- base fields: axes (ρ, T_QK, T_DSP, T_two_regime) + cell volume ---
        rho_u,   _ = p.get_slab_z(('gas', 'density'))
        T_qk_u,  _ = p.get_slab_z(('gas', 'temperature_quokka'))
        T_dsp_u, _ = p.get_slab_z(('gas', 'temperature_despotic'))
        T_2r_u,  _ = p.get_slab_z(('gas', 'temperature_two_regime'))
        dx_u,    _ = p.get_slab_z(('boxlib', 'dx'))
        dy_u,    _ = p.get_slab_z(('boxlib', 'dy'))
        dz_u,    _ = p.get_slab_z(('boxlib', 'dz'))

        rho   = np.asarray(rho_u.in_cgs()).ravel()
        T_qk  = np.asarray(T_qk_u.in_cgs()).ravel()
        T_dsp = np.asarray(T_dsp_u.in_cgs()).ravel()
        T_2r  = np.asarray(T_2r_u.in_cgs()).ravel()
        dV    = (np.asarray(dx_u.in_cgs()) *
                 np.asarray(dy_u.in_cgs()) *
                 np.asarray(dz_u.in_cgs())).ravel()
        del rho_u, T_qk_u, T_dsp_u, T_2r_u, dx_u, dy_u, dz_u

        with np.errstate(divide='ignore', invalid='ignore'):
            log_rho   = np.log10(np.where(rho   > 0, rho,   np.nan))
            log_T_qk  = np.log10(np.where(T_qk  > 0, T_qk,  np.nan))
            log_T_dsp = np.log10(np.where(T_dsp > 0, T_dsp, np.nan))
            log_T_2r  = np.log10(np.where(T_2r  > 0, T_2r,  np.nan))
        del T_qk, T_dsp, T_2r

        rho_lo, rho_hi = float(np.nanmin(log_rho)),   float(np.nanmax(log_rho))
        qk_lo,  qk_hi  = float(np.nanmin(log_T_qk)),  float(np.nanmax(log_T_qk))
        dsp_lo, dsp_hi = float(np.nanmin(log_T_dsp)), float(np.nanmax(log_T_dsp))
        t2r_lo, t2r_hi = float(np.nanmin(log_T_2r)),  float(np.nanmax(log_T_2r))
        y_lo = min(qk_lo, dsp_lo, t2r_lo)
        y_hi = max(qk_hi, dsp_hi, t2r_hi)

        def _aligned_edges(lo: float, hi: float, step: float) -> np.ndarray:
            lo_snap = np.floor(lo / step) * step
            hi_snap = (np.ceil(hi / step) + 0.5) * step
            n_bins = int(np.round((hi_snap - lo_snap) / step))
            return np.linspace(lo_snap, hi_snap, n_bins + 1)

        x_edges     = _aligned_edges(rho_lo, rho_hi, self.bin_dex)
        y_qk_edges  = _aligned_edges(y_lo,   y_hi,   self.bin_dex)
        y_dsp_edges = y_qk_edges.copy()
        y_2r_edges  = y_qk_edges.copy()

        # Held throughout: mass = ρ·dV (the mass column AND colden weight).
        mass = rho * dV
        del rho

        # Histogram one sum-weight against all 3 T rows (NaN coords auto-drop).
        def _sum_hist(weight, store, name):
            H_qk,  _, _ = np.histogram2d(log_rho, log_T_qk,  bins=[x_edges, y_qk_edges],  weights=weight)
            H_dsp, _, _ = np.histogram2d(log_rho, log_T_dsp, bins=[x_edges, y_dsp_edges], weights=weight)
            H_2r,  _, _ = np.histogram2d(log_rho, log_T_2r,  bins=[x_edges, y_2r_edges],  weights=weight)
            store[f'T_QUOKKA_{name}']     = H_qk
            store[f'T_DESPOTIC_{name}']   = H_dsp
            store[f'T_TWO_REGIME_{name}'] = H_2r

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

        # --- NH-ρ column: mass histogram on (TABLE N_H grid × ρ bin grid) ---
        #   Replaces the old ⟨logN_H⟩ "colden mean" column (2026-06-12).
        #   x-bins = DESPOTIC table's 35-pt col_density grid → contour shape
        #   lines up with output/table_plots/<TAG>_L<L>_mass/tg_alldvdr_*.png
        colden_u, _ = p.get_slab_z(('gas', 'column_density_H'))
        colden = np.asarray(colden_u.in_cgs()).ravel()
        del colden_u
        with np.errstate(divide='ignore', invalid='ignore'):
            log_colden = np.log10(np.where(colden > 0, colden, np.nan))
        del colden

        from ...tables.plotting import _log_edges
        table_npz = np.load(self.config.despotic_table_path, allow_pickle=True)
        nh_table_edges = _log_edges(table_npz['col_density_values'])

        valid_nh = (np.isfinite(log_colden) & np.isfinite(log_rho)
                    & np.isfinite(mass))
        nh_rho_hist, _, _ = np.histogram2d(
            log_colden[valid_nh], log_rho[valid_nh],
            bins=[nh_table_edges, x_edges],
            weights=mass[valid_nh],
        )

        return {
            'sum_hists':       sum_hists,
            'nh_rho_hist':     nh_rho_hist,
            'nh_rho_x_edges':  nh_table_edges,
            'nh_rho_y_edges':  x_edges.copy(),
            'x_edges':         x_edges,
            'y_qk_edges':      y_qk_edges,
            'y_dsp_edges':     y_dsp_edges,
            'y_2r_edges':      y_2r_edges,
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
        """Pool (vmin, vmax) for one column across all gathered results.

        Uses the FULL data range (log10 nanmin → log10 nanmax of positive
        bins), per [[plot-full-range-colorbar]].  Previously hard-coded
        to a 4-dex window below peak, which clipped tiny-but-non-zero bins
        to yellow and made distinct species' "low-luminosity outlines"
        look identical (2026-06-19 user feedback).
        """
        if reduction == 'sum':
            pos_all = []
            for res in results_list:
                sh = res['sum_hists']
                for row_key, *_ in _ROWS:
                    H = np.asarray(sh[f'{row_key}_{col_key}'])
                    pos_all.append(H[H > 0])
            pos_all = np.concatenate(pos_all) if pos_all else np.array([])
            if pos_all.size == 0:
                return (0.0, 1.0)
            return (float(np.log10(np.nanmin(pos_all))),
                    float(np.log10(np.nanmax(pos_all))))

        if reduction == 'sum_NH_rho':
            pos_all = []
            for res in results_list:
                H = np.asarray(res['nh_rho_hist'])
                pos_all.append(H[H > 0])
            pos_all = np.concatenate(pos_all) if pos_all else np.array([])
            if pos_all.size == 0:
                return (0.0, 1.0)
            return (float(np.log10(np.nanmin(pos_all))),
                    float(np.log10(np.nanmax(pos_all))))

        return (0.0, 1.0)

    # ── plot ────────────────────────────────────────────────────────────
    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        x_edges = np.asarray(results['x_edges'])
        y_edges = {
            'T_QUOKKA':     np.asarray(results['y_qk_edges']),
            'T_DESPOTIC':   np.asarray(results['y_dsp_edges']),
            'T_TWO_REGIME': np.asarray(results['y_2r_edges']),
        }

        # Pool colour ranges across L_ext partners (incl. self).
        shared = self._gather_results(results)
        # Dedup col_keys: all 3 mass×T panels share col_key 'mass' → same vmin/vmax.
        unique_col_keys = {col_key: reduction
                           for col_key, reduction, _t, _l in _COLUMNS}
        col_ranges = {
            col_key: self._column_range(col_key, reduction, shared)
            for col_key, reduction in unique_col_keys.items()
        }

        # Subset columns if user requested only some (e.g. mass + colden).
        if self.columns is not None:
            cols_used = [c for c in _COLUMNS if c[0] in self.columns]
        else:
            cols_used = list(_COLUMNS)

        # 2026-06-18 layout: 2 rows × 4 cols.
        #   Row 1: mass×T_QK | mass×T_DSP | mass×T_two_regime | NH-ρ
        #   Row 2: CO        | C+         | Hα                | HI    (all y=T_DSP)
        n_panels = len(cols_used)
        n_cols   = 4
        n_rows   = (n_panels + n_cols - 1) // n_cols   # 2 for our 8-panel default
        panel_inch = 3.4
        # Original 1×7 per-panel proportions: each cell ≈ panel_inch × panel_inch.
        # Top-row panels hide their x-axis labels (sharex with bottom row).
        fig = plt.figure(figsize=(panel_inch * n_cols + 0.4,
                                  panel_inch * n_rows + 1.6))
        # Nested GridSpec: outer rows are the row GROUPS (each group =
        # cbar-on-top + panel-below).  Inner controls the cbar↔panel coupling
        # tightly so cbar sits right above its panel; outer hspace controls
        # the LARGER gap between row groups so the bottom-row cbar title
        # doesn't crowd into the top-row panels above it.
        outer = fig.add_gridspec(
            n_rows, 1,
            hspace=0.40,                       # gap between row groups
            left=0.05, right=0.99, top=0.93, bottom=0.07,
        )
        cax_grid: list[list] = []
        ax_grid:  list[list] = []
        for r in range(n_rows):
            inner = outer[r].subgridspec(
                2, n_cols,
                height_ratios=[0.06, 1.0],
                hspace=0.05,                   # cbar→panel tight coupling
                wspace=0.30,
            )
            cax_grid.append([fig.add_subplot(inner[0, c]) for c in range(n_cols)])
            ax_grid.append( [fig.add_subplot(inner[1, c]) for c in range(n_cols)])

        # Helper: (row, col) for the c-th panel in cols_used.
        def _rc(c: int) -> tuple[int, int]:
            return c // n_cols, c % n_cols

        # Share y within each (row, T_group).  Row 2 (species) all use
        # T_two_regime (2026-06-19) → they share y across the row, AND share
        # y with Row 1 col 3 (mass × T_two_regime) — same physical axis.
        # Row 1 cols 1, 2 (T_QK, T_DSP) and col 4 (NH-ρ) have their own y.
        T_2R_idx = [c for c, (_, red, t, _l) in enumerate(cols_used)
                    if red == 'sum' and t == 'T_TWO_REGIME']
        if T_2R_idx:
            base_y = T_2R_idx[0]
            r0, c0 = _rc(base_y)
            for c in T_2R_idx:
                if c == base_y:
                    continue
                r, cc = _rc(c)
                ax_grid[r][cc].sharey(ax_grid[r0][c0])

        # Share x across all ρ-axis panels (everything except NH-ρ).  This
        # spans both rows so density scales line up vertically.
        rho_x_idx = [c for c, (_, red, _t, _l) in enumerate(cols_used)
                     if red == 'sum']
        if rho_x_idx:
            base_x = rho_x_idx[0]
            r0, c0 = _rc(base_x)
            for c in rho_x_idx:
                if c == base_x:
                    continue
                r, cc = _rc(c)
                ax_grid[r][cc].sharex(ax_grid[r0][c0])

        # Track which T_two_regime panel on row 2 (species) gets the y-label
        # (leftmost species panel).
        species_first_idx = next(
            (c for c, (_, red, t, _l) in enumerate(cols_used)
             if red == 'sum' and t == 'T_TWO_REGIME' and _rc(c)[0] == 1),
            None,
        )

        for c, (col_key, reduction, t_row, col_label) in enumerate(cols_used):
            vmin, vmax = col_ranges[col_key]
            r, cc = _rc(c)
            ax = ax_grid[r][cc]
            cax = cax_grid[r][cc]

            if reduction == 'sum_NH_rho':
                H = np.asarray(results['nh_rho_hist'])
                with np.errstate(divide='ignore'):
                    data = np.where(H > 0,
                                    np.log10(np.where(H > 0, H, 1.0)),
                                    np.nan)
                nh_xe  = np.asarray(results['nh_rho_x_edges'])
                rho_ye = np.asarray(results['nh_rho_y_edges'])
                im = ax.imshow(
                    data.T,
                    origin='lower',
                    extent=[nh_xe[0], nh_xe[-1], rho_ye[0], rho_ye[-1]],
                    aspect='auto',
                    cmap='viridis_r',
                    norm=Normalize(vmin=vmin, vmax=vmax),
                )
                ax.set_xlabel(r'$\log_{10}\,N_{\rm H}$ [cm$^{-2}$]', fontsize=10)
                ax.set_ylabel(r'$\log_{10}\,\rho$ [g cm$^{-3}$]',    fontsize=10)
            else:
                ye = y_edges[t_row]
                H = np.asarray(results['sum_hists'][f'{t_row}_{col_key}'])
                with np.errstate(divide='ignore'):
                    data = np.where(H > 0,
                                    np.log10(np.where(H > 0, H, 1.0)),
                                    np.nan)
                im = ax.imshow(
                    data.T,
                    origin='lower',
                    extent=[x_edges[0], x_edges[-1], ye[0], ye[-1]],
                    aspect='auto',
                    cmap='viridis_r',
                    norm=Normalize(vmin=vmin, vmax=vmax),
                )
                ax.set_xlim(x_edges[0], x_edges[-1])
                ax.set_ylim(ye[0], ye[-1])
                # (2026-06-18) Dashed phase-boundary horizontal lines
                # (T=2e4 / T=1e6) removed per user request.
                # y-label: Row 1 T_QK / T_DSP / T_two_regime each get their
                # own label.  Row 2 (species) labels only the leftmost panel.
                if r == 0 and t_row in ('T_QUOKKA', 'T_DESPOTIC', 'T_TWO_REGIME'):
                    label = {
                        'T_QUOKKA':     r'$\log_{10}\,T_{\rm QUOKKA}$ [K]',
                        'T_DESPOTIC':   r'$\log_{10}\,T_{\rm DESPOTIC}$ [K]',
                        'T_TWO_REGIME': r'$\log_{10}\,T_{\rm use}$ [K]',
                    }[t_row]
                    ax.set_ylabel(label, fontsize=10)
                elif r == 1 and species_first_idx is not None and c == species_first_idx:
                    ax.set_ylabel(r'$\log_{10}\,T_{\rm use}$ [K]', fontsize=10)
                # x-label: only on the bottom row (row 1).  Top row shares x
                # with bottom row (rho_x_idx) so its xticklabels are hidden too.
                if r == n_rows - 1:
                    ax.set_xlabel(r'$\log_{10}\,\rho$ [g cm$^{-3}$]', fontsize=10)
                else:
                    ax.tick_params(axis='x', labelbottom=False)

            ax.tick_params(axis='both', labelsize=8)
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=8, top=True, bottom=False,
                                labeltop=True, labelbottom=False)
            cax.set_title(col_label, fontsize=9, pad=4)

        # Hide any trailing unused panels (e.g. if user filters columns).
        for c_unused in range(n_panels, n_rows * n_cols):
            r, cc = _rc(c_unused)
            ax_grid[r][cc].set_visible(False)
            cax_grid[r][cc].set_visible(False)

        # Top-left small bold label.
        cur_lext = float(getattr(self.config, 'column_extension_lateral_kpc', 0.0))
        fig.text(0.005, 0.998,
                 f'$L_{{\\rm ext}}$ = {cur_lext:g} kpc',
                 fontsize=11, fontweight='bold', ha='left', va='top')

        out = context.config.output_dir / self.filename
        fig.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
'''
# ═══════════════════════════════════════════════════════════════════════════
# End of deprecated original code.  Nothing is exported from this module
# anymore — see phase_hist.py + phase_combined_plot.py for the current
# implementation.
# ═══════════════════════════════════════════════════════════════════════════
