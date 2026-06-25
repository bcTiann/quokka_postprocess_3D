"""Plot_PhaseCombined — assemble phase_combined.png from the Build_PhaseHist*
results (Build_PhaseHist ×7, keyed by tag, + Build_PhaseHistNHRho).

A pure Plot task: it computes nothing and reads the Build results fresh from
disk at plot time (see ``_gather_inputs``).  The Build tasks must run first
(``--mode compute``).  The colorbar UNIT comes from each Build result's
yt-derived ``unit_latex`` (not hardcoded); the ``symbol`` gives the quantity.

Layout (2×4) — see `_LAYOUT` below:

   Row 1:  mass × T_QK | mass × T_DSP | mass × T_use | NH-ρ
   Row 2:  CO          | C+           | Hα           | HI    (all y = T_use)

Row 2 species panels share y (T_use) with Row 1 col 3.  Mass columns
share their colorbar range (pooled across the 3 T panels).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ..base import PlotTask, PipelinePlotContext


# Each entry:  (tag, row_y_label_for_top_row, colorbar_group)
#   tag                : matches the `tag` field on the sibling intermediate
#   row_y_label_for... : LaTeX y-axis label when r==0 (Row 1); ignored when r==1
#                        (Row 2 species share a single label via species_first_idx)
#   colorbar_group     : panels in the SAME group share a vmin/vmax pooled
#                        from their H matrices
_LAYOUT = [
    # Row 1
    ('mass_T_QK',   r'$\log_{10}\,T_{\rm QUOKKA}$ [K]',     'mass'),
    ('mass_T_DSP',  r'$\log_{10}\,T_{\rm DESPOTIC}$ [K]',   'mass'),
    ('mass_T_2R',   r'$\log_{10}\,T_{\rm two-regime}$ [K]', 'mass'),
    ('NH_rho',      None,                                   'NH_rho'),
    # Row 2 — all y = T_two_regime, each its own colorbar group
    ('CO_T_2R',     None, 'CO'),
    ('Cplus_T_2R',  None, 'Cplus'),
    ('Halpha_T_2R', None, 'Halpha'),
    ('HI_T_2R',     None, 'HI'),
]


def _coerce_str(value) -> str:
    """h5py returns bytes for str attributes; coerce back to native str."""
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _unit_latex(unit_str: str) -> str:
    """A yt unit NAME (e.g. 'erg/s', 'g') → LaTeX for the colorbar label.

    The unit is rendered to LaTeX here, at plot time (display); the Build tasks
    store only the weight's natural unit STRING (data).  Falls back to the plain
    name if rendering fails or is empty (e.g. dimensionless)."""
    if not unit_str:
        return ''
    try:
        from unyt import Unit
        return Unit(unit_str).latex_repr or unit_str
    except Exception:
        return unit_str


# tag → LaTeX symbol for the colorbar.  This is DISPLAY config and lives here in
# the Plot task, not on the Build task; the data's unit comes from each Build
# result's `weight_unit` and is rendered by `_unit_latex` above.
_SYMBOL = {
    'mass_T_QK':   r'M_{\rm bin}',
    'mass_T_DSP':  r'M_{\rm bin}',
    'mass_T_2R':   r'M_{\rm bin}',
    'NH_rho':      r'M_{\rm bin}',
    'CO_T_2R':     r'L_{\rm CO}',
    'Cplus_T_2R':  r'L_{\rm C^+}',
    'Halpha_T_2R': r'L_{\rm H\alpha}',
    'HI_T_2R':     r'L_{\rm HI}',
}


class Plot_PhaseCombined(PlotTask):
    """Assemble phase_combined.png from the Build_PhaseHist* results."""

    def __init__(self, config, filename: str = 'phase_combined.png'):
        super().__init__(config, name='Plot_PhaseCombined')
        self.filename = filename

    def _gather_inputs(self, context: PipelinePlotContext) -> dict:
        """Load the Build_PhaseHist (×N, keyed by tag) + Build_PhaseHistNHRho
        results fresh from disk (cache-key validated by the loaders)."""
        panels: dict[str, dict] = {}
        for data in self._load_all(context, 'Build_PhaseHist'):
            tag = _coerce_str(data.get('tag', ''))
            if tag:
                panels[tag] = data
        panels['NH_rho'] = self._load_one(context, 'Build_PhaseHistNHRho')

        expected = [tag for tag, _, _ in _LAYOUT]
        missing  = [t for t in expected if t not in panels]
        if missing:
            raise RuntimeError(
                f'Plot_PhaseCombined: missing Build results for tags {missing}. '
                f'Run the Build_PhaseHist / Build_PhaseHistNHRho tasks first '
                f'(--mode compute).'
            )
        return panels

    def plot(self, context: PipelinePlotContext, panels: dict) -> None:
        # ── Pool colorbar ranges per group (full data range, no fixed dex) ──
        group_ranges: dict[str, tuple[float, float]] = {}
        for tag, _, group in _LAYOUT:
            H = np.asarray(panels[tag]['H'])
            pos = H[H > 0]
            if pos.size == 0:
                continue
            lo = float(np.log10(np.nanmin(pos)))
            hi = float(np.log10(np.nanmax(pos)))
            if group in group_ranges:
                g_lo, g_hi = group_ranges[group]
                group_ranges[group] = (min(g_lo, lo), max(g_hi, hi))
            else:
                group_ranges[group] = (lo, hi)

        # ── Figure + nested GridSpec ──
        n_panels = len(_LAYOUT)
        n_cols   = 4
        n_rows   = (n_panels + n_cols - 1) // n_cols
        panel_inch = 3.4
        fig = plt.figure(figsize=(panel_inch * n_cols + 0.4,
                                  panel_inch * n_rows + 1.6))
        outer = fig.add_gridspec(
            n_rows, 1,
            hspace=0.40,
            left=0.05, right=0.99, top=0.93, bottom=0.07,
        )
        cax_grid: list[list] = []
        ax_grid:  list[list] = []
        for r in range(n_rows):
            inner = outer[r].subgridspec(
                2, n_cols,
                height_ratios=[0.06, 1.0],
                hspace=0.05,
                wspace=0.30,
            )
            cax_grid.append([fig.add_subplot(inner[0, c]) for c in range(n_cols)])
            ax_grid.append( [fig.add_subplot(inner[1, c]) for c in range(n_cols)])

        def _rc(idx: int) -> tuple[int, int]:
            return idx // n_cols, idx % n_cols

        # ── Sharex/sharey ──
        # All ρ-axis panels share x (everything except NH-ρ).
        rho_axis_idx = [i for i, (tag, _, _) in enumerate(_LAYOUT) if tag != 'NH_rho']
        if rho_axis_idx:
            base = rho_axis_idx[0]
            r0, c0 = _rc(base)
            for i in rho_axis_idx:
                if i == base:
                    continue
                r, c = _rc(i)
                ax_grid[r][c].sharex(ax_grid[r0][c0])

        # All T_two_regime panels share y (Row 1 col 3 + Row 2 cols 1-4).
        T_2R_idx = [i for i, (tag, _, _) in enumerate(_LAYOUT)
                    if tag.endswith('T_2R')]
        if T_2R_idx:
            base = T_2R_idx[0]
            r0, c0 = _rc(base)
            for i in T_2R_idx:
                if i == base:
                    continue
                r, c = _rc(i)
                ax_grid[r][c].sharey(ax_grid[r0][c0])

        # All Row-2 species panels get a y-label (user 2026-06-20 — was
        # previously only the leftmost).
        species_idx_row2 = {
            i for i, (tag, _, _) in enumerate(_LAYOUT)
            if tag.endswith('T_2R') and _rc(i)[0] == 1
        }

        # ── Compute union of bin-edge extents per axis-share-group ──
        # Each PhaseHistTask uses its own data-derived bin edges (tight to
        # its data).  When multiple panels share an axis (sharex / sharey),
        # we set xlim/ylim to the UNION of their edge ranges so all data
        # is visible and panels align visually.
        def _x_extent(tag: str) -> tuple[float, float]:
            xe = np.asarray(panels[tag]['x_edges'])
            return float(xe[0]), float(xe[-1])

        def _y_extent(tag: str) -> tuple[float, float]:
            ye = np.asarray(panels[tag]['y_edges'])
            return float(ye[0]), float(ye[-1])

        def _union(extents: list[tuple[float, float]]) -> tuple[float, float] | None:
            if not extents:
                return None
            return min(e[0] for e in extents), max(e[1] for e in extents)

        # Union ρ extent across all ρ-axis panels (everything except NH-ρ).
        rho_x_extent = _union([_x_extent(tag) for tag, _, _ in _LAYOUT if tag != 'NH_rho'])

        # Per-y-group extents (T_QK / T_DSP / T_2R each own a y axis).
        y_QK_extent  = _y_extent('mass_T_QK')
        y_DSP_extent = _y_extent('mass_T_DSP')
        y_2R_extent  = _union([_y_extent(tag) for tag, _, _ in _LAYOUT if tag.endswith('T_2R')])

        # NH-ρ panel: own (N_H, ρ) axes.
        nh_x_extent  = _x_extent('NH_rho')
        nh_y_extent  = _y_extent('NH_rho')

        # ── Render each panel ──
        for i, (tag, y_label, group) in enumerate(_LAYOUT):
            data_dict = panels[tag]
            H  = np.asarray(data_dict['H'])
            xe = np.asarray(data_dict['x_edges'])
            ye = np.asarray(data_dict['y_edges'])
            symbol     = _SYMBOL.get(tag, '')                       # display: from layout
            unit_latex = _unit_latex(_coerce_str(data_dict.get('weight_unit', '')))
            cbar_label = (rf'$\log_{{10}}\,{symbol}$  [${unit_latex}$]'
                          if unit_latex else rf'$\log_{{10}}\,{symbol}$')

            r, c = _rc(i)
            ax  = ax_grid[r][c]
            cax = cax_grid[r][c]

            vmin, vmax = group_ranges.get(group, (0.0, 1.0))

            with np.errstate(divide='ignore'):
                view = np.where(H > 0,
                                np.log10(np.where(H > 0, H, 1.0)),
                                np.nan)
            im = ax.imshow(
                view.T,
                origin='lower',
                extent=[xe[0], xe[-1], ye[0], ye[-1]],
                aspect='auto',
                cmap='viridis_r',
                norm=Normalize(vmin=vmin, vmax=vmax),
            )

            # Auto-crop to the non-empty extent for THIS panel's axis group.
            if tag == 'NH_rho':
                xlim = nh_x_extent or (xe[0], xe[-1])
                ylim = nh_y_extent or (ye[0], ye[-1])
            else:
                xlim = rho_x_extent or (xe[0], xe[-1])
                if   tag == 'mass_T_QK':  ylim = y_QK_extent  or (ye[0], ye[-1])
                elif tag == 'mass_T_DSP': ylim = y_DSP_extent or (ye[0], ye[-1])
                else:                     ylim = y_2R_extent  or (ye[0], ye[-1])
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            if tag == 'NH_rho':
                ax.set_xlabel(r'$\log_{10}\,N_{\rm H}$ [cm$^{-2}$]', fontsize=10)
                ax.set_ylabel(r'$\log_{10}\,\rho$ [g cm$^{-3}$]', fontsize=10)
            else:
                if r == 0 and y_label is not None:
                    ax.set_ylabel(y_label, fontsize=10)
                elif r == 1 and i in species_idx_row2:
                    ax.set_ylabel(r'$\log_{10}\,T_{\rm two-regime}$ [K]', fontsize=10)
                # Every ρ panel gets its own x-axis label + ticks — including
                # the top row (user 2026-06-24: the top mass×T panels were
                # unlabeled because of sharex, which looked inconsistent next to
                # the labeled NH-ρ panel).
                ax.set_xlabel(r'$\log_{10}\,\rho$ [g cm$^{-3}$]', fontsize=10)
                ax.tick_params(axis='x', labelbottom=True)

            ax.tick_params(axis='both', labelsize=8)
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=8, top=True, bottom=False,
                                labeltop=True, labelbottom=False)
            cax.set_title(cbar_label, fontsize=9, pad=4)

        # L_ext label, top-left.
        cur_lext = float(getattr(self.config, 'column_extension_lateral_kpc', 0.0))
        fig.text(0.005, 0.998,
                 f'$L_{{\\rm ext}}$ = {cur_lext:g} kpc',
                 fontsize=11, fontweight='bold', ha='left', va='top')

        out = self.config.output_dir / self.filename
        fig.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
