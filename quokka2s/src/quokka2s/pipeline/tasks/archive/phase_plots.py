"""PhasePlotTask: 10-panel ρ-T phase diagrams weighted by mass and 4 luminosities.

Layout (2 rows × 5 columns):
    Row 0 (Y = log10 T_QUOKKA):     mass | CO | C+ | Hα | HI
    Row 1 (Y = log10 T_DESPOTIC):   mass | CO | C+ | Hα | HI

Each panel is a 2D histogram of (log10 ρ, log10 T) where each cell contributes
its weight to the bin it falls in.  Weights are physical cell totals (not
densities):
    mass        = ρ × dV               [g]
    L_species   = ε_species × dV       [erg/s]

The colour shows log10(Σ weight per bin), so brighter bins hold more mass
(or more luminosity) than darker ones.  Within a column (same weight) the
two T-rows share the colour scale so T_QUOKKA vs T_DESPOTIC are directly
comparable.

Output: phase_plots.png
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..base import AnalysisTask, PipelinePlotContext


# (weight_key, column_label) — label uses log10 prefix because the colourbar
# below shows the log10 of the binned weight (so the tick "26" means 10^26).
_WEIGHTS = [
    ('mass',    r'$\log_{10}\,M_{\rm bin}$ [g]'),
    ('CO',      r'$\log_{10}\,L_{\rm CO}$ [erg s$^{-1}$]'),
    ('Cplus',   r'$\log_{10}\,L_{\rm C^+}$ [erg s$^{-1}$]'),
    ('Halpha',  r'$\log_{10}\,L_{\rm H\alpha}$ [erg s$^{-1}$]'),
    ('HI',      r'$\log_{10}\,L_{\rm HI}$ [erg s$^{-1}$]'),
]

# (row_key, label, y_edges_key, y_label)
_ROWS = [
    ('T_QUOKKA',   r'$\log_{10}\,T_{\rm QUOKKA}$ [K]',   'y_qk_edges',  r'$\log_{10}\,T_{\rm QUOKKA}$ [K]'),
    ('T_DESPOTIC', r'$\log_{10}\,T_{\rm DESPOTIC}$ [K]', 'y_dsp_edges', r'$\log_{10}\,T_{\rm DESPOTIC}$ [K]'),
]


class PhasePlotTask(AnalysisTask):
    """10-panel ρ–T phase plot (T_QUOKKA & T_DESPOTIC × {mass, CO, C+, Hα, HI})."""

    # Phase boundaries marked with dashed lines on every panel.
    # T=2e4 K = cool/warm boundary,  T=1e6 K = warm/hot boundary
    # (same convention as Tian+ outflow phase diagrams).
    _PHASE_BOUNDARIES_LOG_T = (np.log10(2.0e4), np.log10(1.0e6))

    # Colour scale spans this many decades down from the per-column max.
    _COLOR_DEX = 4.0

    def __init__(self, config,
                 bin_dex: float = 0.2,
                 filename: str = 'phase_plots.png'):
        super().__init__(config)
        self.bin_dex = float(bin_dex)
        self.filename = filename

    def compute(self, context: PipelinePlotContext) -> dict:
        p = context.provider

        # Load all needed fields.  cgs values + ravel to 1D for histogram2d.
        rho_u,   _ = p.get_slab_z(('gas', 'density'))
        T_qk_u,  _ = p.get_slab_z(('gas', 'temperature_quokka'))
        T_dsp_u, _ = p.get_slab_z(('gas', 'temperature_despotic'))
        CO_u,    _ = p.get_slab_z(('gas', 'CO_luminosity'))
        Cp_u,    _ = p.get_slab_z(('gas', 'C+_luminosity'))
        Ha_u,    _ = p.get_slab_z(('gas', 'H_alpha_luminosity'))
        HI_u,    _ = p.get_slab_z(('gas', 'HI_luminosity'))
        dx_u,    _ = p.get_slab_z(('boxlib', 'dx'))
        dy_u,    _ = p.get_slab_z(('boxlib', 'dy'))
        dz_u,    _ = p.get_slab_z(('boxlib', 'dz'))

        rho   = np.asarray(rho_u.in_cgs()).ravel()
        T_qk  = np.asarray(T_qk_u.in_cgs()).ravel()
        T_dsp = np.asarray(T_dsp_u.in_cgs()).ravel()
        CO    = np.asarray(CO_u.in_cgs()).ravel()
        Cp    = np.asarray(Cp_u.in_cgs()).ravel()
        Ha    = np.asarray(Ha_u.in_cgs()).ravel()
        HI    = np.asarray(HI_u.in_cgs()).ravel()
        dV    = (np.asarray(dx_u.in_cgs()) *
                 np.asarray(dy_u.in_cgs()) *
                 np.asarray(dz_u.in_cgs())).ravel()
        del rho_u, T_qk_u, T_dsp_u, CO_u, Cp_u, Ha_u, HI_u, dx_u, dy_u, dz_u

        # log axes — clamp non-positive to NaN so they fall outside the bin range.
        with np.errstate(divide='ignore', invalid='ignore'):
            log_rho  = np.log10(np.where(rho   > 0, rho,   np.nan))
            log_T_qk = np.log10(np.where(T_qk  > 0, T_qk,  np.nan))
            log_T_dsp= np.log10(np.where(T_dsp > 0, T_dsp, np.nan))

        # Bin edges cover the FULL data range (nanmin → nanmax with a tiny
        # right-edge pad so nanmax lands strictly inside the rightmost bin).
        # This way every cell with finite log_ρ AND finite log_T is captured,
        # which lets us assert mass/luminosity conservation below.  Y axis is
        # SHARED between T_QUOKKA and T_DESPOTIC rows (union of their data
        # ranges) so within a column you can read off vertically where T_DSP
        # sits relative to T_QK.
        rho_lo, rho_hi  = float(np.nanmin(log_rho)),  float(np.nanmax(log_rho))
        qk_lo,  qk_hi   = float(np.nanmin(log_T_qk)), float(np.nanmax(log_T_qk))
        dsp_lo, dsp_hi  = float(np.nanmin(log_T_dsp)),float(np.nanmax(log_T_dsp))
        y_lo = min(qk_lo, dsp_lo)
        y_hi = max(qk_hi, dsp_hi)
        # Fixed-width bins in log space (0.2 dex by default), Tian+ style.
        # We snap the lower edge DOWN and upper edge UP to the next bin
        # boundary so every cell still falls inside a bin (conservation).
        def _aligned_edges(lo: float, hi: float, step: float) -> np.ndarray:
            lo_snap = np.floor(lo / step) * step
            # add 0.5*step so the rightmost edge sits strictly above the max
            hi_snap = (np.ceil(hi / step) + 0.5) * step
            n_bins = int(np.round((hi_snap - lo_snap) / step))
            return np.linspace(lo_snap, hi_snap, n_bins + 1)

        x_edges     = _aligned_edges(rho_lo, rho_hi, self.bin_dex)
        y_qk_edges  = _aligned_edges(y_lo,   y_hi,   self.bin_dex)
        y_dsp_edges = y_qk_edges.copy()

        # Per-cell weights — mass + 4 luminosities, all × dV so each is a total
        # (g for mass, erg/s for luminosities).
        weights = {
            'mass':   rho * dV,
            'CO':     CO  * dV,
            'Cplus':  Cp  * dV,
            'Halpha': Ha  * dV,
            'HI':     HI  * dV,
        }
        # The big arrays are no longer needed after weights are formed.
        del rho, CO, Cp, Ha, HI, dV

        # Mask of cells with finite log axes (i.e. positive ρ and T).  Cells
        # outside this mask are dropped by histogram2d, so they're also the
        # cells we exclude from the reference total.
        valid_qk  = np.isfinite(log_rho) & np.isfinite(log_T_qk)
        valid_dsp = np.isfinite(log_rho) & np.isfinite(log_T_dsp)

        histograms: dict[str, np.ndarray] = {}
        for w_key, w in weights.items():
            H_qk,  _, _ = np.histogram2d(log_rho, log_T_qk,  bins=[x_edges, y_qk_edges],  weights=w)
            H_dsp, _, _ = np.histogram2d(log_rho, log_T_dsp, bins=[x_edges, y_dsp_edges], weights=w)
            histograms[f'T_QUOKKA_{w_key}']   = H_qk
            histograms[f'T_DESPOTIC_{w_key}'] = H_dsp

            # Conservation check: every cell with finite log axes contributes
            # its weight to one bin, so Σ(hist) must equal Σ(weight) over
            # those same cells, up to float-summation rounding.
            for row_key, valid, H in (
                ('T_QUOKKA',   valid_qk,  H_qk),
                ('T_DESPOTIC', valid_dsp, H_dsp),
            ):
                expected = float(np.nansum(w[valid]))
                actual   = float(H.sum())
                denom    = max(abs(expected), 1e-300)
                rel_err  = abs(actual - expected) / denom
                # Float pairwise summation introduces ULP-level drift; 1e-8
                # relative tolerance is generous against that and still catches
                # any real mass/luminosity loss.
                assert rel_err < 1e-8, (
                    f'[{row_key} / {w_key}] histogram sum {actual:.6e} disagrees '
                    f'with cell sum {expected:.6e} (rel_err={rel_err:.2e}).  '
                    f'Probable cause: bin edges miss extreme cells.'
                )
                print(f'  [conservation OK] {row_key:<10s} / {w_key:<7s}: '
                      f'sum={actual:.4e}  rel_err={rel_err:.1e}')

        return {
            'histograms':  histograms,
            'x_edges':     x_edges,
            'y_qk_edges':  y_qk_edges,
            'y_dsp_edges': y_dsp_edges,
        }

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        histograms  = results['histograms']
        x_edges     = results['x_edges']
        y_edges = {
            'T_QUOKKA':   results['y_qk_edges'],
            'T_DESPOTIC': results['y_dsp_edges'],
        }

        n_rows = len(_ROWS)
        n_cols = len(_WEIGHTS)

        # Three-row gridspec: a thin row for colourbars on top, then two equal
        # plot rows.  Splitting the cbar into its own gridspec row guarantees
        # the two T rows have IDENTICAL height (no make_axes_locatable carving
        # from row 0 only, which made row 0 panels shorter than row 1).
        fig = plt.figure(figsize=(2.8 * n_cols, 3.0 * n_rows))
        gs = fig.add_gridspec(
            3, n_cols,
            height_ratios=[0.04, 1.0, 1.0],
            hspace=0.35, wspace=0.30,
            left=0.06, right=0.98, top=0.92, bottom=0.10,
        )
        cax_row = [fig.add_subplot(gs[0, c]) for c in range(n_cols)]
        axes = np.array([
            [fig.add_subplot(gs[1, c]) for c in range(n_cols)],
            [fig.add_subplot(gs[2, c],
                              sharex=None,
                              sharey=None) for c in range(n_cols)],
        ])
        # Link X across all panels and Y across all panels.
        for c in range(n_cols):
            axes[1, c].sharex(axes[0, c])
            axes[1, c].sharey(axes[0, c])
        for c in range(1, n_cols):
            axes[0, c].sharex(axes[0, 0])
            axes[0, c].sharey(axes[0, 0])

        # For each column, share vmin/vmax across the two rows so T_QK vs T_DSP
        # is directly comparable per weight type.  Tian+ convention: colour
        # spans `_COLOR_DEX` decades down from the column's per-column max.
        #
        # We plot log10(H) with a linear Normalize so the colourbar tick labels
        # ARE the log values themselves (a tick "26" means 10^26).
        for c, (w_key, col_label) in enumerate(_WEIGHTS):
            H_combined = []
            for row_key, *_rest in _ROWS:
                H = histograms[f'{row_key}_{w_key}']
                H_combined.append(H[H > 0])
            H_combined = np.concatenate(H_combined) if H_combined else np.array([1.0])
            if H_combined.size == 0:
                log_vmax = 0.0
            else:
                log_vmax = float(np.log10(np.nanmax(H_combined)))
            log_vmin = log_vmax - self._COLOR_DEX

            im = None
            for r, (row_key, _row_label, _y_key, y_axis_label) in enumerate(_ROWS):
                ax = axes[r, c]
                H = histograms[f'{row_key}_{w_key}']
                ye = y_edges[row_key]
                # log10 of bin sums; empty bins (0) become NaN so they're drawn
                # in the cmap's bad-value colour (default white background).
                with np.errstate(divide='ignore'):
                    log_H = np.where(H > 0, np.log10(np.where(H > 0, H, 1.0)), np.nan)
                im = ax.imshow(
                    log_H.T,
                    origin='lower',
                    extent=[x_edges[0], x_edges[-1], ye[0], ye[-1]],
                    aspect='auto',
                    cmap='viridis_r',
                    norm=Normalize(vmin=log_vmin, vmax=log_vmax),
                )
                ax.tick_params(axis='both', labelsize=8)
                # Belt-and-suspenders: explicitly pin axis limits so any
                # auto-rescaling can't sneak in.
                ax.set_xlim(x_edges[0], x_edges[-1])
                ax.set_ylim(ye[0], ye[-1])
                # Phase boundaries: T = 2e4 K and T = 1e6 K horizontal lines.
                for log_T_bnd in self._PHASE_BOUNDARIES_LOG_T:
                    if ye[0] <= log_T_bnd <= ye[-1]:
                        ax.axhline(log_T_bnd, color='gray', ls='--', lw=0.6, alpha=0.7)
                if c == 0:
                    ax.set_ylabel(y_axis_label, fontsize=10)
                if r == n_rows - 1:
                    ax.set_xlabel(r'$\log_{10}\,\rho$ [g cm$^{-3}$]', fontsize=10)
                # Hide x-axis tick labels on the top T_QK row (shared X with bottom).
                if r == 0:
                    ax.tick_params(labelbottom=False)

            cbar = fig.colorbar(im, cax=cax_row[c], orientation='horizontal')
            cbar.ax.tick_params(labelsize=8, top=True, bottom=False,
                                labeltop=True, labelbottom=False)
            cax_row[c].set_title(col_label, fontsize=9, pad=4)

        out = context.config.output_dir / self.filename
        fig.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
