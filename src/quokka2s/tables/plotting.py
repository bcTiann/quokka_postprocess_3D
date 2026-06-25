from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="collision rates not available",
    category=UserWarning,
    module=r"DESPOTIC.*emitterData",
)
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in log",
    category=RuntimeWarning,
    module=r"DESPOTIC.*NL99_GC",
)

import math
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .models import DespoticTable

def _log_edges(values: np.ndarray) -> np.ndarray:
    if values.size < 2:
        raise ValueError("Need at least two grid points to compute edges.")
    log_values = np.log10(values)
    deltas = np.diff(log_values)
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = log_values[:-1] + deltas / 2.0
    edges[0] = log_values[0] - deltas[0] / 2.0
    edges[-1] = log_values[-1] + deltas[-1] / 2.0
    return edges


def _blocky_contour_segments(M2d, thr, nH_edges_lin, col_edges_lin):
    """Walk cell edges of `M2d` (shape n_nH × n_NH); return list of
    ((x0, y0), (x1, y1)) segments where adjacent cells differ in the
    "value >= thr" predicate.  NaN counts as not-above.  Used to draw
    staircase contours that follow the table's true cell boundaries
    (no matplotlib marching-squares interpolation)."""
    above = M2d >= thr
    segs = []
    n_nH, n_NH = above.shape
    # Vertical edges between column j and j+1 (NH-direction)
    for i in range(n_nH):
        for j in range(n_NH - 1):
            if above[i, j] != above[i, j + 1]:
                x = col_edges_lin[j + 1]
                segs.append(((x, nH_edges_lin[i]),
                             (x, nH_edges_lin[i + 1])))
    # Horizontal edges between row i and i+1 (nH-direction)
    for i in range(n_nH - 1):
        for j in range(n_NH):
            if above[i, j] != above[i + 1, j]:
                y = nH_edges_lin[i + 1]
                segs.append(((col_edges_lin[j], y),
                             (col_edges_lin[j + 1], y)))
    return segs

def _get_field_data(table: DespoticTable, token: str) -> tuple[np.ndarray, str]:
    """Retrieve data array and label for a given field token."""
    if token == "tg_final":
        return table.tg_final, "T_g (K)"
    if token == "failure_mask":
        return table.failure_mask.astype(float), "Failure Mask"
    
    if token.startswith("energy:"):
        key = token.split(":", 1)[1]
        if not table.energy_terms or key not in table.energy_terms:
            raise ValueError(f"Energy term '{key}' not found in the table.")
        return table.energy_terms[key], f"Energy Term: {key}"
    
    if token.startswith("species:"):
        _, spec, field = token.split(":")
        record = table.require_species(spec)
        if field == "abundance":
            data = record.abundance
        else:
            if record.line is None:
                raise ValueError(f"Species '{spec}' has no line data; cannot plot '{field}'")
            data = getattr(record.line, field)
        return data, f"{spec}:{field}"
        
    raise ValueError(f"Unknown field token: {token}")

DEFAULT_FIELDS: tuple[str, ...] = (
    "tg_final",
    "energy:dEdtGas",
    "energy:GammaPE",
    "species:CO:lumPerH",
    "species:C+:lumPerH",
)

def _plot_panel(ax, data, title, table, cmap, show_colorbar, fig, samples=None, dvdr_idx=0):
    # 对齐 build_despotic_table.py 的绘图风格：对数坐标、掩蔽非正值、叠加失败遮罩
    nH_edges = np.power(10.0, _log_edges(table.nH_values))
    col_edges = np.power(10.0, _log_edges(table.col_density_values))

    # 修改：将3D data切片成2D
    data_2d = data[:, :, dvdr_idx]

    invalid = ~np.isfinite(data_2d) | (data_2d <= 0)
    # FIX 2026-05-31: also mask failure_mask cells so they render as blank.
    # Without this, failure cells got the convex-filled interp colour AND a
    # grey overlay — visually a "shaded colour", not blank.  User expected
    # blank.  Now failure cells are masked out before pcolormesh and show
    # as white (transparent).
    if table.failure_mask is not None:
        invalid = invalid | table.failure_mask[:, :, dvdr_idx]
    masked = np.ma.masked_array(data_2d, mask=invalid)

    norm = None
    valid = masked.compressed()
    if valid.size:
        norm = plt.cm.colors.LogNorm(vmin=valid.min(), vmax=valid.max())

    mesh = ax.pcolormesh(
        col_edges,
        nH_edges,
        masked,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("Column Density (cm$^{-2}$)")
    ax.set_ylabel("n$_\\mathrm{H}$ (cm$^{-3}$)")

    # FIX 2026-05-31: removed the grey alpha=0.35 overlay.  failure cells are
    # now masked in `invalid` above and render as blank (white).  No overlay
    # needed; in the old code the overlay also collided visually with the
    # samples histogram (both used cmap=Greys, indistinguishable).

    if samples is not None:
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] not in (2, 3, 4):
            raise ValueError("samples must be shape (N, 2) of log10(nH), log10(N_col); "
                             "(N, 3) adds log10(dVdr) for per-slice filtering; "
                             "(N, 4) adds cell mass [g] for mass-weighted contours.")
        # Mass-weighted mode: 4th column is per-cell mass [g] → contour shows
        # log10 M_bin instead of cell count.  Adjusts legend labels too.
        weights = None
        mass_mode = (samples.shape[1] == 4)
        if samples.shape[1] >= 3:
            # Per-dVdr filtering: keep only cells whose dVdr falls in this
            # slice's bin (midpoint edges, same as the table grid binning).
            dv_edges = _log_edges(table.dVdr_values)
            lo, hi = dv_edges[dvdr_idx], dv_edges[dvdr_idx + 1]
            sel = (samples[:, 2] >= lo) & (samples[:, 2] < hi)
            if mass_mode:
                weights = samples[sel, 3]
            samples = samples[sel, :2]
        # FIX 2026-05-31: samples[:, 0] = log10 nH, samples[:, 1] = log10 NH.
        # The old call was `histogram2d(samples[:, 1], samples[:, 0],
        # bins=[nH_edges, col_edges])` — first positional was log_NH but the
        # first bin spec was nH_edges (range ~[-4.5, 6.5] vs log_NH range
        # ~[15.7, 23.3]) → every sample fell OUTSIDE the bin range → counts
        # was all zeros → grey overlay was always blank.  Bug was silent in
        # the old grey-pcolormesh path; surfaced now that we draw contours
        # (no contour visible).  Correct order: x=log_nH paired with
        # nH_edges, y=log_NH paired with col_edges.
        counts, _, _ = np.histogram2d(
            samples[:, 0], samples[:, 1],
            bins=[_log_edges(table.nH_values), _log_edges(table.col_density_values)],
            weights=weights,   # None → cell count; else per-cell mass [g] → bin mass sum
        )
        positive = counts[counts > 0]
        if positive.size:
            # 2026-06-06: switched from "percentile of bin values" (which
            # outlined the top-X% of occupied bins by value) to **mass-
            # enclosure thresholds**: each contour level T_f is chosen so
            # that the bins where M_bin >= T_f collectively contain fraction
            # f of the total mass (or total cell count in cell-count mode).
            # That makes the contour directly interpretable: "this region
            # contains 99 % of the simulation mass landing in this dVdr
            # slice".  Three nesting levels: 50 % (core), 90 % (main body),
            # 99 % (envelope of useful data).
            sorted_desc = np.sort(positive)[::-1]
            cumsum = np.cumsum(sorted_desc)
            total  = cumsum[-1]

            def _enclosure_threshold(frac):
                target = frac * total
                idx = int(np.searchsorted(cumsum, target))
                idx = min(idx, len(sorted_desc) - 1)
                return float(sorted_desc[idx])

            fractions = [0.50, 0.90, 0.99, 1.00]
            # Lower fraction ⇒ higher threshold ⇒ smaller (denser) region.
            # 100% level (added 2026-06-11) = boundary of all occupied bins;
            # this boundary is identical between cell-count and mass modes
            # since only the weight inside each bin changes, not the
            # occupied set.
            raw_levels = [_enclosure_threshold(f) for f in fractions]
            seen = set()
            level_frac_pairs = []
            for f, lvl in zip(fractions, raw_levels):
                if lvl in seen:
                    continue
                seen.add(lvl)
                level_frac_pairs.append((f, lvl))

            if level_frac_pairs:
                # 2026-06-06: blocky cell-edge staircase via LineCollection.
                # 2026-06-11: distinct colour per fraction (was all red) so
                # nested levels are visually unambiguous where they cluster.
                from matplotlib.collections import LineCollection
                nH_edges_lin_r  = np.power(10.0, _log_edges(table.nH_values))
                col_edges_lin_r = np.power(10.0, _log_edges(table.col_density_values))
                # innermost (50 %) red solid → outermost (100 %) blue dashdot
                style_by_frac = {
                    0.50: dict(color='red',        linestyle='solid',   linewidth=1.4),
                    0.90: dict(color='darkorange', linestyle='dashed',  linewidth=1.1),
                    0.99: dict(color='gold',       linestyle='dotted',  linewidth=1.1),
                    1.00: dict(color='royalblue',  linestyle='dashdot', linewidth=1.1),
                }
                for f, lvl in level_frac_pairs:
                    style = style_by_frac.get(f,
                                              dict(color='red', linestyle='solid', linewidth=1.0))
                    segs = _blocky_contour_segments(counts, lvl,
                                                    nH_edges_lin_r,
                                                    col_edges_lin_r)
                    if not segs:
                        continue
                    lc = LineCollection(segs, alpha=0.9, **style)
                    ax.add_collection(lc)
                    # Inline %-text removed 2026-06-12 — colour + linestyle +
                    # legend already disambiguate the 4 levels cleanly.
                # Legend with matching per-level colour/style.  100% level
                # shows the TOTAL signal so the user can cross-check it
                # against the dataset size (e.g. 256×256×2048 = 134,217,728
                # cells / total mass).  Other levels show the per-bin
                # threshold.
                from matplotlib.lines import Line2D
                unit = 'mass' if mass_mode else 'cells'
                def _label(f, lv):
                    if f >= 1.0:
                        if mass_mode:
                            return (f'{int(f*100)}% of {unit} '
                                    f'(total {float(total):.3e} g)')
                        return (f'{int(f*100)}% of {unit} '
                                f'(total {int(total):,} cells)')
                    if mass_mode:
                        return f'{int(f*100)}% of {unit} (≥ {lv:.1e} g / bin)'
                    return f'{int(f*100)}% of {unit} (≥ {int(lv):,} cells / bin)'
                handles = [Line2D([0], [0],
                                  label=_label(f, lv),
                                  **style_by_frac.get(f, dict(color='red',
                                      linestyle='solid', linewidth=1.0)))
                           for f, lv in level_frac_pairs]
                ax.legend(handles=handles, loc='upper right', fontsize=8,
                          framealpha=0.9,
                          title=f'sim {unit} enclosed',
                          title_fontsize=8)

    # T=100/200 K iso-temperature contours on the tg_final panel.
    # 2026-06-01 (v3): blocky cell-boundary contours instead of matplotlib's
    # marching-squares interpolated curves.  For each threshold we walk every
    # interior cell edge and draw a segment exactly where T crosses the level
    # — preserves the table's true 35×35 discretisation.  No smoothing.
    if title.startswith("T_g"):
        from matplotlib.collections import LineCollection
        log_nH_e  = _log_edges(table.nH_values)
        log_col_e = _log_edges(table.col_density_values)
        nH_edges_lin  = np.power(10.0, log_nH_e)        # length n_nH + 1
        col_edges_lin = np.power(10.0, log_col_e)
        T_grid = np.where(np.isfinite(data_2d) & (data_2d > 0), data_2d, np.nan)

        finite_T = T_grid[np.isfinite(T_grid)]
        if finite_T.size:
            tmin, tmax = float(finite_T.min()), float(finite_T.max())
            spec = [(100.0, 'solid', 1.0), (200.0, 'dashed', 1.0)]
            for thr, ls, lw in spec:
                if not (tmin <= thr <= tmax):
                    continue
                segs = _blocky_contour_segments(T_grid, thr,
                                                nH_edges_lin, col_edges_lin)
                if not segs:
                    continue
                lc = LineCollection(segs, colors='black',
                                    linewidths=lw, linestyles=ls)
                ax.add_collection(lc)
                # Label: place a small text annotation at the centroid of
                # the longest segment (rough but readable).
                lengths = [(s[1][0] - s[0][0])**2 + (s[1][1] - s[0][1])**2
                           for s in segs]
                k = int(np.argmax(lengths))
                xmid = 0.5 * (segs[k][0][0] + segs[k][1][0])
                ymid = 0.5 * (segs[k][0][1] + segs[k][1][1])
                ax.text(xmid, ymid, f'{int(thr)} K', fontsize=7,
                        color='black', ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white',
                                  ec='none', alpha=0.6))

    if show_colorbar:
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(title)

    

def plot_table_overview(
    table: DespoticTable,
    *,
    fields: Sequence[str] | None = None,
    ncols: int = 3,
    figsize: tuple[float, float] = (15, 10),
    cmap: str = "viridis",
    show_colorbar: bool = True,
    separate: bool = False,
    samples: np.ndarray | None = None,
    dvdr_idx: int = 0,  # index into the dVdr axis (3rd table dimension)
) -> plt.Figure | list[plt.Figure]:
    """Plot an overview of selected fields from a DespoticTable.

    Parameters:
        table: DespoticTable
            The DESPOTIC table to plot.
        fields: Sequence[str] | None
            List of field tokens to plot. If None, defaults are used.
        ncols: int
            Number of columns in the subplot grid.
        figsize: tuple[float, float]
            Size of the figure.
        cmap: str
            Colormap to use for the plots.
        show_colorbar: bool
            Whether to display colorbars for each subplot.

    Returns:
        plt.Figure
            The matplotlib Figure object containing the plots.
    """

    tokens = list(fields) if fields is not None else list(DEFAULT_FIELDS)
    if not tokens:
        raise ValueError("At least one field token must be specified for plotting.")
    if separate:
        figs = []
        for token in tokens:
            data, title = _get_field_data(table, token)
            fig, ax = plt.subplots(figsize=figsize)
            _plot_panel(ax, data, title, table, cmap, show_colorbar, fig, samples=samples, dvdr_idx=dvdr_idx)
            figs.append(fig)
        return figs

    n_panels = len(tokens)
    ncols = max(1, ncols)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)

    log_nH = [f"{v:1e}" for v in table.nH_values]
    log_colDen = [f"{v:1e}" for v in table.col_density_values]

    for idx, token in enumerate(tokens):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        data, title = _get_field_data(table, token)
        _plot_panel(ax, data, title, table, cmap, show_colorbar, fig, samples=samples, dvdr_idx=dvdr_idx)


    # Hide any unused subplots
    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis("off")

    fig.tight_layout()
    return fig
