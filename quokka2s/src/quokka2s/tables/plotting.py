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

def _plot_panel(ax, data, title, table, cmap, show_colorbar, fig, samples=None):
    # 对齐 build_despotic_table.py 的绘图风格：对数坐标、掩蔽非正值、叠加失败遮罩
    nH_edges = np.power(10.0, _log_edges(table.nH_values))
    col_edges = np.power(10.0, _log_edges(table.col_density_values))

    invalid = ~np.isfinite(data) | (data <= 0)
    masked = np.ma.masked_array(data, mask=invalid)

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

    if table.failure_mask is not None:
        overlay = np.ma.masked_where(~table.failure_mask, np.ones_like(table.failure_mask, dtype=float))
        ax.pcolormesh(
            col_edges,
            nH_edges,
            overlay,
            shading="auto",
            cmap=plt.cm.Greys,
            alpha=0.35,
            vmin=0,
            vmax=1,
        )

    if samples is not None:
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError("samples must be shape (N, 2) of log10(nH), log10(N_col)")
        counts, _, _ = np.histogram2d(samples[:, 1], samples[:, 0], bins=[_log_edges(table.nH_values), _log_edges(table.col_density_values)])
        positive = counts[counts > 0]
        hist_norm = plt.cm.colors.LogNorm(vmin=positive.min(), vmax=positive.max()) if positive.size else None
        overlay_hist = np.ma.masked_where(counts <= 0, counts)
        ax.pcolormesh(
            col_edges,
            nH_edges,
            overlay_hist,
            shading="auto",
            cmap=plt.cm.Greys,
            alpha=0.35,
            norm=hist_norm,
        )

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
            _plot_panel(ax, data, title, table, cmap, show_colorbar, fig, samples=samples)
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
        _plot_panel(ax, data, title, table, cmap, show_colorbar, fig, samples=samples)


    # Hide any unused subplots
    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis("off")

    fig.tight_layout()
    return fig
