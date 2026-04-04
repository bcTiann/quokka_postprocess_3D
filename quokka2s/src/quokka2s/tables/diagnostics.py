from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .models import DespoticTable

def plot_failure_overlay(*_args, **_kwargs) -> None:
    """Placeholder for failure-visualization helper."""
    raise NotImplementedError("plot_failure_overlay is not implemented yet.")

def summarize_failures(*_args, **_kwargs) -> None:
    """Placeholder for failure summary helper."""
    raise NotImplementedError("summarize_failures is not implemented yet.")

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


def plot_sampling_histogram(
        table: DespoticTable,
        samples: np.ndarray,
        *,
        ax: plt.Axes | None = None,
        cmap: str = "viridis",
        log_space: bool = True,
) -> plt.Axes:
    """Plot simulation sampling density per table cell, matching raw-table coverage plot."""
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("samples must be a 2D array with shape (N, 2)")

    logged = np.asarray(samples, dtype=float)
    if not log_space:
        with np.errstate(divide="ignore"):
            logged = np.log10(logged)

    finite_mask = np.all(np.isfinite(logged), axis=1)
    if not np.any(finite_mask):
        raise ValueError("No finite sample pairs available for histogramming.")

    nH_edges_log = _log_edges(table.nH_values)
    col_edges_log = _log_edges(table.col_density_values)

    counts, _, _ = np.histogram2d(
        logged[finite_mask, 0],
        logged[finite_mask, 1],
        bins=[nH_edges_log, col_edges_log],
    )
    counts = counts.astype(float)

    nH_edges = np.power(10.0, nH_edges_log)
    col_edges = np.power(10.0, col_edges_log)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    norm = None
    positive = counts[counts > 0]
    if positive.size:
        norm = LogNorm(vmin=positive.min(), vmax=positive.max())

    mesh = ax.pcolormesh(
        col_edges,
        nH_edges,
        counts,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Column Density (cm$^{-2}$)")
    ax.set_ylabel("n$_\\mathrm{H}$ (cm$^{-3}$)")
    ax.set_title("Simulation Sampling vs Raw Table Coverage")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Voxel count per table cell")

    if table.failure_mask is not None:
        overlay = np.ma.masked_where(~table.failure_mask, np.ones_like(table.failure_mask, dtype=float))
        ax.pcolormesh(
            col_edges,
            nH_edges,
            overlay,
            shading="auto",
            cmap=plt.cm.Greys,
            alpha=0.4,
            vmin=0,
            vmax=1,
        )

    return ax

