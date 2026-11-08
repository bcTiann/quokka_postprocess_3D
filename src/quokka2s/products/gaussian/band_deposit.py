"""exact-index replacement for the established compact histogram deposit.

Only bracket search and repeated immutable-axis validation change. The masks,
NumPy logarithms, corner-major scatter, sorted node output and weights are the
established calculations. No dense per-worker histogram or new threads are added.
"""
from __future__ import annotations

import threading

import numpy as np

from ._histogram_deposit import AxisIndexer, Workspace
from .geometry import BandGeometry, SparseIncrement


_thread_workspaces = threading.local()
_geometry_lock = threading.Lock()


def engine_axis_indexers(engine):
    """Validate the engine's original read-only axes once and retain geometry."""
    existing = getattr(engine, "_indexed_axis_geometry", None)
    if existing is not None:
        return existing
    with _geometry_lock:
        existing = getattr(engine, "_indexed_axis_geometry", None)
        if existing is None:
            existing = initialize_axis_indexers(engine)
            engine._indexed_axis_geometry = existing
    return existing


def initialize_axis_indexers(engine):
    """Construct descriptors while excluding other first-use worker threads."""
    grid = engine.grid
    if grid.mu_core_min is None:
        mu_starts = [0]
        mu_steps = [grid.mu_step]
    else:
        core_bounds = np.asarray([grid.mu_core_min, grid.mu_core_max], dtype=np.float64)
        core_indices = np.searchsorted(engine.mu_nodes, core_bounds, side="left")
        if not np.array_equal(engine.mu_nodes[core_indices], core_bounds):
            raise ValueError("The refined mu boundaries must be exact axis nodes")
        mu_starts = [0, int(core_indices[0]), int(core_indices[1])]
        mu_steps = [grid.mu_wing_step, grid.mu_step, grid.mu_wing_step]
    mu_indexer = AxisIndexer(engine.mu_nodes, mu_starts, mu_steps)
    sigma_indexer = AxisIndexer(engine.log_sigma_nodes, [0], [grid.log_sigma_step])
    return mu_indexer, sigma_indexer


def get_indexed_workspace(node_count, max_profiles=12, *, indexers):
    """Reuse the established compact scratch structure per thread and exact geometry."""
    if not hasattr(_thread_workspaces, "grids"):
        _thread_workspaces.grids = {}
    mu_indexer, sigma_indexer = indexers
    key = (node_count, max_profiles, id(mu_indexer), id(sigma_indexer))
    workspace = _thread_workspaces.grids.get(key)
    if workspace is None:
        workspace = Workspace(
            node_count=node_count,
            max_profiles=max_profiles,
            mu_indexer=mu_indexer,
            sigma_indexer=sigma_indexer,
        )
        _thread_workspaces.grids[key] = workspace
    return workspace


def indexed_deposit(self, centers, sigmas, luminosity):
    """Return the established owned sparse increment using indexed native brackets."""
    centers = np.asarray(centers, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    luminosity = np.asarray(luminosity, dtype=np.float64)
    if luminosity.ndim == 2 and luminosity.shape[1] > 12:
        raise ValueError('The adopted ten-line width groups support at most twelve profiles')
    if centers.ndim != 1 or sigmas.shape != centers.shape:
        raise ValueError('centers and sigmas must have the same one-dimensional shape')
    if luminosity.ndim != 2 or luminosity.shape[0] != centers.size:
        raise ValueError('luminosity must have shape (cell_count, profile_count)')
    if not np.all(np.isfinite(luminosity)) or np.any(luminosity < 0.0):
        raise ValueError('Prepare per-line masks first: luminosity must be finite and nonnegative')

    active = np.any(luminosity != 0.0, axis=1)
    inside = (
        np.isfinite(centers)
        & np.isfinite(sigmas)
        & (centers >= self.grid.mu_min)
        & (centers <= self.grid.mu_max)
        & (sigmas >= self.grid.sigma_min)
        & (sigmas <= self.grid.sigma_max)
    )
    fallback_mask = active & ~inside
    if np.any(fallback_mask):
        outside_centers = centers[fallback_mask]
        outside_sigmas = sigmas[fallback_mask]
        raise ValueError(
            'Pure 2D requires every emitting Gaussian inside its grid; '
            f'outside rows={outside_centers.size}, '
            f'mu=[{outside_centers.min()}, {outside_centers.max()}], '
            f'sigma=[{outside_sigmas.min()}, {outside_sigmas.max()}]'
        )
    selected_rows = np.flatnonzero(active & inside)
    profile_count = luminosity.shape[1]
    if selected_rows.size == 0:
        return SparseIncrement(
            nodes=np.empty(0, dtype=np.int64),
            weights=np.empty((0, profile_count), dtype=np.float64),
            fallback_mask=fallback_mask,
        )

    workspace = get_indexed_workspace(
        self.node_count,
        max_profiles=12,
        indexers=engine_axis_indexers(self),
    )
    nodes, weights = workspace.scatter_parameters(
        rows=selected_rows,
        centers=centers[selected_rows],
        log_sigmas=np.log(sigmas[selected_rows]),
        mu_axis=self.mu_nodes,
        log_sigma_axis=self.log_sigma_nodes,
        luminosity=luminosity,
    )
    if not np.all(np.isfinite(weights)):
        raise FloatingPointError('Histogram luminosity accumulation overflowed')
    return SparseIncrement(nodes=nodes, weights=weights, fallback_mask=fallback_mask)



class GaussianBandEngine(BandGeometry):
    """Use the original indexed corner-major scatter on one sigma band."""

    deposit = indexed_deposit
