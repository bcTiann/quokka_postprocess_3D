"""Pure 2D accumulation with separately resolved narrow and broad Gaussians.

Every emitting row enters exactly one mu/ln(sigma) grid. Each grid uses four
neighboring shapes and its own mu spacing. No row invokes a 1D spectrum kernel.
Global node IDs concatenate the two grids; original batch merging is retained.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .geometry import GridSpec, SparseIncrement
from .band_deposit import GaussianBandEngine


class GaussianHistogramEngine:
    """Deposit into disjoint sigma bands and expand their summed spectra."""

    def __init__(self, grid, *, contraction="einsum"):
        threshold = grid.narrow_sigma_threshold
        if threshold is None or not grid.sigma_min < threshold < grid.sigma_max:
            raise ValueError("A pure 2D grid requires an interior narrow-sigma threshold")
        if grid.narrow_mu_step is None or grid.narrow_mu_step <= 0.0:
            raise ValueError("A pure 2D grid requires a positive narrow-mu spacing")
        self.grid = grid
        self.contraction = contraction
        narrow = replace(grid, sigma_max=threshold, mu_step=grid.narrow_mu_step)
        broad = replace(grid, sigma_min=threshold)
        self.engines = [
            GaussianBandEngine(narrow, contraction=contraction),
            GaussianBandEngine(broad, contraction=contraction),
        ]
        self.offsets = [0, self.engines[0].node_count]
        self.node_count = sum(engine.node_count for engine in self.engines)

    def deposit(self, centers, sigmas, luminosity):
        """Return sorted owned increments; each active input enters one grid."""
        centers = np.asarray(centers, dtype=np.float64)
        sigmas = np.asarray(sigmas, dtype=np.float64)
        luminosity = np.asarray(luminosity, dtype=np.float64)
        if centers.ndim != 1 or sigmas.shape != centers.shape:
            raise ValueError("Gaussian centers and sigmas must be equal one-dimensional arrays")
        if luminosity.ndim != 2 or luminosity.shape[0] != centers.size:
            raise ValueError("Luminosity must have shape (cell_count, profile_count)")
        if np.any(~np.isfinite(luminosity)) or np.any(luminosity < 0.0):
            raise ValueError("Physical line luminosities must be finite and nonnegative")
        active = np.any(luminosity != 0.0, axis=1)
        if np.any(active & (~np.isfinite(centers) | ~np.isfinite(sigmas))):
            raise ValueError("All emitting Gaussian parameters must be finite")
        narrow = sigmas < self.grid.narrow_sigma_threshold
        nodes = []
        weights = []
        for engine, offset, selected in zip(self.engines, self.offsets, (narrow, ~narrow)):
            selected = active & selected
            if not np.any(selected):
                continue
            increment = engine.deposit(centers[selected], sigmas[selected], luminosity[selected])
            if np.any(increment.fallback_mask):
                raise AssertionError("The pure 2D engine cannot return fallback rows")
            nodes.append(increment.nodes + offset)
            weights.append(increment.weights)
        if nodes:
            return SparseIncrement(
                nodes=np.concatenate(nodes),
                weights=np.concatenate(weights, axis=0),
                fallback_mask=np.zeros(centers.size, dtype=bool),
            )
        return SparseIncrement(
            nodes=np.empty(0, dtype=np.int64),
            weights=np.empty((0, luminosity.shape[1]), dtype=np.float64),
            fallback_mask=np.zeros(centers.size, dtype=bool),
        )

    def describe_grids(self):
        return [
            {
                "mu_nodes": engine.mu_nodes.size,
                "sigma_nodes": engine.sigma_nodes.size,
                "node_count": engine.node_count,
                "sigma_min": engine.grid.sigma_min,
                "sigma_max": engine.grid.sigma_max,
                "mu_core_step": engine.grid.mu_step,
                "mu_wing_step": engine.grid.mu_wing_step,
            }
            for engine in self.engines
        ]
