"""Deposit one Gaussian band with NumPy brackets and corner-major scatter."""
from __future__ import annotations

import numpy as np

from .geometry import BandGeometry, SparseIncrement


class GaussianBandEngine(BandGeometry):
    """Distribute each selected luminosity to its four parameter neighbours."""

    def deposit(self, centers, sigmas, luminosity):
        """Return sorted owned increments using four fixed NumPy scatter passes."""
        centers = np.asarray(centers, dtype=np.float64)
        sigmas = np.asarray(sigmas, dtype=np.float64)
        luminosity = np.asarray(luminosity, dtype=np.float64)
        if centers.ndim != 1 or sigmas.shape != centers.shape:
            raise ValueError("centers and sigmas must have the same one-dimensional shape")
        if luminosity.ndim != 2 or luminosity.shape[0] != centers.size:
            raise ValueError("luminosity must have shape (cell_count, profile_count)")
        if luminosity.shape[1] > 12:
            raise ValueError("The adopted width groups support at most twelve profiles")
        if not np.isfinite(luminosity).all() or np.any(luminosity < 0.0):
            raise ValueError("Prepare per-line masks first: luminosity must be finite and nonnegative")
        active = np.any(luminosity != 0.0, axis=1)
        inside = (
            np.isfinite(centers)
            & np.isfinite(sigmas)
            & (centers >= self.grid.mu_min)
            & (centers <= self.grid.mu_max)
            & (sigmas >= self.grid.sigma_min)
            & (sigmas <= self.grid.sigma_max)
        )
        outside = active & ~inside
        if np.any(outside):
            raise ValueError(
                "Pure 2D requires every emitting Gaussian inside its grid; "
                f"outside rows={np.count_nonzero(outside)}, "
                f"mu=[{centers[outside].min()}, {centers[outside].max()}], "
                f"sigma=[{sigmas[outside].min()}, {sigmas[outside].max()}]"
            )
        rows = np.flatnonzero(active)
        profile_count = luminosity.shape[1]
        if rows.size == 0:
            return SparseIncrement(
                nodes=np.empty(0, dtype=np.int64),
                weights=np.empty((0, profile_count)),
                fallback_mask=outside,
            )
        mu = centers[rows]
        log_sigma = np.log(sigmas[rows])
        mu_lower = np.clip(np.searchsorted(self.mu_nodes, mu, side="right") - 1, 0, self.mu_nodes.size - 2)
        sigma_lower = np.clip(np.searchsorted(self.log_sigma_nodes, log_sigma, side="right") - 1, 0, self.log_sigma_nodes.size - 2)
        alpha = (mu - self.mu_nodes[mu_lower]) / (self.mu_nodes[mu_lower + 1] - self.mu_nodes[mu_lower])
        beta = (log_sigma - self.log_sigma_nodes[sigma_lower]) / (self.log_sigma_nodes[sigma_lower + 1] - self.log_sigma_nodes[sigma_lower])
        lower = sigma_lower * self.mu_nodes.size + mu_lower
        corner_nodes = (lower, lower + 1, lower + self.mu_nodes.size, lower + self.mu_nodes.size + 1)
        fractions = ((1.0 - alpha) * (1.0 - beta), alpha * (1.0 - beta), (1.0 - alpha) * beta, alpha * beta)
        nodes = np.unique(np.concatenate([ids[fraction != 0.0] for ids, fraction in zip(corner_nodes, fractions)]))
        weights = np.zeros((nodes.size, profile_count))
        selected_light = luminosity[rows]
        for ids, fraction in zip(corner_nodes, fractions):
            retained = fraction != 0.0
            positions = np.searchsorted(nodes, ids[retained])
            np.add.at(weights, positions, selected_light[retained] * fraction[retained, None])
        if not np.isfinite(weights).all():
            raise FloatingPointError("Histogram luminosity accumulation overflowed")
        return SparseIncrement(nodes=nodes, weights=weights, fallback_mask=outside)
