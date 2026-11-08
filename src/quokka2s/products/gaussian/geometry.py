"""Validated Gaussian histogram geometry and owned sparse increments."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """Bounds in km/s; log_sigma_step is a natural-log (ln) interval.

    mu_step is the maximum linear-mu spacing. The last interval on either axis
    can be shorter to include the exact configured upper bound. No parameter
    outside these bounds is clipped onto the grid.
    """

    mu_min: float
    mu_max: float
    mu_step: float
    sigma_min: float
    sigma_max: float
    log_sigma_step: float
    mu_core_min: float | None = None
    mu_core_max: float | None = None
    mu_wing_step: float | None = None
    narrow_sigma_threshold: float | None = None
    narrow_mu_step: float | None = None

    def __post_init__(self):
        values = (self.mu_min, self.mu_max, self.mu_step, self.sigma_min,
                  self.sigma_max, self.log_sigma_step)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Grid bounds and spacings must be finite")
        if not self.mu_min < self.mu_max or self.mu_step <= 0.0:
            raise ValueError("Require mu_min < mu_max and mu_step > 0")
        if not 0.0 < self.sigma_min < self.sigma_max or self.log_sigma_step <= 0.0:
            raise ValueError("Require 0 < sigma_min < sigma_max and ln step > 0")
        if self.mu_core_min is not None:
            if self.mu_core_max is None or self.mu_wing_step is None:
                raise ValueError("A refined mu core requires both bounds and a wing step")
            if not self.mu_min < self.mu_core_min < self.mu_core_max < self.mu_max:
                raise ValueError("The refined mu core must lie inside the complete range")
            if not math.isfinite(self.mu_wing_step) or self.mu_wing_step <= 0.0:
                raise ValueError("The outer mu step must be finite and positive")


@dataclass(frozen=True)
class SparseIncrement:
    """Unique sigma-major node IDs, (K,P) luminosities, and (R,) zero fallback mask.

    weights are integrated cell luminosities, in the caller's profile order.
    node_id = sigma_index * engine.mu_nodes.size + mu_index. A caller may safely
    add weights into global_weights[nodes] because every ID occurs only once.
    """

    nodes: np.ndarray
    weights: np.ndarray
    fallback_mask: np.ndarray


def _bounded_axis(lower, upper, step):
    """Keep regular interior nodes and the exact two configured endpoints."""
    nodes = np.arange(lower, upper, step, dtype=np.float64)
    nodes = nodes[nodes < upper]
    nodes = np.concatenate((nodes, np.array([upper], dtype=np.float64)))
    if nodes.size < 2 or nodes[0] != lower or not np.all(np.diff(nodes) > 0.0):
        raise ValueError("Grid spacing must produce distinct, increasing float64 nodes")
    nodes.flags.writeable = False
    return nodes


class BandGeometry:
    """Own the accepted immutable mean and logarithmic-sigma axes."""

    def __init__(
        self,
        grid: GridSpec,
        *,
        template_scratch_bytes=64 * 1024**2,
        contraction="matmul",
    ):
        if type(template_scratch_bytes) is not int or template_scratch_bytes < 1:
            raise ValueError("template_scratch_bytes must be a positive integer")
        if contraction not in ("matmul", "einsum"):
            raise ValueError("contraction must be 'matmul' or 'einsum'")
        self.grid = grid
        self.template_scratch_bytes = template_scratch_bytes
        self.contraction = contraction
        if grid.mu_core_min is None:
            self.mu_nodes = _bounded_axis(grid.mu_min, grid.mu_max, grid.mu_step)
        else:
            left = _bounded_axis(grid.mu_min, grid.mu_core_min, grid.mu_wing_step)
            core = _bounded_axis(grid.mu_core_min, grid.mu_core_max, grid.mu_step)
            right = _bounded_axis(grid.mu_core_max, grid.mu_max, grid.mu_wing_step)
            self.mu_nodes = np.concatenate((left[:-1], core[:-1], right))
            self.mu_nodes.flags.writeable = False
        self.log_sigma_nodes = _bounded_axis(
            float(np.log(grid.sigma_min)),
            float(np.log(grid.sigma_max)),
            grid.log_sigma_step,
        )
        self.sigma_nodes = np.exp(self.log_sigma_nodes)
        self.sigma_nodes[0] = grid.sigma_min
        self.sigma_nodes[-1] = grid.sigma_max
        self.sigma_nodes.flags.writeable = False
        self.node_count = int(self.mu_nodes.size) * int(self.sigma_nodes.size)
        if self.node_count > np.iinfo(np.int64).max:
            raise ValueError("Grid node IDs exceed int64 capacity")
