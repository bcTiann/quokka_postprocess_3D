"""Stable logical coordinates for the sigma-scaled sparse histogram."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..line_velocity_moments import MINIMUM_GAUSSIAN_WIDTH_KMS


@dataclass(frozen=True)
class LayeredGridSpec:
    """Define nodes without allocating a rectangular parameter grid.

    Layer k has log(sigma_floor)+k*log_sigma_step, with k=0 restored to the
    exact physical width floor. Its means are mu_anchor+i*mu_spacing_ratio*
    sigma(k), for signed integer i. Both velocities use km/s. The ratio and
    logarithmic step are dimensionless. Keys contain integers (k,i), never
    floating coordinates or mutable storage slots.
    """

    mu_spacing_ratio: float = 0.1
    log_sigma_step: float = 0.03125
    mu_anchor: float = 0.0
    sigma_floor: float = field(default=MINIMUM_GAUSSIAN_WIDTH_KMS, init=False)

    def __post_init__(self):
        values = (self.mu_spacing_ratio, self.log_sigma_step, self.mu_anchor)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Layered coordinate rules must be finite")
        if self.mu_spacing_ratio <= 0.0 or self.log_sigma_step <= 0.0:
            raise ValueError("Require positive mu-spacing ratio and logarithmic sigma step")
        if not math.isfinite(self.mu_spacing_ratio * self.sigma_floor):
            raise ValueError("The first layered mean spacing must be finite")
        minimum_spacing = self.mu_spacing_ratio * self.sigma_floor
        if minimum_spacing <= 0.0:
            raise ValueError("The first layered mean spacing must remain representable")
        if self.mu_anchor + minimum_spacing <= self.mu_anchor or self.mu_anchor - minimum_spacing >= self.mu_anchor:
            raise ValueError("The minimum mu nodes are indistinguishable at this anchor")
        if np.log(self.sigma_floor) + self.log_sigma_step <= np.log(self.sigma_floor):
            raise ValueError("Adjacent logarithmic sigma layers are indistinguishable")

    def sigma_values(self, layers):
        """Restore positive layer widths using the same NumPy exp rule everywhere."""
        layers = np.asarray(layers, dtype=np.int64)
        if np.any(layers < 0):
            raise ValueError("Sigma layer IDs must be nonnegative")
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            values = np.exp(np.log(self.sigma_floor) + layers * self.log_sigma_step)
        values = np.where(layers == 0, self.sigma_floor, values)
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("A logical sigma layer is unrepresentable")
        return values

    def coordinates(self, keys):
        """Restore means/widths [U] from lexicographic integer keys [U,2]."""
        keys = np.asarray(keys)
        if keys.ndim != 2 or keys.shape[1] != 2 or keys.dtype != np.int64:
            raise ValueError("Layered keys must be an int64 array with shape (node count,2)")
        if np.any(keys[:, 1] == np.iinfo(np.int64).max):
            raise ValueError("A logical mean index exceeds integer capacity")
        sigmas = self.sigma_values(keys[:, 0])
        spacing = self.mu_spacing_ratio * sigmas
        with np.errstate(over="ignore", invalid="ignore"):
            centers = self.mu_anchor + keys[:, 1] * spacing
            right = self.mu_anchor + (keys[:, 1] + 1) * spacing
        if not np.isfinite(spacing).all() or np.any(spacing <= 0.0) or not np.isfinite(centers).all() or not np.isfinite(right).all() or np.any(right <= centers):
            raise ValueError("A logical node coordinate is unrepresentable or outside integer capacity")
        return np.ascontiguousarray(centers), np.ascontiguousarray(sigmas)


@dataclass(frozen=True)
class LayeredIncrement:
    """Owned unique logical keys and their integrated luminosity accounts.

    nodes[U,2] contains (sigma_layer, signed_mu_index), sorted in that order.
    weights[U,P] is float64 luminosity in the caller's existing profile order.
    fallback_mask[R] is always false: no cell is sent to another integrator.
    """

    nodes: np.ndarray
    weights: np.ndarray
    fallback_mask: np.ndarray
