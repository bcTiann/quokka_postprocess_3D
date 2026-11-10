"""Stable logical coordinates for the sigma-scaled sparse histogram."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..line_velocity_moments import MINIMUM_GAUSSIAN_WIDTH_KMS
from . import _layered_deposit as native


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
        if self.mu_spacing_ratio * self.sigma_floor <= 0.0:
            raise ValueError("The first layered mean spacing must remain representable")

    def native_parameters(self):
        """Return only the scalar rules consumed by compiled coordinate lookup."""
        return {
            "mu_spacing_ratio": float(self.mu_spacing_ratio),
            "log_sigma_step": float(self.log_sigma_step),
            "sigma_floor": float(self.sigma_floor),
            "mu_anchor": float(self.mu_anchor),
        }

    def coordinates(self, keys):
        """Restore means and widths [U] from integer keys [U,2], in km/s.

        The same compiled layer routine is used during deposition, avoiding a
        different exp implementation when final node parameters are restored.
        """
        keys = np.asarray(keys)
        if keys.ndim != 2 or keys.shape[1] != 2 or keys.dtype != np.dtype(np.int64):
            raise ValueError("Layered keys must be an int64 array with shape (node count,2)")
        return native.coordinates(
            keys=np.ascontiguousarray(keys),
            **self.native_parameters(),
        )


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
