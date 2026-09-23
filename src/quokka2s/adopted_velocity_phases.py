"""Streaming mass-weighted LOS velocity distributions for the adopted cells.

The caller supplies the phase temperature and removes unavailable cells first.
This module makes no chemistry, temperature-selection, or emissivity decisions.
Velocities are not recentered, thermally broadened, or clipped to the plot window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# These are the established cuts in pipeline.utils.classify_temperature_phase.
# Keeping this numerical helper independent avoids loading the yt pipeline.
PHASE_ORDER = ("CNM", "UNM", "WNM", "WIM", "HIM")
PHASE_BOUNDS_K = (200.0, 3000.0, 1.0e4, 10.0**5.5)
GROUP_ORDER = (*PHASE_ORDER, "total")


@dataclass(frozen=True)
class _WeightedMoments:
    count: int = 0
    mass: float = 0.0
    mean: float = 0.0
    centered_second: float = 0.0

    @classmethod
    def from_arrays(cls, velocity, mass):
        if velocity.size == 0:
            return cls()
        # Shift before summing: E[v^2] - E[v]^2 loses the dispersion when
        # a narrow velocity distribution has a large common bulk offset.
        with np.errstate(over="ignore", invalid="ignore"):
            weight = float(np.sum(mass, dtype=np.float64))
            offset = velocity - velocity[0]
            mean_offset = float(np.sum(mass * offset, dtype=np.float64) / weight)
            mean = float(velocity[0] + mean_offset)
            centered_second = float(np.sum(
                mass * (offset - mean_offset)**2, dtype=np.float64))
        if not np.isfinite([weight, mean, centered_second]).all() or weight <= 0.0:
            raise ValueError("Mass or weighted velocity moments overflowed")
        return cls(int(velocity.size), weight, mean, centered_second)

    def merged(self, other):
        if self.count == 0:
            return other
        if other.count == 0:
            return self
        weight = self.mass + other.mass
        delta = other.mean - self.mean
        fraction = other.mass / weight
        mean = self.mean + delta * fraction
        # Parallel/streaming form of the weighted centered second moment.
        second = (self.centered_second + other.centered_second
                  + delta**2 * (self.mass * fraction))
        if not np.isfinite([weight, mean, second]).all():
            raise ValueError("Accumulated mass or weighted moments overflowed")
        return _WeightedMoments(self.count + other.count, weight, mean, second)

    def report(self, global_mean):
        if not self.count:
            return {
                "count": 0, "mass_g": 0.0, "mean_velocity_kms": None,
                "sigma_internal_kms": None,
                "sigma_about_global_mean_kms": None,
            }
        variance = self.centered_second / self.mass
        return {
            "count": self.count,
            "mass_g": self.mass,
            "mean_velocity_kms": self.mean,
            "sigma_internal_kms": float(np.sqrt(variance)),
            "sigma_about_global_mean_kms": float(np.sqrt(
                variance + (self.mean - global_mean)**2)),
        }


@dataclass(frozen=True)
class _Group:
    full: _WeightedMoments
    window: _WeightedMoments
    histogram: np.ndarray
    below_count: int = 0
    below_mass: float = 0.0
    above_count: int = 0
    above_mass: float = 0.0

    @classmethod
    def empty(cls, n_bins):
        return cls(_WeightedMoments(), _WeightedMoments(), np.zeros(n_bins))

    @classmethod
    def from_arrays(cls, velocity, mass, edges):
        below = velocity < edges[0]
        above = velocity > edges[-1]
        inside = ~(below | above)
        # Match np.histogram's inclusive rightmost edge, without its generic
        # weighted cumulative-sum subtraction (which can lose low-mass bins
        # when the chunk spans many decades in mass). Accumulate each bin
        # directly instead. Values outside the window are never clamped.
        indices = np.searchsorted(edges, velocity[inside], side="right") - 1
        indices[indices == edges.size - 1] = edges.size - 2
        histogram = np.bincount(indices, weights=mass[inside], minlength=edges.size - 1)
        if not np.isfinite(histogram).all():
            raise ValueError("Mass histogram overflowed")
        return cls(
            _WeightedMoments.from_arrays(velocity, mass),
            _WeightedMoments.from_arrays(velocity[inside], mass[inside]),
            histogram, int(np.count_nonzero(below)), float(np.sum(mass[below])),
            int(np.count_nonzero(above)), float(np.sum(mass[above])),
        )

    def merged(self, other):
        return _Group(
            self.full.merged(other.full), self.window.merged(other.window),
            self.histogram + other.histogram,
            self.below_count + other.below_count,
            self.below_mass + other.below_mass,
            self.above_count + other.above_count,
            self.above_mass + other.above_mass,
        )


class AdoptedVelocityPhaseAccumulator:
    """Accumulate five gas phases and all gas using mass = density * volume.

    ``add`` takes finite, already-selected one-dimensional cell arrays; volume
    may be a positive scalar or a same-shaped array. Phase temperature must be
    positive, and is intentionally chosen by the caller. Thresholds are lower
    inclusive (e.g. 3000 K belongs to WNM). State scales with the bin count,
    while temporary arrays are bounded by the caller's input chunk.

    ``report`` returns JSON-safe full-range moments plus separate in-window
    moments and counts/masses below and above the histogram window. Both sets
    of ``sigma_about_global_mean_kms`` use the full-range all-gas mean. Empty
    populations have null means/widths, rather than an artificial zero width.
    """

    def __init__(self, velocity_edges_kms):
        edges = np.asarray(velocity_edges_kms, dtype=float)
        if (edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all()
                or not np.all(np.diff(edges) > 0.0)):
            raise ValueError("velocity_edges_kms must be finite, 1D, and strictly increasing")
        self.velocity_edges_kms = edges.copy()
        self.velocity_edges_kms.flags.writeable = False
        self._groups = {key: _Group.empty(edges.size - 1) for key in GROUP_ORDER}

    @property
    def histogram_mass_g(self):
        """Return independent per-bin mass arrays, ordered phases then total."""
        return {key: group.histogram.copy() for key, group in self._groups.items()}

    def add(self, velocity_kms, phase_temperature_K, density_g_cm3, cell_volume_cm3):
        """Validate and add one chunk; failures leave accumulated state unchanged."""
        velocity = np.asarray(velocity_kms, dtype=float)
        temperature = np.asarray(phase_temperature_K, dtype=float)
        density = np.asarray(density_g_cm3, dtype=float)
        volume = np.asarray(cell_volume_cm3, dtype=float)
        if velocity.ndim != 1 or not np.isfinite(velocity).all():
            raise ValueError("velocity_kms must be finite with shape (cells,)")
        for name, value in (("phase_temperature_K", temperature),
                            ("density_g_cm3", density)):
            if (value.shape != velocity.shape or not np.isfinite(value).all()
                    or np.any(value <= 0.0)):
                raise ValueError(f"{name} must be finite and positive with shape (cells,)")
        if (volume.shape not in ((), velocity.shape) or not np.isfinite(volume).all()
                or np.any(volume <= 0.0)):
            raise ValueError("cell_volume_cm3 must be finite and positive, scalar or (cells,)")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            mass = density * volume
        if not np.isfinite(mass).all() or np.any(mass <= 0.0):
            raise ValueError("Cell mass density * volume must be finite and positive")
        phase_index = np.searchsorted(PHASE_BOUNDS_K, temperature, side="right")
        # Compute everything before replacing state, including all-gas moments
        # independently of the five phase populations for cross-checking.
        updated = {}
        for index, key in enumerate(GROUP_ORDER):
            if key == "total":
                v_group, m_group = velocity, mass
            else:
                selected = phase_index == index
                v_group, m_group = velocity[selected], mass[selected]
            delta = _Group.from_arrays(v_group, m_group, self.velocity_edges_kms)
            updated[key] = self._groups[key].merged(delta)
        self._groups = updated
        return self

    def report(self):
        """Return metadata and ``groups[phase_or_total]`` numerical summaries."""
        total = self._groups["total"].full
        global_mean = total.mean if total.count else None
        groups = {}
        for key, group in self._groups.items():
            result = group.full.report(global_mean)
            result.update({
                "mass_fraction": group.full.mass / total.mass if total.count else None,
                "cell_fraction": group.full.count / total.count if total.count else None,
                "in_window": group.window.report(global_mean),
                "below_window": {"count": group.below_count, "mass_g": group.below_mass},
                "above_window": {"count": group.above_count, "mass_g": group.above_mass},
                "histogram_mass_g": float(np.sum(group.histogram)),
            })
            groups[key] = result
        return {
            "phase_order": list(PHASE_ORDER),
            "phase_boundaries_K": list(PHASE_BOUNDS_K),
            "boundary_rule": "lower inclusive; upper exclusive; HIM has no upper bound",
            "velocity_window_kms": [float(self.velocity_edges_kms[0]),
                                    float(self.velocity_edges_kms[-1])],
            "histogram_boundary_rule": "both outer edges included; no clipping",
            "global_mean_velocity_kms": global_mean,
            "sigma_reference": "all valid gas over the full velocity range",
            "groups": groups,
        }
