"""Prepare one standard-normal cumulative table before snapshot processing."""
from __future__ import annotations

import hashlib
import math
from time import perf_counter

import numpy as np

from . import _prefix_expand
from .intervals import interval_probability


class PrefixGaussianTable:
    """Own cumulative masses [M+1] and uniform velocity edges [K+1].

    ``values[0]`` is zero at standard coordinate ``-cutoff_sigma``. Every next
    value adds the exact mass of one complete standard-normal fine cell. Online
    linear interpolation retains the fine-cell area approximation, while
    adjacent physical channels reuse their shared cumulative boundary.

    Velocity edges use km/s; cutoff_sigma and step are dimensionless. This
    table has no mu/sigma axis and is shared by both luminosity-histogram bands.
    """

    method = "standard-normal-prefix"

    def __init__(self, edges, *, cutoff_sigma=5.0, step=1 / 128):
        began = perf_counter()
        self.Z = float(cutoff_sigma)
        self.requested_h = float(step)
        if not math.isfinite(self.Z) or self.Z <= 0.0:
            raise ValueError("The standard-normal cutoff must be finite and positive")
        if not math.isfinite(self.requested_h) or self.requested_h <= 0.0:
            raise ValueError("The standard-normal fine-cell step must be finite and positive")
        fine_cell_count = 2.0 * self.Z / self.requested_h
        if not math.isfinite(fine_cell_count) or fine_cell_count > np.iinfo(np.int32).max:
            raise ValueError("The standard-normal table must fit int32 indexing")
        self.M = int(math.ceil(fine_cell_count))
        if self.M < 1:
            raise ValueError("The standard-normal table needs at least one fine cell")
        self.h = 2.0 * self.Z / self.M
        self.cutoff_sigma = self.Z

        self.edges = np.array(edges, dtype=np.float64, copy=True)
        if self.edges.ndim != 1 or self.edges.size < 2 or not np.isfinite(self.edges).all():
            raise ValueError("Velocity edges must be a finite vector with at least two values")
        widths = np.diff(self.edges)
        self.delta_v = float(widths[0])
        if self.delta_v <= 0.0 or not np.array_equal(widths, np.full_like(widths, self.delta_v)):
            raise ValueError("The prefix expansion requires increasing uniform velocity edges")

        standard_edges = -self.Z + np.arange(self.M + 1, dtype=np.float64) * self.h
        standard_edges[0] = -self.Z
        standard_edges[-1] = self.Z
        fine_cell_masses = interval_probability(standard_edges[:-1], standard_edges[1:])
        self.fine_cell_masses_sha256 = hashlib.sha256(memoryview(fine_cell_masses)).hexdigest()
        self.values = np.empty(self.M + 1, dtype=np.float64)
        self.values[0] = 0.0
        np.cumsum(fine_cell_masses, out=self.values[1:])
        if not np.isfinite(self.values).all() or np.any(np.diff(self.values) < 0.0):
            raise FloatingPointError("The standard-normal prefix must be finite and nondecreasing")
        self.values.flags.writeable = False
        self.edges.flags.writeable = False
        self.build_seconds = perf_counter() - began

    def expand(self, centers, sigmas, weights, contribution, *, fma=False, threads=0):
        """Add node luminosity [N,40] into channel luminosity [K,40].

        centers and sigmas are occupied histogram-node values in km/s. The
        caller divides channel luminosity by its physical bin width afterward.
        The native kernel visits only bins overlapping node +/-cutoff_sigma.
        """
        return _prefix_expand.expand_block(
            centers=centers,
            sigmas=sigmas,
            weights=weights,
            values=self.values,
            contribution=contribution,
            v_min=float(self.edges[0]),
            delta_v=self.delta_v,
            Z=self.Z,
            h=self.h,
            match_numpy_fma=fma,
            threads=threads,
        )

    def info(self):
        """Report actual persistent buffers, excluding temporary build arrays."""
        return {
            "method": self.method,
            "build_seconds": self.build_seconds,
            "cutoff_sigma": self.Z,
            "renormalized": False,
            "lookup_memory_bytes": int(self.values.nbytes),
            "owned_velocity_edge_bytes": int(self.edges.nbytes),
            "memory_bytes": int(self.values.nbytes + self.edges.nbytes),
            "table_count": 1,
            "shared_by_sigma_bands": 2,
            "online_cdf_calls": False,
            "node_by_channel_probability_matrix_allocated": False,
            "fine_cell_masses_sha256": self.fine_cell_masses_sha256,
            "build_rule": "C[0]=0; C[1:]=cumsum(exact standard-normal fine-cell masses)",
            "configuration": {
                "Z": self.Z,
                "requested_h": self.requested_h,
                "actual_h": self.h,
                "M": self.M,
                "channels": int(self.edges.size - 1),
                "dtype": "float64",
                "query": "Linear cumulative interpolation; adjacent channels share boundary values",
                "prefix_used": True,
            },
        }
