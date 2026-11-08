"""Prepare immutable A translation templates before snapshot processing."""
from __future__ import annotations

from dataclasses import replace
import operator
import time

import numpy as np

from .intervals import VelocityGrid
from .a_templates import MethodA
from .pure_engine import GaussianHistogramEngine as PureHistogramEngine


CUTOFF_SIGMA = 5.0


def _buffer_bytes(arrays):
    """Count retained NumPy data once even when some arrays are views."""
    owners = {}
    for array in arrays:
        owner = array
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        owners[id(owner)] = owner
    return sum(owner.nbytes for owner in owners.values())


class TableBundle:
    """Own lookup objects while reusing the original narrow/broad geometry.

    engines contains the two original native grid-geometry objects. tables has
    two A template objects. The memory total
    counts that shared table once. Optional engines avoids even duplicating the
    original axes, and is recommended when installing the snapshot adapter.
    """

    def __init__(self, grid, edges, method, *, engines=None):
        began = time.perf_counter()
        self.grid = grid
        self.method = str(method).upper()
        if self.method != "A":
            raise ValueError("The adopted Gaussian templates require method A")
        self.cutoff_sigma = CUTOFF_SIGMA
        self.edges = np.array(edges, dtype=np.float64, order="C", copy=True)
        if self.edges.ndim != 1 or self.edges.size < 2:
            raise ValueError("Velocity edges must be a one-dimensional array")
        widths = np.diff(self.edges)
        if not np.isfinite(self.edges).all() or np.any(widths <= 0.0):
            raise ValueError("Velocity edges must be finite and increasing")
        delta_v = float(widths[0])
        if not np.array_equal(widths, np.full_like(widths, delta_v)):
            raise ValueError("A template comparison requires uniform channels")
        self.velocity_grid = VelocityGrid(
            v_min=float(self.edges[0]),
            v_max=float(self.edges[-1]),
            delta_v=delta_v,
        )
        if not np.array_equal(self.velocity_grid.edges, self.edges):
            raise ValueError("The lookup velocity grid must reproduce the exact saved edges")
        self.edges.flags.writeable = False
        self.owns_geometry = engines is None
        if engines is None:
            geometry = PureHistogramEngine(grid, contraction="einsum")
            engines = geometry.engines
        self.engines = tuple(engines)
        if len(self.engines) != 2:
            raise ValueError("The pure-2D adapter requires narrow then broad geometry")
        expected_grids = (
            replace(grid, sigma_max=grid.narrow_sigma_threshold, mu_step=grid.narrow_mu_step),
            replace(grid, sigma_min=grid.narrow_sigma_threshold),
        )
        self.alignment = []
        for band, expected_grid in zip(self.engines, expected_grids):
            if band.grid != expected_grid:
                raise ValueError("Lookup geometry differs from the original pure-2D bands")
            ratio = delta_v / band.grid.mu_step
            R = int(round(ratio))
            if R < 1 or not np.isclose(ratio, R, rtol=0.0, atol=1e-12):
                raise ValueError("The mean core spacing must evenly subdivide one channel")
            coordinate = band.mu_nodes / delta_v
            fractional_nodes = (coordinate - np.floor(coordinate)) * R
            error = float(np.max(np.abs(fractional_nodes - np.rint(fractional_nodes)), initial=0.0))
            if error > 1e-10:
                raise ValueError("The original mu axis is not aligned with the fractional lookup nodes")
            self.alignment.append({
                "R": R,
                "mu_node_count": int(band.mu_nodes.size),
                "sigma_node_count": int(band.sigma_nodes.size),
                "fraction_alignment_max_error": error,
                "fraction_alignment_exact": error == 0.0,
                "sigma_nodes_source": "unchanged actual pure-2D band sigma axis",
            })
        self.tables = tuple(
            MethodA(
                grid=self.velocity_grid,
                sigma_nodes=band.sigma_nodes,
                mu_min=grid.mu_min,
                mu_max=grid.mu_max,
                R=alignment["R"],
                cutoff_sigma=self.cutoff_sigma,
            )
            for band, alignment in zip(self.engines, self.alignment)
        )
        self.build_seconds = time.perf_counter() - began

    def evaluate_band(self, band_index, mu, sigma, *, sigma_id=None, chunk_size=16384):
        """Return bin masses[N,K]; transpose/copy and contraction stay in adapter.

        sigma_id is used and strictly validated by A. A callers should pass band_node_id // band_mu_count for the known-node
        fast path. All validation, allocation and online querying remain inside
        this call and therefore inside the final expansion timer.
        """
        band_index = operator.index(band_index)
        if band_index not in (0, 1):
            raise ValueError("band_index must be 0 (narrow) or 1 (broad)")
        table = self.tables[band_index]
        return table.evaluate(mu, sigma, sigma_id=sigma_id, chunk_size=chunk_size)

    def info(self):
        """Return real lookup bytes and preparation cost, excluding process time."""
        unique_tables = {id(table): table for table in self.tables}
        lookup_bytes = sum(int(table.info()["memory_bytes"]) for table in unique_tables.values())
        geometry_bytes = 0
        if self.owns_geometry:
            arrays = [array for band in self.engines for array in (band.mu_nodes, band.sigma_nodes, band.log_sigma_nodes)]
            geometry_bytes = _buffer_bytes(arrays)
        edges_bytes = self.edges.nbytes
        return {
            "method": self.method,
            "cutoff_sigma": self.cutoff_sigma,
            "renormalized": False,
            "build_seconds": float(self.build_seconds),
            "lookup_memory_bytes": lookup_bytes,
            "owned_geometry_bytes": int(geometry_bytes),
            "owned_velocity_edge_bytes": int(edges_bytes),
            "memory_bytes": int(lookup_bytes + geometry_bytes + edges_bytes),
            "memory_scope": "Own persistent NumPy buffers; shared table/geometry views counted once; Python overhead excluded",
            "geometry_reused": not self.owns_geometry,
            "native_query_kernels_reused": True,
            "table_count": len(unique_tables),
            "velocity_grid": {
                "v_min": float(self.velocity_grid.v_min),
                "v_max": float(self.velocity_grid.v_max),
                "delta_v": float(self.velocity_grid.delta_v),
                "K": int(self.velocity_grid.K),
            },
            "band_alignment": self.alignment,
            "band_lookup_info": [table.info() for table in self.tables],
            "preparation_scope": "prepare_tables call including geometry validation/build; module imports are separate; no table construction during evaluation",
        }


def prepare_tables(grid, edges, method, *, engines=None):
    """Construct the requested bundle before starting snapshot process timing."""
    return TableBundle(grid, edges, method, engines=engines)
