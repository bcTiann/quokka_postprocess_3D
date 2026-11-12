"""Sparse logical-node histogram with sigma-scaled mean spacing."""
from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import numpy as np

from .layered_nodes import deposit_layered
from .intervals import interval_probability
from .layered_geometry import LayeredGridSpec, LayeredIncrement
from .layered_gate import (
    RULE,
    TEMPLATE_SCRATCH_BYTES,
    run_gate,
    validate_gate,
)
from .prefix_arithmetic import table_signature
from .prefix_tables import PrefixGaussianTable, select_expansion_workers
from ..velocity_grid import velocity_channel_edges_kms


class LayeredHistogramEngine:
    """Accumulate only visited integer nodes, then expand their luminosities.

    One immutable geometry is shared by batch workers. Each deposit owns its
    temporary NumPy arrays. The main accumulator owns a sorted SparseNodeStore;
    neither structure allocates by logical extent.
    """

    coordinate_rule = "sigma-scaled-layered"
    contraction = "einsum"

    def __init__(self, grid, *, table, gate_result, threads=0):
        self.grid = grid
        self.lookup_tables = table
        self.expansion_threads = select_expansion_workers(threads)
        self.portable_backend = validate_gate(
            gate_result,
            grid,
            table,
            threads=self.expansion_threads,
        )
        self.gaussian_cutoff_sigma = 5.0
        self.template_scratch_bytes = TEMPLATE_SCRATCH_BYTES
        self._gate_table_signature = gate_result["table_signature"]
        self.retained_nodes = None
        self.retained_keys = None
        self.retained_weights = None
        self.untruncated_window_luminosity = None
        self.clipped_window_luminosity = None
        self.expansion_statistics = {
            "expand_calls": 0,
            "occupied_rows": 0,
            "coordinate_reconstruction_seconds": 0.0,
            "independent_window_accounting_seconds": 0.0,
            "lookup_and_contraction_seconds": 0.0,
            "support_bins_visited": 0,
            "nonzero_template_coefficients": 0,
            "nonzero_product_updates": 0,
            "active_profile_count_sum": 0,
            "phi_boundary_evaluations": 0,
            "cutoff_sigma": 5.0,
            "renormalized": False,
            "coordinate_rule": RULE,
            "coordinate_parameters": asdict(grid),
            "template_layout": "One prefix table and bounded node-by-channel coefficients",
            "contraction": "Lexicographic layer/mu nodes; bounded NumPy einsum ki,ip->kp, optimize=False",
            "histogram_retention": "Final expansion retains only sorted occupied logical keys and luminosities",
            "portable_backend": self.portable_backend,
            "portable_gate": gate_result,
            "implementation": "numpy",
            "workers": self.expansion_threads,
            "lookup_table": table.info(),
        }

    def deposit(self, centers, sigmas, luminosity):
        """Return owned sorted logical keys [U,2] and luminosities [U,P]."""
        centers = np.ascontiguousarray(centers, dtype=np.float64)
        sigmas = np.ascontiguousarray(sigmas, dtype=np.float64)
        luminosity = np.ascontiguousarray(luminosity, dtype=np.float64)
        if centers.ndim != 1 or sigmas.shape != centers.shape:
            raise ValueError("Layered Gaussian means and widths must be equal vectors")
        if luminosity.ndim != 2 or luminosity.shape[0] != centers.size:
            raise ValueError("Layered luminosity must have shape (cell count,profile count)")
        if luminosity.shape[1] < 1 or luminosity.shape[1] > 12:
            raise ValueError("The adopted width groups support one to twelve luminosity accounts")
        keys, weights = deposit_layered(self.grid, centers, sigmas, luminosity)
        if not np.isfinite(weights).all():
            raise FloatingPointError("Layered histogram luminosity accumulation overflowed")
        return LayeredIncrement(
            nodes=keys,
            weights=weights,
            fallback_mask=np.zeros(centers.size, dtype=bool),
        )

    def worker_statistics(self):
        """No persistent worker hash/workspace survives NumPy deposition."""
        return {
            "implementation": "numpy",
            "worker_workspaces": 0, "persistent_worker_workspace_bytes": 0,
            "scope": "Temporary array scatter per batch; not a measurement of peak RSS",
        }

    def describe_grids(self):
        """Describe one coordinate rule without claiming a finite node rectangle."""
        return [{
            "coordinate_rule": RULE,
            "key_columns": ["sigma_layer_id", "signed_mu_index"],
            "parameters": asdict(self.grid),
            "mu_spacing": "mu_spacing_ratio * sigma(layer)",
            "preallocated_rectangular_nodes": 0,
        }]

    def expand(self, nodes, weights, edges):
        """Return one [K,40] dL/dv spectrum from sorted occupied logical keys."""
        keys = np.asarray(nodes)
        weights = np.asarray(weights, dtype=np.float64)
        edges = np.asarray(edges, dtype=np.float64)
        if keys.ndim != 2 or keys.shape[1] != 2 or keys.dtype != np.dtype(np.int64):
            raise ValueError("Logical node keys must have shape (node count,2) and dtype int64")
        if weights.shape != (keys.shape[0], 40):
            raise ValueError("Layered expansion requires luminosities with shape (node count,40)")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("Layered luminosities must be finite and nonnegative")
        if not np.array_equal(edges, self.lookup_tables.edges):
            raise ValueError("Layered channels differ from the prepared prefix table")
        if table_signature(self.lookup_tables) != self._gate_table_signature:
            raise ValueError("The layered prefix table changed after its arithmetic gate")
        if keys.shape[0] > 1:
            increasing = (keys[1:, 0] > keys[:-1, 0]) | (
                (keys[1:, 0] == keys[:-1, 0]) & (keys[1:, 1] > keys[:-1, 1])
            )
            if not np.all(increasing):
                raise ValueError("Final logical node keys must be unique and lexicographically increasing")
        if self.expansion_statistics["expand_calls"] != 0:
            raise RuntimeError("A complete layered snapshot expands exactly once")
        self.expansion_statistics["expand_calls"] = 1
        active = np.any(weights != 0.0, axis=1)
        if not np.all(active):
            keys = keys[active]
            weights = weights[active]
        keys = np.ascontiguousarray(keys)
        weights = np.ascontiguousarray(weights)
        self.retained_nodes = keys
        self.retained_keys = keys
        self.retained_weights = weights
        self.untruncated_window_luminosity = np.zeros(40, dtype=np.float64)
        self.clipped_window_luminosity = np.zeros(40, dtype=np.float64)
        widths = np.diff(edges)
        output = np.zeros((widths.size, 40), dtype=np.float64)
        bytes_per_node = (8 * edges.size + widths.size) * 8
        block_nodes = self.template_scratch_bytes // bytes_per_node
        if block_nodes < 1:
            raise ValueError("The expansion block budget must hold at least one logical node")
        self.expansion_statistics["occupied_rows"] = int(keys.shape[0])
        for start in range(0, keys.shape[0], block_nodes):
            stop = min(start + block_nodes, keys.shape[0])
            began = perf_counter()
            centers, sigmas = self.grid.coordinates(keys[start:stop])
            self.expansion_statistics["coordinate_reconstruction_seconds"] += perf_counter() - began
            contribution = np.zeros_like(output)
            began = perf_counter()
            counts = self.lookup_tables.expand(
                centers,
                sigmas,
                weights[start:stop],
                contribution,
                threads=self.expansion_threads,
            )
            self.expansion_statistics["lookup_and_contraction_seconds"] += perf_counter() - began
            for name, value in counts.items():
                self.expansion_statistics[name] += value
            if not np.isfinite(contribution).all():
                raise FloatingPointError("The layered prefix contraction is nonfinite")
            output += contribution
            began = perf_counter()
            z = (edges[[0, -1], None] - centers[None, :]) / sigmas[None, :]
            full_mass = interval_probability(z[0], z[1])
            np.clip(z, -5.0, 5.0, out=z)
            clipped_mass = interval_probability(z[0], z[1])
            self.untruncated_window_luminosity += np.einsum(
                "i,ip->p",
                full_mass,
                weights[start:stop],
                optimize=False,
            )
            self.clipped_window_luminosity += np.einsum(
                "i,ip->p",
                clipped_mass,
                weights[start:stop],
                optimize=False,
            )
            self.expansion_statistics["independent_window_accounting_seconds"] += perf_counter() - began
        output /= widths[:, None]
        if not np.isfinite(output).all():
            raise FloatingPointError("The expanded layered spectrum is nonfinite")
        return output


def prepare_layered_engine(grid_parameters=None):
    """Prepare only coordinate rules, one prefix table and a synthetic gate."""
    began = perf_counter()
    grid = LayeredGridSpec(**({} if grid_parameters is None else grid_parameters))
    table = PrefixGaussianTable(edges=velocity_channel_edges_kms())
    gate = run_gate(grid, table)
    engine = LayeredHistogramEngine(grid, table=table, gate_result=gate)
    return engine, perf_counter() - began
