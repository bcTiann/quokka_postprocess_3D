"""Expand 2D luminosity nodes through a shared standard-normal prefix table."""
from __future__ import annotations

import time

import numpy as np

from . import _prefix_expand as native
from .engine import GaussianHistogramEngine as AccountedEngine
from .intervals import interval_probability
from .prefix_arithmetic import table_signature, validate_gate


class GaussianHistogramEngine(AccountedEngine):
    """Keep original deposition, merges and exact endpoint luminosity accounts.

    Only final occupied-node expansion changes. The native kernel reuses each
    channel's right cumulative boundary as the next channel's left boundary.
    """

    def __init__(self, grid, *, table, gate_result, threads=0):
        super().__init__(grid, contraction="einsum")
        self.expansion_threads = native.select_threads(threads)
        self.portable_backend = validate_gate(
            gate_result,
            grid,
            self.engines,
            table,
            threads=self.expansion_threads,
        )
        self.match_numpy_fma = self.portable_backend == "prefix_fma"
        self.lookup_tables = table
        self._gate_table_signature = gate_result["table_signature"]
        self.expansion_statistics = {
            "expand_calls": 0,
            "occupied_rows": 0,
            "independent_window_accounting_seconds": 0.0,
            "fused_lookup_contraction_seconds": 0.0,
            "support_bins_visited": 0,
            "nonzero_template_coefficients": 0,
            "nonzero_product_updates": 0,
            "active_profile_count_sum": 0,
            "phi_boundary_evaluations": 0,
            "cutoff_sigma": 5.0,
            "renormalized": False,
            "template_layout": "Standard-normal cumulative table; shared adjacent channel boundaries; no node-by-channel matrix",
            "contraction": "Original bands/blocks; ascending nodes and profiles per channel",
            "histogram_retention": "Final expansion retains the accumulated node luminosities",
            "portable_backend": self.portable_backend,
            "portable_gate": gate_result,
            "native_channel_threads": self.expansion_threads,
            "native_channel_threading": native.kernel_info(self.expansion_threads),
            "lookup_table": table.info(),
        }

    def _expand_band(self, band_index, nodes, weights, edges):
        """Return spectral luminosity [K,40] in erg/s/(km/s) for one sigma band."""
        band = self.engines[band_index]
        nodes = np.asarray(nodes)
        weights = np.asarray(weights, dtype=np.float64)
        edges = np.asarray(edges, dtype=np.float64)
        if nodes.ndim != 1 or nodes.dtype.kind not in "iu":
            raise ValueError("Node IDs must be an integer vector")
        if weights.shape != (nodes.size, 40):
            raise ValueError("Node luminosity must have shape (node count,40)")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("Node luminosities must be finite and nonnegative")
        if np.any(nodes < 0) or np.any(nodes >= band.node_count):
            raise ValueError("A node ID lies outside its original sigma band")
        if not np.array_equal(edges, self.lookup_tables.edges):
            raise ValueError("Velocity edges differ from the prepared prefix table")
        if table_signature(self.lookup_tables) != self._gate_table_signature:
            raise ValueError("The prepared prefix table changed after its arithmetic gate")
        widths = np.diff(edges)
        output = np.zeros((widths.size, 40), dtype=np.float64)
        active = np.any(weights != 0.0, axis=1)
        if not np.all(active):
            nodes = nodes[active]
            weights = weights[active]
        bytes_per_node = (edges.size + widths.size) * np.dtype(np.float64).itemsize
        block_nodes = band.template_scratch_bytes // bytes_per_node
        if block_nodes < 1:
            raise ValueError("The original expansion block must hold at least one node")
        self.expansion_statistics["occupied_rows"] += int(nodes.size)
        for start in range(0, nodes.size, block_nodes):
            stop = min(start + block_nodes, nodes.size)
            ids = nodes[start:stop]
            sigma_ids = ids // band.mu_nodes.size
            centers = np.ascontiguousarray(band.mu_nodes[ids % band.mu_nodes.size])
            sigmas = np.ascontiguousarray(band.sigma_nodes[sigma_ids])
            contribution = np.zeros((widths.size, 40), dtype=np.float64)
            began = time.perf_counter()
            counts = self.lookup_tables.expand(
                centers,
                sigmas,
                weights[start:stop],
                contribution,
                fma=self.match_numpy_fma,
                threads=self.expansion_threads,
            )
            self.expansion_statistics["fused_lookup_contraction_seconds"] += time.perf_counter() - began
            for name, value in counts.items():
                self.expansion_statistics[name] += value
            if not np.isfinite(contribution).all():
                raise FloatingPointError("The prefix contraction is nonfinite")
            output += contribution

            # These analytic endpoints are independent accounting references;
            # online channel probabilities themselves use only the prefix table.
            began = time.perf_counter()
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
            self.expansion_statistics["independent_window_accounting_seconds"] += time.perf_counter() - began
        output /= widths[:, None]
        if not np.isfinite(output).all():
            raise FloatingPointError("The expanded prefix spectrum is nonfinite")
        return output
