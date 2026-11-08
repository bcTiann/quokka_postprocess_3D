"""Expand occupied 2D nodes with A templates and exact endpoint accounts."""

import time

import numpy as np

from .intervals import interval_probability
from .pure_engine import GaussianHistogramEngine as OriginalEngine


class GaussianHistogramEngine(OriginalEngine):
    def __init__(self, grid, *, contraction="einsum"):
        if contraction != "einsum":
            raise ValueError("The matched local experiment retains einsum.")
        super().__init__(grid, contraction=contraction)
        self.gaussian_cutoff_sigma = 5.0
        self.lookup_tables = None
        self.retained_nodes = None
        self.retained_weights = None
        self.untruncated_window_luminosity = None
        self.clipped_window_luminosity = None
        self.expansion_statistics = {
            "expand_calls": 0,
            "occupied_rows": 0,
            "probability_generation_seconds": 0.0,
            "template_layout_seconds": 0.0,
            "contraction_seconds": 0.0,
            "independent_window_accounting_seconds": 0.0,
            "cutoff_sigma": 5.0,
            "renormalized": False,
            "template_layout": "C contiguous (400 bins, block nodes), using the adopted A templates",
            "contraction": "einsum ki,ip->kp; optimize=False",
            "histogram_retention": "Final expansion retains the accumulated node luminosities.",
        }

    def _expand_band(self, band_index, nodes, weights, edges):
        engine = self.engines[band_index]
        nodes = np.asarray(nodes)
        weights = np.asarray(weights, dtype=np.float64)
        edges = np.asarray(edges, dtype=np.float64)
        if nodes.ndim != 1 or nodes.dtype.kind not in "iu":
            raise ValueError("Node IDs must be an integer vector.")
        if weights.ndim != 2 or weights.shape[0] != nodes.size:
            raise ValueError("Weights must have shape (node count, profile count).")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("Histogram weights must be finite and nonnegative.")
        if np.any(nodes < 0) or np.any(nodes >= engine.node_count):
            raise ValueError("A node ID lies outside its band.")
        widths = np.diff(edges)
        if edges.ndim != 1 or not np.isfinite(edges).all() or np.any(widths <= 0):
            raise ValueError("Velocity edges must be finite and increasing.")
        output = np.zeros((widths.size, weights.shape[1]), dtype=float)
        active = np.any(weights != 0.0, axis=1)
        if not np.all(active):
            nodes = nodes[active]
            weights = weights[active]
        bytes_per_node = (edges.size + widths.size) * np.dtype(np.float64).itemsize
        block_nodes = engine.template_scratch_bytes // bytes_per_node
        self.expansion_statistics["occupied_rows"] += int(nodes.size)
        for start in range(0, nodes.size, block_nodes):
            stop = min(start + block_nodes, nodes.size)
            ids = nodes[start:stop]
            sigma_ids = ids // engine.mu_nodes.size
            centers = engine.mu_nodes[ids % engine.mu_nodes.size]
            sigmas = engine.sigma_nodes[sigma_ids]
            began = time.perf_counter()
            row_probabilities = self.lookup_tables.evaluate_band(
                band_index,
                centers,
                sigmas,
                sigma_id=sigma_ids,
            )
            self.expansion_statistics["probability_generation_seconds"] += time.perf_counter() - began
            began = time.perf_counter()
            # Every method returns node-major rows. Match the original engine's
            # bin-major contiguous layout before the unchanged contraction.
            bin_integrals = np.ascontiguousarray(row_probabilities.T)
            del row_probabilities
            self.expansion_statistics["template_layout_seconds"] += time.perf_counter() - began
            began = time.perf_counter()
            contribution = np.einsum(
                "ki,ip->kp",
                bin_integrals,
                weights[start:stop],
                optimize=False,
            )
            self.expansion_statistics["contraction_seconds"] += time.perf_counter() - began
            if not np.isfinite(contribution).all():
                raise FloatingPointError("The template contraction is nonfinite.")
            output += contribution
            del bin_integrals

            # Exact endpoint integrals distinguish a physical window loss, a
            # deliberate five-sigma tail loss, and the signed LUT residual.
            began = time.perf_counter()
            z = (edges[[0, -1], None] - centers[None, :]) / sigmas[None, :]
            full_mass = interval_probability(z[0], z[1])
            np.clip(z, -5.0, 5.0, out=z)
            clipped_mass = interval_probability(z[0], z[1])
            self.untruncated_window_luminosity += np.einsum(
                "i,ip->p", full_mass, weights[start:stop], optimize=False,
            )
            self.clipped_window_luminosity += np.einsum(
                "i,ip->p", clipped_mass, weights[start:stop], optimize=False,
            )
            self.expansion_statistics["independent_window_accounting_seconds"] += time.perf_counter() - began
        output /= widths[:, None]
        if not np.isfinite(output).all():
            raise FloatingPointError("The expanded spectrum is nonfinite.")
        return output

    def expand(self, nodes, weights, edges):
        if self.lookup_tables is None:
            raise RuntimeError("Prepare lookup tables before processing the snapshot.")
        self.expansion_statistics["expand_calls"] += 1
        if self.expansion_statistics["expand_calls"] != 1:
            raise RuntimeError("This complete snapshot must expand once.")
        self.retained_nodes = nodes
        self.retained_weights = weights
        self.untruncated_window_luminosity = np.zeros(weights.shape[1])
        self.clipped_window_luminosity = np.zeros(weights.shape[1])
        output = np.zeros((len(edges) - 1, weights.shape[1]), dtype=float)
        for band_index, (engine, offset) in enumerate(zip(self.engines, self.offsets)):
            selected = (nodes >= offset) & (nodes < offset + engine.node_count)
            if np.any(selected):
                output += self._expand_band(
                    band_index=band_index,
                    nodes=nodes[selected] - offset,
                    weights=weights[selected],
                    edges=edges,
                )
        return output
