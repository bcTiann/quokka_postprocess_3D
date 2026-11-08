"""Select sparse arithmetic only after an exact gate against this NumPy build.

The original ABC dense einsum is the explicit fallback. Platform names alone
never authorize a fused/nonfused arithmetic choice or a different reduction.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

from .engine import GaussianHistogramEngine as OriginalABCEngine
from .intervals import interval_probability
from . import _a_templates as native_a
from . import _sparse_expand as sparse_expand_native


BACKENDS = ("sparse_fma", "sparse_separate_muladd", "numpy_einsum")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    values = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(values)).hexdigest()


def runtime_signature():
    """Bind the gate to actual arithmetic binaries, CPU dispatch and versions."""
    try:
        core = importlib.import_module("numpy._core._multiarray_umath")
    except ModuleNotFoundError:
        core = importlib.import_module("numpy.core._multiarray_umath")
    cpu_features = {
        name: bool(value)
        for name, value in sorted(getattr(core, "__cpu_features__", {}).items())
    }
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": list(sys.version_info[:3]),
        "python_implementation": platform.python_implementation(),
        "byteorder": sys.byteorder,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "numpy_cpu_features": cpu_features,
        "numpy_disable_cpu_features": os.environ.get("NPY_DISABLE_CPU_FEATURES"),
        "numpy_einsum_binary_sha256": file_sha256(core.__file__),
        "native_a_binary_sha256": file_sha256(native_a.__file__),
        "sparse_binary_sha256": file_sha256(sparse_expand_native.__file__),
        "sparse_threading": sparse_expand_native.kernel_info(
            sparse_expand_native.select_threads(),
        ),
    }


def band_geometry(engines):
    return [
        {
            "mu_nodes": int(band.mu_nodes.size),
            "sigma_nodes": int(band.sigma_nodes.size),
            "mu_sha256": array_sha256(band.mu_nodes),
            "sigma_sha256": array_sha256(band.sigma_nodes),
            "log_sigma_sha256": array_sha256(band.log_sigma_nodes),
            "template_scratch_bytes": int(band.template_scratch_bytes),
        }
        for band in engines
    ]


def validate_gate(result, grid, engines):
    """Reject stale/foreign gates; accept a declared original-NumPy fallback."""
    if result.get("schema_version") != 1 or result.get("complete") is not True:
        raise ValueError("A completed portable sparse gate is required before process")
    if result.get("selected_backend") not in BACKENDS:
        raise ValueError("The gate does not select a supported contraction backend")
    if result.get("supported_profiles") != 40 or result.get("cutoff_sigma") != 5.0:
        raise ValueError("The gate must retain the current 40-account, five-sigma contract")
    if result.get("runtime_signature") != runtime_signature():
        raise ValueError("The contraction gate belongs to different runtime binaries/CPU dispatch")
    if result.get("grid") != asdict(grid) or result.get("table_geometry") != band_geometry(engines):
        raise ValueError("The contraction gate used a different two-band grid")
    cases = result.get("cases", [])
    if not cases:
        raise ValueError("The contraction gate contains no tested blocks")
    selected = result["selected_backend"]
    if selected != "numpy_einsum":
        if not all(case["modes"][selected]["byte_equal"] for case in cases):
            raise ValueError("The selected sparse arithmetic failed an exact gate")
    return selected


class GaussianHistogramEngine(OriginalABCEngine):
    """Keep original nodes/blocks/bands; use only locally validated arithmetic."""

    def __init__(self, grid, *, contraction="einsum", gate_json=None, gate_result=None):
        if (gate_json is None) == (gate_result is None):
            raise ValueError("Supply exactly one completed gate_json or gate_result before process")
        if gate_json is not None:
            gate_result = json.loads(Path(gate_json).read_text())
        super().__init__(grid, contraction=contraction)
        self.portable_backend = validate_gate(gate_result, grid, self.engines)
        self.match_numpy_fma = self.portable_backend == "sparse_fma"
        self.expansion_threads = sparse_expand_native.select_threads()
        self._gate_edges_sha256 = gate_result["edges_sha256"]
        self.expansion_statistics["portable_backend"] = self.portable_backend
        self.expansion_statistics["portable_gate"] = {
            "schema_version": gate_result["schema_version"],
            "selected_backend": self.portable_backend,
            "selection_reason": gate_result["selection_reason"],
            "tested_blocks": len(gate_result["cases"]),
            "runtime_signature": gate_result["runtime_signature"],
            "numeric_tolerance": "none; byte equality",
            "cross_platform_byte_equality_claimed": False,
        }
        if self.portable_backend != "numpy_einsum":
            for name in ("probability_generation_seconds", "template_layout_seconds", "contraction_seconds"):
                self.expansion_statistics.pop(name)
            arithmetic = "explicit float64 FMA" if self.match_numpy_fma else "separate float64 multiply then add"
            self.expansion_statistics.update({
                "native_channel_threads": self.expansion_threads,
                "native_channel_threading": sparse_expand_native.kernel_info(self.expansion_threads),
                "fused_lookup_contraction_seconds": 0.0,
                "support_bins_visited": 0,
                "nonzero_template_coefficients": 0,
                "nonzero_product_updates": 0,
                "active_profile_count_sum": 0,
                "template_layout": "Direct A-table reads; no node-by-channel template matrix",
                "contraction": "Original ascending nodes/blocks/bands; locally gated " + arithmetic,
            })

    def _expand_band(self, band_index, nodes, weights, edges):
        if self.lookup_tables is None or self.lookup_tables.method != "A":
            raise ValueError("Portable expansion requires the prepared A lookup bundle")
        if self.lookup_tables.cutoff_sigma != 5.0 or self.gaussian_cutoff_sigma != 5.0:
            raise ValueError("The five-sigma support rule cannot change after the gate")
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 2 or weights.shape[1] != 40:
            raise ValueError("The portable gate supports weights (node count, 40)")
        edges = np.asarray(edges, dtype=np.float64)
        if array_sha256(edges) != self._gate_edges_sha256:
            raise ValueError("The process velocity edges differ from the gated edges")
        if self.portable_backend == "numpy_einsum":
            return super()._expand_band(band_index, nodes, weights, edges)
        return self._expand_sparse_band(band_index, nodes, weights, edges)

    def _expand_sparse_band(self, band_index, nodes, weights, edges):
        """Frozen sparse block flow with only the gate-selected FMA flag varied."""
        engine = self.engines[band_index]
        table = self.lookup_tables.tables[band_index]
        nodes = np.asarray(nodes)
        if nodes.ndim != 1 or nodes.dtype.kind not in "iu":
            raise ValueError("Node IDs must be an integer vector")
        if weights.shape[0] != nodes.size:
            raise ValueError("Histogram weights must match the node count")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("Histogram weights must be finite and nonnegative")
        if np.any(nodes < 0) or np.any(nodes >= engine.node_count):
            raise ValueError("A node ID lies outside its band")
        widths = np.diff(edges)
        if edges.ndim != 1 or not np.isfinite(edges).all() or np.any(widths <= 0.0):
            raise ValueError("Velocity edges must be finite and increasing")
        if not np.array_equal(edges, self.lookup_tables.edges):
            raise ValueError("Use the exact velocity grid prepared for the A table")
        output = np.zeros((widths.size, weights.shape[1]), dtype=np.float64)
        active = np.any(weights != 0.0, axis=1)
        if not np.all(active):
            nodes = nodes[active]
            weights = weights[active]
        bytes_per_node = (edges.size + widths.size) * np.dtype(np.float64).itemsize
        block_nodes = engine.template_scratch_bytes // bytes_per_node
        if block_nodes < 1:
            raise ValueError("Template scratch budget must hold at least one node")
        self.expansion_statistics["occupied_rows"] += int(nodes.size)
        for start in range(0, nodes.size, block_nodes):
            stop = min(start + block_nodes, nodes.size)
            ids = nodes[start:stop]
            sigma_ids = np.ascontiguousarray(ids // engine.mu_nodes.size, dtype=np.int64)
            centers = np.ascontiguousarray(engine.mu_nodes[ids % engine.mu_nodes.size])
            sigmas = np.ascontiguousarray(engine.sigma_nodes[sigma_ids])
            contribution = np.zeros((widths.size, weights.shape[1]), dtype=np.float64)
            began = time.perf_counter()
            if not np.array_equal(sigmas, table.sigma_nodes[sigma_ids]):
                raise ValueError("Sigma IDs differ from the exact A table axis")
            counts = sparse_expand_native.expand_block(
                centers, sigmas, sigma_ids, weights[start:stop], table.T, contribution,
                table.grid.v_min, table.grid.delta_v, table.n_max, table.R,
                self.gaussian_cutoff_sigma, self.match_numpy_fma,
                threads=self.expansion_threads,
            )
            self.expansion_statistics["fused_lookup_contraction_seconds"] += time.perf_counter() - began
            for name, value in counts.items():
                self.expansion_statistics[name] += value
            if not np.isfinite(contribution).all():
                raise FloatingPointError("The sparse template contraction is nonfinite")
            output += contribution

            # Preserve the original blockwise endpoint accounting and reductions.
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
            raise FloatingPointError("Sparse histogram spectrum contains nonfinite values")
        return output
