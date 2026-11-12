"""Bind NumPy layered coordinates and prefix integration to their prepared state."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from ...file_provenance import file_sha256
from .prefix_tables import select_expansion_workers
from .prefix_arithmetic import check_block, runtime_signature as prefix_runtime_signature, synthetic_weights, table_signature

RULE = "fixed-log-sigma-layers-with-proportional-mu-spacing-v1"
PROFILE_COUNT = 40
TEMPLATE_SCRATCH_BYTES = 64 * 1024**2


def runtime_signature(threads):
    return {
        "implementation": "numpy",
        "prefix_runtime": prefix_runtime_signature(threads),
        "layered_source_sha256": file_sha256(Path(__file__).with_name("layered_nodes.py")),
        "layered_geometry_sha256": file_sha256(Path(__file__).with_name("layered_geometry.py")),
    }


def coordinate_signature(grid):
    return {"rule": RULE, "parameters": asdict(grid)}


def run_gate(grid, table, *, threads=0):
    """Validate synthetic logical-node coefficients against independent np.interp."""
    began = perf_counter()
    workers = select_expansion_workers(threads)
    rng = np.random.default_rng(921731)
    cases = []
    for count in (1, 17, 64, 129, 513):
        layers = rng.integers(0, 720, size=count, dtype=np.int64)
        layers[:min(count, 4)] = np.array([0, 1, 400, 719])[:min(count, 4)]
        sigmas = grid.sigma_values(layers)
        means = rng.uniform(table.edges[0] - 40 * table.delta_v, table.edges[-1] + 40 * table.delta_v, count)
        indices = np.floor((means - grid.mu_anchor) / (grid.mu_spacing_ratio * sigmas))
        if not np.isfinite(indices).all() or np.any(np.abs(indices) >= 2**62):
            raise ValueError("Synthetic logical indices must be safely representable")
        keys = np.column_stack((layers, indices.astype(np.int64)))
        centers, widths = grid.coordinates(keys)
        weights = synthetic_weights(rng, count)
        if count == 1:
            weights[:] = 1.0
        cases.append(check_block(table, centers, widths, weights, threads=workers, label=f"layered_nodes_{count}"))
    if not all(case["modes"]["numpy_einsum"]["passed"] for case in cases):
        raise RuntimeError("Layered prefix interpolation differs from its independent reference")
    return {
        "schema_version": 2, "complete": True, "implementation": "numpy",
        "coordinate_signature": coordinate_signature(grid), "runtime_signature": runtime_signature(workers),
        "table_signature": table_signature(table), "selected_backend": "numpy_einsum",
        "workers": workers, "supported_profiles": PROFILE_COUNT,
        "template_scratch_bytes": TEMPLATE_SCRATCH_BYTES, "cases": cases,
        "snapshot_payload_reads": 0, "emission_table_queries": 0,
        "numeric_tolerance": "5e-12 of reference maximum; independent same-table interpolation",
        "preparation_seconds_outside_process": perf_counter() - began,
    }


def validate_gate(result, grid, table, *, threads):
    expected = {
        "schema_version": 2, "complete": True, "implementation": "numpy",
        "coordinate_signature": coordinate_signature(grid), "runtime_signature": runtime_signature(threads),
        "table_signature": table_signature(table), "selected_backend": "numpy_einsum",
        "workers": select_expansion_workers(threads), "supported_profiles": PROFILE_COUNT,
        "template_scratch_bytes": TEMPLATE_SCRATCH_BYTES,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Layered validation belongs to different NumPy code, rules or prefix table")
    cases = result.get("cases", [])
    if not cases or not all(case["modes"]["numpy_einsum"]["passed"] for case in cases):
        raise ValueError("The layered prefix coefficient fixtures did not pass")
    return "numpy_einsum"
