"""Bind layered-node expansion to an exact prefix arithmetic gate."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from ...file_provenance import file_sha256
from . import _layered_deposit as layered_native
from . import _prefix_expand as prefix_native
from .prefix_arithmetic import (
    BACKENDS,
    check_block,
    runtime_signature as prefix_runtime_signature,
    synthetic_weights,
    table_signature,
)


RULE = "fixed-log-sigma-layers-with-proportional-mu-spacing-v1"
PROFILE_COUNT = 40
TEMPLATE_SCRATCH_BYTES = 64 * 1024**2


def runtime_signature(threads):
    """Identify the actual integration and logical-coordinate implementations."""
    return {
        "prefix_runtime": prefix_runtime_signature(threads),
        "layered_binary_sha256": file_sha256(layered_native.__file__),
        "layered_gate_source_sha256": file_sha256(__file__),
        "layered_geometry_source_sha256": file_sha256(Path(__file__).with_name("layered_geometry.py")),
    }


def coordinate_signature(grid):
    """Keep node identity independent of creation order and occupied extent."""
    return {"rule": RULE, "parameters": asdict(grid)}


def run_gate(grid, table, *, threads=0):
    """Check synthetic logical nodes before process, without any snapshot reads."""
    began = time.perf_counter()
    selected_threads = prefix_native.select_threads(threads)
    signature = table_signature(table)
    rng = np.random.default_rng(921731)
    bytes_per_node = (table.edges.size + table.edges.size - 1) * 8
    block_nodes = TEMPLATE_SCRATCH_BYTES // bytes_per_node
    if block_nodes < 1:
        raise ValueError("A layered expansion block must hold at least one node")
    cases = []
    for count in (1, 17, 64, 129, block_nodes):
        # Includes widths below/above the old bounded grid. Physical floor is
        # exact at layer zero; no finite rectangular parameter axes are built.
        layers = rng.integers(low=0, high=720, size=count, dtype=np.int64)
        layers[:min(count, 4)] = np.array([0, 1, 400, 719])[:min(count, 4)]
        provisional = np.column_stack((layers, np.zeros(count, dtype=np.int64)))
        _, sigmas = grid.coordinates(provisional)
        means = rng.uniform(
            low=float(table.edges[0]) - 40.0 * table.delta_v,
            high=float(table.edges[-1]) + 40.0 * table.delta_v,
            size=count,
        )
        indices = np.floor((means - grid.mu_anchor) / (grid.mu_spacing_ratio * sigmas))
        if not np.isfinite(indices).all() or np.any(np.abs(indices) >= 2**62):
            raise ValueError("Synthetic logical indices must be safely representable")
        keys = np.ascontiguousarray(np.column_stack((layers, indices.astype(np.int64))))
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        keys = np.ascontiguousarray(keys[order])
        centers, widths = grid.coordinates(keys)
        weights = synthetic_weights(rng, count)
        if count == 1:
            weights[:] = 1.0
        cases.append(check_block(
            table,
            centers,
            widths,
            weights,
            threads=selected_threads,
            label=f"layered_logical_nodes_{count}",
        ))
    matches = {
        backend: all(case["modes"][backend]["byte_equal"] for case in cases)
        for backend in BACKENDS
    }
    selected = next((backend for backend in BACKENDS if matches[backend]), None)
    if selected is None:
        raise RuntimeError("Neither layered prefix arithmetic mode matches NumPy exactly")
    return {
        "schema_version": 1,
        "complete": True,
        "coordinate_signature": coordinate_signature(grid),
        "runtime_signature": runtime_signature(selected_threads),
        "table_signature": signature,
        "selected_backend": selected,
        "match_numpy_fma": selected == "prefix_fma",
        "native_channel_threads": selected_threads,
        "supported_profiles": PROFILE_COUNT,
        "template_scratch_bytes": TEMPLATE_SCRATCH_BYTES,
        "cases": cases,
        "matching_modes": matches,
        "snapshot_payload_reads": 0,
        "emission_table_queries": 0,
        "tolerance": "none; selected prefix coefficient contraction must be byte exact",
        "preparation_seconds_outside_process": time.perf_counter() - began,
    }


def validate_gate(result, grid, table, *, threads):
    """Reject a gate with different rules, tables, binaries or actual team."""
    if result.get("schema_version") != 1 or result.get("complete") is not True:
        raise ValueError("A completed layered prefix arithmetic gate is required")
    if result.get("coordinate_signature") != coordinate_signature(grid):
        raise ValueError("Layered node coordinate rules differ from their arithmetic gate")
    if result.get("runtime_signature") != runtime_signature(threads):
        raise ValueError("Layered prefix arithmetic belongs to different runtime binaries")
    if result.get("table_signature") != table_signature(table):
        raise ValueError("The layered arithmetic gate used different prefix values or channels")
    if result.get("native_channel_threads") != threads or result.get("supported_profiles") != PROFILE_COUNT:
        raise ValueError("The layered gate belongs to another team or account layout")
    if result.get("template_scratch_bytes") != TEMPLATE_SCRATCH_BYTES:
        raise ValueError("The layered gate used a different original block budget")
    selected = result.get("selected_backend")
    cases = result.get("cases", [])
    if selected not in BACKENDS or not cases:
        raise ValueError("The layered gate has no supported arithmetic decision")
    if not all(case["modes"][selected]["byte_equal"] for case in cases):
        raise ValueError("The selected layered arithmetic did not pass every fixture")
    if result.get("match_numpy_fma") != (selected == "prefix_fma"):
        raise ValueError("The layered arithmetic flag disagrees with its backend")
    return selected
