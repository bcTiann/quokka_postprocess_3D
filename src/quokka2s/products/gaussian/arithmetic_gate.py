"""Gate P=40 sparse arithmetic against this host's original ABC einsum.

No snapshot or emission table is read. A single prepared Gaussian A bundle is
shared by both candidate arithmetic modes; callers may pass their ready bundle.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np


def byte_comparison(expected, actual):
    equal = expected.shape == actual.shape and expected.dtype == actual.dtype
    equal = equal and np.array_equal(expected.view(np.uint64), actual.view(np.uint64))
    result = {"byte_equal": bool(equal)}
    if not equal:
        difference = expected.reshape(-1).view(np.uint64) != actual.reshape(-1).view(np.uint64)
        indices = np.flatnonzero(difference)
        first = int(indices[0])
        result.update({
            "differing_values": int(indices.size),
            "first_difference_flat_index": first,
            "expected_hex": float(expected.reshape(-1)[first]).hex(),
            "actual_hex": float(actual.reshape(-1)[first]).hex(),
        })
    return result


def source_slice_probabilities(table, centers, sigma_ids):
    """Read aligned original A slices independently of the sparse contraction.

    Exact stored sigma/fraction nodes are memcpy cases in native_a. If the
    geometry is unaligned, retain the original evaluator as the dense reference;
    the sparse candidate must reject rather than silently round that geometry.
    """
    coordinate = centers / table.grid.delta_v
    integer_mu = np.floor(coordinate).astype(np.int64)
    fractional_ids = np.rint((coordinate - integer_mu) * table.R).astype(np.int64)
    aligned = np.array_equal(coordinate, integer_mu + fractional_ids / table.R)
    sigmas = table.sigma_nodes[sigma_ids]
    original_rows = table.evaluate(centers, sigmas, sigma_id=sigma_ids)
    if not aligned:
        return original_rows, False
    rows = np.empty_like(original_rows)
    for index in range(centers.size):
        offset = table.n_max - integer_mu[index]
        rows[index] = table.T[
            sigma_ids[index], fractional_ids[index], offset:offset + table.grid.K,
        ]
    if not np.array_equal(rows.view(np.uint64), original_rows.view(np.uint64)):
        raise AssertionError("Direct A source slices differ from the original native evaluator")
    return rows, True


def synthetic_block(band, edges, count, seed):
    """Use actual grid nodes, sparse positive P40 weights and support edges."""
    rng = np.random.default_rng(seed)
    delta_v = float(edges[1] - edges[0])
    near_window = np.flatnonzero(
        (band.mu_nodes >= edges[0] - 2.0 * delta_v)
        & (band.mu_nodes <= edges[-1] + 2.0 * delta_v)
    )
    if near_window.size == 0:
        near_window = np.arange(band.mu_nodes.size)
    sigma_ids = rng.integers(0, band.sigma_nodes.size, size=count)
    mu_ids = rng.choice(near_window, size=count)
    edge_mu = np.array([
        band.mu_nodes[0], band.mu_nodes[-1], edges[0], edges[-1],
        -0.5 * delta_v, 0.0, 0.5 * delta_v,
        edges[0] + 0.125 * delta_v, edges[-1] - 0.125 * delta_v,
    ])
    for index, value in enumerate(edge_mu[:count]):
        right = min(int(np.searchsorted(band.mu_nodes, value)), band.mu_nodes.size - 1)
        left = max(right - 1, 0)
        mu_ids[index] = left if abs(band.mu_nodes[left] - value) < abs(band.mu_nodes[right] - value) else right
        sigma_ids[index] = 0 if index % 2 else band.sigma_nodes.size - 1
    ids = np.sort(sigma_ids * band.mu_nodes.size + mu_ids).astype(np.int64)
    weights = np.exp(rng.uniform(-12.0, 12.0, size=(count, 40)))
    weights[rng.random(weights.shape) < 0.65] = 0.0
    weights[:, 0] = 0.0
    weights[::37] = 0.0
    return ids, np.ascontiguousarray(weights)


def check_block(band_index, band, table, ids, weights, name):
    from ._sparse_expand import expand_block

    sigma_ids = np.ascontiguousarray(ids // band.mu_nodes.size, dtype=np.int64)
    centers = np.ascontiguousarray(band.mu_nodes[ids % band.mu_nodes.size])
    sigmas = np.ascontiguousarray(band.sigma_nodes[sigma_ids])
    rows, aligned = source_slice_probabilities(table, centers, sigma_ids)
    templates = np.ascontiguousarray(rows.T)
    expected = np.einsum("ki,ip->kp", templates, weights, optimize=False)
    if not np.isfinite(expected).all():
        raise FloatingPointError("The original reference contraction is nonfinite")
    modes = {}
    for backend, use_fma in (("sparse_fma", True), ("sparse_separate_muladd", False)):
        contribution = np.zeros_like(expected)
        try:
            counts = expand_block(
                centers, sigmas, sigma_ids, weights, table.T, contribution,
                table.grid.v_min, table.grid.delta_v, table.n_max, table.R, 5.0,
                use_fma,
            )
            comparison = byte_comparison(expected, contribution)
            comparison["counts"] = counts
        except ValueError as error:
            comparison = {"byte_equal": False, "rejected_geometry": str(error)}
        modes[backend] = comparison
    return {
        "name": name,
        "band": band_index,
        "nodes": int(ids.size),
        "bins": int(table.grid.K),
        "profiles": 40,
        "original_layout": "C bins-by-nodes templates and C nodes-by-40 weights",
        "source_slices_equal_original_native_a": aligned,
        "zero_weights_and_zero_templates_included": True,
        "modes": modes,
    }


def run_gate(grid, edges, *, engines=None, bundle=None):
    """Return a runtime-bound decision; reuse the supplied one A table bundle.

    The gate checks partial and full original-size blocks in both bands. It
    selects an exact mode only when every block matches. Otherwise the declared
    backend is the original dense ABC einsum, with no numerical tolerance.
    """
    from .engine import GaussianHistogramEngine as OriginalABCEngine
    from .tables import prepare_tables
    from .portable_expansion import array_sha256, band_geometry, runtime_signature

    started = time.perf_counter()
    edges = np.ascontiguousarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0.0):
        raise ValueError("Gate velocity edges must be finite and increasing")
    bundle_reused = bundle is not None
    if engines is None:
        if bundle is not None:
            engines = bundle.engines
        else:
            engines = OriginalABCEngine(grid, contraction="einsum").engines
    if len(engines) != 2:
        raise ValueError("The gate requires the original narrow/broad engine pair")
    if bundle is None:
        bundle = prepare_tables(grid, edges, "A", engines=engines)
    if bundle.grid != grid or bundle.method != "A" or bundle.cutoff_sigma != 5.0:
        raise ValueError("The gate requires the unchanged A/5-sigma bundle")
    if not np.array_equal(edges, bundle.edges):
        raise ValueError("Gate edges must match the prepared lookup bundle")
    cases = []
    for band_index, band in enumerate(engines):
        table = bundle.tables[band_index]
        if not np.array_equal(band.sigma_nodes, table.sigma_nodes):
            raise ValueError("Gate sigma nodes differ from the actual A table nodes")
        bytes_per_node = (edges.size + edges.size - 1) * np.dtype(np.float64).itemsize
        block_nodes = band.template_scratch_bytes // bytes_per_node
        if block_nodes < 1:
            raise ValueError("The original template block must hold at least one node")
        sizes = (min(513, band.node_count, block_nodes), min(block_nodes, band.node_count))
        for case_index, count in enumerate(dict.fromkeys(sizes)):
            ids, weights = synthetic_block(band, edges, count, 90387 + 10 * band_index + case_index)
            cases.append(check_block(
                band_index, band, table, ids, weights,
                "original_full_block" if count == block_nodes else "partial_sparse_block",
            ))
    matches = {
        backend: all(case["modes"][backend]["byte_equal"] for case in cases)
        for backend in ("sparse_fma", "sparse_separate_muladd")
    }
    if matches["sparse_fma"]:
        selected = "sparse_fma"
        reason = "Explicit float64 FMA matches every tested original P40 einsum block byte-for-byte"
    elif matches["sparse_separate_muladd"]:
        selected = "sparse_separate_muladd"
        reason = "Separate float64 multiply/add matches every tested original P40 einsum block byte-for-byte"
    else:
        selected = "numpy_einsum"
        reason = "Neither sparse arithmetic matches this runtime; retain original ABC dense einsum and block order"
    return {
        "schema_version": 1,
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_backend": selected,
        "selection_reason": reason,
        "matching_sparse_modes": matches,
        "runtime_signature": runtime_signature(),
        "grid": asdict(grid),
        "edges_sha256": array_sha256(edges),
        "velocity_bins": int(edges.size - 1),
        "table_geometry": band_geometry(engines),
        "supported_profiles": 40,
        "cutoff_sigma": 5.0,
        "renormalized": False,
        "cases": cases,
        "bundle_reused": bundle_reused,
        "A_bundle_build_count_in_gate": 0 if bundle_reused else 1,
        "gate_wall_seconds_outside_process": time.perf_counter() - started,
        "snapshot_payload_reads": 0,
        "emission_table_queries": 0,
        "tolerance": "none; original dtype/shape/bytes must match",
        "cross_platform_byte_equality_claimed": False,
    }
