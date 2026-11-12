"""Validate the NumPy prefix coefficients and bind their prepared table state."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy

from ...file_provenance import file_sha256
from .prefix_tables import select_expansion_workers

METHOD = "standard-normal-prefix"
PROFILE_COUNT = 40
BACKENDS = ("numpy_einsum",)


def array_identity(values):
    array = np.ascontiguousarray(values)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": hashlib.sha256(memoryview(array)).hexdigest()}


def table_signature(table):
    """Bind immutable values, fine-cell spacing and exact observation edges."""
    if table.method != METHOD or table.Z != 5.0 or table.cutoff_sigma != table.Z:
        raise ValueError("The prefix validation requires the adopted five-sigma table")
    if table.values.flags.writeable or table.edges.flags.writeable:
        raise ValueError("Prepared prefix values and edges must be read-only")
    return {
        "method": table.method,
        "cutoff_sigma": table.Z,
        "fine_cell_width": table.h,
        "fine_cells": table.M,
        "channel_width_kms": table.delta_v,
        "values": array_identity(table.values),
        "edges": array_identity(table.edges),
    }


def runtime_signature(threads=0):
    return {
        "implementation": "numpy",
        "workers": select_expansion_workers(threads),
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": list(sys.version_info[:3]),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "prefix_source_sha256": file_sha256(Path(__file__).with_name("prefix_tables.py")),
        "validation_source_sha256": file_sha256(__file__),
        "custom_compiled_kernel": False,
    }


def band_geometry(engines):
    return [{
        "mu_nodes": array_identity(band.mu_nodes),
        "sigma_nodes": array_identity(band.sigma_nodes),
        "log_sigma_nodes": array_identity(band.log_sigma_nodes),
        "template_scratch_bytes": int(band.template_scratch_bytes),
    } for band in engines]


def reference_probabilities(table, centers, sigmas):
    """Independent np.interp reading of the same fine-cell cumulative table."""
    centers = np.asarray(centers, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    inverse = np.reciprocal(sigmas)
    first = (table.edges[0] - centers) * inverse
    increment = table.delta_v * inverse
    boundaries = first[:, None] + np.arange(table.edges.size)[None, :] * increment[:, None]
    standard_edges = -table.Z + np.arange(table.M + 1) * table.h
    standard_edges[-1] = table.Z
    cumulative = np.interp(boundaries, standard_edges, table.values)
    return np.diff(cumulative, axis=1)


def synthetic_weights(rng, count):
    weights = np.exp(rng.uniform(-12.0, 12.0, size=(count, PROFILE_COUNT)))
    weights[rng.random(weights.shape) < 0.65] = 0.0
    weights[:, 0] = 0.0
    weights[::37] = 0.0
    return np.ascontiguousarray(weights)


def check_block(table, centers, sigmas, weights, *, threads=0, label):
    """Check independent prefix interpolation; no FMA/native-team selection."""
    probabilities = reference_probabilities(table, centers, sigmas)
    expected = np.einsum("ki,ip->kp", np.ascontiguousarray(probabilities.T), weights, optimize=False)
    actual = np.zeros_like(expected)
    counts = table.expand(centers, sigmas, weights, actual, threads=threads)
    maximum = float(np.max(np.abs(actual - expected), initial=0.0))
    scale = float(np.max(np.abs(expected), initial=0.0))
    relative = maximum / scale if scale else maximum
    passed = bool(np.isfinite(actual).all() and np.all(actual >= 0.0) and relative <= 5e-12)
    return {
        "name": label, "nodes": int(centers.size), "bins": table.edges.size - 1,
        "profiles": PROFILE_COUNT,
        "modes": {"numpy_einsum": {"passed": passed, "byte_equal": actual.tobytes() == expected.tobytes(), "maximum_absolute_difference": maximum, "maximum_difference_over_reference_maximum": relative, "counts": counts}},
    }


def run_gate(grid, edges, *, engines, table, threads=0):
    """Check small independent fixtures before process; retain one NumPy backend."""
    began = perf_counter()
    workers = select_expansion_workers(threads)
    if len(engines) != 2 or not np.array_equal(np.asarray(edges), table.edges):
        raise ValueError("Prefix validation requires both bands and the exact prepared edges")
    rng = np.random.default_rng(29171)
    cases = []
    for count in (1, 17, 64, 129):
        centers = rng.uniform(table.edges[0] - 40 * table.delta_v, table.edges[-1] + 40 * table.delta_v, count)
        sigmas = np.exp(rng.uniform(np.log(grid.sigma_min), np.log(grid.sigma_max), count))
        weights = synthetic_weights(rng, count)
        if count == 1:
            weights[:] = 1.0
        cases.append(check_block(table, centers, sigmas, weights, threads=workers, label=f"prefix_nodes_{count}"))
    if not all(case["modes"]["numpy_einsum"]["passed"] for case in cases):
        raise RuntimeError("NumPy prefix interpolation differs from its independent fine-cell reference")
    return {
        "schema_version": 2, "complete": True, "method": METHOD,
        "selected_backend": "numpy_einsum", "implementation": "numpy",
        "supported_profiles": PROFILE_COUNT, "workers": workers,
        "runtime_signature": runtime_signature(workers), "grid": asdict(grid),
        "table_geometry": band_geometry(engines), "table_signature": table_signature(table),
        "velocity_bins": table.edges.size - 1, "cutoff_sigma": table.Z, "renormalized": False,
        "cases": cases, "numeric_tolerance": "5e-12 of reference maximum; independent same-table interpolation",
        "snapshot_payload_reads": 0, "emission_table_queries": 0,
        "gate_wall_seconds_outside_process": perf_counter() - began,
    }


def validate_gate(result, grid, engines, table, *, threads):
    expected = {
        "schema_version": 2, "complete": True, "method": METHOD,
        "selected_backend": "numpy_einsum", "supported_profiles": PROFILE_COUNT,
        "workers": select_expansion_workers(threads), "runtime_signature": runtime_signature(threads),
        "grid": asdict(grid), "table_geometry": band_geometry(engines),
        "table_signature": table_signature(table), "velocity_bins": table.edges.size - 1,
        "cutoff_sigma": table.Z, "renormalized": False,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Prefix validation belongs to different NumPy code, geometry or table state")
    cases = result.get("cases", [])
    if not cases or not all(case["modes"]["numpy_einsum"]["passed"] for case in cases):
        raise ValueError("The prefix coefficient fixtures did not pass")
    return "numpy_einsum"
