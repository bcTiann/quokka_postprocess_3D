"""Validate prefix-table contraction against this runtime's NumPy arithmetic.

Only synthetic Gaussian nodes are used. The prepared prefix samples, actual
native binary, NumPy CPU dispatch, two-band geometry and channel edges are
bound to the result before the numerical process timer starts.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib
import os
import platform
import sys
import time

import numpy as np
import scipy

from . import _prefix_expand as native
from ...file_provenance import file_sha256


METHOD = "standard-normal-prefix"
PROFILE_COUNT = 40
BACKENDS = ("prefix_separate_muladd", "prefix_fma")


def array_identity(values):
    """Describe the exact array bytes used by preparation or expansion."""
    array = np.ascontiguousarray(values)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(memoryview(array)).hexdigest(),
    }


def table_signature(table):
    """Bind immutable prefix samples and velocity edges to the selected rule."""
    if table.method != METHOD or table.Z != 5.0 or table.cutoff_sigma != table.Z:
        raise ValueError("The prefix gate requires the adopted five-sigma table")
    if table.values.flags.writeable or table.edges.flags.writeable:
        raise ValueError("Prepared prefix values and velocity edges must be read-only")
    return {
        "method": table.method,
        "cutoff_sigma": table.Z,
        "fine_cell_width": table.h,
        "fine_cells": table.M,
        "channel_width_kms": table.delta_v,
        "values": array_identity(table.values),
        "edges": array_identity(table.edges),
    }


def runtime_signature(threads):
    """Identify this host's loaded prefix/NumPy kernels and actual native team."""
    try:
        core = importlib.import_module("numpy._core._multiarray_umath")
    except ModuleNotFoundError:
        core = importlib.import_module("numpy.core._multiarray_umath")
    features = {
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
        "numpy_cpu_features": features,
        "numpy_disable_cpu_features": os.environ.get("NPY_DISABLE_CPU_FEATURES"),
        "numpy_einsum_binary_sha256": file_sha256(core.__file__),
        "prefix_binary_sha256": file_sha256(native.__file__),
        "prefix_reference_source_sha256": file_sha256(__file__),
        "prefix_threading": native.kernel_info(threads),
    }


def band_geometry(engines):
    """Record the original narrow/broad axes and accumulation block budgets."""
    return [
        {
            "mu_nodes": array_identity(band.mu_nodes),
            "sigma_nodes": array_identity(band.sigma_nodes),
            "log_sigma_nodes": array_identity(band.log_sigma_nodes),
            "template_scratch_bytes": int(band.template_scratch_bytes),
        }
        for band in engines
    ]


def reference_probabilities(table, centers, sigmas):
    """Evaluate independent NumPy prefix coefficients for synthetic gate nodes.

    Inputs are means and positive standard deviations [N], in km/s. Output is
    dimensionless channel mass [N,K]. Every boundary uses first+j*increment,
    preserving the native non-iterative endpoint arithmetic. Cumulative values
    outside +/-5 remain constant, so adjacent differences implement truncation
    without renormalization or an online CDF call.
    """
    centers = np.asarray(centers, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    if centers.ndim != 1 or sigmas.shape != centers.shape:
        raise ValueError("Gate Gaussian parameters must be equal vectors")
    if not np.isfinite(centers).all() or not np.isfinite(sigmas).all() or np.any(sigmas <= 0.0):
        raise ValueError("Gate Gaussian parameters must be finite with positive widths")
    inverse = np.reciprocal(sigmas)
    first = (table.edges[0] - centers) * inverse
    increment = table.delta_v * inverse
    channels = np.arange(table.edges.size, dtype=np.float64)
    boundaries = first[:, None] + channels[None, :] * increment[:, None]
    if not np.isfinite(boundaries).all():
        raise ValueError("Synthetic gate boundaries must be representable")
    positions = (np.clip(boundaries, -table.Z, table.Z) + table.Z) * (1.0 / table.h)
    indices = np.minimum(positions.astype(np.int64), table.M - 1)
    fractions = positions - indices
    lower = table.values[indices]
    upper = table.values[indices + 1]
    cumulative = lower + fractions * (upper - lower)
    cumulative[boundaries <= -table.Z] = table.values[0]
    cumulative[boundaries >= table.Z] = table.values[-1]
    return np.diff(cumulative, axis=1)


def synthetic_weights(rng, count):
    """Use positive sparse accounts, entirely empty profiles and empty nodes."""
    log_weights = rng.uniform(
        low=-12.0,
        high=12.0,
        size=(count, PROFILE_COUNT),
    )
    weights = np.exp(log_weights)
    weights[rng.random(weights.shape) < 0.65] = 0.0
    weights[:, 0] = 0.0
    weights[::37] = 0.0
    return np.ascontiguousarray(weights)


def check_block(table, centers, sigmas, weights, *, threads, label):
    """Compare separate and FMA contraction with the same reference coefficients."""
    probabilities = reference_probabilities(table, centers, sigmas)
    expected = np.einsum(
        "ki,ip->kp",
        np.ascontiguousarray(probabilities.T),
        weights,
        optimize=False,
    )
    if not np.isfinite(expected).all():
        raise FloatingPointError("The synthetic prefix contraction is nonfinite")
    modes = {}
    for backend, use_fma in ((BACKENDS[0], False), (BACKENDS[1], True)):
        actual = np.zeros_like(expected)
        counts = table.expand(
            centers,
            sigmas,
            weights,
            actual,
            fma=use_fma,
            threads=threads,
        )
        modes[backend] = {
            "byte_equal": actual.tobytes() == expected.tobytes(),
            "maximum_absolute_difference": float(np.max(np.abs(actual - expected), initial=0.0)),
            "counts": counts,
        }
    return {
        "name": label,
        "nodes": int(centers.size),
        "bins": int(table.edges.size - 1),
        "profiles": PROFILE_COUNT,
        "modes": modes,
    }


def run_gate(grid, edges, *, engines, table, threads=0):
    """Select exact contraction using partial and full original-size blocks.

    Preparation reads no snapshot, physical emission table or actual H. If this
    runtime matches neither native arithmetic mode, raise before processing;
    a different Gaussian algorithm is never silently substituted.
    """
    began = time.perf_counter()
    selected_threads = native.select_threads(threads)
    if len(engines) != 2:
        raise ValueError("The prefix gate requires both original sigma bands")
    if not np.array_equal(np.asarray(edges), table.edges):
        raise ValueError("Gate velocity edges must match the prepared prefix table")
    signature = table_signature(table)
    rng = np.random.default_rng(29171)
    cases = []
    margin = 40.0 * table.delta_v
    minimum_mean = table.edges[0] - margin
    maximum_mean = table.edges[-1] + margin
    minimum_log_sigma = np.log(grid.sigma_min)
    maximum_log_sigma = np.log(grid.sigma_max)
    for count in (1, 17, 64, 129):
        centers = rng.uniform(
            low=minimum_mean,
            high=maximum_mean,
            size=count,
        )
        log_sigmas = rng.uniform(
            low=minimum_log_sigma,
            high=maximum_log_sigma,
            size=count,
        )
        sigmas = np.exp(log_sigmas)
        centers = np.ascontiguousarray(centers)
        sigmas = np.ascontiguousarray(sigmas)
        weights = synthetic_weights(rng, count)
        if count == 1:
            weights[:] = 1.0
        cases.append(check_block(
            table,
            centers,
            sigmas,
            weights,
            threads=selected_threads,
            label=f"partial_{count}",
        ))
    for band_index, band in enumerate(engines):
        bytes_per_node = (table.edges.size + table.edges.size - 1) * np.dtype(np.float64).itemsize
        block_nodes = band.template_scratch_bytes // bytes_per_node
        if block_nodes < 1:
            raise ValueError("The original expansion block must hold at least one node")
        count = min(block_nodes, band.node_count)
        ids = rng.integers(low=0, high=band.node_count, size=count)
        ids = np.sort(ids)
        sigma_ids = ids // band.mu_nodes.size
        centers = np.ascontiguousarray(band.mu_nodes[ids % band.mu_nodes.size])
        sigmas = np.ascontiguousarray(band.sigma_nodes[sigma_ids])
        weights = synthetic_weights(rng, count)
        cases.append(check_block(
            table,
            centers,
            sigmas,
            weights,
            threads=selected_threads,
            label=f"band_{band_index}_original_block_{count}",
        ))
    matches = {
        backend: all(case["modes"][backend]["byte_equal"] for case in cases)
        for backend in BACKENDS
    }
    selected = next((backend for backend in BACKENDS if matches[backend]), None)
    if selected is None:
        raise RuntimeError("Neither prefix arithmetic mode matches this NumPy contraction exactly")
    return {
        "schema_version": 1,
        "complete": True,
        "method": METHOD,
        "selected_backend": selected,
        "match_numpy_fma": selected == "prefix_fma",
        "supported_profiles": PROFILE_COUNT,
        "runtime_signature": runtime_signature(selected_threads),
        "grid": asdict(grid),
        "table_geometry": band_geometry(engines),
        "table_signature": signature,
        "velocity_bins": int(table.edges.size - 1),
        "cutoff_sigma": table.Z,
        "renormalized": False,
        "native_channel_threads": selected_threads,
        "cases": cases,
        "matching_modes": matches,
        "numeric_tolerance": "none; byte equality against selected prefix coefficients",
        "snapshot_payload_reads": 0,
        "emission_table_queries": 0,
        "gate_wall_seconds_outside_process": time.perf_counter() - began,
        "cross_platform_byte_equality_claimed": False,
    }


def validate_gate(result, grid, engines, table, *, threads):
    """Reject stale binary, table, channel, geometry or CPU-dispatch decisions."""
    if result.get("schema_version") != 1 or result.get("complete") is not True:
        raise ValueError("A completed prefix arithmetic gate is required")
    selected = result.get("selected_backend")
    if selected not in BACKENDS or result.get("method") != METHOD:
        raise ValueError("The gate must select a supported prefix contraction")
    if result.get("supported_profiles") != PROFILE_COUNT:
        raise ValueError("The prefix gate requires the original 40-account layout")
    if result.get("velocity_bins") != table.edges.size - 1:
        raise ValueError("The prefix gate belongs to a different channel count")
    if result.get("cutoff_sigma") != table.Z or result.get("renormalized") is not False:
        raise ValueError("The prefix gate must retain unnormalized five-sigma clipping")
    if result.get("native_channel_threads") != threads:
        raise ValueError("The prefix gate belongs to a different native team")
    if result.get("runtime_signature") != runtime_signature(threads):
        raise ValueError("The prefix gate belongs to different runtime binaries or native threads")
    if result.get("grid") != asdict(grid) or result.get("table_geometry") != band_geometry(engines):
        raise ValueError("The prefix gate belongs to different Gaussian histogram geometry")
    if result.get("table_signature") != table_signature(table):
        raise ValueError("The prefix gate belongs to different lookup values or velocity edges")
    cases = result.get("cases", [])
    if not cases or not all(case["modes"][selected]["byte_equal"] for case in cases):
        raise ValueError("The selected prefix arithmetic did not pass every exact fixture")
    if result.get("match_numpy_fma") != (selected == "prefix_fma"):
        raise ValueError("The prefix gate arithmetic flag disagrees with its selected backend")
    return selected
