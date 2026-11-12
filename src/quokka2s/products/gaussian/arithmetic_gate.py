"""Validate A's fractional-shift/sigma table with independent SciPy interpolation."""
from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def run_gate(grid, edges, *, engines=None, bundle=None):
    """Check the prepared A coefficients; no native/FMA arithmetic is selected."""
    from .engine import GaussianHistogramEngine
    from .tables import prepare_tables
    from .portable_expansion import array_sha256, band_geometry, runtime_signature

    began = perf_counter()
    edges = np.asarray(edges, dtype=np.float64)
    if engines is None:
        engines = bundle.engines if bundle is not None else GaussianHistogramEngine(grid, contraction="einsum").engines
    if bundle is None:
        bundle = prepare_tables(grid, edges, "A", engines=engines)
    if bundle.grid != grid or bundle.method != "A" or bundle.cutoff_sigma != 5.0 or not np.array_equal(edges, bundle.edges):
        raise ValueError("A validation requires its exact geometry, channels and five-sigma table")
    rng = np.random.default_rng(90387)
    cases = []
    for band_index, (band, table) in enumerate(zip(engines, bundle.tables)):
        count = min(129, band.node_count)
        sigma_ids = rng.integers(0, band.sigma_nodes.size, size=count)
        mu_ids = rng.integers(0, band.mu_nodes.size, size=count)
        centers = band.mu_nodes[mu_ids]
        sigmas = band.sigma_nodes[sigma_ids]
        actual = table.evaluate(centers, sigmas, sigma_id=sigma_ids)
        coordinate = centers / table.grid.delta_v
        integer_mu = np.floor(coordinate).astype(np.int64)
        fraction = coordinate - integer_mu
        channel = table.n_max - integer_mu[:, None] + np.arange(table.grid.K)
        sampler = RegularGridInterpolator((table.sigma_nodes, table.fraction_nodes, np.arange(table.M_A)), table.T, bounds_error=True)
        shape = channel.shape
        points = np.stack((np.broadcast_to(sigmas[:, None], shape), np.broadcast_to(fraction[:, None], shape), channel), axis=-1)
        expected = sampler(points)
        maximum = float(np.max(np.abs(actual - expected), initial=0.0))
        scale = float(np.max(np.abs(expected), initial=0.0))
        relative = maximum / scale if scale else maximum
        passed = bool(np.isfinite(actual).all() and np.all(actual >= 0.0) and relative <= 5e-12)
        cases.append({"band": band_index, "nodes": count, "passed": passed, "byte_equal": actual.tobytes() == expected.tobytes(), "maximum_difference_over_reference_maximum": relative})
    if not all(case["passed"] for case in cases):
        raise RuntimeError("A-template coefficients differ from independent table interpolation")
    return {
        "schema_version": 2, "complete": True, "implementation": "numpy",
        "selected_backend": "numpy_einsum", "supported_profiles": 40, "cutoff_sigma": 5.0,
        "runtime_signature": runtime_signature(), "grid": asdict(grid),
        "edges_sha256": array_sha256(edges), "table_geometry": band_geometry(engines),
        "cases": cases, "numeric_tolerance": "5e-12 of reference maximum; same-table interpolation",
        "snapshot_payload_reads": 0, "emission_table_queries": 0,
        "gate_wall_seconds_outside_process": perf_counter() - began,
    }
