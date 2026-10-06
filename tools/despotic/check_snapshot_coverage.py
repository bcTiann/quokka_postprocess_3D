"""Measure raw DESPOTIC-table coverage of every cell in its source snapshot.

This is a diagnostic only: no table values, pipeline defaults, or snapshot
fields are changed. Invalid inputs are counted and never clipped into valid
queries. Valid out-of-bounds inputs are reported before applying the pipeline's
existing lookup-coordinate clipping. No failure-acceptance threshold is set.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from unyt import unyt_array

from quokka2s.despotic.table_files import load_table
from quokka2s.despotic.lookup import DespoticLookup
from quokka2s.despotic.snapshot_domain import AXIS_NAMES, validate_snapshot_domain
from quokka2s.file_provenance import file_sha256
from quokka2s.physics import gas_fields, settings
from quokka2s.snapshot_reader import Snapshot, slab_windows


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE = ROOT / "inputs/tables/despotic/raw.npz"
DEFAULT_DATASET = ROOT / "inputs/snapshots/plt0655228"


def _table_axes(table):
    axes = tuple(np.asarray(axis, dtype=float) for axis in (
        table.nH_values, table.col_density_values, table.dVdr_values))
    for name, axis in zip(AXIS_NAMES, axes):
        if (axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all()
                or np.any(axis <= 0) or np.any(np.diff(axis) <= 0)):
            raise ValueError(f"{name} requires at least two increasing positive finite nodes")
    shape = tuple(axis.size for axis in axes)
    if table.tg_final.shape != shape:
        raise ValueError("Table field shape does not match its axes")
    if table.failure_mask is None:
        raise ValueError("Raw failure_mask is required; missing provenance is not zero failures")
    return axes


def classify_queries(lookup, nH, NH, dVdr, *, clip_inputs):
    """Return per-query flags, retaining failure provenance and actual RGI results.

    clip_inputs is DespoticLookup.clip_coordinates in the snapshot scan (passed
    as an unbound method: lookup, nH, NH, dVdr). Tests may supply a recorder;
    invalid inputs are excluded before this function is called.
    """
    axes = _table_axes(lookup.table)
    inputs = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (nH, NH, dVdr)))
    shape = inputs[0].shape
    flat = tuple(v.ravel() for v in inputs)
    size = flat[0].size
    invalid = [~np.isfinite(v) | (v <= 0) for v in flat]
    valid = ~np.logical_or.reduce(invalid)
    flags = {f"invalid_{name}": mask for name, mask in zip(AXIS_NAMES, invalid)}
    flags["invalid_input"] = ~valid
    flags["valid_input"] = valid
    raw_outside = []
    for name, values, axis, invalid_axis in zip(AXIS_NAMES, flat, axes, invalid):
        below = ~invalid_axis & (values < axis[0])
        above = ~invalid_axis & (values > axis[-1])
        flags[f"raw_below_{name}"] = below
        flags[f"raw_above_{name}"] = above
        raw_outside.append(below | above)
    # Includes a finite out-of-range axis even if another axis is invalid.
    flags["raw_out_of_bounds"] = np.logical_or.reduce(raw_outside)
    names = (
        "positive_weight_failed_support", "any_failed_among_eight_corners",
        "all_eight_corners_failed", "temperature_query_nonfinite",
        "mu_query_nonfinite", "temperature_or_mu_query_nonfinite",
        "temperature_or_mu_query_nonpositive", "query_nonfinite_without_positive_failed_support",
    )
    flags.update({name: np.zeros(size, dtype=bool) for name in names})
    if np.any(valid):
        indices = np.flatnonzero(valid)
        safe = tuple(np.asarray(v, dtype=float) for v in clip_inputs(
            lookup, *(v[valid] for v in flat)))
        brackets, fractions = [], []
        for axis, values in zip(axes, safe):
            log_axis, point = np.log10(axis), np.log10(values)
            # SciPy RGI uses the interval to the right at exact interior nodes,
            # and the final interval at the upper boundary.
            lower = np.clip(np.searchsorted(log_axis, point, side="right") - 1,
                            0, log_axis.size - 2)
            fraction = (point - log_axis[lower]) / (log_axis[lower + 1] - log_axis[lower])
            brackets.append(lower)
            fractions.append(fraction)
        positive_failed = np.zeros(indices.size, dtype=bool)
        any_failed = np.zeros(indices.size, dtype=bool)
        all_failed = np.ones(indices.size, dtype=bool)
        for corner in itertools.product((0, 1), repeat=3):
            failed = lookup.table.failure_mask[tuple(
                lower + offset for lower, offset in zip(brackets, corner))]
            weight = np.ones(indices.size)
            for fraction, offset in zip(fractions, corner):
                weight *= fraction if offset else 1.0 - fraction
            positive_failed |= failed & (weight > 0.0)
            any_failed |= failed
            all_failed &= failed
        flags["positive_weight_failed_support"][indices] = positive_failed
        flags["any_failed_among_eight_corners"][indices] = any_failed
        flags["all_eight_corners_failed"][indices] = all_failed
        # Evaluate the real DespoticLookup. In three dimensions RGI may propagate
        # NaN from a zero-weight corner, so support flags alone are insufficient.
        queries = lookup.prepare_queries(
            hydrogen_density_cm3=safe[0],
            shielding_NH_cm2=safe[1],
            velocity_gradient_s=safe[2],
        )
        temperature = np.asarray(lookup.temperature(queries=queries), dtype=float)
        mu = np.asarray(lookup.mu(queries=queries), dtype=float)
        bad_t, bad_mu = ~np.isfinite(temperature), ~np.isfinite(mu)
        flags["temperature_query_nonfinite"][indices] = bad_t
        flags["mu_query_nonfinite"][indices] = bad_mu
        flags["temperature_or_mu_query_nonfinite"][indices] = bad_t | bad_mu
        flags["temperature_or_mu_query_nonpositive"][indices] = (
            (~bad_t & (temperature <= 0)) | (~bad_mu & (mu <= 0)))
        flags["query_nonfinite_without_positive_failed_support"][indices] = (
            (bad_t | bad_mu) & ~positive_failed)
    return {name: values.reshape(shape) for name, values in flags.items()}


class CoverageTotals:
    """Streaming counts and mass sums, with explicit denominators."""

    def __init__(self):
        self.groups = {name: dict(cell_count=0, valid_mass_cell_count=0,
                                  total_valid_mass_g=0.0, flags={})
                       for name in ("all_cells", "T_QUOKKA_lt_3000_K")}

    def add(self, flags, temperature_quokka, mass_g):
        temperature, mass = np.broadcast_arrays(np.asarray(temperature_quokka, dtype=float),
                                               np.asarray(mass_g, dtype=float))
        if any(np.shape(value) != temperature.shape for value in flags.values()):
            raise ValueError("Classification, temperature, and mass shapes must agree")
        valid_t = np.isfinite(temperature) & (temperature > 0)
        valid_mass = np.isfinite(mass) & (mass > 0)
        flags = dict(flags, invalid_temperature_quokka=~valid_t, invalid_mass=~valid_mass)
        for name, selection in (
            ("all_cells", np.ones(temperature.shape, dtype=bool)),
            ("T_QUOKKA_lt_3000_K", valid_t & (temperature < settings.EMISSION_TEMPERATURE_BOUNDARY_K)),
        ):
            group = self.groups[name]
            weighted = selection & valid_mass
            group["cell_count"] += int(np.count_nonzero(selection))
            group["valid_mass_cell_count"] += int(np.count_nonzero(weighted))
            group["total_valid_mass_g"] += float(np.sum(mass[weighted], dtype=np.float64))
            for flag, values in flags.items():
                record = group["flags"].setdefault(flag, dict(cell_count=0, mass_g=0.0))
                record["cell_count"] += int(np.count_nonzero(selection & values))
                record["mass_g"] += float(np.sum(mass[weighted & values], dtype=np.float64))

    def result(self):
        groups = {}
        for name, group in self.groups.items():
            count, mass = group["cell_count"], group["total_valid_mass_g"]
            groups[name] = {key: value for key, value in group.items() if key != "flags"}
            groups[name]["flags"] = {
                flag: dict(record,
                           cell_fraction=record["cell_count"] / count if count else None,
                           mass_fraction=record["mass_g"] / mass if mass else None)
                for flag, record in group["flags"].items()
            }
        return groups


def update_coordinate_extrema(extrema, coordinates):
    """Update three raw coordinate ranges using positive finite values only."""
    for name, values in zip(AXIS_NAMES, coordinates):
        valid = np.isfinite(values) & (values > 0)
        if not np.any(valid):
            continue
        limits = extrema[name]
        lower = float(values[valid].min())
        upper = float(values[valid].max())
        limits["minimum"] = lower if limits["minimum"] is None else min(lower, limits["minimum"])
        limits["maximum"] = upper if limits["maximum"] is None else max(upper, limits["maximum"])


def scan_snapshot_coverage(snapshot, lookup, slab_nx, query_chunk):
    """Read native slabs and accumulate coverage flags, masses and raw extrema.

    Only the current slab and batch views remain in memory. The query batching
    and CoverageTotals summation order match the original diagnostic.
    """
    totals = CoverageTotals()
    scanned = 0
    extrema = {name: dict(minimum=None, maximum=None) for name in AXIS_NAMES}

    for x_start, x_stop in slab_windows(
        x_start=0,
        x_stop=snapshot.shape[0],
        slab_nx=slab_nx,
    ):
        slab = snapshot.read_slab(x_start=x_start, x_stop=x_stop)
        for cells in slab.iter_batches(batch_size=query_chunk):

            # Preserve the original unit-aware nH normalization used to scan axes.
            density = unyt_array(cells.density_g_cm3, "g/cm**3")
            nH = gas_fields.hydrogen_number_density(density).to_value("cm**-3")
            coordinates = (nH, cells.shielding_NH_cm2, cells.velocity_gradient_s)
            update_coordinate_extrema(extrema=extrema, coordinates=coordinates)
            flags = classify_queries(
                lookup,
                *coordinates,
                clip_inputs=DespoticLookup.clip_coordinates,
            )
            mass_g = cells.density_g_cm3 * cells.cell_volume_cm3
            totals.add(flags, cells.temperature_QUOKKA_K, mass_g)
            scanned += cells.cell_count
            del density, nH, coordinates, flags, mass_g, cells
        del slab

        percentage = 100.0 * scanned / snapshot.cell_count
        print(
            f"Coverage checked {percentage:.1f}% "
            f"({scanned}/{snapshot.cell_count} cells; {x_stop}/{snapshot.shape[0]} x planes)",
            flush=True,
        )
    return totals, scanned, extrema


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slab-nx", type=int, default=8)
    parser.add_argument("--query-chunk", type=int, default=500_000)
    parser.add_argument("--expected-cells", type=int, default=134_217_728)
    args = parser.parse_args(argv)
    if not args.table.is_file():
        raise FileNotFoundError(f"Candidate table is not complete or is missing: {args.table}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite coverage report: {args.output}")
    if min(args.slab_nx, args.query_chunk, args.expected_cells) < 1:
        raise ValueError("Slab size, query chunk size, and expected cell count must be positive")

    import yt
    import scipy
    from quokka2s.despotic import lookup as lookup_module

    table = load_table(args.table)
    lookup = DespoticLookup(table)
    _table_axes(table)
    # A cleaned table may retain the old failure mask while filling values.
    # Require the candidate's raw failed T/mu outputs so this cannot report a
    # filled table as the requested raw candidate.
    if np.any(table.failure_mask & (
            np.isfinite(table.tg_final) | np.isfinite(table.mu_values))):
        raise ValueError("Failed nodes contain finite T or mu; supply the raw candidate table")
    dataset = args.dataset
    ds = yt.load(str(dataset.resolve()))
    if ds.max_level != 0 or settings.COLUMN_DENSITY_DIRECTIONS != "z":
        raise ValueError("Coverage requires a full-resolution uniform snapshot and full z columns")
    shape = tuple(int(value) for value in ds.domain_dimensions)
    if min(shape) < 2 or int(np.prod(shape)) != args.expected_cells:
        raise ValueError(f"Snapshot shape {shape} does not match expected all-cell count "
                         f"{args.expected_cells}")
    domain = validate_snapshot_domain(
        table=table,
        shape=shape,
        cfg=settings,
    )
    snapshot = Snapshot(dataset=ds)
    totals, scanned, extrema = scan_snapshot_coverage(
        snapshot=snapshot,
        lookup=lookup,
        slab_nx=args.slab_nx,
        query_chunk=args.query_chunk,
    )
    groups = totals.result()
    if scanned != args.expected_cells or groups["all_cells"]["cell_count"] != args.expected_cells:
        raise RuntimeError("Coverage scan did not include every simulation cell exactly once")
    result = {
        "status": "diagnostic only; adoption and failure acceptance are not decided",
        "table": str(args.table.resolve()), "table_sha256": file_sha256(args.table),
        "table_build_metadata": dict(table.build_metadata or {}),
        "dataset": str(dataset.resolve()), "shape": list(shape), "total_cells": scanned,
        "snapshot_domain": domain, "fresh_input_extrema": extrema,
        "source": "fresh snapshot fields; full z columns; periodic x/y velocity differences; no field caches",
        "temperature_source": "raw boxlib temperature, interpreted as K by Snapshot.read_slab",
        "lookup_policy": "invalid inputs skipped; valid inputs use DespoticLookup.clip_coordinates then DespoticLookup",
        "mass_weight": "rho times uniform cell volume; invalid/nonpositive masses excluded and counted",
        "fraction_denominators": "cell_fraction uses all cells in each group; mass_fraction uses its total valid positive mass",
        "support_policy": "positive_weight_failed_support uses weight > 0; any_failed_among_eight_corners also includes zero-weight corners",
        "cold_selection": "finite positive T_QUOKKA < 3000 K; exact boundary is excluded",
        "query_scope": "temperature/mu queried for every valid-input cell, including hot cells for diagnostics",
        "invalid_input_scope": "support and query flags are false for skipped invalid inputs; invalid_input is reported separately",
        "raw_table_nodes": {
            "total": int(table.tg_final.size), "failed": int(np.count_nonzero(table.failure_mask)),
            "temperature_nonfinite": int(np.count_nonzero(~np.isfinite(table.tg_final))),
            "mu_nonfinite": int(np.count_nonzero(~np.isfinite(table.mu_values))),
        },
        "source_sha256": {"script": file_sha256(__file__), "gas_fields": file_sha256(gas_fields.__file__),
                          "snapshot_reader": file_sha256(ROOT / "src/quokka2s/snapshot_reader.py"),
                          "lookup": file_sha256(lookup_module.__file__), "settings": file_sha256(settings.__file__)},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "yt": yt.__version__},
        "slab_nx": args.slab_nx, "query_chunk": args.query_chunk, "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved coverage diagnostic: {args.output}", flush=True)


if __name__ == "__main__":
    main()
