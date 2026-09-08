"""Measure raw DESPOTIC-table coverage of every cell in its source snapshot.

This is a diagnostic only: no table values, pipeline defaults, or snapshot
fields are changed. Invalid inputs are counted and never clipped into valid
queries. Valid out-of-bounds inputs are reported before applying the pipeline's
existing lookup-coordinate clipping. No failure-acceptance threshold is set.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from quokka2s.tables.io import load_table
from quokka2s.tables.lookup import TableLookup


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = (ROOT / "output/despotic_rebuild_snapshot_20260908"
                 / "despotic_table_co10_co21_dvdr_fullrange.npz")
AXIS_NAMES = ("nH", "NH", "dVdr")
COLD_BOUNDARY_K = 3000.0


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

    ``clip_inputs`` is the existing physics_fields._clip_to_table_domain in
    the snapshot scan. Supplying it explicitly keeps synthetic tests independent
    of yt and of the installed DESPOTIC solver.
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
        # Evaluate the real TableLookup. In three dimensions RGI may propagate
        # NaN from a zero-weight corner, so support flags alone are insufficient.
        temperature = np.asarray(lookup.temperature(*safe), dtype=float)
        mu = np.asarray(lookup.mu(*safe), dtype=float)
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
            ("T_QUOKKA_lt_3000_K", valid_t & (temperature < COLD_BOUNDARY_K)),
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


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slab_windows(nx, slab_nx):
    """Non-overlapping x cores with the halo used by the extrema scanner."""
    if nx < 2 or slab_nx < 1:
        raise ValueError("x dimension must be at least two and slab size positive")
    for ix in range(0, nx, slab_nx):
        end = min(ix + slab_nx, nx)
        lo, hi = max(0, ix - 1), min(nx, end + 1)
        yield ix, end, lo, hi, slice(ix - lo, end - lo)


def _validate_scan_provenance(table, dataset, shape, cfg, physics):
    domain = (table.build_metadata or {}).get("snapshot_domain")
    if not domain or domain.get("selection") != "all simulation cells":
        raise ValueError("Candidate lacks all-cell snapshot-domain provenance")
    checks = {
        "dataset": str(dataset.resolve()), "shape": list(shape),
        "total_cells": int(np.prod(shape)), "X_H": float(cfg.X_H),
        "column_mean": cfg.COLUMN_DENSITY_MEAN,
        "column_directions": cfg.COLUMN_DENSITY_DIRECTIONS,
        "physics_source_sha256": _sha256(physics.__file__),
    }
    for name, expected in checks.items():
        if domain.get(name) != expected:
            raise ValueError(f"Snapshot provenance mismatch for {name}: "
                             f"table={domain.get(name)!r}, scan={expected!r}")
    for name, axis in zip(AXIS_NAMES, _table_axes(table)):
        measured = domain["axes"][name]
        if axis[0] != measured["minimum"] or axis[-1] != measured["maximum"]:
            raise ValueError(f"Candidate {name} bounds differ from recorded snapshot extrema")
    return domain


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--dataset", type=Path)
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
    from quokka2s.pipeline.prep import config as cfg
    from quokka2s.pipeline.prep import physics_fields as physics
    from quokka2s.tables import lookup as lookup_module

    table = load_table(args.table)
    lookup = TableLookup(table)
    _table_axes(table)
    # A cleaned table may retain the old failure mask while filling values.
    # Require the candidate's raw failed T/mu outputs so this cannot report a
    # filled table as the requested raw candidate.
    if np.any(table.failure_mask & (
            np.isfinite(table.tg_final) | np.isfinite(table.mu_values))):
        raise ValueError("Failed nodes contain finite T or mu; supply the raw candidate table")
    dataset = args.dataset or Path(cfg.YT_DATASET_PATH)
    ds = yt.load(str(dataset.resolve()))
    if (ds.max_level != 0 or cfg.DOWNSAMPLE_FACTOR != 1
            or cfg.COLUMN_DENSITY_DIRECTIONS != "z"):
        raise ValueError("Coverage requires a full-resolution uniform snapshot and full z columns")
    shape = tuple(int(value) for value in ds.domain_dimensions)
    if min(shape) < 2 or int(np.prod(shape)) != args.expected_cells:
        raise ValueError(f"Snapshot shape {shape} does not match expected all-cell count "
                         f"{args.expected_cells}")
    domain = _validate_scan_provenance(table, dataset, shape, cfg, physics)
    width = ds.domain_width / ds.domain_dimensions
    volume_cm3 = float(np.prod(width.to("cm").value))
    totals = CoverageTotals()
    scanned = 0
    extrema = {name: dict(minimum=None, maximum=None) for name in AXIS_NAMES}
    for ix, end, lo, hi, core in slab_windows(shape[0], args.slab_nx):
        edge = ds.domain_left_edge.copy()
        edge[0] += lo * width[0]
        grid = ds.covering_grid(level=0, left_edge=edge, dims=(hi - lo, *shape[1:]))
        fields = []
        for name, function, unit in (
            ("nH", physics._number_density_H, "cm**-3"),
            ("NH", physics._column_density_H, "cm**-2"),
            ("dVdr", physics._dVdr_lvg, "s**-1"),
        ):
            values = np.array(function(None, grid).to(unit)[core], dtype=float, copy=True).ravel()
            valid = np.isfinite(values) & (values > 0)
            if np.any(valid):
                limits = extrema[name]
                lower, upper = float(values[valid].min()), float(values[valid].max())
                limits["minimum"] = lower if limits["minimum"] is None else min(lower, limits["minimum"])
                limits["maximum"] = upper if limits["maximum"] is None else max(upper, limits["maximum"])
            fields.append(values)
        temperature = np.array(grid[("boxlib", "temperature")][core], dtype=float, copy=True).ravel()
        mass = (np.array(grid[("gas", "density")].to("g/cm**3")[core],
                         dtype=float, copy=True).ravel() * volume_cm3)
        del grid
        expected_slab = (end - ix) * shape[1] * shape[2]
        if any(values.size != expected_slab for values in (*fields, temperature, mass)):
            raise RuntimeError("Slab fields do not contain exactly the non-halo cells")
        for start in range(0, expected_slab, args.query_chunk):
            stop = min(start + args.query_chunk, expected_slab)
            flags = classify_queries(lookup, *(values[start:stop] for values in fields),
                                     clip_inputs=physics._clip_to_table_domain)
            totals.add(flags, temperature[start:stop], mass[start:stop])
        scanned += expected_slab
        print(f"Coverage checked {scanned}/{args.expected_cells} cells ({end}/{shape[0]} x planes)",
              flush=True)
    groups = totals.result()
    if scanned != args.expected_cells or groups["all_cells"]["cell_count"] != args.expected_cells:
        raise RuntimeError("Coverage scan did not include every simulation cell exactly once")
    result = {
        "status": "diagnostic only; adoption and failure acceptance are not decided",
        "table": str(args.table.resolve()), "table_sha256": _sha256(args.table),
        "table_build_metadata": dict(table.build_metadata or {}),
        "dataset": str(dataset.resolve()), "shape": list(shape), "total_cells": scanned,
        "snapshot_domain": domain, "fresh_input_extrema": extrema,
        "source": "fresh snapshot fields; full z columns; one x halo; no field caches",
        "temperature_source": "raw boxlib temperature, interpreted as K as in physics_fields",
        "lookup_policy": "invalid inputs skipped; valid inputs use _clip_to_table_domain then TableLookup",
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
        "source_sha256": {"script": _sha256(__file__), "physics_fields": _sha256(physics.__file__),
                          "lookup": _sha256(lookup_module.__file__), "config": _sha256(cfg.__file__)},
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
