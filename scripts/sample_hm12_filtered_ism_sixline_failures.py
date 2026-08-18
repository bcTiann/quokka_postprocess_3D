#!/usr/bin/env python3
"""Audit simulation support for raw six-line column and Jeans failures.

The regime boundary always uses T_QUOKKA.  For cells with T_QUOKKA < 3000 K,
the Cloudy lookup coordinate is T_DESPOTIC; otherwise it is T_QUOKKA.  Failed
nodes are not filled by this script.  A cell touches a failure when any corner
of its linear interpolation stencil has failure weight greater than 1e-12.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import yt
from yt.units.physical_constants import mh

from quokka2s.pipeline.cache import (
    cache_root_for_dataset,
    compute_cache_key,
    field_cache_key,
    field_cache_path,
)
from quokka2s.pipeline.prep import config as cfg


COLUMN_FIELD = ("gas", "column_density_H")
TDSP_FIELD = ("gas", "temperature_despotic")
SPLIT_K = 3000.0
TOUCH_EPS = 1.0e-12


def _load(path: Path, geometry: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        result = {name: np.asarray(source[name]) for name in source.files}
    expected_axis = (
        "line,log_nH,log_NH,log_T" if geometry == "column"
        else "line,log_nH,log_T"
    )
    if str(result["axis_order"].item()) != expected_axis:
        raise ValueError(f"unexpected axis order in {path}")
    failure = np.asarray(result["failure_mask"], dtype=bool)
    if not all(np.array_equal(failure[0], mask) for mask in failure[1:]):
        raise ValueError(f"line failure masks are not identical in {path}")
    result["union_failure"] = np.any(failure, axis=0)
    return result


def _open_cache(
    dataset: Path,
    despotic_table: Path,
    field: tuple[str, str],
) -> tuple[h5py.File, Path, int]:
    path = field_cache_path(cache_root_for_dataset(dataset), field).resolve()
    handle = h5py.File(path, "r")
    stored_schema = int(handle.attrs.get("schema_version", -1))
    # Schema v20 changed Halpha coefficients and spectrum units only.  The two
    # inputs used here (column density and T_DESPOTIC) are unchanged from v19,
    # so a v19 cache is valid only if its key recomputes exactly under v19.
    if stored_schema not in (19, 20):
        handle.close()
        raise RuntimeError(f"unsupported cache schema {stored_schema}: {path}")
    base_key = compute_cache_key(
        dataset_path=dataset,
        despotic_table_path=despotic_table,
        downsample_factor=cfg.DOWNSAMPLE_FACTOR,
        column_extension_lateral_kpc=cfg.COLUMN_EXTENSION_LATERAL_KPC,
        schema_version=stored_schema,
    )
    expected_key = field_cache_key(base_key, field)
    actual_key = str(handle.attrs.get("cache_key", ""))
    actual_field = (
        str(handle.attrs.get("field_type", "")),
        str(handle.attrs.get("field_name", "")),
    )
    if actual_key != expected_key or actual_field != field:
        handle.close()
        raise RuntimeError(f"stale or mismatched cache: {path}")
    return handle, path, stored_schema


def _brackets(axis: np.ndarray, values: np.ndarray):
    upper = np.searchsorted(axis, values, side="right")
    upper = np.clip(upper, 1, axis.size - 1)
    lower = upper - 1
    fraction = (values - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, fraction


def _failure_weight(mask: np.ndarray, brackets, touched_ids: set[int]):
    output = np.zeros(brackets[0][2].shape, dtype=float)
    shape = mask.shape

    def recurse(axis_index: int, indices: list[np.ndarray], weight: np.ndarray):
        nonlocal output
        if axis_index == len(brackets):
            failed = mask[tuple(indices)]
            output += weight * failed
            contributes = failed & (weight > TOUCH_EPS)
            if np.any(contributes):
                flat = np.ravel_multi_index(
                    tuple(index[contributes] for index in indices), shape
                )
                touched_ids.update(int(value) for value in np.unique(flat))
            return
        lower, upper, fraction = brackets[axis_index]
        recurse(axis_index + 1, indices + [lower], weight * (1.0 - fraction))
        recurse(axis_index + 1, indices + [upper], weight * fraction)

    recurse(0, [], np.ones_like(output))
    return output


def _new_counts() -> dict[str, int | float]:
    return {
        "fully_in_bounds": 0,
        "any_axis_outside": 0,
        "nH_below": 0,
        "nH_above": 0,
        "Tlookup_below": 0,
        "Tlookup_above": 0,
        "NH_below": 0,
        "NH_above": 0,
        "touches_failure": 0,
        "touches_failure_TQUOKKA_lt_3000": 0,
        "touches_failure_TQUOKKA_ge_3000": 0,
        "failure_weight_ge_0p01": 0,
        "failure_weight_ge_0p10": 0,
        "failure_weight_ge_0p25": 0,
        "failure_weight_ge_0p50": 0,
        "failure_weight_nearly_one": 0,
        "maximum_failure_weight": 0.0,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    stem = "cloudy_hm2012_native_plus_filtered_ism_defaultabund_sixline"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--column-table", type=Path,
                        default=root / f"data/{stem}_column_10x10x21.npz")
    parser.add_argument("--jeans-table", type=Path,
                        default=root / f"data/{stem}_jeans_10x21.npz")
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument("--despotic-table", type=Path,
                        default=Path(cfg.DESPOTIC_TABLE_PATH))
    parser.add_argument("--output", type=Path,
                        default=Path(cfg.OUTPUT_DIR) /
                        "hm12_native_filtered_ism_sixline_failure_sampling.json")
    parser.add_argument("--slab-nz", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in ("column_table", "jeans_table", "dataset", "despotic_table", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    if args.slab_nz <= 0:
        raise ValueError("--slab-nz must be positive")

    tables = {
        "column": _load(args.column_table, "column"),
        "jeans": _load(args.jeans_table, "jeans"),
    }
    ds = yt.load(str(args.dataset))
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    nx, ny, nz = dimensions
    cell_width = ds.domain_width.to("cm") / ds.domain_dimensions
    column_cache, column_path, column_cache_schema = _open_cache(
        args.dataset, args.despotic_table, COLUMN_FIELD
    )
    tdsp_cache, tdsp_path, tdsp_cache_schema = _open_cache(
        args.dataset, args.despotic_table, TDSP_FIELD
    )
    for handle, name in ((column_cache, "column"), (tdsp_cache, "T_DESPOTIC")):
        if tuple(handle["data"].shape) != dimensions:
            handle.close()
            raise ValueError(f"{name} cache shape mismatch")

    counts = {geometry: _new_counts() for geometry in tables}
    touched_ids = {geometry: set() for geometry in tables}
    all_cells = 0
    low_cells = 0
    invalid_cells = 0
    hydrogen_mass_g = float(mh.to_value("g"))
    n_slabs = (nz + args.slab_nz - 1) // args.slab_nz
    started = time.perf_counter()
    try:
        for slab_number, iz in enumerate(range(0, nz, args.slab_nz), start=1):
            local_nz = min(args.slab_nz, nz - iz)
            left_edge = ds.domain_left_edge.copy()
            left_edge[2] += iz * cell_width[2]
            grid = ds.covering_grid(
                level=ds.max_level, left_edge=left_edge, dims=(nx, ny, local_nz)
            )
            tq = np.asarray(grid[("boxlib", "temperature")], dtype=float)
            density = np.asarray(grid[("gas", "density")].to("g/cm**3"), dtype=float)
            del grid
            n_h = density * float(cfg.X_H) / hydrogen_mass_g
            del density
            tdsp = np.asarray(tdsp_cache["data"][:, :, iz:iz + local_nz], dtype=float)
            column = np.asarray(
                column_cache["data"][:, :, iz:iz + local_nz], dtype=float
            )
            low = tq < SPLIT_K
            lookup_t = np.where(low, tdsp, tq)
            finite = (
                np.isfinite(tq) & np.isfinite(lookup_t) & np.isfinite(n_h)
                & np.isfinite(column) & (tq > 0.0) & (lookup_t > 0.0)
                & (n_h > 0.0) & (column > 0.0)
            )
            all_cells += int(tq.size)
            low_cells += int(np.count_nonzero(low))
            invalid_cells += int(tq.size - np.count_nonzero(finite))
            log_nh = np.zeros_like(n_h)
            log_t = np.zeros_like(lookup_t)
            log_column = np.zeros_like(column)
            np.log10(n_h, out=log_nh, where=finite)
            np.log10(lookup_t, out=log_t, where=finite)
            np.log10(column, out=log_column, where=finite)

            for geometry, table in tables.items():
                local = counts[geometry]
                n_axis = table["log_nH"].astype(float)
                t_axis = table["log_T"].astype(float)
                n_below = finite & (log_nh < n_axis[0])
                n_above = finite & (log_nh > n_axis[-1])
                t_below = finite & (log_t < t_axis[0])
                t_above = finite & (log_t > t_axis[-1])
                outside = n_below | n_above | t_below | t_above
                for key, mask in (("nH_below", n_below), ("nH_above", n_above),
                                  ("Tlookup_below", t_below),
                                  ("Tlookup_above", t_above)):
                    local[key] += int(np.count_nonzero(mask))
                coordinates = [log_nh]
                axes = [n_axis]
                if geometry == "column":
                    column_axis = table["log_NH"].astype(float)
                    column_below = finite & (log_column < column_axis[0])
                    column_above = finite & (log_column > column_axis[-1])
                    outside |= column_below | column_above
                    local["NH_below"] += int(np.count_nonzero(column_below))
                    local["NH_above"] += int(np.count_nonzero(column_above))
                    coordinates.append(log_column)
                    axes.append(column_axis)
                in_bounds = finite & ~outside
                local["fully_in_bounds"] += int(np.count_nonzero(in_bounds))
                local["any_axis_outside"] += int(np.count_nonzero(outside))
                if not np.any(in_bounds):
                    continue
                coordinates.append(log_t)
                axes.append(t_axis)
                brackets = tuple(
                    _brackets(axis, coordinate[in_bounds])
                    for axis, coordinate in zip(axes, coordinates)
                )
                weight = _failure_weight(
                    table["union_failure"], brackets, touched_ids[geometry]
                )
                touched = weight > TOUCH_EPS
                local["touches_failure"] += int(np.count_nonzero(touched))
                selected_low = low[in_bounds]
                local["touches_failure_TQUOKKA_lt_3000"] += int(
                    np.count_nonzero(touched & selected_low)
                )
                local["touches_failure_TQUOKKA_ge_3000"] += int(
                    np.count_nonzero(touched & ~selected_low)
                )
                for threshold, key in (
                    (0.01, "failure_weight_ge_0p01"),
                    (0.10, "failure_weight_ge_0p10"),
                    (0.25, "failure_weight_ge_0p25"),
                    (0.50, "failure_weight_ge_0p50"),
                    (1.0 - 1.0e-12, "failure_weight_nearly_one"),
                ):
                    local[key] += int(np.count_nonzero(weight >= threshold))
                local["maximum_failure_weight"] = max(
                    float(local["maximum_failure_weight"]), float(weight.max())
                )
            elapsed = time.perf_counter() - started
            print(
                f"[{slab_number:02d}/{n_slabs:02d}] z={iz}:{iz + local_nz} "
                f"column={counts['column']['touches_failure']:,} "
                f"jeans={counts['jeans']['touches_failure']:,} "
                f"elapsed={elapsed / 60.0:.2f} min",
                flush=True,
            )
    finally:
        column_cache.close()
        tdsp_cache.close()

    geometry_results = {}
    for geometry, table in tables.items():
        mask = table["union_failure"]
        axes = [table["log_nH"].astype(float)]
        names = ["log_nH"]
        if geometry == "column":
            axes.append(table["log_NH"].astype(float))
            names.append("log_NH")
        axes.append(table["log_T"].astype(float))
        names.append("log_T")
        nodes = []
        for flat in sorted(touched_ids[geometry]):
            index = np.unravel_index(flat, mask.shape)
            record = {name: float(axis[i]) for name, axis, i in zip(names, axes, index)}
            record["indices"] = [int(value) for value in index]
            record["temperature_K"] = float(10.0 ** record["log_T"])
            nodes.append(record)
        geometry_results[geometry] = {
            "table": str((args.column_table if geometry == "column" else args.jeans_table)),
            "grid_shape": list(mask.shape),
            "raw_failure_nodes": int(np.count_nonzero(mask)),
            "unique_touched_failure_nodes": len(nodes),
            "touched_failure_nodes": nodes,
            "counts": counts[geometry],
            "touched_fraction_of_in_bounds": (
                counts[geometry]["touches_failure"] / counts[geometry]["fully_in_bounds"]
                if counts[geometry]["fully_in_bounds"] else 0.0
            ),
        }
    result = {
        "dataset": str(args.dataset),
        "temperature_policy": (
            "split by T_QUOKKA; lookup T_DESPOTIC below 3000 K, "
            "T_QUOKKA at or above 3000 K"
        ),
        "touch_definition": f"linear interpolation failure weight > {TOUCH_EPS:g}",
        "column_density_cache": str(column_path),
        "column_density_cache_schema": column_cache_schema,
        "temperature_despotic_cache": str(tdsp_path),
        "temperature_despotic_cache_schema": tdsp_cache_schema,
        "legacy_cache_acceptance": (
            "schema v19 accepted only after exact v19 key recomputation; v20 "
            "changed Halpha/spectrum definitions, not column density or T_DESPOTIC"
        ),
        "all_cells": all_cells,
        "TQUOKKA_lt_3000_cells": low_cells,
        "invalid_cells": invalid_cells,
        "geometries": geometry_results,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
