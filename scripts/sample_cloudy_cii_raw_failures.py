"""Count simulation cells whose stencil touches raw Cloudy line-map failures."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import h5py
import numpy as np
import yt
from yt.units.physical_constants import mh

from quokka2s.pipeline.prep import config as cfg


LOOP_RE = re.compile(r"^#\s*(hden|stop column density)\s+(.+?)\s*$")
TOUCH_EPS = 1.0e-12


def read_run(path: Path) -> tuple[float, float, np.ndarray]:
    log_nh = log_nh_column = None
    rows: list[tuple[float, float]] = []
    for line in path.read_text().splitlines():
        match = LOOP_RE.match(line)
        if match:
            if match.group(1) == "hden":
                log_nh = float(match.group(2))
            else:
                log_nh_column = float(match.group(2))
            continue
        if not line or line.startswith("#"):
            continue
        columns = line.split()
        if len(columns) >= 2:
            rows.append((float(columns[0]), float(columns[1])))
    if log_nh is None or log_nh_column is None:
        raise ValueError(f"missing loop metadata: {path}")
    return log_nh, log_nh_column, np.asarray(rows, dtype=float).reshape(-1, 2)


def load_failure_grid(
    input_dir: Path,
    *,
    t_min: float,
    t_max: float,
    t_points: int,
    expected_runs: int = 240,
    expected_nh_points: int = 16,
    expected_column_points: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(input_dir.glob("*_run*.dat"))
    if len(files) != expected_runs:
        raise ValueError(
            f"expected {expected_runs} run files, found {len(files)}"
        )
    records = [read_run(path) for path in files]
    log_nh = np.unique([record[0] for record in records])
    log_column = np.unique([record[1] for record in records])
    expected_shape = (expected_nh_points, expected_column_points)
    if (log_nh.size, log_column.size) != expected_shape:
        raise ValueError(
            f"expected {expected_shape[0]}x{expected_shape[1]} axes, "
            f"found {log_nh.size}x{log_column.size}"
        )
    log_temperature = np.linspace(np.log10(t_min), np.log10(t_max), t_points)
    failure_mask = np.ones((log_nh.size, log_column.size, t_points), dtype=bool)
    nh_index = {value: index for index, value in enumerate(log_nh)}
    column_index = {value: index for index, value in enumerate(log_column)}
    for path, (run_nh, run_column, rows) in zip(files, records):
        indices = np.abs(
            rows[:, 0, None] - log_temperature[None, :]
        ).argmin(axis=1)
        residual = np.abs(rows[:, 0] - log_temperature[indices])
        if np.any(residual > 5.1e-4):
            raise ValueError(f"temperature row is off-grid: {path}")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"duplicate temperature row: {path}")
        failure_mask[nh_index[run_nh], column_index[run_column], indices] = False
    return log_nh, log_column, log_temperature, failure_mask


def brackets(axis: np.ndarray, coordinates: np.ndarray):
    clipped = np.clip(coordinates, axis[0], axis[-1])
    upper = np.searchsorted(axis, clipped, side="right")
    upper = np.clip(upper, 1, axis.size - 1)
    lower = upper - 1
    fraction = (clipped - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, fraction


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "work/cloudy_cooling_tools_history/examples/grackle/hm_2012_cii_cloudy_full_output",
    )
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument(
        "--column-cache",
        type=Path,
        default=project_root / "intermediates/plt0655228/fields/field_gas_column_density_H.h5",
    )
    parser.add_argument("--slab-nz", type=int, default=32)
    parser.add_argument("--t-min", type=float, default=3.0e3)
    parser.add_argument("--t-max", type=float, default=2.7289777828080403e6)
    parser.add_argument("--t-points", type=int, default=31)
    parser.add_argument("--expected-runs", type=int, default=240)
    parser.add_argument("--expected-nh-points", type=int, default=16)
    parser.add_argument("--expected-column-points", type=int, default=15)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "output/plt0655228_down1_Lext15kpc/cloudy_cii_full31_raw_failure_sampling.json",
    )
    args = parser.parse_args()

    log_nh, log_column, log_temperature, failure_mask = load_failure_grid(
        args.input_dir,
        t_min=args.t_min,
        t_max=args.t_max,
        t_points=args.t_points,
        expected_runs=args.expected_runs,
        expected_nh_points=args.expected_nh_points,
        expected_column_points=args.expected_column_points,
    )
    ds = yt.load(str(args.dataset.resolve()))
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    cell_width = ds.domain_width.to("cm") / ds.domain_dimensions
    hydrogen_mass_g = float(mh.to_value("g"))
    nx, ny, nz = dimensions

    counts = {
        "all_cells": 0,
        "cloudy_branch_T_ge_3000": 0,
        "cloudy_table_sampled": 0,
        "temperature_above_table_zero_emissivity": 0,
        "touches_failure_node": 0,
        "nH_outside_table": 0,
        "NH_outside_table": 0,
    }
    maximum_failure_weight = 0.0
    touched_node_ids: set[int] = set()
    start = time.perf_counter()

    with h5py.File(args.column_cache, "r") as column_file:
        if column_file["data"].shape != dimensions:
            raise ValueError(
                f"column cache shape {column_file['data'].shape} != {dimensions}"
            )
        for iz in range(0, nz, args.slab_nz):
            slab_nz = min(args.slab_nz, nz - iz)
            left_edge = ds.domain_left_edge.copy()
            left_edge[2] += iz * cell_width[2]
            grid = ds.covering_grid(
                level=ds.max_level,
                left_edge=left_edge,
                dims=(nx, ny, slab_nz),
            )
            temperature = np.asarray(grid[("boxlib", "temperature")], dtype=float)
            density = np.asarray(grid[("gas", "density")].to("g/cm**3"), dtype=float)
            n_h = density * float(cfg.X_H) / hydrogen_mass_g
            column = np.asarray(column_file["data"][:, :, iz:iz + slab_nz], dtype=float)
            del grid, density

            finite_positive = (
                np.isfinite(temperature) & np.isfinite(n_h) & np.isfinite(column)
                & (temperature > 0.0) & (n_h > 0.0) & (column > 0.0)
            )
            cloudy_branch = finite_positive & (temperature >= args.t_min)
            sampled = cloudy_branch & (temperature <= args.t_max)
            counts["all_cells"] += int(temperature.size)
            counts["cloudy_branch_T_ge_3000"] += int(np.count_nonzero(cloudy_branch))
            counts["cloudy_table_sampled"] += int(np.count_nonzero(sampled))
            counts["temperature_above_table_zero_emissivity"] += int(
                np.count_nonzero(cloudy_branch & (temperature > args.t_max))
            )
            if not np.any(sampled):
                continue

            selected_temperature = temperature[sampled]
            selected_nh = n_h[sampled]
            selected_column = column[sampled]
            log_nh_values = np.log10(selected_nh)
            log_column_values = np.log10(selected_column)
            log_temperature_values = np.log10(selected_temperature)
            counts["nH_outside_table"] += int(np.count_nonzero(
                (log_nh_values < log_nh[0]) | (log_nh_values > log_nh[-1])
            ))
            counts["NH_outside_table"] += int(np.count_nonzero(
                (log_column_values < log_column[0]) | (log_column_values > log_column[-1])
            ))
            axis_brackets = (
                brackets(log_nh, log_nh_values),
                brackets(log_column, log_column_values),
                brackets(log_temperature, log_temperature_values),
            )
            failure_weight = np.zeros(selected_temperature.size, dtype=float)
            for nh_corner in (0, 1):
                ni = axis_brackets[0][nh_corner]
                nw = axis_brackets[0][2] if nh_corner else 1.0 - axis_brackets[0][2]
                for column_corner in (0, 1):
                    ci = axis_brackets[1][column_corner]
                    cw = axis_brackets[1][2] if column_corner else 1.0 - axis_brackets[1][2]
                    for temperature_corner in (0, 1):
                        ti = axis_brackets[2][temperature_corner]
                        tw = (
                            axis_brackets[2][2]
                            if temperature_corner else 1.0 - axis_brackets[2][2]
                        )
                        weight = nw * cw * tw
                        failed = failure_mask[ni, ci, ti]
                        contributes = failed & (weight > TOUCH_EPS)
                        failure_weight += weight * failed
                        if np.any(contributes):
                            node_ids = np.ravel_multi_index(
                                (ni[contributes], ci[contributes], ti[contributes]),
                                failure_mask.shape,
                            )
                            touched_node_ids.update(int(value) for value in np.unique(node_ids))
            touched = failure_weight > TOUCH_EPS
            counts["touches_failure_node"] += int(np.count_nonzero(touched))
            if failure_weight.size:
                maximum_failure_weight = max(
                    maximum_failure_weight, float(np.max(failure_weight))
                )

    touched_nodes = []
    for node_id in sorted(touched_node_ids):
        i, j, k = np.unravel_index(node_id, failure_mask.shape)
        touched_nodes.append({
            "nH_index": int(i),
            "NH_index": int(j),
            "T_index": int(k),
            "log_nH": float(log_nh[i]),
            "log_NH": float(log_column[j]),
            "log_T": float(log_temperature[k]),
            "temperature_K": float(10.0 ** log_temperature[k]),
        })
    sampled_count = counts["cloudy_table_sampled"]
    result = {
        "dataset": str(args.dataset.resolve()),
        "raw_cloudy_output": str(args.input_dir.resolve()),
        "cloudy_grid_shape": list(failure_mask.shape),
        "raw_failure_nodes": int(np.count_nonzero(failure_mask)),
        "touch_definition": f"trilinear failure weight > {TOUCH_EPS:g}",
        "counts": counts,
        "touched_fraction_of_sampled_cells": (
            counts["touches_failure_node"] / sampled_count if sampled_count else 0.0
        ),
        "maximum_failure_weight": maximum_failure_weight,
        "unique_touched_failure_nodes": len(touched_nodes),
        "touched_failure_nodes": touched_nodes,
        "elapsed_minutes": (time.perf_counter() - start) / 60.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
