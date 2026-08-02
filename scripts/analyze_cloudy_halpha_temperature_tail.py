"""Measure the simulation-weighted high-temperature tail of Cloudy H-alpha.

The raw 5x5x21 Cloudy line map stores log10(epsilon_Halpha / n_H^2).
This program samples it for every T_QUOKKA >= 3000 K simulation cell, converts
the local emissivity to cell luminosity, and bins that luminosity in log T.
Failed Cloudy nodes remain unavailable and abort if they receive trilinear
weight greater than 1e-12.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import h5py
import numpy as np
import yt
from yt.units.physical_constants import mh
from yt.units.yt_array import YTQuantity

from quokka2s.pipeline.prep import config as cfg


LOOP_RE = re.compile(r"^#\s*(hden|stop column density)\s+(.+?)\s*$")
TOUCH_EPS = 1.0e-12
ZERO_SENTINEL_MAX = -90.0


def read_run(path: Path) -> tuple[float, float, np.ndarray]:
    log_nh = log_column = None
    rows: list[tuple[float, float]] = []
    for line in path.read_text().splitlines():
        match = LOOP_RE.match(line)
        if match:
            if match.group(1) == "hden":
                log_nh = float(match.group(2))
            else:
                log_column = float(match.group(2))
            continue
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 2:
            rows.append((float(fields[0]), float(fields[1])))
    if log_nh is None or log_column is None:
        raise ValueError(f"missing loop metadata: {path}")
    return log_nh, log_column, np.asarray(rows, dtype=float).reshape(-1, 2)


def load_grid(
    input_dir: Path,
    *,
    t_min: float,
    t_max: float,
    t_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(input_dir.glob("*_run*.dat"))
    if len(files) != 25:
        raise ValueError(f"expected 25 H-alpha run files, found {len(files)}")
    records = [read_run(path) for path in files]
    log_nh = np.unique([record[0] for record in records])
    log_column = np.unique([record[1] for record in records])
    if (log_nh.size, log_column.size) != (5, 5):
        raise ValueError(
            f"expected 5x5 H-alpha axes, found {log_nh.size}x{log_column.size}"
        )
    expected_keys = {(n, c) for n in log_nh for c in log_column}
    if {(record[0], record[1]) for record in records} != expected_keys:
        raise ValueError("H-alpha raw output is not a complete Cartesian run grid")

    log_temperature = np.linspace(np.log10(t_min), np.log10(t_max), t_points)
    raw_log = np.full((log_nh.size, log_column.size, t_points), np.nan)
    nh_index = {value: index for index, value in enumerate(log_nh)}
    column_index = {value: index for index, value in enumerate(log_column)}
    for path, (run_nh, run_column, rows) in zip(files, records):
        indices = np.abs(
            rows[:, 0, None] - log_temperature[None, :]
        ).argmin(axis=1)
        residual = np.abs(rows[:, 0] - log_temperature[indices])
        if np.any(residual > 5.1e-4):
            raise ValueError(f"temperature row is off-grid: {path}")
        i = nh_index[run_nh]
        j = column_index[run_column]
        for row, k in zip(rows, indices):
            if np.isfinite(raw_log[i, j, k]):
                raise ValueError(f"duplicate H-alpha temperature row: {path}")
            raw_log[i, j, k] = float(row[1])

    failure_mask = ~np.isfinite(raw_log)
    zero_mask = np.isfinite(raw_log) & (raw_log <= ZERO_SENTINEL_MAX)
    if zero_mask.any():
        raise ValueError(
            "H-alpha scan contains exact-zero nodes; mixed linear/log "
            "interpolation must be implemented before this analysis"
        )
    return log_nh, log_column, log_temperature, raw_log, failure_mask


def brackets(axis: np.ndarray, coordinates: np.ndarray):
    clipped = np.clip(coordinates, axis[0], axis[-1])
    upper = np.searchsorted(axis, clipped, side="right")
    upper = np.clip(upper, 1, axis.size - 1)
    lower = upper - 1
    fraction = (clipped - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, fraction


def interpolate_log_coefficient(
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    raw_log: np.ndarray,
    failure_mask: np.ndarray,
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    axis_brackets = tuple(
        brackets(axis, coordinate)
        for axis, coordinate in zip(axes, coordinates)
    )
    log_coefficient = np.zeros(coordinates[0].size, dtype=float)
    failure_weight = np.zeros(coordinates[0].size, dtype=float)
    for nh_corner in (0, 1):
        ni = axis_brackets[0][nh_corner]
        nw = axis_brackets[0][2] if nh_corner else 1.0 - axis_brackets[0][2]
        for column_corner in (0, 1):
            ci = axis_brackets[1][column_corner]
            cw = (
                axis_brackets[1][2]
                if column_corner else 1.0 - axis_brackets[1][2]
            )
            for temperature_corner in (0, 1):
                ti = axis_brackets[2][temperature_corner]
                tw = (
                    axis_brackets[2][2]
                    if temperature_corner else 1.0 - axis_brackets[2][2]
                )
                weight = nw * cw * tw
                failed = failure_mask[ni, ci, ti]
                failure_weight += weight * failed
                valid = ~failed
                corner_values = raw_log[ni, ci, ti]
                log_coefficient[valid] += weight[valid] * corner_values[valid]
    return log_coefficient, failure_weight


def cutoff_summary(edges: np.ndarray, luminosity: np.ndarray) -> dict[str, float | int]:
    peak_index = int(np.argmax(luminosity))
    peak = float(luminosity[peak_index])
    threshold = peak * 0.01
    significant_hot = np.flatnonzero(
        (np.arange(luminosity.size) >= peak_index) & (luminosity >= threshold)
    )
    if significant_hot.size == 0 or significant_hot[-1] == luminosity.size - 1:
        cutoff_index = luminosity.size
        cutoff_log_t = float(edges[-1])
        cutoff_available = False
    else:
        cutoff_index = int(significant_hot[-1] + 1)
        cutoff_log_t = float(edges[cutoff_index])
        cutoff_available = True
    total = float(np.sum(luminosity))
    tail = float(np.sum(luminosity[cutoff_index:]))
    return {
        "temperature_bins": int(luminosity.size),
        "peak_bin_index": peak_index,
        "peak_log10_temperature_center": float(
            0.5 * (edges[peak_index] + edges[peak_index + 1])
        ),
        "peak_temperature_K": float(
            10.0 ** (0.5 * (edges[peak_index] + edges[peak_index + 1]))
        ),
        "peak_bin_luminosity_erg_s": peak,
        "one_percent_of_peak_erg_s": threshold,
        "cutoff_available_within_scan": cutoff_available,
        "candidate_cutoff_log10_temperature": cutoff_log_t,
        "candidate_cutoff_temperature_K": float(10.0 ** cutoff_log_t),
        "tail_luminosity_erg_s": tail,
        "tail_fraction_of_total": tail / total if total > 0.0 else 0.0,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path,
        default=project_root / "work/cloudy_cooling_tools_history/examples/grackle/hm_2012_halpha_sparse_output",
    )
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument(
        "--column-cache", type=Path,
        default=project_root / "intermediates/plt0655228/fields/field_gas_column_density_H.h5",
    )
    parser.add_argument("--slab-nz", type=int, default=32)
    parser.add_argument("--t-min", type=float, default=3.0e3)
    parser.add_argument("--t-max", type=float, default=1.0e9)
    parser.add_argument("--t-points", type=int, default=21)
    parser.add_argument(
        "--temperature-bins", type=int, nargs="+", default=[40, 80, 160],
        help="Equal-width log10(T) binnings used to test cutoff robustness.",
    )
    parser.add_argument(
        "--tail-thresholds-K", type=float, nargs="+",
        default=[5.0e4, 8.0e4, 1.0e5, 2.0e5, 1.0e6],
        help="Exact temperature thresholds for cumulative hot-tail totals.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=project_root / "output/plt0655228_down1_Lext15kpc/cloudy_halpha_temperature_tail.json",
    )
    parser.add_argument(
        "--csv-output", type=Path,
        default=project_root / "output/plt0655228_down1_Lext15kpc/cloudy_halpha_temperature_tail_80bins.csv",
    )
    args = parser.parse_args()

    axes_and_values = load_grid(
        args.input_dir, t_min=args.t_min, t_max=args.t_max,
        t_points=args.t_points,
    )
    log_nh, log_column, log_temperature, raw_log, failure_mask = axes_and_values
    axes = (log_nh, log_column, log_temperature)
    bin_counts = sorted(set(args.temperature_bins))
    if any(value < 2 for value in bin_counts):
        raise ValueError("temperature bin counts must be at least two")
    edges_by_count = {
        count: np.linspace(log_temperature[0], log_temperature[-1], count + 1)
        for count in bin_counts
    }
    luminosity_by_count = {
        count: np.zeros(count, dtype=float) for count in bin_counts
    }
    cells_by_count = {
        count: np.zeros(count, dtype=np.int64) for count in bin_counts
    }
    tail_thresholds = sorted(set(args.tail_thresholds_K))
    if any(value < args.t_min or value > args.t_max for value in tail_thresholds):
        raise ValueError("tail thresholds must lie inside the Cloudy temperature scan")
    tail_luminosity = {threshold: 0.0 for threshold in tail_thresholds}
    tail_cells = {threshold: 0 for threshold in tail_thresholds}

    ds = yt.load(str(args.dataset.resolve()))
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    nx, ny, nz = dimensions
    cell_width = ds.domain_width.to("cm") / ds.domain_dimensions
    cell_volume_cm3 = float(np.prod(cell_width.to_value("cm")))
    hydrogen_mass_g = float(mh.to_value("g"))
    lsun_erg_s = float(YTQuantity(1.0, "Lsun").to_value("erg/s"))
    counters = {
        "all_cells": 0,
        "cloudy_branch_T_ge_3000": 0,
        "sampled_cells": 0,
        "temperature_above_scan": 0,
        "nH_outside_table": 0,
        "NH_outside_table": 0,
        "touches_failure_node": 0,
    }
    maximum_failure_weight = 0.0
    total_luminosity = 0.0
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
                level=ds.max_level, left_edge=left_edge,
                dims=(nx, ny, slab_nz),
            )
            temperature = np.asarray(grid[("boxlib", "temperature")], dtype=float)
            density = np.asarray(
                grid[("gas", "density")].to("g/cm**3"), dtype=float,
            )
            n_h = density * float(cfg.X_H) / hydrogen_mass_g
            column = np.asarray(
                column_file["data"][:, :, iz:iz + slab_nz], dtype=float,
            )
            del grid, density

            finite_positive = (
                np.isfinite(temperature) & np.isfinite(n_h) & np.isfinite(column)
                & (temperature > 0.0) & (n_h > 0.0) & (column > 0.0)
            )
            hot = finite_positive & (temperature >= args.t_min)
            sampled = hot & (temperature <= args.t_max)
            counters["all_cells"] += int(temperature.size)
            counters["cloudy_branch_T_ge_3000"] += int(np.count_nonzero(hot))
            counters["sampled_cells"] += int(np.count_nonzero(sampled))
            counters["temperature_above_scan"] += int(
                np.count_nonzero(hot & (temperature > args.t_max))
            )
            if not np.any(sampled):
                continue

            selected_temperature = temperature[sampled]
            selected_nh = n_h[sampled]
            selected_column = column[sampled]
            coordinates = (
                np.log10(selected_nh),
                np.log10(selected_column),
                np.log10(selected_temperature),
            )
            outside_nh = (
                (coordinates[0] < log_nh[0]) | (coordinates[0] > log_nh[-1])
            )
            outside_column = (
                (coordinates[1] < log_column[0])
                | (coordinates[1] > log_column[-1])
            )
            counters["nH_outside_table"] += int(np.count_nonzero(outside_nh))
            counters["NH_outside_table"] += int(np.count_nonzero(outside_column))
            if outside_nh.any() or outside_column.any():
                raise ValueError("simulation density/column lies outside H-alpha grid")

            log_coefficient, failure_weight = interpolate_log_coefficient(
                axes, raw_log, failure_mask, coordinates,
            )
            touched = failure_weight > TOUCH_EPS
            counters["touches_failure_node"] += int(np.count_nonzero(touched))
            maximum_failure_weight = max(
                maximum_failure_weight, float(np.max(failure_weight)),
            )
            if touched.any():
                raise ValueError(
                    "simulation query touches an unavailable H-alpha Cloudy node"
                )

            coefficient = np.power(10.0, log_coefficient)
            cell_luminosity = (
                coefficient * np.square(selected_nh) * cell_volume_cm3
            )
            total_luminosity += float(np.sum(cell_luminosity, dtype=np.float64))
            for threshold in tail_thresholds:
                in_tail = selected_temperature >= threshold
                tail_luminosity[threshold] += float(
                    np.sum(cell_luminosity[in_tail], dtype=np.float64)
                )
                tail_cells[threshold] += int(np.count_nonzero(in_tail))
            for count in bin_counts:
                edges = edges_by_count[count]
                luminosity_by_count[count] += np.histogram(
                    coordinates[2], bins=edges, weights=cell_luminosity,
                )[0]
                cells_by_count[count] += np.histogram(
                    coordinates[2], bins=edges,
                )[0]

    summaries = {
        str(count): cutoff_summary(
            edges_by_count[count], luminosity_by_count[count],
        )
        for count in bin_counts
    }
    for summary in summaries.values():
        summary["peak_bin_luminosity_Lsun"] = (
            summary["peak_bin_luminosity_erg_s"] / lsun_erg_s
        )
        summary["tail_luminosity_Lsun"] = (
            summary["tail_luminosity_erg_s"] / lsun_erg_s
        )
    exact_hot_tails = {
        f"{threshold:.8g}": {
            "temperature_threshold_K": threshold,
            "cell_count": tail_cells[threshold],
            "luminosity_erg_s": tail_luminosity[threshold],
            "luminosity_Lsun": tail_luminosity[threshold] / lsun_erg_s,
            "fraction_of_total_luminosity": (
                tail_luminosity[threshold] / total_luminosity
                if total_luminosity > 0.0 else 0.0
            ),
        }
        for threshold in tail_thresholds
    }

    result = {
        "dataset": str(args.dataset.resolve()),
        "raw_cloudy_output": str(args.input_dir.resolve()),
        "grid_shape": list(raw_log.shape),
        "raw_failure_nodes": int(np.count_nonzero(failure_mask)),
        "interpolation": "trilinear in log10(epsilon_Halpha/n_H^2)",
        "failure_touch_threshold": TOUCH_EPS,
        "cutoff_definition": (
            "upper edge of the hottest bin at or above 1% of the peak "
            "temperature-bin luminosity, considering only bins at/above peak"
        ),
        "counts": counters,
        "maximum_failure_weight": maximum_failure_weight,
        "total_halpha_luminosity_erg_s": total_luminosity,
        "total_halpha_luminosity_Lsun": total_luminosity / lsun_erg_s,
        "binning_sensitivity": summaries,
        "exact_hot_tails": exact_hot_tails,
        "elapsed_minutes": (time.perf_counter() - start) / 60.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    primary_count = 80 if 80 in bin_counts else bin_counts[len(bin_counts) // 2]
    primary_edges = edges_by_count[primary_count]
    primary_luminosity = luminosity_by_count[primary_count]
    primary_cells = cells_by_count[primary_count]
    peak = float(np.max(primary_luminosity))
    with args.csv_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "logT_left", "logT_right", "temperature_center_K", "cell_count",
            "luminosity_erg_s", "luminosity_Lsun", "fraction_of_peak_bin",
            "fraction_of_total_luminosity",
        ])
        for index, luminosity in enumerate(primary_luminosity):
            writer.writerow([
                primary_edges[index], primary_edges[index + 1],
                10.0 ** (0.5 * (primary_edges[index] + primary_edges[index + 1])),
                int(primary_cells[index]), luminosity, luminosity / lsun_erg_s,
                luminosity / peak if peak > 0.0 else 0.0,
                luminosity / total_luminosity if total_luminosity > 0.0 else 0.0,
            ])

    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.csv_output}")


if __name__ == "__main__":
    main()
