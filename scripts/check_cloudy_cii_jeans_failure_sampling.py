"""Measure simulation support for failures in the 2-D Jeans CII table."""
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


TDSP_FIELD = ("gas", "temperature_despotic")
SPLIT_K = 3000.0
TOUCH_EPS = 1.0e-12
POLICIES = ("hybrid_TDESP_low_TQUOKKA_high", "TQUOKKA_only")


def _brackets(axis: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper = np.searchsorted(axis, values, side="right")
    upper = np.clip(upper, 1, axis.size - 1)
    lower = upper - 1
    fraction = (values - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, fraction


def _open_tdsp_cache(
    dataset: Path,
    despotic_table: Path,
    dimensions: tuple[int, int, int],
) -> tuple[h5py.File, Path]:
    base_key = compute_cache_key(
        dataset_path=dataset,
        despotic_table_path=despotic_table,
        downsample_factor=cfg.DOWNSAMPLE_FACTOR,
        column_extension_lateral_kpc=cfg.COLUMN_EXTENSION_LATERAL_KPC,
    )
    expected_key = field_cache_key(base_key, TDSP_FIELD)
    path = field_cache_path(cache_root_for_dataset(dataset), TDSP_FIELD).resolve()
    handle = h5py.File(path, "r")
    if str(handle.attrs.get("cache_key", "")) != expected_key:
        handle.close()
        raise RuntimeError(f"stale T_DESPOTIC cache: {path}")
    if tuple(handle["data"].shape) != dimensions:
        handle.close()
        raise ValueError(f"T_DESPOTIC cache shape mismatch: {path}")
    return handle, path


def _empty_policy_counts() -> dict[str, int]:
    return {
        "sampled_in_bounds": 0,
        "nH_below_table": 0,
        "nH_above_table": 0,
        "T_below_table": 0,
        "T_above_table": 0,
        "either_axis_out_of_bounds": 0,
        "touches_failure": 0,
        "touches_failure_TQUOKKA_lt_3000": 0,
        "touches_failure_TQUOKKA_ge_3000": 0,
        "failure_weight_ge_0p01": 0,
        "failure_weight_ge_0p10": 0,
        "failure_weight_ge_0p25": 0,
        "failure_weight_ge_0p50": 0,
        "failure_weight_nearly_one": 0,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table", type=Path,
        default=root / "data/cloudy_cii_hm2012_z0_baseline_jeans_10x21_T3p6_to1e9.npz",
    )
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument(
        "--despotic-table", type=Path, default=Path(cfg.DESPOTIC_TABLE_PATH),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(cfg.OUTPUT_DIR) / "cloudy_cii_jeans_failure_sampling.json",
    )
    parser.add_argument("--slab-nz", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in ("table", "dataset", "despotic_table", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    if args.slab_nz <= 0:
        raise ValueError("--slab-nz must be positive")

    with np.load(args.table, allow_pickle=False) as table:
        log_nh_axis = np.asarray(table["log_nH"], dtype=float)
        log_t_axis = np.asarray(table["log_T"], dtype=float)
        failure = np.asarray(table["failure_mask"], dtype=bool)
    if failure.shape != (log_nh_axis.size, log_t_axis.size):
        raise ValueError("failure-mask shape does not match axes")
    failure_indices = np.argwhere(failure)
    failure_flat_size = failure.size

    ds = yt.load(str(args.dataset))
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    nx, ny, nz = dimensions
    tdsp_file, tdsp_path = _open_tdsp_cache(
        args.dataset, args.despotic_table, dimensions,
    )
    hydrogen_mass_g = float(mh.to_value("g"))
    counts = {policy: _empty_policy_counts() for policy in POLICIES}
    node_touch_counts = {
        policy: np.zeros(failure_flat_size, dtype=np.int64) for policy in POLICIES
    }
    node_weight_sums = {
        policy: np.zeros(failure_flat_size, dtype=float) for policy in POLICIES
    }
    node_max_weights = {
        policy: np.zeros(failure_flat_size, dtype=float) for policy in POLICIES
    }
    maximum_failure_weight = {policy: 0.0 for policy in POLICIES}
    all_cells = 0
    low_cells = 0
    started = time.perf_counter()

    try:
        for slab_number, iz in enumerate(range(0, nz, args.slab_nz), start=1):
            local_nz = min(args.slab_nz, nz - iz)
            left_edge = ds.domain_left_edge.copy()
            left_edge[2] += iz * (ds.domain_width[2] / ds.domain_dimensions[2])
            grid = ds.covering_grid(
                level=ds.max_level, left_edge=left_edge,
                dims=(nx, ny, local_nz),
            )
            density = np.asarray(
                grid[("gas", "density")].to("g/cm**3"), dtype=float,
            ).reshape(-1)
            temperature_qk = np.asarray(
                grid[("boxlib", "temperature")], dtype=float,
            ).reshape(-1)
            del grid
            temperature_dsp = np.asarray(
                tdsp_file["data"][:, :, iz:iz + local_nz], dtype=float,
            ).reshape(-1)
            n_h = density * float(cfg.X_H) / hydrogen_mass_g
            if not (
                np.isfinite(n_h).all() and np.isfinite(temperature_qk).all()
                and np.isfinite(temperature_dsp).all()
                and np.all(n_h > 0.0) and np.all(temperature_qk > 0.0)
                and np.all(temperature_dsp > 0.0)
            ):
                raise ValueError(f"invalid cell input in z slab {iz}:{iz + local_nz}")
            low = temperature_qk < SPLIT_K
            all_cells += n_h.size
            low_cells += int(np.count_nonzero(low))
            log_nh = np.log10(n_h)
            policy_temperatures = {
                "hybrid_TDESP_low_TQUOKKA_high": np.where(
                    low, temperature_dsp, temperature_qk,
                ),
                "TQUOKKA_only": temperature_qk,
            }

            for policy, temperature in policy_temperatures.items():
                log_t = np.log10(temperature)
                n_below = log_nh < log_nh_axis[0]
                n_above = log_nh > log_nh_axis[-1]
                t_below = log_t < log_t_axis[0]
                t_above = log_t > log_t_axis[-1]
                out = n_below | n_above | t_below | t_above
                local_counts = counts[policy]
                local_counts["nH_below_table"] += int(np.count_nonzero(n_below))
                local_counts["nH_above_table"] += int(np.count_nonzero(n_above))
                local_counts["T_below_table"] += int(np.count_nonzero(t_below))
                local_counts["T_above_table"] += int(np.count_nonzero(t_above))
                local_counts["either_axis_out_of_bounds"] += int(np.count_nonzero(out))
                valid = ~out
                local_counts["sampled_in_bounds"] += int(np.count_nonzero(valid))
                if not np.any(valid):
                    continue
                n_bracket = _brackets(log_nh_axis, log_nh[valid])
                t_bracket = _brackets(log_t_axis, log_t[valid])
                failure_weight = np.zeros(int(np.count_nonzero(valid)))
                for ni in (0, 1):
                    n_index = n_bracket[ni]
                    n_weight = n_bracket[2] if ni else 1.0 - n_bracket[2]
                    for ti in (0, 1):
                        t_index = t_bracket[ti]
                        t_weight = t_bracket[2] if ti else 1.0 - t_bracket[2]
                        weight = n_weight * t_weight
                        corner_failure = failure[n_index, t_index]
                        supported = corner_failure & (weight > TOUCH_EPS)
                        failure_weight += corner_failure * weight
                        if np.any(supported):
                            flat = np.ravel_multi_index(
                                (n_index[supported], t_index[supported]), failure.shape,
                            )
                            support_weight = weight[supported]
                            node_touch_counts[policy] += np.bincount(
                                flat, minlength=failure_flat_size,
                            )
                            node_weight_sums[policy] += np.bincount(
                                flat, weights=support_weight,
                                minlength=failure_flat_size,
                            )
                            np.maximum.at(
                                node_max_weights[policy], flat, support_weight,
                            )
                touched = failure_weight > TOUCH_EPS
                valid_low = low[valid]
                local_counts["touches_failure"] += int(np.count_nonzero(touched))
                local_counts["touches_failure_TQUOKKA_lt_3000"] += int(
                    np.count_nonzero(touched & valid_low)
                )
                local_counts["touches_failure_TQUOKKA_ge_3000"] += int(
                    np.count_nonzero(touched & ~valid_low)
                )
                for threshold, key in (
                    (0.01, "failure_weight_ge_0p01"),
                    (0.10, "failure_weight_ge_0p10"),
                    (0.25, "failure_weight_ge_0p25"),
                    (0.50, "failure_weight_ge_0p50"),
                    (1.0 - 1.0e-12, "failure_weight_nearly_one"),
                ):
                    local_counts[key] += int(np.count_nonzero(failure_weight >= threshold))
                maximum_failure_weight[policy] = max(
                    maximum_failure_weight[policy], float(failure_weight.max()),
                )
            elapsed = time.perf_counter() - started
            print(
                f"[{slab_number:02d}/{(nz + args.slab_nz - 1) // args.slab_nz:02d}] "
                f"z={iz}:{iz + local_nz} elapsed={elapsed / 60.0:.2f} min",
                flush=True,
            )
    finally:
        tdsp_file.close()

    node_reports = {}
    for policy in POLICIES:
        local = []
        for i, j in failure_indices:
            flat = np.ravel_multi_index((i, j), failure.shape)
            touches = int(node_touch_counts[policy][flat])
            if touches == 0:
                continue
            local.append({
                "density_index": int(i), "temperature_index": int(j),
                "log_nH": float(log_nh_axis[i]),
                "log_T": float(log_t_axis[j]),
                "temperature_K": float(10.0 ** log_t_axis[j]),
                "corner_touch_count": touches,
                "corner_weight_sum": float(node_weight_sums[policy][flat]),
                "maximum_corner_weight": float(node_max_weights[policy][flat]),
            })
        node_reports[policy] = local

    result = {
        "dataset": str(args.dataset), "cloudy_table": str(args.table),
        "table_shape": list(failure.shape),
        "table_temperature_bounds_K": [
            float(10.0 ** log_t_axis[0]), float(10.0 ** log_t_axis[-1]),
        ],
        "original_failure_nodes": int(np.count_nonzero(failure)),
        "touch_definition": "bilinear original-failure weight > 1e-12",
        "temperature_policies": {
            "hybrid_TDESP_low_TQUOKKA_high": (
                "T_DESPOTIC where T_QUOKKA<3000 K; otherwise T_QUOKKA"
            ),
            "TQUOKKA_only": "T_QUOKKA for every cell",
        },
        "all_cells": all_cells, "T_QUOKKA_lt_3000_cells": low_cells,
        "counts": counts,
        "fractions": {
            policy: {
                "touched_per_in_bounds_cell": (
                    counts[policy]["touches_failure"]
                    / counts[policy]["sampled_in_bounds"]
                    if counts[policy]["sampled_in_bounds"] else 0.0
                ),
                "out_of_bounds_per_all_cell": (
                    counts[policy]["either_axis_out_of_bounds"] / all_cells
                ),
            }
            for policy in POLICIES
        },
        "maximum_failure_weight": maximum_failure_weight,
        "touched_failure_nodes": node_reports,
        "temperature_despotic_cache": str(tdsp_path),
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
