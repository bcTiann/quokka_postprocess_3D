"""Audit simulation use of failures in the four-state Cloudy line bundle.

The canonical input is produced by
``scripts/build_cloudy_line_physics_ablation_tables.py`` and has axis order
``(state, line, log_nH, log_NH, log_T)``.  This program reads the QUOKKA
temperature and density, plus the cached total-hydrogen column, once per
z-slab.  The same trilinear brackets are then reused for all four physics
states and all three lines.

Failed Cloudy nodes remain unavailable.  This audit never fills them and does
not estimate luminosity from them.  It reports the interpolation weight given
to failed nodes, input-bound coverage, and separate results below/above the
pipeline's 3000 K regime boundary.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
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


EXPECTED_AXIS_ORDER = "state,line,log_nH,log_NH,log_T"
EXPECTED_STATES = ("baseline", "mol", "ct", "mol_ct")
EXPECTED_LINES = ("cii", "halpha", "hi21")
COLUMN_FIELD = ("gas", "column_density_H")
REGIME_SPLIT_K = 3000.0
TOUCH_EPS = 1.0e-12


@dataclass(frozen=True)
class AblationBundle:
    path: Path
    states: tuple[str, ...]
    lines: tuple[str, ...]
    line_labels: tuple[str, ...]
    log_nH: np.ndarray
    log_NH: np.ndarray
    log_T: np.ndarray
    failure_mask: np.ndarray
    original_failure_mask: np.ndarray


@dataclass
class WeightStats:
    sampled_cells: int = 0
    touched_cells: int = 0
    sum_weight: float = 0.0
    maximum_weight: float = 0.0
    weight_ge_001: int = 0
    weight_ge_010: int = 0
    weight_ge_025: int = 0
    weight_ge_050: int = 0
    weight_nearly_one: int = 0

    def update(self, weight: np.ndarray) -> None:
        if weight.size == 0:
            return
        self.sampled_cells += int(weight.size)
        self.touched_cells += int(np.count_nonzero(weight > TOUCH_EPS))
        self.sum_weight += float(np.sum(weight))
        self.maximum_weight = max(self.maximum_weight, float(np.max(weight)))
        self.weight_ge_001 += int(np.count_nonzero(weight >= 0.01))
        self.weight_ge_010 += int(np.count_nonzero(weight >= 0.10))
        self.weight_ge_025 += int(np.count_nonzero(weight >= 0.25))
        self.weight_ge_050 += int(np.count_nonzero(weight >= 0.50))
        self.weight_nearly_one += int(
            np.count_nonzero(weight >= 1.0 - 1.0e-10)
        )

    def as_dict(self) -> dict[str, object]:
        sampled = self.sampled_cells
        return {
            "sampled_cells": sampled,
            "touches_failure_node": self.touched_cells,
            "touched_fraction_of_sampled_cells": (
                self.touched_cells / sampled if sampled else 0.0
            ),
            "mean_failure_interpolation_weight": (
                self.sum_weight / sampled if sampled else 0.0
            ),
            "maximum_failure_interpolation_weight": self.maximum_weight,
            "failure_weight_threshold_counts": {
                "ge_0.01": self.weight_ge_001,
                "ge_0.10": self.weight_ge_010,
                "ge_0.25": self.weight_ge_025,
                "ge_0.50": self.weight_ge_050,
                "nearly_one": self.weight_nearly_one,
            },
        }


def _string_tuple(values: np.ndarray) -> tuple[str, ...]:
    return tuple(str(value) for value in np.asarray(values).tolist())


def load_bundle(path: Path) -> AblationBundle:
    """Load and strictly validate the canonical five-dimensional bundle."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Cloudy ablation bundle does not exist: {path}\n"
            "Build it with scripts/build_cloudy_line_physics_ablation_tables.py."
        )
    with np.load(path, allow_pickle=False) as table:
        required = {
            "bundle_schema_version",
            "axis_order",
            "state_labels",
            "line_keys",
            "line_labels",
            "log_nH",
            "log_NH",
            "log_T",
            "failure_mask",
            "original_failure_mask",
        }
        missing = sorted(required - set(table.files))
        if missing:
            raise ValueError(f"bundle is missing fields: {missing}")
        schema = int(np.asarray(table["bundle_schema_version"]).item())
        axis_order = str(np.asarray(table["axis_order"]).item())
        states = _string_tuple(table["state_labels"])
        lines = _string_tuple(table["line_keys"])
        line_labels = _string_tuple(table["line_labels"])
        log_nH = np.asarray(table["log_nH"], dtype=float)
        log_NH = np.asarray(table["log_NH"], dtype=float)
        log_T = np.asarray(table["log_T"], dtype=float)
        failure_mask = np.asarray(table["failure_mask"], dtype=bool)
        original_failure_mask = np.asarray(
            table["original_failure_mask"], dtype=bool,
        )

    if schema != 1:
        raise ValueError(f"unsupported ablation bundle schema {schema}")
    if axis_order != EXPECTED_AXIS_ORDER:
        raise ValueError(
            f"bundle axis_order={axis_order!r}, expected {EXPECTED_AXIS_ORDER!r}"
        )
    if states != EXPECTED_STATES:
        raise ValueError(f"unexpected state labels: {states}")
    if lines != EXPECTED_LINES:
        raise ValueError(f"unexpected line keys: {lines}")
    if len(line_labels) != len(lines):
        raise ValueError("line_labels and line_keys have different lengths")
    for name, axis in (("log_nH", log_nH), ("log_NH", log_NH), ("log_T", log_T)):
        if axis.ndim != 1 or axis.size < 2 or np.any(np.diff(axis) <= 0.0):
            raise ValueError(f"{name} must be a strictly increasing 1D axis")
    expected_shape = (
        len(states), len(lines), log_nH.size, log_NH.size, log_T.size,
    )
    if failure_mask.shape != expected_shape:
        raise ValueError(
            f"failure_mask shape {failure_mask.shape} != {expected_shape}"
        )
    if original_failure_mask.shape != expected_shape:
        raise ValueError("original_failure_mask shape does not match bundle axes")
    if not np.array_equal(failure_mask, original_failure_mask):
        raise ValueError(
            "bundle failure_mask differs from original_failure_mask; this raw "
            "ablation audit requires an unfilled canonical bundle"
        )

    return AblationBundle(
        path=path.resolve(),
        states=states,
        lines=lines,
        line_labels=line_labels,
        log_nH=log_nH,
        log_NH=log_NH,
        log_T=log_T,
        failure_mask=failure_mask,
        original_failure_mask=original_failure_mask,
    )


def _open_validated_column_cache(
    dataset_path: Path,
    despotic_table_path: str | Path,
    explicit_path: Path | None,
) -> tuple[h5py.File, Path]:
    base_key = compute_cache_key(
        dataset_path=dataset_path,
        despotic_table_path=despotic_table_path,
        downsample_factor=cfg.DOWNSAMPLE_FACTOR,
        column_extension_lateral_kpc=cfg.COLUMN_EXTENSION_LATERAL_KPC,
    )
    expected_key = field_cache_key(base_key, COLUMN_FIELD)
    path = (
        explicit_path.resolve()
        if explicit_path is not None
        else field_cache_path(cache_root_for_dataset(dataset_path), COLUMN_FIELD)
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"required column-density cache does not exist: {path}\n"
            "Run the normal pipeline compute first."
        )
    handle = h5py.File(path, "r")
    if "data" not in handle:
        handle.close()
        raise ValueError(f"column cache has no 'data' dataset: {path}")
    actual_key = str(handle.attrs.get("cache_key", ""))
    actual_field = (
        str(handle.attrs.get("field_type", "")),
        str(handle.attrs.get("field_name", "")),
    )
    if actual_key != expected_key or actual_field != COLUMN_FIELD:
        handle.close()
        raise RuntimeError(
            f"stale or mismatched column cache: {path}\n"
            f"expected key/field {expected_key} {COLUMN_FIELD}, "
            f"found {actual_key} {actual_field}"
        )
    return handle, path


def _brackets(
    axis: np.ndarray,
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lower/upper indices and the upper-node linear weight."""
    clipped = np.clip(coordinates, axis[0], axis[-1])
    upper = np.searchsorted(axis, clipped, side="right")
    upper = np.clip(upper, 1, axis.size - 1)
    lower = upper - 1
    fraction = (clipped - axis[lower]) / (axis[upper] - axis[lower])
    # These tables have only ten or twenty nodes.  Compact indices reduce the
    # peak memory used by a native-resolution slab without changing indexing.
    return lower.astype(np.int16), upper.astype(np.int16), fraction


def _outside_axis(
    axis: np.ndarray,
    coordinates: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify coordinates outside an axis using the lookup's tolerance."""
    tolerance = 1.0e-12 * max(1.0, abs(axis[0]), abs(axis[-1]))
    return (
        valid & (coordinates < axis[0] - tolerance),
        valid & (coordinates > axis[-1] + tolerance),
    )


def _failure_weight(
    mask: np.ndarray,
    brackets: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
    touched_node_ids: set[int] | None = None,
) -> np.ndarray:
    output = np.zeros(brackets[0][2].shape, dtype=float)
    for n_corner in (0, 1):
        n_index = brackets[0][n_corner]
        n_weight = brackets[0][2] if n_corner else 1.0 - brackets[0][2]
        for column_corner in (0, 1):
            column_index = brackets[1][column_corner]
            column_weight = (
                brackets[1][2]
                if column_corner else 1.0 - brackets[1][2]
            )
            for temperature_corner in (0, 1):
                temperature_index = brackets[2][temperature_corner]
                temperature_weight = (
                    brackets[2][2]
                    if temperature_corner else 1.0 - brackets[2][2]
                )
                corner_weight = n_weight * column_weight * temperature_weight
                corner_failed = mask[
                    n_index, column_index, temperature_index,
                ]
                output += corner_weight * corner_failed
                if touched_node_ids is not None:
                    contributes = corner_failed & (corner_weight > TOUCH_EPS)
                    if np.any(contributes):
                        node_ids = np.ravel_multi_index(
                            (
                                n_index[contributes],
                                column_index[contributes],
                                temperature_index[contributes],
                            ),
                            mask.shape,
                        )
                        touched_node_ids.update(
                            int(value) for value in np.unique(node_ids)
                        )
    return output


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], bytes]:
    return mask.shape, np.packbits(mask, bitorder="little").tobytes()


def _new_stats() -> dict[str, WeightStats]:
    return {
        "all_temperatures": WeightStats(),
        "T_lt_3000K": WeightStats(),
        "T_ge_3000K": WeightStats(),
    }


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=(
            root
            / "data/cloudy_lines_hm2012_z0_physics_ablation_4state_3line_10x10x20.npz"
        ),
    )
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument(
        "--despotic-table", type=Path, default=Path(cfg.DESPOTIC_TABLE_PATH),
        help="used only to validate the pipeline column-density cache key",
    )
    parser.add_argument(
        "--column-cache",
        type=Path,
        default=None,
        help="override the normal pipeline column-density cache location",
    )
    parser.add_argument("--slab-nz", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(cfg.OUTPUT_DIR)
            / "cloudy_line_physics_ablation_failure_sampling.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.slab_nz <= 0:
        raise ValueError("--slab-nz must be positive")
    if cfg.DOWNSAMPLE_FACTOR != 1:
        raise NotImplementedError(
            "this native-slab audit currently requires DOWNSAMPLE_FACTOR=1"
        )

    bundle = load_bundle(args.bundle)
    dataset_path = args.dataset.resolve()
    column_file, column_path = _open_validated_column_cache(
        dataset_path,
        args.despotic_table,
        args.column_cache,
    )

    # A state-level result uses the union of its three line failure masks.  In
    # normal CIAOLoop output a Cloudy crash removes all three values together,
    # but taking the union keeps this diagnostic correct if that ever changes.
    state_masks = {
        state: np.any(bundle.failure_mask[state_index], axis=0)
        for state_index, state in enumerate(bundle.states)
    }
    line_masks = {
        (state, line): bundle.failure_mask[state_index, line_index]
        for state_index, state in enumerate(bundle.states)
        for line_index, line in enumerate(bundle.lines)
    }

    # Deduplicate identical masks.  The usual 12 logical line tables share one
    # crash mask per state, so each slab normally needs only four weight sums.
    unique_masks: list[np.ndarray] = []
    mask_index_by_key: dict[tuple[tuple[int, ...], bytes], int] = {}
    state_mask_index: dict[str, int] = {}
    line_mask_index: dict[tuple[str, str], int] = {}

    def register(mask: np.ndarray) -> int:
        key = _mask_key(mask)
        if key not in mask_index_by_key:
            mask_index_by_key[key] = len(unique_masks)
            unique_masks.append(mask)
        return mask_index_by_key[key]

    for state, mask in state_masks.items():
        state_mask_index[state] = register(mask)
    for state_line, mask in line_masks.items():
        line_mask_index[state_line] = register(mask)

    stats = [_new_stats() for _ in unique_masks]
    touched_nodes_by_mask: list[set[int]] = [set() for _ in unique_masks]
    bounds_counts = {
        "all_cells": 0,
        "finite_positive_inputs": 0,
        "invalid_or_nonpositive_inputs": 0,
        "fully_in_bounds": 0,
        "T_below_min": 0,
        "T_above_max": 0,
        "nH_below_min": 0,
        "nH_above_max": 0,
        "NH_below_min": 0,
        "NH_above_max": 0,
        "any_axis_outside": 0,
    }
    input_ranges = {
        "temperature_K": [np.inf, -np.inf],
        "n_H_cm-3": [np.inf, -np.inf],
        "N_H_cm-2": [np.inf, -np.inf],
    }

    ds = yt.load(str(dataset_path))
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    if tuple(column_file["data"].shape) != dimensions:
        column_file.close()
        raise ValueError(
            f"column cache shape {column_file['data'].shape} != dataset {dimensions}"
        )
    nx, ny, nz = dimensions
    cell_width = ds.domain_width.to("cm") / ds.domain_dimensions
    hydrogen_mass_g = float(mh.to_value("g"))
    n_slabs = (nz + args.slab_nz - 1) // args.slab_nz
    started = time.perf_counter()

    try:
        for slab_number, iz in enumerate(range(0, nz, args.slab_nz), start=1):
            local_nz = min(args.slab_nz, nz - iz)
            left_edge = ds.domain_left_edge.copy()
            left_edge[2] += iz * cell_width[2]
            grid = ds.covering_grid(
                level=ds.max_level,
                left_edge=left_edge,
                dims=(nx, ny, local_nz),
            )
            temperature = np.asarray(grid[("boxlib", "temperature")], dtype=float)
            density = np.asarray(
                grid[("gas", "density")].to("g/cm**3"), dtype=float,
            )
            n_H = density * float(cfg.X_H) / hydrogen_mass_g
            column = np.asarray(
                column_file["data"][:, :, iz:iz + local_nz], dtype=float,
            )
            del grid, density

            finite_positive = (
                np.isfinite(temperature)
                & np.isfinite(n_H)
                & np.isfinite(column)
                & (temperature > 0.0)
                & (n_H > 0.0)
                & (column > 0.0)
            )
            log_temperature = np.zeros_like(temperature, dtype=float)
            log_n_H = np.zeros_like(n_H, dtype=float)
            log_column = np.zeros_like(column, dtype=float)
            np.log10(temperature, out=log_temperature, where=finite_positive)
            np.log10(n_H, out=log_n_H, where=finite_positive)
            np.log10(column, out=log_column, where=finite_positive)

            t_below, t_above = _outside_axis(
                bundle.log_T, log_temperature, finite_positive,
            )
            n_below, n_above = _outside_axis(
                bundle.log_nH, log_n_H, finite_positive,
            )
            column_below, column_above = _outside_axis(
                bundle.log_NH, log_column, finite_positive,
            )
            outside = (
                t_below | t_above | n_below | n_above
                | column_below | column_above
            )
            in_bounds = finite_positive & ~outside

            bounds_counts["all_cells"] += int(temperature.size)
            bounds_counts["finite_positive_inputs"] += int(
                np.count_nonzero(finite_positive)
            )
            bounds_counts["invalid_or_nonpositive_inputs"] += int(
                temperature.size - np.count_nonzero(finite_positive)
            )
            bounds_counts["fully_in_bounds"] += int(np.count_nonzero(in_bounds))
            for key, mask in (
                ("T_below_min", t_below),
                ("T_above_max", t_above),
                ("nH_below_min", n_below),
                ("nH_above_max", n_above),
                ("NH_below_min", column_below),
                ("NH_above_max", column_above),
                ("any_axis_outside", outside),
            ):
                bounds_counts[key] += int(np.count_nonzero(mask))

            if np.any(finite_positive):
                for key, values in (
                    ("temperature_K", temperature),
                    ("n_H_cm-3", n_H),
                    ("N_H_cm-2", column),
                ):
                    selected = values[finite_positive]
                    input_ranges[key][0] = min(
                        input_ranges[key][0], float(np.min(selected)),
                    )
                    input_ranges[key][1] = max(
                        input_ranges[key][1], float(np.max(selected)),
                    )

            if np.any(in_bounds):
                selected_T = temperature[in_bounds]
                coordinates = (
                    log_n_H[in_bounds],
                    log_column[in_bounds],
                    log_temperature[in_bounds],
                )
                brackets = tuple(
                    _brackets(axis, coordinate)
                    for axis, coordinate in zip(
                        (bundle.log_nH, bundle.log_NH, bundle.log_T),
                        coordinates,
                    )
                )
                low_regime = selected_T < REGIME_SPLIT_K
                for mask_index, mask in enumerate(unique_masks):
                    weight = _failure_weight(
                        mask, brackets, touched_nodes_by_mask[mask_index],
                    )
                    stats[mask_index]["all_temperatures"].update(weight)
                    stats[mask_index]["T_lt_3000K"].update(weight[low_regime])
                    stats[mask_index]["T_ge_3000K"].update(weight[~low_regime])

            elapsed = time.perf_counter() - started
            rate = slab_number / elapsed if elapsed > 0.0 else 0.0
            remaining = (
                (n_slabs - slab_number) / rate if rate > 0.0 else np.nan
            )
            state_touches = {
                state: stats[index]["all_temperatures"].touched_cells
                for state, index in state_mask_index.items()
            }
            touch_text = " ".join(
                f"{state}={count:,}" for state, count in state_touches.items()
            )
            print(
                f"[{slab_number:02d}/{n_slabs:02d}] z={iz}:{iz + local_nz} "
                f"touches[{touch_text}] elapsed={elapsed / 60.0:.1f} min "
                f"ETA={remaining / 60.0:.1f} min",
                flush=True,
            )
    finally:
        column_file.close()

    state_results: dict[str, dict[str, object]] = {}
    for state_index, state in enumerate(bundle.states):
        mask = state_masks[state]
        touched_node_records: list[dict[str, object]] = []
        for node_id in sorted(touched_nodes_by_mask[state_mask_index[state]]):
            i, j, k = np.unravel_index(node_id, mask.shape)
            touched_node_records.append({
                "nH_index": int(i),
                "NH_index": int(j),
                "T_index": int(k),
                "log_nH": float(bundle.log_nH[i]),
                "log_NH": float(bundle.log_NH[j]),
                "log_T": float(bundle.log_T[k]),
                "temperature_K": float(10.0 ** bundle.log_T[k]),
            })
        line_specific_masks_identical = all(
            np.array_equal(mask, line_masks[(state, line)])
            for line in bundle.lines
        )
        state_results[state] = {
            "union_failure_nodes": int(np.count_nonzero(mask)),
            "unique_touched_failure_nodes": len(touched_node_records),
            "touched_failure_nodes": touched_node_records,
            "line_failure_masks_identical": line_specific_masks_identical,
            "regimes": {
                regime: value.as_dict()
                for regime, value in stats[state_mask_index[state]].items()
            },
            "lines": {
                line: {
                    "line_label": bundle.line_labels[line_index],
                    "failure_nodes": int(
                        np.count_nonzero(line_masks[(state, line)])
                    ),
                    "regimes": {
                        regime: value.as_dict()
                        for regime, value in stats[
                            line_mask_index[(state, line)]
                        ].items()
                    },
                }
                for line_index, line in enumerate(bundle.lines)
            },
        }

    finite_count = bounds_counts["finite_positive_inputs"]
    in_bounds_count = bounds_counts["fully_in_bounds"]
    serializable_input_ranges = {
        key: (
            [float(bounds[0]), float(bounds[1])]
            if np.isfinite(bounds).all() else [None, None]
        )
        for key, bounds in input_ranges.items()
    }
    result = {
        "dataset": str(dataset_path),
        "column_density_cache": str(column_path),
        "canonical_bundle": str(bundle.path),
        "bundle_grid_shape": list(bundle.failure_mask.shape),
        "bundle_axis_order": EXPECTED_AXIS_ORDER.split(","),
        "table_bounds": {
            "temperature_K": [
                float(10.0 ** bundle.log_T[0]),
                float(10.0 ** bundle.log_T[-1]),
            ],
            "n_H_cm-3": [
                float(10.0 ** bundle.log_nH[0]),
                float(10.0 ** bundle.log_nH[-1]),
            ],
            "N_H_cm-2": [
                float(10.0 ** bundle.log_NH[0]),
                float(10.0 ** bundle.log_NH[-1]),
            ],
        },
        "simulation_input_ranges": serializable_input_ranges,
        "bounds_counts": bounds_counts,
        "bounds_fractions": {
            "fully_in_bounds_per_finite_positive_cell": (
                in_bounds_count / finite_count if finite_count else 0.0
            ),
            "outside_per_finite_positive_cell": (
                bounds_counts["any_axis_outside"] / finite_count
                if finite_count else 0.0
            ),
        },
        "touch_definition": (
            f"sum of trilinear weights on unavailable nodes > {TOUCH_EPS:g}; "
            "evaluated only for cells fully inside all three table axes"
        ),
        "automatic_failure_fill": False,
        "luminosity_analysis": {
            "computed": False,
            "reason": (
                "failed nodes are unavailable and the canonical bundle contains "
                "no filled emissivity for them; estimating affected luminosity "
                "would introduce an unrequested interpolation assumption"
            ),
        },
        "unique_failure_masks_evaluated_per_slab": len(unique_masks),
        "states": state_results,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print("\nCloudy line-physics ablation failure audit")
    print("=" * 52)
    print(f"Fully in-bounds simulation cells : {in_bounds_count:,}")
    print(
        f"Outside any table axis          : "
        f"{bounds_counts['any_axis_outside']:,}"
    )
    for state in bundle.states:
        summary = state_results[state]["regimes"]["all_temperatures"]
        print(
            f"{state:8s} touches failures       : "
            f"{summary['touches_failure_node']:,} "
            f"({summary['touched_fraction_of_sampled_cells']:.6%}); "
            f"max weight={summary['maximum_failure_interpolation_weight']:.6g}"
        )
    print(f"JSON result                     : {args.output}")


if __name__ == "__main__":
    main()
