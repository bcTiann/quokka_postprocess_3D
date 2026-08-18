"""Combine the completed low/high HM2012 Cloudy H I 21-cm scans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_cloudy_halpha_temperature_tail import load_grid, read_run


ZERO_SENTINEL_MAX = -90.0


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    example = root / "work/cloudy_cooling_tools_history/examples/grackle"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--low-input", type=Path,
        default=example / "hm_2012_hi21_5x5_low_output",
    )
    parser.add_argument(
        "--high-input", type=Path,
        default=example / "hm_2012_hi21_5x5_high_output",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "data/cloudy_hi21_hm2012_z0_5x5.npz",
    )
    parser.add_argument(
        "--supplement-input", type=Path,
        default=example / "hm_2012_hi21_5x5_high_secant_repair_output",
    )
    parser.add_argument(
        "--failure-manifest", type=Path,
        default=example / "hm_2012_hi21_5x5_failures.json",
    )
    args = parser.parse_args()

    low = load_grid(
        args.low_input, t_min=10.0, t_max=3000.0, t_points=11,
    )
    high = load_grid(
        args.high_input, t_min=3000.0, t_max=1.0e7, t_points=17,
    )
    low_nh, low_column, low_t, low_raw, low_failure = low
    high_nh, high_column, high_t, high_raw, high_failure = high
    if not np.array_equal(low_nh, high_nh):
        raise ValueError("low/high log_nH axes differ")
    if not np.array_equal(low_column, high_column):
        raise ValueError("low/high log_NH axes differ")
    if not np.isclose(low_t[-1], high_t[0], atol=1.0e-12, rtol=0.0):
        raise ValueError("low/high temperature grids do not meet at 3000 K")

    boundary_valid = ~low_failure[..., -1] & ~high_failure[..., 0]
    boundary_difference = np.abs(low_raw[..., -1] - high_raw[..., 0])
    max_boundary_difference = float(np.max(boundary_difference[boundary_valid]))
    if max_boundary_difference > 5.0e-4:
        raise ValueError(
            "independent 3000 K scans disagree by more than output precision: "
            f"{max_boundary_difference:.6g} dex"
        )

    # Keep the low-scan copy of the duplicated 3000 K plane.  The independently
    # generated high-scan copy is checked above and is numerically identical at
    # CIAOLoop_lines output precision.
    log_t = np.concatenate((low_t, high_t[1:]))
    raw_log = np.concatenate((low_raw, high_raw[..., 1:]), axis=-1)
    failure_mask = np.concatenate(
        (low_failure, high_failure[..., 1:]), axis=-1,
    )

    supplement_overlap: list[float] = []
    supplement_fills = 0
    if args.supplement_input.exists():
        nh_index = {value: i for i, value in enumerate(low_nh)}
        column_index = {value: j for j, value in enumerate(low_column)}
        supplement_files = sorted(args.supplement_input.glob("*_run*.dat"))
        if not supplement_files:
            raise ValueError(
                f"no supplement .dat files in {args.supplement_input}"
            )
        for path in supplement_files:
            run_nh, run_column, rows = read_run(path)
            if run_nh not in nh_index or run_column not in column_index:
                raise ValueError(f"supplement point outside primary grid: {path}")
            indices = np.abs(rows[:, 0, None] - log_t[None, :]).argmin(axis=1)
            residual = np.abs(rows[:, 0] - log_t[indices])
            if np.any(residual > 5.1e-4):
                raise ValueError(f"supplement temperature is off-grid: {path}")
            i = nh_index[run_nh]
            j = column_index[run_column]
            for row, k in zip(rows, indices):
                value = float(row[1])
                if failure_mask[i, j, k]:
                    raw_log[i, j, k] = value
                    failure_mask[i, j, k] = False
                    supplement_fills += 1
                else:
                    supplement_overlap.append(abs(value - raw_log[i, j, k]))

    max_supplement_difference = max(supplement_overlap, default=0.0)
    if max_supplement_difference > 0.02:
        raise ValueError(
            "secant/default solver overlap differs by more than 0.02 dex: "
            f"{max_supplement_difference:.6g} dex"
        )

    original_failure_mask = failure_mask.copy()
    interpolated_mask = np.zeros_like(failure_mask)
    # The simulation audit found that only this failed node participates in
    # any interpolation stencil.  Fill exactly this node, as requested, by
    # linear interpolation in log10(epsilon/n_H^2) versus log10(T).
    requested_fill = (1.0, 24.0, 5.238560627359831)
    fill_i = int(np.abs(low_nh - requested_fill[0]).argmin())
    fill_j = int(np.abs(low_column - requested_fill[1]).argmin())
    fill_k = int(np.abs(log_t - requested_fill[2]).argmin())
    if not failure_mask[fill_i, fill_j, fill_k]:
        raise ValueError("requested H I repair node is not marked as failed")
    valid_temperature = ~failure_mask[fill_i, fill_j]
    lower = np.flatnonzero(valid_temperature & (log_t < log_t[fill_k]))
    upper = np.flatnonzero(valid_temperature & (log_t > log_t[fill_k]))
    if lower.size == 0 or upper.size == 0:
        raise ValueError("requested H I repair node is not bracketed in temperature")
    lo = int(lower[-1])
    hi = int(upper[0])
    raw_log[fill_i, fill_j, fill_k] = np.interp(
        log_t[fill_k], log_t[[lo, hi]], raw_log[fill_i, fill_j, [lo, hi]],
    )
    failure_mask[fill_i, fill_j, fill_k] = False
    interpolated_mask[fill_i, fill_j, fill_k] = True
    zero_mask = (~failure_mask) & (raw_log <= ZERO_SENTINEL_MAX)
    coefficient = np.zeros_like(raw_log)
    positive = ~failure_mask & ~zero_mask
    coefficient[positive] = np.power(10.0, raw_log[positive])

    failures = []
    for i, j, k in np.argwhere(failure_mask):
        failures.append({
            "nH_index": int(i),
            "NH_index": int(j),
            "T_index": int(k),
            "log_nH": float(low_nh[i]),
            "log_NH": float(low_column[j]),
            "log_T": float(log_t[k]),
            "temperature_K": float(10.0 ** log_t[k]),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray(2, dtype=np.int32),
        log_nH=low_nh,
        log_NH=low_column,
        log_T=log_t,
        log_emissivity_per_nH2=raw_log,
        emissivity_per_nH2=coefficient,
        failure_mask=failure_mask,
        zero_mask=zero_mask,
        original_failure_mask=original_failure_mask,
        interpolated_mask=interpolated_mask,
        out_of_bounds_policy=np.asarray(
            "temperature_above_max_zero; other_axes_raise"
        ),
        line_label=np.asarray("H  1 21.1207c"),
        uv_background=np.asarray("HM2012 z=0 shielded"),
        cloudy_version=np.asarray("17.02"),
        h2_molecule_enabled=np.asarray(False),
        charge_transfer_enabled=np.asarray(False),
        column_model=np.asarray(
            "explicit stop column density; no Jeans length"
        ),
        interpolation_policy=np.asarray(
            "log coefficient; one simulation-used failed node linearly "
            "interpolated along log_T"
        ),
        failed_node_policy=np.asarray(
            "runtime raises if interpolation weight > 1e-12"
        ),
    )

    args.failure_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.failure_manifest.write_text(json.dumps({
        "low_input": str(args.low_input.resolve()),
        "high_input": str(args.high_input.resolve()),
        "supplement_input": (
            str(args.supplement_input.resolve())
            if args.supplement_input.exists() else None
        ),
        "output": str(args.output.resolve()),
        "grid_shape": list(raw_log.shape),
        "temperature_bounds_K": [
            float(10.0 ** log_t[0]), float(10.0 ** log_t[-1]),
        ],
        "max_low_high_3000K_difference_dex": max_boundary_difference,
        "supplement_filled_nodes": supplement_fills,
        "supplement_overlap_points": len(supplement_overlap),
        "max_default_vs_secant_overlap_difference_dex": (
            max_supplement_difference
        ),
        "linearly_interpolated_nodes": [{
            "nH_index": fill_i,
            "NH_index": fill_j,
            "T_index": fill_k,
            "log_nH": float(low_nh[fill_i]),
            "log_NH": float(low_column[fill_j]),
            "log_T": float(log_t[fill_k]),
            "temperature_K": float(10.0 ** log_t[fill_k]),
            "lower_log_T": float(log_t[lo]),
            "upper_log_T": float(log_t[hi]),
            "filled_log_emissivity_per_nH2": float(
                raw_log[fill_i, fill_j, fill_k]
            ),
        }],
        "missing_calculations": len(failures),
        "failures": failures,
    }, indent=2) + "\n")
    print(
        f"Wrote {args.output}: shape={raw_log.shape}, "
        f"failures={len(failures)}, exact_zeros={int(zero_mask.sum())}, "
        f"3000K_overlap={max_boundary_difference:.6g} dex, "
        f"supplement_fills={supplement_fills}, "
        f"linear_fills={int(interpolated_mask.sum())}, "
        f"max_solver_overlap={max_supplement_difference:.6g} dex"
    )


if __name__ == "__main__":
    main()
