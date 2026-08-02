"""Summarize where a sparse Cloudy [C II] scan becomes exactly zero.

Cloudy's CIAOLoop sentinel ``-99`` is treated as an exact zero.  The report
also records how far the last non-zero point lies below each curve's peak,
but that relative number is diagnostic only and is never used as a cutoff.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


LOOP_RE = re.compile(r"^#\s*(hden|stop column density)\s+(.+?)\s*$")
ZERO_SENTINEL_MAX = -90.0


def read_run(path: Path) -> dict[str, object]:
    log_nh = log_nh_column = None
    rows: list[tuple[float, float]] = []
    for line in path.read_text().splitlines():
        match = LOOP_RE.match(line)
        if match:
            value = float(match.group(2))
            if match.group(1) == "hden":
                log_nh = value
            else:
                log_nh_column = value
            continue
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 2:
            rows.append((float(fields[0]), float(fields[1])))

    if log_nh is None or log_nh_column is None:
        raise ValueError(f"missing loop metadata: {path}")
    values = np.asarray(rows, dtype=float).reshape(-1, 2)
    if values.size == 0:
        raise ValueError(f"no temperature rows: {path}")

    log_t = values[:, 0]
    log_emissivity = values[:, 1]
    positive = log_emissivity > ZERO_SENTINEL_MAX
    zero = ~positive
    positive_indices = np.flatnonzero(positive)
    zero_indices = np.flatnonzero(zero)
    last_positive_index = int(positive_indices[-1]) if positive_indices.size else None
    first_zero_index = int(zero_indices[0]) if zero_indices.size else None
    reappears = bool(
        first_zero_index is not None
        and np.any(positive[first_zero_index + 1 :])
    )
    peak = float(np.max(log_emissivity[positive])) if positive_indices.size else None

    def point(index: int | None) -> dict[str, float] | None:
        if index is None:
            return None
        return {
            "log_T": float(log_t[index]),
            "temperature_K": float(10.0 ** log_t[index]),
            "log_emissivity_per_nH2": float(log_emissivity[index]),
        }

    last_positive = point(last_positive_index)
    if last_positive is not None and peak is not None:
        last_positive["dex_below_curve_peak"] = float(
            peak - last_positive["log_emissivity_per_nH2"]
        )

    return {
        "file": str(path.resolve()),
        "log_nH": float(log_nh),
        "log_NH": float(log_nh_column),
        "temperature_rows": int(values.shape[0]),
        "curve_peak_log_emissivity_per_nH2": peak,
        "last_positive": last_positive,
        "first_exact_zero": point(first_zero_index),
        "positive_reappears_after_first_zero": reappears,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_directory = (
        project_root
        / "work/cloudy_cooling_tools_history/examples/grackle"
        / "hm_2012_cii_zero_sparse_output"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=default_directory)
    parser.add_argument("--expected-runs", type=int, default=25)
    parser.add_argument("--expected-temperature-rows", type=int, default=29)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=default_directory / "zero_scan_summary.json",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=default_directory / "zero_scan_summary.csv",
    )
    args = parser.parse_args()

    files = sorted(args.directory.glob("*_run*.dat"))
    records = [read_run(path) for path in files]
    incomplete = [
        record
        for record in records
        if record["temperature_rows"] != args.expected_temperature_rows
    ]
    no_zero = [record for record in records if record["first_exact_zero"] is None]
    reappearing = [
        record
        for record in records
        if record["positive_reappears_after_first_zero"]
    ]
    complete = (
        len(records) == args.expected_runs
        and not incomplete
        and not no_zero
        and not reappearing
    )
    first_zero_temperatures = [
        record["first_exact_zero"]["temperature_K"]
        for record in records
        if record["first_exact_zero"] is not None
    ]
    summary = {
        "directory": str(args.directory.resolve()),
        "expected_runs": args.expected_runs,
        "runs_found": len(records),
        "expected_temperature_rows_per_run": args.expected_temperature_rows,
        "incomplete_runs": len(incomplete),
        "runs_without_exact_zero": len(no_zero),
        "runs_with_positive_reappearance": len(reappearing),
        "scan_complete_and_usable": complete,
        "largest_first_exact_zero_temperature_K": (
            max(first_zero_temperatures) if first_zero_temperatures else None
        ),
        "records": records,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2) + "\n")

    with args.csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "log_nH",
                "log_NH",
                "temperature_rows",
                "last_positive_log_T",
                "last_positive_temperature_K",
                "last_positive_log_emissivity_per_nH2",
                "last_positive_dex_below_curve_peak",
                "first_zero_log_T",
                "first_zero_temperature_K",
                "positive_reappears_after_first_zero",
            ],
        )
        writer.writeheader()
        for record in records:
            last = record["last_positive"] or {}
            first = record["first_exact_zero"] or {}
            writer.writerow(
                {
                    "log_nH": record["log_nH"],
                    "log_NH": record["log_NH"],
                    "temperature_rows": record["temperature_rows"],
                    "last_positive_log_T": last.get("log_T"),
                    "last_positive_temperature_K": last.get("temperature_K"),
                    "last_positive_log_emissivity_per_nH2": last.get(
                        "log_emissivity_per_nH2"
                    ),
                    "last_positive_dex_below_curve_peak": last.get(
                        "dex_below_curve_peak"
                    ),
                    "first_zero_log_T": first.get("log_T"),
                    "first_zero_temperature_K": first.get("temperature_K"),
                    "positive_reappears_after_first_zero": record[
                        "positive_reappears_after_first_zero"
                    ],
                }
            )

    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.csv_output}")
    print(
        f"runs={len(records)}/{args.expected_runs}, incomplete={len(incomplete)}, "
        f"no_zero={len(no_zero)}, reappearing={len(reappearing)}, "
        f"usable={complete}"
    )
    if first_zero_temperatures:
        print(
            "largest first exact-zero temperature: "
            f"{max(first_zero_temperatures):.6e} K"
        )
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
