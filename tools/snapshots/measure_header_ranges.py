"""Summarize QUOKKA field extrema from text headers, without reading Cell_D.

Header supplies field names and geometry. Level_0/Cell_H supplies one minimum
and maximum per field per native block. This first survey obtains all-cell
rho, T_QUOKKA and nH ranges; NH, dV/dr and hot-only ranges require a cell scan.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import yaml

from quokka2s.physics.gas_fields import HYDROGEN_MASS_G
from quokka2s.physics.settings import X_H
from quokka2s.snapshot_headers import (
    SnapshotHeader,
    check_cgs_metadata,
    read_block_extrema,
    read_box_shape,
    read_extrema_group,
    read_snapshot_header,
    read_snapshot_ranges,
)


def find_snapshots(*, source_root: Path, runs: list[str]) -> list[tuple[str, Path]]:
    """List only plt followed by digits in each explicitly named run directory."""
    snapshots = []
    for run_name in runs:
        run_directory = (source_root / run_name).resolve(strict=True)
        if not run_directory.is_relative_to(source_root):
            raise ValueError(f"Run directory lies outside the source root: {run_name}")
        candidates = sorted(run_directory.iterdir())
        for path in candidates:
            if not re.fullmatch(r"plt\d+", path.name) or not path.is_dir():
                continue
            if not path.resolve().is_relative_to(source_root):
                raise ValueError(f"Snapshot symlink leaves the source root: {path}")
            snapshots.append((run_name, path))
    return snapshots


def summarize_ranges(records: list[dict]) -> dict:
    """Combine successfully read headers and identify each extremum's snapshot."""
    successful = [record for record in records if record["status"] == "header_read_ok"]
    summary = {"snapshots_read": len(successful), "snapshots_failed": len(records) - len(successful)}
    for field_name in ("rho_g_cm3", "temperature_QUOKKA_K", "nH_cm3"):
        if not successful:
            summary[field_name] = None
            continue
        low = min(successful, key=lambda record: record[field_name]["minimum"])
        high = max(successful, key=lambda record: record[field_name]["maximum"])
        summary[field_name] = {
            "minimum": low[field_name]["minimum"],
            "maximum": high[field_name]["maximum"],
            "minimum_source": low["source"],
            "maximum_source": high["source"],
        }
    return summary


def prepare_output_paths(*, source_root: Path, output_directory: Path) -> dict[str, Path]:
    """Keep every new output outside the source tree and inside its destination.

    Resolve symlinks before checking containment. Exclusive file creation later
    also refuses existing files and symlinks, so no existing result is replaced.
    """
    output_directory = output_directory.resolve()
    if output_directory.is_relative_to(source_root):
        raise ValueError("Outputs must be outside the simulation source tree")
    paths = {}
    for name in ("snapshots.csv", "jobs.csv", "summary.json"):
        target = output_directory / name
        if not target.resolve().is_relative_to(output_directory):
            raise ValueError(f"Output symlink escapes the destination: {target}")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Choose a new header-survey output directory: {target}")
        paths[name] = target
    output_directory.mkdir(parents=True, exist_ok=True)
    return paths


def write_range_csv(*, path: Path, records: list[dict]) -> None:
    """Save one row per snapshot/job; columns state physical quantities and units."""
    columns = [
        "run", "snapshot", "status", "shape", "time_s", "block_count",
        "snapshots_read", "snapshots_failed", "error", "source",
    ]
    for field_name in ("rho_g_cm3", "temperature_QUOKKA_K", "nH_cm3"):
        columns.extend([f"{field_name}_min", f"{field_name}_max"])
    with path.open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {name: record[name] for name in columns if name in record}
            if "shape" in record:
                row["shape"] = "x".join(str(size) for size in record["shape"])
            for field_name in ("rho_g_cm3", "temperature_QUOKKA_K", "nH_cm3"):
                if record.get(field_name) is not None:
                    row[f"{field_name}_min"] = record[field_name]["minimum"]
                    row[f"{field_name}_max"] = record[field_name]["maximum"]
            writer.writerow(row)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories relative to source-root")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory outside source-root")
    args = parser.parse_args(argv)
    source_root = args.source_root.expanduser().resolve(strict=True)
    output_paths = prepare_output_paths(source_root=source_root, output_directory=args.output_dir.expanduser())
    snapshots = find_snapshots(source_root=source_root, runs=args.runs)
    print(f"Header survey: {len(snapshots)} snapshots; no Cell_D reads", flush=True)

    records = []
    for index, (run_name, snapshot) in enumerate(snapshots, start=1):
        try:
            record = read_snapshot_ranges(snapshot=snapshot, run_name=run_name)
        except (OSError, ValueError, KeyError, IndexError, StopIteration, yaml.YAMLError) as error:
            record = {
                "run": run_name,
                "snapshot": snapshot.name,
                "source": str(snapshot.resolve()),
                "status": "header_read_failed",
                "error": f"{type(error).__name__}: {error}",
            }
        records.append(record)
        print(f"[{index}/{len(snapshots)}] {run_name}/{snapshot.name}: {record['status']}", flush=True)

    jobs = []
    for run_name in args.runs:
        run_records = [record for record in records if record["run"] == run_name]
        jobs.append({"run": run_name, **summarize_ranges(run_records)})
    result = {
        "survey": "text headers only; all native cells",
        "source_root": str(source_root),
        "X_H": float(X_H),
        "hydrogen_mass_g": float(HYDROGEN_MASS_G),
        "units": {"rho": "g/cm^3", "temperature_QUOKKA": "K", "nH": "cm^-3"},
        "not_measured": ["NH", "dVdr", "hot-only nH and T_QUOKKA", "binary payload completeness"],
        "overall": summarize_ranges(records),
        "jobs": jobs,
        "snapshots": records,
    }
    write_range_csv(path=output_paths["snapshots.csv"], records=records)
    write_range_csv(path=output_paths["jobs.csv"], records=jobs)
    with output_paths["summary.json"].open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(result["overall"], indent=2), flush=True)
    print(f"Saved header statistics: {output_paths['summary.json'].parent}", flush=True)


if __name__ == "__main__":
    main()
