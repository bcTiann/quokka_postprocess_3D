"""Summarize QUOKKA field extrema from text headers, without reading Cell_D.

Header supplies field names and geometry. Level_0/Cell_H supplies one minimum
and maximum per field per native block. This first survey obtains all-cell
rho, T_QUOKKA and nH ranges; NH, dV/dr and hot-only ranges require a cell scan.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
import re

import numpy as np
import yaml

from quokka2s.physics.gas_fields import (
    HYDROGEN_MASS_G,
    hydrogen_number_density_cm3,
)
from quokka2s.physics.settings import X_H


@dataclass(frozen=True)
class SnapshotHeader:
    """Small text metadata: ordered field names, native shape and time [s]."""

    field_names: tuple[str, ...]
    shape: tuple[int, int, int]
    time_s: float


def read_box_shape(box_text: str) -> tuple[int, int, int]:
    """Read inclusive AMReX cell bounds; (0,0,0)..(255,255,2047) gives shape."""
    coordinates = re.findall(r"\(([-\d,\s]+)\)", box_text)
    start = np.fromstring(coordinates[0], sep=",", dtype=int)
    stop = np.fromstring(coordinates[1], sep=",", dtype=int)
    return tuple(int(size) for size in stop - start + 1)


def read_snapshot_header(path: Path) -> SnapshotHeader:
    """Read field order, 3-D level-zero geometry and time from Header only."""
    with path.open("r") as header:
        header.readline()  # Plotfile format name, e.g. HyperCLaw-V1.1.
        field_count = int(header.readline())
        field_names = tuple(header.readline().strip() for _ in range(field_count))
        dimensions = int(header.readline())
        time_s = float(header.readline())
        finest_level = int(header.readline())
        if dimensions != 3 or finest_level != 0:
            raise ValueError("Header survey supports 3-D uniform level-zero snapshots")
        header.readline()  # Physical left edge.
        header.readline()  # Physical right edge.
        header.readline()  # Refinement ratios; blank for these level-zero files.
        shape = read_box_shape(header.readline())
    return SnapshotHeader(field_names=field_names, shape=shape, time_s=time_s)


def check_cgs_metadata(path: Path) -> None:
    """Confirm the recorded units match this pipeline's cgs/K interpretation.

    These QUOKKA snapshots record unit_length/mass/time/temperature = 1.
    Reject other conventions rather than silently treating code units as cgs.
    No yt dataset or binary field is opened by this check.
    """
    with path.open("r") as metadata_file:
        units = yaml.safe_load(metadata_file)["units"]
    for unit_name in ("unit_length", "unit_mass", "unit_time", "unit_temperature"):
        if float(units[unit_name]) != 1.0:
            raise ValueError(f"Header survey requires cgs/K metadata: {unit_name}")


def read_extrema_group(
    *,
    lines,
    block_count: int,
    field_count: int,
) -> np.ndarray:
    """Read one (block_count, field_count) table of minima or maxima.

    Each numeric row has a trailing comma. NumPy reads only the actual field
    columns. Example: '128,8' introduces 128 block rows with eight fields.
    """
    dimensions = tuple(int(value) for value in next(lines).split(","))
    if dimensions != (block_count, field_count):
        raise ValueError(f"Unexpected extrema dimensions: {dimensions}")
    rows = [next(lines) for _ in range(block_count)]
    return np.loadtxt(
        StringIO("\n".join(rows)),
        delimiter=",",
        usecols=range(field_count),
        ndmin=2,
    )


def read_block_extrema(
    *,
    path: Path,
    snapshot_header: SnapshotHeader,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Reduce stored block extrema to two (field_count,) arrays, in Header order.

    Only Cell_H version 1, no-ghost headers with explicit minima/maxima are
    accepted. FabOnDisk rows are skipped; their Cell_D files are never opened.
    """
    lines = iter(line.strip() for line in path.read_text().splitlines() if line.strip())
    version = int(next(lines))
    next(lines)  # Storage layout; file locations are not needed for extrema.
    field_count = int(next(lines))
    ghost_count = int(next(lines))
    if version != 1 or ghost_count != 0:
        raise ValueError("Expected Cell_H version 1 with no ghost-cell extrema")
    if field_count != len(snapshot_header.field_names):
        raise ValueError("Header and Cell_H disagree on the field count")

    block_count = int(next(lines).lstrip("(").split()[0])
    block_shapes = [read_box_shape(next(lines)) for _ in range(block_count)]
    if next(lines) != ")" or int(next(lines)) != block_count:
        raise ValueError("Incomplete Cell_H block list")
    if sum(int(np.prod(shape)) for shape in block_shapes) != int(np.prod(snapshot_header.shape)):
        raise ValueError("Cell_H block cell counts do not match the full native grid")
    for _ in range(block_count):
        if not next(lines).startswith("FabOnDisk:"):
            raise ValueError("Incomplete Cell_H file-location list")

    minima = read_extrema_group(lines=lines, block_count=block_count, field_count=field_count)
    maxima = read_extrema_group(lines=lines, block_count=block_count, field_count=field_count)
    if not np.isfinite(minima).all() or not np.isfinite(maxima).all() or np.any(minima > maxima):
        raise ValueError("Nonfinite or reversed stored field extrema")
    return minima.min(axis=0), maxima.max(axis=0), block_count


def read_snapshot_ranges(*, snapshot: Path, run_name: str) -> dict:
    """Read three text files and return stored extrema plus converted nH.

    All extrema describe all native cells, not the T_QUOKKA >= 3000 K subset.
    File size/mtime are checked before/after reading to flag changing metadata;
    this does not verify the existence or completeness of binary payloads.
    """
    paths = (snapshot / "Header", snapshot / "Level_0/Cell_H", snapshot / "metadata.yaml")
    before = [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    header = read_snapshot_header(paths[0])
    check_cgs_metadata(paths[2])
    minima, maxima, block_count = read_block_extrema(path=paths[1], snapshot_header=header)
    after = [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    if before != after:
        raise ValueError("Snapshot text metadata changed during the read")

    field_ranges = {}
    for index, name in enumerate(header.field_names):
        field_ranges[name] = {"minimum": float(minima[index]), "maximum": float(maxima[index])}
    density_range = field_ranges["gasDensity"]
    hydrogen_range = hydrogen_number_density_cm3(
        density_g_cm3=np.array([density_range["minimum"], density_range["maximum"]]),
    )
    return {
        "run": run_name,
        "snapshot": snapshot.name,
        "source": str(snapshot.resolve()),
        "status": "header_read_ok",
        "shape": list(header.shape),
        "time_s": header.time_s,
        "block_count": block_count,
        "fields_in_native_units": field_ranges,
        "rho_g_cm3": density_range,
        "temperature_QUOKKA_K": field_ranges["temperature"],
        "nH_cm3": {"minimum": float(hydrogen_range[0]), "maximum": float(hydrogen_range[1])},
    }


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
