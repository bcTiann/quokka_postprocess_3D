"""Read QUOKKA geometry and stored extrema from three text headers only.

Header records native fields, grid shape and time. Level_0/Cell_H records
per-block minima/maxima; metadata.yaml confirms the existing cgs/K units.
No Cell_D payload is opened. Returned extrema describe all native cells.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import re

import numpy as np
import yaml

from quokka2s.physics.gas_fields import hydrogen_number_density_cm3


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
