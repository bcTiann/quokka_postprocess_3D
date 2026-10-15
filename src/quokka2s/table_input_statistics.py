"""Accumulate physical table-input ranges without querying emission tables."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from quokka2s.physics import settings
from quokka2s.physics.gas_fields import HYDROGEN_MASS_G
from quokka2s.snapshot_reader import CellBatch, Snapshot


@dataclass(frozen=True)
class SnapshotInputSettings:
    """Inputs needed to scan one snapshot, without emission-product settings.

    dataset is an absolute snapshot Path; names identify its saved record.
    tile_ny counts y layers per complete-z tile; query_chunk counts cells.
    Optional x/y ranges use native [start, stop) indices; max_tiles limits
    a diagnostic scan only. Each tile spans the selected x range.
    Single-snapshot and multi-job commands pass this same small record.
    """

    dataset: Path
    jobname: str
    snapshot_name: str
    tile_ny: int = 8
    query_chunk: int = 1_000_000
    max_tiles: int | None = None
    x_index_range: tuple[int, int] | None = None
    y_index_range: tuple[int, int] | None = None


@dataclass
class FieldRange:
    """Running extrema and invalid counts for one physical input coordinate.

    Only positive finite values enter extrema, because these inputs have
    logarithmic table axes. Invalid values are counted explicitly, not hidden.
    No cell arrays are retained after record() returns.
    """

    unit: str
    cell_count: int = 0
    invalid_cell_count: int = 0
    minimum: float = math.inf
    maximum: float = -math.inf

    def record(self, *, values: np.ndarray) -> None:
        """Update from a (B,) array in self.unit; retain only scalar statistics."""
        usable = np.isfinite(values) & (values > 0.0)
        self.cell_count += values.size
        self.invalid_cell_count += int(values.size - np.count_nonzero(usable))
        if np.any(usable):
            self.minimum = min(self.minimum, float(values[usable].min()))
            self.maximum = max(self.maximum, float(values[usable].max()))

    def as_dict(self) -> dict:
        """Return JSON-ready scalars; empty/invalid-only inputs have null extrema."""
        has_extrema = self.cell_count > self.invalid_cell_count
        return {
            "unit": self.unit,
            "cell_count": self.cell_count,
            "invalid_cell_count": self.invalid_cell_count,
            "minimum": self.minimum if has_extrema else None,
            "maximum": self.maximum if has_extrema else None,
        }

    def merge_saved_range(self, *, measured: dict) -> None:
        """Combine another completed snapshot's scalar range in the same unit."""
        if measured["unit"] != self.unit:
            raise ValueError("Cannot combine table-input ranges with different units")
        self.cell_count += measured["cell_count"]
        self.invalid_cell_count += measured["invalid_cell_count"]
        if measured["minimum"] is not None:
            self.minimum = min(self.minimum, measured["minimum"])
            self.maximum = max(self.maximum, measured["maximum"])


class TableInputStatistics:
    """Record DESPOTIC all-cell inputs and Cloudy hot-cell inputs separately.

    DESPOTIC coordinates are nH/NH/dVdr for every cell, including hot CO cells.
    Cloudy coordinates are nH/NH/T_QUOKKA where T_QUOKKA >= 3000 K. NH stays
    physical here; lookup clipping is reported later during table comparison.
    Example: record_batch(cells=cells) updates six named ranges and counts.
    """

    def __init__(self):
        self.despotic = {
            "nH": FieldRange(unit="cm^-3"),
            "NH": FieldRange(unit="cm^-2"),
            "dVdr": FieldRange(unit="s^-1"),
        }
        self.cloudy = {
            "nH": FieldRange(unit="cm^-3"),
            "NH": FieldRange(unit="cm^-2"),
            "T_QUOKKA": FieldRange(unit="K"),
        }
        self.cell_count = 0
        self.hot_cell_count = 0
        self.invalid_quokka_temperature_count = 0

    def record_batch(self, *, cells: CellBatch) -> None:
        """Record unchanged batch coordinates, then select the QUOKKA hot branch.

        cells comes from slab.iter_batches(). Its fields have shape (B,);
        hydrogen_density_cm3 uses the pipeline's cached rho-to-nH conversion.
        Physical zeros/invalid inputs are counted rather than dropped silently.
        """
        hydrogen_density = cells.hydrogen_density_cm3
        temperature = cells.temperature_QUOKKA_K
        hot_cells = temperature >= settings.EMISSION_TEMPERATURE_BOUNDARY_K

        self.despotic["nH"].record(values=hydrogen_density)
        self.despotic["NH"].record(values=cells.shielding_NH_cm2)
        self.despotic["dVdr"].record(values=cells.velocity_gradient_s)
        self.cloudy["nH"].record(values=hydrogen_density[hot_cells])
        self.cloudy["NH"].record(values=cells.shielding_NH_cm2[hot_cells])
        self.cloudy["T_QUOKKA"].record(values=temperature[hot_cells])

        self.cell_count += temperature.size
        self.hot_cell_count += int(np.count_nonzero(hot_cells))
        invalid_temperature = ~np.isfinite(temperature) | (temperature <= 0.0)
        self.invalid_quokka_temperature_count += int(np.count_nonzero(invalid_temperature))

    def build_ranges(self) -> dict:
        """Return named ranges and branch counts; no table-dependent results."""
        return {
            "cell_count": self.cell_count,
            "hot_cell_count": self.hot_cell_count,
            "invalid_quokka_temperature_count": self.invalid_quokka_temperature_count,
            "despotic": {name: field.as_dict() for name, field in self.despotic.items()},
            "cloudy": {name: field.as_dict() for name, field in self.cloudy.items()},
        }

    def merge_completed_snapshot(self, *, statistics: dict) -> None:
        """Aggregate a full snapshot's saved ranges, without retaining its record.

        statistics comes from a completed scan or its validated checkpoint.
        Partial/region scans are rejected so aggregate ranges describe whole
        snapshots. Raw per-snapshot invalid counts remain visible in totals.
        """
        if statistics["scan_status"] != "complete" or not statistics["full_snapshot"]:
            raise ValueError("Only completed full snapshots enter job/global extrema")
        self.cell_count += statistics["cell_count"]
        self.hot_cell_count += statistics["hot_cell_count"]
        self.invalid_quokka_temperature_count += statistics["invalid_quokka_temperature_count"]
        for name, measured in statistics["despotic"].items():
            self.despotic[name].merge_saved_range(measured=measured)
        for name, measured in statistics["cloudy"].items():
            self.cloudy[name].merge_saved_range(measured=measured)


def input_definitions() -> dict:
    """Record the physical choices that determine reusable coordinate statistics.

    Change definition_version when a numerical field algorithm changes without
    changing its settings. These plain values prevent reuse under another rule.
    """
    return {
        "definition_version": 1,
        "X_H": float(settings.X_H),
        "hydrogen_mass_g": float(HYDROGEN_MASS_G),
        "hot_temperature_boundary_K": settings.EMISSION_TEMPERATURE_BOUNDARY_K,
        "shielding_column_mean": settings.COLUMN_DENSITY_MEAN,
        "shielding_column_directions": settings.COLUMN_DENSITY_DIRECTIONS,
        "velocity_gradient": "abs(div(v))/3; periodic x/y central differences; one-sided z edges",
        "velocity_gradient_floor_s": settings.SIMULATION_DVDR_MIN_S,
    }


def scan_snapshot_input_ranges(
    *,
    snapshot: Snapshot,
    tile_ny: int,
    query_chunk: int,
    max_tiles: int | None = None,
    progress_label: str = "",
) -> dict:
    """Read complete-z y strips and retain scalar ranges, not full field cubes.

    snapshot is the pipeline Snapshot from open_snapshot(), including any x/y
    selection. tile_ny counts y layers; each tile keeps the selected x range.
    This follows the file's x-fastest order, avoiding full-block reads for
    repeated narrow x slices. query_chunk counts cells per in-memory batch.
    max_tiles limits a diagnostic run only. Returns plain saved statistics;
    partial scans are explicitly labelled and cannot certify full coverage.
    """
    statistics = TableInputStatistics()
    prefix = f"[{progress_label}] " if progress_label else ""
    x_start, x_stop = snapshot.xy_region["x"]
    y_start, y_stop = snapshot.xy_region["y"]
    tile_count = math.ceil((y_stop - y_start) / tile_ny)
    if max_tiles is not None:
        tile_count = min(tile_count, max_tiles)
    for tile_index, start in enumerate(range(y_start, y_stop, tile_ny), start=1):
        if tile_index > tile_count:
            break
        stop = min(start + tile_ny, y_stop)
        print(
            f"{prefix}Reading table inputs: tile {tile_index}/{tile_count}, "
            f"x=[{x_start}, {x_stop}), y=[{start}, {stop}), full z",
            flush=True,
        )
        tile = snapshot.read_column_tile(
            x_start=x_start,
            x_stop=x_stop,
            y_start=start,
            y_stop=stop,
        )
        for cells in tile.iter_batches(batch_size=query_chunk):
            statistics.record_batch(cells=cells)
        del cells, tile
        percentage = 100.0 * tile_index / tile_count
        print(f"{prefix}Input statistics: {percentage:.1f}% ({statistics.cell_count:,} cells)", flush=True)

    selection_complete = statistics.cell_count == snapshot.processing_cell_count
    full_snapshot = selection_complete and snapshot.processing_shape == snapshot.shape
    ranges = statistics.build_ranges()
    return {
        "scan_status": "complete" if selection_complete else "partial",
        "full_snapshot": full_snapshot,
        "native_shape": list(snapshot.shape),
        "selected_shape": list(snapshot.processing_shape),
        "xy_region": {axis: list(bounds) for axis, bounds in snapshot.xy_region.items()},
        "expected_selected_cells": snapshot.processing_cell_count,
        "reading": {"tile_axis": "y", "tile_ny": tile_ny, "tile_count": tile_count},
        "time_s": float(snapshot.dataset.current_time.to_value("s")),
        "definitions": input_definitions(),
        **ranges,
    }
