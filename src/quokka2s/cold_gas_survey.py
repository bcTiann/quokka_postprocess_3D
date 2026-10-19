"""Prepare cold nH–NH distributions while keeping simulation inputs read-only."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import closing
from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np
import yaml

from quokka2s.cold_gas_reader import ColdGasColumnReader
from quokka2s.despotic.table_extension import extend_logarithmic_axis
from quokka2s.physics.gas_fields import HYDROGEN_MASS_G
from quokka2s.physics.settings import EMISSION_TEMPERATURE_BOUNDARY_K, X_H
from quokka2s.products.cold_gas_phase import ColdGasPhaseAccumulator
from quokka2s.snapshot_headers import read_snapshot_ranges
from quokka2s.snapshot_reader import Snapshot
from quokka2s.table_input_files import SurveyOutputDirectory
from quokka2s.table_input_jobs import discover_job_snapshots


@dataclass(frozen=True)
class ColdPhaseGrid:
    """Shared log10 histogram edges and physical table-outline endpoints.

    nH uses cm^-3 and NH uses cm^-2. Bounds describe coordinate coverage,
    not successful chemistry or emissivity. No table values are queried.
    """

    edges_nH_log10: np.ndarray
    edges_NH_log10: np.ndarray
    original_table_bounds: dict
    extended_table_bounds: dict

    def accumulator(self, *, query_chunk):
        return ColdGasPhaseAccumulator(
            edges_nH_log10=self.edges_nH_log10,
            edges_NH_log10=self.edges_NH_log10,
            query_chunk=query_chunk,
        )


def read_cold_phase_grid(*, original_table: Path, extension_config: Path):
    """Read only old axis arrays and extend their original log spacing.

    The extension YAML supplies requested bounds. The actual outer nodes,
    including their outward rounding, match the table-extension calculation.
    A partially built raw table and its checkpoint directory are never opened.
    """
    with np.load(original_table, allow_pickle=False) as saved:
        old_axes = {
            "nH": saved["nH_values"].copy(),
            "NH": saved["col_density_values"].copy(),
        }
    requested = yaml.safe_load(extension_config.read_text())["bounds"]
    original = {}
    extended = {}
    for name, values in old_axes.items():
        bounds = requested.get(name, {})
        new_values = extend_logarithmic_axis(
            old_values=values,
            minimum=float(bounds.get("minimum", values[0])),
            maximum=float(bounds.get("maximum", values[-1])),
        )
        original[name] = [float(values[0]), float(values[-1])]
        extended[name] = [float(new_values[0]), float(new_values[-1])]
    return ColdPhaseGrid(
        edges_nH_log10=np.arange(-8.0, 8.0, 0.15),
        # Extra upper bins cover conservative header bounds for denser jobs.
        # The original 10..28 edges and 0.15-dex spacing are preserved exactly.
        edges_NH_log10=np.arange(10.0, 30.00001, 0.15),
        original_table_bounds=original,
        extended_table_bounds=extended,
    )


def cold_phase_definitions():
    """Bind caches to the actual physics, independently of table overlays."""
    return {
        "schema": 1,
        "cold_temperature_rule": "T_QUOKKA < boundary_K",
        "boundary_K": EMISSION_TEMPERATURE_BOUNDARY_K,
        "X_H": float(X_H),
        "hydrogen_mass_g": float(HYDROGEN_MASS_G),
        "NH": "Inclusive full +/-z cellwise sums; 2/(1/Nplus+1/Nminus)",
        "region": "Complete original XYZ grid",
        "reader": "ColdGasColumnReader; density/temperature only",
    }


def source_file_state(*, dataset: Path):
    """Stat all original files, including particle metadata; never hash payloads."""
    return [
        {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(dataset.rglob("*"))
        if path.is_file()
    ]


def process_read_counters():
    """Linux logical/storage I/O counters, or None on systems without /proc."""
    path = Path("/proc/self/io")
    if not path.exists():
        return None
    counters = {}
    for line in path.read_text().splitlines():
        name, value = line.split(":", 1)
        counters[name] = int(value)
    return counters


def protect_simulation_inputs(*, source_root: Path):
    """Reject write opens and metadata mutations under the original source tree."""
    root = source_root.resolve()

    def guard(event, arguments):
        if event == "open" and not isinstance(arguments[0], int):
            path = Path(os.fsdecode(arguments[0])).resolve()
            flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            if path.is_relative_to(root) and arguments[2] & flags:
                raise PermissionError("Simulation inputs are read-only")
        mutations = (
            "os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.chmod",
            "os.chown", "os.utime", "os.truncate", "os.link", "os.symlink",
        )
        if event in mutations:
            for value in arguments[:2]:
                if isinstance(value, (str, bytes, os.PathLike)):
                    if Path(os.fsdecode(value)).resolve().is_relative_to(root):
                        raise PermissionError("Simulation metadata is read-only")

    sys.addaudithook(guard)


class ColdPhaseFiles(SurveyOutputDirectory):
    """Own one snapshot's atomic numerical output and completion record."""

    def save_arrays(self, *, arrays):
        path = self.directory / "prepared.npz"
        self.check_output_path(path=path)
        self.directory.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".cold-phase-", dir=self.directory)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(stream, **arrays)
            self.check_output_path(path=path)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def read_completed(self, *, source, source_state, grid):
        """Reuse only identical complete inputs, rules and bins; never mix stale data."""
        summary_path = self.directory / "summary.json"
        self.check_output_path(path=summary_path)
        if not summary_path.exists():
            return None
        saved = json.loads(summary_path.read_text())
        if saved["status"] != "completed":
            raise ValueError("Existing phase result is incomplete; choose another output directory")
        if (
            saved["source"] != str(source)
            or saved["source_file_state"] != source_state
            or saved["definitions"] != cold_phase_definitions()
        ):
            raise ValueError("Existing phase result belongs to different inputs or physics")
        prepared_path = self.directory / "prepared.npz"
        self.check_output_path(path=prepared_path)
        with np.load(prepared_path, allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        for axis in ("nH", "NH"):
            key = "edges_" + axis + "_log10"
            if not np.array_equal(arrays[key], getattr(grid, key)):
                raise ValueError("Existing phase result uses different histogram bins")
        return saved, arrays


def initialize_scan_worker(source_root):
    """Protect sources and disable yt caches; inherit the one-thread environment."""
    protect_simulation_inputs(source_root=Path(source_root))
    from yt.config import ytcfg

    ytcfg["yt", "store_parameter_files"] = False
    ytcfg["yt", "serialize"] = False


def scan_cold_gas_snapshot(
    *, selection, source_root, output_root, grid, expected_source_state=None,
):
    """Read one complete snapshot into small cold histograms, then release it.

    selection comes from discover_job_snapshots(). A worker holds at most one
    native z layer and its boundary carries. Outputs contain raw histograms,
    prepared display arrays, source identity, timing and population counts.
    """
    started = time.monotonic()
    io_before = process_read_counters()
    source = selection.dataset
    before = source_file_state(dataset=source)
    if expected_source_state is not None and before != expected_source_state:
        raise ValueError("Simulation changed after the batch inventory was frozen")
    directory = output_root / "snapshots" / selection.jobname / selection.snapshot_name
    files = ColdPhaseFiles(
        directory=directory,
        protected_roots=(source_root,),
        allowed_output_root=output_root,
    )
    cached = files.read_completed(source=source, source_state=before, grid=grid)
    if cached is not None:
        record, arrays = cached
        accumulator = grid.accumulator(query_chunk=selection.query_chunk)
        accumulator.add_snapshot(saved_arrays=arrays)
        refreshed = accumulator.build_outputs(
            title=f"{selection.jobname} / {selection.snapshot_name}",
            original_table_bounds=grid.original_table_bounds,
            extended_table_bounds=grid.extended_table_bounds,
        )
        if source_file_state(dataset=source) != before:
            raise ValueError("Simulation files changed while checking a cached result")
        files.save_arrays(arrays=refreshed)
        record["cache_reused"] = True
        record["cache_validation_seconds"] = time.monotonic() - started
        files.write_json(path=directory / "summary.json", values=record)
        return record

    # 1. Fresh text headers can prove there are no cells in the cold branch.
    header = read_snapshot_ranges(snapshot=source, run_name=selection.jobname)
    accumulator = grid.accumulator(query_chunk=selection.query_chunk)
    layers = 0
    native_cells = int(np.prod(header["shape"]))
    no_cold_cells = header["temperature_QUOKKA_K"]["minimum"] >= EMISSION_TEMPERATURE_BOUNDARY_K
    if no_cold_cells:
        accumulator.record_header_proven_no_cold_cells(cell_count=native_cells)
        strategy = "fresh_header_proves_no_cold_cells"
    else:
        # 2. Read rho/T only; retain the exact complete-z NH calculation.
        import yt

        snapshot = Snapshot(dataset=yt.load(str(source)))
        reader = ColdGasColumnReader(snapshot=snapshot)
        expected_layers = len(reader.native.blocks)
        with closing(reader.iter_layers()) as source_layers:
            for layer in source_layers:
                accumulator.add_layer(
                    layer=layer,
                    cell_volume_cm3=snapshot.cell_volume_cm3,
                )
                layers += 1
                files.write_json(
                    path=directory / "progress.json",
                    values={
                        "layers_completed": layers,
                        "expected_layers": expected_layers,
                        "cells_completed": accumulator.all_cell_count,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
        if layers != expected_layers or accumulator.all_cell_count != native_cells:
            raise ValueError("Cold-gas scan did not cover the complete original grid")
        strategy = "native_density_temperature_columns"

    # 3. Prepare numerical plot arrays, verify source identity, then mark complete.
    after = source_file_state(dataset=source)
    if before != after:
        raise ValueError("Simulation files changed during the scan; result not saved")
    arrays = accumulator.build_outputs(
        title=f"{selection.jobname} / {selection.snapshot_name}",
        original_table_bounds=grid.original_table_bounds,
        extended_table_bounds=grid.extended_table_bounds,
    )
    io_after = process_read_counters()
    record = {
        **accumulator.summary(),
        "status": "completed",
        "jobname": selection.jobname,
        "snapshot_name": selection.snapshot_name,
        "source": str(source),
        "source_file_state": before,
        "source_file_state_after": after,
        "definitions": cold_phase_definitions(),
        "native_shape": header["shape"],
        "time_s": header["time_s"],
        "reading_strategy": strategy,
        "payload_cells_scanned": 0 if no_cold_cells else native_cells,
        "native_layers_scanned": layers,
        "elapsed_seconds": time.monotonic() - started,
        "worker_pid": os.getpid(),
        "process_io_delta": None if io_before is None else {
            name: io_after[name] - io_before[name] for name in io_before
        },
        # Persistent workers reuse a PID: this high-water mark includes prior snapshots.
        "worker_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "thread_variables": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
        },
        "cache_reused": False,
    }
    files.save_arrays(arrays=arrays)
    files.write_json(path=directory / "summary.json", values=record)
    return record


def build_job_distributions(*, settings, jobs, grid):
    """Sum saved snapshot populations in inventory order, with no time weighting."""
    records = []
    for jobname, selections in jobs.items():
        accumulator = grid.accumulator(query_chunk=settings.query_chunk)
        for selection in selections:
            path = settings.output_dir / "snapshots" / jobname / selection.snapshot_name / "prepared.npz"
            with np.load(path, allow_pickle=False) as saved:
                accumulator.add_snapshot(saved_arrays=saved)
        files = ColdPhaseFiles(
            directory=settings.output_dir / "jobs" / jobname,
            protected_roots=(settings.source_root,),
            allowed_output_root=settings.output_dir,
        )
        arrays = accumulator.build_outputs(
            title=f"{jobname}: all {len(selections)} snapshots",
            original_table_bounds=grid.original_table_bounds,
            extended_table_bounds=grid.extended_table_bounds,
        )
        files.save_arrays(arrays=arrays)
        record = {"jobname": jobname, **accumulator.summary()}
        files.write_json(path=files.directory / "summary.json", values=record)
        records.append(record)
    return records


def run_cold_gas_survey(*, settings, grid):
    """Dispatch at most snapshot_workers readers and save progress after each result.

    Workers own datasets; the coordinator owns summaries. A failure stops new
    dispatch, lets an already running snapshot finish, and never auto-retries.
    The final job sums follow numeric snapshot order, regardless of finish order.
    """
    if settings.max_columns is not None:
        raise ValueError("Cold-gas survey requires complete snapshots, not max_columns")
    if settings.max_new_snapshots is not None or settings.max_runtime_minutes is not None:
        raise ValueError("Cold-gas survey runs the complete selected inventory; invocation limits are unsupported")
    protect_simulation_inputs(source_root=settings.source_root)
    jobs = discover_job_snapshots(settings=settings)
    if any(not selections for selections in jobs.values()):
        raise ValueError("Every selected job must contain at least one plt snapshot")
    selections = [selection for group in jobs.values() for selection in group]
    outputs = SurveyOutputDirectory(
        directory=settings.output_dir,
        protected_roots=(settings.source_root,),
    )
    records = {}
    frozen_sources = {
        str(selection.dataset): source_file_state(dataset=selection.dataset)
        for selection in selections
    }
    outputs.write_json(
        path=settings.output_dir / "inventory.json",
        values={"sources": frozen_sources, "snapshot_count": len(selections)},
    )
    next_selection = iter(selections)
    pending = {}
    started = time.monotonic()
    print(f"Cold gas survey: {len(selections)} snapshots; {settings.snapshot_workers} workers", flush=True)
    print(f"Source: {settings.source_root}", flush=True)
    print(f"Output: {settings.output_dir}", flush=True)
    context = multiprocessing.get_context("spawn")
    # Spawned interpreters see these before importing NumPy or yt.
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    with ProcessPoolExecutor(
        max_workers=settings.snapshot_workers,
        mp_context=context,
        initializer=initialize_scan_worker,
        initargs=(str(settings.source_root),),
    ) as executor:
        while True:
            while len(pending) < settings.snapshot_workers:
                selection = next(next_selection, None)
                if selection is None:
                    break
                future = executor.submit(
                    scan_cold_gas_snapshot,
                    selection=selection,
                    source_root=settings.source_root,
                    output_root=settings.output_dir,
                    grid=grid,
                    expected_source_state=frozen_sources[str(selection.dataset)],
                )
                pending[future] = selection
            if not pending:
                break
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                selection = pending.pop(future)
                record = future.result()
                key = (selection.jobname, selection.snapshot_name)
                records[key] = record
                print(f"[{len(records)}/{len(selections)}] {key[0]}/{key[1]}: {record['elapsed_seconds']:.1f}s", flush=True)
            outputs.write_json(
                path=settings.output_dir / "progress.json",
                values={
                    "status": "running",
                    "completed_snapshots": len(records),
                    "expected_snapshots": len(selections),
                    "active": [f"{item.jobname}/{item.snapshot_name}" for item in pending.values()],
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
    ordered = [records[(item.jobname, item.snapshot_name)] for item in selections]
    summary = {
        "status": "completed",
        "snapshot_count": len(ordered),
        "snapshot_workers": settings.snapshot_workers,
        "elapsed_seconds": time.monotonic() - started,
        "jobs": build_job_distributions(settings=settings, jobs=jobs, grid=grid),
        "snapshots": ordered,
    }
    outputs.write_json(path=settings.output_dir / "summary.json", values=summary)
    outputs.write_json(path=settings.output_dir / "progress.json", values={
        "status": "completed", "completed_snapshots": len(ordered),
        "expected_snapshots": len(selections), "active": [],
    })
    return summary
