"""Process original full-z FAB columns while retaining global batch merge order.

One tile owns its raw/derived arrays. Finished worker products retain only small
numerical arrays until their original global batch position can be merged.
For a 512x512x4096 grid with 256x256 XY tiles and 131072-cell batches, at most
2040 finished partials wait. No unfinished Gaussian inputs survive a tile.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, is_dataclass
from math import prod
import resource
import sys
from time import monotonic, perf_counter_ns

import numpy as np

from .native_column_fields import derive_native_column_tile
from .products.gaussian.template_batches import process_template_batches
from .result_files import write_status


PROCESS_LIMIT_BYTES = 20 * 1024**3
FINISHED_PARTIAL_BUDGET_BYTES = 512 * 1024


def peak_process_rss_bytes():
    """Return process high-water resident bytes, including every allocator."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _check_process_memory(stage):
    peak = peak_process_rss_bytes()
    if peak > PROCESS_LIMIT_BYTES:
        raise MemoryError(
            f"native-columns exceeded the 20 GiB process RSS guard during {stage}: "
            f"{peak / 1024**3:.3f} GiB"
        )
    return peak


def _validate_plan(config, snapshot, reader):
    """Reject geometry that cuts the frozen original query-batch boundaries."""
    nx, ny, nz = snapshot.shape
    if config.spectral_method != "direct-template" or config.max_slabs is not None:
        raise ValueError("native-columns requires direct-template and max_slabs: null")
    if config.prepared_z_fields is not None:
        raise ValueError("native-columns cannot use prepared_z_fields")
    if snapshot.xy_region != {"x": (0, nx), "y": (0, ny)}:
        raise ValueError("native-columns currently requires the complete original box")
    if reader.native_shape != tuple(snapshot.shape):
        raise ValueError("Native column geometry differs from the original snapshot")
    if config.query_chunk % nz or (ny * nz) % config.query_chunk:
        raise ValueError("native-columns query_chunk must contain full z columns and divide Ny*Nz")
    bounds = tuple(reader.iter_tile_bounds())
    if not bounds:
        raise ValueError("Native column plan contains no tiles")
    waiting = set()
    next_cell = 0
    maximum_pending = 0
    for x_start, x_stop, y_start, y_stop in bounds:
        if ((y_stop - y_start) * nz) % config.query_chunk:
            raise ValueError("Native XY tile width cuts an original query_chunk boundary")
        if (y_start * nz) % config.query_chunk:
            raise ValueError("Native XY tile origin cuts an original query_chunk boundary")
        for x_index in range(x_start, x_stop):
            for y_offset in range(y_start * nz, y_stop * nz, config.query_chunk):
                first_cell = x_index * ny * nz + y_offset
                if first_cell < next_cell or first_cell in waiting:
                    raise ValueError("Native column plan repeats an original query batch")
                waiting.add(first_cell)
                while next_cell in waiting:
                    waiting.remove(next_cell)
                    next_cell += config.query_chunk
                maximum_pending = max(maximum_pending, len(waiting))
    if waiting or next_cell != nx * ny * nz:
        raise ValueError("Native columns do not cover every original query batch exactly once")
    return bounds, maximum_pending


def _memory_plan(config, snapshot, reader, bounds, maximum_pending):
    """Preflight all-process headroom; array counts are not an RSS measurement.

    Reserve 192 float64 vectors plus 32 mask bytes per worker cell, 512 KiB per
    finished partial and 1 GiB for Python/library/transient allocation headroom.
    Runtime high-water checks and the external process guard remain necessary.
    """
    nx, ny, nz = snapshot.shape
    native = snapshot.native_reader
    maximum_cells = max((x_stop - x_start) * (y_stop - y_start) * nz for x_start, x_stop, y_start, y_stop in bounds)
    maximum_faces = max(4 * ((x_stop - x_start) + (y_stop - y_start)) * nz * 8 for x_start, x_stop, y_start, y_stop in bounds)
    x_planes = {(position % nx) for x_start, x_stop, _, _ in bounds for position in (x_start - 1, x_stop)}
    y_planes = {(position % ny) for _, _, y_start, y_stop in bounds for position in (y_start - 1, y_stop)}
    face_cache = 2 * nz * 8 * (len(x_planes) * ny + len(y_planes) * nx)
    largest_native_field = max(block.field_bytes for block in native.blocks)
    read_working = config.z_read_workers * largest_native_field
    derived_working = config.z_reader_threads * 2 * nz * 8
    worker_reserve = config.chunk_workers * config.query_chunk * (192 * 8 + 32)
    pending_reserve = maximum_pending * FINISHED_PARTIAL_BUDGET_BYTES
    baseline = _check_process_memory("native column preflight")
    raw_peak = 5 * maximum_cells * 8 + read_working
    derive_peak = 6 * maximum_cells * 8 + derived_working
    compute_peak = 6 * maximum_cells * 8 + worker_reserve
    estimated = baseline + max(raw_peak, derive_peak, compute_peak) + maximum_faces + face_cache + pending_reserve + 1024**3
    if estimated > PROCESS_LIMIT_BYTES:
        raise MemoryError(
            "native-columns conservative all-process memory plan exceeds 20 GiB: "
            f"{estimated / 1024**3:.3f} GiB; lower query_chunk/chunk_workers or use smaller native FABs"
        )
    return {
        "process_limit_bytes": PROCESS_LIMIT_BYTES,
        "baseline_process_peak_rss_bytes": baseline,
        "maximum_tile_cells": maximum_cells,
        "maximum_raw_five_field_bytes": maximum_cells * 5 * 8,
        "maximum_derived_six_field_bytes": maximum_cells * 6 * 8,
        "maximum_returned_face_bytes": maximum_faces,
        "immutable_face_cache_bytes": face_cache,
        "read_worker_payload_reserve_bytes": read_working,
        "derivation_scratch_reserve_bytes": derived_working,
        "physical_worker_reserve_bytes": worker_reserve,
        "finished_partial_reserve_bytes": pending_reserve,
        "additional_library_allocator_headroom_bytes": 1024**3,
        "conservative_estimated_process_bytes": estimated,
        "note": "Ownership/reserve accounting is not measured RSS; runtime process high-water checks cover every allocator",
    }


def _array_bytes(value, seen):
    if isinstance(value, np.ndarray):
        owner = value
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        if id(owner) in seen:
            return 0
        seen.add(id(owner))
        return owner.nbytes
    if isinstance(value, dict):
        return sum(_array_bytes(element, seen) for element in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_array_bytes(element, seen) for element in value)
    if is_dataclass(value):
        return sum(_array_bytes(getattr(value, field.name), seen) for field in fields(value))
    if hasattr(value, "__dict__"):
        return _array_bytes(vars(value), seen)
    return 0


def _finished_partial_bytes(partial):
    """Count only worker-owned arrays, excluding its shared immutable engine."""
    seen = set()
    spectrum_values = {name: value for name, value in vars(partial.spectra).items() if name != "engine"}
    return sum(_array_bytes(value, seen) for value in (
        vars(partial.images), spectrum_values, vars(partial.phases),
        vars(partial.cold_despotic_luminosity), partial.intrinsic_luminosity_erg_s,
        partial.attenuated_luminosity_erg_s,
    ))


class OrderedFinishedPartials:
    """Merge completed small products in the original global query-batch order."""

    def __init__(self, *, products, query_chunk, cell_count, maximum_pending):
        self.products = products
        self.query_chunk = query_chunk
        self.cell_count = cell_count
        self.maximum_pending = maximum_pending
        self.pending = {}
        self.next_cell = 0
        self.received_cells = 0
        self.received_batches = 0
        self.merged_batches = 0
        self.pending_array_bytes = 0
        self.maximum_pending_count = 0
        self.maximum_pending_array_bytes = 0
        self.maximum_finished_partial_array_bytes = 0
        self.merge_nanoseconds = 0
        self.peak_rss_bytes = 0

    def __call__(self, partial, first_cell_id):
        if partial.spectra._groups or partial.spectra._owned_input_bytes:
            raise AssertionError("A pending native-column partial retained Gaussian input arrays")
        if partial.counts["all"] != self.query_chunk:
            raise AssertionError("A native-column partial changed original query_chunk grouping")
        if first_cell_id % self.query_chunk or not 0 <= first_cell_id < self.cell_count:
            raise AssertionError("A finished native-column batch has an invalid original cell ID")
        if first_cell_id < self.next_cell or first_cell_id in self.pending:
            raise AssertionError("A finished native-column batch was supplied twice")
        array_bytes = _finished_partial_bytes(partial)
        if array_bytes > FINISHED_PARTIAL_BUDGET_BYTES:
            raise MemoryError("A finished partial exceeds the preflight small-product array budget")
        self.maximum_finished_partial_array_bytes = max(self.maximum_finished_partial_array_bytes, array_bytes)
        self.pending[first_cell_id] = (partial, array_bytes)
        self.pending_array_bytes += array_bytes
        self.received_cells += partial.counts["all"]
        self.received_batches += 1
        merged_any = False
        began = perf_counter_ns()
        try:
            while self.next_cell in self.pending:
                ready, ready_bytes = self.pending.pop(self.next_cell)
                self.pending_array_bytes -= ready_bytes
                self.products.merge(ready)
                self.next_cell += self.query_chunk
                self.merged_batches += 1
                merged_any = True
                del ready
        finally:
            merge_ns = perf_counter_ns() - began if merged_any else 0
            self.merge_nanoseconds += merge_ns
            self.products.spectra.record_finished_profile_merges(merge_ns=merge_ns)
        self.maximum_pending_count = max(self.maximum_pending_count, len(self.pending))
        self.maximum_pending_array_bytes = max(self.maximum_pending_array_bytes, self.pending_array_bytes)
        if len(self.pending) > self.maximum_pending:
            raise MemoryError("Native-column pending products exceeded the validated merge-order plan")
        self.peak_rss_bytes = max(self.peak_rss_bytes, _check_process_memory("finished native batch"))

    def finish(self):
        if self.pending or self.next_cell != self.cell_count:
            raise AssertionError("Native-column processing left missing original query batches")
        return {
            "received_batches": self.received_batches,
            "merged_batches": self.merged_batches,
            "received_cells": self.received_cells,
            "maximum_pending_partials": self.maximum_pending_count,
            "maximum_pending_partial_array_bytes": self.maximum_pending_array_bytes,
            "maximum_finished_partial_array_bytes": self.maximum_finished_partial_array_bytes,
            "planned_maximum_pending_partials": self.maximum_pending,
            "actual_ordered_product_merge_seconds": self.merge_nanoseconds * 1e-9,
            "process_peak_rss_bytes": self.peak_rss_bytes,
            "merge_order": "increasing original full-grid first_cell_id with unchanged query_chunk",
        }

    def release(self):
        """Release only completed partials; no raw tile/worker queues are retained."""
        for partial, _ in self.pending.values():
            partial.spectra.release_gaussian_inputs()
        self.pending.clear()
        self.pending_array_bytes = 0


def _process_one_tile(*, config, snapshot, reader, bounds, executor, emission_calculator, products, accumulate_batch, sink):
    """Own all tile arrays/views/futures until processing ends, then drop them."""
    started = monotonic()
    raw = reader.read_tile(bounds)
    read_finished = monotonic()
    _check_process_memory("native tile reading")
    slab, derivation = derive_native_column_tile(
        snapshot=snapshot,
        native_reader=snapshot.native_reader,
        raw_tile=raw,
        threads=config.z_reader_threads,
    )
    del raw
    _check_process_memory("native tile derivation")
    computation_started = monotonic()
    process_template_batches(
        config=config,
        snapshot=snapshot,
        emission_calculator=emission_calculator,
        products=products,
        slab=slab,
        executor=executor,
        accumulate_batch=accumulate_batch,
        finished_partial_sink=sink,
    )
    computation_seconds = monotonic() - computation_started
    cell_count = slab.cell_count
    del slab
    # Return scalars only. process_template_batches has returned, so its final
    # cells/futures/closures cannot keep this tile alive during the next read.
    return {
        "bounds": list(bounds),
        "cell_count": cell_count,
        "read_seconds": read_finished - started,
        "derivation": derivation,
        "compute_and_sink_seconds": computation_seconds,
        "total_seconds": monotonic() - started,
        "process_peak_rss_bytes": _check_process_memory("native tile completion"),
    }


def process_native_columns(*, config, snapshot, reader, emission_calculator, products, accumulate_batch, began, status_began=None):
    """Read each original tile once, derive in RAM and merge unchanged products."""
    bounds, maximum_pending = _validate_plan(config, snapshot, reader)
    memory = _memory_plan(config, snapshot, reader, bounds, maximum_pending)
    sink = OrderedFinishedPartials(
        products=products,
        query_chunk=config.query_chunk,
        cell_count=snapshot.cell_count,
        maximum_pending=maximum_pending,
    )
    records = []
    try:
        with ThreadPoolExecutor(max_workers=config.chunk_workers) as executor:
            for number, tile_bounds in enumerate(bounds, start=1):
                print(f"Native column tile {number}/{len(bounds)}: XY={tile_bounds}; full z; original plt -> RAM", flush=True)
                record = _process_one_tile(
                    config=config,
                    snapshot=snapshot,
                    reader=reader,
                    bounds=tile_bounds,
                    executor=executor,
                    emission_calculator=emission_calculator,
                    products=products,
                    accumulate_batch=accumulate_batch,
                    sink=sink,
                )
                records.append(record)
                processed_fraction = sink.received_cells / snapshot.cell_count
                elapsed = monotonic() - began
                write_status(
                    output_dir=config.output_dir,
                    counts=products.counts,
                    began=began if status_began is None else status_began,
                    state="running",
                    completed_tiles=number,
                    total_tiles=len(bounds),
                    calculated_cells=sink.received_cells,
                    merged_cells=products.counts["all"],
                    pending_finished_batches=len(sink.pending),
                    progress_percent=round(100 * processed_fraction, 1),
                    eta_seconds=round(elapsed * (1.0 - processed_fraction) / processed_fraction),
                )
        ordered = sink.finish()
    finally:
        sink.release()
    read_seconds = sum(record["read_seconds"] for record in records)
    derive_seconds = sum(record["derivation"]["fused_derivation_seconds"] for record in records)
    gradient_seconds = sum(record["derivation"]["gradient_seconds"] for record in records)
    columns_units_seconds = sum(record["derivation"]["units_and_columns_seconds"] for record in records)
    other_derivation_seconds = max(0.0, derive_seconds - gradient_seconds - columns_units_seconds)
    statistics = reader.stats
    return {
        "field_source": "native-columns",
        "read_stage_labels": {
            "original_fields_and_halos": "Original native fields, required halo rereads and RAM gather",
            "velocity_gradient": "Gradient from original momenta and true global neighbour faces",
            "units_and_columns": "In-place unit conversion plus foreground/shielding columns",
            "derivation_other": "Derivation validation, allocation and remaining orchestration",
        },
        "read_stage_totals_seconds": {
            "original_fields_and_halos": read_seconds,
            "velocity_gradient": gradient_seconds,
            "units_and_columns": columns_units_seconds,
            "derivation_other": other_derivation_seconds,
        },
        "columns_total_seconds": None,
        "columns_timing_note": "Foreground/shielding columns share the in-place unit-conversion pass; only their combined pass is timed",
        "read_derive_total_seconds": read_seconds + derive_seconds,
        "compute_and_ordered_merge_total_seconds": sum(record["compute_and_sink_seconds"] for record in records),
        "gaussian": products.spectra.timing_summary(),
        "native_columns": {
            "tiles": records,
            "read_statistics": statistics,
            "ordered_finished_products": ordered,
            "memory_plan": memory,
            "source_layout": "Original AMReX FAB fields: x varies fastest within each block",
            "ram_layout": "One full-z XY tile: float64 C-order XYZ, z varies fastest",
            "disk_copy_created": False,
        },
    }
