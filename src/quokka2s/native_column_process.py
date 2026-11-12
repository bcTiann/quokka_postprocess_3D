"""Process complete-z native tiles and merge each finished wave immediately.

Tiles follow sorted native XY bounds. Batches retain their original cells and
within-tile order; no finished products wait for another tile to be processed.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import resource
import sys
from time import monotonic

import numpy as np

from .native_column_fields import derive_native_column_tile
from .physics.slab_derive import estimated_working_bytes
from .products.gaussian.template_batches import process_template_batches
from .result_files import write_status


PROCESS_LIMIT_BYTES = 20 * 1024**3


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
    """Check every original query batch is covered once, in any tile order."""
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
    bounds = tuple(tuple(tile) for tile in reader.iter_tile_bounds())
    if not bounds:
        raise ValueError("Native column plan contains no tiles")
    for tile in bounds:
        if len(tile) != 4 or any(not isinstance(value, (int, np.integer)) for value in tile):
            raise ValueError("Native XY tile bounds must contain four integer indices")
        x_start, x_stop, y_start, y_stop = tile
        if not (0 <= x_start < x_stop <= nx and 0 <= y_start < y_stop <= ny):
            raise ValueError("Native XY tile lies outside the complete snapshot")
        if ((y_stop - y_start) * nz) % config.query_chunk:
            raise ValueError("Native XY tile width cuts an original query_chunk boundary")
        if (y_start * nz) % config.query_chunk:
            raise ValueError("Native XY tile origin cuts an original query_chunk boundary")
    bounds = tuple(sorted(bounds))
    batches = set()
    for x_start, x_stop, y_start, y_stop in bounds:
        for x_index in range(x_start, x_stop):
            for y_offset in range(y_start * nz, y_stop * nz, config.query_chunk):
                first_cell = x_index * ny * nz + y_offset
                if first_cell in batches:
                    raise ValueError("Native column plan repeats an original query batch")
                batches.add(first_cell)
    expected_batches = snapshot.cell_count // config.query_chunk
    # Bounds and alignment place each ID inside the finite valid batch set.
    # Its expected size plus uniqueness therefore establish complete coverage.
    if len(batches) != expected_batches:
        raise ValueError("Native columns do not cover every original query batch exactly once")
    return bounds, expected_batches


def _memory_plan(config, snapshot, reader, bounds):
    """Reserve arrays, worker space and headroom; this is not measured RSS.

    Finished partials survive only within the existing bounded worker wave.
    Runtime high-water checks and the external process guard remain necessary.
    """
    nx, ny, nz = snapshot.shape
    native = snapshot.native_reader
    maximum_cells = max((x_stop - x_start) * (y_stop - y_start) * nz for x_start, x_stop, y_start, y_stop in bounds)
    maximum_faces = max(4 * ((x_stop - x_start) + (y_stop - y_start)) * nz * 8 for x_start, x_stop, y_start, y_stop in bounds)
    x_planes = {position % nx for x_start, x_stop, _, _ in bounds for position in (x_start - 1, x_stop)}
    y_planes = {position % ny for _, _, y_start, y_stop in bounds for position in (y_start - 1, y_stop)}
    face_cache = 2 * nz * 8 * (len(x_planes) * ny + len(y_planes) * nx)
    largest_native_field = max(block.field_bytes for block in native.blocks)
    read_working = config.z_read_workers * largest_native_field
    derived_working = estimated_working_bytes(
        column_count=maximum_cells // nz,
        column_depth=nz,
        threads=config.z_reader_threads,
    )
    worker_reserve = config.chunk_workers * config.query_chunk * (192 * 8 + 32)
    baseline = _check_process_memory("native column preflight")
    raw_peak = 5 * maximum_cells * 8 + read_working
    derive_peak = 6 * maximum_cells * 8 + derived_working
    compute_peak = 6 * maximum_cells * 8 + worker_reserve
    estimated = baseline + max(raw_peak, derive_peak, compute_peak) + maximum_faces + face_cache + 1024**3
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
        "derivation_working_memory_scope": "Conservative bounded NumPy temporaries; not measured RSS",
        "physical_worker_reserve_bytes": worker_reserve,
        "additional_library_allocator_headroom_bytes": 1024**3,
        "conservative_estimated_process_bytes": estimated,
        "note": "No cross-tile partial queue; runtime process high-water checks cover every allocator",
    }


def _process_one_tile(*, config, snapshot, reader, bounds, executor, emission_calculator, products, accumulate_batch):
    """Own all tile arrays until its batches finish and merge, then drop them."""
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
    merged_before = products.counts["all"]
    wave_count = 0

    def after_wave():
        nonlocal wave_count
        _check_process_memory("native worker wave")
        wave_count += 1

    computation_started = monotonic()
    process_template_batches(
        config=config,
        snapshot=snapshot,
        emission_calculator=emission_calculator,
        products=products,
        slab=slab,
        executor=executor,
        accumulate_batch=accumulate_batch,
        after_wave=after_wave,
    )
    computation_seconds = monotonic() - computation_started
    cell_count = slab.cell_count
    if products.counts["all"] - merged_before != cell_count:
        raise AssertionError("Native tile did not merge every original cell exactly once")
    batch_count = cell_count // config.query_chunk
    expected_waves = (batch_count + config.chunk_workers - 1) // config.chunk_workers
    if wave_count != expected_waves:
        raise AssertionError("Native tile memory guard did not run after every worker wave")
    del slab
    # No cells, raw fields, futures or unmerged products escape this function.
    return {
        "bounds": list(bounds),
        "cell_count": cell_count,
        "merged_batches": batch_count,
        "worker_waves": wave_count,
        "read_seconds": read_finished - started,
        "derivation": derivation,
        "compute_and_merge_seconds": computation_seconds,
        "total_seconds": monotonic() - started,
        "process_peak_rss_bytes": _check_process_memory("native tile completion"),
    }


def process_native_columns(*, config, snapshot, reader, emission_calculator, products, accumulate_batch, began, status_began=None):
    """Read one native tile at a time; immediately merge its finished batches."""
    bounds, expected_batches = _validate_plan(config, snapshot, reader)
    memory = _memory_plan(config, snapshot, reader, bounds)
    if products.counts["all"] != 0:
        raise ValueError("Native column processing requires fresh product accumulators")
    records = []
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
            )
            records.append(record)
            processed_fraction = products.counts["all"] / snapshot.cell_count
            elapsed = monotonic() - began
            write_status(
                output_dir=config.output_dir,
                counts=products.counts,
                began=began if status_began is None else status_began,
                state="running",
                completed_tiles=number,
                total_tiles=len(bounds),
                progress_percent=round(100 * processed_fraction, 1),
                eta_seconds=round(elapsed * (1.0 - processed_fraction) / processed_fraction),
            )
    merged_batches = sum(record["merged_batches"] for record in records)
    if products.counts["all"] != snapshot.cell_count or merged_batches != expected_batches:
        raise AssertionError("Native column processing did not merge the complete snapshot")
    read_seconds = sum(record["read_seconds"] for record in records)
    derive_seconds = sum(record["derivation"]["fused_derivation_seconds"] for record in records)
    gradient_seconds = sum(record["derivation"]["gradient_seconds"] for record in records)
    columns_units_seconds = sum(record["derivation"]["units_and_columns_seconds"] for record in records)
    other_derivation_seconds = max(0.0, derive_seconds - gradient_seconds - columns_units_seconds)
    gaussian_timings = products.spectra.timing_summary()
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
        "compute_and_ordered_merge_total_seconds": sum(record["compute_and_merge_seconds"] for record in records),
        "gaussian": gaussian_timings,
        "native_columns": {
            "tiles": records,
            "read_statistics": reader.stats,
            "finished_products": {
                "merged_cells": products.counts["all"],
                "merged_batches": merged_batches,
                "validated_query_batches": expected_batches,
                "worker_waves": sum(record["worker_waves"] for record in records),
                "actual_product_merge_seconds": gaussian_timings["ordered_product_merge_wave_seconds"],
                "process_peak_rss_bytes": peak_process_rss_bytes(),
                "merge_order": "Sorted native XY tile bounds; original batch order within each tile",
                "cross_tile_partial_queue": False,
            },
            "memory_plan": memory,
            "source_layout": "Original AMReX FAB fields: x varies fastest within each block",
            "ram_layout": "One full-z XY tile: float64 C-order XYZ, z varies fastest",
            "disk_copy_created": False,
        },
    }
