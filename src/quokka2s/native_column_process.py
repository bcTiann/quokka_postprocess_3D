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


# Decimal GB matches the whole-process memory budget.
PROCESS_LIMIT_BYTES = 30_000_000_000


def peak_process_rss_bytes():
    """Return process high-water resident bytes, including every allocator."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _check_process_memory(stage):
    peak = peak_process_rss_bytes()
    if peak > PROCESS_LIMIT_BYTES:
        raise MemoryError(
            f"native-columns exceeded the {PROCESS_LIMIT_BYTES / 1e9:g} GB "
            f"process RSS guard during {stage}: {peak / 1e9:.3f} GB"
        )
    return peak


def _validate_plan(config, snapshot, reader):
    """Check selected native cells are covered once, allowing shorter batches."""
    nx, ny, nz = snapshot.shape
    if reader.native_shape != tuple(snapshot.shape):
        raise ValueError("Native column geometry differs from the original snapshot")
    if not isinstance(config.query_chunk, (int, np.integer)) or config.query_chunk < 1:
        raise ValueError("query_chunk must be a positive integer cell count")
    selected_x_start, selected_x_stop = snapshot.xy_region["x"]
    selected_y_start, selected_y_stop = snapshot.xy_region["y"]
    bounds = tuple(tuple(tile) for tile in reader.iter_tile_bounds())
    if not bounds:
        raise ValueError("Native column plan contains no tiles")
    for tile in bounds:
        if len(tile) != 4 or any(not isinstance(value, (int, np.integer)) for value in tile):
            raise ValueError("Native XY tile bounds must contain four integer indices")
        x_start, x_stop, y_start, y_stop = tile
        if not (
            selected_x_start <= x_start < x_stop <= selected_x_stop
            and selected_y_start <= y_start < y_stop <= selected_y_stop
        ):
            raise ValueError("Native XY tile lies outside the selected region")
    bounds = tuple(sorted(bounds))
    selected_cells = 0
    expected_batches = 0
    for index, (x_start, x_stop, y_start, y_stop) in enumerate(bounds):
        for other in bounds[:index]:
            if (
                max(x_start, other[0]) < min(x_stop, other[1])
                and max(y_start, other[2]) < min(y_stop, other[3])
            ):
                raise ValueError("Native column tiles overlap within the selected region")
        tile_cells = (x_stop - x_start) * (y_stop - y_start) * nz
        selected_cells += tile_cells
        expected_batches += (tile_cells + config.query_chunk - 1) // config.query_chunk
    # Non-overlap and equal total volume establish exact selected-cell coverage.
    if selected_cells != snapshot.processing_cell_count:
        raise ValueError("Native columns do not cover every selected cell exactly once")
    return bounds, expected_batches


def _memory_plan(config, snapshot, bounds):
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
    selected_nx, selected_ny, _ = snapshot.processing_shape
    face_cache = 2 * nz * 8 * (len(x_planes) * selected_ny + len(y_planes) * selected_nx)
    largest_native_field = max(block.field_bytes for block in native.blocks)
    read_working = config.z_read_workers * largest_native_field
    derived_working = estimated_working_bytes(
        column_count=maximum_cells // nz,
        column_depth=nz,
        threads=config.z_reader_threads,
    )
    largest_batch = min(config.query_chunk, maximum_cells)
    active_workers = min(config.chunk_workers, (maximum_cells + config.query_chunk - 1) // config.query_chunk)
    worker_reserve = active_workers * largest_batch * (192 * 8 + 32)
    baseline = _check_process_memory("native column preflight")
    raw_peak = 5 * maximum_cells * 8 + read_working
    derive_peak = 6 * maximum_cells * 8 + derived_working
    compute_peak = 6 * maximum_cells * 8 + worker_reserve
    estimated = baseline + max(raw_peak, derive_peak, compute_peak) + maximum_faces + face_cache + 1024**3
    if estimated > PROCESS_LIMIT_BYTES:
        raise MemoryError(
            "native-columns conservative all-process memory plan exceeds "
            f"{PROCESS_LIMIT_BYTES / 1e9:g} GB: {estimated / 1e9:.3f} GB; "
            "lower query_chunk/chunk_workers or use smaller native FABs"
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


def _process_one_tile(
    *, config, snapshot, reader, bounds, executor, emission_calculator, products,
    accumulate_batch, tile_number, tile_count, batches_before, total_batches,
    process_began,
):
    """Own all tile arrays until its batches finish and merge, then drop them."""
    prefix = f"Tile {tile_number}/{tile_count}"
    tile_cells = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]) * snapshot.shape[2]
    tile_batches = (tile_cells + config.query_chunk - 1) // config.query_chunk
    print(
        f"{prefix}: reading original fields and global neighbour faces "
        f"for {tile_cells:,} cells (up to {config.z_read_workers} readers)...",
        flush=True,
    )
    started = monotonic()
    raw = reader.read_tile(bounds)
    read_finished = monotonic()
    peak = _check_process_memory("native tile reading")
    print(
        f"{prefix}: field reading and RAM copies finished in {read_finished - started:.2f} s; "
        f"peak process RSS {peak / 1e9:.2f} GB",
        flush=True,
    )

    def report_derivation(stage):
        if stage == "gradient_from_raw_fields":
            print(f"{prefix}: calculating velocity gradients (up to {config.z_reader_threads} threads)...", flush=True)
        elif stage == "inplace_units_and_columns":
            print(f"{prefix}: gradients finished; converting units and calculating foreground/shielding columns...", flush=True)

    slab, derivation = derive_native_column_tile(
        snapshot=snapshot,
        native_reader=snapshot.native_reader,
        raw_tile=raw,
        threads=config.z_reader_threads,
        on_stage=report_derivation,
    )
    print(
        f"{prefix}: physical fields ready; gradients {derivation['gradient_seconds']:.2f} s, "
        f"units/columns {derivation['units_and_columns_seconds']:.2f} s, "
        f"derivation total {derivation['fused_derivation_seconds']:.2f} s",
        flush=True,
    )
    del raw
    _check_process_memory("native tile derivation")
    merged_before = products.counts["all"]
    wave_count = 0

    def after_wave():
        nonlocal wave_count
        _check_process_memory("native worker wave")
        wave_count += 1

    def report_wave(*, stage, first_batch, last_batch, cell_count, seconds):
        batch_label = f"{prefix}, batches {first_batch}-{last_batch}/{tile_batches}"
        if stage == "physical":
            print(f"{batch_label}: calculating emission, dust, images, full moments and gas phases for {cell_count:,} cells...", flush=True)
        elif stage == "corners":
            print(f"{batch_label}: physical products ready in {seconds:.2f} s; preparing Gaussian corners and accounts...", flush=True)
        elif stage == "scatter":
            print(f"{batch_label}: Gaussian preparation finished in {seconds:.2f} s; accumulating weighted luminosity at nodes...", flush=True)
        elif stage == "complete":
            merged = products.counts["all"]
            fraction = merged / snapshot.processing_cell_count
            elapsed = monotonic() - process_began
            eta = elapsed * (1.0 - fraction) / fraction
            print(
                f"{batch_label}: wave finished in {seconds:.2f} s; "
                f"snapshot {merged:,}/{snapshot.processing_cell_count:,} cells ({100 * fraction:.2f}%), "
                f"batches {batches_before + last_batch}/{total_batches}; "
                f"cell-processing elapsed {elapsed:.1f} s, ETA {eta:.1f} s",
                flush=True,
            )

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
        on_wave=report_wave,
    )
    computation_seconds = monotonic() - computation_started
    cell_count = slab.cell_count
    if products.counts["all"] - merged_before != cell_count:
        raise AssertionError("Native tile did not merge every original cell exactly once")
    batch_count = (cell_count + config.query_chunk - 1) // config.query_chunk
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


def process_native_columns(
    *, config, snapshot, reader, emission_calculator, products, accumulate_batch,
    calculation_began, process_began,
):
    """Read one native tile at a time; immediately merge its finished batches.

    calculation_began starts the tile loop's elapsed/ETA clock. process_began
    starts the complete process clock, including input and reader initialization.
    Both are monotonic seconds supplied by run_one_snapshot().
    """
    bounds, expected_batches = _validate_plan(config, snapshot, reader)
    memory = _memory_plan(
        config=config,
        snapshot=snapshot,
        bounds=bounds,
    )
    if products.counts["all"] != 0:
        raise ValueError("Native column processing requires fresh product accumulators")
    print(
        f"Processing plan: {len(bounds)} full-z tiles, {expected_batches:,} batches, "
        f"{snapshot.processing_cell_count:,} selected cells; "
        f"estimated peak {memory['conservative_estimated_process_bytes'] / 1e9:.2f} GB "
        f"of {PROCESS_LIMIT_BYTES / 1e9:g} GB limit",
        flush=True,
    )
    records = []
    completed_batches = 0
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
                tile_number=number,
                tile_count=len(bounds),
                batches_before=completed_batches,
                total_batches=expected_batches,
                process_began=calculation_began,
            )
            records.append(record)
            completed_batches += record["merged_batches"]
            print(
                f"Tile {number}/{len(bounds)} finished in {record['total_seconds']:.2f} s: "
                f"read {record['read_seconds']:.2f} s, "
                f"derive {record['derivation']['fused_derivation_seconds']:.2f} s, "
                f"compute/merge {record['compute_and_merge_seconds']:.2f} s",
                flush=True,
            )
            processed_fraction = products.counts["all"] / snapshot.processing_cell_count
            elapsed = monotonic() - calculation_began
            write_status(
                output_dir=config.output_dir,
                counts=products.counts,
                began=process_began,
                state="running",
                completed_tiles=number,
                total_tiles=len(bounds),
                progress_percent=round(100 * processed_fraction, 1),
                eta_seconds=round(elapsed * (1.0 - processed_fraction) / processed_fraction),
            )
    merged_batches = sum(record["merged_batches"] for record in records)
    if products.counts["all"] != snapshot.processing_cell_count or merged_batches != expected_batches:
        raise AssertionError("Native column processing did not merge the selected snapshot cells")
    read_seconds = sum(record["read_seconds"] for record in records)
    derive_seconds = sum(record["derivation"]["fused_derivation_seconds"] for record in records)
    gradient_seconds = sum(record["derivation"]["gradient_seconds"] for record in records)
    columns_units_seconds = sum(record["derivation"]["units_and_columns_seconds"] for record in records)
    other_derivation_seconds = max(0.0, derive_seconds - gradient_seconds - columns_units_seconds)
    gaussian_timings = products.spectra.timing_summary()
    print(
        f"Processing stages finished: original reads {read_seconds:.2f} s; "
        f"gradients {gradient_seconds:.2f} s; units/columns {columns_units_seconds:.2f} s; "
        f"physical products {gaussian_timings['physical_preparation_wave_seconds']:.2f} s; "
        f"Gaussian preparation/accumulation {gaussian_timings['gaussian_wave_seconds_including_analytic_accounts_and_scheduling']:.2f} s; "
        f"ordered merges {gaussian_timings['ordered_product_merge_wave_seconds']:.2f} s",
        flush=True,
    )
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
