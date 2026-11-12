"""Run snapshot reading, shared cell emission, and the three product accumulators.

Read this file for the order of work. The imported modules own the input/output
format, cell physics, image sums, spectral integration, and gas-phase statistics.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import time

from quokka2s.physics.cell_emission import CellEmissionCalculator
from quokka2s.run_settings import (
    ProcessSettings,
    SPECTRAL_METHODS,
    FIELD_SOURCES,
    add_plot_arguments,
    add_run_selection_arguments,
    load_plot_config,
    load_process_configs,
)
from quokka2s.processing_inputs import (
    load_processing_inputs,
    prepare_processing_snapshot,
)
from quokka2s.result_files import write_emission_results, write_status
from quokka2s.result_metadata import ResultMetadata
from quokka2s.processing_report import build_processing_report
from quokka2s.process_progress import READ_STAGE_LABELS, SlabProgress
from quokka2s.process_reuse import processing_input_identity, reuse_completed_results
from quokka2s.products.emission_products import EmissionProducts
from quokka2s.snapshot_reader import slab_windows


def accumulate_batch(cells, emission_calculator: CellEmissionCalculator, products,
                     progress=None):
    """Calculate a batch once, then add its emission to every product.

    cells is a CellBatch of six (N,) arrays plus grid positions/volume.
    emission_calculator holds the shared table readers and dust cross-sections.
    products receives image, spectrum, phase, and diagnostic sums in place.
    progress optionally records a stage label before each existing calculation.
    Direct-template workers may retain owned Gaussian input vectors until the
    next bounded query wave. No CellBatch or LineEmission object is retained."""
    products.failure_context = {
        "first_cell_id": cells.first_cell_id,
        "last_cell_id": cells.last_cell_id,
    }
    if progress is not None:
        progress('DESPOTIC/Cloudy lookup, line emissivity and dust attenuation')
    emission = emission_calculator.calculate(cells=cells)
    if progress is not None:
        progress('accumulating luminosity images')
    products.images.add_batch(cells=cells, emission=emission)
    if progress is not None:
        if products.spectra.spectral_method == "direct-template":
            progress('preparing original-cell Gaussian inputs')
        else:
            progress('integrating Gaussian line spectra')
    products.spectra.add_batch(cells=cells, emission=emission)
    if progress is not None:
        progress('accumulating gas-phase velocity statistics')
    products.phases.add_batch(cells=cells, emission=emission)
    if progress is not None:
        progress('recording missingness, luminosity and cell accounting')
    products.record_batch(cells=cells, emission=emission)
    # A template partial also owns its bounded Gaussian queue; not cells/emission.


def process_parallel_batches(
    config, snapshot, emission_calculator, products, slab, executor, progress=None,
):
    """Process a slab with at most config.chunk_workers batches in flight.

    snapshot, emission_calculator, and slab are shared read-only inputs. Each
    worker owns its EmissionProducts; the main thread merges them in batch order.
    executor is reused across slabs. A successful return leaves no pending sums."""
    batches = slab.iter_batches(batch_size=config.query_chunk)
    pending = deque()
    submitted_batches = 0

    def compute(cells, batch_number):
        """Calculate one batch into independent product sums."""
        stage_progress = None
        if progress is not None:
            progress.start_batch(batch_number)
            stage_progress = lambda stage: progress.set_stage(batch_number, stage)
        partial = EmissionProducts(
            histogram_engine=getattr(products.spectra, 'engine', None),
            histogram_partial=True,
            histogram_merge_threads=config.histogram_merge_threads,
            snapshot=snapshot,
            line_keys=emission_calculator.line_keys,
            spectral_workers=config.spectral_workers,
            spectral_method=config.spectral_method,
        )
        accumulate_batch(
            cells=cells,
            emission_calculator=emission_calculator,
            products=partial,
            progress=stage_progress,
        )
        if progress is not None:
            progress.batch_ready(batch_number)
        return partial

    def submit_next():
        """Keep the pending queue bounded by submitting one remaining batch."""
        nonlocal submitted_batches
        cells = next(batches, None)
        if cells is not None:
            submitted_batches += 1
            future = executor.submit(compute, cells, submitted_batches)
            pending.append((future, cells.first_cell_id, cells.last_cell_id,
                            submitted_batches, cells.cell_count))

    for _ in range(config.chunk_workers):
        submit_next()
    try:
        while pending:
            future, first_cell_id, last_cell_id, batch_number, cell_count = pending.popleft()
            products.failure_context = {
                "first_cell_id": first_cell_id,
                "last_cell_id": last_cell_id,
            }
            partial = future.result()
            if progress is not None:
                progress.set_stage(batch_number, 'merging products in original batch order')
            products.merge(partial)
            if progress is not None:
                progress.batch_merged(batch_number, cell_count)
            del partial, future
            submit_next()
    except BaseException:
        for future, *_ in pending:
            future.cancel()
        raise


def duration_label(seconds):
    """Format seconds for progress, e.g. duration_label(65) -> '1m 5s'."""
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f'{hours}h {minutes}m {seconds}s'
    return f'{minutes}m {seconds}s'


def select_slabs_to_process(config, snapshot):
    """Select core x windows without reading cells.

    snapshot.xy_region supplies the global x range. config.slab_nx limits each
    window's width; max_slabs optionally selects its first windows. read_slab()
    reads gradient neighbours from the original box, even outside the region."""
    region_x_start, region_x_stop = snapshot.xy_region["x"]
    selected_slabs = list(slab_windows(
        x_start=region_x_start,
        x_stop=region_x_stop,
        slab_nx=config.slab_nx,
    ))
    if config.max_slabs is not None:
        selected_slabs = selected_slabs[:config.max_slabs]
    return selected_slabs


def count_cells_in_slabs(snapshot, selected_slabs):
    """Count selected core cells for progress, excluding gradient neighbours.

    snapshot.processing_shape supplies selected Ny and full Nz; each window
    is an exclusive global x range. Gradient neighbours are not counted."""
    cells_per_x_layer = snapshot.processing_shape[1] * snapshot.processing_shape[2]
    total_cells = 0
    for window in selected_slabs:
        x_start, x_stop = window  # E.g. 0, 8; x_stop is excluded.
        x_layers_in_slab = x_stop - x_start  # Usually 8; the last may be shorter.
        total_cells += x_layers_in_slab * cells_per_x_layer
    return total_cells


def process_serial_batches(config, emission_calculator, products, slab, progress=None):
    """Accumulate consecutive batches from one already loaded slab.

    config.query_chunk limits each batch's cell count. Batch arrays are views;
    release the final view before process_one_slab() releases the parent slab."""
    for batch_number, cells in enumerate(slab.iter_batches(batch_size=config.query_chunk), start=1):
        stage_progress = None
        if progress is not None:
            progress.start_batch(batch_number)
            stage_progress = lambda stage: progress.set_stage(batch_number, stage)
        accumulate_batch(
            cells=cells,
            emission_calculator=emission_calculator,
            products=products,
            progress=stage_progress,
        )
        if progress is not None:
            progress.batch_merged(batch_number, cells.cell_count)
        del cells  # Drop the batch views; the parent slab remains until finished.


def process_one_slab(
    config, snapshot, emission_calculator, products, window, executor, progress=None,
):
    """Read one core x window and finish its batches before returning.

    window is (x_start, x_stop), with stop excluded. The default template method
    uses the shared executor even for one worker. All query queues finish before
    the slab is released, progress is reported and the next slab is read."""
    # read_slab() returns six flattened arrays, each normally shape (4194304,).
    # Neighbours are used for gradients and removed before these are returned.
    x_start, x_stop = window
    y_start, y_stop = snapshot.xy_region['y']
    z_stop = snapshot.shape[2]
    slab = snapshot.read_slab(
        x_start=x_start,
        x_stop=x_stop,
        on_stage=progress.set_read_stage if progress is not None else None,
    )
    if progress is not None:
        progress.reading_complete()
    print(
        f'Calculating slab: x=[{x_start}, {x_stop}); y=[{y_start}, {y_stop}); '
        f'z=[0, {z_stop}); {slab.cell_count:,} cells; '
        'line emission, images, spectra and gas phases...',
        flush=True,
    )

    if config.spectral_method == "direct-template":
        from quokka2s.products.gaussian.template_batches import process_template_batches
        process_template_batches(
            config=config,
            snapshot=snapshot,
            emission_calculator=emission_calculator,
            products=products,
            slab=slab,
            executor=executor,
            accumulate_batch=accumulate_batch,
            progress=progress,
        )
    elif config.chunk_workers == 1:
        process_serial_batches(
            config=config,
            emission_calculator=emission_calculator,
            products=products,
            slab=slab,
            progress=progress,
        )
    else:
        process_parallel_batches(
            config=config,
            snapshot=snapshot,
            emission_calculator=emission_calculator,
            products=products,
            slab=slab,
            executor=executor,
            progress=progress,
        )
    del slab  # All batch results are merged before the next slab is read.


def report_snapshot_progress(config, products, began, completed_slabs,
                             total_slabs, total_cells, status_began=None):
    """Print progress and write status.json after a finished slab.

    total_slabs and total_cells cover the selected run, including max_slabs.
    products.counts['all'] counts visited cells, including excluded cells.
    began starts cell calculation for the ETA. status_began optionally includes
    input loading and conversion in the persistent total elapsed time."""
    completed_cells = products.counts['all']  # 4,194,304 after the first slab.
    elapsed_seconds = time.monotonic() - began
    progress_fraction = completed_cells / total_cells  # 1/32 in this example.
    remaining_cells = total_cells - completed_cells
    eta_seconds = elapsed_seconds * remaining_cells / completed_cells

    write_status(
        output_dir=config.output_dir,
        counts=products.counts,
        began=began if status_began is None else status_began,
        state='running',
        completed_slabs=completed_slabs,
        progress_percent=round(100 * progress_fraction, 1),
        eta_seconds=round(eta_seconds),
    )
    print(
        f'Emission: {progress_fraction:.1%} ({completed_slabs}/{total_slabs} slabs); '
        f'elapsed {duration_label(elapsed_seconds)}; '
        f'ETA ~{duration_label(eta_seconds)}',
        flush=True,
    )


def process_snapshot(config, snapshot, emission_calculator, products, began, *, status_began=None, native_column_reader=None):
    """Read bounded fields, accumulate products and retain original batch order.

    Native columns are the default: original full-z XY tiles are derived in
    RAM; finished waves merge immediately in deterministic native-tile order.
    The explicit z-contiguous source retains the bounded x-slab implementation.

    config supplies slab/batch sizes, workers, and max_slabs. snapshot supplies
    grid geometry and read_slab(); emission_calculator holds the shared readers.
    Returns per-slab and total wall timings; the five read stages partition
    read/derive time, and columns_total_seconds is their two-column subtotal.
    products accumulates sums in place; began starts cell calculation for ETA.
    status_began, when supplied, starts the complete process clock, excluding
    separately measured z conversion and prebuilt Gaussian templates.
    Final product checks and saving happen in main()."""
    if config.field_source == "native-columns":
        from .native_column_process import process_native_columns
        if native_column_reader is None:
            raise ValueError("native-columns requires its explicitly prepared original-field reader")
        return process_native_columns(
            config=config,
            snapshot=snapshot,
            reader=native_column_reader,
            emission_calculator=emission_calculator,
            products=products,
            accumulate_batch=accumulate_batch,
            began=began,
            status_began=status_began,
        )
    # 1. Select index windows only; no simulation fields are loaded yet.
    selected_slabs = select_slabs_to_process(config, snapshot)
    total_slabs = len(selected_slabs)  # Full run: 32 slabs for Nx=256, slab_nx=8.
    total_cells = count_cells_in_slabs(snapshot, selected_slabs)  # Full run: 134,217,728.
    print(f'Processing plan: {total_cells:,} cells; slab count: {total_slabs}.', flush=True)
    slab_timings = []
    y_start, y_stop = snapshot.xy_region['y']
    z_stop = snapshot.shape[2]

    # 2. Create the batch worker pool once and reuse it for all selected slabs.
    # With chunk_workers=2, two batches can run together within the current slab.
    with snapshot.native_reader.reuse_slab_fields(
        selected_slabs=selected_slabs,
        selected_y_range=snapshot.xy_region['y'],
        slabs_per_read=config.field_cache_slabs,
        max_cached_bytes=config.field_cache_gib * 1024**3,
    ):
        with ThreadPoolExecutor(max_workers=config.chunk_workers) as executor:
            for slab_number, window in enumerate(selected_slabs, start=1):
                # 3. Read, calculate, accumulate, and release this slab's cell arrays.
                x_start, x_stop = window
                print(
                    f'Reading slab: {slab_number}/{total_slabs}; x=[{x_start}, {x_stop}); '
                    f'y=[{y_start}, {y_stop}); z=[0, {z_stop}); '
                    'reading cell fields and deriving columns and velocity gradients...',
                    flush=True,
                )
                slab_cell_count = (x_stop - x_start) * (y_stop - y_start) * z_stop
                with SlabProgress(
                    slab_number=slab_number,
                    total_slabs=total_slabs,
                    cell_count=slab_cell_count,
                    batch_size=config.query_chunk,
                    workers=config.chunk_workers,
                ) as progress:
                    process_one_slab(
                        config=config,
                        snapshot=snapshot,
                        emission_calculator=emission_calculator,
                        products=products,
                        window=window,
                        executor=executor,
                        progress=progress,
                    )
                slab_timings.append({
                    'slab_number': slab_number,
                    'x_index_range': list(window),
                    'read_stage_seconds': progress.read_stage_seconds.copy(),
                    'read_derive_seconds': progress.read_seconds,
                    'compute_and_ordered_merge_seconds': progress.compute_seconds,
                })

                # 4. Report a finished slab (1/32, 2/32, ...) before reading the next.
                report_snapshot_progress(
                    config=config,
                    products=products,
                    began=began,
                    status_began=status_began,
                    completed_slabs=slab_number,
                    total_slabs=total_slabs,
                    total_cells=total_cells,
                )

    read_stage_totals = {
        stage: sum(record['read_stage_seconds'][stage] for record in slab_timings)
        for stage in READ_STAGE_LABELS
    }
    timings = {
        'read_stage_labels': READ_STAGE_LABELS.copy(),
        'read_stage_totals_seconds': read_stage_totals,
        'columns_total_seconds': (
            read_stage_totals['foreground_column'] + read_stage_totals['shielding_column']
        ),
        'read_derive_total_seconds': sum(record['read_derive_seconds'] for record in slab_timings),
        'compute_and_ordered_merge_total_seconds': sum(
            record['compute_and_ordered_merge_seconds'] for record in slab_timings
        ),
        'slabs': slab_timings,
    }
    if config.spectral_method == "direct-template":
        timings['gaussian'] = products.spectra.timing_summary()
    return timings


def print_run_settings(config, config_path: Path) -> None:
    """Print the selected run, resolved data/output paths, and worker settings.

    config is ProcessSettings from load_process_config(); config_path is the
    supplied YAML filename. This startup summary does not open any input data.
    Example: sigma13.v2/plt0916720 saves under that job/snapshot's processed/.
    """
    print(f'Configuration: {config_path}', flush=True)
    print(f'Processing run: {config.jobname}/{config.snapshot_name}', flush=True)
    print(f'Simulation: {config.dataset}', flush=True)
    print(f'Numerical output: {config.output_dir}', flush=True)
    print(
        f'Execution: slab_nx={config.slab_nx}; query_chunk={config.query_chunk:,}; '
        f'chunk_workers={config.chunk_workers}; spectral_workers={config.spectral_workers}; '
        f'field_cache_slabs={config.field_cache_slabs}; field_cache_gib={config.field_cache_gib}; '
        f'spectral_method={config.spectral_method}; field_source={config.field_source}',
        flush=True,
    )


def run_one_snapshot(config: ProcessSettings, config_path: Path) -> None:
    """Calculate, check and save one snapshot's numerical products without drawing.

    config owns resolved paths and one native x/y selection. All accumulators,
    snapshot fields and lookup readers stay local to this call. The temporary
    z-contiguous copy is removed before the next snapshot or plotting step.
    """
    histogram_engine = None
    gaussian_table_seconds = 0.0
    if config.spectral_method == "direct-template":
        from quokka2s.products.gaussian.template_engine import prepare_template_engine
        histogram_engine, gaussian_table_seconds = prepare_template_engine(config.gaussian_grid)
    elif config.spectral_method == "histogram-2d":
        from quokka2s.products.gaussian.preparation import prepare_gaussian_engine
        histogram_engine, gaussian_table_seconds = prepare_gaussian_engine(config.gaussian_grid)
    elif config.spectral_method == "histogram-layered":
        from quokka2s.products.gaussian.layered_engine import prepare_layered_engine
        histogram_engine, gaussian_table_seconds = prepare_layered_engine(config.gaussian_grid)
    print_run_settings(config=config, config_path=config_path)
    began = time.monotonic()
    input_identity = processing_input_identity(config=config)
    snapshot, emission_calculator = load_processing_inputs(config)
    input_loading_seconds = time.monotonic() - began
    print('Outputs: allocating image, spectrum and gas-phase accumulators...', flush=True)
    products = EmissionProducts(
        histogram_engine=histogram_engine,
        histogram_merge_threads=config.histogram_merge_threads,
        snapshot=snapshot,
        line_keys=emission_calculator.line_keys,
        spectral_workers=config.spectral_workers,
        spectral_method=config.spectral_method,
    )
    write_status(
        output_dir=config.output_dir,
        counts=products.counts,
        began=began,
        state='running',
    )
    try:
        stage = 'preparing ' + config.field_source + ' fields'
        with prepare_processing_snapshot(
            snapshot=snapshot,
            dataset_path=config.dataset,
            work_directory=config.output_dir.parent,
            source_identity=input_identity['snapshot_files'],
            field_source=config.field_source,
            prepared_directory=config.prepared_z_fields,
            reader_threads=config.z_reader_threads,
            read_workers=config.z_read_workers,
        ) as (prepared_snapshot, preprocessing):
            # Exclude only conversion. Input initialization and all numerical
            # slab reads, derivations, calculations, checks and saves stay timed.
            began += preprocessing['conversion_seconds']
            stage = 'processing cells'
            calculation_began = time.monotonic()
            processing_timings = process_snapshot(
                config=config,
                snapshot=prepared_snapshot,
                emission_calculator=emission_calculator,
                products=products,
                began=calculation_began,
                status_began=began,
                native_column_reader=preprocessing.get('native_column_reader'),
            )
            processing_timings['input_loading_seconds'] = input_loading_seconds
            processing_timings['z_conversion_seconds'] = preprocessing['conversion_seconds']
            processing_timings['z_cache_validation_seconds'] = preprocessing['preprocessing_seconds'] if preprocessing['cache_reused'] else 0.0
            processing_timings['field_reader_initialization_seconds'] = preprocessing['reader_initialization_seconds']
            processing_timings['z_reader_initialization_seconds'] = preprocessing['reader_initialization_seconds'] if config.field_source == "z-contiguous" else 0.0
            processing_timings['z_conversion_excluded_from_process'] = config.field_source == "z-contiguous"
            processing_timings['field_source'] = config.field_source
            processing_timings['gaussian_table_build_seconds_excluded'] = gaussian_table_seconds
            stage = 'building results'
            print('Outputs: building image, spectrum and gas-phase results...', flush=True)
            outputs = products.build_outputs()
            metadata = ResultMetadata.from_processing_inputs(
                snapshot=prepared_snapshot,
                emission_calculator=emission_calculator,
            )
            outputs.add_metadata(metadata=metadata)
            stage = 'checking results'
            print('Outputs: checking luminosity conservation and cell accounting...', flush=True)
            image_conservation = products.check_outputs(outputs)
            stage = 'building report'
            print('Outputs: preparing the processing report...', flush=True)
            report = build_processing_report(
                config=config,
                snapshot=prepared_snapshot,
                emission_calculator=emission_calculator,
                products=products,
                outputs=outputs,
                metadata=metadata,
                image_conservation=image_conservation,
                began=began,
                processing_timings=processing_timings,
            )
            if config.field_source == "native-columns":
                native = processing_timings['native_columns']
                report['field_preprocessing'] = {
                    'method': 'original-native-columns-inplace-ram',
                    'field_source': config.field_source,
                    'disk_copy_created': False,
                    'conversion_excluded_from_process': False,
                    'seconds': 0.0,
                    'shape': list(prepared_snapshot.shape),
                    'fields': ['gasDensity', 'temperature', 'x-GasMomentum', 'y-GasMomentum', 'z-GasMomentum'],
                    'source_stored_dtype': native['read_statistics']['native_dtype'],
                    'source_layout': native['source_layout'],
                    'ram_dtype': 'float64',
                    'ram_layout': native['ram_layout'],
                    'cache_file_bytes': 0,
                    'reader_threads': config.z_reader_threads,
                    'read_workers': config.z_read_workers,
                    'native_reading': native['read_statistics'],
                    'memory_plan': native['memory_plan'],
                    'finished_products': native['finished_products'],
                    'retention': 'one RAM tile; released before reading the next tile',
                }
            else:
                cache_identity = preprocessing['identity']
                report['field_preprocessing'] = {
                    'method': 'disk-z-contiguous-bounded-eager-fused',
                    'snapshot_metadata': preprocessing['identity']['snapshot_metadata'],
                    'metadata_sidecar_write': preprocessing['metadata_sidecar_write'],
                    'prepared_cache_reused': preprocessing['cache_reused'],
                    'reader_threads': config.z_reader_threads,
                    'read_workers': config.z_read_workers,
                    'bounded_eager_reading': dict(preprocessing['reader_statistics']),
                    'fused_derivation': dict(preprocessing['derivation_statistics']),
                    'conversion_excluded_from_process': True,
                    'seconds': preprocessing['preprocessing_seconds'],
                    'shape': cache_identity['shape'],
                    'fields': list(cache_identity['fields']),
                    'stored_dtype': cache_identity['stored_dtype'],
                    'layout': cache_identity['layout'],
                    'cache_file_bytes': sum(
                        record['size_bytes']
                        for record in cache_identity['fields'].values()
                    ),
                    'retention': 'retained explicit cache' if preprocessing['cache_reused'] else 'temporary; removed when snapshot processing exits',
                }
            if processing_input_identity(config=config) != input_identity:
                raise ValueError('Processing inputs or source files changed during calculation')
            report['input_identity'] = input_identity
            stage = 'saving results'
            print(f'Outputs: saving numerical products and report to: {config.output_dir}', flush=True)
            write_emission_results(
                directory=config.output_dir,
                outputs=outputs,
                report=report,
            )
        # All field maps and the owned working copy are gone before completion
        # or plotting. Persistent result reuse needs only the original inputs.
        write_status(
            output_dir=config.output_dir,
            counts=products.counts,
            began=began,
            state=report["status"],
            processing_complete=outputs.processing_complete,
            full_snapshot=outputs.full_snapshot,
        )
        print(f'Run finished: {report["status"]}.', flush=True)
        print(f'Saved numerical results: {config.output_dir}', flush=True)
    except BaseException as exc:
        print(f'Failed while {stage}: {type(exc).__name__}: {exc}', flush=True)
        write_status(
            output_dir=config.output_dir,
            counts=products.counts,
            began=began,
            state='failed',
            error=f'{type(exc).__name__}: {exc}',
            stage=stage,
            failure_context=(products.failure_context if stage == "processing cells" else {}),
        )
        raise


def main(argv=None):
    """Process one snapshot or a run's direct plt directories, then draw each.

    --dataset requires --jobname and overrides YAML with an absolute snapshot
    or parent run directory. --jobname alone replaces the YAML output group.
    --no-plot saves numerical products only. Plot choices otherwise use the same
    YAML and format/title flags as plot. Each invocation freezes its inventory;
    completed matching products are reused and conflicting output is preserved.
    --spectral-method explicitly overrides the YAML algorithm. Original-cell
    direct-template queries are the default; retained node, exact and linear-lut
    methods remain explicit comparison options.
    """
    parser = argparse.ArgumentParser(
        prog="process",
        description=__doc__,
    )
    add_run_selection_arguments(parser)
    add_plot_arguments(parser)
    parser.add_argument(
        "--spectral-method",
        choices=SPECTRAL_METHODS,
        default=None,
        help="Override Gaussian method; direct-template is the default",
    )
    parser.add_argument(
        "--field-source",
        choices=FIELD_SOURCES,
        default=None,
        help="Override field source; native-columns reads original plt into RAM by default",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Save numerical products only; draw them later with plot --all",
    )
    args = parser.parse_args(argv)
    if args.dataset is not None and args.jobname is None:
        parser.error("--dataset requires --jobname; explicitly name the output group")
    try:
        configs = load_process_configs(
            path=args.config,
            dataset=args.dataset,
            jobname=args.jobname,
            x_index_range=args.x_index_range,
            y_index_range=args.y_index_range,
            spectral_method=args.spectral_method,
            field_source=args.field_source,
        )
        plot_defaults = None
        if not args.no_plot:
            plot_defaults = load_plot_config(
                path=args.config,
                dataset=args.dataset,
                jobname=args.jobname,
                x_index_range=args.x_index_range,
                y_index_range=args.y_index_range,
                figure_formats=args.figure_formats,
                figure_styles=args.figure_styles,
            )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    started = time.monotonic()
    processed = []
    reused = []
    plotted = []
    failures = []
    total = len(configs)
    group_source = "--jobname" if args.jobname is not None else "YAML"
    print(f"Output group: {configs[0].jobname} (from {group_source})", flush=True)
    print(f"Process inventory: {total} snapshot(s); plotting={'off' if args.no_plot else 'on'}", flush=True)
    for config in configs:
        print(f"  {config.snapshot_name}: {config.dataset} -> {config.output_dir}", flush=True)
    for index, config in enumerate(configs, start=1):
        print(f"Snapshot {index}/{total}: {config.snapshot_name}", flush=True)
        stage = "processing numerical products"
        try:
            if reuse_completed_results(config=config):
                reused.append(config.snapshot_name)
                print(f"Process: reusing completed matching results: {config.output_dir}", flush=True)
            else:
                run_one_snapshot(config=config, config_path=args.config)
                processed.append(config.snapshot_name)
            if plot_defaults is not None:
                stage = "plotting saved products"
                # Import only after numerical processing has released its arrays.
                from quokka2s.plot_emission_results import plot_one_snapshot

                plot_config = replace(
                    plot_defaults,
                    products=config.output_dir,
                    output_dir=config.output_dir.parent / "figures",
                )
                plot_one_snapshot(config=plot_config)
                plotted.append(config.snapshot_name)
        except Exception as exc:
            failures.append((config.snapshot_name, stage, f"{type(exc).__name__}: {exc}"))
            print(f"Snapshot failed ({stage}): {config.snapshot_name}: {exc}", flush=True)
        elapsed = time.monotonic() - started
        eta = elapsed / index * (total - index)
        print(
            f"Snapshots: {index}/{total}; elapsed {duration_label(elapsed)}; "
            f"ETA ~{duration_label(eta)}",
            flush=True,
        )
    print(
        f"Process summary: processed={len(processed)}, reused={len(reused)}, "
        f"plotted={len(plotted)}, failed={len(failures)}; "
        f"elapsed {duration_label(time.monotonic() - started)}",
        flush=True,
    )
    for name, stage, error in failures:
        print(f"  FAILED {name} ({stage}): {error}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
