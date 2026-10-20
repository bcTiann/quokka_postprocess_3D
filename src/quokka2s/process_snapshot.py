"""Run snapshot reading, shared cell emission, and the three product accumulators.

Read this file for the order of work. The imported modules own the input/output
format, cell physics, image sums, spectral integration, and gas-phase statistics.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from quokka2s.physics.cell_emission import CellEmissionCalculator
from quokka2s.run_settings import add_run_selection_arguments, load_process_config
from quokka2s.processing_inputs import load_processing_inputs
from quokka2s.result_files import write_emission_results, write_status
from quokka2s.result_metadata import ResultMetadata
from quokka2s.processing_report import build_processing_report
from quokka2s.process_progress import SlabProgress
from quokka2s.products.emission_products import EmissionProducts
from quokka2s.snapshot_reader import slab_windows


def accumulate_batch(cells, emission_calculator: CellEmissionCalculator, products,
                     progress=None):
    """Calculate a batch once, then add its emission to every product.

    cells is a CellBatch of six (N,) arrays plus grid positions/volume.
    emission_calculator holds the shared table readers and dust cross-sections.
    products receives image, spectrum, phase, and diagnostic sums in place.
    progress optionally records a stage label before each existing calculation.
    Only the sums survive this call; no product retains cells or line arrays."""
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
        progress('integrating Gaussian line spectra')
    products.spectra.add_batch(cells=cells, emission=emission)
    if progress is not None:
        progress('accumulating gas-phase velocity statistics')
    products.phases.add_batch(cells=cells, emission=emission)
    if progress is not None:
        progress('recording missingness, luminosity and cell accounting')
    products.record_batch(cells=cells, emission=emission)
    # Only accumulated results survive this call; no product retains cells/emission.


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
            snapshot=snapshot,
            line_keys=emission_calculator.line_keys,
            spectral_workers=config.spectral_workers,
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

    window is (x_start, x_stop), with stop excluded. The shared executor is used
    when chunk_workers > 1. Only accumulated sums survive; the slab is released
    before progress reporting and the next slab read."""
    # read_slab() returns six flattened arrays, each normally shape (4194304,).
    # Neighbours are used for gradients and removed before these are returned.
    x_start, x_stop = window
    y_start, y_stop = snapshot.xy_region['y']
    z_stop = snapshot.shape[2]
    slab = snapshot.read_slab(
        x_start=x_start,
        x_stop=x_stop,
    )
    if progress is not None:
        progress.reading_complete()
    print(
        f'Calculating slab: x=[{x_start}, {x_stop}); y=[{y_start}, {y_stop}); '
        f'z=[0, {z_stop}); {slab.cell_count:,} cells; '
        'line emission, images, spectra and gas phases...',
        flush=True,
    )

    if config.chunk_workers == 1:
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
                             total_slabs, total_cells):
    """Print progress and write status.json after a finished slab.

    total_slabs and total_cells cover the selected run, including max_slabs.
    products.counts['all'] counts visited cells, including excluded cells.
    began is the run's time.monotonic() start, used for elapsed seconds and ETA."""
    completed_cells = products.counts['all']  # 4,194,304 after the first slab.
    elapsed_seconds = time.monotonic() - began
    progress_fraction = completed_cells / total_cells  # 1/32 in this example.
    remaining_cells = total_cells - completed_cells
    eta_seconds = elapsed_seconds * remaining_cells / completed_cells

    write_status(
        output_dir=config.output_dir,
        counts=products.counts,
        began=began,
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


def process_snapshot(config, snapshot, emission_calculator, products, began):
    """Read, accumulate, and release one slab at a time.

    config supplies slab/batch sizes, workers, and max_slabs. snapshot supplies
    grid geometry and read_slab(); emission_calculator holds the shared readers.
    products accumulates sums in place; began is the time.monotonic() run start.
    Final product checks and saving happen in main()."""
    # 1. Select index windows only; no simulation fields are loaded yet.
    selected_slabs = select_slabs_to_process(config, snapshot)
    total_slabs = len(selected_slabs)  # Full run: 32 slabs for Nx=256, slab_nx=8.
    total_cells = count_cells_in_slabs(snapshot, selected_slabs)  # Full run: 134,217,728.
    print(f'Processing plan: {total_cells:,} cells; slab count: {total_slabs}.', flush=True)
    y_start, y_stop = snapshot.xy_region['y']
    z_stop = snapshot.shape[2]

    # 2. Create the batch worker pool once and reuse it for all selected slabs.
    # With chunk_workers=2, two batches can run together within the current slab.
    with snapshot.native_reader.reuse_slab_fields(
        selected_slabs=selected_slabs,
        selected_y_range=snapshot.xy_region['y'],
        slabs_per_read=4,
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

                # 4. Report a finished slab (1/32, 2/32, ...) before reading the next.
                report_snapshot_progress(
                    config=config,
                    products=products,
                    began=began,
                    completed_slabs=slab_number,
                    total_slabs=total_slabs,
                    total_cells=total_cells,
                )


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
        f'chunk_workers={config.chunk_workers}; spectral_workers={config.spectral_workers}',
        flush=True,
    )


def main(argv=None):
    """Process the selected job/snapshot and save checked numerical products.

    argv is a sequence of command arguments, or None for sys.argv[1:].
    From the repository root, the default is configs/emission.yaml;
    --config PATH selects another YAML file. --dataset supplies an absolute
    snapshot path; --x/--y override native ranges. Omitted options retain YAML.
    """
    parser = argparse.ArgumentParser(
        prog='process',
        description=__doc__,
    )
    add_run_selection_arguments(parser)
    args = parser.parse_args(argv)
    config_path = args.config
    try:
        config = load_process_config(
            path=config_path,
            dataset=args.dataset,
            x_index_range=args.x_index_range,
            y_index_range=args.y_index_range,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    print_run_settings(config=config, config_path=config_path)
    snapshot, emission_calculator = load_processing_inputs(config)
    print('Outputs: allocating image, spectrum and gas-phase accumulators...', flush=True)
    products = EmissionProducts(
        snapshot=snapshot,
        line_keys=emission_calculator.line_keys,
        spectral_workers=config.spectral_workers,
    )
    began = time.monotonic()
    write_status(
        output_dir=config.output_dir,
        counts=products.counts,
        began=began,
        state='running',
    )
    try:
        stage = "processing cells"
        process_snapshot(
            config=config,
            snapshot=snapshot,
            emission_calculator=emission_calculator,
            products=products,
            began=began,
        )
        stage = "building results"
        print('Outputs: building image, spectrum and gas-phase results...', flush=True)
        outputs = products.build_outputs()
        metadata = ResultMetadata.from_processing_inputs(
            snapshot=snapshot,
            emission_calculator=emission_calculator,
        )
        outputs.add_metadata(metadata=metadata)
        stage = "checking results"
        print('Outputs: checking luminosity conservation and cell accounting...', flush=True)
        image_conservation = products.check_outputs(outputs)
        stage = "building report"
        print('Outputs: preparing the processing report...', flush=True)
        report = build_processing_report(
            config=config,
            snapshot=snapshot,
            emission_calculator=emission_calculator,
            products=products,
            outputs=outputs,
            metadata=metadata,
            image_conservation=image_conservation,
            began=began,
        )
        stage = "saving results"
        print(f'Outputs: saving numerical products and report to: {config.output_dir}', flush=True)
        write_emission_results(
            directory=config.output_dir,
            outputs=outputs,
            report=report,
        )
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


if __name__ == '__main__':
    main()
