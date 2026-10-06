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
from quokka2s.run_settings import load_process_config
from quokka2s.processing_inputs import load_processing_inputs
from quokka2s.result_files import write_products_and_report, write_status
from quokka2s.products.emission_products import EmissionProducts


def accumulate_batch(cells, emission_calculator: CellEmissionCalculator, products):
    """Calculate a batch once, then add its emission to every product.

    cells is a CellBatch of six (N,) arrays plus grid positions/volume.
    emission_calculator holds the shared table readers and dust cross-sections.
    products receives image, spectrum, phase, and diagnostic sums in place.
    Only the sums survive this call; no product retains cells or line arrays."""
    products.failure_context = {
        "first_cell_id": cells.first_cell_id,
        "last_cell_id": cells.last_cell_id,
    }
    emission = emission_calculator.calculate(cells=cells)
    products.images.add_batch(cells=cells, emission=emission)
    products.spectra.add_batch(cells=cells, emission=emission)
    products.phases.add_batch(cells=cells, emission=emission)
    products.record_batch(cells=cells, emission=emission)
    # Only accumulated results survive this call; no product retains cells/emission.


def process_parallel_batches(
    config, snapshot, emission_calculator, products, slab, executor,
):
    """Process a slab with at most config.chunk_workers batches in flight.

    snapshot, emission_calculator, and slab are shared read-only inputs. Each
    worker owns its EmissionProducts; the main thread merges them in batch order.
    executor is reused across slabs. A successful return leaves no pending sums."""
    starts = iter(range(0, slab.cell_count, config.query_chunk))
    pending = deque()

    def compute(start, stop):
        """Calculate slab[start:stop] into independent product sums."""
        partial = EmissionProducts(
            snapshot=snapshot,
            line_keys=emission_calculator.line_keys,
            spectral_workers=config.spectral_workers,
        )
        cells = slab.batch(
            start=start,
            stop=stop,
        )
        accumulate_batch(
            cells=cells,
            emission_calculator=emission_calculator,
            products=partial,
        )
        return partial

    def submit_next():
        """Keep the pending queue bounded by submitting one remaining batch."""
        start = next(starts, None)
        if start is not None:
            stop = min(start + config.query_chunk, slab.cell_count)
            pending.append((executor.submit(compute, start, stop), start, stop))

    for _ in range(config.chunk_workers):
        submit_next()
    try:
        while pending:
            future, start, stop = pending.popleft()
            products.failure_context = {
                "first_cell_id": slab.cell_id_at(start),
                "last_cell_id": slab.cell_id_at(stop - 1),
            }
            partial = future.result()
            products.merge(partial)
            del partial, future
            submit_next()
    except BaseException:
        for future, _, _ in pending:
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
    selected_slabs = []
    for x_start in range(region_x_start, region_x_stop, config.slab_nx):
        x_stop = min(x_start + config.slab_nx, region_x_stop)
        selected_slabs.append((x_start, x_stop))
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


def process_serial_batches(config, emission_calculator, products, slab):
    """Accumulate consecutive batches from one already loaded slab.

    config.query_chunk limits each batch's cell count. Batch arrays are views;
    release the final view before process_one_slab() releases the parent slab."""
    for start in range(0, slab.cell_count, config.query_chunk):
        stop = min(start + config.query_chunk, slab.cell_count)
        cells = slab.batch(
            start=start,
            stop=stop,
        )
        accumulate_batch(
            cells=cells,
            emission_calculator=emission_calculator,
            products=products,
        )
        del cells  # Drop the batch views; the parent slab remains until finished.


def process_one_slab(
    config, snapshot, emission_calculator, products, window, executor,
):
    """Read one core x window and finish its batches before returning.

    window is (x_start, x_stop), with stop excluded. The shared executor is used
    when chunk_workers > 1. Only accumulated sums survive; the slab is released
    before progress reporting and the next slab read."""
    # read_slab() returns six flattened arrays, each normally shape (4194304,).
    # Neighbours are used for gradients and removed before these are returned.
    x_start, x_stop = window
    slab = snapshot.read_slab(
        x_start=x_start,
        x_stop=x_stop,
    )

    if config.chunk_workers == 1:
        process_serial_batches(
            config=config,
            emission_calculator=emission_calculator,
            products=products,
            slab=slab,
        )
    else:
        process_parallel_batches(
            config=config,
            snapshot=snapshot,
            emission_calculator=emission_calculator,
            products=products,
            slab=slab,
            executor=executor,
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

    # 2. Create the batch worker pool once and reuse it for all selected slabs.
    # With chunk_workers=2, two batches can run together within the current slab.
    with ThreadPoolExecutor(max_workers=config.chunk_workers) as executor:
        for slab_number, window in enumerate(selected_slabs, start=1):
            # 3. Read, calculate, accumulate, and release this slab's cell arrays.
            process_one_slab(
                config=config,
                snapshot=snapshot,
                emission_calculator=emission_calculator,
                products=products,
                window=window,
                executor=executor,
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


def main(argv=None):
    """Process a snapshot using --config PATH and save checked numerical products.

    argv is a sequence of command arguments, or None for sys.argv[1:]."""
    parser = argparse.ArgumentParser(
        prog='quokka2s-process',
        description=__doc__,
    )
    parser.add_argument(
        '--config',
        required=True,
        type=Path,
        help='YAML file containing the processing input and output paths',
    )
    config_path = parser.parse_args(argv).config
    try:
        config = load_process_config(config_path)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    snapshot, emission_calculator = load_processing_inputs(config)
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
        process_snapshot(
            config=config,
            snapshot=snapshot,
            emission_calculator=emission_calculator,
            products=products,
            began=began,
        )
        outputs = products.build_outputs()
        image_conservation = products.check_outputs(outputs)
        write_products_and_report(
            config=config,
            snapshot=snapshot,
            emission_calculator=emission_calculator,
            products=products,
            outputs=outputs,
            image_conservation=image_conservation,
            began=began,
        )
    except BaseException as exc:
        write_status(
            output_dir=config.output_dir,
            counts=products.counts,
            began=began,
            state='failed',
            error=f'{type(exc).__name__}: {exc}',
            failure_context=products.failure_context,
        )
        raise


if __name__ == '__main__':
    main()
