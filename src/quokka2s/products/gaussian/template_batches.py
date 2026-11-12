"""Process bounded physical batches before their original-cell template queries."""
from __future__ import annotations

from itertools import islice
from time import perf_counter_ns


def process_template_batches(
    *, config, snapshot, emission_calculator, products, slab, executor,
    accumulate_batch, progress=None, after_wave=None,
):
    """Keep at most chunk_workers Gaussian input queues within the current slab.

    The ordinary accumulate_batch function computes the same physical products,
    available-line grouping and complete moments. Each worker then queries its
    owned Gaussians. Results merge in the current slab's batch order. Native
    tiles are visited in their deterministic XY order; sums across tiles may
    therefore differ in floating-point rounding from a full-grid batch walk.
    A one-worker configuration follows this same path. after_wave optionally
    checks process memory after each bounded worker wave completes.
    """
    from ..emission_products import EmissionProducts

    batches = iter(slab.iter_batches(batch_size=config.query_chunk))
    submitted_batches = 0
    while True:
        wave_cells = list(islice(batches, config.chunk_workers))
        if not wave_cells:
            break
        first_batch = submitted_batches + 1
        batch_numbers = list(range(first_batch, first_batch + len(wave_cells)))
        submitted_batches += len(wave_cells)
        physical_started = perf_counter_ns()
        physical_futures = []
        gaussian_futures = []
        partials = []

        def prepare_one(batch_number, cells):
            stage_progress = None
            if progress is not None:
                progress.start_batch(batch_number)
                stage_progress = lambda stage: progress.set_stage(batch_number, stage)
            partial = EmissionProducts(
                histogram_engine=products.spectra.engine,
                histogram_partial=True,
                histogram_merge_threads=config.histogram_merge_threads,
                snapshot=snapshot,
                line_keys=emission_calculator.line_keys,
                spectral_workers=config.spectral_workers,
                spectral_method=config.spectral_method,
            )
            try:
                accumulate_batch(
                    cells=cells,
                    emission_calculator=emission_calculator,
                    products=partial,
                    progress=stage_progress,
                )
            except BaseException:
                partial.spectra.release_gaussian_inputs()
                raise
            if progress is not None:
                progress.set_stage(batch_number, "prepared original-cell Gaussians")
            return partial

        def finish_one(batch_number, partial):
            if progress is not None:
                progress.set_stage(batch_number, "querying Gaussian templates")
            partial.spectra.finish_gaussians()
            if progress is not None:
                progress.batch_ready(batch_number)
            return partial

        try:
            for batch_number, cells in zip(batch_numbers, wave_cells):
                physical_futures.append(executor.submit(prepare_one, batch_number, cells))
            for cells, future in zip(wave_cells, physical_futures):
                products.failure_context = {
                    "first_cell_id": cells.first_cell_id,
                    "last_cell_id": cells.last_cell_id,
                }
                partials.append(future.result())
            physical_finished = perf_counter_ns()
            owned_input_bytes = sum(partial.spectra._owned_input_bytes for partial in partials)
            prepared_groups = sum(partial.spectra._groups_prepared for partial in partials)
            gaussian_started = perf_counter_ns()
            for batch_number, partial in zip(batch_numbers, partials):
                gaussian_futures.append(executor.submit(finish_one, batch_number, partial))
            for cells, future in zip(wave_cells, gaussian_futures):
                products.failure_context = {
                    "first_cell_id": cells.first_cell_id,
                    "last_cell_id": cells.last_cell_id,
                }
                future.result()
            gaussian_finished = perf_counter_ns()
            finished_groups = sum(partial.spectra._groups_finished for partial in partials)
            if prepared_groups != finished_groups:
                raise AssertionError("A Gaussian wave omitted one or more prepared groups")
            if any(partial.spectra._groups for partial in partials):
                raise AssertionError("A completed Gaussian wave retained its input vectors")
            merge_started = perf_counter_ns()
            for batch_number, cells, partial in zip(batch_numbers, wave_cells, partials):
                products.failure_context = {
                    "first_cell_id": cells.first_cell_id,
                    "last_cell_id": cells.last_cell_id,
                }
                if progress is not None:
                    progress.set_stage(batch_number, "merging completed batch products")
                products.merge(partial)
                if progress is not None:
                    progress.batch_merged(batch_number, cells.cell_count)
            merge_finished = perf_counter_ns()
            products.spectra.record_completed_wave(
                physical_ns=physical_finished - physical_started,
                gaussian_ns=gaussian_finished - gaussian_started,
                merge_ns=merge_finished - merge_started,
                owned_input_bytes=owned_input_bytes,
            )
            if after_wave is not None:
                after_wave()
        except BaseException:
            # Wait only for this bounded wave before releasing every surviving
            # partial; no failed task is resubmitted or reused in another wave.
            for future in physical_futures + gaussian_futures:
                future.cancel()
            for future in gaussian_futures:
                if not future.cancelled():
                    try:
                        future.result()
                    except BaseException:
                        pass
            for future in physical_futures:
                if not future.cancelled():
                    try:
                        partial = future.result()
                    except BaseException:
                        continue
                    partial.spectra.release_gaussian_inputs()
            for partial in partials:
                partial.spectra.release_gaussian_inputs()
            raise
        del physical_futures, gaussian_futures, partials, wave_cells
