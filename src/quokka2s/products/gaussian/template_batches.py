"""Prepare physical batches, accumulate original corners, then merge products."""
from __future__ import annotations

from itertools import islice
from time import perf_counter_ns


def process_template_batches(
    *, config, snapshot, emission_calculator, products, slab, executor,
    accumulate_batch, after_wave=None,
):
    """Keep at most chunk_workers Gaussian input queues within the current slab.

    The ordinary accumulate_batch function computes the same physical products,
    available-line grouping and complete moments. Workers then prepare their
    original-corner increments, endpoint accounts and outside-width profiles.
    The wave accumulates into disjoint H profile planes before small products
    merge in the current slab's batch order. Native
    tiles are visited in their deterministic XY order; sums across tiles may
    therefore differ in floating-point rounding from a full-grid batch walk.
    A one-worker configuration follows this same path. after_wave optionally
    checks process memory after each bounded worker wave completes.
    """
    from ..emission_products import EmissionProducts

    batches = iter(slab.iter_batches(batch_size=config.query_chunk))
    while True:
        wave_cells = list(islice(batches, config.chunk_workers))
        if not wave_cells:
            break
        physical_started = perf_counter_ns()
        physical_futures = []
        gaussian_futures = []
        partials = []

        def prepare_one(cells):
            partial = EmissionProducts(
                gaussian_engine=products.spectra.engine,
                batch_partial=True,
                snapshot=snapshot,
                line_keys=emission_calculator.line_keys,
            )
            try:
                accumulate_batch(
                    cells=cells,
                    emission_calculator=emission_calculator,
                    products=partial,
                )
            except BaseException:
                partial.spectra.release_gaussian_inputs()
                raise
            return partial

        try:
            for cells in wave_cells:
                physical_futures.append(executor.submit(prepare_one, cells))
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
            for partial in partials:
                gaussian_futures.append(executor.submit(partial.spectra.prepare_corner_increments))
            for cells, future in zip(wave_cells, gaussian_futures):
                products.failure_context = {
                    "first_cell_id": cells.first_cell_id,
                    "last_cell_id": cells.last_cell_id,
                }
                future.result()
            gaussian_workers_finished = perf_counter_ns()
            finished_groups = sum(partial.spectra._groups_finished for partial in partials)
            if prepared_groups != finished_groups:
                raise AssertionError("A Gaussian wave omitted one or more prepared groups")
            if any(partial.spectra._groups for partial in partials):
                raise AssertionError("A prepared Gaussian wave retained its coordinate queues")
            products.spectra.scatter_corner_wave(
                partials,
                executor=executor,
                workers=config.chunk_workers,
            )
            gaussian_finished = perf_counter_ns()
            merge_started = perf_counter_ns()
            for cells, partial in zip(wave_cells, partials):
                products.failure_context = {
                    "first_cell_id": cells.first_cell_id,
                    "last_cell_id": cells.last_cell_id,
                }
                products.merge(partial)
            merge_finished = perf_counter_ns()
            products.spectra.record_completed_wave(
                physical_ns=physical_finished - physical_started,
                gaussian_ns=gaussian_finished - gaussian_started,
                gaussian_worker_ns=gaussian_workers_finished - gaussian_started,
                merge_ns=merge_finished - merge_started,
                owned_input_bytes=owned_input_bytes,
            )
            if after_wave is not None:
                after_wave()
        except BaseException:
            # Wait only for this bounded wave before releasing every surviving
            # partial; no failed task is resubmitted or reused in another wave.
            products.spectra._corner_failed = True
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
