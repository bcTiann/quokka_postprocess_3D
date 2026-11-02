"""Report real slab/batch stages without inspecting numerical arrays.

One reading thread owns SlabProgress. Batch workers update only small status
records; a separate thread reports them every ten seconds. A batch counts as
completed only after direct serial accumulation or an input-order merge.
"""
from __future__ import annotations

import logging
import sys
from threading import Event, Lock, Thread
from time import monotonic


READ_STAGE_LABELS = {
    'field_reading': 'field reading (native/cache + unit conversion)',
    'foreground_column': 'foreground column',
    'shielding_column': 'shielding column',
    'velocity_gradient': 'velocity gradient',
    'core_preparation_other': 'core preparation / other',
}


class SlabProgress:
    """Console progress for one slab, including time spent waiting on workers.

    cell_count counts core cells, excluding gradient neighbours. batch_size and
    workers describe the existing computation; this reporter does not schedule
    work. interval is the heartbeat period in seconds. stream/clock permit
    deterministic checks without reading a simulation or waiting ten seconds.
    Only integers, strings and timestamps are retained, never cell arrays.
    """

    def __init__(
        self,
        *,
        slab_number: int,
        total_slabs: int,
        cell_count: int,
        batch_size: int,
        workers: int,
        interval: float = 10.0,
        stream=None,
        clock=monotonic,
    ):
        self.label = f'Slab {slab_number}/{total_slabs}'
        self.cell_count = cell_count
        self.total_batches = (cell_count + batch_size - 1) // batch_size
        self.workers = workers
        self.interval = interval
        self.clock = clock
        self.stream = sys.stdout if stream is None else stream
        self.lock = Lock()
        self.stop_event = Event()
        self.thread = None
        self.active_batches = {}
        self.merged_batches = 0
        self.merged_cells = 0
        self.read_started = self.clock()
        self.read_stage = 'core_preparation_other'
        self.read_stage_started = self.read_started
        self.read_stage_seconds = {stage: 0.0 for stage in READ_STAGE_LABELS}
        self.compute_started = None
        self.read_seconds = None
        self.compute_seconds = None
        self.last_merge_report = None
        self.logger = logging.getLogger('quokka2s.process.progress')

    def __enter__(self):
        self.previous_level = self.logger.level
        self.previous_propagate = self.logger.propagate
        self.handler = logging.StreamHandler(self.stream)
        self.handler.setFormatter(logging.Formatter(
            fmt='%(asctime)s %(levelname)-7s %(message)s',
            datefmt='%H:%M:%S',
        ))
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.thread = Thread(
            target=self._report_while_waiting,
            name='quokka-progress',
            daemon=True,
        )
        self.thread.start()
        return self

    def __exit__(self, exception_type, exception, traceback):
        try:
            self.stop_event.set()
            self.thread.join()
            if exception_type is not None:
                self.logger.error('%s | interrupted by %s; no completion reported',
                                  self.label, exception_type.__name__)
            elif self.compute_started is not None and self.merged_batches == self.total_batches:
                self.compute_seconds = self.clock() - self.compute_started
                self.logger.info(
                    '%s | complete | read/derive %.1fs | compute + ordered merge %.1fs',
                    self.label,
                    self.read_seconds,
                    self.compute_seconds,
                )
            else:
                self.logger.warning('%s | ended with %d/%d batches completed',
                                    self.label, self.merged_batches, self.total_batches)
        finally:
            self.logger.removeHandler(self.handler)
            self.handler.close()
            self.logger.setLevel(self.previous_level)
            self.logger.propagate = self.previous_propagate

    def set_read_stage(self, stage: str):
        """Partition read_slab wall time at an observed numerical stage boundary.

        Repeated core-preparation segments accumulate under the same key. The
        initial/final call overhead is included there; all five stages together
        cover read/derive time without overlap, including progress-report overhead.
        """
        if stage not in READ_STAGE_LABELS:
            raise ValueError(f'Unknown slab read stage: {stage}')
        now = self.clock()
        with self.lock:
            if self.compute_started is not None:
                raise ValueError('Cannot change read stage after reading is complete')
            self.read_stage_seconds[self.read_stage] += now - self.read_stage_started
            self.read_stage = stage
            self.read_stage_started = now
        self.logger.info('%s | starting %s', self.label, READ_STAGE_LABELS[stage])

    def reading_complete(self):
        """Mark completion of read_slab, including its derived fields/cache work."""
        now = self.clock()
        with self.lock:
            self.read_stage_seconds[self.read_stage] += now - self.read_stage_started
            self.read_seconds = now - self.read_started
            self.compute_started = now
            stages = self.read_stage_seconds.copy()
        self.logger.info(
            '%s | read/derive finished in %.1fs | %d batches, up to %d concurrent',
            self.label, self.read_seconds, self.total_batches, self.workers,
        )
        self.logger.info(
            '%s | read/derive stages | field reading %.3fs | foreground column %.3fs | '
            'shielding column %.3fs | velocity gradient %.3fs | core/other %.3fs',
            self.label,
            stages['field_reading'],
            stages['foreground_column'],
            stages['shielding_column'],
            stages['velocity_gradient'],
            stages['core_preparation_other'],
        )

    def start_batch(self, batch_number: int):
        with self.lock:
            self.active_batches[batch_number] = {
                'stage': 'allocating batch accumulators',
                'stage_started': self.clock(),
            }

    def set_stage(self, batch_number: int, stage: str):
        """Record the actual call about to run; stage is a short English label."""
        with self.lock:
            self.active_batches[batch_number] = {
                'stage': stage,
                'stage_started': self.clock(),
            }

    def batch_ready(self, batch_number: int):
        self.set_stage(batch_number, 'computed; waiting for ordered merge')

    def batch_merged(self, batch_number: int, cell_count: int):
        """Advance after global accumulation (serial) or ordered merge (parallel)."""
        now = self.clock()
        with self.lock:
            self.active_batches.pop(batch_number, None)
            self.merged_batches += 1
            self.merged_cells += cell_count
            should_report = (
                self.last_merge_report is None
                or now - self.last_merge_report >= 2.0
                or self.merged_batches == self.total_batches
            )
            if should_report:
                self.last_merge_report = now
        if should_report:
            self.report(include_active=False)

    def report(self, *, include_active: bool = True):
        """Take a small locked snapshot, then format/write without holding it."""
        with self.lock:
            compute_started = self.compute_started
            merged_batches = self.merged_batches
            merged_cells = self.merged_cells
            read_stage = self.read_stage
            read_stage_started = self.read_stage_started
            active = {number: dict(state) for number, state in self.active_batches.items()}
        now = self.clock()
        if compute_started is None:
            self.logger.info(
                '%s | %s | stage elapsed %.1fs | read/derive elapsed %.1fs',
                self.label, READ_STAGE_LABELS[read_stage],
                now - read_stage_started, now - self.read_started,
            )
            return
        self.logger.info(
            '%s | completed %d/%d batches | %s/%s cells (%.1f%%) | compute elapsed %.1fs',
            self.label, merged_batches, self.total_batches,
            f'{merged_cells:,}', f'{self.cell_count:,}',
            100.0 * merged_cells / self.cell_count, now - compute_started,
        )
        if include_active:
            for number, state in sorted(active.items()):
                self.logger.info(
                    '  Batch %d/%d | %s | stage elapsed %.1fs',
                    number, self.total_batches, state['stage'], now - state['stage_started'],
                )

    def _report_while_waiting(self):
        while not self.stop_event.wait(self.interval):
            self.report()
