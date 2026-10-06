"""Snapshot orchestration keeps slab boundaries, ordering, and progress intact."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from quokka2s import process_snapshot as processing


class FakeProducts:
    """Keep cell IDs so the scheduler can be checked without line calculations."""

    def __init__(self, *config, **kwargs):
        self.counts = {"all": 0}
        self.cell_ids = []
        self.failure_context = {}

    def merge(self, partial):
        self.counts["all"] += partial.counts["all"]
        self.cell_ids.extend(partial.cell_ids)


class FakeSlab:
    """Represent one slab; batches contain positions but no reference to the slab."""

    def __init__(self, cell_count, first_cell_id):
        self.cell_count = cell_count
        self.first_cell_id = first_cell_id

    def batch(self, start, stop):
        first_cell_id = self.first_cell_id + start
        return SimpleNamespace(
            first_cell_id=first_cell_id,
            last_cell_id=first_cell_id + stop - start - 1,
            cell_count=stop - start,
        )


class ProcessSnapshotFlowTests(unittest.TestCase):
    def run_small_snapshot(self, *, workers, max_slabs=None, fail_at=None,
                           check_release=False):
        """Use a 10 x 2 x 3 box: slabs contain 24, 24, and 12 cells."""
        config = SimpleNamespace(
            slab_nx=4,
            query_chunk=5,
            chunk_workers=workers,
            spectral_workers=1,
            max_slabs=max_slabs,
            output_dir=Path("unused-test-output"),
        )
        snapshot = SimpleNamespace(shape=(10, 2, 3))
        shared_calculator = SimpleNamespace(line_keys=())
        products = FakeProducts()
        windows_read = []
        progress_records = []
        slab_references = []
        failure = ValueError("test batch failed")

        def assert_slabs_released():
            for reference in slab_references:
                self.assertIsNone(reference(), "The previous slab is still retained")

        def read_slab(x_start, x_stop):
            window = (x_start, x_stop)
            # Every preceding slab must already be accumulated before this read.
            self.assertEqual(products.counts["all"], window[0] * 2 * 3)
            if check_release:
                assert_slabs_released()
            windows_read.append(window)
            slab = FakeSlab((window[1] - window[0]) * 2 * 3, first_cell_id=x_start*2*3)
            slab_references.append(weakref.ref(slab))
            return slab

        def accumulate(cells, emission_calculator, products):
            self.assertIs(emission_calculator, shared_calculator)
            products.failure_context = {
                "first_cell_id": cells.first_cell_id,
                "last_cell_id": cells.last_cell_id,
            }
            if cells.first_cell_id == fail_at:
                raise failure
            products.counts["all"] += cells.cell_count
            products.cell_ids.extend(
                range(cells.first_cell_id, cells.last_cell_id + 1)
            )

        def write_status(output_dir, counts, began, state, **progress):
            self.assertEqual(output_dir, config.output_dir)
            self.assertEqual(state, "running")
            self.assertEqual(began, 0.)
            finished_window = windows_read[-1]
            self.assertEqual(counts["all"], finished_window[1] * 2 * 3)
            if check_release:
                assert_slabs_released()
            progress_records.append(dict(cells=counts["all"], **progress))

        output = StringIO()
        caught = None
        real_executor = processing.ThreadPoolExecutor
        snapshot.read_slab = read_slab
        with patch.object(snapshot, "read_slab", side_effect=read_slab), \
                patch.object(processing, "accumulate_batch", side_effect=accumulate), \
                patch.object(processing, "EmissionProducts", side_effect=FakeProducts), \
                patch.object(processing, "write_status", side_effect=write_status), \
                patch.object(processing.time, "monotonic", side_effect=[10., 20., 30.]), \
                patch.object(processing, "ThreadPoolExecutor", wraps=real_executor) as pool, \
                redirect_stdout(output):
            try:
                processing.process_snapshot(
                    config=config,
                    snapshot=snapshot,
                    emission_calculator=shared_calculator,
                    products=products,
                    began=0.,
                )
            except ValueError as error:
                caught = error

        self.assertEqual(pool.call_count, 1, "One pool should be reused across slabs")
        pool.assert_called_once_with(max_workers=workers)
        if fail_at is None:
            self.assertIsNone(caught)
        else:
            self.assertIs(caught, failure)
        return products, windows_read, progress_records, output.getvalue()

    def test_serial_and_parallel_cover_partial_final_slab_without_overlap(self):
        for workers in (1, 2):
            with self.subTest(workers=workers):
                products, windows, progress, output = self.run_small_snapshot(
                    workers=workers,
                )
                self.assertEqual(products.cell_ids, list(range(60)))
                self.assertEqual(products.counts["all"], 60)
                self.assertEqual(
                    [(window[0], window[1]) for window in windows],
                    [(0, 4), (4, 8), (8, 10)],
                )
                self.assertEqual([entry["cells"] for entry in progress], [24, 48, 60])
                self.assertEqual([entry["completed_slabs"] for entry in progress], [1, 2, 3])
                self.assertEqual([entry["progress_percent"] for entry in progress], [40., 80., 100.])
                self.assertEqual([entry["eta_seconds"] for entry in progress], [15, 5, 0])
                self.assertEqual(output.splitlines(), [
                    "Emission: 40.0% (1/3 slabs); elapsed 0m 10s; ETA ~0m 15s",
                    "Emission: 80.0% (2/3 slabs); elapsed 0m 20s; ETA ~0m 5s",
                    "Emission: 100.0% (3/3 slabs); elapsed 0m 30s; ETA ~0m 0s",
                ])

    def test_max_slabs_limits_work_and_progress_but_passes_full_snapshot_to_reader(self):
        for workers in (1, 2):
            with self.subTest(workers=workers):
                products, windows, progress, output = self.run_small_snapshot(
                    workers=workers,
                    max_slabs=2,
                )
                self.assertEqual(products.cell_ids, list(range(48)))
                self.assertEqual(len(windows), 2)
                # The reader receives the original snapshot and core bounds only.
                # It can read x=8 as a neighbour even though this run ends at 8.
                self.assertEqual(windows[-1], (4, 8))
                self.assertEqual([entry["progress_percent"] for entry in progress], [50., 100.])
                self.assertEqual([entry["eta_seconds"] for entry in progress], [10, 0])
                self.assertIn("100.0% (2/2 slabs)", output)

    def test_max_slabs_above_available_slabs_uses_actual_box_size(self):
        products, windows, progress, output = self.run_small_snapshot(
            workers=2,
            max_slabs=99,
        )
        self.assertEqual(products.cell_ids, list(range(60)))
        self.assertEqual(len(windows), 3)
        self.assertEqual(progress[-1]["progress_percent"], 100.)
        self.assertIn("100.0% (3/3 slabs)", output)

    def test_serial_slab_is_released_before_progress_and_next_read(self):
        self.run_small_snapshot(workers=1, check_release=True)

    def test_failed_batch_propagates_without_reporting_or_reading_later_slabs(self):
        for workers in (1, 2):
            with self.subTest(workers=workers):
                products, windows, progress, output = self.run_small_snapshot(
                    workers=workers,
                    fail_at=24,
                )
                self.assertEqual(products.cell_ids, list(range(24)))
                self.assertEqual(len(windows), 2)
                self.assertEqual(len(progress), 1)
                self.assertEqual(len(output.splitlines()), 1)
                self.assertEqual(products.failure_context, {
                    "first_cell_id": 24,
                    "last_cell_id": 28,
                })


if __name__ == "__main__":
    unittest.main()
