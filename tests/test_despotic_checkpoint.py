"""Tiny mocked builds: resume completed successes/failures without solving again."""
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tempfile
import unittest

from quokka2s.tables import builder
from quokka2s.tables.checkpoint import PointCheckpoints
from quokka2s.tables.io import save_table
from quokka2s.tables.models import AttemptRecord, ExplicitGrid, LineLumResult


GRIDS = (ExplicitGrid((1.0, 2.0)), ExplicitGrid((10.0, 20.0)), ExplicitGrid((0.1,)))
METADATA = {"composition": {"xHe": 0.09296}, "solver": "mock-v1"}
CONTEXT = {"snapshot_domain": {"dataset": "test-snapshot"}}
SPECS = (builder.SpeciesSpec("CO", True), builder.SpeciesSpec("H2", False))


def _point(**kwargs):
    row, col, dvdr = (kwargs[name] for name in ("row_idx", "col_idx", "dvdr_idx"))
    failed = (row, col, dvdr) == (0, 1, 0)
    scalar = float("nan") if failed else float(10 * row + col + 1)
    attempt = AttemptRecord(
        row_idx=row, col_idx=col, dvdr_idx=dvdr,
        nH=kwargs["nH_val"], colDen=kwargs["colDen_val"], dvdr=kwargs["dvdr_val"],
        tg_guess=100.0, final_Tg=scalar, converged=not failed,
        message="mock failure" if failed else "mock success", duration=0.25,
    )
    # Multiple attempts must retain their ordering and original diagnostics.
    kwargs["attempt_log"].extend([replace(attempt, converged=False, message="first attempt"), attempt])
    line = LineLumResult(*(scalar + i for i in range(6)))
    return ({"CO": line, "CO21": line}, {"CO": scalar, "H2": scalar / 2},
            scalar, scalar * 2, scalar * 3, scalar * 4,
            {"z_rate": scalar * 5, "a_rate": scalar * 6}, failed)


def _build(directory=None, *, solver=_point, grids=GRIDS, context=CONTEXT,
           metadata=METADATA, source="mock-source-v1", workers=1):
    with (patch.object(builder, "validated_solver_metadata", return_value=metadata),
          patch.object(builder, "source_metadata", return_value={"mock": source}),
          patch.object(builder, "solve_gow_lvg_point", side_effect=solver)):
        return builder.build_gow_lvg_table(
            *grids, species_specs=SPECS, workers=workers, show_progress=False,
            checkpoint_dir=directory, checkpoint_context=context,
        )


def _interrupted_build(directory, ready, hold):
    count = 0

    def solve(**kwargs):
        nonlocal count
        count += 1
        if count == 3:
            ready.set()  # Two completed, durably committed points, third in flight.
            hold.wait(30)
            raise RuntimeError("test did not terminate the interrupted process")
        return _point(**kwargs)

    _build(directory, solver=solve)


def _assert_npz_equal(first, second, tmp_path):
    paths = [tmp_path / "first.npz", tmp_path / "second.npz"]
    for table, path in zip((first, second), paths):
        save_table(table, path)
    with np.load(paths[0], allow_pickle=True) as left, np.load(paths[1], allow_pickle=True) as right:
        assert left.files == right.files
        for name in left.files:
            if name == "attempts":
                for field in left[name].dtype.names:
                    np.testing.assert_equal(left[name][field], right[name][field])
            else:
                np.testing.assert_equal(left[name], right[name])


def _unexpected(**kwargs):
    raise AssertionError("completed or incompatible points must not be solved")


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "checkpoints"

    def test_real_process_termination_resumes_committed_success_and_failure(self):
        ctx = multiprocessing.get_context("spawn")
        ready, hold = ctx.Event(), ctx.Event()
        process = ctx.Process(target=_interrupted_build, args=(self.directory, ready, hold))
        process.start()
        try:
            self.assertTrue(ready.wait(15), "mock child did not reach the in-flight third point")
            process.terminate()
            process.join(5)
            self.assertFalse(process.is_alive())
            self.assertNotEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.kill()
                process.join(5)
        self.assertEqual(len(list(self.directory.glob("row-*/point-*.json"))), 2)
        (self.directory / "row-00000" / ".tmp-abandoned").write_text("incomplete")
        calls = []

        def remaining(**kwargs):
            calls.append((kwargs["row_idx"], kwargs["col_idx"], kwargs["dvdr_idx"]))
            return _point(**kwargs)

        resumed = _build(self.directory, solver=remaining)
        self.assertEqual(calls, [(1, 0, 0), (1, 1, 0)])
        self.assertTrue(resumed.failure_mask[0, 1, 0])
        self.assertTrue(np.isnan(resumed.tg_final[0, 1, 0]))
        self.assertEqual(len(resumed.attempts), 8)
        _assert_npz_equal(resumed, _build(), self.root)
        replay = _build(self.directory, solver=_unexpected)
        _assert_npz_equal(replay, resumed, self.root)

    def test_incompatible_checkpoint_refused_before_any_solver_call(self):
        _build(self.directory)
        for change in ("axis", "context", "composition", "source"):
            with self.subTest(change=change):
                options = {}
                if change == "axis":
                    options["grids"] = (ExplicitGrid((1.0, np.nextafter(2.0, 3.0))), *GRIDS[1:])
                elif change == "context":
                    options["context"] = {"snapshot_domain": {"dataset": "different-snapshot"}}
                elif change == "composition":
                    options["metadata"] = {**METADATA, "composition": {"xHe": 0.1}}
                else:
                    options["source"] = "different-source"
                with self.assertRaisesRegex(RuntimeError, "Incompatible"):
                    _build(self.directory, solver=_unexpected, **options)

    def test_corrupt_committed_record_is_an_error_not_a_cache_miss(self):
        for target in ("manifest", "point"):
            with self.subTest(target=target):
                directory = self.directory / target
                _build(directory)
                path = directory / "manifest.json" if target == "manifest" else next(directory.glob("row-*/point-*.json"))
                envelope = json.loads(path.read_text())
                envelope["payload"] += " "
                path.write_text(json.dumps(envelope))
                with self.assertRaisesRegex(RuntimeError, "Corrupt.*checksum"):
                    _build(directory, solver=_unexpected)

    def test_checkpoint_directory_excludes_a_second_writer(self):
        with PointCheckpoints.open(self.directory, {"identity": "same"}):
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                with PointCheckpoints.open(self.directory, {"identity": "same"}):
                    self.fail("second writer acquired lock")

    def test_worker_count_can_change_when_replaying_completed_table(self):
        original = _build(self.directory)
        replay = _build(self.directory, workers=2, solver=_unexpected)
        _assert_npz_equal(original, replay, self.root)


if __name__ == "__main__":
    unittest.main()
