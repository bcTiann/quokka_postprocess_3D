"""Reuse or measure one snapshot's physical inputs with read-only source access."""
from __future__ import annotations

from pathlib import Path
import time

from quokka2s.processing_inputs import open_snapshot, print_snapshot_summary
from quokka2s.table_input_files import TableInputFiles, snapshot_file_state
from quokka2s.table_input_statistics import SnapshotInputSettings, scan_snapshot_input_ranges


class SnapshotInputSurvey:
    """Own one snapshot's source metadata and guarded result-file locations.

    config names a snapshot, batch size and optional native-column limit.
    No cell data or emission tables are loaded at construction. Completed
    statistics can be checked in the coordinator before a separate worker
    opens the snapshot.
    """

    def __init__(
        self,
        *,
        config: SnapshotInputSettings,
        output_directory: Path,
        protected_source_roots: tuple[Path, ...] = (),
        allowed_output_root: Path | None = None,
    ):
        self.config = config
        self.source_state = snapshot_file_state(dataset=config.dataset)
        protected_roots = {config.dataset.parent.resolve()}
        protected_roots.update(root.resolve() for root in protected_source_roots)
        protected_roots.update(Path(record["path"]).parent for record in self.source_state)
        self.files = TableInputFiles(
            directory=output_directory,
            protected_roots=tuple(protected_roots),
            allowed_output_root=allowed_output_root,
        )
        self.requested_region = {
            "x": config.x_index_range,
            "y": config.y_index_range,
        }

    def read_completed_statistics(self) -> dict | None:
        """Read a current complete JSON record, or return None without cell I/O."""
        return self.files.read_completed_statistics(
            dataset=self.config.dataset,
            source_state=self.source_state,
            xy_region=self.requested_region,
        )

    def measure_input_ranges(self) -> dict:
        """Read this snapshot, calculate scalar ranges, and save its own JSON.

        Cell arrays exist only for the current native z layer; small boundary
        carries retain the complete-column sums. The reader opens source files
        read-only; source sizes/mtimes are checked again before saving.
        No table comparison, CSV or shared summary is written here.
        """
        from yt.config import ytcfg

        if ytcfg["yt", "store_parameter_files"]:
            raise ValueError("Disable yt's store_parameter_files setting before running a survey")
        snapshot = open_snapshot(
            dataset_path=self.config.dataset,
            xy_region=self.requested_region,
        )
        print_snapshot_summary(snapshot=snapshot)
        started = time.monotonic()
        statistics = scan_snapshot_input_ranges(
            snapshot=snapshot,
            query_chunk=self.config.query_chunk,
            max_columns=self.config.max_columns,
            progress_label=f"{self.config.jobname}/{self.config.snapshot_name}",
        )
        if snapshot_file_state(dataset=self.config.dataset) != self.source_state:
            raise ValueError("Snapshot files changed while scanning; statistics were not saved")
        statistics.update({
            "source": str(self.config.dataset.resolve()),
            "jobname": self.config.jobname,
            "snapshot_name": self.config.snapshot_name,
            "source_file_state": self.source_state,
            "requested_xy_region": {
                axis: list(bounds) if bounds is not None else None
                for axis, bounds in self.requested_region.items()
            },
            "elapsed_seconds": time.monotonic() - started,
        })
        self.files.write_json(path=self.files.statistics_path, values=statistics)
        return statistics


def load_or_measure_input_ranges(
    *,
    config: SnapshotInputSettings,
    output_directory: Path,
    protected_source_roots: tuple[Path, ...] = (),
    allowed_output_root: Path | None = None,
) -> tuple[dict, TableInputFiles]:
    """Reuse complete physical statistics, otherwise read this snapshot once.

    Single-snapshot and worker commands share this action. Tables are absent:
    the coordinator compares current table axes after the ranges are available.
    """
    survey = SnapshotInputSurvey(
        config=config,
        output_directory=output_directory,
        protected_source_roots=protected_source_roots,
        allowed_output_root=allowed_output_root,
    )
    statistics = survey.read_completed_statistics()
    if statistics is not None:
        print(f"Reusing completed input statistics: {survey.files.statistics_path}", flush=True)
    else:
        statistics = survey.measure_input_ranges()
    return statistics, survey.files
