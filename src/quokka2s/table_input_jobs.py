"""Select simulation jobs and run the shared snapshot table-input survey."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
import math
import multiprocessing
import os
import re
import signal
import time

import yaml

from quokka2s.cloudy.lookup import CloudyLookup
from quokka2s.despotic.lookup import DespoticLookup
from quokka2s.table_input_survey import SnapshotInputSurvey, load_or_measure_input_ranges
from quokka2s.table_input_worker import finish_snapshot_on_interrupt, scan_snapshot_in_worker
from quokka2s.input_paths import DEFAULT_CLOUDY_TABLE, DEFAULT_DESPOTIC_TABLE
from quokka2s.paths import resolve_path
from quokka2s.table_input_files import (
    StaleInputStatistics,
    SurveyOutputDirectory,
    TableInputFiles,
    UnsafeSurveyOutput,
    snapshot_source_paths,
)
from quokka2s.table_input_range_check import build_table_range_report
from quokka2s.table_input_statistics import SnapshotInputSettings, TableInputStatistics


@dataclass(frozen=True)
class JobInputSurveySettings:
    """Resolved batch paths and the shared sizes used by each snapshot scan.

    jobs maps output job names to absolute run directories inside source_root.
    output_dir is a separate user-owned root. All snapshots retain full x/y/z;
    tile_ny sets y-strip width with full selected x and z. max_tiles is only
    a per-snapshot diagnostic limit, never an accepted scan.
    snapshot_workers limits concurrent readers; max_new_snapshots excludes
    completed caches. max_runtime_minutes stops new dispatch, not active reads.
    """

    source_root: Path
    jobs: dict[str, Path]
    output_dir: Path
    despotic_table: Path = DEFAULT_DESPOTIC_TABLE
    cloudy_table: Path = DEFAULT_CLOUDY_TABLE
    tile_ny: int = 8
    query_chunk: int = 1_000_000
    max_tiles: int | None = None
    snapshot_workers: int = 1
    max_new_snapshots: int | None = None
    max_runtime_minutes: float | None = None


def read_job_survey_config(*, path: Path) -> JobInputSurveySettings:
    """Read source_root/jobs/output_dir; resolve relative paths from this YAML.

    jobs is a nonempty mapping, e.g. sigma13: job41101509-sigma13/run1.
    Mapping keys are single output-folder names. Unknown keys are rejected at
    this input boundary; no plot/dust/emission settings enter the batch scan.
    """
    config_path = resolve_path(path)
    with config_path.open("r") as source:
        supplied = yaml.safe_load(source)
    required = {"source_root", "jobs", "output_dir"}
    optional = {
        "despotic_table", "cloudy_table", "tile_ny", "query_chunk", "max_tiles",
        "snapshot_workers", "max_new_snapshots", "max_runtime_minutes",
    }
    if not isinstance(supplied, dict) or not required.issubset(supplied):
        raise ValueError("Survey config requires source_root, jobs and output_dir")
    unknown = set(supplied) - required - optional
    if unknown:
        raise ValueError(f"Unknown survey setting(s): {', '.join(sorted(unknown))}")
    for name in ("source_root", "output_dir", "despotic_table", "cloudy_table"):
        if name in supplied and (not isinstance(supplied[name], str) or not supplied[name].strip()):
            raise ValueError(f"{name} must be a nonempty path string")
    source_root = resolve_path(supplied["source_root"], base_directory=config_path.parent)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Survey source_root: {source_root}")
    configured_jobs = supplied["jobs"]
    if not isinstance(configured_jobs, dict) or not configured_jobs:
        raise ValueError("jobs must map at least one job name to its run directory")
    jobs = {}
    for jobname, run_path in configured_jobs.items():
        if (
            not isinstance(jobname, str) or not jobname.strip()
            or jobname in (".", "..") or "/" in jobname or "\\" in jobname
        ):
            raise ValueError("Each job name must be a nonempty single folder name")
        if not isinstance(run_path, str) or not run_path.strip():
            raise ValueError(f"Run directory must be a nonempty path string: {jobname}")
        run_directory = resolve_path(run_path, base_directory=source_root)
        if not run_directory.is_relative_to(source_root):
            raise ValueError(f"Job directory leaves source_root: {jobname}")
        if not run_directory.is_dir():
            raise FileNotFoundError(f"Job directory: {run_directory}")
        jobs[jobname] = run_directory

    sizes = {
        "tile_ny": 8,
        "query_chunk": 1_000_000,
        "max_tiles": None,
        "snapshot_workers": 1,
        "max_new_snapshots": None,
    }
    for name, default in sizes.items():
        sizes[name] = supplied.get(name, default)
        if name in ("max_tiles", "max_new_snapshots") and sizes[name] is None:
            continue
        if type(sizes[name]) is not int or sizes[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if sizes["query_chunk"] > 1_000_000:
        raise ValueError("query_chunk must be at most 1000000 cells")
    runtime_minutes = supplied.get("max_runtime_minutes")
    if runtime_minutes is not None:
        if (
            isinstance(runtime_minutes, bool)
            or not isinstance(runtime_minutes, (int, float))
            or not math.isfinite(runtime_minutes)
            or runtime_minutes <= 0
        ):
            raise ValueError("max_runtime_minutes must be a positive finite number")
    return JobInputSurveySettings(
        source_root=source_root,
        jobs=jobs,
        output_dir=resolve_path(supplied["output_dir"], base_directory=config_path.parent),
        despotic_table=resolve_path(supplied.get("despotic_table", DEFAULT_DESPOTIC_TABLE), base_directory=config_path.parent),
        cloudy_table=resolve_path(supplied.get("cloudy_table", DEFAULT_CLOUDY_TABLE), base_directory=config_path.parent),
        max_runtime_minutes=runtime_minutes,
        **sizes,
    )


def discover_job_snapshots(*, settings: JobInputSurveySettings) -> dict[str, list[SnapshotInputSettings]]:
    """List each run's direct plt+digits directories, in numeric snapshot order.

    Header/payload completeness is checked when scanning, not assumed from the
    directory name. Snapshot symlinks must stay inside the common source_root.
    An empty run remains in the inventory and makes the survey incomplete.
    """
    jobs = {}
    for jobname, run_directory in settings.jobs.items():
        candidates = []
        for path in run_directory.iterdir():
            if not re.fullmatch(r"plt\d+", path.name):
                continue
            if path.is_symlink() and not path.exists():
                raise FileNotFoundError(f"Broken snapshot symlink: {path}")
            if path.is_dir():
                candidates.append(path)
        candidates.sort(key=lambda path: int(path.name[3:]))
        snapshots = []
        for path in candidates:
            resolved = path.resolve()
            if not resolved.is_relative_to(settings.source_root):
                raise ValueError(f"Snapshot symlink leaves source_root: {path}")
            snapshots.append(SnapshotInputSettings(
                dataset=resolved,
                jobname=jobname,
                snapshot_name=path.name,
                tile_ny=settings.tile_ny,
                query_chunk=settings.query_chunk,
                max_tiles=settings.max_tiles,
            ))
        jobs[jobname] = snapshots
    return jobs


def check_batch_output_paths(
    *,
    settings: JobInputSurveySettings,
    jobs: dict[str, list[SnapshotInputSettings]],
) -> SurveyOutputDirectory:
    """Check every destination before scanning, including snapshot-dir symlinks.

    The entire common source_root is protected, not just the selected job.
    Per-snapshot writers also protect any resolved external input-file locations
    when they load source metadata. This stage creates no output directories.
    """
    protected_roots = {settings.source_root}
    for snapshots in jobs.values():
        for selection in snapshots:
            for path in snapshot_source_paths(dataset=selection.dataset):
                protected_roots.add(path.resolve().parent)
    output = SurveyOutputDirectory(
        directory=settings.output_dir,
        protected_roots=tuple(protected_roots),
    )
    for name in ("snapshots.csv", "jobs.csv", "summary.json"):
        output.check_output_path(path=output.directory / name)
    for snapshots in jobs.values():
        for selection in snapshots:
            directory = settings.output_dir / "snapshots" / selection.jobname / selection.snapshot_name
            TableInputFiles(
                directory=directory,
                protected_roots=tuple(protected_roots),
                allowed_output_root=settings.output_dir,
            )
    return output


class SnapshotSurveyCoordinator:
    """Arrange snapshot scans and own all table reports and shared summaries.

    Only small records and Paths belong to this object. Spawned workers own
    their yt datasets and slab arrays. records remains in inventory order even
    when workers finish in another order, preserving the summary merge order.
    """

    def __init__(
        self,
        *,
        settings: JobInputSurveySettings,
        despotic: DespoticLookup,
        cloudy: CloudyLookup,
        max_new_snapshots: int | None,
    ):
        self.settings = settings
        self.despotic = despotic
        self.cloudy = cloudy
        self.max_new_snapshots = max_new_snapshots
        self.started = time.monotonic()
        self.records = []  # One scalar record per snapshot, in inventory order.
        self.selections = []
        self.surveys = {}
        self.snapshots_to_scan = []
        self.next_snapshot = 0  # Next entry in snapshots_to_scan, not a global cell index.
        self.new_scans_started = 0
        self.active_scans = {}  # {Future: inventory index}, at most snapshot_workers entries.
        self.stop_reason = None

    def prepare_snapshot_records(
        self,
        *,
        jobs: dict[str, list[SnapshotInputSettings]],
        max_snapshots: int | None,
    ) -> None:
        """Check all selected caches before opening any snapshot for new scans.

        Complete caches immediately enter the report and do not consume the
        new-scan budget. Missing/partial records enter snapshots_to_scan.
        max_snapshots retains its explicit first-N-inventory selection meaning.
        """
        for snapshots in jobs.values():
            self.selections.extend(snapshots)
        for index, selection in enumerate(self.selections):
            self.records.append({
                "jobname": selection.jobname,
                "snapshot_name": selection.snapshot_name,
                "source": str(selection.dataset),
                "status": "not_selected",
            })
            if max_snapshots is not None and index >= max_snapshots:
                continue
            directory = (
                self.settings.output_dir / "snapshots"
                / selection.jobname / selection.snapshot_name
            )
            try:
                survey = SnapshotInputSurvey(
                    config=selection,
                    output_directory=directory,
                    protected_source_roots=(self.settings.source_root,),
                    allowed_output_root=self.settings.output_dir,
                )
                self.surveys[index] = survey
                statistics = survey.read_completed_statistics()
            except (UnsafeSurveyOutput, StaleInputStatistics):
                raise
            except (OSError, ValueError, KeyError) as error:
                self.record_failure(index=index, error=error)
                continue
            if statistics is None:
                self.snapshots_to_scan.append(index)
            else:
                print(f"Reusing completed snapshot: {selection.jobname}/{selection.snapshot_name}", flush=True)
                self.record_statistics(index=index, statistics=statistics)
        print(
            f"Scan session: {self.settings.snapshot_workers} worker(s); "
            f"{len(self.snapshots_to_scan)} unfinished snapshots; "
            f"new-scan limit={self.max_new_snapshots}; "
            f"dispatch budget={self.settings.max_runtime_minutes} minutes",
            flush=True,
        )

    def may_start_snapshot(self) -> bool:
        """Check count/time limits before each launch, without cancelling active work."""
        if self.stop_reason is not None:
            return False
        if self.max_new_snapshots is not None:
            if self.new_scans_started >= self.max_new_snapshots:
                self.stop_reason = "new_snapshot_limit"
                return False
        if self.settings.max_runtime_minutes is not None:
            elapsed_minutes = (time.monotonic() - self.started) / 60.0
            if elapsed_minutes >= self.settings.max_runtime_minutes:
                self.stop_reason = "runtime_limit"
                print("Time budget reached; finishing active snapshots without starting more.", flush=True)
                return False
        return True

    def announce_snapshot_start(self, *, index: int) -> None:
        """Show the inventory position and count this newly dispatched scan."""
        selection = self.selections[index]
        self.new_scans_started += 1
        print(
            f"Starting snapshot {index + 1}/{len(self.selections)}: "
            f"{selection.jobname}/{selection.snapshot_name}",
            flush=True,
        )

    def record_statistics(
        self,
        *,
        index: int,
        statistics: dict,
        execution: dict | None = None,
    ) -> None:
        """Compare current table axes in the parent; write this snapshot's report."""
        report = build_table_range_report(
            statistics=statistics,
            despotic=self.despotic,
            cloudy=self.cloudy,
        )
        report["tables"] = {
            "despotic": str(self.settings.despotic_table),
            "cloudy": str(self.settings.cloudy_table),
        }
        if execution is not None:
            report["execution"] = execution
            self.records[index]["execution"] = execution
        files = self.surveys[index].files
        files.write_json(path=files.report_path, values=report)
        self.records[index]["statistics"] = statistics
        self.records[index]["table_range_report"] = report
        self.records[index]["status"] = "complete" if statistics["full_snapshot"] else "partial"
        selection = self.selections[index]
        print(f"Snapshot ready: {selection.jobname}/{selection.snapshot_name}", flush=True)

    def record_failure(self, *, index: int, error: Exception) -> None:
        """Keep an ordinary source-read failure visible in the final inventory."""
        message = f"{type(error).__name__}: {error}"
        self.records[index]["status"] = "failed"
        self.records[index]["error"] = message
        selection = self.selections[index]
        print(f"Snapshot failed: {selection.jobname}/{selection.snapshot_name}: {message}", flush=True)

    def scan_serially(self) -> None:
        """Use the same scan action directly when only one worker is requested."""
        for index in self.snapshots_to_scan:
            if not self.may_start_snapshot():
                break
            self.announce_snapshot_start(index=index)
            try:
                statistics, _ = load_or_measure_input_ranges(
                    config=self.selections[index],
                    output_directory=self.surveys[index].files.directory,
                    protected_source_roots=(self.settings.source_root,),
                    allowed_output_root=self.settings.output_dir,
                )
            except (UnsafeSurveyOutput, StaleInputStatistics):
                raise
            except (OSError, ValueError, KeyError) as error:
                self.record_failure(index=index, error=error)
                continue
            except KeyboardInterrupt:
                self.stop_reason = "interrupted"
                print("Stopped; completed snapshots remain available for the next run.", flush=True)
                break
            self.record_statistics(index=index, statistics=statistics)

    def dispatch_available_snapshots(self, *, executor: ProcessPoolExecutor) -> None:
        """Submit at most one task per free worker slot; retain no backlog."""
        while len(self.active_scans) < self.settings.snapshot_workers:
            if self.next_snapshot >= len(self.snapshots_to_scan):
                return
            if not self.may_start_snapshot():
                return
            index = self.snapshots_to_scan[self.next_snapshot]
            self.announce_snapshot_start(index=index)
            future = executor.submit(
                scan_snapshot_in_worker,
                config=self.selections[index],
                output_directory=self.surveys[index].files.directory,
                protected_source_roots=(self.settings.source_root,),
                allowed_output_root=self.settings.output_dir,
            )
            self.active_scans[future] = index
            self.next_snapshot += 1

    def collect_completed_scans(self, *, completed: set[Future]) -> None:
        """Resolve every finished worker before another snapshot can be launched.

        Safety, stale-cache, worker-crash and output-write errors stop dispatch.
        Source-read failures become failed rows. No table or yt object crosses
        the process boundary; the returned dictionaries contain scalars only.
        """
        completed = sorted(completed, key=lambda future: self.active_scans[future])
        results = []
        for future in completed:
            index = self.active_scans[future]
            try:
                result = future.result()
            except (UnsafeSurveyOutput, StaleInputStatistics):
                raise
            except (OSError, ValueError, KeyError) as error:
                self.record_failure(index=index, error=error)
                del self.active_scans[future]
                continue
            results.append((future, index, result))
        for future, index, result in results:
            self.record_statistics(
                index=index,
                statistics=result["statistics"],
                execution=result["execution"],
            )
            del self.active_scans[future]

    def request_graceful_stop(self, signal_number, stack_frame) -> None:
        """Handle Ctrl-C by stopping dispatch without interrupting result writes."""
        self.stop_reason = "interrupted"

    def scan_in_parallel(self) -> None:
        """Spawn independent readers, drain active work at the time/count boundary."""
        # Children must see these limits before importing NumPy/SciPy.
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        previous_handler = signal.signal(signal.SIGINT, self.request_graceful_stop)
        try:
            with ProcessPoolExecutor(
                max_workers=self.settings.snapshot_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=finish_snapshot_on_interrupt,
            ) as executor:
                while True:
                    self.dispatch_available_snapshots(executor=executor)
                    if not self.active_scans:
                        break
                    completed, _ = wait(self.active_scans, return_when=FIRST_COMPLETED)
                    self.collect_completed_scans(completed=completed)
        finally:
            signal.signal(signal.SIGINT, previous_handler)
        if self.stop_reason == "interrupted":
            print("Stopped after active snapshots finished; completed records are saved.", flush=True)

    def finish_records(self) -> list[dict]:
        """Label unstarted entries and return records in the original inventory order."""
        for index in self.snapshots_to_scan:
            if self.records[index]["status"] == "not_selected":
                self.records[index]["not_scanned_reason"] = self.stop_reason
        return self.records


def scan_job_snapshots(
    *,
    settings: JobInputSurveySettings,
    jobs: dict[str, list[SnapshotInputSettings]],
    despotic: DespoticLookup,
    cloudy: CloudyLookup,
    max_snapshots: int | None = None,
    max_new_snapshots: int | None = None,
) -> list[dict]:
    """Reuse current completed scans; scan remaining snapshots within this session.

    max_snapshots selects an inventory prefix. max_new_snapshots overrides the
    YAML new-scan limit and excludes completed caches from that count. Time
    budget stops new dispatch only; already-started snapshots finish and save.
    """
    if max_new_snapshots is None:
        max_new_snapshots = settings.max_new_snapshots
    coordinator = SnapshotSurveyCoordinator(
        settings=settings,
        despotic=despotic,
        cloudy=cloudy,
        max_new_snapshots=max_new_snapshots,
    )
    coordinator.prepare_snapshot_records(jobs=jobs, max_snapshots=max_snapshots)
    if settings.snapshot_workers == 1:
        coordinator.scan_serially()
    elif coordinator.snapshots_to_scan:
        coordinator.scan_in_parallel()
    return coordinator.finish_records()


def summarize_snapshot_records(
    *,
    records: list[dict],
    despotic: DespoticLookup,
    cloudy: CloudyLookup,
) -> dict:
    """Merge full successes only; retain failed/partial/unselected counts.

    records comes from the current inventory, never a recursive cache search.
    A range-covered result is false for an empty or incomplete scope. This
    comparison certifies coordinate ranges only, not remaining table holes.
    """
    statistics = TableInputStatistics()
    status_counts = Counter(record["status"] for record in records)
    for record in records:
        if record["status"] == "complete":
            statistics.merge_completed_snapshot(statistics=record["statistics"])
    ranges = statistics.build_ranges()
    complete = bool(records) and status_counts["complete"] == len(records)
    measured = {
        "scan_status": "complete" if complete else "partial",
        "full_snapshot": complete,
        **ranges,
    }
    report = build_table_range_report(statistics=measured, despotic=despotic, cloudy=cloudy)
    return {
        "snapshots_discovered": len(records),
        "snapshot_status_counts": dict(status_counts),
        "survey_complete": complete,
        "ranges": ranges,
        "table_range_report": report,
    }


def build_job_survey_summary(
    *,
    settings: JobInputSurveySettings,
    records: list[dict],
    despotic: DespoticLookup,
    cloudy: CloudyLookup,
) -> dict:
    """Build job and global summaries from this run's explicit snapshot inventory."""
    job_summaries = []
    for jobname in settings.jobs:
        job_records = [record for record in records if record["jobname"] == jobname]
        summary = summarize_snapshot_records(records=job_records, despotic=despotic, cloudy=cloudy)
        job_summaries.append({"jobname": jobname, **summary})
    overall = summarize_snapshot_records(records=records, despotic=despotic, cloudy=cloudy)
    if any(not job["survey_complete"] for job in job_summaries):
        overall["survey_complete"] = False
        overall["table_range_report"]["scan_status"] = "partial"
        overall["table_range_report"]["full_snapshot"] = False
        overall["table_range_report"]["selected_input_ranges_covered"] = False
    return {
        "check": "table input coordinate ranges only",
        "source_root": str(settings.source_root),
        "output_dir": str(settings.output_dir),
        "tables": {"despotic": str(settings.despotic_table), "cloudy": str(settings.cloudy_table)},
        "overall": overall,
        "jobs": job_summaries,
        "snapshots": records,
    }
