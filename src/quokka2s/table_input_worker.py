"""Scan one snapshot in a process; return ranges and small resource counters."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import signal
import sys
import time

from quokka2s.table_input_statistics import SnapshotInputSettings
from quokka2s.table_input_survey import load_or_measure_input_ranges


def finish_snapshot_on_interrupt() -> None:
    """Let the coordinator handle Ctrl-C while this worker finishes its scan."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def scan_snapshot_in_worker(
    *,
    config: SnapshotInputSettings,
    output_directory: Path,
    protected_source_roots: tuple[Path, ...],
    allowed_output_root: Path,
) -> dict:
    """Read one snapshot and return scalar statistics plus process measurements.

    Inputs are settings and Paths, never a yt dataset or table lookup object.
    This worker saves only its own input_ranges.json. Peak memory is the
    process high-water RSS; a reused worker's peak includes its earlier scans.
    Example: result['execution']['wall_seconds'] measures this snapshot call.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_SELF)
    statistics, _ = load_or_measure_input_ranges(
        config=config,
        output_directory=output_directory,
        protected_source_roots=protected_source_roots,
        allowed_output_root=allowed_output_root,
    )
    after = resource.getrusage(resource.RUSAGE_SELF)
    peak_memory_bytes = after.ru_maxrss
    if sys.platform != "darwin":
        peak_memory_bytes *= 1024
    return {
        "statistics": statistics,
        "execution": {
            "process_id": os.getpid(),
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.monotonic() - started,
            "user_cpu_seconds": after.ru_utime - before.ru_utime,
            "system_cpu_seconds": after.ru_stime - before.ru_stime,
            "process_peak_memory_bytes": int(peak_memory_bytes),
        },
    }
