"""Format process-only console messages without changing numerical products."""
from __future__ import annotations

from contextlib import contextmanager
import logging
import math


def format_elapsed(seconds: float) -> str:
    """Format nonnegative wall seconds as MM:SS, or HH:MM:SS after one hour."""
    whole_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


@contextmanager
def quiet_yt_metadata():
    """Hide yt INFO only while process opens metadata; retain stricter levels.

    The caller includes both yt.load() and Snapshot construction in this scope.
    Always restore the exact original logger level, including after an error.
    Other tools retain their own yt logging because they do not use this scope.
    """
    from yt.funcs import mylog

    original_level = mylog.level
    mylog.setLevel(max(mylog.getEffectiveLevel(), logging.WARNING))
    try:
        yield
    finally:
        mylog.setLevel(original_level)


def process_stage_seconds(*, timings, process_seconds, checking_seconds, saving_seconds):
    """Return non-overlapping wall stages and the unassigned process remainder.

    All inputs are seconds. Read and Gaussian entries come from completed
    process receipts, not sums of worker-call times. Unit conversion and column
    calculation share one measured pass and therefore remain a single row.
    Input loading, reader setup and ordered merges use their existing measured
    intervals. The remainder covers orchestration, reporting and final status
    saving. No saved timing/report field is edited.
    """
    reads = timings["read_stage_totals_seconds"]
    gaussian = timings["gaussian"]
    stages = [
        ("Load inputs", timings["input_loading_seconds"]),
        ("Initialize field reader", timings["field_reader_initialization_seconds"]),
        ("Initialize Gaussian node store", gaussian["snapshot_corner_store_initialization_seconds"]),
        ("Read original fields", reads["original_fields_and_halos"]),
        ("Velocity gradients", reads["velocity_gradient"]),
        ("Units + column densities", reads["units_and_columns"]),
        ("Other field preparation", reads["derivation_other"]),
        ("Physical products", gaussian["physical_preparation_wave_seconds"]),
        (
            "Gaussian preparation + accumulation",
            gaussian["gaussian_wave_seconds_including_analytic_accounts_and_scheduling"],
        ),
        ("Merge batch products", gaussian["ordered_product_merge_wave_seconds"]),
        ("400-bin spectrum expansion", gaussian["snapshot_corner_finalization_seconds"]),
        (
            "Other output preparation",
            timings["output_construction_seconds"] - gaussian["snapshot_corner_finalization_seconds"],
        ),
        ("Output checks", checking_seconds),
        ("Save numerical products", saving_seconds),
    ]
    for name, seconds in [("Complete process", process_seconds), *stages]:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"{name} must have finite nonnegative wall seconds")
    remaining_seconds = process_seconds - sum(seconds for _, seconds in stages)
    if remaining_seconds < -1e-6:
        raise ValueError(
            f"Stage wall times overlap the complete process by {-remaining_seconds:.6f} s"
        )
    stages.append(("Other orchestration + reporting", max(0.0, remaining_seconds)))
    return stages


def print_process_summary(
    *, timings, process_seconds, checking_seconds, saving_seconds,
    peak_rss_bytes, gaussian_table_seconds, status,
):
    """Print a vertical console summary; template construction stays excluded.

    Invalid overlapping intervals are reported explicitly instead of inventing
    an additive breakdown or changing a successfully saved numerical status.
    """
    print(f"\nSummary: {status}", flush=True)
    print("  Wall time", flush=True)
    try:
        stages = process_stage_seconds(
            timings=timings,
            process_seconds=process_seconds,
            checking_seconds=checking_seconds,
            saving_seconds=saving_seconds,
        )
    except ValueError as error:
        print(f"  Stage breakdown unavailable: {error}", flush=True)
    else:
        for name, seconds in stages:
            print(f"  {name:<38} {seconds:>10.3f} s", flush=True)
    print(
        f"  {'Complete process':<38} {format_elapsed(process_seconds)} "
        f"({process_seconds:.3f} s)",
        flush=True,
    )
    print(f"  {'Peak process RSS':<38} {peak_rss_bytes / 1e9:>5.2f} GB", flush=True)
    print(
        f"  {'Gaussian templates (excluded)':<38} {gaussian_table_seconds:>10.3f} s",
        flush=True,
    )
