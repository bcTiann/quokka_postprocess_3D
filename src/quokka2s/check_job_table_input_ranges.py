"""Survey all snapshots in explicitly configured jobs and save range summaries."""
from __future__ import annotations

import argparse
import csv
from io import StringIO
from pathlib import Path

from quokka2s.cloudy.lookup import CloudyLookup
from quokka2s.processing_inputs import load_despotic_lookup
from quokka2s.table_input_jobs import (
    build_job_survey_summary,
    check_batch_output_paths,
    discover_job_snapshots,
    read_job_survey_config,
    scan_job_snapshots,
)


DEFAULT_SURVEY_CONFIG = Path("configs/table_input_survey.yaml")


def range_csv_text(*, rows: list[dict], summary_rows: bool = False) -> str:
    """Prepare readable snapshot/job rows using the already-measured ranges.

    Snapshot rows retain explicit scan status/error. Job rows retain completeness
    and snapshot counts. Six physical axes have min/max/unit columns; Cloudy
    physical NH is shown, not substituted with its clipped lookup coordinate.
    """
    columns = [
        "jobname", "snapshot_name", "status", "error", "source",
        "survey_complete", "snapshots_discovered", "snapshots_complete",
        "snapshots_partial", "snapshots_failed", "snapshots_not_selected",
        "shape", "full_snapshot", "cell_count", "hot_cell_count",
        "elapsed_seconds", "input_ranges_covered",
    ]
    axes = {
        "despotic": ("nH", "NH", "dVdr"),
        "cloudy": ("nH", "NH", "T_QUOKKA"),
    }
    for table_name, names in axes.items():
        for name in names:
            columns.extend((
                f"{table_name}_{name}_min",
                f"{table_name}_{name}_max",
                f"{table_name}_{name}_unit",
            ))
    text = StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=columns)
    writer.writeheader()
    for record in rows:
        row = {name: record[name] for name in columns if name in record}
        statistics = record.get("statistics")
        if summary_rows:
            statistics = record["ranges"]
            for status in ("complete", "partial", "failed", "not_selected"):
                row[f"snapshots_{status}"] = record["snapshot_status_counts"].get(status, 0)
        report = record.get("table_range_report")
        if report is not None:
            row["input_ranges_covered"] = report["selected_input_ranges_covered"]
        if statistics is not None:
            for name in ("cell_count", "hot_cell_count", "elapsed_seconds"):
                if name in statistics:
                    row[name] = statistics[name]
            if "native_shape" in statistics:
                row["shape"] = "x".join(str(size) for size in statistics["native_shape"])
                row["full_snapshot"] = statistics["full_snapshot"]
            for table_name, names in axes.items():
                for name in names:
                    measured = statistics[table_name][name]
                    row[f"{table_name}_{name}_min"] = measured["minimum"]
                    row[f"{table_name}_{name}_max"] = measured["maximum"]
                    row[f"{table_name}_{name}_unit"] = measured["unit"]
        writer.writerow(row)
    return text.getvalue()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SURVEY_CONFIG)
    parser.add_argument("--max-snapshots", type=int, help="Scan only the first N inventory entries for a diagnostic invocation")
    parser.add_argument("--max-new-snapshots", type=int, help="Limit newly scanned snapshots; complete caches do not count")
    args = parser.parse_args(argv)
    if args.max_snapshots is not None and args.max_snapshots <= 0:
        raise ValueError("max-snapshots must be positive")

    if args.max_new_snapshots is not None and args.max_new_snapshots <= 0:
        raise ValueError("max-new-snapshots must be positive")

    settings = read_job_survey_config(path=args.config)
    jobs = discover_job_snapshots(settings=settings)
    outputs = check_batch_output_paths(settings=settings, jobs=jobs)
    snapshot_count = sum(len(snapshots) for snapshots in jobs.values())
    print(f"Batch survey source: {settings.source_root}", flush=True)
    print(f"Batch survey outputs: {outputs.directory}", flush=True)
    for jobname, snapshots in jobs.items():
        print(f"Job {jobname}: {len(snapshots)} snapshots", flush=True)
    print(f"Total inventory: {snapshot_count} snapshots", flush=True)
    print("Loading tables once to read their axes...", flush=True)
    despotic = load_despotic_lookup(table_path=settings.despotic_table)
    cloudy = CloudyLookup(path=settings.cloudy_table)

    records = scan_job_snapshots(
        settings=settings,
        jobs=jobs,
        despotic=despotic,
        cloudy=cloudy,
        max_snapshots=args.max_snapshots,
        max_new_snapshots=args.max_new_snapshots,
    )
    summary = build_job_survey_summary(settings=settings, records=records, despotic=despotic, cloudy=cloudy)
    outputs.write_json(path=outputs.directory / "summary.json", values=summary)
    outputs.write_text(path=outputs.directory / "snapshots.csv", text=range_csv_text(rows=records))
    outputs.write_text(path=outputs.directory / "jobs.csv", text=range_csv_text(rows=summary["jobs"], summary_rows=True))
    print(f"Survey complete: {summary['overall']['survey_complete']}", flush=True)
    print(f"Snapshot statuses: {summary['overall']['snapshot_status_counts']}", flush=True)
    print(f"Saved summaries: {outputs.directory}", flush=True)


if __name__ == "__main__":
    main()
