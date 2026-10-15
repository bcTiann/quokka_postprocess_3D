"""Measure a snapshot's physical inputs and compare them with current table axes.

Uses the same emission.yaml selection, reader, derived fields and table paths
as process. It saves scalar statistics only, without calculating line emission,
dust attenuation, images or spectra. Saved ranges can be compared again without
reading the snapshot cells.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from quokka2s.cloudy.lookup import CloudyLookup
from quokka2s.paths import resolve_path
from quokka2s.processing_inputs import load_despotic_lookup
from quokka2s.run_settings import DEFAULT_CONFIG, load_process_config
from quokka2s.table_input_range_check import build_table_range_report
from quokka2s.table_input_statistics import SnapshotInputSettings
from quokka2s.table_input_survey import load_or_measure_input_ranges


def print_range_report(*, report: dict) -> None:
    """Display measured/table ranges and the explicitly scoped coverage result."""
    for table_name in ("despotic", "cloudy"):
        print(f"{table_name.upper()} input ranges:", flush=True)
        for name, axis in report[table_name].items():
            print(
                f"  {name}: [{axis['physical_minimum']}, {axis['physical_maximum']}] "
                f"{axis['unit']}; table [{axis['table_minimum']:.6g}, "
                f"{axis['table_maximum']:.6g}]; covered={axis['input_range_covered']}",
                flush=True,
            )
    print(
        f"Scan={report['scan_status']}; full snapshot={report['full_snapshot']}; "
        f"selected input ranges covered={report['selected_input_ranges_covered']}",
        flush=True,
    )
    print("This checks coordinate ranges only. Remaining table holes are not checked.", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, help="Override the default selected-run/table_input_check directory")
    parser.add_argument("--tile-ny", type=int, default=8, help="Y cells per complete-z reading tile (default: 8)")
    parser.add_argument("--max-tiles", type=int, help="Read only this many y tiles for a labelled partial scan")
    args = parser.parse_args(argv)
    if args.tile_ny < 1 or (args.max_tiles is not None and args.max_tiles < 1):
        parser.error("--tile-ny and --max-tiles must be positive")
    config = load_process_config(path=args.config)
    output_directory = config.output_dir.parent / "table_input_check"
    if args.output_dir is not None:
        output_directory = resolve_path(args.output_dir)
    print(f"Table-input survey: {config.jobname}/{config.snapshot_name}", flush=True)
    print(f"Snapshot: {config.dataset}", flush=True)
    print(f"Statistics directory: {output_directory}", flush=True)
    scan_settings = SnapshotInputSettings(
        dataset=config.dataset,
        jobname=config.jobname,
        snapshot_name=config.snapshot_name,
        tile_ny=args.tile_ny,
        query_chunk=config.query_chunk,
        max_tiles=args.max_tiles,
        x_index_range=config.x_index_range,
        y_index_range=config.y_index_range,
    )
    statistics, files = load_or_measure_input_ranges(
        config=scan_settings,
        output_directory=output_directory,
    )

    print("Comparing saved ranges with current table axes...", flush=True)
    despotic = load_despotic_lookup(table_path=config.despotic_table)
    cloudy = CloudyLookup(path=config.cloudy_table)
    report = build_table_range_report(statistics=statistics, despotic=despotic, cloudy=cloudy)
    report["tables"] = {"despotic": str(config.despotic_table), "cloudy": str(config.cloudy_table)}
    files.write_json(path=files.report_path, values=report)
    print_range_report(report=report)
    print(f"Saved: {files.statistics_path}\nSaved: {files.report_path}", flush=True)


if __name__ == "__main__":
    main()
