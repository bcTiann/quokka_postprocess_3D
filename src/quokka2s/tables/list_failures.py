"""List failed ``(nH, N_H, dVdr)`` cells from a 3D GOW/LVG table."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import load_table


def collect_failures(table):
    if table.failure_mask is None:
        return []
    attempts_by_cell = defaultdict(list)
    for attempt in table.attempts:
        if attempt.dvdr_idx is not None:
            attempts_by_cell[(attempt.row_idx, attempt.col_idx, attempt.dvdr_idx)].append(attempt)

    failures = []
    for row_idx, col_idx, dvdr_idx in np.argwhere(table.failure_mask):
        key = (int(row_idx), int(col_idx), int(dvdr_idx))
        failures.append((
            *key,
            float(table.nH_values[row_idx]),
            float(table.col_density_values[col_idx]),
            float(table.dVdr_values[dvdr_idx]),
            float(table.tg_final[row_idx, col_idx, dvdr_idx]),
            attempts_by_cell.get(key, []),
        ))
    return failures


def write_csv(path: Path, failures) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "row_idx", "col_idx", "dvdr_idx", "nH_cgs", "colDen_cgs",
            "dVdr_s-1", "final_Tg", "converged", "message", "duration_s",
        ])
        for row, col, dvdr_idx, nH, column, dvdr, tg, history in failures:
            attempt = history[-1] if history else None
            writer.writerow([
                row, col, dvdr_idx, nH, column, dvdr, tg,
                "" if attempt is None else attempt.converged,
                "" if attempt is None else attempt.message or "",
                "" if attempt is None or attempt.duration is None else attempt.duration,
            ])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)

    table = load_table(args.table)
    failures = collect_failures(table)
    print(f"Failed cells: {len(failures)} / {table.tg_final.size}")
    for row, col, dvdr_idx, nH, column, dvdr, tg, history in failures[:50]:
        suffix = "" if history else " (attempt dVdr metadata unavailable in legacy v4 table)"
        print(
            f"[{row},{col},{dvdr_idx}] nH={nH:.3e} N_H={column:.3e} "
            f"dVdr={dvdr:.3e} Tg={tg:.3g}{suffix}"
        )
    if len(failures) > 50:
        print(f"... {len(failures) - 50} additional failures; use --csv for the full list")
    if args.csv:
        write_csv(args.csv, failures)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
