from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict
import csv

import numpy as np

from . import load_table


def format_record(idx: int, rec) -> str:
    return (
        f"    #{idx}: guess={rec.tg_guess:.3g} K "
        f"final={rec.final_Tg:.3g} K "
        f"converged={rec.converged} "
        f"message={rec.message or '-'} "
        f"duration={rec.duration if rec.duration is not None else float('nan'):.3g}s"
    )


def collect_failures(table) -> list[tuple[int, int, int, float, float, float, float, list]]:
    failure_mask = np.asarray(table.failure_mask, dtype=bool)
    coords = np.argwhere(failure_mask)
    attempts_by_cell: dict[tuple[int, int], list] = defaultdict(list)
    for rec in table.attempts:
        attempts_by_cell[(rec.row_idx, rec.col_idx)].append(rec)

    failures: list[tuple[int, int, int, float, float, float, float, list]] = []
    for row_idx, col_idx, t_idx in coords:
        nH = table.nH_values[row_idx]
        col = table.col_density_values[col_idx]
        t_val = table.T_values[t_idx]  # 新增：提取失败的温度
        final_tg = table.tg_final[row_idx, col_idx, t_idx]  # 修改：加上t_idx
        history = attempts_by_cell.get((row_idx, col_idx), [])
        failures.append((row_idx, col_idx, t_idx, nH, col, t_val, final_tg, history))
    return failures


def print_failures(table, failures) -> None:
    if not failures:
        print("No failing cells were recorded.")
        return

    print(f"Found {len(failures)} failing cells:")
    for row_idx, col_idx, t_idx, nH, col, t_val, final_tg, history in failures:
        print(
            f"- cell[{row_idx},{col_idx},{t_idx}] nH={nH:.3e} cm^-3 "
            f"col={col:.3e} cm^-2 T={t_val:.3g} K final_Tg={final_tg:.3g} K"
        )
        if not history:
            print("    (no attempt records)")
            continue
        for idx, rec in enumerate(history, start=1):
            print(format_record(idx, rec))


def write_csv(path: Path, failures) -> None:
    headers = [
        "row_idx",
        "col_idx",
        "t_idx",
        "nH_cgs",
        "colDen_cgs",
        "T_K",
        "final_Tg",
        "attempt_idx",
        "tg_guess",
        "attempt_final_Tg",
        "converged",
        "message",
        "duration_s",
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row_idx, col_idx, t_idx, nH, col, t_val, final_tg, history in failures:
            if history:
                for idx, rec in enumerate(history, start=1):
                    writer.writerow(
                        [
                            row_idx,
                            col_idx,
                            t_idx,
                            nH,
                            col,
                            t_val,
                            final_tg,
                            idx,
                            rec.tg_guess,
                            rec.final_Tg,
                            rec.converged,
                            rec.message or "",
                            rec.duration if rec.duration is not None else "",
                        ]
                    )
            else:
                writer.writerow(
                    [row_idx, col_idx, t_idx, nH, col, t_val, final_tg, "", "", "", "", "", ""]
                )
    print(f"Wrote failure details to {path}")


def write_attempt_history(path: Path, attempts) -> None:
    headers = [
        "row_idx",
        "col_idx",
        "nH_cgs",
        "colDen_cgs",
        "tg_guess",
        "final_Tg",
        "converged",
        "message",
        "duration_s",
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for rec in attempts:
            writer.writerow(
                [
                    rec.row_idx,
                    rec.col_idx,
                    rec.nH,
                    rec.colDen,
                    rec.tg_guess,
                    rec.final_Tg,
                    rec.converged,
                    rec.message or "",
                    rec.duration if rec.duration is not None else "",
                ]
            )
    print(f"Wrote attempt history to {path}")


def list_failures(
    path: Path,
    csv_output: Path | None = None,
    attempts_output: Path | None = None,
) -> None:
    table = load_table(path)
    if table.failure_mask is None:
        print("Table has no failure mask; nothing to report.")
        return
    failures = collect_failures(table)
    if not failures:
        print("No failing cells were recorded.")
        return
    print_failures(table, failures)
    target_csv = csv_output or path.with_suffix("").with_name(path.stem + "_failures.csv")
    write_csv(target_csv, failures)

    attempts_csv = (
        attempts_output or path.with_suffix("").with_name(path.stem + "_attempts.csv")
    )
    write_attempt_history(attempts_csv, table.attempts)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit(
            "Usage: python -m quokka2s.tables.list_failures <table_path.npz> [failures_csv] [attempts_csv]"
        )
    table_path = Path(args[0])
    csv_path = Path(args[1]) if len(args) > 1 else None
    attempts_path = Path(args[2]) if len(args) > 2 else None
    list_failures(table_path, csv_path, attempts_path)


if __name__ == "__main__":
    main()
