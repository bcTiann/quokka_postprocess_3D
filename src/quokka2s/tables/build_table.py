"""Build the canonical 35^3 GOW/LVG DESPOTIC lookup table.

This is the only production table-building entry point.  Chemistry, escape
geometry, grid ranges, and species are intentionally fixed so a command cannot
silently create a physically different table under the canonical filename.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .builder import GOW_LVG_SPECIES, build_gow_lvg_table
from .io import save_table
from .models import LogGrid


N_H_RANGE = (1e-4, 1e6)
COL_DEN_RANGE = (1e15, 1e24)
DVDR_RANGE = (1e-19, 1e-12)
GRID_POINTS = 35
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "output_tables_3D_GOW_LVG" / "despotic_table.npz"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output NPZ path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="joblib worker count (-1 uses all CPUs; default: -1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing output table",
    )
    return parser.parse_args(argv)


def _write_readme(path: Path, elapsed: float, table) -> Path:
    failed = int(np.count_nonzero(table.failure_mask)) if table.failure_mask is not None else 0
    nan_t = int(np.count_nonzero(~np.isfinite(table.tg_final)))
    readme = path.parent / "README.txt"
    species = ", ".join(
        name + ("(em)" if table.species_data[name].is_emitter else "")
        for name in table.species
    )
    text = (
        "DESPOTIC 3D table\n"
        "=================\n"
        "network         : GOW\n"
        "escape geometry : LVG\n"
        "evolveTemp      : iterateDust\n"
        f"grid            : nH {N_H_RANGE[0]:.0e}..{N_H_RANGE[1]:.0e}, "
        f"NH {COL_DEN_RANGE[0]:.0e}..{COL_DEN_RANGE[1]:.0e}, "
        f"dVdr {DVDR_RANGE[0]:.0e}..{DVDR_RANGE[1]:.0e}, {GRID_POINTS}^3\n"
        f"species         : {species}\n"
        f"failed cells    : {failed} / {table.tg_final.size}\n"
        f"non-finite Tg   : {nan_t} / {table.tg_final.size}\n"
        f"build time      : {elapsed / 3600:.2f} h ({elapsed:.0f} s)\n"
        f"completed at    : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"output file     : {path}\n"
    )
    readme.write_text(text)
    return readme


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing table: {output}\nPass --force to replace it.")

    nH_grid = LogGrid(*N_H_RANGE, num_points=GRID_POINTS)
    col_grid = LogGrid(*COL_DEN_RANGE, num_points=GRID_POINTS)
    dVdr_grid = LogGrid(*DVDR_RANGE, num_points=GRID_POINTS)
    species = ", ".join(s.name + ("(em)" if s.is_emitter else "") for s in GOW_LVG_SPECIES)

    print("[build_table] network = GOW")
    print("[build_table] geometry = LVG")
    print(f"[build_table] species = {species}")
    print(f"[build_table] grid = {GRID_POINTS}^3")
    print(f"[build_table] output = {output}")

    started = time.time()
    table = build_gow_lvg_table(
        nH_grid,
        col_grid,
        dVdr_grid,
        show_progress=True,
        workers=args.workers,
    )
    elapsed = time.time() - started

    output.parent.mkdir(parents=True, exist_ok=True)
    save_table(table, output)
    readme = _write_readme(output, elapsed, table)
    print(f"[build_table] saved -> {output} ({elapsed / 3600:.2f}h)")
    print(f"[build_table] README sidecar -> {readme}")


if __name__ == "__main__":
    main()
