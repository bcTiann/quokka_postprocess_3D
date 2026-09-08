"""Build the canonical snapshot-covering GOW/LVG DESPOTIC lookup table.

Chemistry, escape geometry, and species are fixed. Use --snapshot-domain for
measured all-cell input extrema; omission retains the legacy grid for older
commands. The output records the measured domain when supplied.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import time
from pathlib import Path

import numpy as np

from .builder import GOW_LVG_SPECIES, build_gow_lvg_table
from .io import save_table
from .dvdr_domain import extended_dvdr_values
from .models import ExplicitGrid, LogGrid


N_H_RANGE = (1e-4, 1e6)
COL_DEN_RANGE = (1e15, 1e24)
GRID_POINTS = 35
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "output_tables_3D_GOW_LVG"
    / "despotic_table_co10_co21_dvdr_fullrange.npz"
)


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
        "--snapshot-domain", type=Path,
        help="All-cell extrema JSON from scripts/measure_despotic_snapshot_domain.py; "
             "use these endpoints with 35 x 35 x 53 logarithmic nodes.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        help="Save each completed point and resume matching completed points from "
             "this directory; incompatible settings or corrupt results are rejected.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing output table",
    )
    return parser.parse_args(argv)


def _snapshot_grids(path: Path):
    domain = json.loads(path.read_text())
    if domain.get("selection") != "all simulation cells" or domain.get("total_cells", 0) <= 0:
        raise ValueError("Domain must describe all simulation cells")
    grids = []
    for name, count in (("nH", 35), ("NH", 35), ("dVdr", 53)):
        axis = domain["axes"][name]
        lo, hi = float(axis["minimum"]), float(axis["maximum"])
        if (axis["count"] != domain["total_cells"] or axis["invalid_count"] != 0
                or not np.isfinite([lo, hi]).all() or not 0.0 < lo < hi):
            raise ValueError(f"Invalid or incomplete snapshot bounds: {name}")
        values = np.geomspace(lo, hi, count)
        values[0], values[-1] = lo, hi
        grids.append(ExplicitGrid(tuple(values)))
    return (*grids, domain)


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
        f"grid            : nH {table.nH_values[0]:.12e}..{table.nH_values[-1]:.12e}, "
        f"NH {table.col_density_values[0]:.12e}..{table.col_density_values[-1]:.12e}, "
        f"dVdr {table.dVdr_values[0]:.6e}..{table.dVdr_values[-1]:.6e}, "
        f"shape {table.tg_final.shape}\n"
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

    domain = None
    if args.snapshot_domain:
        nH_grid, col_grid, dVdr_grid, domain = _snapshot_grids(args.snapshot_domain)
    else:
        nH_grid = LogGrid(*N_H_RANGE, num_points=GRID_POINTS)
        col_grid = LogGrid(*COL_DEN_RANGE, num_points=GRID_POINTS)
        dVdr_grid = ExplicitGrid(tuple(extended_dvdr_values()))
    dVdr_values = dVdr_grid.sample()
    species = ", ".join(s.name + ("(em)" if s.is_emitter else "") for s in GOW_LVG_SPECIES)

    print("[build_table] network = GOW")
    print("[build_table] geometry = LVG")
    print(f"[build_table] species = {species}")
    print(f"[build_table] grid = {GRID_POINTS} x {GRID_POINTS} x {len(dVdr_values)}")
    print(f"[build_table] output = {output}")
    for name, grid in (("nH", nH_grid), ("NH", col_grid), ("dVdr", dVdr_grid)):
        values = grid.sample()
        print(f"[build_table] {name} range = {values[0]:.16e} .. {values[-1]:.16e}")

    started = time.time()
    checkpoint_options = {}
    if args.checkpoint_dir is not None:
        checkpoint_options = {
            "checkpoint_dir": args.checkpoint_dir.expanduser().resolve(),
            "checkpoint_context": {"snapshot_domain": domain},
        }
    table = build_gow_lvg_table(
        nH_grid,
        col_grid,
        dVdr_grid,
        show_progress=True,
        workers=args.workers,
        **checkpoint_options,
    )
    elapsed = time.time() - started

    if domain is not None:
        metadata = dict(table.build_metadata)
        metadata["snapshot_domain"] = domain
        metadata["grid_sampling"] = "35 x 35 x 53 logarithmic nodes; exact snapshot endpoints"
        table = replace(table, build_metadata=metadata)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_table(table, output)
    readme = _write_readme(output, elapsed, table)
    print(f"[build_table] saved -> {output} ({elapsed / 3600:.2f}h)")
    print(f"[build_table] README sidecar -> {readme}")


if __name__ == "__main__":
    main()
