"""Extend a raw DESPOTIC table by solving only new physical coordinates."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
import yaml

from quokka2s.despotic.cell_solver import validated_solver_metadata
from quokka2s.despotic.table_builder import GOW_LVG_SPECIES, build_gow_lvg_table
from quokka2s.despotic.table_extension import (
    extend_table_axes,
    read_raw_table_for_extension,
    validate_reuse_settings,
)
from quokka2s.despotic.table_files import save_table
from quokka2s.paths import resolve_path


DEFAULT_CONFIG = Path("configs/despotic_extension.yaml")


@dataclass(frozen=True)
class TableExtensionSettings:
    """Resolved old raw input, separate new output and requested axis bounds.

    bounds contains nH [cm^-3], NH [cm^-2] and dVdr [s^-1]. Omitted bounds
    keep their source endpoints. checkpoints belong to this extended grid;
    they are not the old table's checkpoint directory.
    """

    source_table: Path
    output: Path
    checkpoint_dir: Path
    bounds: dict
    workers: int


def read_extension_config(path: Path) -> TableExtensionSettings:
    """Read one YAML; resolve its paths relative to the YAML's directory."""
    config_path = resolve_path(path)
    with config_path.open("r") as source:
        values = yaml.safe_load(source)
    required = {"source_table", "output", "checkpoint_dir", "bounds"}
    if not isinstance(values, dict) or not required.issubset(values):
        raise ValueError("Extension config requires source_table, output, checkpoint_dir and bounds")
    if set(values) - required - {"workers"}:
        raise ValueError("Unknown DESPOTIC extension config field")
    bounds = values["bounds"]
    if not isinstance(bounds, dict) or set(bounds) - {"nH", "NH", "dVdr"}:
        raise ValueError("Extension bounds use only nH, NH and dVdr")
    for axis_bounds in bounds.values():
        if not isinstance(axis_bounds, dict) or set(axis_bounds) - {"minimum", "maximum"}:
            raise ValueError("Each axis bound uses minimum and/or maximum")
    workers = values.get("workers", 1)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    settings = TableExtensionSettings(
        source_table=resolve_path(values["source_table"], base_directory=config_path.parent),
        output=resolve_path(values["output"], base_directory=config_path.parent),
        checkpoint_dir=resolve_path(values["checkpoint_dir"], base_directory=config_path.parent),
        bounds=bounds,
        workers=workers,
    )
    if settings.output == settings.source_table:
        raise ValueError("The extended output must differ from the old raw table")
    if settings.output.exists():
        raise FileExistsError(f"Choose a new extended output; refusing to overwrite {settings.output}")
    return settings


def prepare_extension(settings: TableExtensionSettings):
    """Read/verify the source and return exact extended grids plus a small plan.

    This performs no DESPOTIC point solves and creates no output/checkpoint
    directories. The returned plan states counts and actual outward endpoints.
    """
    source_table = read_raw_table_for_extension(settings.source_table)
    validate_reuse_settings(
        source_table=source_table,
        current_metadata=validated_solver_metadata(),
        species_specs=GOW_LVG_SPECIES,
    )
    grids = extend_table_axes(source_table=source_table, bounds=settings.bounds)
    axes = [grid.sample() for grid in grids]
    shape = tuple(len(axis) for axis in axes)
    new_count = int(np.prod(shape)) - source_table.tg_final.size
    plan = {
        "source_table": str(settings.source_table),
        "output": str(settings.output),
        "checkpoint_dir": str(settings.checkpoint_dir),
        "old_shape": list(source_table.tg_final.shape),
        "extended_shape": list(shape),
        "reused_successful_nodes": int(np.count_nonzero(~source_table.failure_mask)),
        "retained_old_failed_nodes": int(np.count_nonzero(source_table.failure_mask)),
        "new_coordinate_nodes": new_count,
        "workers": settings.workers,
        "axes": {
            name: {"minimum": float(axis[0]), "maximum": float(axis[-1]), "count": len(axis)}
            for name, axis in zip(("nH", "NH", "dVdr"), axes)
        },
        "policy": "Copy exact raw successes; do not retry old failures; solve new coordinates only.",
    }
    return grids, plan


def run_extension(settings: TableExtensionSettings, grids) -> None:
    """Build and save a new raw NPZ; leave the source raw/filled tables intact."""
    table = build_gow_lvg_table(
        nH_grid=grids[0],
        col_grid=grids[1],
        dVdr_grid=grids[2],
        reuse_raw_table=settings.source_table,
        checkpoint_dir=settings.checkpoint_dir,
        workers=settings.workers,
    )
    metadata = dict(table.build_metadata)
    metadata["grid_sampling"] = "Exact old coordinates plus outward nodes at the old log spacing"
    metadata["table_extension"] = {
        **metadata["table_extension"],
        "requested_bounds": settings.bounds,
    }
    table = replace(table, build_metadata=metadata)
    # Recheck after a long build so an output created meanwhile is not replaced.
    if settings.output.exists():
        raise FileExistsError(f"Extended output appeared during the build: {settings.output}")
    save_table(table, settings.output)
    print(f"Saved extended raw table: {settings.output}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Print counts and axes without solving or saving")
    args = parser.parse_args(argv)
    settings = read_extension_config(args.config)
    grids, plan = prepare_extension(settings)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print("Dry run: no nodes solved; no files written.", flush=True)
        return
    run_extension(settings, grids)


if __name__ == "__main__":
    main()
