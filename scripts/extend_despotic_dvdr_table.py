#!/usr/bin/env python3
"""Extend the canonical DESPOTIC table without recomputing old nodes.

The existing 35 dV/dr nodes are copied bit-for-bit.  Only nodes below
1e-19 s^-1 and above 1e-12 s^-1 are solved, after which the two tables are
merged and the existing convex-hull-only cleaner is run on the result.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from quokka2s.tables import ExplicitGrid, build_gow_lvg_table, load_table, save_table
from quokka2s.tables.dvdr_domain import (
    added_dvdr_values,
    extended_dvdr_values,
    legacy_dvdr_values,
)
from quokka2s.tables.models import (
    AttemptRecord,
    DespoticTable,
    SpeciesLineGrid,
    SpeciesRecord,
)
from quokka2s.tables.solver import LINE_RESULT_FIELDS, validated_solver_metadata


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "output_tables_3D_GOW_LVG"
DEFAULT_SOURCE = TABLE_DIR / "despotic_table_co10_co21.npz"
DEFAULT_OUTPUT = TABLE_DIR / "despotic_table_co10_co21_dvdr_fullrange.npz"
DEFAULT_CLEAN_OUTPUT = (
    TABLE_DIR / "despotic_table_co10_co21_dvdr_fullrange_clean.npz"
)
DEFAULT_PARTS_DIR = TABLE_DIR / "despotic_table_co10_co21_dvdr_fullrange_parts"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clean-output", type=Path, default=DEFAULT_CLEAN_OUTPUT)
    parser.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS_DIR)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    return parser.parse_args()


def _require_matching_build_metadata(reference, candidate, context: str) -> None:
    if reference is None or candidate is None:
        raise ValueError(
            f"{context}: build provenance is unknown; rebuild the full table with "
            "quokka2s.tables.build_table instead of reusing legacy nodes."
        )
    if reference != candidate:
        raise ValueError(
            f"{context}: composition or solver build metadata differs; "
            "rebuild the full table instead of mixing old and new nodes."
        )


def _concatenate_dvdr_tables(tables: list[DespoticTable]) -> DespoticTable:
    """Join checkpoint tables along their dV/dr axis."""
    if not tables:
        raise ValueError("at least one checkpoint table is required")
    reference = tables[0]
    for table in tables[1:]:
        _require_matching_build_metadata(
            reference.build_metadata, table.build_metadata, "checkpoint tables"
        )
        if not np.array_equal(reference.nH_values, table.nH_values):
            raise ValueError("checkpoint nH axes differ")
        if not np.array_equal(reference.col_density_values, table.col_density_values):
            raise ValueError("checkpoint column-density axes differ")
        if reference.species != table.species:
            raise ValueError("checkpoint species differ")
        for name in ("chemistry_network", "escape_geometry", "temperature_mode"):
            if getattr(reference, name) != getattr(table, name):
                raise ValueError(f"checkpoint metadata differs for {name}")

    dVdr_values = np.concatenate([table.dVdr_values for table in tables])
    if np.any(np.diff(dVdr_values) <= 0.0):
        raise ValueError("checkpoint dV/dr axes are not strictly increasing")

    concatenate = lambda arrays: np.concatenate(arrays, axis=2)  # noqa: E731
    species_data: dict[str, SpeciesRecord] = {}
    for name in reference.species:
        records = [table.species_data[name] for table in tables]
        abundance = concatenate([record.abundance for record in records])
        line = None
        if records[0].is_emitter:
            if any(record.line is None for record in records):
                raise ValueError(f"missing line data for emitter {name}")
            fields = {
                field: concatenate([
                    getattr(record.line, field)  # type: ignore[union-attr]
                    for record in records
                ])
                for field in LINE_RESULT_FIELDS
            }
            line = SpeciesLineGrid(**fields, abundance=abundance)
        species_data[name] = SpeciesRecord(
            name=name,
            abundance=abundance,
            line=line,
            is_emitter=records[0].is_emitter,
        )

    failure_mask = concatenate([
        np.zeros(table.tg_final.shape, dtype=bool)
        if table.failure_mask is None else table.failure_mask
        for table in tables
    ]).astype(bool)

    energy_terms = None
    if any(table.energy_terms for table in tables):
        names = sorted({
            name
            for table in tables
            for name in (table.energy_terms or {})
        })
        energy_terms = {
            name: concatenate([
                (table.energy_terms or {}).get(
                    name, np.full(table.tg_final.shape, np.nan)
                )
                for table in tables
            ])
            for name in names
        }

    attempts: list[AttemptRecord] = []
    offset = 0
    for table in tables:
        for record in table.attempts:
            local_idx = record.dvdr_idx
            if local_idx is None:
                if record.dvdr is None:
                    attempts.append(record)
                    continue
                local_idx = int(_positions(
                    table.dVdr_values, np.asarray([record.dvdr])
                )[0])
            attempts.append(replace(record, dvdr_idx=offset + local_idx))
        offset += table.dVdr_values.size

    return DespoticTable(
        species_data=species_data,
        tg_final=concatenate([table.tg_final for table in tables]),
        nH_values=np.array(reference.nH_values, copy=True),
        col_density_values=np.array(reference.col_density_values, copy=True),
        dVdr_values=dVdr_values,
        mu_values=concatenate([table.mu_values for table in tables]),
        cv_values=concatenate([table.cv_values for table in tables]),
        Eint_values=concatenate([table.Eint_values for table in tables]),
        failure_mask=failure_mask,
        energy_terms=energy_terms,
        attempts=tuple(attempts),
        chemistry_network=reference.chemistry_network,
        escape_geometry=reference.escape_geometry,
        temperature_mode=reference.temperature_mode,
        build_metadata=reference.build_metadata,
    )


def _positions(combined: np.ndarray, subset: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(combined, subset)
    if np.any(positions >= combined.size) or not np.allclose(
        combined[positions], subset, rtol=2.0e-14, atol=0.0
    ):
        raise ValueError("dV/dr subset is not contained in the combined axis")
    return positions


def _merge_array(
    old: np.ndarray,
    added: np.ndarray,
    old_positions: np.ndarray,
    added_positions: np.ndarray,
    combined_shape: tuple[int, int, int],
) -> np.ndarray:
    old = np.asarray(old)
    added = np.asarray(added)
    if old.shape[:2] != combined_shape[:2] or added.shape[:2] != combined_shape[:2]:
        raise ValueError("table arrays disagree on the nH/NH dimensions")
    output = np.empty(combined_shape, dtype=np.result_type(old, added))
    output[..., old_positions] = old
    output[..., added_positions] = added
    if not np.array_equal(output[..., old_positions], old, equal_nan=True):
        raise RuntimeError("old table values changed during merge")
    return output


def _remap_attempts(
    attempts: tuple[AttemptRecord, ...],
    source_axis: np.ndarray,
    combined_axis: np.ndarray,
) -> tuple[AttemptRecord, ...]:
    remapped = []
    for record in attempts:
        value = record.dvdr
        if value is None and record.dvdr_idx is not None:
            value = float(source_axis[record.dvdr_idx])
        if value is None:
            remapped.append(record)
            continue
        position = int(_positions(
            combined_axis, np.asarray([value], dtype=float)
        )[0])
        remapped.append(replace(record, dvdr_idx=position, dvdr=float(value)))
    return tuple(remapped)


def _merge_tables(old: DespoticTable, added: DespoticTable) -> DespoticTable:
    _require_matching_build_metadata(old.build_metadata, added.build_metadata, "table merge")
    if not np.array_equal(old.nH_values, added.nH_values):
        raise ValueError("nH axes differ")
    if not np.array_equal(old.col_density_values, added.col_density_values):
        raise ValueError("column-density axes differ")
    if old.species != added.species:
        raise ValueError(f"species differ: {old.species} vs {added.species}")
    for name in ("chemistry_network", "escape_geometry", "temperature_mode"):
        if getattr(old, name) != getattr(added, name):
            raise ValueError(f"table metadata differs for {name}")

    combined_axis = extended_dvdr_values()
    old_positions = _positions(combined_axis, old.dVdr_values)
    added_positions = _positions(combined_axis, added.dVdr_values)
    if np.intersect1d(old_positions, added_positions).size:
        raise ValueError("old and added dV/dr nodes overlap")
    if not np.array_equal(
        np.sort(np.concatenate((old_positions, added_positions))),
        np.arange(combined_axis.size),
    ):
        raise ValueError("old and added nodes do not cover the combined axis")
    combined_shape = (
        old.nH_values.size,
        old.col_density_values.size,
        combined_axis.size,
    )

    merge = lambda a, b: _merge_array(  # noqa: E731
        a, b, old_positions, added_positions, combined_shape
    )
    species_data: dict[str, SpeciesRecord] = {}
    for name in old.species:
        old_record = old.species_data[name]
        added_record = added.species_data[name]
        abundance = merge(old_record.abundance, added_record.abundance)
        line = None
        if old_record.is_emitter:
            if old_record.line is None or added_record.line is None:
                raise ValueError(f"missing line data for emitter {name}")
            fields = {
                field: merge(
                    getattr(old_record.line, field),
                    getattr(added_record.line, field),
                )
                for field in LINE_RESULT_FIELDS
            }
            line = SpeciesLineGrid(**fields, abundance=abundance)
        species_data[name] = SpeciesRecord(
            name=name,
            abundance=abundance,
            line=line,
            is_emitter=old_record.is_emitter,
        )

    old_failures = (
        np.zeros(old.tg_final.shape, dtype=bool)
        if old.failure_mask is None else old.failure_mask
    )
    added_failures = (
        np.zeros(added.tg_final.shape, dtype=bool)
        if added.failure_mask is None else added.failure_mask
    )
    failure_mask = merge(old_failures, added_failures).astype(bool)

    energy_terms = None
    if old.energy_terms or added.energy_terms:
        energy_terms = {}
        names = sorted(set(old.energy_terms or {}) | set(added.energy_terms or {}))
        for name in names:
            old_values = (old.energy_terms or {}).get(name)
            added_values = (added.energy_terms or {}).get(name)
            if old_values is None:
                old_values = np.full(old.tg_final.shape, np.nan)
            if added_values is None:
                added_values = np.full(added.tg_final.shape, np.nan)
            energy_terms[name] = merge(old_values, added_values)

    attempts = (
        _remap_attempts(old.attempts, old.dVdr_values, combined_axis)
        + _remap_attempts(added.attempts, added.dVdr_values, combined_axis)
    )
    return DespoticTable(
        species_data=species_data,
        tg_final=merge(old.tg_final, added.tg_final),
        nH_values=np.array(old.nH_values, copy=True),
        col_density_values=np.array(old.col_density_values, copy=True),
        dVdr_values=combined_axis,
        mu_values=merge(old.mu_values, added.mu_values),
        cv_values=merge(old.cv_values, added.cv_values),
        Eint_values=merge(old.Eint_values, added.Eint_values),
        failure_mask=failure_mask,
        energy_terms=energy_terms,
        attempts=attempts,
        chemistry_network=old.chemistry_network,
        escape_geometry=old.escape_geometry,
        temperature_mode=old.temperature_mode,
        build_metadata=old.build_metadata,
    )


def main() -> None:
    args = _parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    clean_output = args.clean_output.expanduser().resolve()
    parts_dir = args.parts_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    protected = [output]
    if not args.skip_clean:
        protected.append(clean_output)
    existing = [path for path in protected if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing)
        )

    old = load_table(source)
    build_metadata = validated_solver_metadata()
    _require_matching_build_metadata(old.build_metadata, build_metadata, "source table")
    expected_old = legacy_dvdr_values()
    if not np.allclose(old.dVdr_values, expected_old, rtol=2.0e-14, atol=0.0):
        raise ValueError("source table does not have the expected legacy dV/dr axis")

    added_axis = added_dvdr_values()
    print(f"[source] {source}")
    print(f"[legacy] {old.tg_final.shape}")
    print(
        f"[added] {added_axis.size} dV/dr nodes, "
        f"{added_axis[0]:.6e}..{added_axis[-1]:.6e} s^-1"
    )
    started = time.time()
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_tables: list[DespoticTable] = []
    reused_parts = 0
    for index, dvdr in enumerate(added_axis):
        part_path = parts_dir / f"added_dvdr_{index:02d}.npz"
        if part_path.is_file():
            part = load_table(part_path)
            _require_matching_build_metadata(
                build_metadata, part.build_metadata, f"checkpoint {part_path}"
            )
            if part.tg_final.shape != (old.nH_values.size, old.col_density_values.size, 1):
                raise ValueError(f"invalid checkpoint shape: {part_path}")
            if not np.array_equal(part.nH_values, old.nH_values):
                raise ValueError(f"invalid checkpoint nH axis: {part_path}")
            if not np.array_equal(part.col_density_values, old.col_density_values):
                raise ValueError(f"invalid checkpoint column axis: {part_path}")
            if not np.array_equal(part.dVdr_values, np.asarray([dvdr])):
                raise ValueError(f"invalid checkpoint dV/dr value: {part_path}")
            reused_parts += 1
            print(
                f"[part {index + 1:02d}/{added_axis.size:02d}] "
                f"reuse {dvdr:.6e} s^-1"
            )
        else:
            print(
                f"[part {index + 1:02d}/{added_axis.size:02d}] "
                f"solve {dvdr:.6e} s^-1"
            )
            part = build_gow_lvg_table(
                ExplicitGrid(tuple(old.nH_values)),
                ExplicitGrid(tuple(old.col_density_values)),
                ExplicitGrid((float(dvdr),)),
                show_progress=True,
                workers=args.workers,
            )
            save_table(part, part_path)
            print(f"[part saved] {part_path}")
        part_tables.append(part)

    added = _concatenate_dvdr_tables(part_tables)
    if not np.array_equal(added.dVdr_values, added_axis):
        raise RuntimeError("checkpoint merge changed the requested dV/dr axis")
    merged = _merge_tables(old, added)
    save_table(merged, output)
    elapsed = time.time() - started
    print(f"[saved raw] {output}")

    if not args.skip_clean:
        cleaner = ROOT / "scripts" / "fill_table_convex_hull_only.py"
        subprocess.run(
            [sys.executable, str(cleaner), str(output), str(clean_output)],
            check=True,
        )
        print(f"[saved clean] {clean_output}")

    report = {
        "source": str(source),
        "raw_output": str(output),
        "clean_output": None if args.skip_clean else str(clean_output),
        "legacy_shape": list(old.tg_final.shape),
        "extended_shape": list(merged.tg_final.shape),
        "legacy_dvdr_nodes": int(old.dVdr_values.size),
        "added_dvdr_nodes": int(added_axis.size),
        "reused_checkpoint_nodes": int(reused_parts),
        "extended_dvdr_nodes": int(merged.dVdr_values.size),
        "dvdr_min_s-1": float(merged.dVdr_values[0]),
        "dvdr_max_s-1": float(merged.dVdr_values[-1]),
        "added_failure_nodes": int(np.count_nonzero(added.failure_mask)),
        "combined_failure_nodes": int(np.count_nonzero(merged.failure_mask)),
        "old_raw_nodes_preserved_exactly": True,
        "elapsed_seconds": elapsed,
    }
    report_path = output.with_suffix(".build.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[report] {report_path}")


if __name__ == "__main__":
    main()
