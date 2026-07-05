"""Versioned NPZ I/O for the canonical 3D GOW/LVG table."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .models import AttemptRecord, DespoticTable, SpeciesLineGrid, SpeciesRecord


TABLE_VERSION = 5
SUPPORTED_TABLE_VERSIONS = (4, 5)
_LINE_FIELDS = ("freq", "intIntensity", "intTB", "lumPerH", "tau", "tauDust")


def _attempts_to_array(attempts: Iterable[AttemptRecord]) -> np.ndarray:
    records = list(attempts)
    arr = np.empty(len(records), dtype=[
        ("row_idx", np.int32), ("col_idx", np.int32),
        ("dvdr_idx", np.int32), ("nH", float), ("colDen", float), ("dvdr", float),
        ("tg_guess", float), ("final_Tg", float), ("converged", np.bool_),
        ("message", object), ("duration", float),
    ])
    for idx, rec in enumerate(records):
        arr[idx] = (
            rec.row_idx, rec.col_idx, -1 if rec.dvdr_idx is None else rec.dvdr_idx,
            rec.nH, rec.colDen, np.nan if rec.dvdr is None else rec.dvdr,
            rec.tg_guess, rec.final_Tg, rec.converged, rec.message,
            np.nan if rec.duration is None else rec.duration,
        )
    return arr


def _attempts_from_array(data: np.ndarray) -> tuple[AttemptRecord, ...]:
    names = set(data.dtype.names or ())
    attempts = []
    for row in data:
        dvdr_idx = int(row["dvdr_idx"]) if "dvdr_idx" in names else -1
        dvdr = float(row["dvdr"]) if "dvdr" in names else np.nan
        attempts.append(AttemptRecord(
            row_idx=int(row["row_idx"]), col_idx=int(row["col_idx"]),
            nH=float(row["nH"]), colDen=float(row["colDen"]),
            tg_guess=float(row["tg_guess"]), final_Tg=float(row["final_Tg"]),
            converged=bool(row["converged"]),
            message=str(row["message"]) if row["message"] else None,
            duration=None if np.isnan(row["duration"]) else float(row["duration"]),
            dvdr_idx=None if dvdr_idx < 0 else dvdr_idx,
            dvdr=None if np.isnan(dvdr) else dvdr,
        ))
    return tuple(attempts)


def save_table(table: DespoticTable, path: str | Path) -> None:
    """Atomically save a version-5 table."""
    path = Path(path)
    payload: dict[str, np.ndarray] = {
        "version": np.array([TABLE_VERSION], dtype=np.int32),
        "chemistry_network": np.array(table.chemistry_network),
        "escape_geometry": np.array(table.escape_geometry),
        "temperature_mode": np.array(table.temperature_mode),
        "nH_values": np.asarray(table.nH_values),
        "col_density_values": np.asarray(table.col_density_values),
        "dVdr_values": np.asarray(table.dVdr_values),
        "tg_final": np.asarray(table.tg_final),
        "mu_values": np.asarray(table.mu_values),
        "cv_values": np.asarray(table.cv_values),
        "Eint_values": np.asarray(table.Eint_values),
        "species_names": np.array(table.species, dtype=object),
        "species_is_emitter": np.array([table.species_data[n].is_emitter for n in table.species]),
        "attempts": _attempts_to_array(table.attempts),
    }
    if table.failure_mask is not None:
        payload["failure_mask"] = np.asarray(table.failure_mask, dtype=bool)
    for name, record in table.species_data.items():
        payload[f"{name}_abundance"] = np.asarray(record.abundance)
        if record.line is not None:
            for field in _LINE_FIELDS:
                payload[f"{name}_{field}"] = np.asarray(getattr(record.line, field))
    if table.energy_terms:
        payload["energy_term_names"] = np.array(tuple(table.energy_terms), dtype=object)
        for name, values in table.energy_terms.items():
            payload[f"energy::{name}"] = np.asarray(values)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _scalar_string(blob, name: str, default: str) -> str:
    return str(np.asarray(blob[name]).item()) if name in blob.files else default


def load_table(path: str | Path) -> DespoticTable:
    """Load current version 5 or the existing version-4 GOW/LVG table."""
    with np.load(Path(path), allow_pickle=True) as blob:
        version = int(np.asarray(blob["version"]).flat[0])
        if version not in SUPPORTED_TABLE_VERSIONS:
            raise ValueError(
                f"Unsupported DESPOTIC table version {version}; supported: {SUPPORTED_TABLE_VERSIONS}"
            )
        species_names = [str(value) for value in blob["species_names"]]
        emitter_flags = np.asarray(blob["species_is_emitter"], dtype=bool)
        species_data: dict[str, SpeciesRecord] = {}
        for name, is_emitter in zip(species_names, emitter_flags):
            abundance = np.asarray(blob[f"{name}_abundance"], dtype=float)
            line = None
            if is_emitter:
                fields: Mapping[str, np.ndarray] = {
                    field: np.asarray(blob[f"{name}_{field}"], dtype=float)
                    for field in _LINE_FIELDS
                }
                line = SpeciesLineGrid(**fields, abundance=abundance)
            species_data[name] = SpeciesRecord(name, abundance, line, bool(is_emitter))

        energy_terms = None
        if "energy_term_names" in blob.files:
            energy_terms = {
                str(name): np.asarray(blob[f"energy::{name}"], dtype=float)
                for name in blob["energy_term_names"]
            }
        failure_mask = (
            np.asarray(blob["failure_mask"], dtype=bool)
            if "failure_mask" in blob.files else None
        )
        attempts = _attempts_from_array(blob["attempts"]) if "attempts" in blob.files else ()
        return DespoticTable(
            species_data=species_data,
            tg_final=np.asarray(blob["tg_final"], dtype=float),
            nH_values=np.asarray(blob["nH_values"], dtype=float),
            col_density_values=np.asarray(blob["col_density_values"], dtype=float),
            dVdr_values=np.asarray(blob["dVdr_values"], dtype=float),
            mu_values=np.asarray(blob["mu_values"], dtype=float),
            cv_values=np.asarray(blob["cv_values"], dtype=float),
            Eint_values=np.asarray(blob["Eint_values"], dtype=float),
            failure_mask=failure_mask,
            energy_terms=energy_terms,
            attempts=attempts,
            chemistry_network=_scalar_string(blob, "chemistry_network", "GOW"),
            escape_geometry=_scalar_string(blob, "escape_geometry", "LVG"),
            temperature_mode=_scalar_string(blob, "temperature_mode", "iterateDust"),
        )
