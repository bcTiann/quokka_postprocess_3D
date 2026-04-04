# quokka2s/src/quokka2s/tables/io.py
from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="collision rates not available",
    category=UserWarning,
    module=r"DESPOTIC.*emitterData",
)
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in log",
    category=RuntimeWarning,
    module=r"DESPOTIC.*NL99_GC",
)

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .models import AttemptRecord, DespoticTable, SpeciesLineGrid, SpeciesRecord

TABLE_VERSION = 2
_LINE_FIELDS = ("freq", "intIntensity", "intTB", "lumPerH", "tau", "tauDust")


def _attempts_to_array(attempts: Iterable[AttemptRecord]) -> np.ndarray:
    records = list(attempts)
    arr = np.empty(len(records), dtype=[
        ("row_idx", np.int32),
        ("col_idx", np.int32),
        ("nH", float),
        ("colDen", float),
        ("tg_guess", float),
        ("final_Tg", float),
        ("converged", np.bool_),
        ("message", object),
        ("duration", float),
    ])
    for idx, rec in enumerate(records):
        arr[idx] = (
            rec.row_idx,
            rec.col_idx,
            rec.nH,
            rec.colDen,
            rec.tg_guess,
            rec.final_Tg,
            rec.converged,
            rec.message,
            rec.duration if rec.duration is not None else np.nan,
        )
    return arr


def _attempts_from_array(data: np.ndarray) -> tuple[AttemptRecord, ...]:
    attempts: list[AttemptRecord] = []
    for row in data:
        attempts.append(
            AttemptRecord(
                row_idx=int(row["row_idx"]),
                col_idx=int(row["col_idx"]),
                nH=float(row["nH"]),
                colDen=float(row["colDen"]),
                tg_guess=float(row["tg_guess"]),
                final_Tg=float(row["final_Tg"]),
                converged=bool(row["converged"]),
                message=row["message"] if row["message"] else None,
                duration=None if np.isnan(row["duration"]) else float(row["duration"]),
            )
        )
    return tuple(attempts)



def save_table(table: DespoticTable, path: str | Path) -> None:
    path = Path(path)
    payload: dict[str, np.ndarray] = {
        "version": np.array([TABLE_VERSION], dtype=np.int32),
        "nH_values": np.array(table.nH_values),
        "col_density_values": np.array(table.col_density_values),
        "tg_final": np.array(table.tg_final),
        "failure_mask": np.asarray(table.failure_mask) if table.failure_mask is not None else None,
        "species_names": np.array(table.species, dtype=object),
        "species_is_emitter": np.array([table.species_data[name].is_emitter for name in table.species], dtype=np.bool_),
        "attempts": _attempts_to_array(table.attempts),
    }
    for species, record in table.species_data.items():
        payload[f"{species}_abundance"] = np.array(record.abundance)
        if record.is_emitter and record.line is not None:
            for field in _LINE_FIELDS:
                payload[f"{species}_{field}"] = np.array(getattr(record.line, field))
    if table.energy_terms:
        payload["energy_term_names"] = np.array(list(table.energy_terms.keys()), dtype=object)
        for name, grid in table.energy_terms.items():
            payload[f"energy::{name}"] = np.asarray(grid)

    payload = {key: value for key, value in payload.items() if value is not None} # Remove any None (key, value) pairs
    np.savez_compressed(path, **payload)


def load_table(path: str | Path) -> DespoticTable:
    path = Path(path)
    blob = np.load(path, allow_pickle=True)
    version = int(blob["version"])

    if version != TABLE_VERSION:
        raise ValueError(f"Unsupported DESPOTIC table version: {version} (expected {TABLE_VERSION})")
    
    nH_values = np.array(blob["nH_values"])
    col_density_values = np.array(blob["col_density_values"], dtype=float)
    tg_final = np.array(blob["tg_final"], dtype=float)

    failure_mask = blob.get("failure_mask")
    if failure_mask is not None:
        failure_mask = np.array(failure_mask, dtype=bool)

    energy_fields = None
    energy_names = blob.get("energy_term_names")
    if energy_names is not None:
        energy_fields = {
            str(name): np.array(blob[f"energy::{name}"], dtype=float)
            for name in energy_names
        }

    species_names = [str(x) for x in blob["species_names"]]
    emitter_flags = np.array(blob["species_is_emitter"], dtype=bool)
    species_data: dict[str, SpeciesRecord] = {}
    for name, is_emitter in zip(species_names, emitter_flags):
        abundance = np.array(blob[f"{name}_abundance"], dtype=float)
        line_grid = None
        if is_emitter:
            fields: Mapping[str, np.ndarray] = {
                field: np.array(blob[f"{name}_{field}"], dtype=float)
                for field in _LINE_FIELDS
            }
            line_grid = SpeciesLineGrid(**fields, abundance=abundance)
        species_data[name] = SpeciesRecord(
            name=name,
            abundance=abundance,
            line=line_grid,
            is_emitter=bool(is_emitter),
        )
        
    attempts = _attempts_from_array(blob["attempts"])

    return DespoticTable(
        species_data=species_data,
        tg_final=tg_final,
        nH_values=nH_values,
        col_density_values=col_density_values,
        attempts=attempts,
        failure_mask=failure_mask,
        energy_terms=energy_fields,
    )
