"""Parallel construction of the canonical 3D GOW/LVG table."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Sequence

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from .models import AttemptRecord, DespoticTable, LineLumResult, LogGrid, SpeciesLineGrid, SpeciesRecord
from .solver import CO21_TABLE_TOKEN, LINE_RESULT_FIELDS, solve_gow_lvg_point


LOGGER = logging.getLogger(__name__)
DEFAULT_LINE_RESULT = LineLumResult(*([float("nan")] * len(LINE_RESULT_FIELDS)))


@dataclass(frozen=True)
class SpeciesSpec:
    name: str
    is_emitter: bool


GOW_LVG_SPECIES: tuple[SpeciesSpec, ...] = (
    SpeciesSpec("CO", True),
    SpeciesSpec("C", True),
    SpeciesSpec("C+", True),
    SpeciesSpec("HCO+", True),
    SpeciesSpec("O", True),
    SpeciesSpec("e-", False),
    SpeciesSpec("H+", False),
    SpeciesSpec("H2", False),
    SpeciesSpec("H", False),
)


def build_gow_lvg_table(
    nH_grid: LogGrid,
    col_grid: LogGrid,
    dVdr_grid: LogGrid,
    *,
    species_specs: Sequence[SpeciesSpec] = GOW_LVG_SPECIES,
    show_progress: bool = True,
    workers: int | None = None,
) -> DespoticTable:
    """Build a true 3-input ``(nH, N_H, dVdr)`` GOW/LVG table.

    Each grid cell independently solves GOW chemistry and dust/gas thermal
    equilibrium.  Emitters are present during the equilibrium solve, so LVG
    line cooling contributes to the converged temperature.

    ``species_specs`` is exposed only so the sparse smoke test can exercise a
    cheaper subset.  The production CLI always uses :data:`GOW_LVG_SPECIES`.
    """
    specs = tuple(species_specs)
    nH_vals = nH_grid.sample()
    col_vals = col_grid.sample()
    dvdr_vals = dVdr_grid.sample()
    shape = (len(nH_vals), len(col_vals), len(dvdr_vals))
    num_rows, num_cols, num_dvdr = shape

    tg_table = np.full(shape, np.nan)
    failure_mask = np.zeros(shape, dtype=bool)
    abundance_map = {spec.name: np.full(shape, np.nan) for spec in specs}
    mu_grid = np.full(shape, np.nan)
    cv_grid = np.full(shape, np.nan)
    eint_grid = np.full(shape, np.nan)
    energy_fields: dict[str, np.ndarray] = {}

    emitter_names = tuple(spec.name for spec in specs if spec.is_emitter)
    abundance_only = tuple(spec.name for spec in specs if not spec.is_emitter)
    line_output_names = emitter_names + (
        (CO21_TABLE_TOKEN,) if "CO" in emitter_names else ()
    )
    line_buffers = {
        name: {field: np.full(shape, np.nan) for field in LINE_RESULT_FIELDS}
        for name in line_output_names
    }

    def _solve_row(row_idx: int):
        row_shape = (num_cols, num_dvdr)
        tg_row = np.full(row_shape, np.nan)
        failure_row = np.zeros(row_shape, dtype=bool)
        mu_row = np.full(row_shape, np.nan)
        cv_row = np.full(row_shape, np.nan)
        eint_row = np.full(row_shape, np.nan)
        line_rows = {
            name: {field: np.full(row_shape, np.nan) for field in LINE_RESULT_FIELDS}
            for name in line_output_names
        }
        abundance_rows = {spec.name: np.full(row_shape, np.nan) for spec in specs}
        energy_rows: dict[str, np.ndarray] = {}
        attempts_row: list[AttemptRecord] = []

        for col_idx, col_val in enumerate(col_vals):
            for dvdr_idx, dvdr_val in enumerate(dvdr_vals):
                result = solve_gow_lvg_point(
                    nH_val=float(nH_vals[row_idx]),
                    colDen_val=float(col_val),
                    dvdr_val=float(dvdr_val),
                    species=emitter_names,
                    abundance_only=abundance_only,
                    row_idx=row_idx,
                    col_idx=col_idx,
                    dvdr_idx=dvdr_idx,
                    Tg_init=100.0,
                    log_failures=True,
                    attempt_log=attempts_row,
                )
                line_results, chem_abunds, mu, cv, eint, tg, energy_terms, failed = result
                tg_row[col_idx, dvdr_idx] = tg
                failure_row[col_idx, dvdr_idx] = failed
                mu_row[col_idx, dvdr_idx] = mu
                cv_row[col_idx, dvdr_idx] = cv
                eint_row[col_idx, dvdr_idx] = eint

                for spec in specs:
                    abundance_rows[spec.name][col_idx, dvdr_idx] = chem_abunds.get(spec.name, np.nan)
                for name in line_output_names:
                    line = line_results.get(name, DEFAULT_LINE_RESULT)
                    for field in LINE_RESULT_FIELDS:
                        line_rows[name][field][col_idx, dvdr_idx] = getattr(line, field)
                for term, value in energy_terms.items():
                    energy_rows.setdefault(term, np.full(row_shape, np.nan))[col_idx, dvdr_idx] = value

        return row_idx, tg_row, failure_row, line_rows, abundance_rows, energy_rows, mu_row, cv_row, eint_row, attempts_row

    if workers is None:
        workers = -1
    solve_row = partial(_solve_row)
    tasks = range(num_rows)
    if show_progress:
        with tqdm_joblib(tqdm(total=num_rows, desc="DESPOTIC rows", unit="row")):
            results = Parallel(n_jobs=workers)(delayed(solve_row)(idx) for idx in tasks)
    else:
        results = Parallel(n_jobs=workers)(delayed(solve_row)(idx) for idx in tasks)

    attempts: list[AttemptRecord] = []
    for row_idx, tg_row, failure_row, line_rows, abundance_rows, energy_rows, mu_row, cv_row, eint_row, attempts_row in results:
        tg_table[row_idx] = tg_row
        failure_mask[row_idx] = failure_row
        mu_grid[row_idx] = mu_row
        cv_grid[row_idx] = cv_row
        eint_grid[row_idx] = eint_row
        attempts.extend(attempts_row)
        for name, values in abundance_rows.items():
            abundance_map[name][row_idx] = values
        for name, fields in line_rows.items():
            for field, values in fields.items():
                line_buffers[name][field][row_idx] = values
        for term, values in energy_rows.items():
            energy_fields.setdefault(term, np.full(shape, np.nan))[row_idx] = values

    failed_cells = int(np.count_nonzero(failure_mask))
    if failed_cells:
        LOGGER.warning("GOW/LVG table: %s/%s cells failed", failed_cells, failure_mask.size)

    species_data: dict[str, SpeciesRecord] = {}
    for spec in specs:
        abundance = abundance_map[spec.name]
        line = None
        if spec.is_emitter:
            fields = line_buffers[spec.name]
            line = SpeciesLineGrid(
                freq=fields["freq"], intIntensity=fields["intIntensity"], intTB=fields["intTB"],
                lumPerH=fields["lumPerH"], tau=fields["tau"], tauDust=fields["tauDust"],
                abundance=abundance,
            )
        species_data[spec.name] = SpeciesRecord(spec.name, abundance, line, spec.is_emitter)

    if CO21_TABLE_TOKEN in line_buffers:
        fields = line_buffers[CO21_TABLE_TOKEN]
        co_abundance = abundance_map["CO"]
        line = SpeciesLineGrid(
            freq=fields["freq"], intIntensity=fields["intIntensity"], intTB=fields["intTB"],
            lumPerH=fields["lumPerH"], tau=fields["tau"], tauDust=fields["tauDust"],
            abundance=co_abundance,
        )
        species_data[CO21_TABLE_TOKEN] = SpeciesRecord(
            CO21_TABLE_TOKEN, co_abundance, line, True,
        )

    return DespoticTable(
        species_data=species_data,
        tg_final=tg_table,
        nH_values=nH_vals,
        col_density_values=col_vals,
        dVdr_values=dvdr_vals,
        mu_values=mu_grid,
        cv_values=cv_grid,
        Eint_values=eint_grid,
        failure_mask=failure_mask,
        energy_terms=energy_fields or None,
        attempts=tuple(attempts),
    )
