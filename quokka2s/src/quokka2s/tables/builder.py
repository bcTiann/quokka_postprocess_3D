from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="collision rates not available")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Mapping, Sequence

import logging
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from despotic.chemistry import NL99, NL99_GC

from .models import (
    AttemptRecord,
    DespoticTable,
    LineLumResult,
    LogGrid,
    SpeciesLineGrid,
    SpeciesRecord,
)
from .solver import LINE_RESULT_FIELDS, calculate_single_despotic_point


LOGGER = logging.getLogger(__name__)
DEFAULT_LINE_RESULT = LineLumResult(*([float("nan")] * len(LINE_RESULT_FIELDS)))

@dataclass(frozen=True)
class SpeciesSpec:
    name: str
    is_emitter: bool

SPECIES_SPECS = (
    SpeciesSpec("H+", False),
    SpeciesSpec("CO", True),
    SpeciesSpec("C", True),
    SpeciesSpec("C+", True),
    SpeciesSpec("HCO+", True),
    SpeciesSpec("e-", False),
)



def build_table(
    nH_grid: LogGrid,
    col_grid: LogGrid,
    tg_guesses: Sequence[float],
    *,
    species_specs: Sequence[SpeciesSpec],
    chem_network=NL99,
    show_progress: bool = True,
    workers: int | None = None,
) -> DespoticTable:
    """
    Run DESPOTIC across logarithmic density/column grids and cache the results.

    Parameters
    ----------
    nH_grid : LogGrid
        Hydrogen number-density grid (cm^-3); defines table rows.
    col_grid : LogGrid
        Column-density grid ; defines table columns.
    tg_guesses : Sequence[float]
        Initial gas-temperature guesses passed to the DESPOTIC solver.
    species : Sequence[str], optional
        Species for which line luminosities/abundances are tabulated.
    chem_network : callable, optional
        Chemistry network constructor provided to DESPOTIC (defaults to NL99_GC).
    show_progress : bool, optional
        Whether to wrap the row loop with tqdm progress reporting.

    Returns
    -------
    DespoticTable
        Structured table containing per-species line grids, final gas temperatures,
        failure mask, and heating/cooling energy rates on the sampled grid.
    """

    specs = tuple(species_specs)
    nH_vals = nH_grid.sample()
    col_vals = col_grid.sample()

    num_rows = len(nH_vals)
    num_cols = len(col_vals) 
    tg_table = np.full((num_rows, num_cols), np.nan, dtype=float)
    failure_mask = np.zeros((num_rows, num_cols), dtype=bool)

    abundance_map: dict[str, np.ndarray] = {
        spec.name: np.full((num_rows, num_cols), np.nan, dtype=float) for spec in specs
    }
    line_buffers: dict[str, dict[str, np.ndarray]] = {
        spec.name: {
            field: np.full((num_rows, num_cols), np.nan, dtype=float)
            for field in LINE_RESULT_FIELDS
        }
        for spec in specs
        if spec.is_emitter
    }
    energy_fields: dict[str, np.ndarray] = {}

    def _flatten_energy(term: str, value, target: dict[str, np.ndarray], idx: int) -> None:
        if isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                _flatten_energy(f"{term}.{sub_key}", sub_value, target, idx)
            return
        grid = target.setdefault(term, np.full(len(col_vals), np.nan, dtype=float))
        grid[idx] = float(value)

    def _solve_row(row_idx: int) -> tuple[
        int, np.ndarray, np.ndarray, dict[str, dict[str, np.ndarray]],
        dict[str, np.ndarray], dict[str, np.ndarray], list[AttemptRecord]
    ]:
        tg_row = np.full(col_vals.shape, np.nan, dtype=float)
        failure_row = np.zeros(col_vals.shape, dtype=bool)
        line_rows: dict[str, dict[str, np.ndarray]] = {
            spec.name: {field: np.full(col_vals.shape, np.nan, dtype=float) for field in LINE_RESULT_FIELDS}
            for spec in specs if spec.is_emitter
        }
        abundance_rows: dict[str, np.ndarray] = {
            spec.name: np.full(col_vals.shape, np.nan, dtype=float) for spec in specs
        }
        energy_rows: dict[str, np.ndarray] = {}
        attempts_row: list[AttemptRecord] = []

        for col_idx, col_val in enumerate(col_vals):
            line_results, emitter_abunds, chem_abunds, final_tg, energy_terms, failed = (
                calculate_single_despotic_point(
                    nH_val=nH_vals[row_idx],
                    colDen_val=col_val,
                    initial_Tg_guesses=tg_guesses,
                    species=[spec.name for spec in specs if spec.is_emitter],
                    abundance_only=[spec.name for spec in specs if not spec.is_emitter],
                    chem_network=chem_network,
                    row_idx=row_idx,
                    col_idx=col_idx,
                    log_failures=True,
                    attempt_log=attempts_row,
                )
            )
            tg_row[col_idx] = final_tg
            failure_row[col_idx] = failed

            for spec in specs:
                abundance_rows[spec.name][col_idx] = chem_abunds.get(spec.name, float("nan"))
                if spec.is_emitter:
                    result = line_results.get(spec.name, DEFAULT_LINE_RESULT)
                    for field in LINE_RESULT_FIELDS:
                        line_rows[spec.name][field][col_idx] = getattr(result, field)

            for term, value in energy_terms.items():
                _flatten_energy(term, value, energy_rows, col_idx)

        return row_idx, tg_row, failure_row, line_rows, abundance_rows, energy_rows, attempts_row

    if workers is None:
        workers = -1
    tasks = range(num_rows)
    solve_row = partial(_solve_row)

    if show_progress:
        progress = tqdm(total=num_rows, desc="DESPOTIC rows", unit="row")
        with tqdm_joblib(progress):
            results = Parallel(n_jobs=workers)(delayed(solve_row)(row_idx) for row_idx in tasks)
    else:
        results = Parallel(n_jobs=workers)(delayed(solve_row)(row_idx) for row_idx in tasks)

    attempts: list[AttemptRecord] = []
    for row_idx, tg_row, failure_row, line_rows, abundance_rows, energy_rows, attempts_row in results:
        tg_table[row_idx, :] = tg_row
        failure_mask[row_idx, :] = failure_row
        attempts.extend(attempts_row)

        for name, row_values in abundance_rows.items():
            abundance_map[name][row_idx, :] = row_values
        for name, fields in line_rows.items():
            buffer = line_buffers[name]
            for field, values in fields.items():
                buffer[field][row_idx, :] = values
        for term, values in energy_rows.items():
            grid = energy_fields.setdefault(term, np.full((num_rows, num_cols), np.nan, dtype=float))
            grid[row_idx, :] = values

    failed_cells = int(np.count_nonzero(failure_mask))
    if failed_cells:
        LOGGER.warning("DESPOTIC table: %s/%s cells failed to converge", failed_cells, num_rows * num_cols)
    else:
        LOGGER.info("DESPOTIC table converged for all %s cells", num_rows * num_cols)

    species_data: dict[str, SpeciesRecord] = {}
    for spec in specs:
        abundance_grid = abundance_map[spec.name]
        line_grid = None
        if spec.is_emitter:
            buf = line_buffers[spec.name]
            line_grid = SpeciesLineGrid(
                freq=buf["freq"],
                intIntensity=buf["intIntensity"],
                intTB=buf["intTB"],
                lumPerH=buf["lumPerH"],
                tau=buf["tau"],
                tauDust=buf["tauDust"],
                abundance=abundance_grid,
            )
        species_data[spec.name] = SpeciesRecord(
            name=spec.name,
            abundance=abundance_grid,
            line=line_grid,
            is_emitter=spec.is_emitter,
        )

    return DespoticTable(
        species_data=species_data,
        tg_final=tg_table,
        nH_values=nH_vals,
        col_density_values=col_vals,
        failure_mask=failure_mask,
        energy_terms=energy_fields or None,
        attempts=tuple(attempts),
    )



def plot_table(*_args, **_kwargs) -> None:
    """Placeholder for future plotting utilities."""
    raise NotImplementedError("plot_table is not implemented yet.")
