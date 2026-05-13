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
from despotic.chemistry import NL99, NL99_GC, GOW

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
    dVdr_grid: LogGrid,
    *,
    species_specs: Sequence[SpeciesSpec],
    chem_network=GOW,
    show_progress: bool = True,
    full_parallel: bool = False,
    workers: int | None = None,
    escape_geom: str = 'LVG',
) -> DespoticTable:
    """Build a 3D DESPOTIC table on (nH, colDen, dVdr).

    Architecture (validated empirically):
    - Outer loop over (nH, colDen): chemistry + thermal balance is solved
      ONCE per (nH, colDen) point with `evolveTemp="iterateDust"`.
    - Inner loop over dVdr: only line emission is recomputed
      (`cell.lineLum(escapeProbGeom='LVG')`) — no chemistry rerun.
    - Tg, abundances, mu, cv, Eint do not depend on dVdr; they are
      broadcast across the dVdr axis at storage time.
    - Line fields (lumPerH, tau, intIntensity, intTB, tauDust, freq)
      are genuinely 3D.
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
    Eint_dimless = np.full(shape, np.nan)

    line_buffers: dict[str, dict[str, np.ndarray]] = {
        spec.name: {
            field: np.full(shape, np.nan, dtype=float)
            for field in LINE_RESULT_FIELDS
        }
        for spec in specs
        if spec.is_emitter
    }
    energy_fields: dict[str, np.ndarray] = {}

    def _flatten_energy(term: str, value, target: dict[str, np.ndarray], col_i: int) -> None:
        """Flatten one (col, dVdr-broadcast) energy-term entry into target."""
        if isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                _flatten_energy(f"{term}.{sub_key}", sub_value, target, col_i)
            return
        grid = target.setdefault(term, np.full((num_cols, num_dvdr), np.nan, dtype=float))
        grid[col_i, :] = float(value)

    def _solve_row(row_idx: int) -> tuple[
        int,
        np.ndarray,                                # tg_row (num_cols, num_dvdr)
        np.ndarray,                                # failure_row
        dict[str, dict[str, np.ndarray]],          # line_rows
        dict[str, np.ndarray],                     # abundance_rows
        dict[str, np.ndarray],                     # energy_rows
        np.ndarray, np.ndarray, np.ndarray,        # mu/cv/eint
        list[AttemptRecord],
    ]:
        tg_row      = np.full((num_cols, num_dvdr), np.nan)
        failure_row = np.zeros((num_cols, num_dvdr), dtype=bool)
        mu_row      = np.full((num_cols, num_dvdr), np.nan)
        cv_row      = np.full((num_cols, num_dvdr), np.nan)
        eint_row    = np.full((num_cols, num_dvdr), np.nan)

        line_rows: dict[str, dict[str, np.ndarray]] = {
            spec.name: {field: np.full((num_cols, num_dvdr), np.nan) for field in LINE_RESULT_FIELDS}
            for spec in specs if spec.is_emitter
        }
        abundance_rows: dict[str, np.ndarray] = {
            spec.name: np.full((num_cols, num_dvdr), np.nan) for spec in specs
        }
        energy_rows: dict[str, np.ndarray] = {}
        attempts_row: list[AttemptRecord] = []

        emitter_names = [spec.name for spec in specs if spec.is_emitter]

        for col_idx, col_val in enumerate(col_vals):
            line_results, _emit_abunds, chem_abunds, mu_val, cv_val, eint_val, final_tg, energy_terms, failed = (
                calculate_single_despotic_point(
                    nH_val=nH_vals[row_idx],
                    colDen_val=col_val,
                    dvdr_grid=tuple(dvdr_vals.tolist()),
                    species=emitter_names,
                    abundance_only=tuple(spec.name for spec in specs if not spec.is_emitter),
                    chem_network=chem_network,
                    row_idx=row_idx,
                    col_idx=col_idx,
                    Tg_init=100.0,
                    log_failures=True,
                    attempt_log=attempts_row,
                    escape_geom=escape_geom,
                )
            )

            # 2D fields broadcast across the dvdr axis.
            tg_row[col_idx, :]      = final_tg
            failure_row[col_idx, :] = failed
            mu_row[col_idx, :]      = mu_val
            cv_row[col_idx, :]      = cv_val
            eint_row[col_idx, :]    = eint_val

            for spec in specs:
                abundance_rows[spec.name][col_idx, :] = chem_abunds.get(spec.name, float("nan"))

            # 3D line fields filled per dvdr index.
            for spec in specs:
                if not spec.is_emitter:
                    continue
                results_list = line_results.get(spec.name, [DEFAULT_LINE_RESULT] * num_dvdr)
                for d_idx, result in enumerate(results_list):
                    if d_idx >= num_dvdr:
                        break
                    for field in LINE_RESULT_FIELDS:
                        line_rows[spec.name][field][col_idx, d_idx] = getattr(result, field)

            for term, value in energy_terms.items():
                _flatten_energy(term, value, energy_rows, col_idx)

        return row_idx, tg_row, failure_row, line_rows, abundance_rows, energy_rows, mu_row, cv_row, eint_row, attempts_row

    # ============ Main process: collect parallel results ============

    if full_parallel:
        raise NotImplementedError("3D full_parallel is currently not updated. Please use row-wise parallel.")
    else:
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

        for row_idx, tg_row, failure_row, line_rows, abundance_rows, energy_rows, mu_row, cv_row, eint_row, attempts_row in results:
            tg_table[row_idx, :, :]    = tg_row
            failure_mask[row_idx, :, :] = failure_row
            mu_grid[row_idx, :, :]     = mu_row
            cv_grid[row_idx, :, :]     = cv_row
            Eint_dimless[row_idx, :, :] = eint_row
            attempts.extend(attempts_row)

            for name, row_values in abundance_rows.items():
                abundance_map[name][row_idx, :, :] = row_values
            for name, fields in line_rows.items():
                buffer = line_buffers[name]
                for field, values in fields.items():
                    buffer[field][row_idx, :, :] = values
            for term, values in energy_rows.items():
                grid = energy_fields.setdefault(term, np.full(shape, np.nan))
                grid[row_idx, :, :] = values

        failed_cells = int(np.count_nonzero(failure_mask))
        total_cells  = num_rows * num_cols * num_dvdr
        if failed_cells:
            LOGGER.warning("DESPOTIC table: %s/%s cells failed to converge", failed_cells, total_cells)
        else:
            LOGGER.info("DESPOTIC table converged for all %s cells", total_cells)

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
            dVdr_values=dvdr_vals,
            mu_values=mu_grid,
            cv_values=cv_grid,
            Eint_values=Eint_dimless,
            failure_mask=failure_mask,
            energy_terms=energy_fields or None,
            attempts=tuple(attempts),
        )


def plot_table(*_args, **_kwargs) -> None:
    """Placeholder for future plotting utilities."""
    raise NotImplementedError("plot_table is not implemented yet.")
