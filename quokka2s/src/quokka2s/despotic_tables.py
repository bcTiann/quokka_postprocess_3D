"""Utilities for building DESPOTIC lookup tables."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Mapping
from types import MappingProxyType
import warnings

import contextlib
import io
import logging
import time
import math
import re


import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import RectBivariateSpline, griddata
try:  # prefer rich-rendered progress bars for better visibility
    from tqdm.rich import tqdm
except Exception:  # pragma: no cover - fallback when rich support unavailable
    from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib


from despotic import cloud
from despotic.chemistry import NL99, NL99_GC, GOW

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class LogGrid:
    """Logarithmically spaced grid specification."""

    min_value: float
    max_value: float
    num_points: int
    round_digits: int | None = None

    def __post_init__(self) -> None:
        if self.min_value <= 0 or self.max_value <= 0:
            raise ValueError("LogGrid values must be positive for log10 sampling")
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be smaller than max_value")
        if self.num_points < 2:
            raise ValueError("num_points must be >= 2")

    def sample(self) -> np.ndarray:
        values = np.logspace(np.log10(self.min_value), np.log10(self.max_value), self.num_points)
        if self.round_digits is not None:
            values = np.round(values, self.round_digits)
        return values

@dataclass(frozen=True)
class LineLumResult:
    """Single-species line luminosity output from DESPOTIC."""

    int_tb: float
    int_intensity: float
    lum_per_h: float
    tau: float
    tau_dust: float
    tex: float
    freq: float


@dataclass(frozen=True)
class SpeciesLineGrid:
    """Grid of line luminosity outputs for an emitting species."""

    int_tb: np.ndarray
    int_intensity: np.ndarray
    lum_per_h: np.ndarray
    tau: np.ndarray
    tau_dust: np.ndarray
    tex: np.ndarray
    freq: np.ndarray


@dataclass(frozen=True)
class AttemptRecord:
    row_idx: int | None
    col_idx: int | None
    nH: float
    colDen: float
    tg_guess: float
    final_Tg: float
    attempt_number: int
    attempt_type: str  # "successful" / "single_attempt" / "exception" / "all_guesses_failed"
    converged: bool
    repeat_equilibrium: int
    line_results: Mapping[str, LineLumResult] = field(default_factory=dict)
    residual_trace: tuple[float, ...] = tuple()
    temperature_converged: bool = False
    error_message: str | None = None
    duration: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "line_results", MappingProxyType(dict(self.line_results)))

    def _line_result(self, species: str = "CO") -> LineLumResult | None:
        if species in self.line_results:
            return self.line_results[species]
        if self.line_results:
            return next(iter(self.line_results.values()))
        return None

    @property
    def co_int_TB(self) -> float:
        result = self._line_result()
        return result.int_tb if result is not None else float("nan")

    @property
    def int_intensity(self) -> float:
        result = self._line_result()
        return result.int_intensity if result is not None else float("nan")

    @property
    def lum_per_h(self) -> float:
        result = self._line_result()
        return result.lum_per_h if result is not None else float("nan")

    @property
    def tau(self) -> float:
        result = self._line_result()
        return result.tau if result is not None else float("nan")

    @property
    def tau_dust(self) -> float:
        result = self._line_result()
        return result.tau_dust if result is not None else float("nan")

    @property
    def tex(self) -> float:
        result = self._line_result()
        return result.tex if result is not None else float("nan")

    @property
    def frequency(self) -> float:
        result = self._line_result()
        return result.freq if result is not None else float("nan")

    @property
    def max_residual(self) -> float:
        if not self.residual_trace:
            return float("nan")
        return max(self.residual_trace)


@dataclass(frozen=True)
class DespoticTable:
    """Container for DESPOTIC lookup table outputs."""

    species_data: Mapping[str, SpeciesLineGrid]
    tg_final: np.ndarray
    nH_values: np.ndarray
    col_density_values: np.ndarray
    emitter_abundances: Mapping[str, float]
    attempts: Tuple[AttemptRecord, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "species_data", MappingProxyType(dict(self.species_data)))
        object.__setattr__(self, "emitter_abundances", MappingProxyType(dict(self.emitter_abundances)))

    @property
    def species(self) -> Tuple[str, ...]:
        return tuple(self.species_data.keys())
    
    def require_species(self, species: str) -> tuple[str, SpeciesLineGrid]:
        if not self.species:
            raise ValueError("DespoticTable contains no species data.")
        if species not in self.species_data:
            available = ", ".join(self.species)
            raise ValueError(
                f"Requested species '{species}' not found; available species: {available}"
            )
        return species, self.species_data[species]



    def get_species_grid(self, species: str) -> SpeciesLineGrid:
        return self.species_data[species]

    def clone_species_fields(self) -> dict[str, dict[str, np.ndarray]]:
        field_map: dict[str, dict[str, np.ndarray]] = {}
        for name, grid in self.species_data.items():
            field_map[name] = {
                field: np.array(getattr(grid, field), copy=True)  # type: ignore[arg-type]
                for field in LINE_RESULT_FIELDS
            }
        return field_map

    def with_updated_fields(
        self,
        *,
        species_fields: Mapping[str, Mapping[str, np.ndarray]],
        tg_final: np.ndarray | None = None,
        attempts: Sequence[AttemptRecord] | None = None,
    ) -> "DespoticTable":
        updated_species = {
            name: SpeciesLineGrid(
                int_tb=np.asarray(fields["int_tb"]),
                int_intensity=np.asarray(fields["int_intensity"]),
                lum_per_h=np.asarray(fields["lum_per_h"]),
                tau=np.asarray(fields["tau"]),
                tau_dust=np.asarray(fields["tau_dust"]),
                tex=np.asarray(fields["tex"]),
                freq=np.asarray(fields["freq"]),
            )
            for name, fields in species_fields.items()
        }
        tg_array = np.asarray(tg_final) if tg_final is not None else self.tg_final
        attempt_records: Tuple[AttemptRecord, ...]
        if attempts is None:
            attempt_records = self.attempts
        else:
            attempt_records = tuple(attempts)
        return DespoticTable(
            species_data=updated_species,
            tg_final=tg_array,
            nH_values=self.nH_values,
            col_density_values=self.col_density_values,
            emitter_abundances=self.emitter_abundances,
            attempts=attempt_records,
        )




CO_ABUNDANCE = 8.0e-9
CP_ABUNDANCE = 1.1e-4
DEFAULT_EMITTER_ABUNDANCES: Mapping[str, float] = MappingProxyType(
    {"CO": CO_ABUNDANCE, "C+": CP_ABUNDANCE}
)



LINE_RESULT_FIELDS: Tuple[str, ...] = (
    "int_tb",
    "int_intensity",
    "lum_per_h",
    "tau",
    "tau_dust",
    "tex",
    "freq",
)

_NAN_LINE_RESULT = LineLumResult(
    int_tb=float("nan"),
    int_intensity=float("nan"),
    lum_per_h=float("nan"),
    tau=float("nan"),
    tau_dust=float("nan"),
    tex=float("nan"),
    freq=float("nan"),
)


def _nan_line_result() -> LineLumResult:
    return _NAN_LINE_RESULT


def _empty_line_results(species: Sequence[str]) -> dict[str, LineLumResult]:
    return {sp: _nan_line_result() for sp in species}


def _extract_line_result(transitions: Sequence[Mapping[str, float]]) -> LineLumResult:
    if not transitions:
        return _nan_line_result()
    entry = transitions[0]
    return LineLumResult(
        int_tb=float(entry.get("intTB", float("nan"))),
        int_intensity=float(entry.get("intIntensity", float("nan"))),
        lum_per_h=float(entry.get("lumPerH", float("nan"))),
        tau=float(entry.get("tau", float("nan"))),
        tau_dust=float(entry.get("tauDust", float("nan"))),
        tex=float(entry.get("Tex", float("nan"))),
        freq=float(entry.get("freq", float("nan"))),
    )


_RESIDUAL_RE = re.compile(r"residual\s*=\s*([0-9eE.+-]+)")


def _extract_residuals(text: str) -> list[float]:
    return [float(match) for match in _RESIDUAL_RE.findall(text)]


def _log_despotic_stdout(output: io.StringIO | str) -> None:
    """Flush redirected stdout from DESPOTIC calls into the logger."""
    if isinstance(output, io.StringIO):
        text = output.getvalue()
        output.truncate(0)
        output.seek(0)
    else:
        text = output
    if not text:
        return
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("make: ***"):
            continue
        if stripped.startswith("setChemEquil:") or "Temperature converged!" in stripped:
            LOGGER.debug("DESPOTIC: %s", stripped)
            continue
        LOGGER.warning("DESPOTIC: %s", stripped)


def calculate_single_despotic_point(
    nH_val: float,
    colDen_val: float,
    initial_Tg_guesses: Sequence[float],
    *,
    log_failures: bool = False,
    emitter_abundances: Mapping[str, float] = DEFAULT_EMITTER_ABUNDANCES,
    repeat_equilibrium: int = 0,
    chem_network=NL99,
    row_idx: int | None = None,
    col_idx: int | None = None,
    attempt_log: list[AttemptRecord] | None = None,
    reuse_failed_tg: bool = False,
    reuse_max_insertions: int = 3,
    dVdr_val: float = 1e-14,
) -> Tuple[Mapping[str, LineLumResult], float]:
    """Run DESPOTIC for one (nH, column density) pair.

    Returns
    -------
    Tuple[Mapping[str, LineLumResult], float]
        Mapping from species name to line luminosity metrics together with
        the final gas temperature. Values are ``nan`` if all guesses fail.
    """
    species_order = tuple(emitter_abundances.keys())
#   emitter_abundances = {"CO": 8.0e-9, "C+": 1.1e-4}
#   species_order = tuple(emitter_abundances.keys())  # 结果是 ("CO", "C+")

    last_guess: float | None = None
    attempt_number = 0
    last_final_tg = float("nan")
    last_line_results: dict[str, LineLumResult] = _empty_line_results(species_order)
    ######
    # {
    # "CO": LineLumResult(int_tb=nan, int_intensity=nan, lum_per_h=nan,
    #                     tau=nan, tau_dust=nan, tex=nan, freq=nan),
    # "C+": LineLumResult(... 全是 nan ...)
    # }
    #######
    last_residual_trace: tuple[float, ...] = tuple()
    last_temp_converged = False
    last_duration = float("nan")

    pending_guesses: list[float] = [float(g) for g in initial_Tg_guesses]
    seen_guesses: list[float] = []
    reuse_insertions = 0

    def _should_skip(value: float) -> bool:
        return any(math.isclose(value, prev, rel_tol=1e-3, abs_tol=1e-2) for prev in seen_guesses)

    def _enqueue_retry(value: float) -> None:
        nonlocal reuse_insertions
        if not reuse_failed_tg:
            return
        if reuse_insertions >= reuse_max_insertions:
            return
        if not np.isfinite(value) or value <= 0:
            return
        value = float(value)
        if _should_skip(value):
            return
        pending_guesses.insert(0, value)
        reuse_insertions += 1

    while pending_guesses:
        guess = pending_guesses.pop(0)
        if reuse_failed_tg and _should_skip(guess):
            continue
        seen_guesses.append(guess)
        attempt_number += 1
        last_guess = guess
        attempt_start = time.perf_counter()
        try:
            temp_converged_run = False
            cell = cloud()
            cell.noWarn = True
            cell.nH = nH_val
            cell.colDen = colDen_val
            cell.Tg = guess

            cell.sigmaNT = 2.0e5
            cell.dVdr = dVdr_val
            cell.comp.xoH2 = 0.1
            cell.comp.xpH2 = 0.4
            cell.comp.xHe = 0.1

            cell.dust.alphaGD = 3.2e-34
            cell.dust.sigma10 = 2.0e-25
            cell.dust.sigmaPE = 1.0e-21
            cell.dust.sigmaISRF = 3.0e-22
            cell.dust.beta = 2.0
            cell.dust.Zd = 1.0

            cell.Td = 10.0
            cell.rad.TCMB = 2.73
            cell.rad.TradDust = 0.0
            cell.rad.ionRate = 2.0e-17
            cell.rad.chi = 1.0

            cell.comp.computeDerived(cell.nH)


            residual_trace_run: list[float] = []
            stdout_buffer = io.StringIO()
            # cell.setTempEq()
            with contextlib.redirect_stdout(stdout_buffer):
                converge = cell.setChemEq(
                    network=chem_network,
                    tol=1e-6, 
                    maxTime=1e18,
                    maxTempIter=200,
                    evolveTemp="iterateDust",
                    verbose=True,
                )

            output = stdout_buffer.getvalue()
            residual_trace_run.extend(_extract_residuals(output))
            temp_converged_run = temp_converged_run or "Temperature converged!" in output
            _log_despotic_stdout(output)

            if not converge:
                final_tg = float(cell.Tg)
                last_final_tg = final_tg
                line_results = _empty_line_results(species_order)
                last_line_results = dict(line_results)
                last_residual_trace = tuple(residual_trace_run)
                last_temp_converged = temp_converged_run
                if attempt_log is not None:
                    attempt_log.append(
                        AttemptRecord(
                            row_idx=row_idx,
                            col_idx=col_idx,
                            nH=cell.nH,
                            colDen=cell.colDen,
                            tg_guess=guess,
                            final_Tg=final_tg,
                            attempt_number=attempt_number,
                            attempt_type="single_attempt",
                            converged=False,
                            repeat_equilibrium=repeat_equilibrium,
                            line_results=line_results,
                            residual_trace=last_residual_trace,
                            temperature_converged=temp_converged_run,
                        )
                    )
                _enqueue_retry(final_tg)
                continue

            line_results: dict[str, LineLumResult] = {}
            for species in species_order:
                stdout_buffer = io.StringIO()
                with contextlib.redirect_stdout(stdout_buffer):
                    transitions = cell.lineLum(species, escapeProbGeom='LVG')
                _log_despotic_stdout(stdout_buffer.getvalue())
                line_results[species] = _extract_line_result(transitions)

            final_Tg = float(cell.Tg)
            last_final_tg = final_Tg
            last_line_results = dict(line_results)
            last_residual_trace = tuple(residual_trace_run)
            last_temp_converged = temp_converged_run
            attempt_duration = time.perf_counter() - attempt_start
            last_duration = attempt_duration

            if attempt_log is not None:
                attempt_log.append(
                    AttemptRecord(
                        row_idx=row_idx,
                        col_idx=col_idx,
                        nH=cell.nH,
                        colDen=cell.colDen,
                        tg_guess=guess,
                        final_Tg=final_Tg,
                        attempt_number=attempt_number,
                        attempt_type="successful",
                        converged=True,
                        repeat_equilibrium=repeat_equilibrium,
                        line_results=line_results,
                        residual_trace=last_residual_trace,
                        temperature_converged=temp_converged_run,
                        duration=attempt_duration,
                    )
                )
            return MappingProxyType(dict(line_results)), final_Tg

        except Exception as exc:  # pragma: no cover - DESPOTIC exceptions vary
            attempt_duration = time.perf_counter() - attempt_start
            fallback_cell = locals().get("cell")
            fallback_tg = float(getattr(fallback_cell, "Tg", float("nan"))) if fallback_cell is not None else float("nan")

            last_final_tg = fallback_tg
            line_results = _empty_line_results(species_order)
            if fallback_cell is not None:
                recovered_results: dict[str, LineLumResult] = {}
                for species in species_order:
                    try:
                        stdout_buffer = io.StringIO()
                        with contextlib.redirect_stdout(stdout_buffer):
                            transitions = fallback_cell.lineLum(species, escapeProbGeom='LVG')
                        _log_despotic_stdout(stdout_buffer.getvalue())
                        recovered_results[species] = _extract_line_result(transitions)
                    except Exception:
                        recovered_results[species] = _nan_line_result()
                line_results = recovered_results
            last_line_results = dict(line_results)
            last_residual_trace = tuple(residual_trace_run)
            last_temp_converged = temp_converged_run
            last_duration = attempt_duration
            if attempt_log is not None:
                attempt_log.append(
                    AttemptRecord(
                        row_idx=row_idx,
                        col_idx=col_idx,
                        nH=nH_val,
                        colDen=colDen_val,
                        tg_guess=guess,
                        final_Tg=fallback_tg,
                        attempt_number=attempt_number,
                        attempt_type="exception",
                        converged=False,
                        repeat_equilibrium=repeat_equilibrium,
                        line_results=line_results,
                        residual_trace=tuple(residual_trace_run),
                        temperature_converged=temp_converged_run,
                        error_message=str(exc),
                        duration=attempt_duration,
                    )
                )
            if log_failures:
                warnings.warn(
                    f"DESPOTIC failed for Tg guess {guess:.2f} K (nH={nH_val}, colDen={colDen_val}): {exc}",
                    RuntimeWarning,
                )
            _enqueue_retry(fallback_tg)
            continue

    if attempt_log is not None:
        attempt_log.append(
            AttemptRecord(
                row_idx=row_idx,
                col_idx=col_idx,
                nH=nH_val,
                colDen=colDen_val,
                tg_guess=last_guess if last_guess is not None else float("nan"),
                final_Tg=last_final_tg,
                attempt_number=attempt_number,
                attempt_type="all_guesses_failed",
                converged=False,
                repeat_equilibrium=repeat_equilibrium,
                line_results=last_line_results,
                residual_trace=last_residual_trace,
                temperature_converged=last_temp_converged,
                duration=last_duration,
            )
        )
    return MappingProxyType(dict(last_line_results)), float("nan")

def _compute_row(
    nH: float,
    col_den_points: np.ndarray,
    guess_list: Sequence[float],
    interpolator: Optional[RectBivariateSpline],
    *,
    chem_network=NL99,
    emitter_abundances: Mapping[str, float],
    row_logs: list[AttemptRecord] | None = None,
    repeat_equilibrium: int = 0,
    log_failures: bool = False,
    row_idx: int,
    reuse_failed_tg: bool = False,
    reuse_max_insertions: int = 5,
) -> Tuple[
    List[float],
    dict[str, dict[str, List[float]]],
    list[AttemptRecord],
]:
    species_order = tuple(emitter_abundances.keys())
    species_buffers: dict[str, dict[str, List[float]]] = {
        species: {field: [] for field in LINE_RESULT_FIELDS}
        for species in species_order
    }
    tg_row: List[float] = []

    if row_logs is None:
        row_logs = []

    for col_idx, colDen in enumerate(col_den_points):
        dynamic_guesses: Sequence[float] = guess_list
        if interpolator is not None:
            try:
                predicted_log_T = interpolator(np.log10(nH), np.log10(colDen))[0][0]
                predicted_T = float(10 ** predicted_log_T)
                if np.isfinite(predicted_T) and predicted_T > 0:
                    dynamic_guesses = [predicted_T] + list(guess_list)
            except Exception as exc:
                if log_failures:
                    warnings.warn(
                        f"Temperature interpolation failed at (nH={nH}, colDen={colDen}): {exc}",
                        RuntimeWarning,
                    )

        line_results_map, tg_val = calculate_single_despotic_point(
            nH_val=nH,
            colDen_val=colDen,
            initial_Tg_guesses=dynamic_guesses,
            log_failures=log_failures,
            emitter_abundances=emitter_abundances,
            repeat_equilibrium=repeat_equilibrium,
            chem_network=chem_network,
            row_idx=row_idx,
            col_idx=col_idx,
            attempt_log=row_logs,
            reuse_failed_tg=reuse_failed_tg,
            reuse_max_insertions=reuse_max_insertions,
        )

        tg_row.append(tg_val)
        line_results_dict = dict(line_results_map)
        for species in species_order:
            result = line_results_dict.get(species, _nan_line_result())
            for field in LINE_RESULT_FIELDS:
                species_buffers[species][field].append(getattr(result, field))

    return tg_row, species_buffers, row_logs

def build_table(
    nH_grid: LogGrid,
    col_den_grid: LogGrid,
    tg_guesses: Sequence[float],
    *,
    chem_network=NL99,
    emitter_abundances: Mapping[str, float] = DEFAULT_EMITTER_ABUNDANCES,
    interpolator: Optional[RectBivariateSpline] = None,
    n_jobs: int = -1,
    repeat_equilibrium: int = 0,
    show_progress: bool = False,
    log_failures: bool = False,
    reuse_failed_tg: bool = False,
    reuse_max_insertions: int = 3,
) -> DespoticTable:
    """Build a DESPOTIC lookup table for a pair of logarithmic grids."""
    nH_points = nH_grid.sample()
    col_den_points = col_den_grid.sample()
    species_order = tuple(emitter_abundances.keys())

    progress_bar = None
    progress_manager = contextlib.nullcontext()
    if show_progress:
        progress_bar = tqdm(total=nH_points.size, desc="DESPOTIC rows")
        progress_manager = tqdm_joblib(progress_bar)

    with progress_manager:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_row)(
                nH,
                col_den_points,
                tg_guesses,
                interpolator,
                chem_network=chem_network,
                emitter_abundances=emitter_abundances,
                repeat_equilibrium=repeat_equilibrium,
                log_failures=log_failures,
                row_idx=row_idx,
                reuse_failed_tg=reuse_failed_tg,
                reuse_max_insertions=reuse_max_insertions,
            )
            for row_idx, nH in enumerate(nH_points)
        )

    if progress_bar is not None:
        progress_bar.close()

    tg_rows, species_rows_list, attempt_lists = zip(*results)
    tg_final = np.asarray(tg_rows, dtype=float)

    species_data: dict[str, SpeciesLineGrid] = {}
    for species in species_order:
        field_arrays: dict[str, np.ndarray] = {}
        for field in LINE_RESULT_FIELDS:
            field_arrays[field] = np.asarray(
                [row_data[species][field] for row_data in species_rows_list],
                dtype=float,
            )
        species_data[species] = SpeciesLineGrid(
            int_tb=field_arrays["int_tb"],
            int_intensity=field_arrays["int_intensity"],
            lum_per_h=field_arrays["lum_per_h"],
            tau=field_arrays["tau"],
            tau_dust=field_arrays["tau_dust"],
            tex=field_arrays["tex"],
            freq=field_arrays["freq"],
        )

    from itertools import chain

    attempts = tuple(chain.from_iterable(attempt_lists))

    return DespoticTable(
        species_data=species_data,
        tg_final=tg_final,
        nH_values=nH_points,
        col_density_values=col_den_points,
        emitter_abundances=emitter_abundances,
        attempts=attempts,
    )



# def make_temperature_interpolator(
#     nH_values: Sequence[float],
#     col_density_values: Sequence[float],
#     tg_table: np.ndarray,
#     *,
#     kx: int = 3,
#     ky: int = 3,
# ) -> RectBivariateSpline:
#     """Create a spline interpolator in log-space for Tg data."""
#     nH_values = np.asarray(nH_values)
#     col_density_values = np.asarray(col_density_values)
#     tg_table = np.asarray(tg_table)

#     if tg_table.shape != (nH_values.size, col_density_values.size):
#         raise ValueError(
#             "tg_table shape must match the lengths of nH_values and col_density_values"
#         )

#     log_nH = np.log10(nH_values)
#     log_col = np.log10(col_density_values)
#     if np.any(~np.isfinite(log_nH)) or np.any(~np.isfinite(log_col)):
#         raise ValueError("nH_values and col_density_values must be positive and finite")

#     log_tg = np.log10(tg_table)
#     if np.any(~np.isfinite(log_tg)):
#         raise ValueError("tg_table must contain only positive, finite values")

#     return RectBivariateSpline(log_nH, log_col, log_tg, kx=kx, ky=ky)

# def refine_table(
#     coarse_table: DespoticTable,
#     fine_nH_grid: LogGrid,
#     fine_col_den_grid: LogGrid,
#     tg_guesses: Sequence[float],
#     *,
#     chem_network=NL99,
#     emitter_abundances: Mapping[str, float] | None = None,
#     interpolator: Optional[RectBivariateSpline] = None,
#     n_jobs: int = -1,
#     repeat_equilibrium: int = 0,
#     show_progress: bool = False,
#     log_failures: bool = False,
#     reuse_failed_tg: bool = False,
#     reuse_max_insertions: int = 3,
# ) -> DespoticTable:
#     """Use a coarse table to guide computation on a finer grid."""
#     if emitter_abundances is None:
#         emitter_abundances = coarse_table.emitter_abundances

#     if interpolator is None:
#         interpolator = make_temperature_interpolator(
#             coarse_table.nH_values,
#             coarse_table.col_density_values,
#             coarse_table.tg_final,
#         )

#     return build_table(
#         fine_nH_grid,
#         fine_col_den_grid,
#         tg_guesses,
#         chem_network=chem_network,
#         emitter_abundances=emitter_abundances,
#         interpolator=interpolator,
#         n_jobs=n_jobs,
#         repeat_equilibrium=repeat_equilibrium,
#         show_progress=show_progress,
#         log_failures=log_failures,
#         reuse_failed_tg=reuse_failed_tg,
#         reuse_max_insertions=reuse_max_insertions,
#     )


# def fill_missing_values(table: DespoticTable) -> DespoticTable:
#     log_nH = np.log10(table.nH_values)
#     log_col = np.log10(table.col_density_values)
#     log_col_grid, log_nH_grid = np.meshgrid(log_col, log_nH, indexing="xy")

#     def _fill_grid(values: np.ndarray, *, log_space: bool) -> np.ndarray:
#         grid = values.copy()
#         mask = ~np.isfinite(grid)
#         if log_space:
#             mask |= grid <= 0
#         if not mask.any():
#             return grid

#         if log_space:
#             safe = grid.copy()
#             safe[mask] = 1.0
#             work = np.log10(safe)
#         else:
#             work = grid

#         points = np.column_stack((log_nH_grid[~mask], log_col_grid[~mask]))
#         targets = np.column_stack((log_nH_grid[mask], log_col_grid[mask]))
#         filled = griddata(points, work[~mask], targets, method="linear")

#         if np.isnan(filled).any():
#             fallback = griddata(points, work[~mask], targets, method="nearest")
#             filled = np.where(np.isnan(filled), fallback, filled)

#         result = work.copy()
#         result[mask] = filled
#         return np.power(10.0, result) if log_space else result

#     species_filled: dict[str, SpeciesLineGrid] = {}
#     for species, grid in table.species_data.items():
#         species_filled[species] = SpeciesLineGrid(
#             int_tb=_fill_grid(grid.int_tb, log_space=False),
#             int_intensity=_fill_grid(grid.int_intensity, log_space=True),
#             lum_per_h=_fill_grid(grid.lum_per_h, log_space=True),
#             tau=_fill_grid(grid.tau, log_space=False),
#             tau_dust=_fill_grid(grid.tau_dust, log_space=False),
#             tex=_fill_grid(grid.tex, log_space=False),
#             freq=np.array(grid.freq, copy=True),
#         )

#     tg_filled = _fill_grid(table.tg_final, log_space=True)

#     return DespoticTable(
#         species_data=species_filled,
#         tg_final=tg_filled,
#         nH_values=table.nH_values,
#         col_density_values=table.col_density_values,
#         emitter_abundances=table.emitter_abundances,
#         attempts=table.attempts,
#     )

def compute_average(
    components: Sequence[np.ndarray],
    *,
    method: str = "arithmetic",
) -> np.ndarray:
    """Combine column density components with the requested averaging scheme.

    Parameters
    ----------
    components
        Sequence of arrays (all same shape) to combine.
    method
        One of ``"arithmetic"``/``"mean"``, ``"geometric"``/``"geom"``, or
        ``"harmonic"``/``"inverse"``.

    Returns
    -------
    np.ndarray
        Array with the same shape as the inputs after applying the requested
        averaging operation.
    """

    if not components:
        raise ValueError("components must contain at least one array")

    stack = np.stack(components, axis=0)
    method_key = method.lower()

    if method_key in {"arithmetic", "mean", "avg"}:
        return np.mean(stack, axis=0)

    if method_key in {"geometric", "geom"}:
        if np.any(stack <= 0):
            raise ValueError("Geometric mean requires all values to be positive")
        return np.exp(np.mean(np.log(stack), axis=0))

    if method_key in {"harmonic", "inverse"}:
        if np.any(stack <= 0):
            raise ValueError("Harmonic mean requires all values to be positive")
        return stack.shape[0] / np.sum(1.0 / stack, axis=0)

    raise ValueError(
        "Unsupported method '{method}'. Choose from 'arithmetic', 'geometric', or 'harmonic'."
        .format(method=method)
    )


def ensure_species_fields(
    field_map: dict[str, dict[str, np.ndarray]],
    species: str,
    shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    arrays = field_map.get(species)
    if arrays is None:
        arrays = {field: np.full(shape, np.nan, dtype=float) for field in LINE_RESULT_FIELDS}
        field_map[species] = arrays
    return arrays


def select_indices(
    values: np.ndarray,
    *,
    index_span: tuple[int, int] | None = None,
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    if index_span is not None:
        start, end = index_span
        start = max(start, 0)
        end = min(end, values.size)
        return np.arange(start, end)
    if value_range is not None:
        low, high = value_range
        return np.where((values >= low) & (values <= high))[0]
    return np.arange(values.size)




__all__ = [
    "LogGrid",
    "DespoticTable",
    "LineLumResult",
    "SpeciesLineGrid",
    "calculate_single_despotic_point",
    "build_table",
    "compute_average",
    "ensure_species_fields",
    "select_indices",
]
