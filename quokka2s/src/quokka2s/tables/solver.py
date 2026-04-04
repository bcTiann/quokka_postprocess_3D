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
import contextlib
import io
import logging
import math
import time
from typing import Mapping, Sequence, Tuple

import numpy as np
from despotic import cloud
from despotic.chemistry import NL99, NL99_GC, GOW
from types import MappingProxyType

from .models import AttemptRecord, LineLumResult

DEFAULT_SPECIES = ("CO", "C+", "HCO+")
LOGGER = logging.getLogger(__name__)

LINE_RESULT_FIELDS = [
    "freq",
    "intIntensity",
    "intTB",
    "lumPerH",
    "tau",
    "tauDust",
]

_NAN_LINE_RESULT = LineLumResult(
    *(float("nan") for _ in LINE_RESULT_FIELDS)
)

def _nan_line_result() -> LineLumResult:
    """Return a LineLumResult with all fields set to NaN."""
    return _NAN_LINE_RESULT 


def _empty_line_results(species: Sequence[str]) -> dict[str, LineLumResult]:
    """Return a dict of species to NaN LineLumResults."""
    return {sp: _nan_line_result() for sp in species}

def _extract_line_result(transitions: Sequence[Mapping[str, float]]) -> LineLumResult:
    if not transitions:
        return _nan_line_result()
    entry = transitions[0]
    return LineLumResult(
        freq=entry.get("freq", float("nan")),
        intIntensity=entry.get("intIntensity", float("nan")),
        intTB=entry.get("intTB", float("nan")),
        lumPerH=entry.get("lumPerH", float("nan")),
        tau=entry.get("tau", float("nan")),
        tauDust=entry.get("tauDust", float("nan")),
    )

def _log_despotic_stdout(output: io.StringIO | str) -> None:
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
    species: Sequence[str] = DEFAULT_SPECIES,
    abundance_only: Sequence[str] = ("e-", ),
    chem_network=NL99_GC,
    log_failures: bool = True,
    row_idx: int | None = None,
    col_idx: int | None = None,
    attempt_log: list[AttemptRecord] | None = None,
) -> Tuple[Mapping[str, LineLumResult], Mapping[str, float], Mapping[str, float], float, Mapping[str, float], bool]:
    """
    Calculate DESPOTIC line for a single point in (nH, colDen) .

    Parameters
    ----------
    nH_val : float
        Hydrogen number density in cm^-3.
    colDen_val : float
        Hydrogen column density in cm^-2.
    initial_Tg_guesses : Sequence[float]
        Initial guesses for gas temperature in K.
    chem_network : despotic.chemistry.ChemicalNetwork, optional
        Chemical network to use. Default is NL99_GC.
    log_failures : bool, optional
        Whether to log failures. Default is True.
    row_idx : int | None, optional
        Row index in the table grid, for logging purposes.
    col_idx : int | None, optional
        Column index in the table grid, for logging purposes.
    attempt_log : List[AttemptRecord] | None, optional
        List to append attempt records to.

    Returns
    -------
    line_results : Mapping[str, LineLumResult]
        Mapping of species to their line luminosity results.
    final_Tg : float
        Final gas temperature in K.
    failed_flag : bool
        Whether the calculation converged.
    """
    species_order = tuple(species)
    pending_guesses = [float(g) for g in initial_Tg_guesses]
    seen_guesses: list[float] = []

    last_line_results: dict[str, LineLumResult] = _empty_line_results(species_order)
    last_abundances: dict[str, float] = {sp: float("nan") for sp in species_order}
    last_chem_abundances: dict[str, float] = {}
    last_energy_terms: dict[str, float] = {}
    last_final_tg = float("nan")
    failed = True
    
    while pending_guesses:
        guess = pending_guesses.pop(0)
        if any(math.isclose(guess, prev, rel_tol=1e-2, abs_tol=1e-2) for prev in seen_guesses):
            continue
        seen_guesses.append(guess)
        attempt_start_time = time.perf_counter()

        energy_terms: dict[str, float] = {}
        final_tg = float("nan")
        stdout_buffer = io.StringIO()
        
        try:
            cell = cloud()
            cell.nH = nH_val
            cell.colDen = colDen_val
            cell.Tg = guess


            cell.sigmaNT = 2.0e5
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

            with contextlib.redirect_stdout(stdout_buffer):
                converged = cell.setChemEq(
                    network=chem_network,
                    evolveTemp="iterateDust",
                    tol=1e-6,
                    maxTime=1e22,
                    maxTempIter=50,
                )
            chem_abundances = dict(cell.chemabundances)
            last_chem_abundances = dict(chem_abundances)
            _log_despotic_stdout(stdout_buffer)

            line_results: dict[str, LineLumResult] = {}
            species_abundances: dict[str, float] = {}
            for species in species_order:
                cell.addEmitter(species, cell.chemabundances[species])
                transitions = cell.lineLum(species)
                line_results[species] = _extract_line_result(transitions)
                species_abundances[species] = float(cell.emitters[species].abundance)

            energy_terms = dict(cell.dEdt())
            final_tg = float(cell.Tg)
            failed = not converged

            last_line_results = dict(line_results)
            last_abundances = dict(species_abundances)
            last_energy_terms = dict(energy_terms)
            last_final_tg = final_tg

            if attempt_log is not None:
                attempt_log.append(
                    AttemptRecord(
                        row_idx=row_idx if row_idx is not None else -1,
                        col_idx=col_idx if col_idx is not None else -1,
                        nH=nH_val,
                        colDen=colDen_val,
                        tg_guess=guess,
                        final_Tg=final_tg,
                        converged=converged,
                        message= "Success",
                        duration=time.perf_counter() - attempt_start_time,
                    )
                )    
            return (
                MappingProxyType(last_line_results),
                MappingProxyType(last_abundances),
                MappingProxyType(last_chem_abundances),
                last_final_tg,
                MappingProxyType(last_energy_terms),
                failed,
            )

        except Exception as exc:
            if attempt_log is not None:
                attempt_log.append(
                    AttemptRecord(
                        row_idx=row_idx if row_idx is not None else -1,
                        col_idx=col_idx if col_idx is not None else -1,
                        nH=nH_val,
                        colDen=colDen_val,
                        tg_guess=guess,
                        final_Tg=final_tg,
                        converged=False,
                        message=str(exc),
                        duration=time.perf_counter() - attempt_start_time,
                    )
            )


    if log_failures:
        LOGGER.warning("All guesses failed for nH=%s colDen=%s", nH_val, colDen_val)
    return (
    MappingProxyType(last_line_results),
    MappingProxyType(last_abundances),
    MappingProxyType(last_chem_abundances),
    last_final_tg,
    MappingProxyType(last_energy_terms),
    failed,
    )


