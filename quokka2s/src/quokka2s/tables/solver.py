from __future__ import annotations
import os
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

# Point DESPOTIC at its bundled LAMDA molecular-data directory if the
# user has not set DESPOTIC_HOME themselves. Without this, addEmitter
# silently falls back to fetching LAMDA files from the web on first use
# — fragile when the network is flaky.
if "DESPOTIC_HOME" not in os.environ:
    _despotic_home = "/Users/baochen/despotic/despotic/chemistry"
    if os.path.isdir(os.path.join(_despotic_home, "LAMDA")):
        os.environ["DESPOTIC_HOME"] = _despotic_home

import contextlib
import io
import logging
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


def _empty_line_results_per_dvdr(species: Sequence[str], num_dvdr: int) -> dict[str, list[LineLumResult]]:
    """Return a dict of species → list of NaN LineLumResults (one per dVdr point)."""
    return {sp: [_nan_line_result()] * num_dvdr for sp in species}

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
    dvdr_grid: Sequence[float],
    *,
    species: Sequence[str] = DEFAULT_SPECIES,
    abundance_only: Sequence[str] = ("e-", ),
    chem_network=GOW,
    log_failures: bool = True,
    row_idx: int | None = None,
    col_idx: int | None = None,
    Tg_init: float = 100.0,
    Tg_fixed: float | None = None,
    attempt_log: list[AttemptRecord] | None = None,
    escape_geom: str = 'LVG',
) -> Tuple[
    Mapping[str, list[LineLumResult]],
    Mapping[str, float],
    Mapping[str, float],
    float, float, float,
    float,
    Mapping[str, float],
    bool,
]:
    """Run DESPOTIC at one (nH, colDen) point and sweep dVdr for line emission.

    Two-stage build optimization (verified empirically: at fixed nH, colDen,
    Tg and chemical abundances are independent of dVdr because dVdr enters
    DESPOTIC only via LVG escape probabilities, which feed back into thermal
    balance only through line cooling — empirically a sub-dominant cooling
    channel across the whole physical ISM range):

    1.  Solve chemistry + thermal balance once via
        ``cell.setChemEq(evolveTemp="iterateDust")`` at a canonical dVdr.
    2.  For each dVdr in ``dvdr_grid``: set ``cell.dVdr`` and call

    Fixed-T mode (``Tg_fixed`` given): the gas temperature is pinned to
    ``Tg_fixed`` and chemistry is solved with ``evolveTemp="fixed"`` (no
    thermal balance).  ``mu/cv/Eint`` and abundances are then the values at
    that imposed T, and ``final_Tg == Tg_fixed``.  This is the build mode for
    the (nH, N_H, dVdr, T) table consumed by the μγ bisection.
        ``cell.lineLum(species, escapeProbGeom='LVG')``. Internally this
        re-solves level populations and escape probabilities (cheap)
        without rerunning chemistry.

    Returns
    -------
    line_results : Mapping[str, list[LineLumResult]]
        For each emitting species, a list of LineLumResult (one per
        entry in ``dvdr_grid``, in order).
    species_abundances, chem_abundances : Mapping[str, float]
        Single-value-per-species mappings (independent of dVdr).
    mu, cv, eint : float
        Single-value composition derivatives (independent of dVdr).
    final_Tg : float
        Self-consistent gas temperature from setChemEq.
    energy_terms : Mapping[str, float]
        Single-cell dEdt() dictionary at convergence.
    failed : bool
        True if setChemEq did not converge or an exception occurred.
    """
    species_order = tuple(species)
    num_dvdr = len(dvdr_grid)
    canonical_dvdr = float(dvdr_grid[num_dvdr // 2]) if num_dvdr > 0 else 1e-14

    # Failure-mode defaults.
    last_line_results: dict[str, list[LineLumResult]] = _empty_line_results_per_dvdr(species_order, num_dvdr)
    last_abundances: dict[str, float] = {sp: float("nan") for sp in species_order}
    last_chem_abundances: dict[str, float] = {}
    last_energy_terms: dict[str, float] = {}
    last_final_tg = float("nan")
    last_mu_val = float("nan")
    last_cv_val = float("nan")
    last_eint_val = float("nan")
    failed = True

    attempt_start_time = time.perf_counter()
    stdout_buffer = io.StringIO()

    try:
        cell = cloud()
        cell.nH = nH_val
        cell.colDen = colDen_val
        cell.Tg = Tg_init if Tg_fixed is None else float(Tg_fixed)
        cell.dVdr = canonical_dvdr

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

        # Initialise composition-derived quantities (mu, cv, ...) before
        # setChemEq. NL99's dxdt reads cloud.comp.mu (for sound speed); without
        # this call mu is 0 and every cell hits ZeroDivisionError. GOW's dxdt
        # doesn't depend on mu so it didn't surface there.
        cell.comp.computeDerived(cell.nH)

        with contextlib.redirect_stdout(stdout_buffer):
            if Tg_fixed is None:
                converged = cell.setChemEq(
                    network=chem_network,
                    evolveTemp="iterateDust",
                    tol=1e-6,
                    maxTime=1e22,
                    maxTempIter=200,
                )
            else:
                # Pin T; solve chemistry only (no thermal balance).
                cell.Tg = float(Tg_fixed)
                converged = cell.setChemEq(
                    network=chem_network,
                    evolveTemp="fixed",
                    tol=1e-6,
                    maxTime=1e22,
                )
        _log_despotic_stdout(stdout_buffer)

        cell.comp.computeDerived(cell.nH)
        last_mu_val = float(cell.comp.mu)
        last_cv_val = float(cell.comp.computeCv(cell.Tg))
        last_eint_val = float(cell.comp.computeEint(cell.Tg))
        last_chem_abundances = dict(cell.chemabundances)
        last_final_tg = float(cell.Tg)
        last_energy_terms = dict(cell.dEdt())

        # Add all emitters once at the converged abundances.
        species_abundances: dict[str, float] = {}
        for sp in species_order:
            cell.addEmitter(sp, cell.chemabundances[sp])
            species_abundances[sp] = float(cell.emitters[sp].abundance)
        last_abundances = species_abundances

        # Compute line emission. LVG: sweep dVdr; sphere: compute once then
        # broadcast across the dVdr axis (sphere τ formula doesn't read dVdr).
        line_results: dict[str, list[LineLumResult]] = {sp: [] for sp in species_order}
        if escape_geom == 'LVG':
            for dvdr in dvdr_grid:
                cell.dVdr = float(dvdr)
                with contextlib.redirect_stdout(stdout_buffer):
                    for sp in species_order:
                        transitions = cell.lineLum(sp, escapeProbGeom='LVG')
                        line_results[sp].append(_extract_line_result(transitions))
                _log_despotic_stdout(stdout_buffer)
        elif escape_geom == 'sphere':
            with contextlib.redirect_stdout(stdout_buffer):
                for sp in species_order:
                    transitions = cell.lineLum(sp, escapeProbGeom='sphere')
                    single = _extract_line_result(transitions)
                    line_results[sp] = [single] * num_dvdr
            _log_despotic_stdout(stdout_buffer)
        else:
            raise ValueError(f"unknown escape_geom: {escape_geom!r}; expected 'LVG' or 'sphere'")
        last_line_results = line_results

        failed = not converged

        if attempt_log is not None:
            attempt_log.append(
                AttemptRecord(
                    row_idx=row_idx if row_idx is not None else -1,
                    col_idx=col_idx if col_idx is not None else -1,
                    nH=nH_val,
                    colDen=colDen_val,
                    tg_guess=Tg_init,
                    final_Tg=last_final_tg,
                    converged=converged,
                    message="Success" if converged else "Did not converge",
                    duration=time.perf_counter() - attempt_start_time,
                )
            )

    except Exception as exc:
        if attempt_log is not None:
            attempt_log.append(
                AttemptRecord(
                    row_idx=row_idx if row_idx is not None else -1,
                    col_idx=col_idx if col_idx is not None else -1,
                    nH=nH_val,
                    colDen=colDen_val,
                    tg_guess=Tg_init,
                    final_Tg=last_final_tg,
                    converged=False,
                    message=str(exc),
                    duration=time.perf_counter() - attempt_start_time,
                )
            )
        if log_failures:
            LOGGER.warning("Exception at nH=%s colDen=%s: %s", nH_val, colDen_val, exc)

    if failed and log_failures:
        LOGGER.warning("Failed at nH=%s colDen=%s", nH_val, colDen_val)

    return (
        MappingProxyType(last_line_results),
        MappingProxyType(last_abundances),
        MappingProxyType(last_chem_abundances),
        last_mu_val,
        last_cv_val,
        last_eint_val,
        last_final_tg,
        MappingProxyType(last_energy_terms),
        failed,
    )
