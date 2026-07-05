"""Single-cell solver for the canonical GOW/LVG DESPOTIC table."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import logging
import os
import time
import warnings
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from .models import AttemptRecord, LineLumResult


LOGGER = logging.getLogger(__name__)
LINE_RESULT_FIELDS = ("freq", "intIntensity", "intTB", "lumPerH", "tau", "tauDust")
DEFAULT_EMITTERS = ("CO", "C", "C+", "HCO+", "O")
LVG_GEOMETRY = "LVG"

warnings.filterwarnings(
    "ignore",
    message="collision rates not available",
    category=UserWarning,
    module=r"despotic\.emitterData",
)

_NAN_LINE_RESULT = LineLumResult(*([float("nan")] * len(LINE_RESULT_FIELDS)))


def _configure_despotic_home() -> None:
    """Point DESPOTIC at a directory containing the required LAMDA files."""
    if "DESPOTIC_HOME" in os.environ:
        home = Path(os.environ["DESPOTIC_HOME"]).expanduser()
        if not (home / "LAMDA").is_dir():
            raise RuntimeError(f"DESPOTIC_HOME does not contain LAMDA/: {home}")
        return

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [repo_root]
    spec = importlib.util.find_spec("despotic")
    if spec and spec.submodule_search_locations:
        candidates.insert(0, Path(next(iter(spec.submodule_search_locations))) / "chemistry")

    required = ("co.dat", "catom.dat", "c+.dat", "hco+.dat", "oatom.dat")
    for candidate in candidates:
        lamda = candidate / "LAMDA"
        if lamda.is_dir() and all((lamda / name).is_file() for name in required):
            os.environ["DESPOTIC_HOME"] = str(candidate)
            return
    raise RuntimeError(
        "Could not find the required LAMDA files. Keep repo-root LAMDA/ intact "
        "or set DESPOTIC_HOME to a directory containing LAMDA/."
    )


def _nan_line_results(species: Sequence[str]) -> dict[str, LineLumResult]:
    return {name: _NAN_LINE_RESULT for name in species}


def _extract_line_result(transitions: Sequence[Mapping[str, float]]) -> LineLumResult:
    if not transitions:
        return _NAN_LINE_RESULT
    entry = transitions[0]
    return LineLumResult(*(float(entry.get(field, float("nan"))) for field in LINE_RESULT_FIELDS))


def _log_despotic_stdout(output: io.StringIO) -> None:
    text = output.getvalue()
    output.truncate(0)
    output.seek(0)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("make: ***"):
            continue
        if stripped.startswith("setChemEquil:") or "Temperature converged!" in stripped:
            LOGGER.debug("DESPOTIC: %s", stripped)
        else:
            LOGGER.warning("DESPOTIC: %s", stripped)


def _flatten_energy_terms(rates: Mapping[str, object], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in rates.items():
        key = f"{prefix}{name}" if prefix else name
        if isinstance(value, Mapping):
            out.update(_flatten_energy_terms(value, prefix=f"{key}."))
        else:
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                pass
    return out


def solve_gow_lvg_point(
    nH_val: float,
    colDen_val: float,
    dvdr_val: float,
    *,
    species: Sequence[str] = DEFAULT_EMITTERS,
    abundance_only: Sequence[str] = ("e-", "H+", "H2", "H"),
    log_failures: bool = True,
    row_idx: int | None = None,
    col_idx: int | None = None,
    dvdr_idx: int | None = None,
    Tg_init: float = 100.0,
    attempt_log: list[AttemptRecord] | None = None,
) -> tuple[
    Mapping[str, LineLumResult],
    Mapping[str, float],
    float,
    float,
    float,
    float,
    Mapping[str, float],
    bool,
]:
    """Solve one ``(nH, N_H, dVdr)`` point with GOW and LVG thermal balance."""
    _configure_despotic_home()
    from despotic import cloud
    from despotic.chemistry import GOW

    species_order = tuple(species)
    last_lines = _nan_line_results(species_order)
    last_abundances: dict[str, float] = {}
    last_energy: dict[str, float] = {}
    last_mu = last_cv = last_eint = last_tg = float("nan")
    failed = True
    started = time.perf_counter()
    output = io.StringIO()

    try:
        cell = cloud()
        cell.nH = float(nH_val)
        cell.colDen = float(colDen_val)
        cell.Tg = float(Tg_init)
        cell.dVdr = float(dvdr_val)

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

        # Emitters must exist before setChemEq so their LVG line cooling enters
        # the iterateDust thermal-balance solve. GOW replaces the zero
        # placeholders with equilibrium abundances.
        for name in species_order:
            cell.addEmitter(name, 0.0)
        cell.comp.computeDerived(cell.nH)

        with contextlib.redirect_stdout(output):
            converged = cell.setChemEq(
                network=GOW,
                evolveTemp="iterateDust",
                tol=1e-6,
                maxTime=1e22,
                maxTempIter=200,
                tempEqParam={"escapeProbGeom": LVG_GEOMETRY},
            )
        _log_despotic_stdout(output)

        cell.comp.computeDerived(cell.nH)
        last_mu = float(cell.comp.mu)
        last_cv = float(cell.comp.computeCv(cell.Tg))
        last_eint = float(cell.comp.computeEint(cell.Tg))
        last_tg = float(cell.Tg)
        last_abundances = dict(cell.chemabundances)
        last_energy = _flatten_energy_terms(dict(cell.dEdt()))

        lines: dict[str, LineLumResult] = {}
        with contextlib.redirect_stdout(output):
            for name in species_order:
                lines[name] = _extract_line_result(cell.lineLum(name, escapeProbGeom=LVG_GEOMETRY))
        _log_despotic_stdout(output)
        last_lines = lines
        failed = not bool(converged)

        if attempt_log is not None:
            attempt_log.append(AttemptRecord(
                row_idx=-1 if row_idx is None else row_idx,
                col_idx=-1 if col_idx is None else col_idx,
                nH=float(nH_val), colDen=float(colDen_val), tg_guess=float(Tg_init),
                final_Tg=last_tg, converged=bool(converged),
                message="Success" if converged else "Did not converge",
                duration=time.perf_counter() - started,
                dvdr_idx=dvdr_idx, dvdr=float(dvdr_val),
            ))
    except Exception as exc:
        if attempt_log is not None:
            attempt_log.append(AttemptRecord(
                row_idx=-1 if row_idx is None else row_idx,
                col_idx=-1 if col_idx is None else col_idx,
                nH=float(nH_val), colDen=float(colDen_val), tg_guess=float(Tg_init),
                final_Tg=last_tg, converged=False, message=str(exc),
                duration=time.perf_counter() - started,
                dvdr_idx=dvdr_idx, dvdr=float(dvdr_val),
            ))
        if log_failures:
            LOGGER.warning("Exception at nH=%s N_H=%s dVdr=%s: %s", nH_val, colDen_val, dvdr_val, exc)

    if failed and log_failures:
        LOGGER.warning("Failed at nH=%s N_H=%s dVdr=%s", nH_val, colDen_val, dvdr_val)

    # Ensure requested abundance-only names exist even if DESPOTIC omitted one.
    for name in abundance_only:
        last_abundances.setdefault(name, float("nan"))
    return (
        MappingProxyType(last_lines),
        MappingProxyType(last_abundances),
        last_mu, last_cv, last_eint, last_tg,
        MappingProxyType(last_energy),
        failed,
    )
