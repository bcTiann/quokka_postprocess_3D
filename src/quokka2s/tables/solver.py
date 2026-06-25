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


def _flatten_energy_terms(rates: Mapping[str, object], prefix: str = "") -> dict[str, float]:
    """Flatten dEdt() output into a single-level {key: float} dict.

    DESPOTIC's ``cell.dEdt()`` returns a dict with one nested dict
    (``LambdaLine`` → {emitter_name: rate}); everything else is scalar. We
    flatten ``LambdaLine`` into ``LambdaLine.CO``, ``LambdaLine.C+`` etc. so the
    full per-species line cooling is preserved in the table for later analysis.
    """
    out: dict[str, float] = {}
    for k, v in rates.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, Mapping):
            out.update(_flatten_energy_terms(v, prefix=f"{key}."))
        else:
            try:
                out[key] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def calculate_single_despotic_point(
    nH_val: float,
    colDen_val: float,
    dvdr_val: float,
    *,
    species: Sequence[str] = DEFAULT_SPECIES,
    abundance_only: Sequence[str] = ("e-", ),
    chem_network=None,   # None → GOW, resolved lazily in-body (see below)
    log_failures: bool = True,
    row_idx: int | None = None,
    col_idx: int | None = None,
    dvdr_idx: int | None = None,
    Tg_init: float = 100.0,
    Tg_fixed: float | None = None,
    attempt_log: list[AttemptRecord] | None = None,
    escape_geom: str = 'LVG',
) -> Tuple[
    Mapping[str, LineLumResult],
    Mapping[str, float],
    Mapping[str, float],
    float, float, float,
    float,
    Mapping[str, float],
    bool,
]:
    """Run DESPOTIC at a single (nH, colDen, dVdr) point — true 3-input solve.

    Re-architected 2026-05-29 to fix two critical bugs in the old builder:

    1.  **Emitters were never added before setChemEq.** GOW's
        ``applyAbundances(addEmitters=True)`` only auto-adds emitters that are
        already in ``cell.emitters`` via DESPOTIC's case-insensitive lookup
        (it overwrites existing ones rather than creating new ones).  Without
        a pre-population call, ``cell.emitters`` stayed empty throughout
        ``setChemEq``'s ``iterateDust`` loop, so ``setTempEq``'s ``dEdt`` saw
        zero line cooling from CO/C+/C/HCO+/O — the dominant ISM coolants.
        This pushed CNM cells to ~10⁴ K (the dust+CR+PE equilibrium) instead
        of ~10²–10³ K (with C+ 158 μm cooling).  Verified empirically: at
        nH=10, NH=1e20, Tg was 9379 K without pre-add and 74 K with pre-add
        (127× error).

        Fix: before ``setChemEq``, call ``cell.addEmitter(sp, 0.0)`` for each
        species we want in the line-cooling sum (CO, C+, C, HCO+, O).  The
        zero is a placeholder; ``applyAbundances`` updates it to the
        equilibrium abundance during the chemistry solve.

    2.  **The dVdr axis was broadcast, not solved.** A "two-stage" build
        optimization set ``cell.dVdr = canonical_dvdr`` (median of the grid),
        solved chemistry + thermal balance once, and broadcast Tg/μ/cv/Eint/
        abundances across the 35 dVdr storage indices.  This was justified by
        an empirical claim that line cooling is a sub-dominant channel — but
        with bug #1 above, line cooling was literally zero, so the claim was
        trivially satisfied.  With #1 fixed, line cooling re-enters dEdt and
        dVdr genuinely affects thermal balance via LVG escape probabilities.

        Fix: ``calculate_single_despotic_point`` now takes a single
        ``dvdr_val`` (no more grid).  The builder calls it once per
        (nH, colDen, dVdr) cell, so every point in the 35³ table is the
        product of an independent ``setChemEq`` solve.

    Fixed-T mode (``Tg_fixed`` given): the gas temperature is pinned to
    ``Tg_fixed`` and chemistry is solved with ``evolveTemp="fixed"`` (no
    thermal balance).  ``mu/cv/Eint`` and abundances are then the values at
    that imposed T, and ``final_Tg == Tg_fixed``.  This is the build mode for
    the (nH, N_H, dVdr, T) table consumed by the μγ bisection.

    Returns
    -------
    line_results : Mapping[str, LineLumResult]
        For each emitting species, the line-luminosity result at this dVdr.
    species_abundances, chem_abundances : Mapping[str, float]
        Per-species composition (abundance per H nucleus) at the converged
        chemistry.
    mu, cv, eint : float
        Composition derivatives evaluated at the final Tg.
    final_Tg : float
        Self-consistent gas temperature from setChemEq (or Tg_fixed).
    energy_terms : Mapping[str, float]
        Flattened ``cell.dEdt()`` output — includes ``LambdaLine.CO``,
        ``LambdaLine.C+`` etc. so per-emitter cooling is preserved.
    failed : bool
        True if setChemEq did not converge or an exception occurred.
    """
    # despotic is an optional, table-building-only dependency — import it lazily
    # so the runtime pipeline never needs it installed.  ``chem_network=None``
    # resolves to GOW here (the historical default), so behaviour is unchanged.
    from despotic import cloud
    if chem_network is None:
        from despotic.chemistry import GOW
        chem_network = GOW

    species_order = tuple(species)
    dvdr_val = float(dvdr_val)

    # Failure-mode defaults.
    last_line_results: dict[str, LineLumResult] = _empty_line_results(species_order)
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
        cell.dVdr = dvdr_val

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

        # CRITICAL FIX #1: pre-add line-cooling emitters so setTempEq inside
        # iterateDust actually sees CO/C+/C/HCO+/O contributions.  Abundance
        # 0.0 is a placeholder — applyAbundances overwrites it with the
        # network's equilibrium value during the chemistry solve.  See the
        # docstring above for the full rationale.
        for sp in species_order:
            cell.addEmitter(sp, 0.0)

        # Initialise composition-derived quantities (mu, cv, ...) before
        # setChemEq. NL99's dxdt reads cloud.comp.mu (for sound speed); without
        # this call mu is 0 and every cell hits ZeroDivisionError. GOW's dxdt
        # doesn't depend on mu so it didn't surface there.
        cell.comp.computeDerived(cell.nH)

        # CRITICAL FIX #3 (2026-05-30): pass escapeProbGeom through to setTempEq.
        # Without this, setChemEq's inner setTempEq() defaults to sphere
        # geometry — and sphere escape probability does NOT read cell.dVdr
        # (only LVG does, emitter.py:684).  So Tg ends up dVdr-invariant even
        # though we set cell.dVdr per call.  Passing tempEqParam pipes the
        # caller's escape_geom all the way down to em.setLevPopEscapeProb
        # inside _gdTempResid → dEdt, so thermal balance uses the same
        # geometry as our post-solve lineLum call.
        _temp_eq_param = {'escapeProbGeom': escape_geom}
        with contextlib.redirect_stdout(stdout_buffer):
            if Tg_fixed is None:
                converged = cell.setChemEq(
                    network=chem_network,
                    evolveTemp="iterateDust",
                    tol=1e-6,
                    maxTime=1e22,
                    maxTempIter=200,
                    tempEqParam=_temp_eq_param,
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

        # Capture per-emitter line cooling in energy_terms (this is why
        # addEmitter was moved before setChemEq — so cell.emitters is
        # populated when dEdt iterates over them at line 979 of cloud.py).
        species_abundances: dict[str, float] = {}
        for sp in species_order:
            if sp in cell.emitters:
                species_abundances[sp] = float(cell.emitters[sp].abundance)
            else:
                species_abundances[sp] = float(cell.chemabundances.get(sp, float("nan")))
        last_abundances = species_abundances

        # dEdt returns LambdaLine as a nested dict {emitter_name: rate}; we
        # flatten it to LambdaLine.CO, LambdaLine.C+ etc. so all per-species
        # cooling channels live in the same flat energy_terms namespace.
        last_energy_terms = _flatten_energy_terms(dict(cell.dEdt()))

        # Line emission at this (single) dVdr.
        line_results: dict[str, LineLumResult] = {}
        if escape_geom in ('LVG', 'sphere'):
            with contextlib.redirect_stdout(stdout_buffer):
                for sp in species_order:
                    transitions = cell.lineLum(sp, escapeProbGeom=escape_geom)
                    line_results[sp] = _extract_line_result(transitions)
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
                    dvdr_idx=dvdr_idx,
                    dvdr=dvdr_val,
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
                    dvdr_idx=dvdr_idx,
                    dvdr=dvdr_val,
                )
            )
        if log_failures:
            LOGGER.warning("Exception at nH=%s colDen=%s dVdr=%s: %s",
                           nH_val, colDen_val, dvdr_val, exc)

    if failed and log_failures:
        LOGGER.warning("Failed at nH=%s colDen=%s dVdr=%s", nH_val, colDen_val, dvdr_val)

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
