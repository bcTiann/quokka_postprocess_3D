"""Single-cell solver for the canonical GOW/LVG DESPOTIC table."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
from importlib.metadata import distribution
import io
import logging
import os
import time
import warnings
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .abundances import GOW_ELEMENTAL_ABUNDANCES, abundance_metadata
from .models import AttemptRecord, LineLumResult, ThermalSolveError


LOGGER = logging.getLogger(__name__)
LINE_RESULT_FIELDS = ("freq", "intIntensity", "intTB", "lumPerH", "tau", "tauDust")
DEFAULT_EMITTERS = ("CO", "C", "C+", "HCO+", "O")
LVG_GEOMETRY = "LVG"
CO21_TABLE_TOKEN = "CO21"
DESPOTIC_REQUIRED_COMMIT = "ed18e5669adb7306f795a3d30d8919995793bc61"
GOW_REFRESH_SOURCE_SHA256 = "0abcc94bd0a7e72d0fc1c8f826fa1aad45b5ad0f7bff41ba178a8ed20f1b1ed3"
CHECKED_CHEMISTRY_SOURCE_SHA256 = "c1376df46886bd8f7c2638df7323125a110631c17b3b9ccf8dc74d9aea9d7b09"
FALLBACK_GAS_BRACKET_UPPER_K = 1000.0
FINAL_THERMAL_RESIDUAL_TOLERANCE = 1e-4
ELEMENT_TOTAL_RELATIVE_TOLERANCE = 1e-8

warnings.filterwarnings(
    "ignore",
    message="collision rates not available",
    category=UserWarning,
    module=r"despotic\.emitterData",
)

_NAN_LINE_RESULT = LineLumResult(*([float("nan")] * len(LINE_RESULT_FIELDS)))
_PointResult = tuple[
    Mapping[str, LineLumResult], Mapping[str, float], float, float, float, float,
    Mapping[str, float], bool,
]


def validated_solver_metadata() -> dict[str, object]:
    """Check the reviewed DESPOTIC patches and record build provenance."""
    package = distribution("despotic")
    source = Path(package.locate_file("despotic/chemistry/GOW.py"))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != GOW_REFRESH_SOURCE_SHA256:
        raise RuntimeError(
            "DESPOTIC table builds require the reviewed GOW composition refresh. "
            "Run `python scripts/apply_despotic_gow_patch.py` with the table-building "
            f"Python environment before rebuilding. Found GOW.py SHA256 {digest}."
        )
    chemistry_source = Path(package.locate_file("despotic/chemistry/chemEvol.py"))
    chemistry_digest = hashlib.sha256(chemistry_source.read_bytes()).hexdigest()
    if chemistry_digest != CHECKED_CHEMISTRY_SOURCE_SHA256:
        raise RuntimeError(
            "DESPOTIC table builds require checked GOW chemical integrations. "
            "Run python scripts/apply_despotic_chemistry_patch.py with the table-building "
            f"Python environment before rebuilding. Found chemEvol.py SHA256 {chemistry_digest}."
        )
    return {
        "composition": abundance_metadata(),
        "despotic": {
            "version": package.version,
            "required_upstream_commit": DESPOTIC_REQUIRED_COMMIT,
            "gow_source_sha256": digest,
            "gow_patch": "computeDerived after applyAbundances hydrogen consistency check",
            "chemistry_source_sha256": chemistry_digest,
            "chemistry_patch": "check fixed-temperature GOW integration; retry failed segment from same state",
            "scipy_version": distribution("scipy").version,
        },
        "numerical_settings": {
            "chemistry_tolerance": 1e-6,
            "chemistry_acceptance": "native setChemEq relative species-change and temperature criteria",
            "chemistry_max_time_s": 1e22,
            "max_temperature_iterations": 200,
            "initial_gas_temperature_K": 100.0,
            "odeint": {
                "rtol": 1e-8, "atol": 1e-12, "mxstep": 10000,
                "acceptance": "Integration successful; complete finite output",
                "retry": "failed segment only; same initial state; local time with physical RHS time restored",
            },
            "thermal_retry": {
                "trigger": "thermal subsolve failure only",
                "state": "restart whole point on a fresh cloud",
                "initial_gas_bracket_upper_K": FALLBACK_GAS_BRACKET_UPPER_K,
                "expansion": "decades until a valid sign change; at most 12 expansions",
            },
        },
        "final_validation": {
            "thermal_relative_residual_tolerance": FINAL_THERMAL_RESIDUAL_TOLERANCE,
            "element_total_relative_tolerance": ELEMENT_TOTAL_RELATIVE_TOLERANCE,
            "derived_relative_tolerance": 1e-12,
            "level_populations": "native dEdt LVG retries; finite normalized populations required",
            "line_output": "noRecompute; same populations as final thermal residual",
            "failed_points": "NaN physical outputs; original failure retained in attempt log",
        },
    }


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
    names = list(species)
    if "CO" in names:
        names.append(CO21_TABLE_TOKEN)
    return {name: _NAN_LINE_RESULT for name in names}


def _extract_line_result(
    transitions: Sequence[Mapping[str, float]],
    index: int = 0,
) -> LineLumResult:
    if index < 0 or index >= len(transitions):
        return _NAN_LINE_RESULT
    entry = transitions[index]
    return LineLumResult(*(float(entry.get(field, float("nan"))) for field in LINE_RESULT_FIELDS))


def _extract_transition_result(
    transitions: Sequence[Mapping[str, float]],
    upper: int,
    lower: int,
) -> LineLumResult:
    """Extract a transition by level numbers, independent of LAMDA order."""
    for entry in transitions:
        matches_upper = int(entry.get("upper", -1)) == upper
        matches_lower = int(entry.get("lower", -1)) == lower
        if matches_upper and matches_lower:
            return LineLumResult(
                *(float(entry.get(field, float("nan"))) for field in LINE_RESULT_FIELDS)
            )
    return _NAN_LINE_RESULT


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


def _relative_energy_residual(rates: Mapping[str, float], component: str) -> float:
    net = float(rates[f"dEdt{component}"])
    scale = float(rates[f"maxAbsdEdt{component}"])
    if not np.isfinite(net) or not np.isfinite(scale) or scale < 0.0:
        raise RuntimeError(f"Non-finite or invalid final {component.lower()} energy rates")
    residual = abs(net) / scale if scale > 0.0 else (0.0 if net == 0.0 else float("inf"))
    if residual >= FINAL_THERMAL_RESIDUAL_TOLERANCE:
        raise RuntimeError(
            f"Final {component.lower()} thermal residual {residual:.6g} exceeds "
            f"the {FINAL_THERMAL_RESIDUAL_TOLERANCE:g} acceptance threshold"
        )
    return residual


def _validate_final_state(cell, species: Sequence[str]) -> dict[str, float]:
    """Validate fixed chemistry/temperatures and prepare one checked line state."""
    abundances = dict(cell.chemabundances)
    physical_state = (float(cell.Tg), float(cell.Td), abundances)
    for name, value in (("Tg", cell.Tg), ("Td", cell.Td), ("mu", cell.comp.mu)):
        if not np.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"Final {name} must be finite and positive")
    if not all(np.isfinite(value) for value in abundances.values()):
        raise RuntimeError("Final chemical abundances are non-finite")
    if any(value < 0.0 for value in abundances.values()):
        raise RuntimeError("Final chemical abundances are negative")

    hydrogen_total = sum(
        count * abundances[name] for name, count in (
            ("H", 1), ("H+", 1), ("H2", 2), ("H2+", 2), ("H3+", 3),
            ("CHx", 1), ("OHx", 1), ("HCO+", 1),
        )
    )
    if not np.isclose(hydrogen_total, 1.0, rtol=ELEMENT_TOTAL_RELATIVE_TOLERANCE, atol=0.0):
        raise RuntimeError("Final network hydrogen total is not conserved")

    element_species = {
        "xHe": ("He", "He+"),
        "xC": ("C", "C+", "CO", "HCO+", "CHx"),
        "xO": ("O", "O+", "CO", "HCO+", "OHx"),
        "xSi": ("Si", "Si+"),
    }
    for element, names in element_species.items():
        total = sum(abundances[name] for name in names)
        if not np.isclose(total, GOW_ELEMENTAL_ABUNDANCES[element],
                          rtol=ELEMENT_TOTAL_RELATIVE_TOLERANCE, atol=0.0):
            raise RuntimeError(f"Final {element} elemental total does not match the adopted composition")
    if not np.isclose(cell.comp.xHe, GOW_ELEMENTAL_ABUNDANCES["xHe"],
                      rtol=ELEMENT_TOTAL_RELATIVE_TOLERANCE, atol=0.0):
        raise RuntimeError("Final bulk helium abundance does not match the adopted composition")

    # Recompute on a separate composition object to detect stale cached values
    # without silently repairing the state whose temperature was just solved.
    derived = copy.copy(cell.comp)
    derived.computeDerived(cell.nH)
    for name in ("mu", "muH", "qIon"):
        actual, expected = float(getattr(cell.comp, name)), float(getattr(derived, name))
        if not np.isfinite(actual) or not np.isfinite(expected) or not np.isclose(
            actual, expected, rtol=1e-12, atol=0.0
        ):
            raise RuntimeError(f"Final composition-derived {name} is inconsistent with the abundances")

    # dEdt checks convergence and performs its native damping retries. A direct
    # lineLum recomputation would ignore a False level-solver return value.
    energy = _flatten_energy_terms(dict(cell.dEdt(escapeProbGeom=LVG_GEOMETRY)))
    if not all(np.isfinite(value) for value in energy.values()):
        raise RuntimeError("Final heating or cooling terms are non-finite")
    for name in species:
        emitter = cell.emitters[name]
        if emitter.energySkip:
            raise RuntimeError(f"Final {name} populations were excluded from the thermal validation")
        populations = np.asarray(emitter.levPop, dtype=float)
        if (not emitter.levPopInitialized or not emitter.escapeProbInitialized
                or not np.isfinite(populations).all() or (populations < 0.0).any()
                or not np.isclose(populations.sum(), 1.0, rtol=0.0, atol=1e-8)
                or not np.isfinite(emitter.escapeProb).all()
                or not np.isfinite(emitter.tau).all()):
            raise RuntimeError(f"Final {name} level populations or escape probabilities are invalid")
    rates = cell.dEdt(escapeProbGeom=LVG_GEOMETRY, sumOnly=True, fixedLevPop=True)
    for component in ("Gas", "Dust"):
        energy[f"validation.{component.lower()}_relative_residual"] = _relative_energy_residual(
            rates, component
        )
    if (float(cell.Tg), float(cell.Td), dict(cell.chemabundances)) != physical_state:
        raise RuntimeError("Final validation changed the temperatures or chemical abundances")
    return energy


def _make_despotic_cloud(initial_gas_upper=None):
    # Keep the optional DESPOTIC dependency out of prebuilt-table readers.
    from .thermal_solver import CheckedCloud
    return CheckedCloud(initial_gas_upper=initial_gas_upper)


def _solve_gow_lvg_point_once(
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
    sigma_nt_cms: float = 2.0e5,
    attempt_log: list[AttemptRecord] | None = None,
    initial_gas_upper: float | None = None,
) -> tuple[_PointResult, bool]:
    """Solve one ``(nH, N_H, dVdr)`` point with GOW and LVG thermal balance.

    ``sigma_nt_cms`` defaults to the canonical table value of 2 km/s.  It is
    exposed so diagnostics can measure DESPOTIC's non-LVG uses of sigmaNT
    without changing production behavior.
    """
    if sigma_nt_cms < 0.0:
        raise ValueError("sigma_nt_cms must be non-negative")
    _configure_despotic_home()
    from despotic.chemistry import GOW

    species_order = tuple(species)
    last_lines = _nan_line_results(species_order)
    last_abundances: dict[str, float] = {}
    last_energy: dict[str, float] = {}
    last_mu = last_cv = last_eint = last_tg = float("nan")
    failed = True
    retryable_thermal_failure = False
    started = time.perf_counter()
    output = io.StringIO()
    cell = None

    try:
        cell = _make_despotic_cloud(initial_gas_upper)
        cell.nH = float(nH_val)
        cell.colDen = float(colDen_val)
        cell.Tg = float(Tg_init)
        cell.dVdr = float(dvdr_val)

        cell.sigmaNT = float(sigma_nt_cms)
        cell.comp.xoH2 = 0.1
        cell.comp.xpH2 = 0.4
        cell.comp.xHe = GOW_ELEMENTAL_ABUNDANCES["xHe"]
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
                info=dict(GOW_ELEMENTAL_ABUNDANCES),
                evolveTemp="iterateDust",
                tol=1e-6,
                maxTime=1e22,
                maxTempIter=200,
                tempEqParam={"escapeProbGeom": LVG_GEOMETRY},
            )
        _log_despotic_stdout(output)

        if not converged:
            raise RuntimeError("Chemical and thermal equilibrium did not converge")
        with contextlib.redirect_stdout(output):
            validated_energy = _validate_final_state(cell, species_order)
        cell.comp.computeDerived(cell.nH)
        last_mu = float(cell.comp.mu)
        last_cv = float(cell.comp.computeCv(cell.Tg))
        last_eint = float(cell.comp.computeEint(cell.Tg))
        last_tg = float(cell.Tg)
        last_abundances = dict(cell.chemabundances)
        last_energy = validated_energy
        for name, value in (("mu", last_mu), ("cv", last_cv), ("Eint", last_eint), ("Tg", last_tg)):
            if not np.isfinite(value) or value <= 0.0:
                raise RuntimeError(f"Final {name} output must be finite and positive")

        lines: dict[str, LineLumResult] = {}
        with contextlib.redirect_stdout(output):
            for name in species_order:
                transitions = cell.lineLum(name, escapeProbGeom=LVG_GEOMETRY, noRecompute=True)
                if name == "CO":
                    # Keep the legacy CO token for 1-0 and expose 2-1 without
                    # adding a generic transition dimension. Match by levels
                    # so a future LAMDA ordering change cannot swap the lines.
                    lines[name] = _extract_transition_result(transitions, 1, 0)
                    lines[CO21_TABLE_TOKEN] = _extract_transition_result(transitions, 2, 1)
                else:
                    lines[name] = _extract_line_result(transitions)
        _log_despotic_stdout(output)
        for name, line in lines.items():
            if (line.freq <= 0.0 or not all(
                    np.isfinite(getattr(line, field)) for field in LINE_RESULT_FIELDS)):
                raise RuntimeError(f"Final {name} line output is missing or non-finite")
        last_lines = lines
        failed = False

        if attempt_log is not None:
            integrations = getattr(cell, "_fixed_chemistry_attempts", ())
            local_retries = sum(item["offset_s"] != 0.0 for item in integrations)
            attempt_log.append(AttemptRecord(
                row_idx=-1 if row_idx is None else row_idx,
                col_idx=-1 if col_idx is None else col_idx,
                nH=float(nH_val), colDen=float(colDen_val), tg_guess=float(Tg_init),
                final_Tg=last_tg, converged=bool(converged),
                message=(f"Success; gas bracket upper={initial_gas_upper or 'native'}; "
                         f"chemical integrations={len(integrations)}; local-time retries={local_retries}"),
                duration=time.perf_counter() - started,
                dvdr_idx=dvdr_idx, dvdr=float(dvdr_val),
            ))
    except Exception as exc:
        retryable_thermal_failure = isinstance(exc, ThermalSolveError)
        if attempt_log is not None:
            attempt_log.append(AttemptRecord(
                row_idx=-1 if row_idx is None else row_idx,
                col_idx=-1 if col_idx is None else col_idx,
                nH=float(nH_val), colDen=float(colDen_val), tg_guess=float(Tg_init),
                final_Tg=float(cell.Tg) if cell is not None else last_tg,
                converged=False, message=str(exc),
                duration=time.perf_counter() - started,
                dvdr_idx=dvdr_idx, dvdr=float(dvdr_val),
            ))
        if log_failures:
            LOGGER.warning("Exception at nH=%s N_H=%s dVdr=%s: %s", nH_val, colDen_val, dvdr_val, exc)

    if failed and log_failures:
        LOGGER.warning("Failed at nH=%s N_H=%s dVdr=%s", nH_val, colDen_val, dvdr_val)

    _log_despotic_stdout(output)
    if failed:
        last_lines = _nan_line_results(species_order)
        last_abundances = {
            name: float("nan") for name in set(last_abundances) | set(species_order) | set(abundance_only)
        }
        last_energy = {name: float("nan") for name in last_energy}
        last_mu = last_cv = last_eint = last_tg = float("nan")

    # Ensure requested abundance-only names exist even if DESPOTIC omitted one.
    for name in abundance_only:
        last_abundances.setdefault(name, float("nan"))
    return (
        (
            MappingProxyType(last_lines),
            MappingProxyType(last_abundances),
            last_mu, last_cv, last_eint, last_tg,
            MappingProxyType(last_energy),
            failed,
        ),
        retryable_thermal_failure,
    )


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
    sigma_nt_cms: float = 2.0e5,
    attempt_log: list[AttemptRecord] | None = None,
) -> _PointResult:
    """Solve a point, retrying a failed thermal bracket on a fresh cloud.

    The alternative 1000 K upper endpoint is an initial bracket, not a
    temperature cap; decade expansion retains access to hotter roots.
    Successful native solves and non-thermal failures are not retried.
    """
    kwargs = dict(
        species=species, abundance_only=abundance_only, log_failures=log_failures,
        row_idx=row_idx, col_idx=col_idx, dvdr_idx=dvdr_idx, Tg_init=Tg_init,
        sigma_nt_cms=sigma_nt_cms, attempt_log=attempt_log,
    )
    result, retryable = _solve_gow_lvg_point_once(nH_val, colDen_val, dvdr_val, **kwargs)
    if result[-1] and retryable:
        result, _ = _solve_gow_lvg_point_once(
            nH_val, colDen_val, dvdr_val, **kwargs,
            initial_gas_upper=FALLBACK_GAS_BRACKET_UPPER_K,
        )
    return result
