#!/usr/bin/env python3
"""Append two lower-density planes without recomputing existing Cloudy nodes.

Run ``plan``, ``smoke``, ``build``, then ``merge`` with the same arguments.
The smoke runs both new densities and every existing attenuation field at
3.6 K, 1e5 K and 1e9 K by default. Production uses the source table's complete
temperature axis. All raw maps and Cloudy diagnostics remain in runtime-dir.
The source archive is read-only; merge requires a distinct, new output path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from quokka2s.cloudy.table_definition import EXPECTED_AXIS_ORDER, LINES
from quokka2s.file_provenance import file_sha256
from quokka2s.paths import resolve_path
from quokka2s.physics.composition import QUOKKA_MASS_FRACTIONS


NODE_KEYS = (
    "log_emissivity_per_nH2",
    "emissivity_per_nH2",
    "failure_mask",
    "original_failure_mask",
    "zero_mask",
    "interpolated_mask",
)
PROVENANCE_KEY = "density_extension_metadata_json"
T_TOLERANCE_DEX = 5.1e-4
ZERO_LIMIT = -90.0
RUN_RE = re.compile(r"_run([1-9][0-9]*)\.dat$")
HDEN_RE = re.compile(r"^#\s*hden\s+(.+?)\s*$")
INIT_RE = re.compile(r'^#\s*init\s+"[^\"]*logNH([0-9.]+)\.out"\s*$')


def _json_text(value: dict) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _write_new_or_identical(path: Path, content: str) -> None:
    """Allow a planned parameter file to be reused, never replace other data."""
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f"different existing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as destination:
        destination.write(content)


def _identity(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": file_sha256(path)}


def _read_source(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        source = {key: archive[key] for key in archive.files}
    if PROVENANCE_KEY in source:
        raise ValueError("source already contains a density extension; choose the original table")
    if str(source["axis_order"]) != EXPECTED_AXIS_ORDER:
        raise ValueError("unsupported source axis order")
    if tuple(source["line_keys"].tolist()) != tuple(item[0] for item in LINES):
        raise ValueError("source lines differ from the eight-line CIAOLoop recipe")
    for key in ("log_NH_attenuation", "log_nH", "log_T"):
        axis = source[key]
        if axis.ndim != 1 or axis.size < 2:
            raise ValueError(f"invalid source axis: {key}")
        if not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError(f"source axis must increase strictly: {key}")
    log_t = source["log_T"]
    uniform_log_t = np.linspace(log_t[0], log_t[-1], log_t.size)
    if not np.allclose(log_t, uniform_log_t, rtol=0.0, atol=1e-12):
        raise ValueError("CIAOLoop requires a uniformly spaced log-temperature axis")
    expected_shape = (
        len(LINES),
        source["log_NH_attenuation"].size,
        source["log_nH"].size,
        log_t.size,
    )
    for key in NODE_KEYS:
        if source[key].shape != expected_shape:
            raise ValueError(f"unexpected shape for {key}: {source[key].shape}")
    return source


def _new_densities(source: dict[str, np.ndarray]) -> np.ndarray:
    """Continue the first source density interval downward by exactly two nodes."""
    old_axis = source["log_nH"]
    spacing = float(old_axis[1] - old_axis[0])
    # Match the decimal precision sent to CIAOLoop so map coordinates are exact.
    new_values = [float(f"{old_axis[0] - offset * spacing:.15g}") for offset in (2, 1)]
    return np.asarray(new_values, dtype=old_axis.dtype)


def _recipe(path: Path, source: dict[str, np.ndarray]) -> list[str]:
    """Reuse the historical commands and reject an incompatible geometry recipe."""
    settings: dict[str, list[str]] = {}
    commands = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("command "):
            commands.append(line)
        elif "=" in line:
            key = line.split("=", 1)[0].strip()
            value = line.split("=", 1)[1].strip()
            settings.setdefault(key, []).append(value)
        elif not line.startswith("loop "):
            raise ValueError(f"unsupported recipe statement: {line}")
    if settings.get("lineMapLine") != [item[1] for item in LINES]:
        raise ValueError("historical recipe has a different ordered line list")
    required = {"cloudyRunMode": "4", "coolingMapUseJeansLength": "1", "coolingScaleFactor": "1"}
    for key, value in required.items():
        if settings.get(key) != [value]:
            raise ValueError(f"historical recipe must set {key} = {value}")
    cap = float(settings["coolingMapMaximumJeansLength"][-1])
    if cap != float(source["jeans_length_cap_cm"]):
        raise ValueError("historical Jeans cap differs from source metadata")
    recipe_t = np.linspace(
        np.log10(float(settings["coolingMapTmin"][-1])),
        np.log10(float(settings["coolingMapTmax"][-1])),
        int(settings["coolingMapTpoints"][-1]),
    )
    if recipe_t.shape != source["log_T"].shape or not np.allclose(
        recipe_t, source["log_T"], rtol=0.0, atol=1e-12
    ):
        raise ValueError("historical temperature grid differs from source table")
    historical_hash = str(source.get("parameter_file_sha256", ""))
    if historical_hash and historical_hash != file_sha256(path):
        raise ValueError("recipe SHA256 differs from source table's historical parameter file")
    if not commands:
        raise ValueError("historical recipe has no Cloudy commands")
    return commands


def _inputs(args: argparse.Namespace) -> tuple[dict, dict[str, np.ndarray], list[str]]:
    source = _read_source(args.source_table)
    commands = _recipe(args.recipe_parameter, source)
    for path in (args.cloudy_exe, args.radiation_directory, args.runtime_dir):
        if any(character.isspace() for character in str(path)):
            raise ValueError(f"CIAOLoop paths cannot contain whitespace: {path}")
    if args.source_table == args.output_table or args.output_table.exists():
        raise FileExistsError(f"output must be a new path distinct from source: {args.output_table}")
    radiation = []
    sed_directory_names = set()
    for value in source["log_NH_attenuation"]:
        init_path = args.radiation_directory / f"logNH{value:g}.out"
        radiation.append(_identity(init_path))
        references = re.findall(r'^table SED "([^\"]+)"\s*$', init_path.read_text(), re.MULTILINE)
        if len(references) != 1:
            raise ValueError(f"expected one table SED reference: {init_path}")
        reference = Path(references[0])
        if reference.is_absolute() or len(reference.parts) != 2 or reference.name != f"logNH{value:g}.sed":
            raise ValueError(f"unsupported relative SED reference: {reference}")
        sed_directory_names.add(reference.parts[0])
        radiation.append(_identity(args.radiation_directory / reference.name))
    if len(sed_directory_names) != 1:
        raise ValueError("all radiation init files must use the same relative SED directory")
    smoke_log_t = args.smoke_log_temperature
    if smoke_log_t is None:
        smoke_log_t = [float(source["log_T"][0]), 5.0, float(source["log_T"][-1])]
    if not np.isfinite(smoke_log_t).all() or len(set(smoke_log_t)) != len(smoke_log_t):
        raise ValueError("smoke log-temperatures must be finite and distinct")
    plan = {
        "source_table": _identity(args.source_table),
        "output_table": str(args.output_table),
        "recipe_parameter": _identity(args.recipe_parameter),
        "cloudy_executable": _identity(args.cloudy_exe),
        "cialoop": _identity(args.cialoop),
        "radiation_files": radiation,
        "radiation_relative_directory": next(iter(sed_directory_names)),
        "new_log_nH": _new_densities(source).tolist(),
        "unchanged_log_NH_attenuation": source["log_NH_attenuation"].tolist(),
        "unchanged_log_T": source["log_T"].tolist(),
        "smoke_log_temperatures": smoke_log_t,
        "new_jeans_hydrogen_mass_fraction": QUOKKA_MASS_FRACTIONS["X"],
        "old_metadata_policy": "all original metadata retained verbatim; extension recipe applies only to new planes",
        "new_state_count": int(2 * source["log_NH_attenuation"].size * source["log_T"].size),
        "workers": args.workers,
    }
    digest = hashlib.sha256(_json_text(plan).encode()).hexdigest()
    plan["input_fingerprint"] = digest
    return plan, source, commands


def _parameter_text(args: argparse.Namespace, plan: dict, source: dict, commands: list[str], phase: str, log_t: np.ndarray) -> str:
    output_dir = args.runtime_dir / phase / "maps"
    lines = [
        "# Incremental density extension; existing table nodes are never run.",
        f"cloudyExe = {args.cloudy_exe}",
        "saveCloudyOutputFiles = 1",
        "exitOnCrash = 0",
        f"outputFilePrefix = density_extension_{phase}",
        f"outputDir = {output_dir}",
        "runStartIndex = 1",
        "test = 0",
        "cloudyRunMode = 4",
        *(f"lineMapLine = {item[1]}" for item in LINES),
        f"coolingMapTmin = {10.0 ** float(log_t[0]):.17g}",
        f"coolingMapTmax = {10.0 ** float(log_t[-1]):.17g}",
        f"coolingMapTpoints = {log_t.size}",
        "coolingScaleFactor = 1",
        "coolingMapUseJeansLength = 1",
        f"coolingMapHydrogenMassFraction = {QUOKKA_MASS_FRACTIONS['X']!r}",
        f"coolingMapMaximumJeansLength = {float(source['jeans_length_cap_cm']):.17g}",
        *commands,
        "loop [hden] " + " ".join(f"{value:.15g}" for value in plan["new_log_nH"]),
        f'loop [init "{plan["radiation_relative_directory"]}/logNH*.out"] '
        + " ".join(f"{value:g}" for value in source["log_NH_attenuation"]),
    ]
    return "\n".join(lines) + "\n"


def _prepare(args: argparse.Namespace, plan: dict, source: dict, commands: list[str]) -> None:
    _write_new_or_identical(args.runtime_dir / "plan.json", _json_text(plan))
    phases = [(f"smoke_{index}", np.asarray([value])) for index, value in enumerate(plan["smoke_log_temperatures"])]
    phases.append(("production", source["log_T"]))
    for phase, log_t in phases:
        path = args.runtime_dir / phase / "extension.par"
        content = _parameter_text(args, plan, source, commands, phase, log_t)
        _write_new_or_identical(path, content)
        # The historical init commands contain relative table-SED paths.
        # Preserve their contents and provide their exact directory in each cwd.
        radiation_link = path.parent / plan["radiation_relative_directory"]
        if radiation_link.is_symlink():
            if radiation_link.resolve() != args.radiation_directory:
                raise ValueError(f"radiation symlink points elsewhere: {radiation_link}")
        elif radiation_link.exists():
            raise FileExistsError(f"radiation link destination already exists: {radiation_link}")
        else:
            radiation_link.symlink_to(args.radiation_directory, target_is_directory=True)


def _run(args: argparse.Namespace, phase: str) -> None:
    directory = args.runtime_dir / phase
    output = directory / "maps"
    if output.exists():
        raise FileExistsError(f"phase output already exists; preserve it and use a new runtime directory: {output}")
    log_path = directory / "cialoop.log"
    command = ["perl", str(args.cialoop), "-np", str(args.workers), "extension.par"]
    with log_path.open("x") as log:
        process = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        raise RuntimeError(f"CIAOLoop exited {process.returncode}; read {log_path}")


def _axis_index(value: float, axis: np.ndarray, tolerance: float, path: Path) -> int:
    index = int(np.abs(axis - value).argmin())
    if abs(float(axis[index]) - value) > tolerance:
        raise ValueError(f"off-grid coordinate {value}: {path}")
    return index


def _load_maps(directory: Path, attenuation: np.ndarray, density: np.ndarray, log_t: np.ndarray) -> tuple[np.ndarray, list[Path]]:
    """Read complete map coordinates, retaining crash rows and valid-zero sentinels."""
    if list(directory.glob("*.mach")):
        raise RuntimeError(f"CIAOLoop workers are still active: {directory}")
    files = {}
    for path in directory.glob("*_run*.dat"):
        match = RUN_RE.search(path.name)
        if match is None:
            raise ValueError(f"unexpected map filename: {path}")
        run = int(match.group(1))
        if run in files:
            raise ValueError(f"duplicate map run index {run}: {directory}")
        files[run] = path
    if set(files) != set(range(1, attenuation.size * density.size + 1)):
        raise ValueError(f"incomplete or extra map run indices: {directory}")
    raw = np.full((len(LINES), attenuation.size, density.size, log_t.size), np.nan)
    seen = np.zeros(raw.shape[1:], dtype=bool)
    for path in files.values():
        map_density = None
        map_attenuation = None
        header = None
        rows = []
        for line in path.read_text().splitlines():
            match = HDEN_RE.match(line)
            if match:
                map_density = float(match.group(1))
            match = INIT_RE.match(line)
            if match:
                map_attenuation = float(match.group(1))
            if line.startswith("#Te"):
                header = tuple(line.split()[1:])
            if line.strip() and not line.lstrip().startswith("#"):
                rows.append(line.split())
        if header != tuple(item[2] for item in LINES):
            raise ValueError(f"wrong ordered line header: {path}")
        if map_density is None or map_attenuation is None or len(rows) != log_t.size:
            raise ValueError(f"incomplete map coordinates or temperature rows: {path}")
        i = _axis_index(map_attenuation, attenuation, 1e-12, path)
        j = _axis_index(map_density, density, 1e-12, path)
        for row in rows:
            if len(row) not in (1, 1 + len(LINES)):
                raise ValueError(f"wrong map row column count: {path}")
            k = _axis_index(float(row[0]), log_t, T_TOLERANCE_DEX, path)
            if seen[i, j, k]:
                raise ValueError(f"duplicate physical state: {path}")
            seen[i, j, k] = True
            if len(row) > 1:
                values = np.asarray(row[1:], dtype=float)
                # CIAOLoop crash rows have one column. A non-finite line value
                # remains unavailable independently of the other seven lines.
                raw[:, i, j, k] = np.where(np.isfinite(values), values, np.nan)
    if not seen.all():
        raise ValueError(f"incomplete physical state grid: {directory}")
    return raw, [files[index] for index in sorted(files)]


def _diagnostics(files: list[Path], log_t: np.ndarray) -> dict:
    """Check actual Cloudy exits and collect warnings in addition to map rows."""
    errors = []
    warning_files = []
    evidence = []
    for path in files:
        stem = path.with_suffix("")
        process_path = stem.with_suffix(".process")
        output_path = stem.with_suffix(".cloudyOut")
        if not process_path.is_file() or not output_path.is_file():
            errors.append(f"missing saved process/output evidence: {stem}")
            continue
        status = np.loadtxt(process_path, ndmin=2)
        if status.shape != (log_t.size, 2) or np.any(status[:, 1] != 0):
            errors.append(f"nonzero or incomplete Cloudy process statuses: {process_path}")
        elif not np.allclose(np.log10(status[:, 0]), log_t, rtol=0.0, atol=1e-6):
            errors.append(f"Cloudy process temperature grid differs: {process_path}")
        output = output_path.read_text(errors="replace")
        if output.count("Cloudy exited OK") != log_t.size:
            errors.append(f"wrong number of successful Cloudy exits: {output_path}")
        if re.search(r"DISASTER|BOTCHED|PROBLEM\s+.*(?:failed|failure|abort)", output, re.IGNORECASE):
            errors.append(f"Cloudy fatal diagnostic: {output_path}")
        warning_path = stem.with_suffix(".warnings")
        if warning_path.exists():
            warning_files.append(_identity(warning_path))
            warnings = warning_path.read_text(errors="replace")
            if re.search(r"(?:converg\w*.*(?:fail|not)|(?:fail|not).*converg)", warnings, re.IGNORECASE):
                errors.append(f"Cloudy convergence warning needs review: {warning_path}")
        evidence.extend([_identity(path), _identity(process_path), _identity(output_path)])
    return {"errors": errors, "warning_files": warning_files, "evidence": evidence}


def _check_phase(args: argparse.Namespace, plan: dict, source: dict, phase: str, log_t: np.ndarray) -> tuple[np.ndarray, dict]:
    raw, files = _load_maps(
        args.runtime_dir / phase / "maps",
        source["log_NH_attenuation"],
        np.asarray(plan["new_log_nH"]),
        log_t,
    )
    diagnostic = _diagnostics(files, log_t)
    failures = ~np.isfinite(raw)
    failure_states = np.any(failures, axis=0)
    line_maxima = np.max(np.where(np.isfinite(raw), raw, -np.inf), axis=(1, 2, 3))
    positive_lines = line_maxima > ZERO_LIMIT
    if phase == "production" and not positive_lines.all():
        missing = [item[0] for item, positive in zip(LINES, positive_lines) if not positive]
        diagnostic["errors"].append(f"production lines have no positive coefficient: {missing}")
    report = {
        "phase": phase,
        "input_fingerprint": plan["input_fingerprint"],
        "state_count": int(np.prod(raw.shape[1:])),
        "line_failure_count": int(np.count_nonzero(failures)),
        "failure_state_count": int(np.count_nonzero(failure_states)),
        "failure_states": [
            {
                "log_NH_attenuation": float(source["log_NH_attenuation"][index[0]]),
                "log_nH": float(plan["new_log_nH"][index[1]]),
                "log_T": float(log_t[index[2]]),
                "line_keys": [
                    item[0]
                    for line_index, item in enumerate(LINES)
                    if failures[(line_index, *index)]
                ],
            }
            for index in np.argwhere(failure_states)
        ],
        "zero_line_count": int(np.count_nonzero(np.isfinite(raw) & (raw <= ZERO_LIMIT))),
        "line_maximum_log_coefficient": [float(value) if np.isfinite(value) else None for value in line_maxima],
        "line_has_positive_coefficient": positive_lines.tolist(),
        "passed": bool(not failures.any() and not diagnostic["errors"]),
        **diagnostic,
    }
    return raw, report


def _summarize_smoke(plan: dict, reports: list[dict]) -> dict:
    """Require ionized-carbon emission somewhere across the three smoke temperatures."""
    required = ("ciii_977", "ciii_1907", "ciii_1909", "civ_1548", "civ_1551")
    positive = np.any([item["line_has_positive_coefficient"] for item in reports], axis=0)
    missing = [item[0] for item, found in zip(LINES, positive) if item[0] in required and not found]
    return {
        "input_fingerprint": plan["input_fingerprint"],
        "passed": bool(all(item["passed"] for item in reports) and not missing),
        "state_count": sum(item["state_count"] for item in reports),
        "required_positive_line_keys": list(required),
        "missing_required_positive_line_keys": missing,
        "line_has_positive_coefficient": positive.tolist(),
        "phases": reports,
    }


def _require_smoke(args: argparse.Namespace, plan: dict) -> dict:
    report = json.loads((args.runtime_dir / "smoke_report.json").read_text())
    if not report["passed"] or report["input_fingerprint"] != plan["input_fingerprint"]:
        raise ValueError("successful smoke report for these exact inputs is required")
    for phase in report["phases"]:
        for record in phase["evidence"] + phase["warning_files"]:
            if file_sha256(record["path"]) != record["sha256"]:
                raise ValueError(f"smoke evidence changed after validation: {record['path']}")
    return report


def _verify_input_files(plan: dict) -> None:
    """Detect source, executable, runner or radiation changes during a long run."""
    records = [plan[key] for key in ("source_table", "recipe_parameter", "cloudy_executable", "cialoop")]
    records.extend(plan["radiation_files"])
    for record in records:
        if file_sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"input changed after planning: {record['path']}")


def _payload(raw: np.ndarray) -> dict[str, np.ndarray]:
    failure = ~np.isfinite(raw)
    zero = ~failure & (raw <= ZERO_LIMIT)
    coefficient = np.zeros_like(raw)
    positive = ~failure & ~zero
    coefficient[positive] = np.power(10.0, raw[positive])
    return {
        "log_emissivity_per_nH2": raw,
        "emissivity_per_nH2": coefficient,
        "failure_mask": failure,
        "original_failure_mask": failure.copy(),
        "zero_mask": zero,
        "interpolated_mask": np.zeros_like(failure),
    }


def _verify_preserved(source: dict, merged: dict) -> None:
    for key, original in source.items():
        if key == "log_nH":
            restored = merged[key][2:]
        elif key in NODE_KEYS:
            restored = merged[key][:, :, 2:, :]
        else:
            restored = merged[key]
        if restored.dtype != original.dtype or restored.shape != original.shape:
            raise ValueError(f"old dtype or shape changed: {key}")
        if restored.tobytes(order="C") != original.tobytes(order="C"):
            raise ValueError(f"old member bytes changed: {key}")


def _merge(args: argparse.Namespace, plan: dict, source: dict, raw: np.ndarray, report: dict) -> dict:
    if not report["passed"]:
        raise ValueError("production validation failed; inspect production_report.json")
    merged = dict(source)
    merged["log_nH"] = np.concatenate((_new_densities(source), source["log_nH"]))
    for key, extension in _payload(raw).items():
        extension = extension.astype(source[key].dtype, copy=False)
        merged[key] = np.concatenate((extension, source[key]), axis=2)
    provenance = {
        "plan": plan,
        "extension_script": _identity(Path(__file__).resolve()),
        "smoke_report": _identity(args.runtime_dir / "smoke_report.json"),
        "production_report": _identity(args.runtime_dir / "production_report.json"),
        "new_plane_geometry": "Jeans length with existing cap; explicit new-plane X_H only",
        "old_member_validation": "all old node and metadata dtypes, shapes and C-order bytes identical",
        "new_failure_policy": "no fill; unavailable nodes retain masks; merge requires all new states valid",
        "created_unix_seconds": time.time(),
    }
    merged[PROVENANCE_KEY] = np.asarray(_json_text(provenance))
    _verify_preserved(source, merged)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create also protects against another process adopting this path.
    with args.output_table.open("xb") as destination:
        np.savez_compressed(destination, **merged)
    with np.load(args.output_table, allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    _verify_preserved(source, saved)
    result = {
        "output_table": _identity(args.output_table),
        "source_table": plan["source_table"],
        "new_shape": list(saved[NODE_KEYS[0]].shape),
        "all_old_members_preserved": True,
        "new_state_count": plan["new_state_count"],
        "new_log_nH": plan["new_log_nH"],
        "passed": True,
    }
    _write_new_or_identical(args.runtime_dir / "merge_report.json", _json_text(result))
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "smoke", "build", "merge"))
    parser.add_argument("--source-table", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--cloudy-exe", type=Path, required=True)
    parser.add_argument("--radiation-directory", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke-log-temperature", type=float, nargs="+")
    parser.add_argument(
        "--recipe-parameter",
        type=Path,
        default=root / "runtime/cloudy_eightline/examples/grackle/hm2012_attgrid_ism_nh21_cmb_cr_defaultabund_eightline_jeans_7x10x21.par",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for key in ("source_table", "output_table", "cloudy_exe", "radiation_directory", "runtime_dir", "recipe_parameter"):
        setattr(args, key, resolve_path(getattr(args, key)))
    args.cialoop = root / "vendor/cloudy_cooling_tools/CIAOLoop_lines"
    plan, source, commands = _inputs(args)
    _prepare(args, plan, source, commands)
    if args.phase == "plan":
        print(_json_text(plan))
        return
    if args.phase == "smoke":
        reports = []
        for index, value in enumerate(plan["smoke_log_temperatures"]):
            phase = f"smoke_{index}"
            _run(args, phase)
            _verify_input_files(plan)
            _, report = _check_phase(args, plan, source, phase, np.asarray([value]))
            reports.append(report)
        report = _summarize_smoke(plan, reports)
        _write_new_or_identical(args.runtime_dir / "smoke_report.json", _json_text(report))
    else:
        _require_smoke(args, plan)
        if args.phase == "build":
            _run(args, "production")
        _verify_input_files(plan)
        raw, report = _check_phase(args, plan, source, "production", source["log_T"])
        _write_new_or_identical(args.runtime_dir / "production_report.json", _json_text(report))
        if args.phase == "merge":
            report = _merge(args, plan, source, raw, report)
    print(_json_text(report))
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
