#!/usr/bin/env python3
"""Validate every attempted fixed-depth state using retained Cloudy evidence.

This is an adoption gate separate from packaging: finite map coefficients do
not substitute for a successful process, converged output, or physical checks.
Original maps and diagnostic aggregates are never changed or filled.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from quokka2s.tables.abundances import abundance_metadata
from scripts.build_cloudy_model_depth_bundle import (
    AXIS_NAMES, AXIS_ORDER, LINES, T_TOLERANCE_DEX, COORDINATE_TOLERANCE_DEX,
    _axis_index, _parse_map, _path, _validate_parameter_file,
)
from scripts.cloudy_model_depth_common import (
    PC_IN_CM, common_commands, inspect_direct_output, reference_log_abundances, sha256,
)
from scripts.build_cloudy_sixline_tables import SED_DIRECTORY_NAME

SECTION_TYPES = {".cloudyIn": "Input", ".cloudyOut": "Output", ".lines": "Line punch",
                 ".physical": "Physical conditions", ".radius": "Radius"}
SECTION_RE = re.compile(r"^## (.+?) for T = ([+\-0-9.eE]+)\.\s*$", re.M)
MAP_ROUNDING_TOLERANCE_DEX = 5.1e-5


def _split_sections(path: Path, section_type: str, log_t: np.ndarray) -> tuple[dict, list[str]]:
    if not path.is_file():
        return {}, [f"missing aggregate {path.name}"]
    text = path.read_text(errors="replace")
    markers = list(SECTION_RE.finditer(text))
    errors, sections = [], {}
    if not markers:
        return {}, [f"no temperature sections in {path.name}"]
    if text[:markers[0].start()].strip():
        errors.append(f"unattributed content before first temperature in {path.name}")
    for number, marker in enumerate(markers):
        try:
            if marker[1] != section_type:
                raise ValueError(f"unexpected section type {marker[1]}")
            temperature = float(marker[2])
            if not np.isfinite(temperature) or temperature <= 0:
                raise ValueError("nonpositive or nonfinite temperature marker")
            index = _axis_index(float(np.log10(temperature)), log_t, "section temperature", T_TOLERANCE_DEX)
            if index in sections:
                raise ValueError(f"duplicate temperature section for axis index {index}")
            stop = markers[number + 1].start() if number + 1 < len(markers) else len(text)
            sections[index] = text[marker.end():stop].lstrip("\n")
        except ValueError as exc:
            errors.append(f"{path.name}: {exc}")
    return sections, errors


def _process_sections(path: Path, log_t: np.ndarray) -> tuple[dict, list[str]]:
    if not path.is_file():
        return {}, [f"missing aggregate {path.name}"]
    result, errors = {}, []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            parts = line.split()
            if len(parts) != 2:
                raise ValueError("process row requires temperature and raw wait status")
            temperature, raw_status = float(parts[0]), int(parts[1])
            if not np.isfinite(temperature) or temperature <= 0:
                raise ValueError("invalid process temperature")
            if raw_status > 65535:
                raise ValueError("raw process status exceeds the POSIX wait-status range")
            # The process sidecar preserves six mantissa decimals, unlike the
            # three-decimal aggregate section labels.
            index = _axis_index(float(np.log10(temperature)), log_t, "process temperature", 5.1e-7)
            if index in result:
                raise ValueError(f"duplicate process temperature for axis index {index}")
            if raw_status < 0:
                result[index] = dict(raw_wait_status=raw_status, returncode=None, exit_code=None,
                                     signal=None, available=False)
            else:
                returncode = os.waitstatus_to_exitcode(raw_status)
                result[index] = dict(raw_wait_status=raw_status, returncode=returncode,
                                     exit_code=returncode if returncode >= 0 else None,
                                     signal=-returncode if returncode < 0 else None, available=True)
        except ValueError as exc:
            errors.append(f"{path.name}:{number}: {exc}")
    return result, errors


def _input_issues(text: str, *, column: float, density: float, log_t: float, depth: float) -> list[str]:
    rows = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    issues = []
    for required in common_commands():
        if rows.count(required) != 1:
            issues.append(f"input missing or repeating common command: {required}")
    try:
        tokens = [shlex.split(line) for line in rows]
        for prefix, expected in ((["hden"], [density]), (["radius"], [1e30, PC_IN_CM * 10**depth])):
            found = [parts for parts in tokens if parts[:len(prefix)] == prefix]
            if len(found) != 1:
                raise ValueError(f"input requires one {' '.join(prefix)} command")
            parts = found[0]
            if prefix == ["hden"]:
                if len(parts) != 2 or abs(float(parts[1]) - density) > COORDINATE_TOLERANCE_DEX:
                    raise ValueError("input hden differs from manifest")
            elif (len(parts) != 4 or parts[-1] != "linear" or float(parts[1]) != expected[0]
                  or not np.isclose(float(parts[2]), expected[1], rtol=2e-12, atol=0)):
                raise ValueError("input fixed model depth differs from manifest")
        init = [parts for parts in tokens if parts[0] == "init"]
        if init != [["init", f"{SED_DIRECTORY_NAME}/logNH{column:g}.out"]]:
            raise ValueError("input attenuation SED differs from manifest")
        temperature = [parts for parts in tokens if parts[:2] == ["constant", "temperature"]]
        expected_t = float(f"{10.0**log_t:.6e}")
        if (len(temperature) != 1 or len(temperature[0]) != 5 or temperature[0][3:] != ["K", "linear"]
                or not np.isclose(float(temperature[0][2]), expected_t, rtol=1e-12, atol=5.1e-7)):
            raise ValueError("input constant temperature differs from manifest")
        abundance = [parts for parts in tokens if parts[0].lower() in ("abundances", "element", "metals")]
        if len(abundance) != 3 or abundance[0] != ["abundances", "default.abn"]:
            raise ValueError("input includes conflicting abundance commands or ordering")
        line_start = [i for i, row in enumerate(rows) if row.startswith("save last lines")]
        if len(line_start) != 1:
            raise ValueError("input requires one line-emissivity save command")
        start = line_start[0] + 1
        if rows[start:start + len(LINES)] != [item[1] for item in LINES] or rows[start + len(LINES)] != "end of lines":
            raise ValueError("input line-emissivity definitions differ from the adopted eight lines")
        if any(parts[0].lower() in ("grains", "coronal") or
               parts[:2] in (["stop", "column"], ["stop", "thickness"]) for parts in tokens):
            raise ValueError("input contains a conflicting physical command")
    except (ValueError, IndexError, OverflowError) as exc:
        issues.append(str(exc))
    return issues


def _reference_file(manifest: dict, base: Path, explicit: Path | None) -> tuple[Path, str | None]:
    provenance = manifest.get("provenance", {})
    declared = provenance.get("default_abn")
    declared_path = declared.get("path") if isinstance(declared, dict) else declared
    declared_hash = (declared.get("sha256") if isinstance(declared, dict) else None) or provenance.get("default_abn_sha256")
    selected = Path(explicit).expanduser().resolve() if explicit is not None else declared_path
    if selected is None:
        raise ValueError("Provide --default-abn or manifest provenance.default_abn; the reference is never inferred from the executable path")
    return _path(str(selected), base), declared_hash


def validate_maps(manifest_path: Path, output_report: Path, *, default_abn: Path | None = None) -> dict:
    """Write an independent evidence report, retaining every failed state."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    output_report = Path(output_report).expanduser().resolve()
    if output_report.exists():
        raise FileExistsError(output_report)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("axis_order") != AXIS_ORDER:
        raise ValueError("unexpected fixed-depth manifest schema")
    if manifest.get("status") != "completed":
        raise ValueError("execution must be completed before per-state validation")
    if manifest.get("cloudy_version") != "17.02" or manifest.get("abundance") != abundance_metadata():
        raise ValueError("manifest differs from the adopted Cloudy version or composition")
    if manifest.get("pc_in_cm") != PC_IN_CM:
        raise ValueError("manifest pc_in_cm differs from the audited direct-output checker")
    axes = {name: np.asarray(manifest["axes"][name], dtype=float) for name in AXIS_NAMES}
    for name, axis in axes.items():
        if axis.ndim != 1 or not axis.size or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError(f"invalid {name} axis")
    shape = tuple(axes[name].size for name in AXIS_NAMES)
    if len(manifest["maps"]) != shape[0] * shape[1] * shape[3]:
        raise ValueError("manifest map grid is incomplete")
    reference, declared_reference_hash = _reference_file(manifest, manifest_path.parent, default_abn)
    reference_hash = sha256(reference)
    expected_abundances = reference_log_abundances(reference)
    if not all(np.isfinite(value) for value in expected_abundances.values()):
        raise ValueError("default.abn reference must contain finite positive abundances")
    global_issues = []
    if declared_reference_hash is not None and reference_hash != declared_reference_hash:
        global_issues.append("default.abn SHA256 differs from declared provenance")
    if manifest.get("process_returncode", 0) != 0:
        global_issues.append("manifest records unsuccessful or unavailable CIAOLoop process return code")
    provenance = manifest.get("provenance", {})
    source_files = []
    parameter = _path(manifest["parameter_file"], manifest_path.parent)
    try:
        _validate_parameter_file(parameter, axes, PC_IN_CM)
        digest = sha256(parameter)
        source_files.append(dict(path=str(parameter), sha256=digest))
        expected = provenance.get("parameter_sha256", provenance.get("parameter_file_sha256"))
        if expected is not None and digest != expected:
            global_issues.append("parameter file SHA256 differs from declared provenance")
    except (ValueError, OSError) as exc:
        global_issues.append(f"parameter validation failed: {exc}")
    for source in provenance.get("sed_files", []):
        path = _path(source["path"], manifest_path.parent)
        try:
            digest = sha256(path)
            source_files.append(dict(path=str(path), sha256=digest))
            if digest != source["sha256"]:
                global_issues.append(f"SED source SHA256 differs from declared provenance: {path}")
        except OSError as exc:
            global_issues.append(f"missing SED source: {exc}")

    states, maps, seen, seen_paths = [], [], set(), set()
    diagnostic_failure = np.ones(shape, dtype=bool)
    raw_failure = np.ones((len(LINES), *shape), dtype=bool)
    with tempfile.TemporaryDirectory(prefix="cloudy-map-validation-") as temporary:
        extracted = Path(temporary) / "state"
        for record in manifest["maps"]:
            path = _path(record["path"], manifest_path.parent)
            if path in seen_paths:
                raise ValueError(f"duplicate map path: {path}")
            seen_paths.add(path)
            if any(path.parent.glob("*.mach")):
                raise ValueError(f"CIAOLoop worker marker remains near {path}")
            index = tuple(_axis_index(float(record[name]), axes[name], name, COORDINATE_TOLERANCE_DEX)
                          for name in ("log_NH_attenuation", "log_nH", "log_L_model_pc"))
            if index in seen:
                raise ValueError(f"duplicate manifest map coordinates: {index}")
            seen.add(index)
            map_issues, temperature_rows, hashes = [], {}, {}
            try:
                coords, rows = _parse_map(path, PC_IN_CM)
                hashes[".dat"] = sha256(path)
                if record.get("sha256") is not None and hashes[".dat"] != record["sha256"]:
                    map_issues.append("map SHA256 differs from declared provenance")
                for name, actual in coords.items():
                    if abs(actual - float(record[name])) > COORDINATE_TOLERANCE_DEX:
                        map_issues.append(f"map header {name} differs from manifest")
                for temperature, values in rows.items():
                    k = _axis_index(temperature, axes["log_T"], "map temperature", T_TOLERANCE_DEX)
                    if k in temperature_rows:
                        raise ValueError("duplicate map temperature assignment")
                    temperature_rows[k] = values
            except (ValueError, OSError) as exc:
                map_issues.append(f"map parsing failed: {exc}")
            sections = {}
            for suffix, section_type in SECTION_TYPES.items():
                sidecar = path.with_suffix(suffix)
                sections[suffix], errors = _split_sections(sidecar, section_type, axes["log_T"])
                # A missing entire diagnostic is represented per state below.
                map_issues.extend(error for error in errors if not error.startswith("missing aggregate"))
                if sidecar.is_file():
                    hashes[suffix] = sha256(sidecar)
            process_path = path.with_suffix(".process")
            processes, errors = _process_sections(process_path, axes["log_T"])
            map_issues.extend(error for error in errors if not error.startswith("missing aggregate"))
            if process_path.is_file():
                hashes[".process"] = sha256(process_path)
            maps.append(dict(path=str(path), coordinates={name: record[name] for name in
                        ("log_NH_attenuation", "log_nH", "log_L_model_pc")}, hashes=hashes, issues=map_issues))
            i, j, l = index
            for k, log_t in enumerate(axes["log_T"]):
                issues = list(global_issues) + list(map_issues)
                values = temperature_rows.get(k)
                row_present = k in temperature_rows
                if not row_present:
                    issues.append("missing requested map temperature row")
                elif values is None:
                    issues.append("Cloudy crash row contains only temperature")
                else:
                    raw_failure[:, i, j, k, l] = ~np.isfinite(values)
                    if not np.isfinite(values).all():
                        issues.append("map row contains nonfinite coefficients")
                process = processes.get(k, dict(available=False, raw_wait_status=None,
                                                returncode=None, exit_code=None, signal=None))
                if not process["available"]:
                    issues.append("missing or unavailable actual Cloudy process status")
                elif process["returncode"] != 0:
                    issues.append(f"Cloudy process failed: exit={process['exit_code']}, signal={process['signal']}")
                for suffix in SECTION_TYPES:
                    if k not in sections[suffix]:
                        issues.append(f"missing {suffix} evidence for requested temperature")
                if k in sections[".cloudyIn"]:
                    issues.extend(_input_issues(sections[".cloudyIn"][k], column=float(record["log_NH_attenuation"]),
                                               density=float(record["log_nH"]), log_t=float(log_t),
                                               depth=float(record["log_L_model_pc"])))
                if k in sections[".lines"]:
                    headers = [line.split("\t")[1:] for line in sections[".lines"][k].splitlines()
                               if line.startswith("#depth\t")]
                    expected_lines = [" ".join(item[1].split()) for item in LINES]
                    if len(headers) != 1 or [" ".join(value.split()) for value in headers[0]] != expected_lines:
                        issues.append("saved emissivity line header differs from adopted line order")
                checks = None
                if k in sections[".cloudyOut"]:
                    for source_suffix, temporary_suffix in ((".cloudyOut", ".out"), (".lines", ".lines"),
                                                            (".physical", ".physical"), (".radius", ".radius")):
                        target = extracted.with_suffix(temporary_suffix)
                        target.unlink(missing_ok=True)
                        if k in sections[source_suffix]:
                            target.write_text(sections[source_suffix][k])
                    try:
                        checks = inspect_direct_output(extracted, log_nH=float(record["log_nH"]),
                                  log_T=float(log_t), log_L_pc=float(record["log_L_model_pc"]),
                                  returncode=process["returncode"], expected_abundances=expected_abundances)
                        issues.extend(checks["issues"])
                    except (ValueError, OSError, IndexError, OverflowError) as exc:
                        issues.append(f"direct-output validation could not complete: {exc}")
                rounding_error = None
                if checks is not None and "emissivity_per_nH2" in checks and values is not None:
                    coefficient = np.asarray(checks["emissivity_per_nH2"])
                    if coefficient.shape == values.shape and np.isfinite(coefficient).all() and np.all(coefficient >= 0):
                        positive = coefficient > 0
                        if np.any(values[~positive] != -99) or np.any(values[positive] == -99):
                            issues.append("raw map true-zero sentinel differs from saved emission")
                        if np.any(positive) and np.isfinite(values[positive]).all():
                            rounding_error = float(np.max(np.abs(np.log10(coefficient[positive]) - values[positive])))
                            if rounding_error > MAP_ROUNDING_TOLERANCE_DEX:
                                issues.append("raw map coefficient differs from saved local emission beyond four-decimal rounding")
                issues = list(dict.fromkeys(issues))
                diagnostic_failure[i, j, k, l] = bool(issues)
                states.append(dict(indices=[i, j, k, l], map_path=str(path),
                                   **{name: float(axes[name][axis_index]) for name, axis_index in zip(AXIS_NAMES, (i, j, k, l))},
                                   valid=not issues, issues=issues, process=process,
                                   raw_map_row_present=row_present, raw_map_crash_row=row_present and values is None,
                                   raw_line_failure_mask=raw_failure[:, i, j, k, l].tolist(),
                                   map_saved_emission_max_error_dex=rounding_error, physical_checks=checks))

    report = dict(schema_version=1, manifest=str(manifest_path), manifest_sha256=sha256(manifest_path),
                  passed=not diagnostic_failure.any(), execution_status=manifest["status"],
                  state_count=len(states), valid_state_count=int((~diagnostic_failure).sum()),
                  invalid_state_count=int(diagnostic_failure.sum()), global_issues=global_issues,
                  axis_order=",".join(AXIS_NAMES), axes={name: axis.tolist() for name, axis in axes.items()},
                  diagnostic_failure_mask=diagnostic_failure.tolist(), raw_map_failure_mask=raw_failure.tolist(),
                  raw_map_failure_axis_order=AXIS_ORDER, states=states, maps=maps,
                  reference=dict(path=str(reference), sha256=reference_hash, declared_sha256=declared_reference_hash),
                  source_files=source_files,
                  checker_sources={str(path): sha256(path) for path in (Path(__file__).resolve(), ROOT / "scripts/cloudy_model_depth_common.py")},
                  interpretation="Independent per-state adoption gate; failed evidence remains unavailable. No raw values were changed, filled, packed, or adopted.")
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with output_report.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--default-abn", type=Path)
    args = parser.parse_args()
    report = validate_maps(args.manifest, args.output_report, default_abn=args.default_abn)
    print(json.dumps({key: report[key] for key in ("passed", "state_count", "valid_state_count", "invalid_state_count")}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
