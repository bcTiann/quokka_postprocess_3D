#!/usr/bin/env python3
"""Prepare or explicitly run shared-composition, fixed-depth Cloudy maps.

Preparation never launches Cloudy. Execution requires --run, uses CIAOLoop's
existing -np worker mechanism, and retains raw outputs for later per-state
validation. A completed build records attempted states, not converged models.
Packing is a separate explicit operation using build_cloudy_model_depth_bundle.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from quokka2s.tables.abundances import abundance_metadata
from scripts.build_cloudy_sixline_tables import HM12_LOG_NH, LOG_NH_DENSITY, LINES, SED_DIRECTORY_NAME
from scripts.build_cloudy_model_depth_bundle import (
    AXIS_NAMES, AXIS_ORDER, COORDINATE_TOLERANCE_DEX, T_TOLERANCE_DEX,
    _axis_index, _parse_map, _validate_parameter_file,
)
from scripts.cloudy_model_depth_common import LOG_DEPTH_PC, LOG_T, PC_IN_CM, common_commands, sha256

STEM = "hm2012_attgrid_ism_nh21_cmb_cr_sharedabund_eightline_fixeddepth"
MANIFEST_NAME = "build_manifest.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    """Replace our status manifest atomically, without partial JSON readers."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_runtime_path(path: Path) -> Path:
    # CIAOLoop constructs shell commands without quoting paths. Preparation
    # must reject shell metacharacters as well as spaces before any launch.
    path = Path(path).expanduser().resolve()
    if not re.fullmatch(r"[A-Za-z0-9_./+:-]+", str(path)):
        raise ValueError(f"CIAOLoop runtime paths cannot contain spaces or shell metacharacters: {path}")
    return path


def _axes(depth_points: int = 10) -> dict[str, np.ndarray]:
    if depth_points not in (10, 19, 37):
        raise ValueError("depth_points must be 10, 19, or 37")
    return dict(log_NH_attenuation=np.asarray(HM12_LOG_NH),
                log_nH=np.asarray(LOG_NH_DENSITY), log_T=np.asarray(LOG_T),
                log_L_model_pc=np.linspace(LOG_DEPTH_PC[0], LOG_DEPTH_PC[-1], depth_points))


def _sed_sources(sed_dir: Path) -> tuple[dict, list[Path]]:
    """Require the prior SED roundtrip report and its exact init filenames."""
    report_path = sed_dir / "build_report.json"
    report = json.loads(report_path.read_text())
    if tuple(report.get("hm12_log_NH_attenuation", ())) != HM12_LOG_NH:
        raise ValueError("SED report does not describe the seven adopted attenuation fields")
    if report.get("ism_log_NH_attenuation") != 21.0:
        raise ValueError("SED report does not use the adopted ISM attenuation")
    if report.get("extinguish_leak") != 0.0 or report.get("external_grackle_hm12_used") is not False:
        raise ValueError("SED report differs from the adopted Cloudy-native, zero-leak prescription")
    if report.get("energy_mesh_identical_for_all_exports") is not True:
        raise ValueError("SED report lacks the validated common energy mesh")
    tolerance = float(report["roundtrip_maximum_allowed_error_dex"])
    if not np.isfinite(tolerance) or not 0 < tolerance <= 0.001:
        raise ValueError("SED roundtrip tolerance is missing or exceeds the adopted 0.001 dex")
    entries = report.get("entries", [])
    if len(entries) != len(HM12_LOG_NH):
        raise ValueError("SED report has incomplete or duplicate entries")
    by_column = {float(entry["hm12_log_NH_attenuation"]): entry for entry in entries}
    if set(by_column) != set(HM12_LOG_NH):
        raise ValueError("SED report entries differ from the adopted attenuation fields")
    files = [report_path]
    for column in HM12_LOG_NH:
        entry = by_column[column]
        error = float(entry["roundtrip"]["maximum_relevant_absolute_error_dex"])
        if not np.isfinite(error) or error > tolerance or error < 0:
            raise ValueError(f"SED roundtrip did not pass for log NH = {column}")
        init = sed_dir / f"logNH{column:.17g}.out"
        sed = sed_dir / f"logNH{column:.17g}.sed"
        for path in (init, sed):
            if not path.is_file():
                raise FileNotFoundError(path)
        expected = [f'table SED "{SED_DIRECTORY_NAME}/{sed.name}"',
                    entry["fnu_normalization_command"]]
        if [line.strip() for line in init.read_text().splitlines() if line.strip()] != expected:
            raise ValueError(f"SED init differs from the reported input: {init}")
        files.extend((init, sed))
    return report, files


def write_parameter_file(destination: Path, *, cloudy_exe: Path, raw_dir: Path,
                         depth_points: int = 10) -> None:
    """Use fixed depth as the innermost loop; all numeric axes retain 17 digits."""
    axes = _axes(depth_points)
    lines = [
        f"# Shared composition; explicit fixed model depth; 7 x 10 x 21 x {depth_points} states",
        f"cloudyExe = {cloudy_exe}", "saveCloudyOutputFiles = 1", "exitOnCrash = 0",
        f"outputFilePrefix = {STEM}", f"outputDir = {raw_dir}",
        "runStartIndex = 1", "test = 0", "cloudyRunMode = 4",
        *(f"lineMapLine = {line}" for line in LINES),
        f"coolingMapTmin = {10.0 ** axes['log_T'][0]:.17g}",
        f"coolingMapTmax = {10.0 ** axes['log_T'][-1]:.17g}",
        f"coolingMapTpoints = {axes['log_T'].size}",
        "coolingScaleFactor = 1", "coolingMapUseJeansLength = 0",
        *(f"command {command}" for command in common_commands()),
        "loop [hden] " + " ".join(f"{value:.17g}" for value in axes["log_nH"]),
        f'loop [init "{SED_DIRECTORY_NAME}/logNH*.out"] '
        + " ".join(f"{value:.17g}" for value in axes["log_NH_attenuation"]),
        "loop [radius 1e30 * linear] "
        + " ".join(f"{PC_IN_CM * 10.0 ** value:.17g}" for value in axes["log_L_model_pc"]),
    ]
    with destination.open("x") as handle:
        handle.write("\n".join(lines) + "\n")


def expected_maps(depth_points: int = 10) -> list[dict]:
    """Predicted filenames only; actual headers are independently checked later."""
    records = []
    for density in LOG_NH_DENSITY:
        for column in HM12_LOG_NH:
            for depth in _axes(depth_points)["log_L_model_pc"]:
                run = len(records) + 1
                records.append(dict(path=f"raw_maps/{STEM}_run{run}.dat", run_index=run,
                                    log_nH=float(density), log_NH_attenuation=float(column),
                                    log_L_model_pc=float(depth)))
    return records


def _command(cialoop: Path, workers: int, parameter_name: str) -> list[str]:
    command = ["perl", str(cialoop), "-np", str(workers), parameter_name]
    if shutil.which("caffeinate") and os.uname().sysname == "Darwin":
        command = ["caffeinate", "-dimsu", *command]
    return command


def prepare_build(*, output_dir: Path, cloudy_exe: Path, sed_dir: Path,
                  workers: int, cialoop: Path | None = None,
                  depth_points: int = 10) -> dict:
    """Create a fresh reviewable build directory; never overwrite an old one."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    axes = _axes(depth_points)
    output_dir = _safe_runtime_path(output_dir)
    cloudy_exe = _safe_runtime_path(cloudy_exe)
    cialoop = _safe_runtime_path(cialoop or ROOT / "vendor/cloudy_cooling_tools/CIAOLoop_lines")
    sed_dir = Path(sed_dir).expanduser().resolve()
    if not cloudy_exe.is_file() or not os.access(cloudy_exe, os.X_OK):
        raise ValueError(f"Cloudy executable is absent or not executable: {cloudy_exe}")
    if not cialoop.is_file():
        raise FileNotFoundError(cialoop)
    if output_dir.exists():
        raise FileExistsError(f"build directory already exists; inspect it before any restart: {output_dir}")
    sed_report, sources = _sed_sources(sed_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    copied_dir = output_dir / SED_DIRECTORY_NAME
    copied_dir.mkdir()
    sed_records = []
    for source in sources:
        destination = copied_dir / source.name
        digest = sha256(source)
        shutil.copyfile(source, destination)
        if sha256(destination) != digest or sha256(source) != digest:
            raise RuntimeError(f"SED source changed during its copy: {source}")
        sed_records.append(dict(source=str(source), path=str(destination.relative_to(output_dir)),
                                sha256=digest))
    raw_dir = output_dir / "raw_maps"
    raw_dir.mkdir()
    parameter = output_dir / "fixed_depth.par"
    write_parameter_file(parameter, cloudy_exe=cloudy_exe, raw_dir=raw_dir,
                         depth_points=depth_points)
    _validate_parameter_file(parameter, axes, PC_IN_CM)
    # Preserve the existing macOS sleep prevention wrapper; worker scheduling
    # remains inside CIAOLoop, not a new Python process pool.
    command = _command(cialoop, workers, parameter.name)
    manifest = dict(
        schema_version=1, axis_order=AXIS_ORDER, status="prepared", created_at=_now(),
        cloudy_version="17.02", pc_in_cm=PC_IN_CM, abundance=abundance_metadata(),
        axes={name: value.tolist() for name, value in axes.items()},
        parameter_file=parameter.name, maps=expected_maps(depth_points),
        expected_map_count=len(expected_maps(depth_points)),
        expected_state_count=len(expected_maps(depth_points)) * len(LOG_T),
        workers=workers, command=command, working_directory=str(output_dir),
        radiation=dict(definition=sed_report.get("definition"),
                       hm12_log_NH_attenuation=list(HM12_LOG_NH),
                       ism_log_NH_attenuation=21.0,
                       extinguish_leak=0.0, external_grackle_hm12_used=False,
                       sed_report=f"{SED_DIRECTORY_NAME}/build_report.json",
                       cmb="CMB redshift 0", cosmic_rays="cosmic rays rate -16.698970"),
        provenance=dict(cloudy_executable=str(cloudy_exe), cloudy_executable_sha256=sha256(cloudy_exe),
                        cialoop=str(cialoop), cialoop_sha256=sha256(cialoop),
                        parameter_sha256=sha256(parameter), sed_files=sed_records,
                        common_commands=common_commands(),
                        source_scripts={str(path): sha256(path) for path in
                                        (Path(__file__).resolve(), ROOT / "scripts/cloudy_model_depth_common.py")}),
        validation=dict(complete_coordinate_and_temperature_coverage=False,
                        cloudy_convergence_independently_verified=False,
                        geometry_and_abundance_per_state_verified=False,
                        interpretation="Preparation only; no Cloudy model has been launched or validated."),
    )
    _write_json(output_dir / MANIFEST_NAME, manifest)
    return manifest


def _verify_prepared(output_dir: Path, manifest: dict) -> None:
    if manifest.get("status") != "prepared":
        raise ValueError("Only a prepared, never-launched build can run; inspect existing process and output evidence before resuming")
    depth_points = len(manifest.get("axes", {}).get("log_L_model_pc", []))
    axes = _axes(depth_points)
    if manifest.get("axis_order") != AXIS_ORDER or manifest.get("axes") != {
        name: value.tolist() for name, value in axes.items()
    } or manifest.get("abundance") != abundance_metadata():
        raise ValueError("Prepared manifest differs from the adopted grid or composition")
    if manifest["working_directory"] != str(output_dir):
        raise ValueError("Prepared build moved; absolute CIAOLoop paths require a new preparation")
    provenance = manifest["provenance"]
    if (not isinstance(manifest["workers"], int) or manifest["workers"] <= 0 or
            manifest["command"] != _command(Path(provenance["cialoop"]), manifest["workers"], "fixed_depth.par") or
            manifest["parameter_file"] != "fixed_depth.par"):
        raise ValueError("Prepared execution command differs from the recorded CIAOLoop inputs")
    if (manifest.get("expected_map_count") != len(expected_maps(depth_points)) or
            manifest.get("expected_state_count") != len(expected_maps(depth_points)) * len(LOG_T)):
        raise ValueError("Prepared map or state count changed")
    for path, expected in (
        (output_dir / manifest["parameter_file"], provenance["parameter_sha256"]),
        (Path(provenance["cloudy_executable"]), provenance["cloudy_executable_sha256"]),
        (Path(provenance["cialoop"]), provenance["cialoop_sha256"]),
        *((output_dir / item["path"], item["sha256"]) for item in provenance["sed_files"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"Prepared input changed: {path}")
    _validate_parameter_file(output_dir / manifest["parameter_file"], axes, PC_IN_CM)
    if manifest["maps"] != expected_maps(depth_points):
        raise ValueError("Prepared map list changed")
    if any((output_dir / "raw_maps").iterdir()):
        raise FileExistsError("Raw output exists in a prepared build; inspect it before restarting")
    for name in ("run.log", "launch.json"):
        if (output_dir / name).exists():
            raise FileExistsError(f"Launch evidence already exists: {output_dir / name}")


def _collect_completed_maps(output_dir: Path, manifest: dict) -> list[dict]:
    """Associate exact grid coordinates with actual headers, never run ordering."""
    axes = {name: np.asarray(manifest["axes"][name]) for name in AXIS_NAMES}
    raw_dir = output_dir / "raw_maps"
    if any(raw_dir.glob("*.mach")):
        raise ValueError("CIAOLoop worker markers remain after the parent exited")
    paths = sorted(raw_dir.glob("*.dat"))
    if {str(path.relative_to(output_dir)) for path in paths} != {
        item["path"] for item in expected_maps(len(axes["log_L_model_pc"]))
    }:
        raise ValueError("Raw map filenames are missing or unexpected")
    records, seen = [], set()
    for path in paths:
        coordinate, rows = _parse_map(path, PC_IN_CM)
        index = tuple(_axis_index(coordinate[name], axes[name], name, COORDINATE_TOLERANCE_DEX)
                      for name in ("log_NH_attenuation", "log_nH", "log_L_model_pc"))
        if index in seen:
            raise ValueError(f"Duplicate map coordinates: {path}")
        seen.add(index)
        indices_t = [_axis_index(value, axes["log_T"], "temperature", T_TOLERANCE_DEX)
                     for value in rows]
        if len(indices_t) != len(axes["log_T"]) or len(set(indices_t)) != len(axes["log_T"]):
            raise ValueError(f"Incomplete temperature attempts: {path}")
        records.append(dict(path=str(path.relative_to(output_dir)), sha256=sha256(path),
                            **{name: float(axes[name][index[i]]) for i, name in
                               enumerate(("log_NH_attenuation", "log_nH", "log_L_model_pc"))},
                            header_coordinates=coordinate,
                            crash_row_count=sum(value is None for value in rows.values())))
    if len(seen) != manifest["expected_map_count"]:
        raise ValueError("Map grid is not the complete expected Cartesian product")
    return records


def run_build(output_dir: Path) -> dict:
    """Launch once and retain its authoritative process outcome and raw evidence."""
    output_dir = _safe_runtime_path(output_dir)
    manifest_path = output_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    _verify_prepared(output_dir, manifest)
    # Exclusive creation prevents two callers from launching the same maps.
    with (output_dir / "launch.json").open("x") as handle:
        json.dump(dict(requested_at=_now(), command=manifest["command"]), handle)
        handle.write("\n")
    process = None
    try:
        with (output_dir / "run.log").open("x") as log:
            process = subprocess.Popen(manifest["command"], cwd=output_dir,
                                       stdout=log, stderr=subprocess.STDOUT, text=True)
            manifest.update(status="running", started_at=_now(), process_pid=process.pid)
            manifest["validation"]["interpretation"] = "Cloudy process launched; model convergence is not yet independently validated."
            _write_json(manifest_path, manifest)
            returncode = process.wait()
        manifest.update(process_returncode=returncode, process_finished_at=_now())
        if returncode != 0:
            raise RuntimeError(f"CIAOLoop process exited with status {returncode}")
        manifest["maps"] = _collect_completed_maps(output_dir, manifest)
        manifest["validation"].update(
            complete_coordinate_and_temperature_coverage=True,
            interpretation="Every requested map row was attempted. Per-state geometry, abundance, and convergence validation remains required; no packing or adoption was performed.")
        manifest.update(status="completed", completed_at=_now())
        _write_json(manifest_path, manifest)
    except BaseException as exc:
        # An interrupted live child remains running/unknown, not falsely failed.
        returncode = process.poll() if process is not None else None
        if process is not None and returncode is None:
            manifest.update(status="running", observation_interrupted_at=_now(),
                            observation_error=f"{type(exc).__name__}: {exc}")
        else:
            manifest.update(status="failed", failed_at=_now(), process_returncode=returncode,
                            error=f"{type(exc).__name__}: {exc}")
        _write_json(manifest_path, manifest)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cloudy-exe", type=Path, required=True)
    parser.add_argument("--sed-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--depth-points", type=int, choices=(10, 19, 37), default=10)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    output_dir = _safe_runtime_path(args.output_dir)
    if args.prepare_only or not output_dir.exists():
        manifest = prepare_build(output_dir=output_dir, cloudy_exe=args.cloudy_exe,
                                 sed_dir=args.sed_dir, workers=args.workers,
                                 depth_points=args.depth_points)
    else:
        manifest = json.loads((output_dir / MANIFEST_NAME).read_text())
        if (manifest["workers"] != args.workers or
                manifest["axes"]["log_L_model_pc"] != _axes(args.depth_points)["log_L_model_pc"].tolist() or
                Path(manifest["provenance"]["cloudy_executable"]) != args.cloudy_exe.expanduser().resolve() or
                {Path(item["source"]).parent for item in manifest["provenance"]["sed_files"]} != {args.sed_dir.expanduser().resolve()}):
            parser.error("arguments differ from the existing preparation; inspect it instead of changing it in place")
    if args.run:
        manifest = run_build(output_dir)
    print(json.dumps(dict(status=manifest["status"], manifest=str(output_dir / MANIFEST_NAME),
                          expected_maps=manifest["expected_map_count"],
                          expected_states=manifest["expected_state_count"],
                          model_convergence_validated=False), indent=2))


if __name__ == "__main__":
    main()
