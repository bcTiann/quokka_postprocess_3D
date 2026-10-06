#!/usr/bin/env python3
"""Build the seven-radiation-field, eight-line Cloudy Jeans lookup table.

The user-facing inputs are the Cloudy 17.02 executable and worker count. This
orchestrator builds the seven incident SEDs, renders the CIAOLoop parameter
file, runs a seven-point smoke test, runs the 7 x 10 x 21 production grid, and
packages the raw maps without filling failed Cloudy nodes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from quokka2s.cloudy.table_definition import (
    CMB_REDSHIFT,
    COSMIC_RAY_H0_IONIZATION_RATE_S,
    HM12_LOG_NH,
    JEANS_CAP_CLOUDY,
    LINES as LINE_SPEC,
    LOG_NH_DENSITY,
    N_T,
    SED_DIRECTORY_NAME,
    STEM,
    T_MAX_CLOUDY,
    T_MIN_CLOUDY,
)
from quokka2s.physics.composition import QUOKKA_MASS_FRACTIONS


# CIAOLoop needs only the ordered parameter labels, while the packager uses
# each complete (key, parameter label, map header) record from LINE_SPEC.
CLOUDY_LINE_LABELS = tuple(line[1] for line in LINE_SPEC)


def _require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def _ensure_no_whitespace(path: Path, description: str) -> None:
    if any(character.isspace() for character in str(path)):
        raise ValueError(
            f"{description} cannot contain whitespace because CIAOLoop does "
            f"not shell-quote it: {path}"
        )


def _write_parameter_file(
    destination: Path,
    *,
    cloudy_exe: Path,
    output_dir: Path,
    smoke: bool,
) -> None:
    title = "seven-point smoke" if smoke else "7x10x21 production"
    cosmic_ray_log_rate = np.log10(COSMIC_RAY_H0_IONIZATION_RATE_S)
    lines = [
        "#########################################################################",
        f"## Eight-line Jeans {title}: HM2012 attenuation grid + ISM + CMB + CR",
        "#########################################################################",
        f"cloudyExe = {cloudy_exe}",
        f"saveCloudyOutputFiles = {1 if smoke else 0}",
        f"exitOnCrash = {1 if smoke else 0}",
        f"outputFilePrefix = {STEM}_{'smoke' if smoke else '7x10x21'}",
        f"outputDir = {output_dir}",
        "runStartIndex = 1",
        "test = 0",
        "cloudyRunMode = 4",
        *(f"lineMapLine = {line}" for line in CLOUDY_LINE_LABELS),
        f"coolingMapTmin = {'1e5' if smoke else T_MIN_CLOUDY}",
        f"coolingMapTmax = {'1e5' if smoke else T_MAX_CLOUDY}",
        f"coolingMapTpoints = {1 if smoke else N_T}",
        "coolingScaleFactor = 1",
        "coolingMapUseJeansLength = 1",
        f"coolingMapHydrogenMassFraction = {QUOKKA_MASS_FRACTIONS['X']!r}",
        f"coolingMapMaximumJeansLength = {JEANS_CAP_CLOUDY}",
        "command iterate to convergence",
        "command stop temperature off",
        f"command cosmic rays rate {cosmic_ray_log_rate:.6f}",
        "# Cloudy's default simple molecular network and charge transfer remain on.",
        "# No grains, turbulence, or custom elemental abundances are added.",
        f"command CMB redshift {CMB_REDSHIFT:g}",
        "loop [hden] "
        + (
            "0"
            if smoke
            else " ".join(f"{x:.15g}" for x in LOG_NH_DENSITY)
        ),
        (
            f'loop [init "{SED_DIRECTORY_NAME}/logNH*.out"] '
            + " ".join(f"{x:g}" for x in HM12_LOG_NH)
        ),
    ]
    destination.write_text("\n".join(lines) + "\n")


def _run_logged(command: list[str], *, cwd: Path, log_path: Path) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _remove_or_refuse(path: Path, *, force: bool) -> None:
    if not path.exists():
        return
    if not force:
        raise FileExistsError(
            f"generated output already exists: {path}\n"
            "Pass --force to replace it."
        )
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _validate_smoke_output(directory: Path) -> None:
    files = sorted(directory.glob("*_run*.dat"))
    if len(files) != len(HM12_LOG_NH):
        raise ValueError(
            f"expected {len(HM12_LOG_NH)} smoke maps, found {len(files)}"
        )
    expected_init = {
        f'# init "{SED_DIRECTORY_NAME}/logNH{value:g}.out"'
        for value in HM12_LOG_NH
    }
    found_init: set[str] = set()
    # CIAOLoop applies a few label-specific formatting rules. Read the first
    # file's exact eight labels and require the same ordered header in all seven.
    canonical_header = None
    line_maxima = np.full(len(CLOUDY_LINE_LABELS), -np.inf)
    for path in files:
        lines = path.read_text().splitlines()
        found_init.update(line for line in lines if line.startswith("# init "))
        header = next((line for line in lines if line.startswith("#Te")), None)
        if header is None:
            raise ValueError(f"missing line header in smoke map: {path}")
        if canonical_header is None:
            canonical_header = header
        elif header != canonical_header:
            raise ValueError(f"inconsistent smoke-map line header: {path}")
        rows = [line for line in lines if line.strip() and not line.startswith("#")]
        if len(rows) != 1 or len(rows[0].split()) != 1 + len(CLOUDY_LINE_LABELS):
            raise ValueError(f"smoke map does not contain one complete row: {path}")
        values = [float(value) for value in rows[0].split()]
        if not all(np.isfinite(values)):
            raise ValueError(f"non-finite smoke result: {path}")
        line_maxima = np.maximum(line_maxima, np.asarray(values[1:]))
    if found_init != expected_init:
        raise ValueError(
            f"smoke attenuation fields differ: found={sorted(found_init)}"
        )
    if np.any(line_maxima[-2:] <= -90.0):
        raise ValueError(
            "C IV smoke lines are absent or zero at every attenuation setup: "
            f"maxima={line_maxima[-2:].tolist()}"
        )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cloudy-exe",
        type=Path,
        default=Path(os.environ["CLOUDY_EXE"]) if "CLOUDY_EXE" in os.environ else None,
        help="Cloudy 17.02 executable (or set CLOUDY_EXE)",
    )
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument(
        "--runtime-dir", type=Path, default=root / "runtime/cloudy_eightline"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "inputs/tables/cloudy"
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="build and verify the SEDs, then run only seven Cloudy points",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.cloudy_exe is None:
        parser.error("--cloudy-exe is required unless CLOUDY_EXE is set")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    cloudy_exe = _require_file(args.cloudy_exe, "Cloudy executable")
    cialoop = _require_file(
        root / "vendor/cloudy_cooling_tools/CIAOLoop_lines", "CIAOLoop_lines"
    )
    runtime_dir = args.runtime_dir.expanduser().resolve()
    runtime_grackle = runtime_dir / "examples/grackle"
    logs = runtime_dir / "logs"
    output_dir = args.output_dir.expanduser().resolve()
    _ensure_no_whitespace(cloudy_exe, "Cloudy executable path")
    _ensure_no_whitespace(runtime_dir, "runtime directory")
    runtime_grackle.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    sed_dir = runtime_grackle / SED_DIRECTORY_NAME
    smoke_output = runtime_grackle / f"{STEM}_smoke_output"
    production_output = runtime_grackle / f"{STEM}_7x10x21_output"
    smoke_parameter = runtime_grackle / f"{STEM}_smoke.par"
    production_parameter = runtime_grackle / f"{STEM}_7x10x21.par"
    table_output = output_dir / "emission.npz"
    failure_output = output_dir / f"cloudy_{STEM}_failure_nodes.json"

    for generated in (sed_dir, smoke_output):
        _remove_or_refuse(generated, force=args.force)
    if not args.smoke_only:
        for generated in (production_output, table_output, failure_output):
            _remove_or_refuse(generated, force=args.force)

    subprocess.run(
        [
            sys.executable,
            str(root / "tools/cloudy/generate_incident_radiation_fields.py"),
            "--cloudy-exe",
            str(cloudy_exe),
            "--output-dir",
            str(sed_dir),
        ],
        cwd=root,
        check=True,
    )
    sed_report = json.loads((sed_dir / "build_report.json").read_text())
    if tuple(sed_report["hm12_log_NH_attenuation"]) != HM12_LOG_NH:
        raise ValueError("SED builder returned the wrong HM2012 attenuation grid")

    _write_parameter_file(
        smoke_parameter,
        cloudy_exe=cloudy_exe,
        output_dir=smoke_output,
        smoke=True,
    )
    base_command = ["perl", str(cialoop)]
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        base_command = ["caffeinate", "-dimsu", *base_command]
    _run_logged(
        [*base_command, "-np", str(min(args.workers, len(HM12_LOG_NH))), smoke_parameter.name],
        cwd=runtime_grackle,
        log_path=logs / f"{STEM}_smoke.log",
    )
    _validate_smoke_output(smoke_output)
    if args.smoke_only:
        print(f"\nSeven-point smoke test completed: {smoke_output}")
        return

    _write_parameter_file(
        production_parameter,
        cloudy_exe=cloudy_exe,
        output_dir=production_output,
        smoke=False,
    )
    _run_logged(
        [*base_command, "-np", str(args.workers), production_parameter.name],
        cwd=runtime_grackle,
        log_path=logs / f"{STEM}_7x10x21.log",
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "tools/cloudy/package_emission_table.py"),
            "--stem",
            STEM,
            "--runtime-grackle-dir",
            str(runtime_grackle),
            "--output-dir",
            str(output_dir),
            "--parameter-file",
            str(production_parameter),
            "--hm12-log-nh",
            *(f"{value:g}" for value in HM12_LOG_NH),
        ],
        cwd=root,
        check=True,
    )
    for path in (table_output, failure_output):
        if not path.is_file():
            raise FileNotFoundError(path)
    print("\nCloudy table build completed:")
    print(f"  {table_output}")
    print(f"  {failure_output}")


if __name__ == "__main__":
    main()
