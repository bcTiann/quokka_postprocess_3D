#!/usr/bin/env python3
"""Build the six-line Cloudy lookup tables.

The only machine-specific input is a Cloudy 17.02 executable.  All generated
SEDs, rendered CIAOLoop parameter files, raw maps, and logs are written under
``runtime/cloudy_sixline`` by default.  Final compact tables are written under
``data``; both directories are ignored by Git.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


STEM = "hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline"
GEOMETRIES = ("column_10x10x21", "jeans_10x21")


def _require_path(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def _ensure_no_whitespace(path: Path, description: str) -> None:
    if any(character.isspace() for character in str(path)):
        raise ValueError(
            f"{description} cannot contain whitespace because CIAOLoop passes "
            f"it to the shell without quoting: {path}"
        )


def _render_parameter_file(
    template: Path,
    destination: Path,
    *,
    cloudy_exe: Path,
    output_dir: Path,
) -> None:
    text = template.read_text()
    replacements = {
        "@CLOUDY_EXE@": str(cloudy_exe),
        "@OUTPUT_DIR@": str(output_dir),
    }
    for token, value in replacements.items():
        if token not in text:
            raise ValueError(f"missing template token {token}: {template}")
        text = text.replace(token, value)
    if "@CLOUDY_EXE@" in text or "@OUTPUT_DIR@" in text:
        raise ValueError(f"unresolved template token: {template}")
    destination.write_text(text)


def _run_logged(command: list[str], *, cwd: Path, log_path: Path) -> None:
    printable = " ".join(command)
    print(f"\n$ {printable}", flush=True)
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
            f"runtime output already exists: {path}\n"
            "Pass --force to replace this generated runtime directory."
        )
    shutil.rmtree(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cloudy-exe",
        type=Path,
        default=Path(os.environ["CLOUDY_EXE"]) if "CLOUDY_EXE" in os.environ else None,
        help="Cloudy 17.02 executable (or set CLOUDY_EXE)",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--runtime-dir", type=Path, default=root / "runtime/cloudy_sixline"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "data")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="build the incident SED and run only the one-point smoke test",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace generated runtime outputs for the requested run",
    )
    args = parser.parse_args()
    if args.cloudy_exe is None:
        parser.error("--cloudy-exe is required unless CLOUDY_EXE is set")
    if args.workers <= 0:
        parser.error("--workers must be a positive integer")

    cloudy_exe = _require_path(args.cloudy_exe, "Cloudy executable")
    cialoop = _require_path(
        root / "vendor/cloudy_cooling_tools/CIAOLoop_lines", "CIAOLoop_lines"
    )
    template_dir = root / "vendor/cloudy_cooling_tools/examples/grackle"
    runtime_dir = args.runtime_dir.expanduser().resolve()
    runtime_grackle = runtime_dir / "examples/grackle"
    output_dir = args.output_dir.expanduser().resolve()
    _ensure_no_whitespace(cloudy_exe, "Cloudy executable path")
    _ensure_no_whitespace(runtime_dir, "runtime directory")

    runtime_grackle.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs = runtime_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    products = [
        output_dir / f"cloudy_{STEM}_{suffix}.npz" for suffix in GEOMETRIES
    ]
    products.append(output_dir / f"cloudy_{STEM}_failure_nodes.json")
    if not args.smoke_only:
        existing_products = [path for path in products if path.exists()]
        if existing_products and not args.force:
            formatted = "\n".join(f"  {path}" for path in existing_products)
            raise FileExistsError(
                "final table products already exist:\n"
                f"{formatted}\nPass --force to replace generated tables."
            )

    sed_dir = runtime_grackle / "HM12_NATIVE_ISM_NH21"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/build_hm12_filtered_ism_sed.py"),
            "--cloudy-exe",
            str(cloudy_exe),
            "--output-dir",
            str(sed_dir),
        ],
        cwd=root,
        check=True,
    )

    names = ["smoke"] if args.smoke_only else ["smoke", *GEOMETRIES]
    rendered: dict[str, Path] = {}
    for suffix in names:
        template = template_dir / f"{STEM}_{suffix}.par.in"
        _require_path(template, "CIAOLoop parameter template")
        run_output = runtime_grackle / f"{STEM}_{suffix}_output"
        _remove_or_refuse(run_output, force=args.force)
        rendered_path = runtime_grackle / f"{STEM}_{suffix}.par"
        _render_parameter_file(
            template,
            rendered_path,
            cloudy_exe=cloudy_exe,
            output_dir=run_output,
        )
        rendered[suffix] = rendered_path

    base_command = ["perl", str(cialoop)]
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        base_command = ["caffeinate", "-dimsu", *base_command]

    _run_logged(
        [*base_command, rendered["smoke"].name],
        cwd=runtime_grackle,
        log_path=logs / "smoke.log",
    )
    if args.smoke_only:
        print(f"\nSmoke test completed. Runtime: {runtime_dir}")
        return

    for suffix in GEOMETRIES:
        _run_logged(
            [*base_command, "-np", str(args.workers), rendered[suffix].name],
            cwd=runtime_grackle,
            log_path=logs / f"{suffix}.log",
        )

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/build_hm12_filtered_ism_sixline_bundles.py"),
            "--stem",
            STEM,
            "--runtime-grackle-dir",
            str(runtime_grackle),
            "--output-dir",
            str(output_dir),
            "--charge-transfer-enabled",
            "--cosmic-ray-rate-s",
            "2e-17",
            "--cmb-redshift",
            "0",
            "--molecular-network-enabled",
        ],
        cwd=root,
        check=True,
    )

    missing = [path for path in products if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"expected products were not created: {missing}")
    print("\nCloudy table build completed:")
    for path in products:
        print(f"  {path}")


if __name__ == "__main__":
    main()
