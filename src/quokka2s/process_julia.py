"""Launch the pinned Julia processor from this editable uv project.

YAML selects a snapshot and runtime settings. This module validates configuration,
prepares only the child environment, and launches Julia with argument vectors.
Numerical processing remains in julia/src/; plot reads saved products only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import zipfile

import yaml

from quokka2s.input_paths import (
    DEFAULT_CLOUDY_TABLE,
    DEFAULT_DESPOTIC_TABLE,
    DEFAULT_DRAINE_TABLE,
    REPOSITORY_ROOT,
)


JULIA_VERSION = "1.13.2"
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/emission_julia.yaml"
RUNTIME_ROOT = REPOSITORY_ROOT / "runtime/julia"
DEFAULTS = {
    "assets": str(RUNTIME_ROOT / "assets"),
    "julia_executable": "julia",
    "julia_depot": str(RUNTIME_ROOT / "depot"),
    "threads": 1,
    "batch_cells": 131072,
    "reader_cache_mib": 1024,
    "x_index_range": None,
    "y_index_range": None,
    "figure_formats": ["png"],
    "figure_styles": ["titled"],
    "image_downsample_factor": 1,
    "no_plot": False,
    "raw_luminosity": False,
    "save_cells": False,
}
PATH_KEYS = ("dataset", "output", "assets", "julia_depot")
INTEGER_KEYS = (
    "threads",
    "batch_cells",
    "reader_cache_mib",
    "image_downsample_factor",
)
BOOLEAN_KEYS = ("no_plot", "raw_luminosity", "save_cells")
ALLOWED_KEYS = set(DEFAULTS) | {"dataset", "jobname", "output"}


class ConfigurationLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys rather than silently replacing a setting."""


def unique_mapping(loader, node):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            raise ValueError("YAML setting names must be strings")
        if key in mapping:
            raise ValueError(f"Duplicate YAML setting: {key}")
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


ConfigurationLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    unique_mapping,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--setup", action="store_true", help="Install pinned Julia packages and export or verify fixed assets.")
    mode.add_argument("--dry-run", action="store_true", help="Print resolved settings and command without opening inputs or Julia.")
    parser.add_argument("--dataset", help="Absolute snapshot directory; requires explicit --jobname.")
    parser.add_argument("--jobname")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--julia-executable")
    parser.add_argument("--julia-depot", type=Path)
    for name in INTEGER_KEYS:
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for axis in ("x", "y"):
        selection = parser.add_mutually_exclusive_group()
        selection.add_argument("--" + axis, nargs=2, type=int, metavar=("START", "STOP"))
        selection.add_argument("--all-" + axis, action="store_true", help=f"Select the full {axis} axis, overriding YAML.")
    parser.add_argument("--format", nargs="+", choices=("png", "pdf"), dest="figure_formats")
    parser.add_argument("--style", nargs="+", choices=("titled", "untitled"), dest="figure_styles")
    plotting = parser.add_mutually_exclusive_group()
    plotting.add_argument("--no-plot", dest="no_plot", action="store_true")
    plotting.add_argument("--plot", dest="no_plot", action="store_false")
    spectra = parser.add_mutually_exclusive_group()
    spectra.add_argument("--raw-luminosity", dest="raw_luminosity", action="store_true")
    spectra.add_argument("--per-area", dest="raw_luminosity", action="store_false")
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument("--save-cells", dest="save_cells", action="store_true")
    diagnostics.add_argument("--no-save-cells", dest="save_cells", action="store_false")
    parser.set_defaults(**{name: None for name in BOOLEAN_KEYS})
    return parser


def absolute_path(value, base):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    # Keep resolution lexical: --dry-run must not inspect snapshot/asset paths.
    return Path(os.path.abspath(path))


def executable_setting(value, base):
    expanded = os.path.expanduser(value)
    if "/" in expanded or os.sep in expanded:
        return str(absolute_path(expanded, base))
    return expanded


def validate_settings(settings, *, setup=False):
    for name in ALLOWED_KEYS & settings.keys():
        value = settings[name]
        if name in PATH_KEYS or name in ("jobname", "julia_executable"):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        elif name in INTEGER_KEYS:
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        elif name in BOOLEAN_KEYS:
            if type(value) is not bool:
                raise ValueError(f"{name} must be a YAML boolean (true or false)")
        elif name in ("x_index_range", "y_index_range"):
            if value is not None:
                if not isinstance(value, list) or len(value) != 2:
                    raise ValueError(f"{name} must be null or [START, STOP]")
                if any(type(bound) is not int for bound in value):
                    raise ValueError(f"{name} bounds must be integers")
                if not 0 <= value[0] < value[1]:
                    raise ValueError(f"{name} requires 0 <= START < STOP")
        elif name in ("figure_formats", "figure_styles"):
            allowed = ("png", "pdf") if name == "figure_formats" else ("titled", "untitled")
            if not isinstance(value, list) or not value:
                raise ValueError(f"{name} must be a nonempty list")
            if any(not isinstance(item, str) or item not in allowed for item in value):
                raise ValueError(f"{name} supports only {allowed}")
            if len(set(value)) != len(value):
                raise ValueError(f"{name} must not contain duplicates")
    if "jobname" in settings:
        name = settings["jobname"]
        if name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError("jobname must be one explicit simulation label, without path separators")
    if not setup:
        for name in ("dataset", "jobname", "output"):
            if name not in settings:
                raise ValueError(f"Missing required setting: {name}")


def load_settings(arguments):
    """Resolve defaults < YAML < CLI, preserving independent native x/y ranges."""
    config_path = absolute_path(arguments.config, Path.cwd())
    config_bytes = config_path.read_bytes()
    loaded = yaml.load(config_bytes, Loader=ConfigurationLoader)
    if not isinstance(loaded, dict):
        raise ValueError("Configuration must be a YAML mapping")
    unknown = sorted(set(loaded) - ALLOWED_KEYS)
    if unknown:
        raise ValueError("Unknown YAML settings: " + ", ".join(unknown))
    settings = dict(DEFAULTS)
    settings.update(loaded)
    validate_settings(settings, setup=True)
    for name in PATH_KEYS:
        if name in settings:
            settings[name] = str(absolute_path(settings[name], config_path.parent))
    settings["julia_executable"] = executable_setting(settings["julia_executable"], config_path.parent)
    for name in ALLOWED_KEYS:
        if name in ("x_index_range", "y_index_range"):
            continue
        value = getattr(arguments, name, None)
        if value is None:
            continue
        if name in PATH_KEYS:
            value = str(absolute_path(value, Path.cwd()))
        elif name == "julia_executable":
            value = executable_setting(value, Path.cwd())
        settings[name] = value
    for axis in ("x", "y"):
        bounds = getattr(arguments, axis)
        if bounds is not None:
            settings[axis + "_index_range"] = bounds
        elif getattr(arguments, "all_" + axis):
            settings[axis + "_index_range"] = None
    validate_settings(settings, setup=arguments.setup)
    return settings, config_path, hashlib.sha256(config_bytes).hexdigest()


def child_environment(settings):
    """Keep cache paths and native thread limits local to the launched process."""
    environment = os.environ.copy()
    environment["JULIA_DEPOT_PATH"] = settings["julia_depot"]
    environment["JULIA_NUM_PRECOMPILE_TASKS"] = str(min(settings["threads"], 4))
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["MPLCONFIGDIR"] = str(RUNTIME_ROOT / "matplotlib")
    return environment


def launch_record(settings, config_path, config_hash, environment):
    return {
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "effective_settings": settings,
        "python_executable": sys.executable,
        "julia_version_required": JULIA_VERSION,
        "project": str(REPOSITORY_ROOT / "julia"),
        "environment": {
            name: environment[name]
            for name in (
                "JULIA_DEPOT_PATH",
                "JULIA_NUM_PRECOMPILE_TASKS",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MPLCONFIGDIR",
            )
        },
    }


def julia_command(settings, record):
    command = [
        settings["julia_executable"],
        "--startup-file=no",
        "--project=" + str(REPOSITORY_ROOT / "julia"),
        "--threads=" + str(settings["threads"]),
        str(REPOSITORY_ROOT / "julia/bin/process.jl"),
    ]
    options = {
        "--dataset": settings["dataset"],
        "--jobname": settings["jobname"],
        "--assets": settings["assets"],
        "--output": settings["output"],
        "--threads": str(settings["threads"]),
        "--batch-cells": str(settings["batch_cells"]),
        "--reader-cache-mib": str(settings["reader_cache_mib"]),
        "--image-downsample-factor": str(settings["image_downsample_factor"]),
        "--python": sys.executable,
        "--launch-record": json.dumps(record, separators=(",", ":"), allow_nan=False),
    }
    for option, value in options.items():
        command.extend((option, value))
    for axis in ("x", "y"):
        bounds = settings[axis + "_index_range"]
        if bounds is not None:
            command.extend(("--" + axis, str(bounds[0]), str(bounds[1])))
    command.extend(("--format", *settings["figure_formats"]))
    command.extend(("--style", *settings["figure_styles"]))
    for name in BOOLEAN_KEYS:
        if settings[name]:
            command.append("--" + name.replace("_", "-"))
    return command


def run_child(command, environment, *, capture=False):
    """Run without a shell; let the foreground process group receive Ctrl-C."""
    process = subprocess.Popen(
        command,
        env=environment,
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
    )
    previous_termination_handler = signal.getsignal(signal.SIGTERM)

    def forward_termination(signum, frame):
        if process.poll() is None:
            process.send_signal(signum)

    signal.signal(signal.SIGTERM, forward_termination)
    try:
        try:
            stdout, stderr = process.communicate()
        except KeyboardInterrupt:
            # Terminal Ctrl-C reaches Julia too. Forward for non-terminal callers.
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            process.wait()
            raise
    finally:
        signal.signal(signal.SIGTERM, previous_termination_handler)
    if process.returncode:
        if capture:
            detail = (stderr or stdout or "").strip()
            if detail:
                print(detail, file=sys.stderr)
        raise subprocess.CalledProcessError(process.returncode, command)
    return stdout or ""


def verify_julia(settings, environment):
    executable = shutil.which(settings["julia_executable"])
    if executable is None:
        raise ValueError(
            "Julia executable not found. Install Julia " + JULIA_VERSION
            + " and set julia_executable in YAML or --julia-executable."
        )
    settings["julia_executable"] = str(Path(executable).absolute())
    version = run_child(
        [executable, "--startup-file=no", "--version"],
        environment,
        capture=True,
    ).strip()
    if version != "julia version " + JULIA_VERSION:
        raise ValueError(f"Expected Julia {JULIA_VERSION}; found {version!r}")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(directory):
    """Verify adopted source provenance and exported NPZ bytes, preserving files."""
    if not directory.exists():
        raise ValueError(f"Julia assets are missing at {directory}; run uv run process-julia --setup first.")
    physics_path = directory / "physics.json"
    numerical_path = directory / "tables.npz"
    if not physics_path.is_file() or not numerical_path.is_file():
        raise ValueError(f"Incomplete Julia assets at {directory}; preserve them and choose a new assets directory.")
    metadata = json.loads(physics_path.read_text())
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported Julia asset schema")
    if metadata.get("numerical_assets_sha256") != file_hash(numerical_path):
        raise ValueError("Julia tables.npz checksum differs from physics.json")
    sources = metadata.get("sources", {})
    if not isinstance(sources, dict):
        raise ValueError("Julia asset sources must be a JSON mapping")
    expected_sources = {
        "despotic": DEFAULT_DESPOTIC_TABLE,
        "cloudy": DEFAULT_CLOUDY_TABLE,
        "dust": DEFAULT_DRAINE_TABLE,
    }
    for name, path in expected_sources.items():
        source = sources.get(name)
        if not isinstance(source, dict) or source.get("sha256") != file_hash(path):
            raise ValueError(f"Julia assets conflict with the current {name} input; choose a new assets directory and run --setup.")
    with zipfile.ZipFile(numerical_path) as archive:
        damaged = archive.testzip()
    if damaged is not None:
        raise ValueError(f"Damaged Julia asset member: {damaged}")


def protect_input_paths(settings, environment):
    """Refuse writable runtime/output paths inside the selected snapshot."""
    if "dataset" not in settings:
        return
    snapshot = Path(settings["dataset"]).resolve()
    writable = {
        name: Path(settings[name]).resolve()
        for name in ("assets", "julia_depot", "output")
        if name in settings
    }
    writable["matplotlib cache"] = Path(environment["MPLCONFIGDIR"]).resolve()
    writable["uv cache"] = (REPOSITORY_ROOT / "runtime/uv-cache").resolve()
    for name, path in writable.items():
        if path.is_relative_to(snapshot):
            raise ValueError(f"{name} must be outside the snapshot directory: {path}")


def setup_runtime(settings, environment):
    directory = Path(settings["assets"])
    if directory.exists():
        verify_assets(directory)
    verify_julia(settings, environment)
    print("Installing the pinned Julia project (one-time setup).", flush=True)
    run_child(
        [
            settings["julia_executable"],
            "--startup-file=no",
            "--project=" + str(REPOSITORY_ROOT / "julia"),
            "--threads=" + str(settings["threads"]),
            "-e",
            "using Pkg; Pkg.instantiate(); Pkg.precompile()",
        ],
        environment,
    )
    if not directory.exists():
        print("Exporting the adopted fixed tables.", flush=True)
        run_child(
            [sys.executable, str(REPOSITORY_ROOT / "julia/export_tables.py"), "--output", str(directory)],
            environment,
        )
    verify_assets(directory)
    print(f"Julia setup complete. Assets: {directory}", flush=True)


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.dataset is not None:
        if arguments.jobname is None:
            parser.error("--dataset requires explicit --jobname")
        if not Path(arguments.dataset).expanduser().is_absolute():
            parser.error("--dataset must be absolute after expanding ~")
    try:
        settings, config_path, config_hash = load_settings(arguments)
        environment = child_environment(settings)
        if arguments.setup:
            protect_input_paths(settings, environment)
            setup_runtime(settings, environment)
            return 0
        record = launch_record(settings, config_path, config_hash, environment)
        if arguments.dry_run:
            print(json.dumps({"launch_record": record, "command": julia_command(settings, record)}, indent=2, allow_nan=False))
            return 0
        protect_input_paths(settings, environment)
        verify_assets(Path(settings["assets"]))
        verify_julia(settings, environment)
        record = launch_record(settings, config_path, config_hash, environment)
        print("Processing snapshot: " + settings["dataset"], flush=True)
        print("Output: " + settings["output"], flush=True)
        run_child(julia_command(settings, record), environment)
        return 0
    except (ValueError, OSError, yaml.YAMLError, zipfile.BadZipFile) as error:
        parser.exit(2, f"process-julia: {error}\n")
    except subprocess.CalledProcessError as error:
        code = error.returncode
        return code if code > 0 else 128 - code
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
