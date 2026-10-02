"""Read the small YAML inputs for the emission ``process`` and ``plot`` commands.

Only the paths and optional execution settings listed below are accepted. A
relative path in the YAML file is relative to that file, not to the shell's
current working directory. Environment variables are never expanded.
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import yaml

from .dust_attenuation import DEFAULT_DRAINE_TABLE


_PROCESS_REQUIRED_PATHS = (
    "dataset", "despotic_table", "cloudy_table", "output_dir",
)
_PROCESS_OPTIONAL_PATHS = ("dust_opacity_table",)
_PROCESS_INTEGERS = ("slab_nx", "query_chunk", "spectral_workers", "max_slabs")
_PROCESS_DEFAULTS = {
    "dust_opacity_table": str(DEFAULT_DRAINE_TABLE),
    "slab_nx": 8,
    "query_chunk": 100000,
    "spectral_workers": 6,
    "max_slabs": None,
}
_PLOT_REQUIRED_PATHS = ("products", "output_dir")
_PLOT_DEFAULTS = {"raw_luminosity": False, "allow_partial": False}


def _read_mapping(path: str | Path, required: set[str], allowed: set[str]) -> tuple[Path, dict]:
    config_path = Path(path).resolve()
    try:
        with config_path.open(encoding="utf-8") as source:
            values = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(values, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    if any(type(key) is not str for key in values):
        raise ValueError(f"{config_path} must use string setting names")
    unknown = set(values) - allowed
    missing = required - set(values)
    if unknown:
        raise ValueError(f"Unknown setting(s) in {config_path}: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"Missing setting(s) in {config_path}: {', '.join(sorted(missing))}")
    return config_path, values


def _path_value(name: str, value: object, directory: Path, *, nullable: bool = False) -> Path | None:
    if value is None and nullable:
        return None
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty path string")
    path = Path(value)
    return (directory / path).resolve()


def _positive_int(name: str, value: object, *, nullable: bool = False) -> int | None:
    if value is None and nullable:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_process_config(path: str | Path) -> Namespace:
    """Return the process command's validated settings as an argparse Namespace."""
    required = set(_PROCESS_REQUIRED_PATHS)
    allowed = required | set(_PROCESS_OPTIONAL_PATHS) | set(_PROCESS_INTEGERS)
    config_path, supplied = _read_mapping(path, required, allowed)
    values = {**_PROCESS_DEFAULTS, **supplied}
    for name in _PROCESS_REQUIRED_PATHS:
        values[name] = _path_value(name, values[name], config_path.parent)
    for name in _PROCESS_OPTIONAL_PATHS:
        values[name] = _path_value(name, values[name], config_path.parent)
    for name in _PROCESS_INTEGERS:
        values[name] = _positive_int(name, values[name], nullable=name == "max_slabs")
    if values["query_chunk"] > 100000:
        raise ValueError("query_chunk must be at most 100000 cells")
    return Namespace(**values)


def load_plot_config(path: str | Path) -> Namespace:
    """Return the plotting command's validated settings as an argparse Namespace."""
    required = set(_PLOT_REQUIRED_PATHS)
    config_path, supplied = _read_mapping(path, required, required | set(_PLOT_DEFAULTS))
    values = {**_PLOT_DEFAULTS, **supplied}
    for name in _PLOT_REQUIRED_PATHS:
        values[name] = _path_value(name, values[name], config_path.parent)
    for name in _PLOT_DEFAULTS:
        if type(values[name]) is not bool:
            raise ValueError(f"{name} must be true or false")
    return Namespace(**values)
