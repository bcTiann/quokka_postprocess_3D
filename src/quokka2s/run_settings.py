"""Read the small YAML inputs for the emission ``process`` and ``plot`` commands.

Only the paths and optional execution settings listed below are accepted. A
relative path in the YAML file is relative to that file, not to the shell's
current working directory. A leading ~ expands to the user's home directory;
environment variables such as $HOME are not expanded.
"""
from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from pathlib import Path

import yaml

from quokka2s.input_paths import DEFAULT_DRAINE_TABLE


# Relative names only: the entry points use the working directory; standalone
# Figure tools explicitly anchor these same names to their repository root.
DEFAULT_PROCESS_CONFIG = Path("configs/emission_process.yaml")
DEFAULT_PLOT_CONFIG = Path("configs/emission_plot.yaml")


@dataclass(frozen=True)
class ProcessSettings:
    """Resolved inputs and execution settings for one snapshot process.

    Paths are absolute. Slab/query sizes count cells; workers count threads.
    Each concurrent batch can use spectral_workers integration threads:
    chunk_workers=2 and spectral_workers=3 allow at most six spectral tasks.
    Index ranges are [start, stop), or None for the full native axis.
    """

    dataset: Path
    despotic_table: Path
    cloudy_table: Path
    output_dir: Path
    dust_opacity_table: Path = DEFAULT_DRAINE_TABLE
    slab_nx: int = 8
    query_chunk: int = 100000
    chunk_workers: int = 1
    spectral_workers: int = 6
    max_slabs: int | None = None
    x_index_range: tuple[int, int] | None = None
    y_index_range: tuple[int, int] | None = None


@dataclass(frozen=True)
class PlotSettings:
    """Saved-result paths and display choices; no snapshot or table inputs.

    image_downsample_factor=2 selects the saved 2x2-summed image product.
    A missing titled_output_dir omits the separate standalone figure version.
    """

    products: Path
    output_dir: Path
    titled_output_dir: Path | None = None
    raw_luminosity: bool = False
    allow_partial: bool = False
    image_downsample_factor: int = 1


_PROCESS_REQUIRED_PATHS = (
    "dataset", "despotic_table", "cloudy_table", "output_dir",
)
_PROCESS_OPTIONAL_PATHS = ("dust_opacity_table",)
_PROCESS_INTEGERS = ("slab_nx", "query_chunk", "chunk_workers", "spectral_workers", "max_slabs")
_PROCESS_INDEX_RANGES = ("x_index_range", "y_index_range")
_PROCESS_DEFAULTS = {
    field.name: field.default
    for field in fields(ProcessSettings)
    if field.default is not MISSING
}
# YAML path conversion accepts strings; retain the same bundled default path.
_PROCESS_DEFAULTS["dust_opacity_table"] = str(DEFAULT_DRAINE_TABLE)
_PLOT_REQUIRED_PATHS = ("products", "output_dir")
_PLOT_OPTIONAL_PATHS = ("titled_output_dir",)
_PLOT_DEFAULTS = {
    field.name: field.default
    for field in fields(PlotSettings)
    if field.default is not MISSING
}


def _read_mapping(path: str | Path, required: set[str], allowed: set[str]) -> tuple[Path, dict]:
    """Read a YAML mapping and reject missing or unknown setting names.

    Return its absolute path and values; the path anchors relative input paths."""
    config_path = Path(path).expanduser().resolve()
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
    """Expand ~, then anchor a relative path to the YAML directory.

    Absolute paths keep their own location. None is accepted only for nullable
    settings; environment variables are not expanded.
    Example: ~/snapshots/plt0655228 uses the user's home, not configs/."""
    if value is None and nullable:
        return None
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = directory / path
    return path.resolve()


def _positive_int(name: str, value: object, *, nullable: bool = False) -> int | None:
    """Accept a positive integer without coercion; optionally permit None."""
    if value is None and nullable:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _index_range(name: str, value: object) -> tuple[int, int] | None:
    """Read [start, stop] native cell indices, with stop excluded.

    None selects the full axis. Snapshot checks the upper bound after loading
    its geometry. Example: [64, 128] selects indices 64 through 127."""
    if value is None:
        return None
    if (not isinstance(value, list) or len(value) != 2
            or any(type(index) is not int for index in value)):
        raise ValueError(f"{name} must contain two integer cell indices [start, stop]")
    start, stop = value
    if start < 0 or stop <= start:
        raise ValueError(f"{name} must satisfy 0 <= start < stop")
    return start, stop


def load_process_config(path: str | Path) -> ProcessSettings:
    """Read snapshot/table/output Paths and execution settings from process YAML.

    Paths are relative to the YAML directory. Slab and query sizes count cells;
    worker settings count threads. x/y_index_range selects native cell indices;
    omitted axes use the full extent. The z axis always remains complete."""
    required = set(_PROCESS_REQUIRED_PATHS)
    allowed = (
        required | set(_PROCESS_OPTIONAL_PATHS) | set(_PROCESS_INTEGERS)
        | set(_PROCESS_INDEX_RANGES)
    )
    config_path, supplied = _read_mapping(path, required, allowed)
    values = {**_PROCESS_DEFAULTS, **supplied}
    for name in _PROCESS_REQUIRED_PATHS:
        values[name] = _path_value(name, values[name], config_path.parent)
    for name in _PROCESS_OPTIONAL_PATHS:
        values[name] = _path_value(name, values[name], config_path.parent)
    for name in _PROCESS_INTEGERS:
        values[name] = _positive_int(name, values[name], nullable=name == "max_slabs")
    for name in _PROCESS_INDEX_RANGES:
        values[name] = _index_range(name, values[name])
    if values["query_chunk"] > 1000000:
        raise ValueError("query_chunk must be at most 1000000 cells")
    return ProcessSettings(**values)


def load_plot_config(path: str | Path) -> PlotSettings:
    """Read saved-product Paths and display settings from plot YAML.

    Paths are relative to the YAML directory. image_downsample_factor counts
    native pixels per image axis; raw_luminosity and allow_partial are Boolean."""
    required = set(_PLOT_REQUIRED_PATHS)
    config_path, supplied = _read_mapping(
        path, required, required | set(_PLOT_OPTIONAL_PATHS) | set(_PLOT_DEFAULTS))
    values = {**_PLOT_DEFAULTS, **supplied}
    for name in _PLOT_REQUIRED_PATHS:
        values[name] = _path_value(name, values[name], config_path.parent)
    for name in _PLOT_OPTIONAL_PATHS:
        values[name] = _path_value(name, values.get(name), config_path.parent, nullable=True)
    for name in ("raw_luminosity", "allow_partial"):
        if type(values[name]) is not bool:
            raise ValueError(f"{name} must be true or false")
    values["image_downsample_factor"] = _positive_int(
        "image_downsample_factor", values["image_downsample_factor"])
    return PlotSettings(**values)
