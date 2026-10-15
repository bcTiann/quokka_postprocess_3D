"""Read one YAML file shared by snapshot processing and result plotting.

Only the paths and optional execution settings listed below are accepted. A
relative path in the YAML file is relative to that file, not to the shell's
current working directory. A leading ~ expands to the user's home directory;
environment variables such as $HOME are not expanded.
"""
from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from pathlib import Path

import yaml

from quokka2s.input_paths import (
    DEFAULT_CLOUDY_TABLE,
    DEFAULT_DESPOTIC_TABLE,
    DEFAULT_DRAINE_TABLE,
)
from quokka2s.paths import resolve_path


# Relative names only: the entry points use the working directory; standalone
# Figure tools explicitly anchor these same names to their repository root.
DEFAULT_CONFIG = Path("configs/emission.yaml")


@dataclass(frozen=True)
class RunSelection:
    """One selected job/snapshot, native x/y ranges, and its result directory.

    snapshot_name comes from the supplied dataset folder name, without opening
    the snapshot or following its symlink. Index ranges are parsed [start, stop)
    tuples, or None for a full axis. directory is an absolute Path: the snapshot
    directory for a full box, or its named region subdirectory for explicit ranges.
    Example: x=(64, 66), y=None selects plt0916720/region_x64-66_yall/.
    """

    jobname: str
    snapshot_name: str
    x_index_range: tuple[int, int] | None
    y_index_range: tuple[int, int] | None
    directory: Path


@dataclass(frozen=True)
class ProcessSettings:
    """Resolved inputs and execution settings for one snapshot process.

    Paths are absolute. Slab/query sizes count cells; workers count threads.
    Omitted execution settings use the same defaults as the shipped YAML.
    Each concurrent batch can use spectral_workers integration threads:
    chunk_workers=2 and spectral_workers=3 allow at most six spectral tasks.
    Index ranges are [start, stop), or None for the full native axis.
    """

    dataset: Path
    output_dir: Path
    jobname: str
    snapshot_name: str
    despotic_table: Path = DEFAULT_DESPOTIC_TABLE
    cloudy_table: Path = DEFAULT_CLOUDY_TABLE
    dust_opacity_table: Path = DEFAULT_DRAINE_TABLE
    slab_nx: int = 8
    query_chunk: int = 1_000_000
    chunk_workers: int = 2
    spectral_workers: int = 3
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


_RUN_REQUIRED_SETTINGS = {"dataset", "jobname", "results_dir"}
_PROCESS_TABLE_PATHS = ("despotic_table", "cloudy_table", "dust_opacity_table")
_PROCESS_INTEGERS = ("slab_nx", "query_chunk", "chunk_workers", "spectral_workers", "max_slabs")
_REGION_INDEX_RANGES = ("x_index_range", "y_index_range")
_PROCESS_DEFAULTS = {
    field.name: field.default
    for field in fields(ProcessSettings)
    if field.default is not MISSING and field.name not in _REGION_INDEX_RANGES
}
for name in _PROCESS_TABLE_PATHS:
    # Bundled defaults stay anchored to this repository, even for external YAML.
    _PROCESS_DEFAULTS[name] = str(_PROCESS_DEFAULTS[name])
_PLOT_DEFAULTS = {
    field.name: field.default
    for field in fields(PlotSettings)
    if field.default is not MISSING and field.name != "titled_output_dir"
}
_ALLOWED_SETTINGS = (
    _RUN_REQUIRED_SETTINGS
    | set(_PROCESS_DEFAULTS)
    | set(_REGION_INDEX_RANGES)
    | set(_PLOT_DEFAULTS)
    | {"titled_figures"}
)


def _read_mapping(path: str | Path) -> tuple[Path, dict]:
    """Read a YAML mapping and reject missing or unknown setting names.

    Return its absolute path and values; the path anchors relative input paths."""
    config_path = resolve_path(path)
    try:
        with config_path.open(encoding="utf-8") as source:
            values = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(values, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    if any(type(key) is not str for key in values):
        raise ValueError(f"{config_path} must use string setting names")
    unknown = set(values) - _ALLOWED_SETTINGS
    missing = _RUN_REQUIRED_SETTINGS - set(values)
    if unknown:
        raise ValueError(f"Unknown setting(s) in {config_path}: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"Missing setting(s) in {config_path}: {', '.join(sorted(missing))}")
    return config_path, values


def _path_value(name: str, value: object, directory: Path) -> Path:
    """Expand ~, then anchor a relative path to the YAML directory.

    Absolute paths keep their own location; environment variables are not expanded.
    Example: ~/snapshots/plt0655228 uses the user's home, not configs/."""
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty path string")
    return resolve_path(
        path=value,
        base_directory=directory,
    )


def _folder_name(name: str, value: object) -> str:
    """Accept one directory name, including dots and hyphens used in job names."""
    if (type(value) is not str or not value.strip()
            or value in (".", "..") or "/" in value or "\\" in value):
        raise ValueError(f"{name} must be a nonempty single folder name")
    return value


def read_run_selection(supplied: dict, config_directory: Path) -> RunSelection:
    """Select the job/snapshot and optional region from the shared YAML settings.

    Paths are relative to config_directory. Only the dataset string's last
    folder name is used here: plot can retain a Setonix path on a laptop.
    Both commands reuse these parsed ranges and the same result directory.
    Example: x=[64, 66], y=[64, 66] add region_x64-66_y64-66/ under the
    snapshot directory; omitted or null axes are named 'all'. No cells are read.
    """
    dataset = supplied["dataset"]
    if type(dataset) is not str or not dataset.strip():
        raise ValueError("dataset must be a nonempty path string")
    snapshot_name = _folder_name("dataset folder name", Path(dataset).name)
    jobname = _folder_name("jobname", supplied["jobname"])
    results_dir = _path_value("results_dir", supplied["results_dir"], config_directory)
    x_index_range = _index_range("x_index_range", supplied.get("x_index_range"))
    y_index_range = _index_range("y_index_range", supplied.get("y_index_range"))
    directory = results_dir / jobname / snapshot_name
    region_name = region_directory_name(
        x_index_range=x_index_range,
        y_index_range=y_index_range,
    )
    if region_name is not None:
        directory = directory / region_name
    return RunSelection(
        jobname=jobname,
        snapshot_name=snapshot_name,
        x_index_range=x_index_range,
        y_index_range=y_index_range,
        directory=directory,
    )


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


def region_directory_name(
    x_index_range: tuple[int, int] | None,
    y_index_range: tuple[int, int] | None,
) -> str | None:
    """Name explicit native x/y ranges, or return None for an unselected full box.

    Inputs are the validated tuples from _index_range(); stops are excluded.
    An omitted axis uses 'all', without needing the snapshot's grid dimensions.
    Example: (64, 66), None -> 'region_x64-66_yall'. Both None keeps the
    snapshot's existing full-box directory, without an extra region folder.
    """
    if x_index_range is None and y_index_range is None:
        return None
    x_label = "all"
    if x_index_range is not None:
        x_start, x_stop = x_index_range
        x_label = f"{x_start}-{x_stop}"
    y_label = "all"
    if y_index_range is not None:
        y_start, y_stop = y_index_range
        y_label = f"{y_start}-{y_stop}"
    return f"region_x{x_label}_y{y_label}"


def load_process_config(path: str | Path) -> ProcessSettings:
    """Read processing settings and derive the selected processed/ directory.

    Paths are relative to the YAML directory. Slab and query sizes count cells;
    worker settings count threads. x/y_index_range selects native cell indices;
    omitted axes use the full extent. The z axis always remains complete.
    Plot options in the shared file are accepted but not used by process.
    """
    config_path, supplied = _read_mapping(path)
    run = read_run_selection(
        supplied=supplied,
        config_directory=config_path.parent,
    )
    values = {
        name: supplied.get(name, default)
        for name, default in _PROCESS_DEFAULTS.items()
    }
    for name in _PROCESS_TABLE_PATHS:
        values[name] = _path_value(name, values[name], config_path.parent)
    for name in _PROCESS_INTEGERS:
        values[name] = _positive_int(name, values[name], nullable=name == "max_slabs")
    if values["query_chunk"] > 1000000:
        raise ValueError("query_chunk must be at most 1000000 cells")
    return ProcessSettings(
        dataset=_path_value("dataset", supplied["dataset"], config_path.parent),
        output_dir=run.directory / "processed",
        jobname=run.jobname,
        snapshot_name=run.snapshot_name,
        x_index_range=run.x_index_range,
        y_index_range=run.y_index_range,
        **values,
    )


def load_plot_config(path: str | Path) -> PlotSettings:
    """Locate the selected job/snapshot/region's products and figure directories.

    Paths are relative to the YAML directory. image_downsample_factor counts
    native pixels per image axis; raw_luminosity and allow_partial are Boolean.
    Only the dataset folder name is used; snapshot/table paths are not opened.
    """
    config_path, supplied = _read_mapping(path)
    run = read_run_selection(
        supplied=supplied,
        config_directory=config_path.parent,
    )
    values = {
        name: supplied.get(name, default)
        for name, default in _PLOT_DEFAULTS.items()
    }
    for name in ("raw_luminosity", "allow_partial"):
        if type(values[name]) is not bool:
            raise ValueError(f"{name} must be true or false")
    values["image_downsample_factor"] = _positive_int(
        "image_downsample_factor", values["image_downsample_factor"])
    titled_figures = supplied.get("titled_figures", True)
    if type(titled_figures) is not bool:
        raise ValueError("titled_figures must be true or false")
    titled_output_dir = None
    if titled_figures:
        titled_output_dir = run.directory / "figures_titled"
    return PlotSettings(
        products=run.directory / "processed",
        output_dir=run.directory / "figures",
        titled_output_dir=titled_output_dir,
        **values,
    )
