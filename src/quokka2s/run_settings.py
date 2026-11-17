"""Read shared YAML defaults and explicit selection or plotting arguments.

Only the paths, execution and plotting settings listed below are accepted.
dataset must be an absolute path to one snapshot. Other relative YAML paths
start at that file's directory. A leading ~ expands to the user's home directory;
environment variables such as $HOME are not expanded.
"""
from __future__ import annotations

import argparse
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
    """Resolved inputs and settings for one explicitly selected snapshot.

    dataset is an absolute path to a directory containing Header. Index ranges
    are native [start, stop) cell indices; omitted axes select their full extent.
    query_chunk bounds cells per batch and chunk_workers bounds concurrent batches.
    Gaussian templates use logarithmically placed sigma nodes with linear phase
    and sigma weights. Valid uncovered widths use direct stable integrals.
    The process selects its work exclusively through the x/y region.
    """

    dataset: Path
    output_dir: Path
    jobname: str
    snapshot_name: str
    despotic_table: Path = DEFAULT_DESPOTIC_TABLE
    cloudy_table: Path = DEFAULT_CLOUDY_TABLE
    dust_opacity_table: Path = DEFAULT_DRAINE_TABLE
    query_chunk: int = 262_144
    chunk_workers: int = 32
    x_index_range: tuple[int, int] | None = None
    y_index_range: tuple[int, int] | None = None
    gaussian_grid: dict | None = None
    z_reader_threads: int = 32
    z_read_workers: int = 5


@dataclass(frozen=True)
class PlotSettings:
    """Saved-result paths and display choices; no snapshot or table inputs.

    image_downsample_factor=2 selects the saved 2x2-summed image product.
    output_dir is the untitled figure directory; the titled version uses its
    name plus "_titled". figure_formats and figure_styles select which files
    are written. Omitted choices save PNG figures with titles only.
    """

    products: Path
    output_dir: Path
    figure_formats: tuple[str, ...] = ("png",)
    figure_styles: tuple[str, ...] = ("titled",)
    raw_luminosity: bool = False
    allow_partial: bool = False
    image_downsample_factor: int = 1


_RUN_REQUIRED_SETTINGS = {"dataset", "jobname", "results_dir"}
_PROCESS_TABLE_PATHS = ("despotic_table", "cloudy_table", "dust_opacity_table")
_PROCESS_INTEGERS = (
    "query_chunk", "chunk_workers", "z_reader_threads", "z_read_workers",
)
_REGION_INDEX_RANGES = ("x_index_range", "y_index_range")
_PROCESS_SETTING_NAMES = (
    *_PROCESS_TABLE_PATHS,
    *_PROCESS_INTEGERS,
    "gaussian_grid",
)
_PROCESS_DEFAULTS = {
    field.name: field.default
    for field in fields(ProcessSettings)
    if field.name in _PROCESS_SETTING_NAMES
}
for name in _PROCESS_TABLE_PATHS:
    # Bundled defaults remain anchored to this repository for external YAML files.
    _PROCESS_DEFAULTS[name] = str(_PROCESS_DEFAULTS[name])
_PLOT_DEFAULTS = {
    field.name: field.default
    for field in fields(PlotSettings)
    if field.default is not MISSING
}
_FIGURE_FORMATS = ("png", "pdf")
_FIGURE_STYLES = ("untitled", "titled")
_ALLOWED_SETTINGS = (
    _RUN_REQUIRED_SETTINGS
    | set(_PROCESS_DEFAULTS)
    | set(_REGION_INDEX_RANGES)
    | set(_PLOT_DEFAULTS)
)


def add_run_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add one absolute snapshot path, an output group and native x/y overrides.

    Example: --snapshot /simulations/plt0008703 --x 64 66 --y 64 66.
    Omitted options retain YAML values. Stops are exclusive; z remains complete.
    """
    parser.add_argument(
        "--snapshot",
        type=Path,
        metavar="PATH",
        help=(
            "Absolute path to one snapshot containing Header; process never scans "
            "a parent directory. Plot uses its folder name to select saved products"
        ),
    )
    parser.add_argument(
        "--jobname",
        metavar="NAME",
        help=(
            "Output job group overriding YAML for this invocation only"
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        type=Path,
        help="Shared emission YAML file (default: configs/emission.yaml)",
    )
    for axis in ("x", "y"):
        parser.add_argument(
            f"--{axis}",
            dest=f"{axis}_index_range",
            type=int,
            nargs=2,
            metavar=("START", "STOP"),
            help=f"Override YAML {axis} indices [START, STOP); z remains complete",
        )


def add_plot_arguments(parser: argparse.ArgumentParser) -> None:
    """Select output formats and title styles independently of YAML choices.

    Omitted categories retain YAML values. Example: --pdf selects only PDF;
    --png --pdf selects both. --titled --untitled writes both title versions.
    Single-dash spellings such as -png are accepted as explicit aliases.
    """
    for extension in _FIGURE_FORMATS:
        parser.add_argument(
            f"--{extension}",
            f"-{extension}",
            dest="figure_formats",
            action="append_const",
            const=extension,
            default=None,
            help=f"Save {extension.upper()} figures; format flags replace YAML figure_formats",
        )
    for style in _FIGURE_STYLES:
        parser.add_argument(
            f"--{style}",
            f"-{style}",
            dest="figure_styles",
            action="append_const",
            const=style,
            default=None,
            help=f"Save {style} figures; title flags replace YAML figure_styles",
        )


def _figure_choices(
    name: str,
    values: object,
    allowed: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate a nonempty choice list and return unique choices in save order."""
    message = f"{name} must be a nonempty list containing {', '.join(allowed)}"
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(message)
    if any(type(value) is not str or value not in allowed for value in values):
        raise ValueError(message)
    return tuple(value for value in allowed if value in values)


def _read_mapping(path: str | Path) -> tuple[Path, dict]:
    """Read a YAML mapping and reject unknown setting names.

    Return the absolute config path and values, without opening any input data.
    """
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
    if "source_root" in values:
        raise ValueError("Remove source_root from the emission YAML; set dataset to the snapshot path instead")
    unknown = set(values) - _ALLOWED_SETTINGS
    if unknown:
        raise ValueError(f"Unknown setting(s) in {config_path}: {', '.join(sorted(unknown))}")
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


def read_run_selection(
    supplied: dict,
    config_directory: Path,
    *,
    dataset: str | Path | None = None,
    jobname: str | None = None,
    x_index_range: list[int] | None = None,
    y_index_range: list[int] | None = None,
) -> RunSelection:
    """Select one saved-result directory without accessing original inputs.

    Explicit CLI options replace only their corresponding YAML values, without
    updating the file. dataset supplies the spelled snapshot name without opening
    its source, so plot works after the original snapshot has been moved away.
    """
    required = {"results_dir"}
    if dataset is None:
        required.add("dataset")
    if jobname is None:
        required.add("jobname")
    missing = required - set(supplied)
    if missing:
        raise ValueError(f"Missing setting(s): {', '.join(sorted(missing))}")
    dataset_name = _selected_dataset(supplied, dataset)
    snapshot_name = _folder_name("dataset folder name", Path(dataset_name).name)
    jobname = _folder_name("jobname", supplied["jobname"] if jobname is None else jobname)
    if x_index_range is None:
        x_index_range = supplied.get("x_index_range")
    if y_index_range is None:
        y_index_range = supplied.get("y_index_range")
    results_dir = _path_value("results_dir", supplied["results_dir"], config_directory)
    x_index_range = _index_range("x_index_range", x_index_range)
    y_index_range = _index_range("y_index_range", y_index_range)
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


def _selected_dataset(supplied: dict, dataset: str | Path | None) -> str:
    """Require one absolute snapshot path without opening the original data.

    Both YAML and CLI paths expand ~. The spelled basename selects saved products
    independently of source availability and symlink targets.
    """
    value = supplied.get("dataset") if dataset is None else dataset
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(
            "Supply --snapshot /absolute/path/to/pltNNNNNNN, or set an absolute "
            "dataset path in the YAML file"
        )
    expanded = Path(value).expanduser()
    if not expanded.is_absolute():
        raise ValueError("dataset must be an absolute path to one snapshot")
    return str(expanded)


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


def load_process_config(
    path: str | Path,
    *,
    dataset: str | Path | None = None,
    jobname: str | None = None,
    x_index_range: list[int] | None = None,
    y_index_range: list[int] | None = None,
) -> ProcessSettings:
    """Read one snapshot's settings and select its processed/ result directory.

    dataset must be absolute and contain Header. A parent run directory is
    rejected rather than expanded into multiple snapshots. Other YAML paths
    are relative to the YAML file. Plot options are accepted but unused here.
    """
    config_path, supplied = _read_mapping(path)
    run = read_run_selection(
        supplied=supplied,
        config_directory=config_path.parent,
        dataset=dataset,
        jobname=jobname,
        x_index_range=x_index_range,
        y_index_range=y_index_range,
    )
    dataset = Path(_selected_dataset(supplied, dataset))
    if not (dataset / "Header").is_file():
        raise ValueError(
            "dataset must point to ONE snapshot directory containing Header, "
            f"not its parent: {dataset}"
        )
    if run.directory.resolve().is_relative_to(dataset.resolve()):
        raise ValueError("results_dir must be outside the snapshot")
    values = {
        name: supplied.get(name, default)
        for name, default in _PROCESS_DEFAULTS.items()
    }
    for name in _PROCESS_TABLE_PATHS:
        values[name] = _path_value(name, values[name], config_path.parent)
    for name in _PROCESS_INTEGERS:
        values[name] = _positive_int(name, values[name])
    if values["z_read_workers"] > 5:
        raise ValueError("z_read_workers cannot exceed the five native fields")
    grid = values["gaussian_grid"]
    if grid is not None:
        if not isinstance(grid, dict):
            raise ValueError("gaussian_grid must be a mapping of template settings or null")
        supported = {
            "sigma_min", "sigma_max", "log_sigma_step",
            "mu_spacing_ratio", "maximum_phase_step",
        }
        unsupported = set(grid) - supported
        if unsupported:
            names = ", ".join(sorted(str(name) for name in unsupported))
            raise ValueError("Unsupported gaussian_grid keys: " + names)
    if values["query_chunk"] > 1_000_000:
        raise ValueError("query_chunk must be at most 1000000 cells")
    return ProcessSettings(
        dataset=dataset.resolve(),
        output_dir=run.directory / "processed",
        jobname=run.jobname,
        snapshot_name=run.snapshot_name,
        x_index_range=run.x_index_range,
        y_index_range=run.y_index_range,
        **values,
    )


def load_plot_config(
    path: str | Path,
    *,
    dataset: str | Path | None = None,
    jobname: str | None = None,
    x_index_range: list[int] | None = None,
    y_index_range: list[int] | None = None,
    figure_formats: list[str] | None = None,
    figure_styles: list[str] | None = None,
) -> PlotSettings:
    """Locate the selected job/snapshot/region's products and figure directories.

    Paths are relative to the YAML directory. image_downsample_factor counts
    native pixels per image axis; raw_luminosity and allow_partial are Boolean.
    Only the dataset folder name is used; snapshot/table paths are not opened.
    CLI figure selections replace only their specified category. Other choices
    use YAML values, then the PNG/titled defaults. Each selection is a nonempty
    list; repeated choices are saved once in the standard format/style order.
    Overrides do not modify YAML. jobname alone may select a different saved
    output group; the original snapshot does not need to remain available.
    """
    config_path, supplied = _read_mapping(path)
    return _plot_settings(
        config_path=config_path,
        supplied=supplied,
        dataset=dataset,
        jobname=jobname,
        x_index_range=x_index_range,
        y_index_range=y_index_range,
        figure_formats=figure_formats,
        figure_styles=figure_styles,
    )


def _plot_settings(
    *,
    config_path: Path,
    supplied: dict,
    dataset: str | Path | None,
    jobname: str | None,
    x_index_range: list[int] | None,
    y_index_range: list[int] | None,
    figure_formats: list[str] | None,
    figure_styles: list[str] | None,
) -> PlotSettings:
    """Validate saved-result selection and figure choices from one YAML mapping."""
    run = read_run_selection(
        supplied=supplied,
        config_directory=config_path.parent,
        dataset=dataset,
        jobname=jobname,
        x_index_range=x_index_range,
        y_index_range=y_index_range,
    )
    values = {
        name: supplied.get(name, default)
        for name, default in _PLOT_DEFAULTS.items()
    }
    if figure_formats is not None:
        values["figure_formats"] = figure_formats
    if figure_styles is not None:
        values["figure_styles"] = figure_styles
    values["figure_formats"] = _figure_choices(
        name="figure_formats",
        values=values["figure_formats"],
        allowed=_FIGURE_FORMATS,
    )
    values["figure_styles"] = _figure_choices(
        name="figure_styles",
        values=values["figure_styles"],
        allowed=_FIGURE_STYLES,
    )
    for name in ("raw_luminosity", "allow_partial"):
        if type(values[name]) is not bool:
            raise ValueError(f"{name} must be true or false")
    values["image_downsample_factor"] = _positive_int(
        "image_downsample_factor", values["image_downsample_factor"])
    return PlotSettings(
        products=run.directory / "processed",
        output_dir=run.directory / "figures",
        **values,
    )
