"""Read shared YAML defaults and explicit selection or plotting arguments.

Only the paths, execution and plotting settings listed below are accepted. A
relative path in the YAML file is relative to that file, not to the shell's
current working directory. A leading ~ expands to the user's home directory;
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
from quokka2s.snapshot_inventory import is_snapshot_name, list_snapshot_directories


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
    chunk_workers=8 and spectral_workers=3 allow at most 24 spectral tasks.
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
    query_chunk: int = 131_072
    chunk_workers: int = 8
    spectral_workers: int = 3
    max_slabs: int | None = None
    x_index_range: tuple[int, int] | None = None
    y_index_range: tuple[int, int] | None = None


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
_PROCESS_INTEGERS = ("slab_nx", "query_chunk", "chunk_workers", "spectral_workers", "max_slabs")
_REGION_INDEX_RANGES = ("x_index_range", "y_index_range")
_PROCESS_DEFAULTS = {
    field.name: field.default
    for field in fields(ProcessSettings)
    if field.default is not MISSING
    and field.name not in _REGION_INDEX_RANGES
}
for name in _PROCESS_TABLE_PATHS:
    # Bundled defaults stay anchored to this repository, even for external YAML.
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
    """Add shared YAML defaults and explicit job, dataset and native x/y overrides.

    Example: --dataset /simulations/plt0008703 --jobname sigma50 --x 64 128.
    An explicit dataset requires an explicit jobname. Omitted options retain
    YAML values; overrides affect this invocation only and stops remain exclusive.
    """
    parser.add_argument(
        "--dataset",
        type=Path,
        metavar="PATH",
        help=(
            "Absolute dataset path overriding YAML: process accepts one snapshot "
            "or its run directory; plot selects one snapshot by folder name; "
            "requires --jobname"
        ),
    )
    parser.add_argument(
        "--jobname",
        metavar="NAME",
        help=(
            "Output job group overriding YAML for this invocation only; "
            "required with --dataset"
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
    updating the file. An explicit dataset requires an explicit jobname rather
    than silently retaining the YAML output group. dataset supplies the spelled
    snapshot name without inspecting its source; plot works without that source.
    """
    if dataset is not None and jobname is None:
        raise ValueError("--dataset requires --jobname; select the output job group explicitly")
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
    """Validate a spelled dataset path without accessing the original source.

    YAML may use paths relative to its own directory. CLI overrides must be
    absolute after expanding ~; preserving the spelled basename keeps plot
    independent of source availability and symlink targets.
    """
    if dataset is not None:
        if not isinstance(dataset, (str, Path)) or not str(dataset).strip():
            raise ValueError("--dataset must be a nonempty absolute snapshot path")
        expanded = Path(dataset).expanduser()
        if not expanded.is_absolute():
            raise ValueError("--dataset must be an absolute snapshot path")
        return str(expanded)
    value = supplied.get("dataset")
    if type(value) is not str or not value.strip():
        raise ValueError("dataset must be a nonempty path string")
    return value


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
    """Read processing settings and derive the selected processed/ directory.

    Paths are relative to the YAML directory. Slab and query sizes count cells;
    worker settings count threads. x/y_index_range selects native cell indices;
    omitted axes use the full extent. The z axis always remains complete.
    Plot options in the shared file are accepted but not used by process.
    Explicit overrides replace only their matching YAML settings for this run.
    A CLI dataset must be absolute and accompanied by an explicit jobname.
    """
    config_path, supplied = _read_mapping(path)
    return _process_settings(
        config_path=config_path,
        supplied=supplied,
        dataset=dataset,
        jobname=jobname,
        x_index_range=x_index_range,
        y_index_range=y_index_range,
    )


def load_process_configs(
    path: str | Path,
    *,
    dataset: str | Path | None = None,
    jobname: str | None = None,
    x_index_range: list[int] | None = None,
    y_index_range: list[int] | None = None,
) -> list[ProcessSettings]:
    """Select one snapshot or the direct plt+digits children of a run directory.

    A directory containing Header, or a supplied plt+digits name, selects one
    snapshot. A missing Header on a named snapshot remains a processing failure;
    it is not treated as an empty run. All snapshots share the selected x/y
    ranges, output job and execution settings. An explicit dataset requires an
    explicit jobname. Directory discovery never reads cell data.
    """
    config_path, supplied = _read_mapping(path)
    selected = _process_settings(
        config_path=config_path,
        supplied=supplied,
        dataset=dataset,
        jobname=jobname,
        x_index_range=x_index_range,
        y_index_range=y_index_range,
    )
    if is_snapshot_name(selected.snapshot_name) or (selected.dataset / "Header").is_file():
        return [selected]

    snapshots = list_snapshot_directories(directory=selected.dataset)
    if not snapshots:
        raise ValueError(f"No plt+digits snapshot directories found in {selected.dataset}")
    configurations = []
    for snapshot in snapshots:
        config = _process_settings(
            config_path=config_path,
            supplied=supplied,
            dataset=snapshot,
            jobname=selected.jobname,
            x_index_range=x_index_range,
            y_index_range=y_index_range,
        )
        if config.output_dir.resolve().is_relative_to(selected.dataset):
            raise ValueError("results_dir must be outside the input run directory")
        configurations.append(config)
    return configurations


def _process_settings(
    *,
    config_path: Path,
    supplied: dict,
    dataset: str | Path | None,
    jobname: str | None,
    x_index_range: list[int] | None,
    y_index_range: list[int] | None,
) -> ProcessSettings:
    """Validate one snapshot's paths, index ranges and calculation settings."""
    run = read_run_selection(
        supplied=supplied,
        config_directory=config_path.parent,
        dataset=dataset,
        jobname=jobname,
        x_index_range=x_index_range,
        y_index_range=y_index_range,
    )
    dataset = _path_value("dataset", _selected_dataset(supplied, dataset), config_path.parent)
    if run.directory.resolve().is_relative_to(dataset):
        raise ValueError("results_dir must be outside the snapshot")
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
        dataset=dataset,
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
    Explicit dataset selection requires an explicit jobname. Neither override
    modifies YAML; jobname alone may select a different saved output group.
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


def load_plot_configs(
    path: str | Path,
    *,
    all_snapshots: bool = False,
    dataset: str | Path | None = None,
    jobname: str | None = None,
    x_index_range: list[int] | None = None,
    y_index_range: list[int] | None = None,
    figure_formats: list[str] | None = None,
    figure_styles: list[str] | None = None,
) -> list[PlotSettings]:
    """Select one saved result or every plt+digits directory in the output job.

    all_snapshots ignores YAML dataset entirely and lists results_dir/jobname
    only. Source snapshots and emission tables may be absent. The same native
    x/y region and figure choices apply to each selected result; missing products
    remain visible to the plotting caller rather than disappearing from the list.
    jobname overrides the YAML group only for this invocation. Without --all,
    an explicit dataset also requires an explicit jobname.
    """
    if all_snapshots and dataset is not None:
        raise ValueError("--all cannot be combined with --dataset")
    config_path, supplied = _read_mapping(path)
    if not all_snapshots:
        return [_plot_settings(
            config_path=config_path,
            supplied=supplied,
            dataset=dataset,
            jobname=jobname,
            x_index_range=x_index_range,
            y_index_range=y_index_range,
            figure_formats=figure_formats,
            figure_styles=figure_styles,
        )]

    required = {"results_dir"}
    if jobname is None:
        required.add("jobname")
    missing = required - set(supplied)
    if missing:
        raise ValueError(f"Missing setting(s): {', '.join(sorted(missing))}")
    jobname = _folder_name("jobname", supplied["jobname"] if jobname is None else jobname)
    results_dir = _path_value("results_dir", supplied["results_dir"], config_path.parent)
    job_directory = results_dir / jobname
    snapshots = list_snapshot_directories(directory=job_directory)
    if not snapshots:
        raise ValueError(f"No plt+digits result directories found in {job_directory}")
    configurations = []
    for snapshot in snapshots:
        config = _plot_settings(
            config_path=config_path,
            supplied=supplied,
            dataset=snapshot,
            jobname=jobname,
            x_index_range=x_index_range,
            y_index_range=y_index_range,
            figure_formats=figure_formats,
            figure_styles=figure_styles,
        )
        configurations.append(config)
    return configurations


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
