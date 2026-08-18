"""Build four-state Cloudy CII, H-alpha, and H I 21-cm spectra.

The input bundle has axes ``(state, line, log_nH, log_NH, log_T)``.  Cloudy
failure nodes are never interpreted as physical zeros.  A failed node is
filled only when two successful nodes bracket it along one grid axis; the
stored value is linearly interpolated in the logarithmic table coordinate.
Temperature is preferred, followed by density and column density.  Failures
that cannot be bracketed remain unavailable and abort if a simulation query
gives them non-zero trilinear weight.

Cloudy's successful ``-99`` sentinel remains an exact zero.  Normal table
sampling follows the schema-2 lookup convention: interpolate log emissivity
when all contributing corners are positive, and interpolate the non-negative
linear coefficient when a true-zero corner contributes.

The spectrum calculation is streamed in z slabs.  For each spectral line the
Gaussian velocity-bin kernel is evaluated once and multiplied by all four
physics states and both temperature selections at the same time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# The outer spectrum kernel supplies the requested worker parallelism.  Keep
# each individual matrix multiplication single-threaded to avoid 11x11 nested
# oversubscription on Apple Accelerate/OpenBLAS builds.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt
from scipy.special import erf as scipy_erf
from yt.units.physical_constants import kb, mh
from yt.units.yt_array import YTArray

from quokka2s.pipeline.cache import (
    cache_root_for_dataset,
    compute_cache_key,
    field_cache_key,
    field_cache_path,
)
from quokka2s.pipeline.prep import config as cfg
from quokka2s.pipeline.spectrum_units import (
    DSIGMA_DV_UNIT,
    SPEED_OF_LIGHT_CGS,
    dsigma_dv_ylabel,
)


EXPECTED_STATES = ("baseline", "mol", "ct", "mol_ct")
EXPECTED_LINES = ("cii", "halpha", "hi21")
STATE_LABELS = {
    "baseline": "baseline",
    "mol": "molecular network",
    "ct": "charge transfer",
    "mol_ct": "molecular + charge transfer",
}
STATE_STYLES = {
    "baseline": ("#0072B2", "-"),
    "mol": ("#D55E00", "--"),
    "ct": ("#009E73", "-."),
    "mol_ct": ("#CC79A7", ":"),
}
LINE_TITLES = {
    "cii": "[C II] 158 μm",
    "halpha": r"H$\alpha$",
    "hi21": "H I 21 cm",
}
LINE_FILENAMES = {
    "cii": "Cloudy_ablation_CII_Tsplit_Rinf_linear_failure_fill.png",
    "halpha": "Cloudy_ablation_Halpha_Tsplit_Rinf_linear_failure_fill.png",
    "hi21": "Cloudy_ablation_HI21_Tsplit_Rinf_linear_failure_fill.png",
}
REGIME_LABELS = (r"$T_{\rm QUOKKA}<3000\,$K", r"$T_{\rm QUOKKA}\geq3000\,$K")
COLUMN_FIELD = ("gas", "column_density_H")
REGIME_SPLIT_K = 3000.0
V_RANGE_KMS = 50.0
N_CHANNELS = 300
TOUCH_EPS = 1.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _brackets(
    axis: np.ndarray,
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper = np.searchsorted(axis, coordinates, side="right")
    upper = np.clip(upper, 1, axis.size - 1)
    lower = upper - 1
    fraction = (coordinates - axis[lower]) / (axis[upper] - axis[lower])
    return lower.astype(np.int16), upper.astype(np.int16), fraction


def _load_bundle(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as table:
        required = {
            "axis_order", "state_labels", "line_keys", "line_labels",
            "log_nH", "log_NH", "log_T", "log_emissivity_per_nH2",
            "emissivity_per_nH2", "failure_mask", "zero_mask",
        }
        missing = sorted(required - set(table.files))
        if missing:
            raise ValueError(f"bundle is missing fields: {missing}")
        data = {name: np.asarray(table[name]) for name in required}

    axis_order = str(data["axis_order"].item())
    states = tuple(str(value) for value in data["state_labels"].tolist())
    lines = tuple(str(value) for value in data["line_keys"].tolist())
    if axis_order != "state,line,log_nH,log_NH,log_T":
        raise ValueError(f"unexpected bundle axis order: {axis_order!r}")
    if states != EXPECTED_STATES:
        raise ValueError(f"unexpected states: {states}")
    if lines != EXPECTED_LINES:
        raise ValueError(f"unexpected lines: {lines}")

    shape = (
        len(states), len(lines), data["log_nH"].size,
        data["log_NH"].size, data["log_T"].size,
    )
    for name in (
        "log_emissivity_per_nH2", "emissivity_per_nH2",
        "failure_mask", "zero_mask",
    ):
        if data[name].shape != shape:
            raise ValueError(f"{name} has shape {data[name].shape}, expected {shape}")
    return data


def _nearest_valid_bracket(
    failure_mask: np.ndarray,
    index: tuple[int, int, int],
    axis_number: int,
) -> tuple[int, int] | None:
    lower = None
    upper = None
    for position in range(index[axis_number] - 1, -1, -1):
        candidate = list(index)
        candidate[axis_number] = position
        if not failure_mask[tuple(candidate)]:
            lower = position
            break
    for position in range(index[axis_number] + 1, failure_mask.shape[axis_number]):
        candidate = list(index)
        candidate[axis_number] = position
        if not failure_mask[tuple(candidate)]:
            upper = position
            break
    if lower is None or upper is None:
        return None
    return lower, upper


def linear_fill_bracketed_failures(
    log_values: np.ndarray,
    coefficients: np.ndarray,
    failure_mask: np.ndarray,
    zero_mask: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Fill only genuinely bracketed failures; never extrapolate boundaries."""
    filled_log = np.array(log_values, dtype=float, copy=True)
    filled_coefficient = np.array(coefficients, dtype=float, copy=True)
    remaining = np.array(failure_mask, dtype=bool, copy=True)
    records: list[dict[str, object]] = []

    # Prefer temperature because these are temperature sweeps.  A boundary
    # failure can still use a density or column bracket when one exists.
    axis_priority = (2, 0, 1)
    axis_names = ("log_nH", "log_NH", "log_T")
    states, lines = failure_mask.shape[:2]
    for state_index in range(states):
        for line_index in range(lines):
            original_mask = failure_mask[state_index, line_index]
            for raw_index in np.argwhere(original_mask):
                index = tuple(int(value) for value in raw_index)
                chosen = None
                for axis_number in axis_priority:
                    bracket = _nearest_valid_bracket(
                        original_mask, index, axis_number,
                    )
                    if bracket is not None:
                        chosen = (axis_number, *bracket)
                        break
                if chosen is None:
                    records.append({
                        "state_index": state_index,
                        "line_index": line_index,
                        "grid_index": list(index),
                        "status": "unfilled_unbracketed",
                    })
                    continue

                axis_number, lower_position, upper_position = chosen
                lower_index = list(index)
                upper_index = list(index)
                lower_index[axis_number] = lower_position
                upper_index[axis_number] = upper_position
                lower_index = tuple(lower_index)
                upper_index = tuple(upper_index)
                axis = axes[axis_number]
                fraction = (
                    (axis[index[axis_number]] - axis[lower_position])
                    / (axis[upper_position] - axis[lower_position])
                )

                lower_coefficient = filled_coefficient[
                    state_index, line_index, *lower_index,
                ]
                upper_coefficient = filled_coefficient[
                    state_index, line_index, *upper_index,
                ]
                bracket_has_zero = bool(
                    zero_mask[state_index, line_index, *lower_index]
                    or zero_mask[state_index, line_index, *upper_index]
                    or lower_coefficient == 0.0
                    or upper_coefficient == 0.0
                )
                if bracket_has_zero:
                    value = (
                        (1.0 - fraction) * lower_coefficient
                        + fraction * upper_coefficient
                    )
                    log_value = np.log10(value) if value > 0.0 else -99.0
                    value_space = "linear_coefficient"
                else:
                    lower_log = filled_log[state_index, line_index, *lower_index]
                    upper_log = filled_log[state_index, line_index, *upper_index]
                    log_value = (1.0 - fraction) * lower_log + fraction * upper_log
                    value = 10.0 ** log_value
                    value_space = "log10_coefficient"

                filled_log[state_index, line_index, *index] = log_value
                filled_coefficient[state_index, line_index, *index] = value
                remaining[state_index, line_index, *index] = False
                records.append({
                    "state_index": state_index,
                    "line_index": line_index,
                    "grid_index": list(index),
                    "status": "filled_linear",
                    "axis": axis_names[axis_number],
                    "bracket_indices": [lower_position, upper_position],
                    "fraction": float(fraction),
                    "value_space": value_space,
                    "log10_coefficient": float(log_value),
                })

    return filled_log, filled_coefficient, remaining, records


def interpolate_line_coefficients(
    line_index: int,
    filled_log: np.ndarray,
    filled_coefficient: np.ndarray,
    remaining_failure_mask: np.ndarray,
    zero_mask: np.ndarray,
    brackets: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    """Evaluate all four physics states at one line's cell coordinates."""
    n_cells = brackets[0][2].size
    linear_sum = np.zeros((len(EXPECTED_STATES), n_cells), dtype=float)
    log_sum = np.zeros_like(linear_sum)
    zero_support = np.zeros_like(linear_sum, dtype=bool)
    remaining_failure_weight = np.zeros_like(linear_sum)

    for n_corner in (0, 1):
        n_index = brackets[0][n_corner]
        n_weight = brackets[0][2] if n_corner else 1.0 - brackets[0][2]
        for column_corner in (0, 1):
            column_index = brackets[1][column_corner]
            column_weight = (
                brackets[1][2] if column_corner else 1.0 - brackets[1][2]
            )
            for temperature_corner in (0, 1):
                temperature_index = brackets[2][temperature_corner]
                temperature_weight = (
                    brackets[2][2]
                    if temperature_corner else 1.0 - brackets[2][2]
                )
                weight = n_weight * column_weight * temperature_weight
                coefficient = filled_coefficient[
                    :, line_index, n_index, column_index, temperature_index,
                ]
                log_value = filled_log[
                    :, line_index, n_index, column_index, temperature_index,
                ]
                corner_failure = remaining_failure_mask[
                    :, line_index, n_index, column_index, temperature_index,
                ]
                corner_zero = zero_mask[
                    :, line_index, n_index, column_index, temperature_index,
                ]
                linear_sum += coefficient * weight
                # Unresolved nodes have NaN log placeholders.  Their weight is
                # checked below; replacing NaN here prevents 0*NaN pollution.
                log_sum += np.where(np.isfinite(log_value), log_value, 0.0) * weight
                zero_support |= corner_zero & (weight > TOUCH_EPS)
                remaining_failure_weight += corner_failure * weight

    touched = remaining_failure_weight > TOUCH_EPS
    if np.any(touched):
        details = {
            EXPECTED_STATES[state_index]: int(np.count_nonzero(touched[state_index]))
            for state_index in range(len(EXPECTED_STATES))
            if np.any(touched[state_index])
        }
        raise RuntimeError(
            "simulation still touches unbracketed Cloudy failures after linear "
            f"fill: {details}"
        )
    return np.where(zero_support, linear_sum, np.power(10.0, log_sum))


def accumulate_velocity_spectra(
    velocity_kms: np.ndarray,
    thermal_width_kms: np.ndarray,
    luminosity_matrix: np.ndarray,
    velocity_edges_kms: np.ndarray,
    *,
    cell_chunk: int = 32768,
    workers: int = 1,
) -> np.ndarray:
    """Accumulate dL/dv for several luminosity models with one Gaussian kernel."""
    velocity = np.asarray(velocity_kms, dtype=float).reshape(-1)
    thermal = np.asarray(thermal_width_kms, dtype=float).reshape(-1)
    luminosity = np.asarray(luminosity_matrix, dtype=float)
    if luminosity.ndim != 2 or luminosity.shape[0] != velocity.size:
        raise ValueError("luminosity_matrix must have shape (n_cells, n_models)")
    if thermal.shape != velocity.shape:
        raise ValueError("thermal width and velocity must have matching shapes")
    if not np.all(np.diff(velocity_edges_kms) > 0.0):
        raise ValueError("velocity edges must be strictly increasing")

    emitting = np.any(luminosity != 0.0, axis=1)
    velocity = velocity[emitting]
    thermal = thermal[emitting]
    luminosity = luminosity[emitting]
    output_shape = (velocity_edges_kms.size - 1, luminosity.shape[1])
    delta_v = float(velocity_edges_kms[1] - velocity_edges_kms[0])
    channel_chunk = 150

    def accumulate_chunk(cell0: int) -> np.ndarray:
        cell1 = min(cell0 + cell_chunk, velocity.size)
        centers = velocity[cell0:cell1][None, :]
        values = luminosity[cell0:cell1]
        sigma = np.maximum(thermal[cell0:cell1], 1.0e-5)[None, :]
        denominator = np.sqrt(2.0) * sigma
        partial = np.zeros(output_shape, dtype=float)
        for channel0 in range(0, output_shape[0], channel_chunk):
            channel1 = min(channel0 + channel_chunk, output_shape[0])
            edges = velocity_edges_kms[channel0:channel1 + 1, None]
            erf_edges = (edges - centers) / denominator
            scipy_erf(erf_edges, out=erf_edges)
            fractions = 0.5 * (erf_edges[1:] - erf_edges[:-1])
            partial[channel0:channel1] = np.einsum(
                "kc,cm->km", fractions, values, optimize=False,
            ) / delta_v
        return partial

    chunk_starts = range(0, velocity.size, cell_chunk)
    output = np.zeros(output_shape, dtype=float)
    if workers <= 1:
        for cell0 in chunk_starts:
            output += accumulate_chunk(cell0)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # executor.map yields in input order, keeping the floating-point
            # reduction deterministic across repeated runs.
            for partial in executor.map(accumulate_chunk, chunk_starts):
                output += partial
    return output


def _open_column_cache(
    dataset_path: Path,
    despotic_table_path: Path,
    explicit_path: Path | None,
) -> tuple[h5py.File, Path]:
    base_key = compute_cache_key(
        dataset_path=dataset_path,
        despotic_table_path=despotic_table_path,
        downsample_factor=cfg.DOWNSAMPLE_FACTOR,
        column_extension_lateral_kpc=cfg.COLUMN_EXTENSION_LATERAL_KPC,
    )
    expected_key = field_cache_key(base_key, COLUMN_FIELD)
    path = (
        explicit_path.resolve()
        if explicit_path is not None
        else field_cache_path(cache_root_for_dataset(dataset_path), COLUMN_FIELD)
    )
    handle = h5py.File(path, "r")
    actual_key = str(handle.attrs.get("cache_key", ""))
    actual_field = (
        str(handle.attrs.get("field_type", "")),
        str(handle.attrs.get("field_name", "")),
    )
    if actual_key != expected_key or actual_field != COLUMN_FIELD:
        handle.close()
        raise RuntimeError(f"stale or mismatched column-density cache: {path}")
    return handle, path


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path,
        default=root / "data/cloudy_lines_hm2012_z0_physics_ablation_4state_3line_10x10x20.npz",
    )
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument(
        "--despotic-table", type=Path, default=Path(cfg.DESPOTIC_TABLE_PATH),
        help="used only to validate the existing column-density cache",
    )
    parser.add_argument("--column-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(cfg.OUTPUT_DIR))
    parser.add_argument("--slab-nz", type=int, default=32)
    parser.add_argument("--max-slabs", type=int, default=None)
    parser.add_argument("--cell-chunk", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _plot_line(
    output_path: Path,
    line: str,
    velocity: np.ndarray,
    spectra: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    shared_max = float(np.nanmax(spectra))
    if not np.isfinite(shared_max) or shared_max <= 0.0:
        raise ValueError(f"{line} spectra have no positive finite values")
    branch_peaks = np.nanmax(spectra, axis=(0, 2))

    for branch_index, (axis, branch_label) in enumerate(zip(axes, REGIME_LABELS)):
        for state_index, state in enumerate(EXPECTED_STATES):
            color, linestyle = STATE_STYLES[state]
            axis.plot(
                velocity, spectra[state_index, branch_index],
                color=color, linestyle=linestyle, linewidth=1.6,
                drawstyle="steps-mid", label=STATE_LABELS[state],
            )
        axis.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
        axis.set_xlabel(r"Velocity [km s$^{-1}$]")
        axis.set_ylabel(dsigma_dv_ylabel(DSIGMA_DV_UNIT))
        axis.set_title(branch_label)
        axis.set_ylim(0.0, 1.05 * shared_max)
        axis.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        axis.legend(fontsize=8.5, frameon=False)
        axis.ticklabel_format(
            style="sci", axis="y", scilimits=(0, 0), useMathText=True,
        )
        # Matplotlib hides right-panel labels for shared axes by default; the
        # comparison figures deliberately retain numeric ticks on both sides.
        axis.tick_params(axis="y", labelleft=True)

        if 0.0 < branch_peaks[branch_index] < 0.15 * shared_max:
            inset = axis.inset_axes([0.08, 0.52, 0.48, 0.40])
            for state_index, state in enumerate(EXPECTED_STATES):
                color, linestyle = STATE_STYLES[state]
                inset.plot(
                    velocity, spectra[state_index, branch_index],
                    color=color, linestyle=linestyle, linewidth=1.2,
                    drawstyle="steps-mid",
                )
            inset.set_xlim(-35.0, 35.0)
            inset.set_ylim(0.0, 1.08 * float(branch_peaks[branch_index]))
            inset.ticklabel_format(
                style="sci", axis="y", scilimits=(0, 0), useMathText=True,
            )
            inset.tick_params(labelsize=7)
            inset.grid(True, alpha=0.2, linestyle="--", linewidth=0.4)
            inset.set_title("zoom", fontsize=8)

    fig.suptitle(
        f"Effect of molecular network and charge transfer on {LINE_TITLES[line]}, "
        r"LOS y, $R=\infty$"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if args.slab_nz <= 0 or args.cell_chunk <= 0 or args.workers <= 0:
        raise ValueError("--slab-nz, --cell-chunk, and --workers must be positive")
    if cfg.DOWNSAMPLE_FACTOR != 1:
        raise NotImplementedError("this native-slab spectrum builder requires downsample=1")

    args.bundle = args.bundle.resolve()
    args.dataset = args.dataset.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spectra_path = args.output_dir / "cloudy_line_physics_ablation_spectra_Rinf_linear_failure_fill.npz"
    provenance_path = args.output_dir / "cloudy_line_physics_ablation_spectra_Rinf_linear_failure_fill.json"
    figure_paths = {
        line: args.output_dir / filename
        for line, filename in LINE_FILENAMES.items()
    }
    products = [spectra_path, provenance_path, *figure_paths.values()]
    existing = [path for path in products if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "refusing to overwrite existing products; pass --force:\n"
            + "\n".join(str(path) for path in existing)
        )

    bundle = _load_bundle(args.bundle)
    axes = (
        np.asarray(bundle["log_nH"], dtype=float),
        np.asarray(bundle["log_NH"], dtype=float),
        np.asarray(bundle["log_T"], dtype=float),
    )
    filled_log, filled_coefficient, remaining_failure, fill_records = (
        linear_fill_bracketed_failures(
            np.asarray(bundle["log_emissivity_per_nH2"], dtype=float),
            np.asarray(bundle["emissivity_per_nH2"], dtype=float),
            np.asarray(bundle["failure_mask"], dtype=bool),
            np.asarray(bundle["zero_mask"], dtype=bool),
            axes,
        )
    )
    filled_count = sum(record["status"] == "filled_linear" for record in fill_records)
    unresolved_count = int(np.count_nonzero(remaining_failure))
    print(
        f"Failure handling: filled line-nodes={filled_count}, "
        f"unbracketed line-nodes={unresolved_count}",
        flush=True,
    )

    column_file, column_path = _open_column_cache(
        args.dataset, args.despotic_table.resolve(), args.column_cache,
    )
    ds = yt.load(str(args.dataset))
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    if tuple(column_file["data"].shape) != dimensions:
        column_file.close()
        raise ValueError("column cache shape does not match the simulation")
    nx, ny, nz = dimensions
    cell_width_cm = np.asarray(
        (ds.domain_width.to("cm") / ds.domain_dimensions), dtype=float,
    )
    cell_volume_cm3 = float(np.prod(cell_width_cm))
    projected_area_cm2 = float(nx * nz * cell_width_cm[0] * cell_width_cm[2])
    hydrogen_mass_g = float(mh.to_value("g"))
    boltzmann_cgs = float(kb.to_value("erg/K"))
    amu_g = 1.66053906660e-24
    line_masses_g = {
        "cii": 12.01 * amu_g,
        "halpha": 1.00794 * amu_g,
        "hi21": 1.00794 * amu_g,
    }

    velocity_edges = np.linspace(-V_RANGE_KMS, V_RANGE_KMS, N_CHANNELS + 1)
    velocity_axis = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])
    accumulated = np.zeros(
        (len(EXPECTED_LINES), len(EXPECTED_STATES), 2, N_CHANNELS),
        dtype=float,
    )
    total_slabs = (nz + args.slab_nz - 1) // args.slab_nz
    if args.max_slabs is not None:
        total_slabs = min(total_slabs, args.max_slabs)
    started = time.perf_counter()

    try:
        for slab_number, iz in enumerate(
            range(0, min(nz, total_slabs * args.slab_nz), args.slab_nz),
            start=1,
        ):
            local_nz = min(args.slab_nz, nz - iz)
            left_edge = ds.domain_left_edge.copy()
            left_edge[2] += iz * (ds.domain_width[2] / ds.domain_dimensions[2])
            grid = ds.covering_grid(
                level=ds.max_level,
                left_edge=left_edge,
                dims=(nx, ny, local_nz),
            )
            temperature = np.asarray(grid[("boxlib", "temperature")], dtype=float)
            density = np.asarray(grid[("gas", "density")].to("g/cm**3"), dtype=float)
            velocity_y = np.asarray(grid[("gas", "velocity_y")].to("km/s"), dtype=float)
            n_H = density * float(cfg.X_H) / hydrogen_mass_g
            column = np.asarray(
                column_file["data"][:, :, iz:iz + local_nz], dtype=float,
            )
            del grid, density

            if not (
                np.isfinite(temperature).all()
                and np.isfinite(n_H).all()
                and np.isfinite(column).all()
                and np.isfinite(velocity_y).all()
                and np.all(temperature > 0.0)
                and np.all(n_H > 0.0)
                and np.all(column > 0.0)
            ):
                raise ValueError(f"invalid simulation inputs in z slab {iz}:{iz + local_nz}")

            coordinates = (
                np.log10(n_H).reshape(-1),
                np.log10(column).reshape(-1),
                np.log10(temperature).reshape(-1),
            )
            for name, axis, values in zip(("nH", "NH", "T"), axes, coordinates):
                tolerance = 1.0e-12 * max(1.0, abs(axis[0]), abs(axis[-1]))
                if np.any(values < axis[0] - tolerance) or np.any(values > axis[-1] + tolerance):
                    raise ValueError(f"{name} outside Cloudy table in z slab {iz}:{iz + local_nz}")
            brackets = tuple(
                _brackets(axis, values)
                for axis, values in zip(axes, coordinates)
            )
            flat_temperature = temperature.reshape(-1)
            low = flat_temperature < REGIME_SPLIT_K
            n_H_squared_volume = np.square(n_H.reshape(-1)) * cell_volume_cm3
            flat_velocity = velocity_y.reshape(-1)

            # C II has a carbon thermal kernel.  H-alpha and H I share the
            # same hydrogen thermal kernel, so accumulate those two lines in
            # a single matrix multiplication.
            for line_group in (("cii",), ("halpha", "hi21")):
                model_count = len(line_group) * len(EXPECTED_STATES) * 2
                luminosities = np.zeros((flat_temperature.size, model_count), dtype=float)
                for group_line_index, line in enumerate(line_group):
                    line_index = EXPECTED_LINES.index(line)
                    coefficients = interpolate_line_coefficients(
                        line_index,
                        filled_log,
                        filled_coefficient,
                        remaining_failure,
                        np.asarray(bundle["zero_mask"], dtype=bool),
                        brackets,
                    )
                    cell_luminosity = coefficients.T * n_H_squared_volume[:, None]
                    base = group_line_index * len(EXPECTED_STATES) * 2
                    luminosities[:, base:base + 8:2] = np.where(
                        low[:, None], cell_luminosity, 0.0,
                    )
                    luminosities[:, base + 1:base + 8:2] = np.where(
                        low[:, None], 0.0, cell_luminosity,
                    )

                mass_g = line_masses_g[line_group[0]]
                thermal_width_kms = np.sqrt(
                    boltzmann_cgs * flat_temperature / mass_g,
                ) / 1.0e5
                # SpectrumStore builds the Gaussian in frequency space with
                # sigma_nu = nu_shifted * sigma_v / c.  Expressing that same
                # kernel on the velocity axis therefore contributes the tiny
                # Doppler factor (1-v/c) to sigma_v.
                c_kms = float(SPEED_OF_LIGHT_CGS.to_value("cm/s")) / 1.0e5
                thermal_width_kms *= 1.0 - flat_velocity / c_kms
                group_spectra = accumulate_velocity_spectra(
                    flat_velocity,
                    thermal_width_kms,
                    luminosities,
                    velocity_edges,
                    cell_chunk=args.cell_chunk,
                    workers=args.workers,
                )
                for group_line_index, line in enumerate(line_group):
                    line_index = EXPECTED_LINES.index(line)
                    base = group_line_index * len(EXPECTED_STATES) * 2
                    for state_index in range(len(EXPECTED_STATES)):
                        accumulated[line_index, state_index, 0] += group_spectra[
                            :, base + 2 * state_index,
                        ]
                        accumulated[line_index, state_index, 1] += group_spectra[
                            :, base + 2 * state_index + 1,
                        ]
                del luminosities, group_spectra

            elapsed = time.perf_counter() - started
            rate = slab_number / elapsed
            eta = (total_slabs - slab_number) / rate if rate > 0.0 else np.nan
            print(
                f"[{slab_number:02d}/{total_slabs:02d}] z={iz}:{iz + local_nz} "
                f"elapsed={elapsed / 60.0:.1f} min ETA={eta / 60.0:.1f} min",
                flush=True,
            )
    finally:
        column_file.close()

    spectra = YTArray(
        accumulated / projected_area_cm2,
        "erg/s/cm**2/(km/s)",
    ).to(DSIGMA_DV_UNIT).value
    completed_full_domain = total_slabs * args.slab_nz >= nz
    np.savez_compressed(
        spectra_path,
        velocity_kms=velocity_axis,
        dsigma_dv=spectra,
        dsigma_dv_units=np.asarray(DSIGMA_DV_UNIT),
        state_labels=np.asarray(EXPECTED_STATES),
        line_keys=np.asarray(EXPECTED_LINES),
        regime_labels=np.asarray(("T_lt_3000K", "T_ge_3000K")),
        regime_split_K=np.asarray(REGIME_SPLIT_K),
        los=np.asarray("y"),
        spectral_resolution_R=np.asarray(np.inf),
        completed_full_domain=np.asarray(completed_full_domain),
    )

    for line_index, line in enumerate(EXPECTED_LINES):
        _plot_line(figure_paths[line], line, velocity_axis, spectra[line_index])
        print(f"Saved: {figure_paths[line]}", flush=True)

    provenance = {
        "dataset": str(args.dataset),
        "bundle": str(args.bundle),
        "bundle_sha256": _sha256(args.bundle),
        "column_density_cache": str(column_path),
        "spectra": str(spectra_path),
        "figures": {line: str(path) for line, path in figure_paths.items()},
        "grid_shape": list(dimensions),
        "processed_z_cells": int(min(nz, total_slabs * args.slab_nz)),
        "completed_full_domain": completed_full_domain,
        "states": list(EXPECTED_STATES),
        "lines": list(EXPECTED_LINES),
        "temperature_regimes": ["T < 3000 K", "T >= 3000 K"],
        "los": "y",
        "spectral_resolution_R": "infinity",
        "spectrum_units": DSIGMA_DV_UNIT,
        "spectrum_workers": int(args.workers),
        "failure_policy": (
            "linear interpolation only between successful bracketing nodes; "
            "temperature axis preferred, then density, then column; no extrapolation"
        ),
        "original_failed_line_nodes": int(np.count_nonzero(bundle["failure_mask"])),
        "linearly_filled_line_nodes": int(filled_count),
        "unbracketed_line_nodes": int(unresolved_count),
        "unbracketed_nodes_touched_by_simulation": 0,
        "true_zero_policy": "Cloudy -99 remains exact zero",
        "fill_records": fill_records,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved: {spectra_path}")
    print(f"Saved: {provenance_path}")


if __name__ == "__main__":
    main()
