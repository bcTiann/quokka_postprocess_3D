"""Compare DESPOTIC and Cloudy CII spectra across the T_QUOKKA split.

The regime selection is always made with ``T_QUOKKA``.  For cells with
``T_QUOKKA < 3000 K``, the Cloudy table temperature coordinate and the CII
thermal width use ``T_DESPOTIC``.  The first diagnostic version clips only
``T_DESPOTIC`` values below the current Cloudy table minimum to that minimum;
the count is recorded explicitly for replacement by an extended table.

Only the low-temperature Cloudy spectra are recomputed.  Existing completed
products supply the high-temperature Cloudy spectra and both DESPOTIC curves.
This keeps the diagnostic targeted and avoids recomputing unrelated lines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt
from yt.units.physical_constants import kb, mh
from yt.units.yt_array import YTArray

from plot_cloudy_line_physics_ablation_spectra import (
    EXPECTED_LINES,
    EXPECTED_STATES,
    N_CHANNELS,
    REGIME_SPLIT_K,
    TOUCH_EPS,
    V_RANGE_KMS,
    _brackets,
    _load_bundle,
    accumulate_velocity_spectra,
    interpolate_line_coefficients,
    linear_fill_bracketed_failures,
)
from plot_expanded_four_model_spectra import _orient_like, _read_curve
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


COLUMN_FIELD = ("gas", "column_density_H")
TDSP_FIELD = ("gas", "temperature_despotic")
SELECTED_STATES = ("baseline", "mol_ct")
MODEL_ORDER = ("despotic", "cloudy_baseline", "cloudy_mol_ct")
MODEL_LABELS = {
    "despotic": "DESPOTIC",
    "cloudy_baseline": "Cloudy baseline",
    "cloudy_mol_ct": "Cloudy molecular + charge transfer",
}
MODEL_STYLES = {
    "despotic": ("#0072B2", "--"),
    "cloudy_baseline": ("#D55E00", "-."),
    "cloudy_mol_ct": ("#CC79A7", ":"),
}
COLUMN_COMPARISON_ORDER = (
    "despotic",
    "cloudy_baseline_full_nh",
    "cloudy_baseline_half_nh",
    "cloudy_mol_ct_full_nh",
    "cloudy_mol_ct_half_nh",
    "cloudy_draine_gow_cr_half_nh",
)
COLUMN_COMPARISON_LABELS = {
    "despotic": "DESPOTIC",
    "cloudy_baseline_full_nh": r"Cloudy baseline, $N_{\rm H}$",
    "cloudy_baseline_half_nh": r"Cloudy, $N_{\rm H}/2$",
    "cloudy_mol_ct_full_nh": (
        r"Cloudy molecular + charge transfer, $N_{\rm H}$"
    ),
    "cloudy_mol_ct_half_nh": (
        r"Cloudy molecular + charge transfer, $N_{\rm H}/2$"
    ),
    "cloudy_draine_gow_cr_half_nh": (
        r"Cloudy $N_{\rm H}/2$ + Draine + CR + GOW"
    ),
}
COLUMN_COMPARISON_STYLES = {
    "despotic": ("#0072B2", "--"),
    "cloudy_baseline_full_nh": ("#D55E00", "-"),
    "cloudy_baseline_half_nh": ("#D55E00", ":"),
    "cloudy_mol_ct_full_nh": ("#CC79A7", "-"),
    "cloudy_mol_ct_half_nh": ("#CC79A7", ":"),
    "cloudy_draine_gow_cr_half_nh": ("#009E73", "-."),
}
OUTPUT_STEM = "CII_DESPOTIC_CloudyBaseline_MolCT_Tsplit_CloudyLowUsesTDESPOTIC_Rinf"
EXTENDED_OUTPUT_STEM = (
    "CII_DESPOTIC_CloudyBaseline_MolCT_Tsplit_CloudyLowUsesTDESPOTIC_"
    "Tmin3p6K_Rinf"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_validated_cache(
    dataset: Path,
    despotic_table: Path,
    field: tuple[str, str],
    explicit: Path | None,
    dimensions: tuple[int, int, int],
) -> tuple[h5py.File, Path]:
    base_key = compute_cache_key(
        dataset_path=dataset,
        despotic_table_path=despotic_table,
        downsample_factor=cfg.DOWNSAMPLE_FACTOR,
        column_extension_lateral_kpc=cfg.COLUMN_EXTENSION_LATERAL_KPC,
    )
    expected_key = field_cache_key(base_key, field)
    path = (
        explicit.resolve()
        if explicit is not None
        else field_cache_path(cache_root_for_dataset(dataset), field)
    )
    handle = h5py.File(path, "r")
    actual_key = str(handle.attrs.get("cache_key", ""))
    actual_field = (
        str(handle.attrs.get("field_type", "")),
        str(handle.attrs.get("field_name", "")),
    )
    # Schema 20 changed Halpha emissivity and spectral output units, but did
    # not change these two expensive, purely thermodynamic/geometric fields.
    # Accept their schema-19 caches only when every other key component still
    # matches exactly; all other fields remain strict schema-20 reads.
    legacy_base_key = compute_cache_key(
        dataset_path=dataset,
        despotic_table_path=despotic_table,
        downsample_factor=cfg.DOWNSAMPLE_FACTOR,
        column_extension_lateral_kpc=cfg.COLUMN_EXTENSION_LATERAL_KPC,
        schema_version=19,
    )
    legacy_safe = (
        field in {COLUMN_FIELD, TDSP_FIELD}
        and int(handle.attrs.get("schema_version", -1)) == 19
        and actual_key == field_cache_key(legacy_base_key, field)
    )
    if (actual_key != expected_key and not legacy_safe) or actual_field != field:
        handle.close()
        raise RuntimeError(f"stale or mismatched field cache: {path}")
    if tuple(handle["data"].shape) != dimensions:
        handle.close()
        raise ValueError(f"cache shape does not match simulation: {path}")
    return handle, path


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    output = Path(cfg.OUTPUT_DIR)
    task = output / "task_intermediates"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path,
        default=root / "data/cloudy_lines_hm2012_z0_physics_ablation_4state_3line_10x10x20.npz",
    )
    parser.add_argument(
        "--extended-cii-bundle", dest="extended_cii_bundle",
        type=Path, default=None,
        help="optional two-state CII table extending the temperature axis below 10 K",
    )
    parser.add_argument(
        "--extra-cii-bundle", type=Path, default=None,
        help=(
            "optional single-state explicit-column CII table to add at the "
            "requested column scale"
        ),
    )
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument(
        "--despotic-table", type=Path, default=Path(cfg.DESPOTIC_TABLE_PATH),
    )
    parser.add_argument(
        "--existing-cloudy-spectra", type=Path,
        default=output / "cloudy_line_physics_ablation_spectra_Rinf_linear_failure_fill.npz",
    )
    parser.add_argument(
        "--existing-full-column-comparison", type=Path,
        default=(
            output
            / "CII_DESPOTIC_CloudyBaseline_MolCT_Tsplit_"
              "CloudyLowUsesTDESPOTIC_Tmin3p6K_Rinf.npz"
        ),
        help="completed N_H comparison used alongside a scaled-column run",
    )
    parser.add_argument(
        "--despotic-low", type=Path,
        default=task / "Build_CplusLowCloudyComparison_c389c117.h5",
    )
    parser.add_argument(
        "--despotic-high", type=Path,
        default=task / "Build_CplusHighModelComparison_3a471490.h5",
    )
    parser.add_argument("--column-cache", type=Path, default=None)
    parser.add_argument("--tdsp-cache", type=Path, default=None)
    parser.add_argument(
        "--column-scale", "--low-column-scale", dest="column_scale",
        type=float, default=1.0,
        help=(
            "multiply N_H by this factor when querying both temperature "
            "regimes; for example, 0.5 adds an N_H/2 comparison"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument("--slab-nz", type=int, default=64)
    parser.add_argument("--cell-chunk", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--max-slabs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_extended_cii_bundle(path: Path) -> dict[str, np.ndarray]:
    required = {
        "axis_order", "state_labels", "line_keys", "log_nH", "log_NH",
        "log_T", "log_emissivity_per_nH2", "emissivity_per_nH2",
        "failure_mask", "zero_mask",
    }
    with np.load(path, allow_pickle=False) as table:
        missing = sorted(required - set(table.files))
        if missing:
            raise ValueError(f"extended CII table is missing fields: {missing}")
        data = {key: np.asarray(table[key]) for key in required}
    if str(data["axis_order"].item()) != "state,line,log_nH,log_NH,log_T":
        raise ValueError("unexpected extended CII axis order")
    states = tuple(str(value) for value in data["state_labels"].tolist())
    lines = tuple(str(value) for value in data["line_keys"].tolist())
    if states != SELECTED_STATES or lines != ("cii",):
        raise ValueError(f"unexpected extended CII labels: states={states}, lines={lines}")
    expected = (2, 1, data["log_nH"].size, data["log_NH"].size, data["log_T"].size)
    for key in ("log_emissivity_per_nH2", "emissivity_per_nH2", "failure_mask", "zero_mask"):
        if data[key].shape != expected:
            raise ValueError(f"{key} has shape {data[key].shape}, expected {expected}")
    return data


def _load_single_cii_bundle(path: Path) -> dict[str, np.ndarray]:
    bundle_required = {
        "axis_order", "state_labels", "line_keys", "log_nH", "log_NH",
        "log_T", "log_emissivity_per_nH2", "emissivity_per_nH2",
        "failure_mask", "zero_mask",
    }
    with np.load(path, allow_pickle=False) as table:
        files = set(table.files)
        if bundle_required <= files:
            data = {key: np.asarray(table[key]) for key in bundle_required}
        else:
            view_required = {
                "axis_order", "state_label", "line_key", "log_nH", "log_NH",
                "log_T", "log_emissivity_per_nH2", "emissivity_per_nH2",
                "failure_mask", "zero_mask",
            }
            missing = sorted(view_required - files)
            if missing:
                raise ValueError(f"extra CII table is missing fields: {missing}")
            view = {key: np.asarray(table[key]) for key in view_required}
            if str(view["axis_order"].item()) != "log_nH,log_NH,log_T":
                raise ValueError("unexpected extra CII view axis order")
            if str(view["line_key"].item()) != "cii":
                raise ValueError("extra CII view does not contain CII")
            data = {
                "axis_order": np.asarray("state,line,log_nH,log_NH,log_T"),
                "state_labels": np.asarray([str(view["state_label"].item())]),
                "line_keys": np.asarray(["cii"]),
                "log_nH": view["log_nH"],
                "log_NH": view["log_NH"],
                "log_T": view["log_T"],
            }
            for key in (
                "log_emissivity_per_nH2", "emissivity_per_nH2",
                "failure_mask", "zero_mask",
            ):
                data[key] = view[key][None, None, ...]
    if str(data["axis_order"].item()) != "state,line,log_nH,log_NH,log_T":
        raise ValueError("unexpected extra CII axis order")
    states = tuple(str(value) for value in data["state_labels"].tolist())
    lines = tuple(str(value) for value in data["line_keys"].tolist())
    if len(states) != 1 or lines != ("cii",):
        raise ValueError(f"unexpected extra CII labels: states={states}, lines={lines}")
    expected = (
        1, 1, data["log_nH"].size, data["log_NH"].size,
        data["log_T"].size,
    )
    for key in (
        "log_emissivity_per_nH2", "emissivity_per_nH2", "failure_mask",
        "zero_mask",
    ):
        if data[key].shape != expected:
            raise ValueError(f"{key} has shape {data[key].shape}, expected {expected}")
    return data


def _interpolate_selected_cii(
    filled_log: np.ndarray,
    filled_coefficient: np.ndarray,
    remaining_failure: np.ndarray,
    zero_mask: np.ndarray,
    brackets: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    """Trilinearly interpolate all states and the sole CII line."""
    n_cells = brackets[0][2].size
    n_states = filled_coefficient.shape[0]
    linear_sum = np.zeros((n_states, n_cells))
    log_sum = np.zeros_like(linear_sum)
    zero_support = np.zeros_like(linear_sum, dtype=bool)
    failure_weight = np.zeros_like(linear_sum)
    for ni in (0, 1):
        n_index = brackets[0][ni]
        n_weight = brackets[0][2] if ni else 1.0 - brackets[0][2]
        for ci in (0, 1):
            c_index = brackets[1][ci]
            c_weight = brackets[1][2] if ci else 1.0 - brackets[1][2]
            for ti in (0, 1):
                t_index = brackets[2][ti]
                t_weight = brackets[2][2] if ti else 1.0 - brackets[2][2]
                weight = n_weight * c_weight * t_weight
                coefficient = filled_coefficient[:, 0, n_index, c_index, t_index]
                log_value = filled_log[:, 0, n_index, c_index, t_index]
                corner_failure = remaining_failure[:, 0, n_index, c_index, t_index]
                corner_zero = zero_mask[:, 0, n_index, c_index, t_index]
                linear_sum += coefficient * weight
                log_sum += np.where(np.isfinite(log_value), log_value, 0.0) * weight
                zero_support |= corner_zero & (weight > TOUCH_EPS)
                failure_weight += corner_failure * weight
    touched = failure_weight > TOUCH_EPS
    if np.any(touched):
        details = {
            str(index): int(np.count_nonzero(touched[index]))
            for index in range(n_states) if np.any(touched[index])
        }
        raise RuntimeError(f"simulation touches unbracketed extended-table failures: {details}")
    return np.where(zero_support, linear_sum, np.power(10.0, log_sum))


def _failure_support_weights(
    state_failure_mask: np.ndarray,
    brackets: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    """Return each state's trilinear weight carried by original failures."""
    weights = np.zeros((state_failure_mask.shape[0], brackets[0][2].size))
    for ni in (0, 1):
        n_index = brackets[0][ni]
        n_weight = brackets[0][2] if ni else 1.0 - brackets[0][2]
        for ci in (0, 1):
            c_index = brackets[1][ci]
            c_weight = brackets[1][2] if ci else 1.0 - brackets[1][2]
            for ti in (0, 1):
                t_index = brackets[2][ti]
                t_weight = brackets[2][2] if ti else 1.0 - brackets[2][2]
                weights += (
                    state_failure_mask[:, n_index, c_index, t_index]
                    * n_weight * c_weight * t_weight
                )
    return weights


def _load_high_cloudy(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as table:
        if not bool(np.asarray(table["completed_full_domain"]).item()):
            raise RuntimeError(f"existing Cloudy spectra are incomplete: {path}")
        velocity = np.asarray(table["velocity_kms"], dtype=float)
        spectra = np.asarray(table["dsigma_dv"], dtype=float)
        states = tuple(str(value) for value in table["state_labels"].tolist())
        lines = tuple(str(value) for value in table["line_keys"].tolist())
    cii_index = lines.index("cii")
    return velocity, {
        state: spectra[cii_index, states.index(state), 1]
        for state in SELECTED_STATES
    }


def _load_full_column_comparison(
    path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as table:
        if not bool(np.asarray(table["completed_full_domain"]).item()):
            raise RuntimeError(f"full-column comparison is incomplete: {path}")
        velocity = np.asarray(table["velocity_kms"], dtype=float)
        spectra = np.asarray(table["dsigma_dv"], dtype=float)
        models = tuple(str(value) for value in table["model_keys"].tolist())
    required = ("despotic", "cloudy_baseline", "cloudy_mol_ct")
    missing = [model for model in required if model not in models]
    if missing:
        raise ValueError(f"full-column comparison lacks models {missing}: {path}")
    return velocity, {
        model: spectra[models.index(model)] for model in required
    }


def _plot(
    output: Path,
    velocity: np.ndarray,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    column_scale: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    shared_max = max(
        float(np.nanmax(curves[model][branch]))
        for model in MODEL_ORDER for branch in (0, 1)
    )
    for branch, axis in enumerate(axes):
        for model in MODEL_ORDER:
            color, linestyle = MODEL_STYLES[model]
            label = MODEL_LABELS[model]
            if branch == 0 and column_scale == 0.5:
                if model == "cloudy_baseline":
                    label = r"Cloudy, $N_{\rm H}/2$"
                elif model == "cloudy_mol_ct":
                    label = r"Cloudy molecular + charge transfer, $N_{\rm H}/2$"
            axis.plot(
                velocity, curves[model][branch], color=color,
                linestyle=linestyle, linewidth=1.7, drawstyle="steps-mid",
                label=label,
            )
        axis.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
        axis.set_xlabel(r"Velocity [km s$^{-1}$]")
        axis.set_ylabel(dsigma_dv_ylabel(DSIGMA_DV_UNIT))
        axis.set_title(
            r"$T_{\rm QUOKKA}<3000\,$K"
            if branch == 0 else r"$T_{\rm QUOKKA}\geq3000\,$K"
        )
        axis.set_ylim(0.0, 1.05 * shared_max)
        axis.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        axis.legend(fontsize=8.5, frameon=False)
        axis.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
        axis.tick_params(axis="y", labelleft=True)

        # DESPOTIC dominates the low branch in the existing comparison.  Keep
        # the requested common y limits while showing both Cloudy curves.
        if branch == 0:
            cloud_peak = max(
                float(np.nanmax(curves[model][branch]))
                for model in ("cloudy_baseline", "cloudy_mol_ct")
            )
            if cloud_peak > 0.0:
                inset = axis.inset_axes([0.08, 0.52, 0.48, 0.40])
                for model in ("cloudy_baseline", "cloudy_mol_ct"):
                    color, linestyle = MODEL_STYLES[model]
                    inset.plot(
                        velocity, curves[model][branch], color=color,
                        linestyle=linestyle, linewidth=1.2,
                        drawstyle="steps-mid",
                    )
                inset.set_xlim(-35.0, 35.0)
                inset.set_ylim(0.0, 1.08 * cloud_peak)
                inset.ticklabel_format(
                    style="sci", axis="y", scilimits=(0, 0), useMathText=True,
                )
                inset.tick_params(labelsize=7)
                inset.grid(True, alpha=0.2, linestyle="--", linewidth=0.4)
                inset.set_title("zoom", fontsize=8)

    fig.suptitle(
        r"Comparison of the [C II] 158 $\mu$m spectrum, LOS y, $R=\infty$"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _plot_column_comparison(
    output: Path,
    velocity: np.ndarray,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    active_order = tuple(
        model for model in COLUMN_COMPARISON_ORDER if model in curves
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    shared_max = max(
        float(np.nanmax(curves[model][branch]))
        for model in active_order for branch in (0, 1)
    )
    cloudy_models = active_order[1:]
    for branch, axis in enumerate(axes):
        for model in active_order:
            color, linestyle = COLUMN_COMPARISON_STYLES[model]
            axis.plot(
                velocity, curves[model][branch], color=color,
                linestyle=linestyle, linewidth=1.7, drawstyle="steps-mid",
                label=COLUMN_COMPARISON_LABELS[model],
            )
        axis.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
        axis.set_xlabel(r"Velocity [km s$^{-1}$]")
        axis.set_ylabel(dsigma_dv_ylabel(DSIGMA_DV_UNIT))
        axis.set_title(
            r"$T_{\rm QUOKKA}<3000\,$K"
            if branch == 0 else r"$T_{\rm QUOKKA}\geq3000\,$K"
        )
        axis.set_ylim(0.0, 1.05 * shared_max)
        axis.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        axis.legend(fontsize=7.5, frameon=False)
        axis.ticklabel_format(
            style="sci", axis="y", scilimits=(0, 0), useMathText=True,
        )
        axis.tick_params(axis="y", labelleft=True)

        cloud_peak = max(
            float(np.nanmax(curves[model][branch])) for model in cloudy_models
        )
        if cloud_peak > 0.0:
            inset = axis.inset_axes([0.08, 0.52, 0.48, 0.40])
            for model in cloudy_models:
                color, linestyle = COLUMN_COMPARISON_STYLES[model]
                inset.plot(
                    velocity, curves[model][branch], color=color,
                    linestyle=linestyle, linewidth=1.2,
                    drawstyle="steps-mid",
                )
            inset.set_xlim(-35.0, 35.0)
            inset.set_ylim(0.0, 1.08 * cloud_peak)
            inset.ticklabel_format(
                style="sci", axis="y", scilimits=(0, 0), useMathText=True,
            )
            inset.tick_params(labelsize=7)
            inset.grid(True, alpha=0.2, linestyle="--", linewidth=0.4)
            inset.set_title("zoom", fontsize=8)

    fig.suptitle(
        r"[C II] 158 $\mu$m: Cloudy $N_{\rm H}$ versus $N_{\rm H}/2$, "
        r"LOS y, $R=\infty$"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if args.slab_nz <= 0 or args.cell_chunk <= 0 or args.workers <= 0:
        raise ValueError("slab, chunk, and worker counts must be positive")
    if not np.isfinite(args.column_scale) or args.column_scale <= 0.0:
        raise ValueError("--column-scale must be finite and positive")
    if cfg.DOWNSAMPLE_FACTOR != 1:
        raise NotImplementedError("this native-slab diagnostic requires downsample=1")
    for name in (
        "bundle", "dataset", "despotic_table", "existing_cloudy_spectra",
        "existing_full_column_comparison", "despotic_low", "despotic_high",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.extended_cii_bundle is not None:
        args.extended_cii_bundle = args.extended_cii_bundle.resolve()
    if args.extra_cii_bundle is not None:
        args.extra_cii_bundle = args.extra_cii_bundle.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = EXTENDED_OUTPUT_STEM if args.extended_cii_bundle is not None else OUTPUT_STEM
    if args.column_scale == 0.5:
        temperature_suffix = "_Tmin3p6K" if args.extended_cii_bundle is not None else ""
        output_stem = (
            "CII_DESPOTIC_CloudyNH_vs_NHdiv2_MolCT_Tsplit_"
            f"CloudyLowUsesTDESPOTIC{temperature_suffix}_Rinf"
        )
    elif args.column_scale != 1.0:
        scale_label = f"{args.column_scale:g}".replace(".", "p")
        output_stem += f"_LowNHx{scale_label}"
    if args.extra_cii_bundle is not None:
        output_stem += "_DraineGOWCR"
    spectra_path = args.output_dir / f"{output_stem}.npz"
    figure_path = args.output_dir / f"{output_stem}.png"
    provenance_path = args.output_dir / f"{output_stem}.json"
    products = (spectra_path, figure_path, provenance_path)
    if any(path.exists() for path in products) and not args.force:
        raise FileExistsError(
            "refusing to overwrite existing diagnostic; pass --force:\n"
            + "\n".join(str(path) for path in products if path.exists())
        )

    bundle = (
        _load_extended_cii_bundle(args.extended_cii_bundle)
        if args.extended_cii_bundle is not None else _load_bundle(args.bundle)
    )
    axes = tuple(
        np.asarray(bundle[name], dtype=float)
        for name in ("log_nH", "log_NH", "log_T")
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
    original_failure_mask = np.asarray(bundle["failure_mask"], dtype=bool)

    extra_bundle = None
    extra_axes = None
    extra_filled_log = None
    extra_filled_coefficient = None
    extra_remaining_failure = None
    extra_fill_records: list[dict[str, object]] = []
    if args.extra_cii_bundle is not None:
        extra_bundle = _load_single_cii_bundle(args.extra_cii_bundle)
        extra_axes = tuple(
            np.asarray(extra_bundle[name], dtype=float)
            for name in ("log_nH", "log_NH", "log_T")
        )
        (
            extra_filled_log,
            extra_filled_coefficient,
            extra_remaining_failure,
            extra_fill_records,
        ) = linear_fill_bracketed_failures(
            np.asarray(extra_bundle["log_emissivity_per_nH2"], dtype=float),
            np.asarray(extra_bundle["emissivity_per_nH2"], dtype=float),
            np.asarray(extra_bundle["failure_mask"], dtype=bool),
            np.asarray(extra_bundle["zero_mask"], dtype=bool),
            extra_axes,
        )

    ds = yt.load(str(args.dataset))
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    nx, ny, nz = dimensions
    column_file, column_path = _open_validated_cache(
        args.dataset, args.despotic_table, COLUMN_FIELD,
        args.column_cache, dimensions,
    )
    tdsp_file, tdsp_path = _open_validated_cache(
        args.dataset, args.despotic_table, TDSP_FIELD,
        args.tdsp_cache, dimensions,
    )

    cell_width_cm = np.asarray(
        ds.domain_width.to("cm") / ds.domain_dimensions, dtype=float,
    )
    cell_volume_cm3 = float(np.prod(cell_width_cm))
    projected_area_cm2 = float(nx * nz * cell_width_cm[0] * cell_width_cm[2])
    hydrogen_mass_g = float(mh.to_value("g"))
    carbon_mass_g = 12.01 * 1.66053906660e-24
    boltzmann_cgs = float(kb.to_value("erg/K"))
    c_kms = float(SPEED_OF_LIGHT_CGS.to_value("cm/s")) / 1.0e5
    velocity_edges = np.linspace(-V_RANGE_KMS, V_RANGE_KMS, N_CHANNELS + 1)
    velocity = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])
    accumulated_scaled = np.zeros(
        (len(SELECTED_STATES), 2, N_CHANNELS), dtype=float,
    )
    accumulated_extra = np.zeros((2, N_CHANNELS), dtype=float)
    state_indices = [EXPECTED_STATES.index(state) for state in SELECTED_STATES]
    cii_index = EXPECTED_LINES.index("cii")
    table_t_min = float(10.0 ** axes[2][0])
    table_t_max = float(10.0 ** axes[2][-1])
    total_slabs = (nz + args.slab_nz - 1) // args.slab_nz
    if args.max_slabs is not None:
        total_slabs = min(total_slabs, args.max_slabs)
    counts = {
        "T_QUOKKA_lt_3000_cells": 0,
        "T_QUOKKA_ge_3000_cells": 0,
        "T_DESPOTIC_below_table_min_clipped": 0,
        "T_DESPOTIC_ge_3000_within_low_selection": 0,
        "touches_any_original_failure": 0,
        "touches_original_failure_baseline": 0,
        "touches_original_failure_mol_ct": 0,
        "touches_any_new_below_10K_failure": 0,
        "extra_touches_original_failure": 0,
    }
    maximum_original_failure_weight = np.zeros(len(SELECTED_STATES))
    total_low_luminosity = np.zeros(len(SELECTED_STATES))
    failure_touched_low_luminosity = np.zeros(len(SELECTED_STATES))
    tdsp_selected_min = np.inf
    tdsp_selected_max = -np.inf
    started = time.perf_counter()

    try:
        for slab_number, iz in enumerate(
            range(0, min(nz, total_slabs * args.slab_nz), args.slab_nz), start=1,
        ):
            local_nz = min(args.slab_nz, nz - iz)
            left_edge = ds.domain_left_edge.copy()
            left_edge[2] += iz * (ds.domain_width[2] / ds.domain_dimensions[2])
            grid = ds.covering_grid(
                level=ds.max_level, left_edge=left_edge,
                dims=(nx, ny, local_nz),
            )
            temperature_qk = np.asarray(
                grid[("boxlib", "temperature")], dtype=float,
            ).reshape(-1)
            low = temperature_qk < REGIME_SPLIT_K
            density = np.asarray(
                grid[("gas", "density")].to("g/cm**3"), dtype=float,
            ).reshape(-1)
            velocity_y = np.asarray(
                grid[("gas", "velocity_y")].to("km/s"), dtype=float,
            ).reshape(-1)
            del grid
            n_H_all = density * float(cfg.X_H) / hydrogen_mass_g
            column_all = np.asarray(
                column_file["data"][:, :, iz:iz + local_nz], dtype=float,
            ).reshape(-1) * args.column_scale
            temperature_dsp_all = np.asarray(
                tdsp_file["data"][:, :, iz:iz + local_nz], dtype=float,
            ).reshape(-1)
            if not (
                np.isfinite(n_H_all).all() and np.isfinite(column_all).all()
                and np.isfinite(temperature_qk).all()
                and np.isfinite(temperature_dsp_all).all()
                and np.isfinite(velocity_y).all()
                and np.all(n_H_all > 0.0) and np.all(column_all > 0.0)
                and np.all(temperature_qk > 0.0)
                and np.all(temperature_dsp_all > 0.0)
            ):
                raise ValueError(f"invalid cell inputs in z slab {iz}:{iz + local_nz}")

            counts["T_QUOKKA_lt_3000_cells"] += int(low.sum())
            counts["T_QUOKKA_ge_3000_cells"] += int((~low).sum())
            selected_tdsp = temperature_dsp_all[low]
            clipped = selected_tdsp < table_t_min
            counts["T_DESPOTIC_below_table_min_clipped"] += int(
                np.count_nonzero(clipped)
            )
            counts["T_DESPOTIC_ge_3000_within_low_selection"] += int(
                np.count_nonzero(selected_tdsp >= REGIME_SPLIT_K)
            )
            if selected_tdsp.size:
                tdsp_selected_min = min(tdsp_selected_min, float(selected_tdsp.min()))
                tdsp_selected_max = max(tdsp_selected_max, float(selected_tdsp.max()))

            for branch_index, selected in enumerate((low, ~low)):
                if not np.any(selected):
                    continue
                n_H = n_H_all[selected]
                column = column_all[selected]
                selected_velocity = velocity_y[selected]
                raw_temperature = (
                    temperature_dsp_all[selected]
                    if branch_index == 0 else temperature_qk[selected]
                )
                temperature_lookup = (
                    np.maximum(raw_temperature, table_t_min)
                    if branch_index == 0 else raw_temperature
                )
                if np.any(temperature_lookup > table_t_max):
                    label = "T_DESPOTIC" if branch_index == 0 else "T_QUOKKA"
                    raise ValueError(
                        f"{label} exceeds Cloudy table in z slab {iz}:{iz + local_nz}"
                    )

                coordinates = (
                    np.log10(n_H), np.log10(column), np.log10(temperature_lookup),
                )
                temperature_name = "T_DESPOTIC" if branch_index == 0 else "T_QUOKKA"
                for name, axis, values in zip(
                    ("nH", "NH", temperature_name), axes, coordinates,
                ):
                    tolerance = 1.0e-12 * max(1.0, abs(axis[0]), abs(axis[-1]))
                    if (
                        np.any(values < axis[0] - tolerance)
                        or np.any(values > axis[-1] + tolerance)
                    ):
                        raise ValueError(
                            f"{name} outside Cloudy table in z slab {iz}:{iz + local_nz}"
                        )
                brackets = tuple(
                    _brackets(axis, values)
                    for axis, values in zip(axes, coordinates)
                )
                if args.extended_cii_bundle is not None:
                    selected_original_failure = original_failure_mask[:, 0]
                else:
                    selected_original_failure = original_failure_mask[
                        state_indices, cii_index,
                    ]
                original_failure_weight = _failure_support_weights(
                    selected_original_failure, brackets,
                )
                touched_by_state = original_failure_weight > TOUCH_EPS
                counts["touches_original_failure_baseline"] += int(
                    np.count_nonzero(touched_by_state[0])
                )
                counts["touches_original_failure_mol_ct"] += int(
                    np.count_nonzero(touched_by_state[1])
                )
                counts["touches_any_original_failure"] += int(
                    np.count_nonzero(np.any(touched_by_state, axis=0))
                )
                maximum_original_failure_weight = np.maximum(
                    maximum_original_failure_weight,
                    np.max(original_failure_weight, axis=1),
                )
                if args.extended_cii_bundle is not None and branch_index == 0:
                    new_low_failure = selected_original_failure.copy()
                    new_low_failure[:, :, :, axes[2] >= 1.0] = False
                    new_low_weight = _failure_support_weights(
                        new_low_failure, brackets,
                    )
                    counts["touches_any_new_below_10K_failure"] += int(
                        np.count_nonzero(
                            np.any(new_low_weight > TOUCH_EPS, axis=0)
                        )
                    )
                if args.extended_cii_bundle is not None:
                    coefficients = _interpolate_selected_cii(
                        filled_log, filled_coefficient, remaining_failure,
                        np.asarray(bundle["zero_mask"], dtype=bool), brackets,
                    )
                else:
                    coefficients = interpolate_line_coefficients(
                        cii_index, filled_log, filled_coefficient,
                        remaining_failure,
                        np.asarray(bundle["zero_mask"], dtype=bool), brackets,
                    )[state_indices]
                luminosities = (
                    coefficients.T * np.square(n_H)[:, None] * cell_volume_cm3
                )
                if branch_index == 0:
                    total_low_luminosity += np.sum(luminosities, axis=0)
                    for state_index in range(len(SELECTED_STATES)):
                        failure_touched_low_luminosity[state_index] += np.sum(
                            luminosities[
                                touched_by_state[state_index], state_index,
                            ]
                        )
                thermal_width = np.sqrt(
                    boltzmann_cgs * temperature_lookup / carbon_mass_g,
                ) / 1.0e5
                thermal_width *= 1.0 - selected_velocity / c_kms
                slab_spectra = accumulate_velocity_spectra(
                    selected_velocity, thermal_width, luminosities, velocity_edges,
                    cell_chunk=args.cell_chunk, workers=args.workers,
                )
                accumulated_scaled[:, branch_index] += slab_spectra.T

                if extra_bundle is not None:
                    assert extra_axes is not None
                    assert extra_filled_log is not None
                    assert extra_filled_coefficient is not None
                    assert extra_remaining_failure is not None
                    extra_coordinates = coordinates
                    for name, axis, values in zip(
                        ("nH", "NH", temperature_name),
                        extra_axes, extra_coordinates,
                    ):
                        tolerance = 1.0e-12 * max(
                            1.0, abs(axis[0]), abs(axis[-1]),
                        )
                        if (
                            np.any(values < axis[0] - tolerance)
                            or np.any(values > axis[-1] + tolerance)
                        ):
                            raise ValueError(
                                f"{name} outside extra Cloudy table in "
                                f"z slab {iz}:{iz + local_nz}"
                            )
                    extra_brackets = tuple(
                        _brackets(axis, values)
                        for axis, values in zip(extra_axes, extra_coordinates)
                    )
                    extra_original_weight = _failure_support_weights(
                        np.asarray(extra_bundle["failure_mask"], dtype=bool)[:, 0],
                        extra_brackets,
                    )
                    counts["extra_touches_original_failure"] += int(
                        np.count_nonzero(extra_original_weight[0] > TOUCH_EPS)
                    )
                    extra_coefficients = _interpolate_selected_cii(
                        extra_filled_log,
                        extra_filled_coefficient,
                        extra_remaining_failure,
                        np.asarray(extra_bundle["zero_mask"], dtype=bool),
                        extra_brackets,
                    )[0]
                    extra_luminosity = (
                        extra_coefficients * np.square(n_H) * cell_volume_cm3
                    )
                    extra_spectrum = accumulate_velocity_spectra(
                        selected_velocity,
                        thermal_width,
                        extra_luminosity[:, None],
                        velocity_edges,
                        cell_chunk=args.cell_chunk,
                        workers=args.workers,
                    )
                    accumulated_extra[branch_index] += extra_spectrum[:, 0]

            elapsed = time.perf_counter() - started
            rate = slab_number / elapsed
            eta = (total_slabs - slab_number) / rate if rate > 0.0 else np.nan
            print(
                f"[{slab_number:02d}/{total_slabs:02d}] z={iz}:{iz + local_nz} "
                f"low_cells={counts['T_QUOKKA_lt_3000_cells']} "
                f"clipped_lt_{table_t_min:g}K={counts['T_DESPOTIC_below_table_min_clipped']} "
                f"elapsed={elapsed / 60.0:.1f} min ETA={eta / 60.0:.1f} min",
                flush=True,
            )
    finally:
        column_file.close()
        tdsp_file.close()

    completed_full_domain = total_slabs * args.slab_nz >= nz
    scaled_cloudy = YTArray(
        accumulated_scaled / projected_area_cm2,
        "erg/s/cm**2/(km/s)",
    ).to(DSIGMA_DV_UNIT).value
    extra_cloudy = YTArray(
        accumulated_extra / projected_area_cm2,
        "erg/s/cm**2/(km/s)",
    ).to(DSIGMA_DV_UNIT).value
    if args.column_scale == 1.0:
        high_velocity, high_cloudy_raw = _load_high_cloudy(
            args.existing_cloudy_spectra,
        )
        high_cloudy = {
            state: _orient_like(high_velocity, values, velocity)
            for state, values in high_cloudy_raw.items()
        }
        low_dsp_velocity, low_dsp = _read_curve(
            args.despotic_low, "CPLUS_DESPOTIC_TQK_LT3000_DIAGNOSTIC",
            los_group=False,
        )
        high_dsp_velocity, high_dsp = _read_curve(
            args.despotic_high, "CPLUS_DESPOTIC_TQK_GE3000", los_group=True,
        )
        curves = {
            "despotic": (
                _orient_like(low_dsp_velocity, low_dsp, velocity),
                _orient_like(high_dsp_velocity, high_dsp, velocity),
            ),
            "cloudy_baseline": (
                scaled_cloudy[0, 0], high_cloudy["baseline"],
            ),
            "cloudy_mol_ct": (
                scaled_cloudy[1, 0], high_cloudy["mol_ct"],
            ),
        }
        output_model_order = MODEL_ORDER
    else:
        full_velocity, full_raw = _load_full_column_comparison(
            args.existing_full_column_comparison,
        )
        full = {
            model: np.asarray([
                _orient_like(full_velocity, values[0], velocity),
                _orient_like(full_velocity, values[1], velocity),
            ])
            for model, values in full_raw.items()
        }
        curves = {
            "despotic": tuple(full["despotic"]),
            "cloudy_baseline_full_nh": tuple(full["cloudy_baseline"]),
            "cloudy_baseline_half_nh": tuple(scaled_cloudy[0]),
            "cloudy_mol_ct_full_nh": tuple(full["cloudy_mol_ct"]),
            "cloudy_mol_ct_half_nh": tuple(scaled_cloudy[1]),
        }
        if extra_bundle is not None:
            curves["cloudy_draine_gow_cr_half_nh"] = tuple(extra_cloudy)
        output_model_order = tuple(
            model for model in COLUMN_COMPARISON_ORDER if model in curves
        )

    np.savez_compressed(
        spectra_path,
        velocity_kms=velocity,
        dsigma_dv=np.asarray([curves[model] for model in output_model_order]),
        model_keys=np.asarray(output_model_order),
        regime_keys=np.asarray(("T_QUOKKA_lt_3000K", "T_QUOKKA_ge_3000K")),
        dsigma_dv_units=np.asarray(DSIGMA_DV_UNIT),
        completed_full_domain=np.asarray(completed_full_domain),
        low_cloudy_temperature_coordinate=np.asarray(
            "T_DESPOTIC" if counts["T_DESPOTIC_below_table_min_clipped"] == 0
            else "max(T_DESPOTIC, table_Tmin)"
        ),
        high_cloudy_temperature_coordinate=np.asarray("T_QUOKKA"),
        regime_selection_temperature=np.asarray("T_QUOKKA"),
    )
    if args.column_scale == 1.0:
        _plot(figure_path, velocity, curves, args.column_scale)
    else:
        _plot_column_comparison(figure_path, velocity, curves)

    provenance = {
        "dataset": str(args.dataset),
        "cloudy_bundle": str(args.extended_cii_bundle or args.bundle),
        "cloudy_bundle_sha256": _sha256(args.extended_cii_bundle or args.bundle),
        "extra_cii_bundle": (
            str(args.extra_cii_bundle) if args.extra_cii_bundle is not None else None
        ),
        "extra_cii_bundle_sha256": (
            _sha256(args.extra_cii_bundle)
            if args.extra_cii_bundle is not None else None
        ),
        "existing_high_cloudy_spectra": str(args.existing_cloudy_spectra),
        "despotic_low_spectrum": str(args.despotic_low),
        "despotic_high_spectrum": str(args.despotic_high),
        "column_density_cache": str(column_path),
        "temperature_despotic_cache": str(tdsp_path),
        "regime_selection": "T_QUOKKA < 3000 K or T_QUOKKA >= 3000 K",
        "cloudy_temperature_policy": {
            "T_QUOKKA_lt_3000K": (
                "T_DESPOTIC without temperature clipping"
                if counts["T_DESPOTIC_below_table_min_clipped"] == 0
                else "T_DESPOTIC, clipped only below current table minimum"
            ),
            "T_QUOKKA_ge_3000K": "T_QUOKKA (reused completed spectrum)",
        },
        "cloudy_column_policy": {
            "plotted_full_column": "simulation N_H in both temperature regimes",
            "plotted_scaled_column": (
                f"{args.column_scale:g} * simulation N_H "
                "in both temperature regimes"
            ),
        },
        "column_scale": float(args.column_scale),
        "existing_full_column_comparison": str(
            args.existing_full_column_comparison
        ),
        "cloudy_table_temperature_bounds_K": [table_t_min, table_t_max],
        "raw_T_DESPOTIC_range_for_low_selection_K": [
            tdsp_selected_min, tdsp_selected_max,
        ],
        "counts": counts,
        "temporary_boundary_policy": (
            "none; the extended table covers the selected T_DESPOTIC range"
            if counts["T_DESPOTIC_below_table_min_clipped"] == 0 else
            f"{counts['T_DESPOTIC_below_table_min_clipped']} cells with "
            f"T_DESPOTIC < {table_t_min:g} K were evaluated at {table_t_min:g} K"
        ),
        "cloudy_states": list(SELECTED_STATES),
        "failure_policy": (
            "linear interpolation only between successful bracketing nodes; "
            "temperature axis preferred, then density, then column"
        ),
        "remaining_unbracketed_line_nodes": int(np.count_nonzero(remaining_failure)),
        "simulation_touched_remaining_failure_nodes": 0,
        "maximum_original_failure_weight_by_state": {
            state: float(maximum_original_failure_weight[index])
            for index, state in enumerate(SELECTED_STATES)
        },
        "original_failure_touched_luminosity_fraction_by_state": {
            state: float(
                failure_touched_low_luminosity[index]
                / total_low_luminosity[index]
            ) if total_low_luminosity[index] > 0.0 else 0.0
            for index, state in enumerate(SELECTED_STATES)
        },
        "linear_failure_fill_records": fill_records,
        "extra_linear_failure_fill_records": extra_fill_records,
        "extra_remaining_unbracketed_line_nodes": (
            int(np.count_nonzero(extra_remaining_failure))
            if extra_remaining_failure is not None else 0
        ),
        "los": "y",
        "spectral_resolution_R": "infinity",
        "spectrum_units": DSIGMA_DV_UNIT,
        "workers": int(args.workers),
        "completed_full_domain": completed_full_domain,
        "spectra": str(spectra_path),
        "figure": str(figure_path),
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved: {spectra_path}")
    print(f"Saved: {figure_path}")
    print(f"Saved: {provenance_path}")


if __name__ == "__main__":
    main()
