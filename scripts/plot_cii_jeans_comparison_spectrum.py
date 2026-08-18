"""Add the HM2012+Draine Jeans-length Cloudy CII spectrum to the comparison."""
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

from plot_cloudy_line_physics_ablation_spectra import accumulate_velocity_spectra
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


TDSP_FIELD = ("gas", "temperature_despotic")
SPLIT_K = 3000.0
TOUCH_EPS = 1.0e-12
DEFAULT_OUTPUT_STEM = "CII_DESPOTIC_CloudyBaseline_MolCT_JeansDraine_Tsplit_Rinf"
MODEL_LABELS = {
    "despotic": "DESPOTIC",
    "cloudy_baseline": "Cloudy baseline",
    "cloudy_mol_ct": "Cloudy molecular + charge transfer",
    "cloudy_jeans_draine": "Cloudy (Jeans length + Draine)",
    "cloudy_jeans_draine_gow_only": "Cloudy (Jeans + Draine + GOW)",
    "cloudy_jeans_draine_cr_only": "Cloudy (Jeans + Draine + CR)",
    "cloudy_jeans_draine_gow_cr": "Cloudy (Jeans + Draine + GOW + CR)",
    "cloudy_jeans_draine_gow_grains": "Cloudy (Jeans + Draine + GOW + grains)",
}
MODEL_STYLES = {
    "despotic": ("#0072B2", "--"),
    "cloudy_baseline": ("#D55E00", "-."),
    "cloudy_mol_ct": ("#CC79A7", ":"),
    "cloudy_jeans_draine": ("#009E73", "-"),
    "cloudy_jeans_draine_gow_only": ("#9467BD", "--"),
    "cloudy_jeans_draine_cr_only": ("#17BECF", "-."),
    "cloudy_jeans_draine_gow_cr": ("#E69F00", "-"),
    "cloudy_jeans_draine_gow_grains": ("#222222", ":"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _brackets(axis: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper = np.searchsorted(axis, values, side="right")
    upper = np.clip(upper, 1, axis.size - 1)
    lower = upper - 1
    fraction = (values - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, fraction


def _interpolate(
    log_values: np.ndarray,
    coefficient: np.ndarray,
    failure: np.ndarray,
    zero: np.ndarray,
    n_bracket: tuple[np.ndarray, np.ndarray, np.ndarray],
    t_bracket: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    n_cells = n_bracket[2].size
    linear_sum = np.zeros(n_cells)
    log_sum = np.zeros(n_cells)
    zero_support = np.zeros(n_cells, dtype=bool)
    failure_weight = np.zeros(n_cells)
    for ni in (0, 1):
        n_index = n_bracket[ni]
        n_weight = n_bracket[2] if ni else 1.0 - n_bracket[2]
        for ti in (0, 1):
            t_index = t_bracket[ti]
            t_weight = t_bracket[2] if ti else 1.0 - t_bracket[2]
            weight = n_weight * t_weight
            local_log = log_values[n_index, t_index]
            linear_sum += coefficient[n_index, t_index] * weight
            log_sum += np.where(np.isfinite(local_log), local_log, 0.0) * weight
            zero_support |= zero[n_index, t_index] & (weight > TOUCH_EPS)
            failure_weight += failure[n_index, t_index] * weight
    return np.where(zero_support, linear_sum, np.power(10.0, log_sum)), failure_weight


def _open_tdsp_cache(
    dataset: Path,
    despotic_table: Path,
    dimensions: tuple[int, int, int],
) -> tuple[h5py.File, Path]:
    base_key = compute_cache_key(
        dataset_path=dataset,
        despotic_table_path=despotic_table,
        downsample_factor=cfg.DOWNSAMPLE_FACTOR,
        column_extension_lateral_kpc=cfg.COLUMN_EXTENSION_LATERAL_KPC,
    )
    expected_key = field_cache_key(base_key, TDSP_FIELD)
    path = field_cache_path(cache_root_for_dataset(dataset), TDSP_FIELD).resolve()
    handle = h5py.File(path, "r")
    if str(handle.attrs.get("cache_key", "")) != expected_key:
        handle.close()
        raise RuntimeError(f"stale T_DESPOTIC cache: {path}")
    if tuple(handle["data"].shape) != dimensions:
        handle.close()
        raise ValueError(f"T_DESPOTIC cache shape mismatch: {path}")
    return handle, path


def _plot(
    path: Path,
    velocity: np.ndarray,
    curves: np.ndarray,
    model_keys: tuple[str, ...],
    model_labels: dict[str, str],
    model_styles: dict[str, tuple[str, str]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    shared_max = float(np.nanmax(curves))
    cloudy_indices = tuple(
        index for index, model in enumerate(model_keys) if model != "despotic"
    )
    for branch, axis in enumerate(axes):
        for model_index, model in enumerate(model_keys):
            color, linestyle = model_styles[model]
            axis.plot(
                velocity, curves[model_index, branch], color=color,
                linestyle=linestyle, linewidth=1.7, drawstyle="steps-mid",
                label=model_labels[model],
            )
        axis.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
        axis.set_xlabel(r"Velocity [km s$^{-1}$]")
        axis.set_ylabel(dsigma_dv_ylabel(DSIGMA_DV_UNIT))
        axis.set_title(
            r"$T_{\rm QUOKKA}<3000\,$K" if branch == 0
            else r"$T_{\rm QUOKKA}\geq3000\,$K"
        )
        axis.set_ylim(0.0, 1.05 * shared_max)
        axis.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        axis.legend(fontsize=8.1, frameon=False)
        axis.ticklabel_format(
            style="sci", axis="y", scilimits=(0, 0), useMathText=True,
        )
        axis.tick_params(axis="y", labelleft=True)
        if branch == 0:
            cloudy_peak = max(
                float(np.nanmax(curves[index, branch]))
                for index in cloudy_indices
            )
            if cloudy_peak > 0.0:
                inset = axis.inset_axes([0.08, 0.52, 0.48, 0.40])
                for model_index in cloudy_indices:
                    model = model_keys[model_index]
                    color, linestyle = model_styles[model]
                    inset.plot(
                        velocity, curves[model_index, branch], color=color,
                        linestyle=linestyle, linewidth=1.2,
                        drawstyle="steps-mid",
                    )
                inset.set_xlim(-35.0, 35.0)
                inset.set_ylim(0.0, 1.08 * cloudy_peak)
                inset.ticklabel_format(
                    style="sci", axis="y", scilimits=(0, 0), useMathText=True,
                )
                inset.tick_params(labelsize=7)
                inset.grid(True, alpha=0.2, linestyle="--", linewidth=0.4)
                inset.set_title("zoom", fontsize=8)
    fig.suptitle(r"Comparison of the [C II] 158 $\mu$m spectrum, LOS y, $R=\infty$")
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = Path(cfg.OUTPUT_DIR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jeans-table", type=Path,
        default=root / "data/cloudy_cii_hm2012_plus_draine_z0_baseline_jeans_10x21_T3p6_to1e9.npz",
    )
    parser.add_argument(
        "--existing-comparison", type=Path,
        default=output_dir / "CII_DESPOTIC_CloudyBaseline_MolCT_Tsplit_CloudyLowUsesTDESPOTIC_Tmin3p6K_Rinf.npz",
    )
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument(
        "--despotic-table", type=Path, default=Path(cfg.DESPOTIC_TABLE_PATH),
    )
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--slab-nz", type=int, default=64)
    parser.add_argument("--cell-chunk", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--model-key", default="cloudy_jeans_draine")
    parser.add_argument(
        "--model-label", default="Cloudy (Jeans length + Draine)",
    )
    parser.add_argument("--model-color", default="#009E73")
    parser.add_argument("--model-linestyle", default="-")
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in ("jeans_table", "existing_comparison", "dataset", "despotic_table", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.slab_nz <= 0 or args.cell_chunk <= 0 or args.workers <= 0:
        raise ValueError("slab, cell-chunk, and worker counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / f"{args.output_stem}.png"
    spectra_path = args.output_dir / f"{args.output_stem}.npz"
    provenance_path = args.output_dir / f"{args.output_stem}.json"
    products = (figure_path, spectra_path, provenance_path)
    if any(path.exists() for path in products) and not args.force:
        raise FileExistsError("output exists; pass --force to replace it")

    with np.load(args.jeans_table, allow_pickle=False) as table:
        log_nh_axis = np.asarray(table["log_nH"], dtype=float)
        log_t_axis = np.asarray(table["log_T"], dtype=float)
        log_values = np.asarray(table["log_emissivity_per_nH2"], dtype=float)
        coefficient = np.asarray(table["emissivity_per_nH2"], dtype=float)
        failure = np.asarray(table["failure_mask"], dtype=bool)
        zero = np.asarray(table["zero_mask"], dtype=bool)
    with np.load(args.existing_comparison, allow_pickle=False) as old:
        velocity = np.asarray(old["velocity_kms"], dtype=float)
        old_curves = np.asarray(old["dsigma_dv"], dtype=float)
        old_models = tuple(str(value) for value in old["model_keys"].tolist())
        old_regimes = tuple(str(value) for value in old["regime_keys"].tolist())
        old_units = str(old["dsigma_dv_units"].item())
    if old_regimes != ("T_QUOKKA_lt_3000K", "T_QUOKKA_ge_3000K"):
        raise ValueError("existing comparison regime labels differ from expected")
    if old_units != DSIGMA_DV_UNIT or old_curves.shape != (
        len(old_models), 2, velocity.size,
    ):
        raise ValueError("existing comparison shape or units differ from expected")
    if args.model_key in old_models:
        raise ValueError(f"model key already exists: {args.model_key}")
    model_keys = old_models + (args.model_key,)
    model_labels = {
        model: MODEL_LABELS.get(model, model) for model in old_models
    }
    model_labels[args.model_key] = args.model_label
    model_styles = {
        model: MODEL_STYLES.get(model, (f"C{index}", "-"))
        for index, model in enumerate(old_models)
    }
    model_styles[args.model_key] = (args.model_color, args.model_linestyle)
    delta_v = float(np.diff(velocity).mean())
    velocity_edges = np.concatenate((
        [velocity[0] - 0.5 * delta_v], velocity + 0.5 * delta_v,
    ))

    ds = yt.load(str(args.dataset))
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    nx, ny, nz = dimensions
    tdsp_file, tdsp_path = _open_tdsp_cache(
        args.dataset, args.despotic_table, dimensions,
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
    accumulated = np.zeros((velocity.size, 2))
    counts = {
        "all_cells": 0, "T_QUOKKA_lt_3000_cells": 0,
        "T_QUOKKA_ge_3000_cells": 0,
        "emitting_cells_T_QUOKKA_lt_3000": 0,
        "emitting_cells_T_QUOKKA_ge_3000": 0,
        "touches_failure": 0, "either_axis_out_of_bounds": 0,
    }
    maximum_failure_weight = 0.0
    started = time.perf_counter()
    total_slabs = (nz + args.slab_nz - 1) // args.slab_nz
    try:
        for slab_number, iz in enumerate(range(0, nz, args.slab_nz), start=1):
            local_nz = min(args.slab_nz, nz - iz)
            left_edge = ds.domain_left_edge.copy()
            left_edge[2] += iz * (ds.domain_width[2] / ds.domain_dimensions[2])
            grid = ds.covering_grid(
                level=ds.max_level, left_edge=left_edge,
                dims=(nx, ny, local_nz),
            )
            density = np.asarray(
                grid[("gas", "density")].to("g/cm**3"), dtype=float,
            ).reshape(-1)
            temperature_qk = np.asarray(
                grid[("boxlib", "temperature")], dtype=float,
            ).reshape(-1)
            velocity_y = np.asarray(
                grid[("gas", "velocity_y")].to("km/s"), dtype=float,
            ).reshape(-1)
            del grid
            temperature_dsp = np.asarray(
                tdsp_file["data"][:, :, iz:iz + local_nz], dtype=float,
            ).reshape(-1)
            n_h = density * float(cfg.X_H) / hydrogen_mass_g
            low = temperature_qk < SPLIT_K
            temperature_lookup = np.where(low, temperature_dsp, temperature_qk)
            log_nh = np.log10(n_h)
            log_t = np.log10(temperature_lookup)
            out = (
                (log_nh < log_nh_axis[0]) | (log_nh > log_nh_axis[-1])
                | (log_t < log_t_axis[0]) | (log_t > log_t_axis[-1])
            )
            counts["all_cells"] += n_h.size
            counts["T_QUOKKA_lt_3000_cells"] += int(np.count_nonzero(low))
            counts["T_QUOKKA_ge_3000_cells"] += int(np.count_nonzero(~low))
            counts["either_axis_out_of_bounds"] += int(np.count_nonzero(out))
            if np.any(out):
                raise RuntimeError(f"simulation leaves Jeans table in z slab {iz}:{iz + local_nz}")
            n_bracket = _brackets(log_nh_axis, log_nh)
            t_bracket = _brackets(log_t_axis, log_t)
            local_coefficient, failure_weight = _interpolate(
                log_values, coefficient, failure, zero, n_bracket, t_bracket,
            )
            touched = failure_weight > TOUCH_EPS
            counts["touches_failure"] += int(np.count_nonzero(touched))
            maximum_failure_weight = max(
                maximum_failure_weight, float(failure_weight.max()),
            )
            if np.any(touched):
                raise RuntimeError(f"simulation touches Jeans failures in z slab {iz}:{iz + local_nz}")
            luminosity = local_coefficient * np.square(n_h) * cell_volume_cm3
            counts["emitting_cells_T_QUOKKA_lt_3000"] += int(
                np.count_nonzero((luminosity > 0.0) & low)
            )
            counts["emitting_cells_T_QUOKKA_ge_3000"] += int(
                np.count_nonzero((luminosity > 0.0) & ~low)
            )
            luminosity_by_branch = np.column_stack((
                np.where(low, luminosity, 0.0),
                np.where(~low, luminosity, 0.0),
            ))
            thermal_width = np.sqrt(
                boltzmann_cgs * temperature_lookup / carbon_mass_g,
            ) / 1.0e5
            thermal_width *= 1.0 - velocity_y / c_kms
            accumulated += accumulate_velocity_spectra(
                velocity_y, thermal_width, luminosity_by_branch,
                velocity_edges, cell_chunk=args.cell_chunk, workers=args.workers,
            )
            elapsed = time.perf_counter() - started
            print(
                f"[{slab_number:02d}/{total_slabs:02d}] z={iz}:{iz + local_nz} "
                f"emitting=({counts['emitting_cells_T_QUOKKA_lt_3000']},"
                f"{counts['emitting_cells_T_QUOKKA_ge_3000']}) "
                f"elapsed={elapsed / 60.0:.2f} min",
                flush=True,
            )
    finally:
        tdsp_file.close()

    jeans_curves = YTArray(
        accumulated / projected_area_cm2, "erg/s/cm**2/(km/s)",
    ).to(DSIGMA_DV_UNIT).value.T
    curves = np.concatenate((old_curves, jeans_curves[None, :, :]), axis=0)
    _plot(
        figure_path, velocity, curves, model_keys, model_labels, model_styles,
    )
    np.savez_compressed(
        spectra_path, velocity_kms=velocity, dsigma_dv=curves,
        model_keys=np.asarray(model_keys),
        regime_keys=np.asarray(old_regimes),
        dsigma_dv_units=np.asarray(DSIGMA_DV_UNIT),
        completed_full_domain=np.asarray(True),
        regime_selection_temperature=np.asarray("T_QUOKKA"),
        added_cloudy_temperature_coordinate=np.asarray(
            "T_DESPOTIC where T_QUOKKA<3000 K; otherwise T_QUOKKA"
        ),
    )
    provenance = {
        "dataset": str(args.dataset),
        "jeans_table": str(args.jeans_table),
        "jeans_table_sha256": _sha256(args.jeans_table),
        "existing_comparison": str(args.existing_comparison),
        "existing_comparison_sha256": _sha256(args.existing_comparison),
        "temperature_despotic_cache": str(tdsp_path),
        "regime_selection": "T_QUOKKA < 3000 K or T_QUOKKA >= 3000 K",
        "jeans_temperature_policy": (
            "T_DESPOTIC for low-selected cells; T_QUOKKA for high-selected cells"
        ),
        "added_model_key": args.model_key,
        "added_model_label": args.model_label,
        "failure_policy": "no fill; abort if original failure weight > 1e-12",
        "counts": counts,
        "maximum_failure_weight": maximum_failure_weight,
        "workers": args.workers,
        "spectrum_units": DSIGMA_DV_UNIT,
        "peak_by_model_and_regime": {
            model: {
                old_regimes[branch]: float(np.max(curves[model_index, branch]))
                for branch in (0, 1)
            }
            for model_index, model in enumerate(model_keys)
        },
        "integrated_by_model_and_regime": {
            model: {
                old_regimes[branch]: float(np.sum(curves[model_index, branch]) * delta_v)
                for branch in (0, 1)
            }
            for model_index, model in enumerate(model_keys)
        },
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
        "figure": str(figure_path), "spectra": str(spectra_path),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved: {figure_path}")
    print(f"Saved: {spectra_path}")
    print(f"Saved: {provenance_path}")


if __name__ == "__main__":
    main()
