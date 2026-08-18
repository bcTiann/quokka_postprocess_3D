"""Plot default-abundance CII spectra for HM2012, Draine, and matched CR."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt
from yt.units.physical_constants import kb, mh
from yt.units.yt_array import YTArray

from plot_cii_cloudy_tdsp_split_spectra import (
    COLUMN_FIELD,
    TDSP_FIELD,
    _failure_support_weights,
    _interpolate_selected_cii,
    _load_single_cii_bundle,
    _open_validated_cache,
)
from plot_cii_gow_cr_comparisons import _load_despotic, _prepare_table
from plot_cii_jeans_comparison_spectrum import _brackets as _brackets_2d
from plot_cii_jeans_comparison_spectrum import _interpolate as _interpolate_2d
from plot_cloudy_line_physics_ablation_spectra import (
    N_CHANNELS,
    REGIME_SPLIT_K,
    TOUCH_EPS,
    V_RANGE_KMS,
    _brackets,
    accumulate_velocity_spectra,
    linear_fill_bracketed_failures,
)
from quokka2s.pipeline.prep import config as cfg
from quokka2s.pipeline.spectrum_units import (
    DSIGMA_DV_UNIT,
    SPEED_OF_LIGHT_CGS,
    dsigma_dv_ylabel,
)


STATE_ORDER = ("hm2012", "hm2012_draine", "hm2012_draine_cr")
STATE_LABELS = {
    "hm2012": "Cloudy HM2012",
    "hm2012_draine": "Cloudy HM2012 + Draine",
    "hm2012_draine_cr": "Cloudy HM2012 + Draine + CR",
    "draine_cr_no_hm": "Cloudy Draine + CR",
    "draine_only": "Cloudy Draine only",
    "cr_only": "Cloudy CR only",
}
STATE_COLORS = {
    "hm2012": "#9467BD",
    "hm2012_draine": "#D55E00",
    "hm2012_draine_cr": "#009E73",
    "draine_cr_no_hm": "#CC79A7",
    "draine_only": "#56B4E9",
    "cr_only": "#E69F00",
}
REGIME_KEYS = ("T_QUOKKA_lt_3000K", "T_QUOKKA_ge_3000K")


def _plot(
    output: Path,
    velocity: np.ndarray,
    curves: np.ndarray,
    keys: tuple[str, ...],
    labels: tuple[str, ...],
    styles: tuple[tuple[str, str], ...],
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    shared_max = float(np.nanmax(curves))
    for branch, axis in enumerate(axes):
        for index, (key, label, style) in enumerate(zip(keys, labels, styles)):
            axis.plot(
                velocity, curves[index, branch], color=style[0],
                linestyle=style[1], linewidth=1.75, drawstyle="steps-mid",
                label=label,
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
        axis.legend(fontsize=7.4 if len(keys) > 5 else 8.5, frameon=False)
        axis.ticklabel_format(
            style="sci", axis="y", scilimits=(0, 0), useMathText=True,
        )
        axis.tick_params(axis="y", labelleft=True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _prepare_jeans(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as table:
        log_nh = np.asarray(table["log_nH"], dtype=float)
        log_t = np.asarray(table["log_T"], dtype=float)
        raw = np.asarray(table["log_emissivity_per_nH2"], dtype=float)
        coefficient = np.asarray(table["emissivity_per_nH2"], dtype=float)
        failure = np.asarray(table["failure_mask"], dtype=bool)
        zero = np.asarray(table["zero_mask"], dtype=bool)
    expanded = tuple(
        value[None, None, :, None, :]
        for value in (raw, coefficient, failure, zero)
    )
    filled_log, filled_coefficient, remaining, records = (
        linear_fill_bracketed_failures(
            expanded[0], expanded[1], expanded[2], expanded[3],
            (log_nh, np.asarray([0.0]), log_t),
        )
    )
    return {
        "log_nH": log_nh,
        "log_T": log_t,
        "log": filled_log[0, 0, :, 0],
        "coefficient": filled_coefficient[0, 0, :, 0],
        "remaining": remaining[0, 0, :, 0],
        "failure": failure,
        "zero": zero,
        "fill_records": records,
    }


def _failure_weight_2d(
    failure: np.ndarray,
    n_bracket: tuple[np.ndarray, np.ndarray, np.ndarray],
    t_bracket: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    weight_sum = np.zeros(n_bracket[2].size)
    for ni in (0, 1):
        nw = n_bracket[2] if ni else 1.0 - n_bracket[2]
        for ti in (0, 1):
            tw = t_bracket[2] if ti else 1.0 - t_bracket[2]
            weight_sum += failure[n_bracket[ni], t_bracket[ti]] * nw * tw
    return weight_sum


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = Path(cfg.OUTPUT_DIR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument("--despotic-table", type=Path, default=Path(cfg.DESPOTIC_TABLE_PATH))
    parser.add_argument(
        "--despotic-spectrum-source", type=Path,
        default=output / "CII_GOW_vs_GOWCR_explicit_column_NH_and_NHdiv2_Tsplit_Rinf_nozoom.npz",
    )
    for state in STATE_ORDER:
        parser.add_argument(f"--column-{state.replace('_', '-')}", type=Path, required=True)
        parser.add_argument(f"--jeans-{state.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--column-draine-cr-no-hm", type=Path)
    parser.add_argument("--jeans-draine-cr-no-hm", type=Path)
    parser.add_argument("--column-draine-only", type=Path)
    parser.add_argument("--jeans-draine-only", type=Path)
    parser.add_argument("--column-cr-only", type=Path)
    parser.add_argument("--jeans-cr-only", type=Path)
    parser.add_argument("--nh-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument("--slab-nz", type=int, default=64)
    parser.add_argument("--cell-chunk", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in ("dataset", "despotic_table", "despotic_spectrum_source", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    optional_pairs = (
        ("draine_cr_no_hm", args.column_draine_cr_no_hm, args.jeans_draine_cr_no_hm),
        ("draine_only", args.column_draine_only, args.jeans_draine_only),
        ("cr_only", args.column_cr_only, args.jeans_cr_only),
    )
    for state, column_path, jeans_path in optional_pairs:
        if (column_path is None) != (jeans_path is None):
            raise ValueError(f"provide both {state} tables or neither")
    state_order = STATE_ORDER
    state_order += tuple(state for state, column_path, _ in optional_pairs if column_path is not None)
    optional_column_paths = {state: path for state, path, _ in optional_pairs}
    optional_jeans_paths = {state: path for state, _, path in optional_pairs}
    column_paths = {
        state: (
            optional_column_paths[state] if state in optional_column_paths
            else getattr(args, f"column_{state}")
        ).resolve() for state in state_order
    }
    jeans_paths = {
        state: (
            optional_jeans_paths[state] if state in optional_jeans_paths
            else getattr(args, f"jeans_{state}")
        ).resolve() for state in state_order
    }
    column_tables = {state: _prepare_table(path) for state, path in column_paths.items()}
    jeans_tables = {state: _prepare_jeans(path) for state, path in jeans_paths.items()}

    ds = yt.load(str(args.dataset))
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    nx, ny, nz = dimensions
    column_file, column_cache_path = _open_validated_cache(
        args.dataset, args.despotic_table, COLUMN_FIELD, None, dimensions,
    )
    tdsp_file, tdsp_cache_path = _open_validated_cache(
        args.dataset, args.despotic_table, TDSP_FIELD, None, dimensions,
    )
    cell_width_cm = np.asarray(ds.domain_width.to("cm") / ds.domain_dimensions, dtype=float)
    cell_volume_cm3 = float(np.prod(cell_width_cm))
    projected_area_cm2 = float(nx * nz * cell_width_cm[0] * cell_width_cm[2])
    hydrogen_mass_g = float(mh.to_value("g"))
    carbon_mass_g = 12.01 * 1.66053906660e-24
    boltzmann_cgs = float(kb.to_value("erg/K"))
    c_kms = float(SPEED_OF_LIGHT_CGS.to_value("cm/s")) / 1.0e5
    velocity_edges = np.linspace(-V_RANGE_KMS, V_RANGE_KMS, N_CHANNELS + 1)
    velocity = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])
    column_scales = (1.0,) if args.nh_only else (1.0, 0.5)
    column_specs = tuple((state, scale) for state in state_order for scale in column_scales)
    model_specs = column_specs + tuple((state, None) for state in state_order)
    accumulated = np.zeros((len(model_specs), 2, N_CHANNELS))
    counts = {
        "all_cells": 0,
        "T_QUOKKA_lt_3000_cells": 0,
        "T_QUOKKA_ge_3000_cells": 0,
        "original_failure_touches": {
            f"column_{state}_{'NH' if scale == 1.0 else 'NHdiv2'}": 0
            for state, scale in column_specs
        } | {f"jeans_{state}": 0 for state in state_order},
    }
    started = time.perf_counter()
    total_slabs = (nz + args.slab_nz - 1) // args.slab_nz
    try:
        for slab_number, iz in enumerate(range(0, nz, args.slab_nz), start=1):
            local_nz = min(args.slab_nz, nz - iz)
            left_edge = ds.domain_left_edge.copy()
            left_edge[2] += iz * (ds.domain_width[2] / ds.domain_dimensions[2])
            grid = ds.covering_grid(
                level=ds.max_level, left_edge=left_edge, dims=(nx, ny, local_nz),
            )
            density = np.asarray(grid[("gas", "density")].to("g/cm**3"), dtype=float).reshape(-1)
            temperature_qk = np.asarray(grid[("boxlib", "temperature")], dtype=float).reshape(-1)
            velocity_y = np.asarray(grid[("gas", "velocity_y")].to("km/s"), dtype=float).reshape(-1)
            del grid
            n_h_all = density * float(cfg.X_H) / hydrogen_mass_g
            column_all = np.asarray(column_file["data"][:, :, iz:iz + local_nz], dtype=float).reshape(-1)
            tdsp_all = np.asarray(tdsp_file["data"][:, :, iz:iz + local_nz], dtype=float).reshape(-1)
            low = temperature_qk < REGIME_SPLIT_K
            counts["all_cells"] += int(low.size)
            counts["T_QUOKKA_lt_3000_cells"] += int(np.count_nonzero(low))
            counts["T_QUOKKA_ge_3000_cells"] += int(np.count_nonzero(~low))

            for branch, selected in enumerate((low, ~low)):
                if not np.any(selected):
                    continue
                n_h = n_h_all[selected]
                base_column = column_all[selected]
                selected_velocity = velocity_y[selected]
                temperature = tdsp_all[selected] if branch == 0 else temperature_qk[selected]
                luminosity = np.zeros((n_h.size, len(model_specs)))
                for model_index, (state, scale) in enumerate(model_specs):
                    if scale is not None:
                        table = column_tables[state]
                        axes = table["axes"]
                        lookup_t = np.clip(temperature, 10.0 ** axes[2][0], 10.0 ** axes[2][-1])
                        coordinates = (np.log10(n_h), np.log10(base_column * scale), np.log10(lookup_t))
                        brackets = tuple(_brackets(axis, values) for axis, values in zip(axes, coordinates))
                        original_weight = _failure_support_weights(
                            np.asarray(table["bundle"]["failure_mask"], dtype=bool)[:, 0], brackets,
                        )[0]
                        key = f"column_{state}_{'NH' if scale == 1.0 else 'NHdiv2'}"
                        counts["original_failure_touches"][key] += int(np.count_nonzero(original_weight > TOUCH_EPS))
                        coefficient = _interpolate_selected_cii(
                            table["filled_log"], table["filled_coefficient"],
                            table["remaining_failure"],
                            np.asarray(table["bundle"]["zero_mask"], dtype=bool), brackets,
                        )[0]
                    else:
                        table = jeans_tables[state]
                        lookup_t = np.clip(temperature, 10.0 ** table["log_T"][0], 10.0 ** table["log_T"][-1])
                        n_bracket = _brackets_2d(table["log_nH"], np.log10(n_h))
                        t_bracket = _brackets_2d(table["log_T"], np.log10(lookup_t))
                        original_weight = _failure_weight_2d(table["failure"], n_bracket, t_bracket)
                        counts["original_failure_touches"][f"jeans_{state}"] += int(
                            np.count_nonzero(original_weight > TOUCH_EPS)
                        )
                        coefficient, remaining_weight = _interpolate_2d(
                            table["log"], table["coefficient"], table["remaining"],
                            table["zero"], n_bracket, t_bracket,
                        )
                        if np.any(remaining_weight > TOUCH_EPS):
                            raise RuntimeError(f"simulation touches unfilled Jeans failure: {state}")
                    luminosity[:, model_index] = coefficient * np.square(n_h) * cell_volume_cm3
                thermal_width = np.sqrt(boltzmann_cgs * temperature / carbon_mass_g) / 1.0e5
                thermal_width *= 1.0 - selected_velocity / c_kms
                accumulated[:, branch] += accumulate_velocity_spectra(
                    selected_velocity, thermal_width, luminosity, velocity_edges,
                    cell_chunk=args.cell_chunk, workers=args.workers,
                ).T
            print(f"[{slab_number:02d}/{total_slabs:02d}] elapsed={(time.perf_counter()-started)/60:.2f} min", flush=True)
    finally:
        column_file.close()
        tdsp_file.close()

    cloudy = YTArray(accumulated / projected_area_cm2, "erg/s/cm**2/(km/s)").to(DSIGMA_DV_UNIT).value
    old_velocity, despotic = _load_despotic(args.despotic_spectrum_source)
    if not np.allclose(velocity, old_velocity):
        raise ValueError("velocity axes differ")
    column_curves = np.concatenate((despotic[None], cloudy[:len(column_specs)]))
    jeans_curves = np.concatenate((despotic[None], cloudy[len(column_specs):]))
    column_keys = ("despotic",) + tuple(
        f"{state}_{'NH' if scale == 1.0 else 'NHdiv2'}" for state, scale in column_specs
    )
    column_labels = ("DESPOTIC",) + tuple(
        STATE_LABELS[state] + (
            "" if args.nh_only else
            (r", $N_{\rm H}$" if scale == 1.0 else r", $N_{\rm H}/2$")
        ) for state, scale in column_specs
    )
    column_styles = (("#0072B2", "--"),) + tuple(
        (STATE_COLORS[state], "-" if scale == 1.0 else ":") for state, scale in column_specs
    )
    jeans_keys = ("despotic",) + state_order
    jeans_labels = ("DESPOTIC",) + tuple(STATE_LABELS[state] for state in state_order)
    jeans_styles = (("#0072B2", "--"),) + tuple((STATE_COLORS[state], "-") for state in state_order)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    optional_suffixes = {
        "draine_cr_no_hm": "DraineCR",
        "draine_only": "DraineOnly",
        "cr_only": "CROnly",
    }
    selected_suffixes = [optional_suffixes[state] for state in state_order if state in optional_suffixes]
    suffix = "_with_" + "_".join(selected_suffixes) + "_noHM" if selected_suffixes else ""
    column_scope = "NHonly" if args.nh_only else "NH_and_NHdiv2"
    column_stem = f"CII_default_abund_radiation_comparison_column_{column_scope}_Tsplit_Rinf_nozoom{suffix}"
    jeans_stem = f"CII_default_abund_radiation_comparison_JeansLength_Tsplit_Rinf_nozoom{suffix}"
    column_png = args.output_dir / f"{column_stem}.png"
    jeans_png = args.output_dir / f"{jeans_stem}.png"
    for path in (column_png, jeans_png):
        if path.exists() and not args.force:
            raise FileExistsError(f"output exists; pass --force: {path}")
    _plot(column_png, velocity, column_curves, column_keys, column_labels, column_styles,
          r"[C II] 158 $\mu$m ($N_{\rm H}, n_{\rm H}, T$), LOS y, $R=\infty$")
    _plot(jeans_png, velocity, jeans_curves, jeans_keys, jeans_labels, jeans_styles,
          r"[C II] 158 $\mu$m ($n_{\rm H}, T$; Jeans length), LOS y, $R=\infty$")
    for stem, curves, keys in ((column_stem, column_curves, column_keys), (jeans_stem, jeans_curves, jeans_keys)):
        np.savez_compressed(
            args.output_dir / f"{stem}.npz", velocity_kms=velocity, dsigma_dv=curves,
            model_keys=np.asarray(keys), regime_keys=np.asarray(REGIME_KEYS),
            dsigma_dv_units=np.asarray(DSIGMA_DV_UNIT), completed_full_domain=np.asarray(True),
        )
    metadata = {
        "dataset": str(args.dataset),
        "composition": "Cloudy 17.02 default.abn; no element abundance overrides",
        "states": {
            "hm2012": "HM2012",
            "hm2012_draine": "HM2012 + table Draine",
            "hm2012_draine_cr": "HM2012 + table Draine + cosmic rays rate 2e-17 s^-1",
            **({"draine_cr_no_hm": "table Draine + cosmic rays rate 2e-17 s^-1; no HM2012"}
               if "draine_cr_no_hm" in state_order else {}),
            **({"draine_only": "table Draine only; no HM2012 and no explicit cosmic rays"}
               if "draine_only" in state_order else {}),
            **({"cr_only": "cosmic rays rate 2e-17 s^-1; no incident radiation and no HM2012"}
               if "cr_only" in state_order else {}),
        },
        "column_tables": {state: str(path) for state, path in column_paths.items()},
        "jeans_tables": {state: str(path) for state, path in jeans_paths.items()},
        "column_cache": str(column_cache_path),
        "temperature_despotic_cache": str(tdsp_cache_path),
        "temperature_policy": "T_QUOKKA split; T_DESPOTIC lookup below 3000 K; T_QUOKKA otherwise",
        "failure_policy": "linear fill only for bracketed failed grid nodes",
        "counts": counts,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    (args.output_dir / f"CII_default_abund_radiation_comparison_Tsplit_Rinf_nozoom{suffix}.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Saved: {column_png}")
    print(f"Saved: {jeans_png}")


if __name__ == "__main__":
    main()
