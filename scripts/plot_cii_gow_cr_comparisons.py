"""Plot matched GOW versus GOW+CR CII spectra for column and Jeans inputs."""
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


COLUMN_KEYS = (
    "despotic",
    "cloudy_gow_nh",
    "cloudy_gow_nh2",
    "cloudy_gow_cr_nh",
    "cloudy_gow_cr_nh2",
)
COLUMN_LABELS = {
    "despotic": "DESPOTIC",
    "cloudy_gow_nh": r"Cloudy HM2012 + Draine + GOW, $N_{\rm H}$",
    "cloudy_gow_nh2": r"Cloudy HM2012 + Draine + GOW, $N_{\rm H}/2$",
    "cloudy_gow_cr_nh": r"Cloudy HM2012 + Draine + GOW + CR, $N_{\rm H}$",
    "cloudy_gow_cr_nh2": r"Cloudy HM2012 + Draine + GOW + CR, $N_{\rm H}/2$",
}
JEANS_KEYS = (
    "despotic",
    "cloudy_jeans_draine_gow_only",
    "cloudy_jeans_draine_gow_cr",
)
JEANS_LABELS = {
    "despotic": "DESPOTIC",
    "cloudy_jeans_draine_gow_only": "Cloudy HM2012 + Draine + GOW",
    "cloudy_jeans_draine_gow_cr": "Cloudy HM2012 + Draine + GOW + CR",
}
STYLES = {
    "despotic": ("#0072B2", "--"),
    "cloudy_gow_nh": ("#D55E00", "-"),
    "cloudy_gow_nh2": ("#D55E00", ":"),
    "cloudy_gow_cr_nh": ("#009E73", "-"),
    "cloudy_gow_cr_nh2": ("#009E73", ":"),
    "cloudy_jeans_draine_gow_only": ("#D55E00", "-"),
    "cloudy_jeans_draine_gow_cr": ("#009E73", "-"),
}
REGIME_KEYS = ("T_QUOKKA_lt_3000K", "T_QUOKKA_ge_3000K")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plot(
    path: Path,
    velocity: np.ndarray,
    curves: np.ndarray,
    keys: tuple[str, ...],
    labels: dict[str, str],
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    shared_max = float(np.nanmax(curves))
    for branch, axis in enumerate(axes):
        for index, key in enumerate(keys):
            color, linestyle = STYLES[key]
            axis.plot(
                velocity,
                curves[index, branch],
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                drawstyle="steps-mid",
                label=labels[key],
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
        axis.legend(fontsize=7.7 if len(keys) > 3 else 8.7, frameon=False)
        axis.ticklabel_format(
            style="sci", axis="y", scilimits=(0, 0), useMathText=True,
        )
        axis.tick_params(axis="y", labelleft=True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _load_despotic(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as table:
        velocity = np.asarray(table["velocity_kms"], dtype=float)
        curves = np.asarray(table["dsigma_dv"], dtype=float)
        models = tuple(str(value) for value in table["model_keys"].tolist())
        units = str(np.asarray(table["dsigma_dv_units"]).item())
    if "despotic" not in models:
        raise ValueError(f"invalid DESPOTIC spectrum source: {path}")
    try:
        converted = YTArray(curves, units).to(DSIGMA_DV_UNIT).d
    except Exception as exc:
        raise ValueError(
            f"DESPOTIC spectrum units {units!r} are not convertible to "
            f"{DSIGMA_DV_UNIT}: {path}"
        ) from exc
    return velocity, converted[models.index("despotic")]


def _prepare_table(path: Path) -> dict[str, object]:
    bundle = _load_single_cii_bundle(path)
    axes = tuple(
        np.asarray(bundle[name], dtype=float)
        for name in ("log_nH", "log_NH", "log_T")
    )
    filled_log, filled_coefficient, remaining_failure, records = (
        linear_fill_bracketed_failures(
            np.asarray(bundle["log_emissivity_per_nH2"], dtype=float),
            np.asarray(bundle["emissivity_per_nH2"], dtype=float),
            np.asarray(bundle["failure_mask"], dtype=bool),
            np.asarray(bundle["zero_mask"], dtype=bool),
            axes,
        )
    )
    return {
        "bundle": bundle,
        "axes": axes,
        "filled_log": filled_log,
        "filled_coefficient": filled_coefficient,
        "remaining_failure": remaining_failure,
        "fill_records": records,
    }


def _column_spectra(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    tables = {
        "gow": _prepare_table(args.column_gow_table),
        "gow_cr": _prepare_table(args.column_gow_cr_table),
    }
    for name, table in tables.items():
        if tuple(np.asarray(axis).shape for axis in table["axes"]) != (
            (10,), (10,), (21,),
        ):
            raise ValueError(f"unexpected {name} column-table axes")

    ds = yt.load(str(args.dataset))
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    nx, ny, nz = dimensions
    column_file, column_path = _open_validated_cache(
        args.dataset, args.despotic_table, COLUMN_FIELD, None, dimensions,
    )
    tdsp_file, tdsp_path = _open_validated_cache(
        args.dataset, args.despotic_table, TDSP_FIELD, None, dimensions,
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
    model_specs = (("gow", 1.0), ("gow", 0.5), ("gow_cr", 1.0), ("gow_cr", 0.5))
    accumulated = np.zeros((len(model_specs), 2, N_CHANNELS))
    counts = {
        "all_cells": 0,
        "T_QUOKKA_lt_3000_cells": 0,
        "T_QUOKKA_ge_3000_cells": 0,
        "original_failure_touches": {
            f"{name}_{'NH' if scale == 1.0 else 'NHdiv2'}": 0
            for name, scale in model_specs
        },
    }
    maxima = {key: 0.0 for key in counts["original_failure_touches"]}
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
            n_h_all = density * float(cfg.X_H) / hydrogen_mass_g
            column_all = np.asarray(
                column_file["data"][:, :, iz:iz + local_nz], dtype=float,
            ).reshape(-1)
            tdsp_all = np.asarray(
                tdsp_file["data"][:, :, iz:iz + local_nz], dtype=float,
            ).reshape(-1)
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
                for model_index, (table_name, scale) in enumerate(model_specs):
                    table = tables[table_name]
                    axes = table["axes"]
                    temperature_lookup = np.clip(
                        temperature, 10.0 ** axes[2][0], 10.0 ** axes[2][-1],
                    )
                    coordinates = (
                        np.log10(n_h),
                        np.log10(base_column * scale),
                        np.log10(temperature_lookup),
                    )
                    for axis, values, label in zip(axes, coordinates, ("nH", "NH", "T")):
                        if np.any(values < axis[0]) or np.any(values > axis[-1]):
                            raise ValueError(
                                f"{table_name} {label} outside table in slab {iz}:{iz + local_nz}"
                            )
                    brackets = tuple(
                        _brackets(axis, values) for axis, values in zip(axes, coordinates)
                    )
                    failure_weight = _failure_support_weights(
                        np.asarray(table["bundle"]["failure_mask"], dtype=bool)[:, 0],
                        brackets,
                    )[0]
                    failure_key = f"{table_name}_{'NH' if scale == 1.0 else 'NHdiv2'}"
                    counts["original_failure_touches"][failure_key] += int(
                        np.count_nonzero(failure_weight > TOUCH_EPS)
                    )
                    maxima[failure_key] = max(
                        maxima[failure_key], float(np.max(failure_weight)),
                    )
                    coefficient = _interpolate_selected_cii(
                        table["filled_log"],
                        table["filled_coefficient"],
                        table["remaining_failure"],
                        np.asarray(table["bundle"]["zero_mask"], dtype=bool),
                        brackets,
                    )[0]
                    luminosity[:, model_index] = coefficient * np.square(n_h) * cell_volume_cm3
                thermal_width = np.sqrt(
                    boltzmann_cgs * temperature / carbon_mass_g,
                ) / 1.0e5
                thermal_width *= 1.0 - selected_velocity / c_kms
                accumulated[:, branch] += accumulate_velocity_spectra(
                    selected_velocity,
                    thermal_width,
                    luminosity,
                    velocity_edges,
                    cell_chunk=args.cell_chunk,
                    workers=args.workers,
                ).T
            print(
                f"[{slab_number:02d}/{total_slabs:02d}] z={iz}:{iz + local_nz} "
                f"elapsed={(time.perf_counter() - started) / 60.0:.2f} min",
                flush=True,
            )
    finally:
        column_file.close()
        tdsp_file.close()

    cloudy = YTArray(
        accumulated / projected_area_cm2, "erg/s/cm**2/(km/s)",
    ).to(DSIGMA_DV_UNIT).value
    old_velocity, despotic = _load_despotic(args.despotic_spectrum_source)
    if not np.allclose(velocity, old_velocity):
        raise ValueError("DESPOTIC and reconstructed Cloudy velocity axes differ")
    curves = np.concatenate((despotic[None], cloudy), axis=0)
    details = {
        "dataset": str(args.dataset),
        "column_cache": str(column_path),
        "temperature_despotic_cache": str(tdsp_path),
        "column_gow_table": str(args.column_gow_table),
        "column_gow_cr_table": str(args.column_gow_cr_table),
        "column_gow_table_sha256": _sha256(args.column_gow_table),
        "column_gow_cr_table_sha256": _sha256(args.column_gow_cr_table),
        "temperature_policy": (
            "select with T_QUOKKA; lookup and broaden with T_DESPOTIC below "
            "3000 K, otherwise T_QUOKKA"
        ),
        "failure_policy": "linear fill only along bracketed log-temperature nodes",
        "linear_failure_fills": {
            name: len(table["fill_records"]) for name, table in tables.items()
        },
        "counts": counts,
        "maximum_original_failure_weight": maxima,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    return velocity, curves, details


def _jeans_spectra(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as table:
        velocity = np.asarray(table["velocity_kms"], dtype=float)
        all_curves = np.asarray(table["dsigma_dv"], dtype=float)
        models = tuple(str(value) for value in table["model_keys"].tolist())
    missing = [key for key in JEANS_KEYS if key not in models]
    if missing:
        raise ValueError(f"Jeans source lacks models: {missing}")
    return velocity, np.stack([all_curves[models.index(key)] for key in JEANS_KEYS])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = Path(cfg.OUTPUT_DIR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--column-gow-table", type=Path,
        default=root / "data/cloudy_cii_hm2012_plus_draine_z0_gowabund_column_10x10x21_T3p6_to1e9.npz",
    )
    parser.add_argument(
        "--column-gow-cr-table", type=Path,
        default=root / "data/cloudy_cii_hm2012_plus_draine_z0_gowabund_cr2e-17_column_10x10x21_T3p6_to1e9.npz",
    )
    parser.add_argument(
        "--jeans-source", type=Path,
        default=output / "CII_DESPOTIC_CloudyAblations_JeansDraine_GOW_CR_Grains_Tsplit_Rinf.npz",
    )
    parser.add_argument(
        "--despotic-spectrum-source", type=Path,
        default=output / "CII_DESPOTIC_CloudyNH_vs_NHdiv2_MolCT_Tsplit_CloudyLowUsesTDESPOTIC_Tmin3p6K_Rinf_DraineGOWCR.npz",
    )
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument("--despotic-table", type=Path, default=Path(cfg.DESPOTIC_TABLE_PATH))
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument("--slab-nz", type=int, default=64)
    parser.add_argument("--cell-chunk", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in (
        "column_gow_table", "column_gow_cr_table", "jeans_source",
        "despotic_spectrum_source", "dataset", "despotic_table", "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    column_stem = "CII_GOW_vs_GOWCR_explicit_column_NH_and_NHdiv2_Tsplit_Rinf_nozoom"
    jeans_stem = "CII_GOW_vs_GOWCR_JeansLength_Tsplit_Rinf_nozoom"
    paths = {
        "column_png": args.output_dir / f"{column_stem}.png",
        "column_npz": args.output_dir / f"{column_stem}.npz",
        "column_json": args.output_dir / f"{column_stem}.json",
        "jeans_png": args.output_dir / f"{jeans_stem}.png",
        "jeans_npz": args.output_dir / f"{jeans_stem}.npz",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.force:
        raise FileExistsError("outputs exist; pass --force:\n" + "\n".join(map(str, existing)))

    velocity, column_curves, details = _column_spectra(args)
    _plot(
        paths["column_png"], velocity, column_curves, COLUMN_KEYS, COLUMN_LABELS,
        r"[C II] 158 $\mu$m: explicit-column comparison, LOS y, $R=\infty$",
    )
    np.savez_compressed(
        paths["column_npz"], velocity_kms=velocity, dsigma_dv=column_curves,
        model_keys=np.asarray(COLUMN_KEYS), regime_keys=np.asarray(REGIME_KEYS),
        dsigma_dv_units=np.asarray(DSIGMA_DV_UNIT),
        completed_full_domain=np.asarray(True),
    )
    paths["column_json"].write_text(json.dumps(details, indent=2) + "\n")

    jeans_velocity, jeans_curves = _jeans_spectra(args.jeans_source)
    _plot(
        paths["jeans_png"], jeans_velocity, jeans_curves, JEANS_KEYS, JEANS_LABELS,
        r"[C II] 158 $\mu$m: Jeans-length comparison, LOS y, $R=\infty$",
    )
    np.savez_compressed(
        paths["jeans_npz"], velocity_kms=jeans_velocity, dsigma_dv=jeans_curves,
        model_keys=np.asarray(JEANS_KEYS), regime_keys=np.asarray(REGIME_KEYS),
        dsigma_dv_units=np.asarray(DSIGMA_DV_UNIT),
        completed_full_domain=np.asarray(True),
    )
    for path in paths.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
