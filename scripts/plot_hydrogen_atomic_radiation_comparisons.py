"""Plot H-alpha and H I 21-cm spectra for three Cloudy radiation states.

The six Cloudy grids contain both lines, so this script streams the simulation
only once and accumulates both line spectra for the explicit-column and Jeans-
length geometries together.  Cells are split by T_QUOKKA; the low-temperature
branch looks up Cloudy with T_DESPOTIC and the high-temperature branch with
T_QUOKKA, matching the established C II comparison policy.
"""
from __future__ import annotations

import argparse
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

from plot_cii_cloudy_tdsp_split_spectra import (
    COLUMN_FIELD,
    TDSP_FIELD,
    _failure_support_weights,
    _interpolate_selected_cii,
    _open_validated_cache,
)
from plot_cii_defaultabund_radiation_cr_comparisons import (
    STATE_COLORS,
    STATE_LABELS,
    STATE_ORDER,
    _failure_weight_2d,
    _prepare_jeans,
)
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


SPECIES = ("halpha", "hi21")
SPECIES_TITLES = {"halpha": r"H$\alpha$", "hi21": "H I 21 cm"}
REGIME_KEYS = ("T_QUOKKA_lt_3000K", "T_QUOKKA_ge_3000K")


def _prepare_column(path: Path) -> dict[str, object]:
    """Load one generic single-line column view into the shared 5-D layout."""
    with np.load(path, allow_pickle=False) as table:
        axes = tuple(
            np.asarray(table[name], dtype=float)
            for name in ("log_nH", "log_NH", "log_T")
        )
        raw = np.asarray(table["log_emissivity_per_nH2"], dtype=float)
        coefficient = np.asarray(table["emissivity_per_nH2"], dtype=float)
        failure = np.asarray(table["failure_mask"], dtype=bool)
        zero = np.asarray(table["zero_mask"], dtype=bool)
    expected = tuple(axis.size for axis in axes)
    if raw.shape != expected:
        raise ValueError(f"unexpected column table shape {raw.shape}: {path}")
    bundle = {
        "failure_mask": failure[None, None],
        "zero_mask": zero[None, None],
    }
    filled_log, filled_coefficient, remaining_failure, records = (
        linear_fill_bracketed_failures(
            raw[None, None], coefficient[None, None],
            failure[None, None], zero[None, None], axes,
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


def _orient(source_v: np.ndarray, source: np.ndarray, target_v: np.ndarray) -> np.ndarray:
    if np.allclose(source_v, target_v, rtol=0.0, atol=1.0e-10):
        return source
    if np.allclose(source_v[::-1], target_v, rtol=0.0, atol=1.0e-10):
        return source[::-1]
    raise ValueError("reference and Cloudy velocity axes differ")


def _load_halpha_reference(path: Path, target_v: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as table:
        velocity = np.asarray(table["velocity_kms"], dtype=float)
        spectra = np.asarray(table["dsigma_dv"], dtype=float)
        models = tuple(str(value) for value in table["model_keys"].tolist())
        units = str(np.asarray(table["dsigma_dv_units"]).item())
    converted = YTArray(spectra, units).to(DSIGMA_DV_UNIT).d
    # Canonical pipeline: DESPOTIC chemistry below 3000 K and QUOKKA's
    # mean-molecular-weight electron fraction at and above 3000 K.
    reference = np.stack((
        converted[models.index("despotic"), 0],
        converted[models.index("quokka"), 1],
    ))
    return np.stack(tuple(_orient(velocity, curve, target_v) for curve in reference))


def _load_hi_reference(path: Path, target_v: np.ndarray) -> np.ndarray:
    curves = []
    with h5py.File(path, "r") as handle:
        # Canonical pipeline: DESPOTIC n_HI below 3000 K and QUOKKA's
        # mean-molecular-weight neutral fraction at and above 3000 K.
        for group in ("HI_DESPOTIC_LOW", "HI_QUOKKA_HIGH"):
            base = handle["spectra"][group]["y"]
            velocity = np.asarray(base["v_axis"], dtype=float)
            raw = np.asarray(base["dsigma_dv"], dtype=float)
            units = str(base["dsigma_dv"].attrs["units"])
            converted = YTArray(raw, units).to(DSIGMA_DV_UNIT).d
            curves.append(_orient(velocity, converted, target_v))
    return np.stack(curves)


def _plot(path: Path, velocity: np.ndarray, curves: np.ndarray, title: str) -> None:
    # curves: [reference + three Cloudy states, low/high, channel]
    labels = ("pipeline",) + tuple(STATE_LABELS[state] for state in STATE_ORDER)
    styles = (("#0072B2", "--"),) + tuple(
        (STATE_COLORS[state], "-") for state in STATE_ORDER
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    shared_max = float(np.nanmax(curves))
    for branch, axis in enumerate(axes):
        for model, (label, style) in enumerate(zip(labels, styles)):
            axis.plot(
                velocity, curves[model, branch], color=style[0],
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
        axis.legend(fontsize=8.2, frameon=False)
        axis.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
        axis.tick_params(axis="y", labelleft=True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    data = root / "data/cloudy_atomic_defaultabund_radiation_3state_views"
    output = Path(cfg.OUTPUT_DIR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument("--despotic-table", type=Path, default=Path(cfg.DESPOTIC_TABLE_PATH))
    parser.add_argument("--table-dir", type=Path, default=data)
    parser.add_argument("--halpha-reference", type=Path, required=True)
    parser.add_argument(
        "--hi-reference", type=Path,
        default=output / "task_intermediates/Build_HICloudyComparison_9bca5349.h5",
    )
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument("--slab-nz", type=int, default=32)
    parser.add_argument("--cell-chunk", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--replot-existing",
        action="store_true",
        help="reuse existing Cloudy spectra and replace only their reference curve",
    )
    return parser.parse_args()


def _replot_existing(args: argparse.Namespace) -> None:
    references = {}
    for species in SPECIES:
        source = args.output_dir / (
            f"{species}_default_abund_radiation_comparison_"
            "column_Tsplit_Rinf_nozoom.npz"
        )
        with np.load(source, allow_pickle=False) as table:
            velocity = np.asarray(table["velocity_kms"], dtype=float)
        references[species] = (
            _load_halpha_reference(args.halpha_reference, velocity)
            if species == "halpha"
            else _load_hi_reference(args.hi_reference, velocity)
        )

    for species in SPECIES:
        for suffix, title_geometry in (
            ("column", r"$(N_{\rm H}, n_{\rm H}, T)$"),
            ("JeansLength", r"$(n_{\rm H}, T;\ \mathrm{Jeans\ length})$"),
        ):
            source = args.output_dir / (
                f"{species}_default_abund_radiation_comparison_"
                f"{suffix}_Tsplit_Rinf_nozoom.npz"
            )
            with np.load(source, allow_pickle=False) as table:
                velocity = np.asarray(table["velocity_kms"], dtype=float)
                old_curves = np.asarray(table["dsigma_dv"], dtype=float)
                units = str(np.asarray(table["dsigma_dv_units"]).item())
            cloudy = YTArray(old_curves[1:], units).to(DSIGMA_DV_UNIT).d
            curves = np.concatenate((references[species][None], cloudy))
            stem = (
                f"{species}_pipeline_vs_default_abund_radiation_"
                f"{suffix}_Tsplit_Rinf_nozoom"
            )
            png = args.output_dir / f"{stem}.png"
            npz = args.output_dir / f"{stem}.npz"
            if (png.exists() or npz.exists()) and not args.force:
                raise FileExistsError(f"output exists; pass --force: {png}")
            _plot(
                png, velocity, curves,
                rf"{SPECIES_TITLES[species]} {title_geometry}, LOS y, $R=\infty$",
            )
            np.savez_compressed(
                npz, velocity_kms=velocity, dsigma_dv=curves,
                model_keys=np.asarray(("pipeline",) + STATE_ORDER),
                regime_keys=np.asarray(REGIME_KEYS),
                dsigma_dv_units=np.asarray(DSIGMA_DV_UNIT),
                completed_full_domain=np.asarray(True),
                reference_policy=np.asarray(
                    "DESPOTIC below 3000 K; QUOKKA mean-molecular-weight branch otherwise"
                ),
            )
            print(f"Saved: {png}")


def main() -> None:
    args = _args()
    for name in ("dataset", "despotic_table", "table_dir", "halpha_reference", "hi_reference", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.replot_existing:
        _replot_existing(args)
        return

    column_tables = {
        species: {
            state: _prepare_column(args.table_dir / f"cloudy_{species}_{state}_column_10x10x21.npz")
            for state in STATE_ORDER
        } for species in SPECIES
    }
    jeans_tables = {
        species: {
            state: _prepare_jeans(args.table_dir / f"cloudy_{species}_{state}_jeans_10x21.npz")
            for state in STATE_ORDER
        } for species in SPECIES
    }

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
    line_mass_g = 1.00794 * 1.66053906660e-24
    boltzmann_cgs = float(kb.to_value("erg/K"))
    c_kms = float(SPEED_OF_LIGHT_CGS.to_value("cm/s")) / 1.0e5
    velocity_edges = np.linspace(-V_RANGE_KMS, V_RANGE_KMS, N_CHANNELS + 1)
    velocity = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])
    model_specs = tuple(("column", state) for state in STATE_ORDER) + tuple(
        ("jeans", state) for state in STATE_ORDER
    )
    accumulated = np.zeros((len(SPECIES), len(model_specs), 2, N_CHANNELS))
    counts = {
        "all_cells": 0,
        "T_QUOKKA_lt_3000_cells": 0,
        "T_QUOKKA_ge_3000_cells": 0,
        "original_failure_touches": {
            f"{species}_{geometry}_{state}": 0
            for species in SPECIES for geometry, state in model_specs
        },
    }
    started = time.perf_counter()
    total_slabs = (nz + args.slab_nz - 1) // args.slab_nz
    try:
        for slab_number, iz in enumerate(range(0, nz, args.slab_nz), start=1):
            local_nz = min(args.slab_nz, nz - iz)
            left_edge = ds.domain_left_edge.copy()
            left_edge[2] += iz * (ds.domain_width[2] / ds.domain_dimensions[2])
            grid = ds.covering_grid(level=ds.max_level, left_edge=left_edge, dims=(nx, ny, local_nz))
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
                column = column_all[selected]
                selected_velocity = velocity_y[selected]
                temperature = tdsp_all[selected] if branch == 0 else temperature_qk[selected]
                luminosity = np.zeros((n_h.size, len(SPECIES) * len(model_specs)))
                for species_index, species in enumerate(SPECIES):
                    for model_index, (geometry, state) in enumerate(model_specs):
                        if geometry == "column":
                            table = column_tables[species][state]
                            axes = table["axes"]
                            lookup_t = np.clip(temperature, 10.0 ** axes[2][0], 10.0 ** axes[2][-1])
                            coordinates = (np.log10(n_h), np.log10(column), np.log10(lookup_t))
                            brackets = tuple(_brackets(axis, values) for axis, values in zip(axes, coordinates))
                            original_weight = _failure_support_weights(
                                np.asarray(table["bundle"]["failure_mask"], dtype=bool)[:, 0], brackets,
                            )[0]
                            coefficient = _interpolate_selected_cii(
                                table["filled_log"], table["filled_coefficient"],
                                table["remaining_failure"],
                                np.asarray(table["bundle"]["zero_mask"], dtype=bool), brackets,
                            )[0]
                        else:
                            table = jeans_tables[species][state]
                            lookup_t = np.clip(temperature, 10.0 ** table["log_T"][0], 10.0 ** table["log_T"][-1])
                            n_bracket = _brackets_2d(table["log_nH"], np.log10(n_h))
                            t_bracket = _brackets_2d(table["log_T"], np.log10(lookup_t))
                            original_weight = _failure_weight_2d(table["failure"], n_bracket, t_bracket)
                            coefficient, remaining_weight = _interpolate_2d(
                                table["log"], table["coefficient"], table["remaining"],
                                table["zero"], n_bracket, t_bracket,
                            )
                            if np.any(remaining_weight > TOUCH_EPS):
                                raise RuntimeError(f"simulation touches unfilled {species} Jeans failure: {state}")
                        counts["original_failure_touches"][f"{species}_{geometry}_{state}"] += int(
                            np.count_nonzero(original_weight > TOUCH_EPS)
                        )
                        column_index = species_index * len(model_specs) + model_index
                        luminosity[:, column_index] = coefficient * np.square(n_h) * cell_volume_cm3
                thermal_width = np.sqrt(boltzmann_cgs * temperature / line_mass_g) / 1.0e5
                thermal_width *= 1.0 - selected_velocity / c_kms
                slab_spectra = accumulate_velocity_spectra(
                    selected_velocity, thermal_width, luminosity, velocity_edges,
                    cell_chunk=args.cell_chunk, workers=args.workers,
                ).T.reshape(len(SPECIES), len(model_specs), N_CHANNELS)
                accumulated[:, :, branch] += slab_spectra
            print(f"[{slab_number:02d}/{total_slabs:02d}] elapsed={(time.perf_counter()-started)/60:.2f} min", flush=True)
    finally:
        column_file.close()
        tdsp_file.close()

    cloudy = YTArray(accumulated / projected_area_cm2, "erg/s/cm**2/(km/s)").to(DSIGMA_DV_UNIT).d
    references = {
        "halpha": _load_halpha_reference(args.halpha_reference, velocity),
        "hi21": _load_hi_reference(args.hi_reference, velocity),
    }
    outputs = {}
    for species_index, species in enumerate(SPECIES):
        for geometry, offset, suffix, title_geometry in (
            ("column", 0, "column", r"$(N_{\rm H}, n_{\rm H}, T)$"),
            ("jeans", len(STATE_ORDER), "JeansLength", r"$(n_{\rm H}, T;\ \mathrm{Jeans\ length})$"),
        ):
            curves = np.concatenate((references[species][None], cloudy[species_index, offset:offset + len(STATE_ORDER)]))
            stem = f"{species}_default_abund_radiation_comparison_{suffix}_Tsplit_Rinf_nozoom"
            png = args.output_dir / f"{stem}.png"
            npz = args.output_dir / f"{stem}.npz"
            if (png.exists() or npz.exists()) and not args.force:
                raise FileExistsError(f"output exists; pass --force: {png}")
            _plot(
                png, velocity, curves,
                rf"{SPECIES_TITLES[species]} {title_geometry}, LOS y, $R=\infty$",
            )
            np.savez_compressed(
                npz, velocity_kms=velocity, dsigma_dv=curves,
                model_keys=np.asarray(("pipeline",) + STATE_ORDER),
                regime_keys=np.asarray(REGIME_KEYS),
                dsigma_dv_units=np.asarray(DSIGMA_DV_UNIT),
                completed_full_domain=np.asarray(True),
            )
            outputs[f"{species}_{geometry}"] = str(png)

    metadata = {
        "dataset": str(args.dataset),
        "temperature_policy": "T_QUOKKA split; T_DESPOTIC lookup below 3000 K; T_QUOKKA otherwise",
        "cloudy_states": list(STATE_ORDER),
        "failure_policy": "linear fill only for bracketed failed table nodes",
        "counts": counts,
        "column_cache": str(column_cache_path),
        "temperature_despotic_cache": str(tdsp_cache_path),
        "outputs": outputs,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    report = args.output_dir / "hydrogen_default_abund_radiation_comparison_Tsplit_Rinf_nozoom.json"
    report.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved: {report}")


if __name__ == "__main__":
    main()
