"""Build the standard split-temperature H-alpha comparison for LOS z.

The layout and model set match the established LOS-y H-alpha/CII figures:
pipeline plus Cloudy HM2012, HM2012+Draine, and HM2012+Draine+CR, shown
separately below and above T_QUOKKA=3000 K.  Both explicit-column and
Jeans-length Cloudy geometries are produced.  All spectra are normalized by
the full x-y projected domain area and retain the existing R=infinity thermal
line-profile convention.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt
from yt.units.physical_constants import kb, mh
from yt.units.yt_array import YTArray

from plot_cii_cloudy_tdsp_split_spectra import (
    _interpolate_selected_cii,
)
from plot_cii_defaultabund_radiation_cr_comparisons import (
    STATE_COLORS,
    STATE_LABELS,
    STATE_ORDER,
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
)
from plot_halpha_huang_figure2_losz_check import _open_caches
from plot_hydrogen_atomic_radiation_comparisons import (
    _prepare_column,
    _prepare_jeans,
)
from quokka2s.line_regimes import electron_fraction_from_mean_molecular_weight
from quokka2s.pipeline.prep import config as cfg
from quokka2s.pipeline.prep.physics_fields import (
    _clip_to_table_domain,
    c,
    effective_halpha_recombination_coefficient,
    h,
    lambda_Halpha,
)
from quokka2s.pipeline.spectrum_units import (
    DSIGMA_DV_UNIT,
    SPEED_OF_LIGHT_CGS,
    dsigma_dv_ylabel,
)
from quokka2s.tables import load_table
from quokka2s.tables.lookup import TableLookup


MODEL_KEYS = ("pipeline",) + tuple(
    f"column_{state}" for state in STATE_ORDER
) + tuple(f"jeans_{state}" for state in STATE_ORDER)
REGIME_KEYS = ("T_QUOKKA_lt_3000K", "T_QUOKKA_ge_3000K")


def _plot(
    path: Path,
    velocity: np.ndarray,
    curves: np.ndarray,
    title: str,
) -> None:
    labels = ("pipeline",) + tuple(STATE_LABELS[state] for state in STATE_ORDER)
    styles = (("#0072B2", "--"),) + tuple(
        (STATE_COLORS[state], "-") for state in STATE_ORDER
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    shared_max = float(np.nanmax(curves))
    for branch, axis in enumerate(axes):
        for model, (label, style) in enumerate(zip(labels, styles)):
            axis.plot(
                velocity,
                curves[model, branch],
                color=style[0],
                linestyle=style[1],
                linewidth=1.75,
                drawstyle="steps-mid",
                label=label,
            )
        axis.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
        axis.set_xlabel(r"Velocity [km s$^{-1}$]")
        axis.set_ylabel(dsigma_dv_ylabel(DSIGMA_DV_UNIT))
        axis.set_title(
            r"$T_{\rm QUOKKA}<3000\,$K"
            if branch == 0
            else r"$T_{\rm QUOKKA}\geq3000\,$K"
        )
        axis.set_ylim(0.0, 1.05 * shared_max)
        axis.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        axis.legend(fontsize=8.2, frameon=False)
        axis.ticklabel_format(
            style="sci", axis="y", scilimits=(0, 0), useMathText=True,
        )
        axis.tick_params(axis="y", labelleft=True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    output = Path(cfg.OUTPUT_DIR) / "atomic3line_rerun"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument(
        "--despotic-table", type=Path, default=Path(cfg.DESPOTIC_TABLE_PATH),
    )
    parser.add_argument(
        "--table-dir", type=Path,
        default=root / "data/cloudy_atomic_defaultabund_radiation_3state_views",
    )
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument("--slab-nz", type=int, default=32)
    parser.add_argument("--cell-chunk", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _args()
    for name in ("dataset", "despotic_table", "table_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stems = {
        "column": "halpha_pipeline_vs_default_abund_radiation_column_Tsplit_Rinf_nozoom_LOSz",
        "jeans": "halpha_pipeline_vs_default_abund_radiation_JeansLength_Tsplit_Rinf_nozoom_LOSz",
    }
    paths = {
        key: (args.output_dir / f"{stem}.png", args.output_dir / f"{stem}.npz")
        for key, stem in stems.items()
    }
    existing = [path for pair in paths.values() for path in pair if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "refusing to overwrite LOS-z outputs; pass --force:\n"
            + "\n".join(str(path) for path in existing)
        )

    column_tables = {
        state: _prepare_column(
            args.table_dir / f"cloudy_halpha_{state}_column_10x10x21.npz"
        )
        for state in STATE_ORDER
    }
    jeans_tables = {
        state: _prepare_jeans(
            args.table_dir / f"cloudy_halpha_{state}_jeans_10x21.npz"
        )
        for state in STATE_ORDER
    }

    ds = yt.load(str(args.dataset))
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    nx, ny, nz = dimensions
    handles, cache_paths = _open_caches(args.dataset, args.despotic_table, dimensions)
    lookup = TableLookup(load_table(args.despotic_table))

    cell_width_cm = np.asarray(
        ds.domain_width.to("cm") / ds.domain_dimensions, dtype=float,
    )
    cell_volume_cm3 = float(np.prod(cell_width_cm))
    projected_area_cm2 = float(nx * ny * cell_width_cm[0] * cell_width_cm[1])
    hydrogen_mass_g = float(mh.to_value("g"))
    boltzmann_cgs = float(kb.to_value("erg/K"))
    photon_energy = float(((h * c) / lambda_Halpha).in_cgs().value)
    c_kms = float(SPEED_OF_LIGHT_CGS.to_value("cm/s")) / 1.0e5
    velocity_edges = np.linspace(-V_RANGE_KMS, V_RANGE_KMS, N_CHANNELS + 1)
    velocity = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])
    accumulated = np.zeros((len(MODEL_KEYS), 2, N_CHANNELS), dtype=float)
    counts = {
        "all_cells": 0,
        "T_QUOKKA_lt_3000_cells": 0,
        "T_QUOKKA_ge_3000_cells": 0,
    }
    started = time.perf_counter()
    n_slabs = (nz + args.slab_nz - 1) // args.slab_nz

    try:
        for slab_number, iz in enumerate(range(0, nz, args.slab_nz), start=1):
            local_nz = min(args.slab_nz, nz - iz)
            left_edge = ds.domain_left_edge.copy()
            left_edge[2] += iz * (ds.domain_width[2] / ds.domain_dimensions[2])
            grid = ds.covering_grid(
                level=ds.max_level,
                left_edge=left_edge,
                dims=(nx, ny, local_nz),
            )
            temperature_qk = np.asarray(
                grid[("boxlib", "temperature")], dtype=float,
            ).reshape(-1)
            density = np.asarray(
                grid[("gas", "density")].to("g/cm**3"), dtype=float,
            ).reshape(-1)
            velocity_z = np.asarray(
                grid[("gas", "velocity_z")].to("km/s"), dtype=float,
            ).reshape(-1)
            total_energy = np.asarray(
                grid[("gas", "total_energy_density")].to("erg/cm**3"),
                dtype=float,
            ).reshape(-1)
            kinetic_energy = np.asarray(
                grid[("gas", "kinetic_energy_density")].to("erg/cm**3"),
                dtype=float,
            ).reshape(-1)
            del grid

            n_h_all = density * float(cfg.X_H) / hydrogen_mass_g
            column_all = np.asarray(
                handles[("gas", "column_density_H")]["data"][:, :, iz:iz + local_nz],
                dtype=float,
            ).reshape(-1)
            dvdr_all = np.asarray(
                handles[("gas", "dVdr_lvg")]["data"][:, :, iz:iz + local_nz],
                dtype=float,
            ).reshape(-1)
            tdsp_all = np.asarray(
                handles[("gas", "temperature_despotic")]["data"][:, :, iz:iz + local_nz],
                dtype=float,
            ).reshape(-1)
            internal_energy = total_energy - kinetic_energy
            low = temperature_qk < REGIME_SPLIT_K
            counts["all_cells"] += int(low.size)
            counts["T_QUOKKA_lt_3000_cells"] += int(np.count_nonzero(low))
            counts["T_QUOKKA_ge_3000_cells"] += int(np.count_nonzero(~low))

            # Canonical pipeline H-alpha: DESPOTIC chemistry below the split,
            # QUOKKA mean-molecular-weight electron fraction above the split.
            safe = _clip_to_table_domain(lookup, n_h_all, column_all, dvdr_all)
            densities = lookup.number_densities(("e-", "H+"), *safe)
            n_e_dsp = np.nan_to_num(densities["e-"], nan=0.0)
            n_hp_dsp = np.nan_to_num(densities["H+"], nan=0.0)
            x_e = electron_fraction_from_mean_molecular_weight(
                internal_energy,
                density,
                temperature_qk,
                hydrogen_mass_g=hydrogen_mass_g,
                boltzmann_erg_K=boltzmann_cgs,
            )
            n_e_qk = x_e * n_h_all
            n_hp_qk = np.minimum(x_e, 1.0) * n_h_all

            for branch, selected in enumerate((low, ~low)):
                if not np.any(selected):
                    continue
                n_h = n_h_all[selected]
                column = column_all[selected]
                selected_velocity = velocity_z[selected]
                selected_temperature = (
                    tdsp_all[selected] if branch == 0 else temperature_qk[selected]
                )
                luminosity = np.zeros((n_h.size, len(MODEL_KEYS)), dtype=float)
                if branch == 0:
                    pipeline_epsilon = (
                        photon_energy
                        * effective_halpha_recombination_coefficient(selected_temperature)
                        * n_e_dsp[selected]
                        * n_hp_dsp[selected]
                    )
                else:
                    pipeline_epsilon = (
                        photon_energy
                        * effective_halpha_recombination_coefficient(selected_temperature)
                        * n_e_qk[selected]
                        * n_hp_qk[selected]
                    )
                luminosity[:, 0] = pipeline_epsilon * cell_volume_cm3

                for state_index, state in enumerate(STATE_ORDER):
                    table = column_tables[state]
                    axes = table["axes"]
                    lookup_t = np.clip(
                        selected_temperature,
                        10.0 ** axes[2][0],
                        10.0 ** axes[2][-1],
                    )
                    coordinates = (
                        np.log10(n_h), np.log10(column), np.log10(lookup_t),
                    )
                    brackets = tuple(
                        _brackets(axis, values)
                        for axis, values in zip(axes, coordinates)
                    )
                    coefficient = _interpolate_selected_cii(
                        table["filled_log"],
                        table["filled_coefficient"],
                        table["remaining_failure"],
                        np.asarray(table["bundle"]["zero_mask"], dtype=bool),
                        brackets,
                    )[0]
                    luminosity[:, 1 + state_index] = (
                        coefficient * np.square(n_h) * cell_volume_cm3
                    )

                    table_2d = jeans_tables[state]
                    lookup_t_2d = np.clip(
                        selected_temperature,
                        10.0 ** table_2d["log_T"][0],
                        10.0 ** table_2d["log_T"][-1],
                    )
                    n_bracket = _brackets_2d(table_2d["log_nH"], np.log10(n_h))
                    t_bracket = _brackets_2d(
                        table_2d["log_T"], np.log10(lookup_t_2d),
                    )
                    coefficient_2d, remaining_weight = _interpolate_2d(
                        table_2d["log"],
                        table_2d["coefficient"],
                        table_2d["remaining"],
                        table_2d["zero"],
                        n_bracket,
                        t_bracket,
                    )
                    if np.any(remaining_weight > TOUCH_EPS):
                        raise RuntimeError(
                            f"simulation touches unfilled Halpha Jeans failure: {state}"
                        )
                    luminosity[:, 1 + len(STATE_ORDER) + state_index] = (
                        coefficient_2d * np.square(n_h) * cell_volume_cm3
                    )

                thermal_width = np.sqrt(
                    boltzmann_cgs * selected_temperature / hydrogen_mass_g,
                ) / 1.0e5
                thermal_width *= 1.0 - selected_velocity / c_kms
                slab_spectra = accumulate_velocity_spectra(
                    selected_velocity,
                    thermal_width,
                    luminosity,
                    velocity_edges,
                    cell_chunk=args.cell_chunk,
                    workers=args.workers,
                ).T
                accumulated[:, branch] += slab_spectra

            elapsed = time.perf_counter() - started
            rate = slab_number / elapsed
            eta = (n_slabs - slab_number) / rate if rate > 0.0 else np.nan
            print(
                f"Halpha LOS z [{slab_number:02d}/{n_slabs:02d}] "
                f"elapsed={elapsed / 60.0:.1f} min ETA={eta / 60.0:.1f} min",
                flush=True,
            )
    finally:
        for handle in handles.values():
            handle.close()

    spectra = YTArray(
        accumulated / projected_area_cm2,
        "erg/s/cm**2/(km/s)",
    ).to(DSIGMA_DV_UNIT).d

    geometry_specs = {
        "column": (
            np.concatenate((spectra[0:1], spectra[1:1 + len(STATE_ORDER)])),
            r"H$\alpha$ $(N_{\rm H}, n_{\rm H}, T)$, LOS z, $R=\infty$",
        ),
        "jeans": (
            np.concatenate((spectra[0:1], spectra[1 + len(STATE_ORDER):])),
            r"H$\alpha$ $(n_{\rm H}, T;\ \mathrm{Jeans\ length})$, LOS z, $R=\infty$",
        ),
    }
    outputs = {}
    for geometry, (curves, title) in geometry_specs.items():
        png, npz = paths[geometry]
        _plot(png, velocity, curves, title)
        np.savez_compressed(
            npz,
            velocity_kms=velocity,
            dsigma_dv=curves,
            model_keys=np.asarray(("pipeline",) + STATE_ORDER),
            regime_keys=np.asarray(REGIME_KEYS),
            dsigma_dv_units=np.asarray(DSIGMA_DV_UNIT),
            los=np.asarray("z"),
            projected_area_cm2=np.asarray(projected_area_cm2),
            completed_full_domain=np.asarray(True),
        )
        outputs[geometry] = {"png": str(png), "npz": str(npz)}
        print(f"Saved: {png}")

    report = args.output_dir / "halpha_LOSz_standard_comparison.json"
    report.write_text(json.dumps({
        "dataset": str(args.dataset),
        "dataset_time_Myr": float(ds.current_time.to_value("Myr")),
        "domain_dimensions": dimensions,
        "los": "z",
        "projected_area_cm2": projected_area_cm2,
        "projected_area_kpc2": float(
            ds.domain_width[0].to_value("kpc") * ds.domain_width[1].to_value("kpc")
        ),
        "temperature_policy": (
            "T_QUOKKA split; T_DESPOTIC lookup/thermal width below 3000 K; "
            "T_QUOKKA otherwise"
        ),
        "pipeline_policy": (
            "DESPOTIC e-/H+ below 3000 K; QUOKKA mu-derived e-/H+ otherwise"
        ),
        "cloudy_states": list(STATE_ORDER),
        "cache_paths": cache_paths,
        "counts": counts,
        "outputs": outputs,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }, indent=2) + "\n")
    print(f"Saved: {report}")


if __name__ == "__main__":
    main()
