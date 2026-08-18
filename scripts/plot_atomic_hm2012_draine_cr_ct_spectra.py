"""Sample and plot the HM2012+Draine+CR+charge-transfer atomic-line tables."""
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
    _open_validated_cache,
)
from plot_cii_defaultabund_radiation_cr_comparisons import _failure_weight_2d
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
from plot_hydrogen_atomic_radiation_comparisons import (
    _prepare_column,
    _prepare_jeans,
)
from quokka2s.pipeline.prep import config as cfg
from quokka2s.pipeline.spectrum_units import (
    DSIGMA_DV_UNIT,
    SPEED_OF_LIGHT_CGS,
    dsigma_dv_ylabel,
)


SPECIES = ("cii", "halpha", "hi21")
SPECIES_TITLES = {
    "cii": r"[C II] 158 $\mu$m",
    "halpha": r"H$\alpha$",
    "hi21": "H I 21 cm",
}
STATE = "hm2012_draine_cr_ct"
NEW_LABEL = "Cloudy HM2012 + Draine + CR + charge transfer"
REGIME_KEYS = ("T_QUOKKA_lt_3000K", "T_QUOKKA_ge_3000K")


def _plot(
    path: Path,
    velocity: np.ndarray,
    curves: np.ndarray,
    labels: tuple[str, ...],
    title: str,
) -> None:
    colors = ("#0072B2", "#9467BD", "#D55E00", "#CC79A7", "#009E73")
    linestyles = ("--", "-", "-", "-", "-")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    shared_max = float(np.nanmax(curves))
    for branch, axis in enumerate(axes):
        for index, label in enumerate(labels):
            axis.plot(
                velocity,
                curves[index, branch],
                color=colors[index],
                linestyle=linestyles[index],
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
        axis.legend(fontsize=7.9, frameon=False)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument(
        "--despotic-table", type=Path, default=Path(cfg.DESPOTIC_TABLE_PATH),
    )
    parser.add_argument(
        "--table-dir", type=Path,
        default=root / "data/cloudy_atomic_defaultabund_radiation_3state_views",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(cfg.OUTPUT_DIR) / "atomic3line_rerun",
    )
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
    spectra_path = args.output_dir / "atomic_hm2012_draine_cr_ct_Tsplit_Rinf.npz"
    report_path = args.output_dir / "atomic_hm2012_draine_cr_ct_Tsplit_Rinf.json"
    if (spectra_path.exists() or report_path.exists()) and not args.force:
        raise FileExistsError("outputs exist; pass --force")

    column_tables = {
        species: _prepare_column(
            args.table_dir / f"cloudy_{species}_{STATE}_column_10x10x21.npz"
        )
        for species in SPECIES
    }
    jeans_tables = {
        species: _prepare_jeans(
            args.table_dir / f"cloudy_{species}_{STATE}_jeans_10x21.npz"
        )
        for species in SPECIES
    }

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
    boltzmann_cgs = float(kb.to_value("erg/K"))
    c_kms = float(SPEED_OF_LIGHT_CGS.to_value("cm/s")) / 1.0e5
    amu_g = 1.66053906660e-24
    line_masses = {
        "cii": 12.01 * amu_g,
        "halpha": 1.00794 * amu_g,
        "hi21": 1.00794 * amu_g,
    }
    velocity_edges = np.linspace(-V_RANGE_KMS, V_RANGE_KMS, N_CHANNELS + 1)
    velocity = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])
    accumulated = np.zeros((len(SPECIES), 2, 2, N_CHANNELS), dtype=float)
    counts = {
        "all_cells": 0,
        "T_QUOKKA_lt_3000_cells": 0,
        "T_QUOKKA_ge_3000_cells": 0,
        "original_failure_touches": {
            f"{species}_{geometry}": 0
            for species in SPECIES for geometry in ("column", "jeans")
        },
    }
    started = time.perf_counter()
    total_slabs = (nz + args.slab_nz - 1) // args.slab_nz

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
                column = column_all[selected]
                selected_velocity = velocity_y[selected]
                selected_temperature = (
                    tdsp_all[selected] if branch == 0 else temperature_qk[selected]
                )
                coefficients = np.zeros((len(SPECIES), 2, n_h.size), dtype=float)
                for species_index, species in enumerate(SPECIES):
                    column_table = column_tables[species]
                    axes = column_table["axes"]
                    lookup_t = np.clip(
                        selected_temperature,
                        10.0 ** axes[2][0],
                        10.0 ** axes[2][-1],
                    )
                    brackets = tuple(
                        _brackets(axis, values)
                        for axis, values in zip(
                            axes, (np.log10(n_h), np.log10(column), np.log10(lookup_t))
                        )
                    )
                    original_weight = _failure_support_weights(
                        np.asarray(
                            column_table["bundle"]["failure_mask"], dtype=bool,
                        )[:, 0],
                        brackets,
                    )[0]
                    counts["original_failure_touches"][f"{species}_column"] += int(
                        np.count_nonzero(original_weight > TOUCH_EPS)
                    )
                    coefficients[species_index, 0] = _interpolate_selected_cii(
                        column_table["filled_log"],
                        column_table["filled_coefficient"],
                        column_table["remaining_failure"],
                        np.asarray(
                            column_table["bundle"]["zero_mask"], dtype=bool,
                        ),
                        brackets,
                    )[0]

                    jeans_table = jeans_tables[species]
                    lookup_t_2d = np.clip(
                        selected_temperature,
                        10.0 ** jeans_table["log_T"][0],
                        10.0 ** jeans_table["log_T"][-1],
                    )
                    n_bracket = _brackets_2d(jeans_table["log_nH"], np.log10(n_h))
                    t_bracket = _brackets_2d(
                        jeans_table["log_T"], np.log10(lookup_t_2d),
                    )
                    original_weight_2d = _failure_weight_2d(
                        jeans_table["failure"], n_bracket, t_bracket,
                    )
                    counts["original_failure_touches"][f"{species}_jeans"] += int(
                        np.count_nonzero(original_weight_2d > TOUCH_EPS)
                    )
                    coefficient_2d, remaining_weight = _interpolate_2d(
                        jeans_table["log"],
                        jeans_table["coefficient"],
                        jeans_table["remaining"],
                        jeans_table["zero"],
                        n_bracket,
                        t_bracket,
                    )
                    if np.any(remaining_weight > TOUCH_EPS):
                        raise RuntimeError(
                            f"simulation touches unfilled {species} Jeans failure"
                        )
                    coefficients[species_index, 1] = coefficient_2d

                n_h2_volume = np.square(n_h) * cell_volume_cm3
                for line_group in (("cii",), ("halpha", "hi21")):
                    luminosity = np.column_stack(tuple(
                        coefficients[SPECIES.index(species), geometry] * n_h2_volume
                        for species in line_group for geometry in range(2)
                    ))
                    thermal_width = np.sqrt(
                        boltzmann_cgs * selected_temperature / line_masses[line_group[0]],
                    ) / 1.0e5
                    thermal_width *= 1.0 - selected_velocity / c_kms
                    group_spectra = accumulate_velocity_spectra(
                        selected_velocity,
                        thermal_width,
                        luminosity,
                        velocity_edges,
                        cell_chunk=args.cell_chunk,
                        workers=args.workers,
                    ).T
                    for line_offset, species in enumerate(line_group):
                        species_index = SPECIES.index(species)
                        accumulated[species_index, :, branch] += group_spectra[
                            2 * line_offset:2 * line_offset + 2
                        ]

            elapsed = time.perf_counter() - started
            rate = slab_number / elapsed
            eta = (total_slabs - slab_number) / rate if rate > 0.0 else np.nan
            print(
                f"[{slab_number:02d}/{total_slabs:02d}] "
                f"elapsed={elapsed / 60.0:.1f} min ETA={eta / 60.0:.1f} min",
                flush=True,
            )
    finally:
        column_file.close()
        tdsp_file.close()

    spectra = YTArray(
        accumulated / projected_area_cm2,
        "erg/s/cm**2/(km/s)",
    ).to(DSIGMA_DV_UNIT).d
    np.savez_compressed(
        spectra_path,
        velocity_kms=velocity,
        dsigma_dv=spectra,
        species_keys=np.asarray(SPECIES),
        geometry_keys=np.asarray(("column", "jeans")),
        regime_keys=np.asarray(REGIME_KEYS),
        dsigma_dv_units=np.asarray(DSIGMA_DV_UNIT),
        state=np.asarray(STATE),
        completed_full_domain=np.asarray(True),
    )

    figures = {}
    for species_index, species in enumerate(SPECIES):
        for geometry_index, (geometry, title_geometry) in enumerate((
            ("column", r"$(N_{\rm H}, n_{\rm H}, T)$"),
            ("JeansLength", r"$(n_{\rm H}, T;\ \mathrm{Jeans\ length})$"),
        )):
            if species == "cii":
                source_name = (
                    "CII_default_abund_radiation_comparison_column_NHonly_Tsplit_Rinf_nozoom.npz"
                    if geometry == "column"
                    else "CII_default_abund_radiation_comparison_JeansLength_Tsplit_Rinf_nozoom.npz"
                )
                with np.load(args.output_dir / source_name, allow_pickle=False) as table:
                    old_velocity = np.asarray(table["velocity_kms"], dtype=float)
                    old = np.asarray(table["dsigma_dv"], dtype=float)
                labels = ("DESPOTIC", "Cloudy HM2012", "Cloudy HM2012 + Draine", NEW_LABEL)
            else:
                source = args.output_dir / (
                    f"{species}_pipeline_vs_default_abund_radiation_"
                    f"{geometry}_Tsplit_Rinf_nozoom_without_CR.npz"
                )
                with np.load(source, allow_pickle=False) as table:
                    old_velocity = np.asarray(table["velocity_kms"], dtype=float)
                    old = np.asarray(table["dsigma_dv"], dtype=float)
                labels = ("pipeline", "Cloudy HM2012", "Cloudy HM2012 + Draine", NEW_LABEL)
            if not np.allclose(old_velocity, velocity, rtol=0.0, atol=1.0e-10):
                if np.allclose(old_velocity[::-1], velocity, rtol=0.0, atol=1.0e-10):
                    old = old[..., ::-1]
                else:
                    raise ValueError("reference and new velocity axes differ")
            curves = np.concatenate((
                old[:3], spectra[species_index, geometry_index][None],
            ))
            output = args.output_dir / (
                f"{species}_HM2012_Draine_CR_CT_{geometry}_Tsplit_Rinf.png"
            )
            _plot(
                output,
                velocity,
                curves,
                labels,
                rf"{SPECIES_TITLES[species]} {title_geometry}, LOS y, $R=\infty$",
            )
            figures[f"{species}_{geometry}"] = str(output)

    report = {
        "dataset": str(args.dataset),
        "state": STATE,
        "temperature_policy": (
            "T_QUOKKA split; T_DESPOTIC lookup/thermal width below 3000 K; "
            "T_QUOKKA otherwise"
        ),
        "failure_policy": (
            "count original failure support; linear fill only when bracketed; "
            "abort on any remaining failure support"
        ),
        "counts": counts,
        "column_cache": str(column_path),
        "temperature_despotic_cache": str(tdsp_path),
        "spectra": str(spectra_path),
        "figures": figures,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
