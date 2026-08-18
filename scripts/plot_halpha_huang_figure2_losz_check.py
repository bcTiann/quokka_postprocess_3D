"""Audit Huang et al. Figure 2 normalization and make a LOS-z comparison.

The simulation curves follow the paper's Equation (6): delta-function cell
velocities, 5 km/s bins over +/-200 km/s, and division by the full projected
x-y domain area.  Two emissivity prescriptions are accumulated:

* the project's canonical hybrid pipeline (cold DESPOTIC, hot QUOKKA mu), and
* the Huang et al. all-temperature QUOKKA-mu Case-B prescription.

The public QEDIV notebook's published broad Gaussian is also shown before and
after replacing its single-cell area by the full 512x512 domain area.
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

from plot_expanded_four_model_spectra import (
    COLUMN_FIELD,
    DVDR_FIELD,
    TDSP_FIELD,
    _validated_cache,
)
from quokka2s.line_regimes import electron_fraction_from_mean_molecular_weight
from quokka2s.pipeline.cache import (
    cache_root_for_dataset,
    compute_cache_key,
    field_cache_key,
    field_cache_path,
)
from quokka2s.pipeline.prep import config as cfg
from quokka2s.pipeline.prep.physics_fields import (
    _clip_to_table_domain,
    c,
    effective_halpha_recombination_coefficient,
    h,
    lambda_Halpha,
)
from quokka2s.pipeline.spectrum_units import DSIGMA_DV_UNIT, dsigma_dv_ylabel
from quokka2s.tables import load_table
from quokka2s.tables.lookup import TableLookup


HUANG_X = 0.706
HUANG_Y = 0.281
PAPER_GRID_NX = 512
PAPER_GRID_NY = 512
PAPER_FIT_AMPLITUDE_CELL_AREA = 4.46e-2
PAPER_FIT_MU_KMS = -8.1
PAPER_FIT_SIGMA_KMS = 25.4


def _gaussian(v: np.ndarray, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(
        -0.5 * np.square((v - PAPER_FIT_MU_KMS) / PAPER_FIT_SIGMA_KMS)
    )


def _open_caches(
    dataset: Path,
    despotic_table: Path,
    dimensions: tuple[int, int, int],
) -> tuple[dict[tuple[str, str], object], dict[str, str]]:
    base = compute_cache_key(
        dataset, despotic_table, cfg.DOWNSAMPLE_FACTOR,
        cfg.COLUMN_EXTENSION_LATERAL_KPC,
    )
    legacy = compute_cache_key(
        dataset, despotic_table, cfg.DOWNSAMPLE_FACTOR,
        cfg.COLUMN_EXTENSION_LATERAL_KPC, schema_version=19,
    )
    handles = {}
    paths = {}
    for field in (COLUMN_FIELD, DVDR_FIELD, TDSP_FIELD):
        path = field_cache_path(cache_root_for_dataset(dataset), field)
        handles[field] = _validated_cache(
            path, field_cache_key(base, field), field, dimensions,
            legacy_key=field_cache_key(legacy, field),
        )
        paths[field[1]] = str(path)
    return handles, paths


def _plot_normalization_audit(output: Path) -> dict[str, float]:
    velocity = np.linspace(-200.0, 200.0, 1000)
    pixel_factor = PAPER_GRID_NX * PAPER_GRID_NY
    corrected_amplitude = PAPER_FIT_AMPLITUDE_CELL_AREA / pixel_factor
    published = _gaussian(velocity, PAPER_FIT_AMPLITUDE_CELL_AREA)
    corrected = _gaussian(velocity, corrected_amplitude)

    fig, axis = plt.subplots(figsize=(8.0, 5.4))
    axis.plot(velocity, published, color="#D55E00", linewidth=2.0,
              label=r"QEDIV notebook: divide by $\Delta x\,\Delta y$")
    axis.plot(velocity, corrected, color="#0072B2", linewidth=2.0,
              label=r"Equation (6): divide by $(N_x\Delta x)(N_y\Delta y)$")
    axis.set_yscale("log")
    axis.set_xlim(-200.0, 200.0)
    axis.set_ylim(1.0e-10, 1.0e-1)
    axis.set_xlabel(r"Velocity [km s$^{-1}$]")
    axis.set_ylabel(dsigma_dv_ylabel(DSIGMA_DV_UNIT))
    axis.set_title(r"Huang et al. Figure 2 broad-fit area normalization")
    axis.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return {
        "paper_grid_projected_cells": pixel_factor,
        "notebook_peak_cell_area_normalized": PAPER_FIT_AMPLITUDE_CELL_AREA,
        "equation6_peak_full_area_normalized": corrected_amplitude,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument("--despotic-table", type=Path, default=Path(cfg.DESPOTIC_TABLE_PATH))
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "output/plt0655228_down1_Lext15kpc/huang_figure2_check",
    )
    parser.add_argument("--slab-nz", type=int, default=32)
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.despotic_table = args.despotic_table.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit_png = args.output_dir / "Huang_Figure2_area_normalization_audit.png"
    audit = _plot_normalization_audit(audit_png)

    ds = yt.load(str(args.dataset))
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    nx, ny, nz = dimensions
    handles, cache_paths = _open_caches(args.dataset, args.despotic_table, dimensions)
    lookup = TableLookup(load_table(args.despotic_table))

    widths = np.asarray(ds.domain_width.to("cm") / ds.domain_dimensions, dtype=float)
    volume_cm3 = float(np.prod(widths))
    projected_area_cm2 = float(nx * ny * widths[0] * widths[1])
    proton_mass_g = float(mh.to_value("g"))
    project_h_mass_g = 1.00794 * 1.66053906660e-24
    boltzmann = float(kb.to_value("erg/K"))
    photon_energy = float(((h * c) / lambda_Halpha).in_cgs().value)
    edges = np.arange(-200.0, 200.0 + 5.0, 5.0)
    velocity = 0.5 * (edges[:-1] + edges[1:])
    histogram = np.zeros((2, velocity.size), dtype=float)
    counts = {"all_cells": 0, "velocity_outside_200_kms": 0, "huang_negative_xe": 0}
    started = time.perf_counter()
    n_slabs = (nz + args.slab_nz - 1) // args.slab_nz

    try:
        for slab_number, iz in enumerate(range(0, nz, args.slab_nz), start=1):
            local_nz = min(args.slab_nz, nz - iz)
            left_edge = ds.domain_left_edge.copy()
            left_edge[2] += iz * (ds.domain_width[2] / ds.domain_dimensions[2])
            grid = ds.covering_grid(
                level=ds.max_level, left_edge=left_edge, dims=(nx, ny, local_nz),
            )
            temperature_qk = np.asarray(grid[("boxlib", "temperature")], dtype=float).reshape(-1)
            density = np.asarray(grid[("gas", "density")].to("g/cm**3"), dtype=float).reshape(-1)
            velocity_z = np.asarray(grid[("gas", "velocity_z")].to("km/s"), dtype=float).reshape(-1)
            total_energy = np.asarray(
                grid[("gas", "total_energy_density")].to("erg/cm**3"), dtype=float,
            ).reshape(-1)
            kinetic_energy = np.asarray(
                grid[("gas", "kinetic_energy_density")].to("erg/cm**3"), dtype=float,
            ).reshape(-1)
            del grid
            internal_energy = total_energy - kinetic_energy

            column = np.asarray(
                handles[COLUMN_FIELD]["data"][:, :, iz:iz + local_nz], dtype=float,
            ).reshape(-1)
            dvdr = np.asarray(
                handles[DVDR_FIELD]["data"][:, :, iz:iz + local_nz], dtype=float,
            ).reshape(-1)
            temperature_dsp = np.asarray(
                handles[TDSP_FIELD]["data"][:, :, iz:iz + local_nz], dtype=float,
            ).reshape(-1)

            # Project pipeline: cold DESPOTIC chemistry, hot QUOKKA mu.
            n_h_project = density * float(cfg.X_H) / project_h_mass_g
            safe = _clip_to_table_domain(lookup, n_h_project, column, dvdr)
            densities = lookup.number_densities(("e-", "H+"), *safe)
            n_e_dsp = np.nan_to_num(densities["e-"], nan=0.0)
            n_hp_dsp = np.nan_to_num(densities["H+"], nan=0.0)
            x_e_project = electron_fraction_from_mean_molecular_weight(
                internal_energy, density, temperature_qk,
                hydrogen_mass_g=project_h_mass_g, boltzmann_erg_K=boltzmann,
            )
            n_e_qk = x_e_project * n_h_project
            n_hp_qk = np.minimum(x_e_project, 1.0) * n_h_project
            low = temperature_qk < 3000.0
            temperature_hybrid = np.where(low, temperature_dsp, temperature_qk)
            epsilon_hybrid = (
                photon_energy
                * effective_halpha_recombination_coefficient(temperature_hybrid)
                * np.where(low, n_e_dsp, n_e_qk)
                * np.where(low, n_hp_dsp, n_hp_qk)
            )

            # Huang et al.: all-temperature Grackle/QUOKKA mu inference.
            inverse_mu = (
                (5.0 / 3.0 - 1.0) * proton_mass_g * internal_energy
                / (density * boltzmann * temperature_qk)
            )
            x_e_huang_raw = (inverse_mu - HUANG_X - HUANG_Y / 4.0) / HUANG_X
            counts["huang_negative_xe"] += int(np.count_nonzero(x_e_huang_raw < 0.0))
            # The public code replaces negative values with the smallest
            # non-negative value in the full grid. Zero is the limiting value
            # and changes no observable H-alpha emission at plotting precision.
            x_e_huang = np.maximum(x_e_huang_raw, 0.0)
            n_h_huang = density * HUANG_X / proton_mass_g
            n_e_huang = x_e_huang * n_h_huang
            n_p_huang = np.minimum(x_e_huang, 1.0) * n_h_huang
            epsilon_huang = (
                photon_energy
                * effective_halpha_recombination_coefficient(temperature_qk)
                * n_e_huang * n_p_huang
            )

            for index, epsilon in enumerate((epsilon_hybrid, epsilon_huang)):
                histogram[index] += np.histogram(
                    velocity_z, bins=edges, weights=epsilon * volume_cm3,
                )[0]
            counts["all_cells"] += int(velocity_z.size)
            counts["velocity_outside_200_kms"] += int(
                np.count_nonzero(np.abs(velocity_z) >= 200.0)
            )
            print(
                f"[{slab_number:02d}/{n_slabs:02d}] "
                f"elapsed={(time.perf_counter() - started) / 60.0:.2f} min",
                flush=True,
            )
    finally:
        for handle in handles.values():
            handle.close()

    spectra = histogram / 5.0 / projected_area_cm2
    corrected_paper = _gaussian(
        velocity, PAPER_FIT_AMPLITUDE_CELL_AREA / (PAPER_GRID_NX * PAPER_GRID_NY),
    )
    fig, axis = plt.subplots(figsize=(9.2, 5.8))
    axis.plot(velocity, np.where(spectra[0] > 0.0, spectra[0], np.nan),
              color="#0072B2", linewidth=1.8, drawstyle="steps-mid",
              label="This work: hybrid pipeline")
    axis.plot(velocity, np.where(spectra[1] > 0.0, spectra[1], np.nan),
              color="#009E73", linewidth=1.8, drawstyle="steps-mid",
              label=r"This work: Huang-style $\mu$ prescription")
    axis.plot(velocity, corrected_paper, color="black", linestyle="--", linewidth=2.0,
              label=r"Huang Figure 2 broad fit / $512^2$")
    axis.axvline(-50.0, color="0.65", linestyle="--", linewidth=0.9)
    axis.axvline(50.0, color="0.65", linestyle="--", linewidth=0.9)
    axis.set_yscale("log")
    axis.set_xlim(-200.0, 200.0)
    positive = np.concatenate((spectra[spectra > 0.0], corrected_paper))
    axis.set_ylim(max(float(positive.min()) * 0.5, 1.0e-14), float(positive.max()) * 2.0)
    axis.set_xlabel(r"z-velocity [km s$^{-1}$]")
    axis.set_ylabel(dsigma_dv_ylabel(DSIGMA_DV_UNIT))
    axis.set_title(r"H$\alpha$ LOS z, full projected-area normalization")
    axis.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    axis.legend(frameon=False)
    fig.tight_layout()
    comparison_png = args.output_dir / "Halpha_LOSz_vs_Huang_corrected_Figure2.png"
    fig.savefig(comparison_png, dpi=250, bbox_inches="tight")
    plt.close(fig)

    npz = args.output_dir / "Halpha_LOSz_vs_Huang_corrected_Figure2.npz"
    np.savez_compressed(
        npz, velocity_kms=velocity,
        dsigma_dv=np.vstack((spectra, corrected_paper)),
        model_keys=np.asarray(("hybrid_pipeline", "huang_mu", "huang_figure2_corrected")),
        dsigma_dv_units=np.asarray(DSIGMA_DV_UNIT),
        velocity_bin_width_kms=np.asarray(5.0),
    )
    report = {
        "dataset": str(args.dataset),
        "dataset_time_Myr": float(ds.current_time.to_value("Myr")),
        "domain_dimensions": dimensions,
        "los": "z",
        "velocity_range_kms": [-200.0, 200.0],
        "velocity_bin_width_kms": 5.0,
        "line_profile": "delta function, matching Huang et al. Equation (6)",
        "projected_area_cm2": projected_area_cm2,
        "projected_area_kpc2": float(ds.domain_width[0].to_value("kpc") * ds.domain_width[1].to_value("kpc")),
        "normalization_audit": audit,
        "counts": counts,
        "integrated_surface_brightness": {
            "hybrid_pipeline": float(histogram[0].sum() / projected_area_cm2),
            "huang_mu": float(histogram[1].sum() / projected_area_cm2),
            "huang_figure2_corrected_gaussian": float(np.trapezoid(corrected_paper, velocity)),
        },
        "peak_dsigma_dv": {
            "hybrid_pipeline": float(spectra[0].max()),
            "huang_mu": float(spectra[1].max()),
            "huang_figure2_corrected_gaussian": float(corrected_paper.max()),
        },
        "cache_paths": cache_paths,
        "outputs": {"audit": str(audit_png), "comparison": str(comparison_png), "data": str(npz)},
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    report_path = args.output_dir / "Halpha_LOSz_vs_Huang_corrected_Figure2.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved: {audit_png}")
    print(f"Saved: {comparison_png}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
