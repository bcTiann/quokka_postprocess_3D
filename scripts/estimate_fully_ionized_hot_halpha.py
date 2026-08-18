#!/usr/bin/env python3
"""Order-of-magnitude H-alpha estimate for fully ionized T > 1e5 K gas."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yt
from yt.units.physical_constants import mh

from quokka2s.pipeline.prep import config as cfg
from quokka2s.pipeline.prep.physics_fields import (
    c,
    effective_halpha_recombination_coefficient,
    h,
    lambda_Halpha,
)


HUANG_X = 0.706
HUANG_Y = 0.281
T_MIN_K = 1.0e5
V_MIN_KMS = -200.0
V_MAX_KMS = 200.0
DV_KMS = 2.0  # matches the central-bump bins in the public QEDIV notebook


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "output/plt0655228_down1_Lext15kpc/atomic3line_rerun/fully_ionized_Tgt1e5_halpha_estimate.json",
    )
    parser.add_argument("--slab-nz", type=int, default=32)
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ds = yt.load(str(args.dataset))
    ds.force_periodicity()
    nx, ny, nz = (int(value) for value in ds.domain_dimensions)
    widths_cm = np.asarray(ds.domain_width.to("cm") / ds.domain_dimensions, dtype=float)
    cell_volume_cm3 = float(np.prod(widths_cm))
    cell_area_cm2 = float(widths_cm[0] * widths_cm[1])
    plane_area_cm2 = float(nx * ny * cell_area_cm2)
    proton_mass_g = float(mh.to_value("g"))
    photon_energy_erg = float(((h * c) / lambda_Halpha).in_cgs().value)
    solar_mass_g = 1.988409870698051e33

    edges = np.arange(V_MIN_KMS, V_MAX_KMS + DV_KMS, DV_KMS)
    velocity = 0.5 * (edges[:-1] + edges[1:])
    luminosity_bins = np.zeros(velocity.size, dtype=float)
    hot_mass_g = 0.0
    hot_luminosity_all_velocity = 0.0
    hot_cells = 0
    total_cells = 0
    emissivity_min = np.inf
    emissivity_max = 0.0
    luminosity_weighted_temperature_numerator = 0.0

    started = time.perf_counter()
    n_slabs = (nz + args.slab_nz - 1) // args.slab_nz
    for slab_number, iz in enumerate(range(0, nz, args.slab_nz), start=1):
        local_nz = min(args.slab_nz, nz - iz)
        left_edge = ds.domain_left_edge.copy()
        left_edge[2] += iz * (ds.domain_width[2] / ds.domain_dimensions[2])
        grid = ds.covering_grid(
            level=ds.max_level, left_edge=left_edge, dims=(nx, ny, local_nz),
        )
        temperature = np.asarray(grid[("boxlib", "temperature")], dtype=float).reshape(-1)
        density = np.asarray(grid[("gas", "density")].to("g/cm**3"), dtype=float).reshape(-1)
        velocity_z = np.asarray(grid[("gas", "velocity_z")].to("km/s"), dtype=float).reshape(-1)
        del grid

        selected = np.isfinite(temperature) & np.isfinite(density) & (temperature > T_MIN_K)
        total_cells += int(temperature.size)
        hot_cells += int(np.count_nonzero(selected))
        if np.any(selected):
            t = temperature[selected]
            rho = density[selected]
            vz = velocity_z[selected]

            # Fully ionized H and He using Huang et al.'s X and Y.
            n_hp = HUANG_X * rho / proton_mass_g
            n_e = (HUANG_X + 0.5 * HUANG_Y) * rho / proton_mass_g
            emissivity = (
                photon_energy_erg
                * effective_halpha_recombination_coefficient(t)
                * n_e
                * n_hp
            )
            luminosity = emissivity * cell_volume_cm3

            hot_mass_g += float(np.sum(rho) * cell_volume_cm3)
            hot_luminosity_all_velocity += float(np.sum(luminosity))
            luminosity_weighted_temperature_numerator += float(np.sum(t * luminosity))
            positive = emissivity[emissivity > 0.0]
            if positive.size:
                emissivity_min = min(emissivity_min, float(positive.min()))
                emissivity_max = max(emissivity_max, float(positive.max()))
            luminosity_bins += np.histogram(vz, bins=edges, weights=luminosity)[0]

        print(
            f"[{slab_number:02d}/{n_slabs:02d}] hot_cells={hot_cells} "
            f"elapsed={(time.perf_counter() - started) / 60.0:.2f} min",
            flush=True,
        )

    luminosity_in_velocity_range = float(luminosity_bins.sum())
    rongjun_spectrum = luminosity_bins / DV_KMS / cell_area_cm2
    plane_mean_spectrum = luminosity_bins / DV_KMS / plane_area_cm2
    report = {
        "dataset": str(args.dataset),
        "assumptions": {
            "temperature_selection": "T_QUOKKA > 1e5 K",
            "ionization": "H fully ionized; He fully doubly ionized",
            "hydrogen_mass_fraction": HUANG_X,
            "helium_mass_fraction": HUANG_Y,
            "halpha_coefficient": "Huang et al. 2025 Eq. 1 Case B",
            "line_profile": "delta-function cell LOS-z velocity; no thermal broadening",
            "velocity_range_km_s": [V_MIN_KMS, V_MAX_KMS],
            "velocity_bin_width_km_s": DV_KMS,
        },
        "domain_dimensions": [nx, ny, nz],
        "counts": {
            "all_cells": total_cells,
            "T_gt_1e5_cells": hot_cells,
            "T_gt_1e5_fraction": hot_cells / total_cells,
        },
        "geometry": {
            "cell_widths_cm": widths_cm.tolist(),
            "cell_volume_cm3": cell_volume_cm3,
            "cell_area_xy_cm2": cell_area_cm2,
            "plane_area_xy_cm2": plane_area_cm2,
            "projected_pixel_count": nx * ny,
        },
        "hot_gas": {
            "mass_g": hot_mass_g,
            "mass_Msun": hot_mass_g / solar_mass_g,
            "halpha_luminosity_all_velocity_erg_s": hot_luminosity_all_velocity,
            "halpha_luminosity_abs_vz_lt_200_erg_s": luminosity_in_velocity_range,
            "luminosity_fraction_abs_vz_lt_200": (
                luminosity_in_velocity_range / hot_luminosity_all_velocity
                if hot_luminosity_all_velocity > 0.0 else 0.0
            ),
            "luminosity_weighted_temperature_K": (
                luminosity_weighted_temperature_numerator / hot_luminosity_all_velocity
                if hot_luminosity_all_velocity > 0.0 else None
            ),
            "cell_emissivity_positive_range_erg_s_cm3": [
                emissivity_min if np.isfinite(emissivity_min) else None,
                emissivity_max,
            ],
        },
        "spectrum": {
            "rongjun_cell_area_normalized_peak_erg_s_cm2_per_km_s": float(rongjun_spectrum.max()),
            "full_plane_mean_peak_erg_s_cm2_per_km_s": float(plane_mean_spectrum.max()),
            "rongjun_integrated_erg_s_cm2": float(luminosity_in_velocity_range / cell_area_cm2),
            "full_plane_mean_integrated_erg_s_cm2": float(luminosity_in_velocity_range / plane_area_cm2),
            "normalization_ratio": nx * ny,
            "velocity_at_peak_km_s": float(velocity[int(np.argmax(rongjun_spectrum))]),
        },
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        velocity_km_s=velocity,
        luminosity_per_velocity_erg_s_per_km_s=luminosity_bins / DV_KMS,
        rongjun_cell_area_normalized=rongjun_spectrum,
        full_plane_mean=plane_mean_spectrum,
    )
    print(json.dumps(report, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
