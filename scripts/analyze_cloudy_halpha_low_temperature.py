"""Estimate simulation H-alpha luminosity below 3000 K from the low-T scan."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import yt
from yt.units.physical_constants import mh
from yt.units.yt_array import YTQuantity

from analyze_cloudy_halpha_temperature_tail import (
    TOUCH_EPS,
    interpolate_log_coefficient,
    load_grid,
)
from quokka2s.pipeline.prep import config as cfg


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / (
            "work/cloudy_cooling_tools_history/examples/grackle/"
            "hm_2012_halpha_lowT_sparse_output"
        ),
    )
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument(
        "--column-cache",
        type=Path,
        default=root
        / "intermediates/plt0655228/fields/field_gas_column_density_H.h5",
    )
    parser.add_argument(
        "--hot-result",
        type=Path,
        default=root
        / "output/plt0655228_down1_Lext15kpc/"
        "cloudy_halpha_temperature_tail.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "output/plt0655228_down1_Lext15kpc/"
        "cloudy_halpha_below_3000K.json",
    )
    parser.add_argument("--slab-nz", type=int, default=32)
    args = parser.parse_args()

    t_min = 10.0
    t_split = 3000.0
    log_nh, log_column, log_temperature, raw_log, failure_mask = load_grid(
        args.input_dir, t_min=t_min, t_max=t_split, t_points=21
    )
    if np.any(failure_mask):
        raise ValueError("low-temperature H-alpha table unexpectedly has failures")

    ds = yt.load(str(args.dataset.resolve()))
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    nx, ny, nz = dimensions
    cell_width = ds.domain_width.to("cm") / ds.domain_dimensions
    cell_volume_cm3 = float(np.prod(cell_width.to_value("cm")))
    hydrogen_mass_g = float(mh.to_value("g"))
    lsun_erg_s = float(YTQuantity(1.0, "Lsun").to_value("erg/s"))

    counts = {
        "all_cells": 0,
        "finite_positive_cells": 0,
        "temperature_below_10K_not_sampled": 0,
        "low_branch_10K_le_T_lt_3000K": 0,
        "nH_outside_table": 0,
        "NH_outside_table": 0,
        "touches_failure_node": 0,
    }
    low_luminosity = 0.0
    start = time.perf_counter()

    with h5py.File(args.column_cache, "r") as column_file:
        if column_file["data"].shape != dimensions:
            raise ValueError("column-density cache shape does not match dataset")
        for iz in range(0, nz, args.slab_nz):
            slab_nz = min(args.slab_nz, nz - iz)
            left_edge = ds.domain_left_edge.copy()
            left_edge[2] += iz * cell_width[2]
            grid = ds.covering_grid(
                level=ds.max_level,
                left_edge=left_edge,
                dims=(nx, ny, slab_nz),
            )
            temperature = np.asarray(grid[("boxlib", "temperature")], dtype=float)
            density = np.asarray(
                grid[("gas", "density")].to("g/cm**3"), dtype=float
            )
            n_h = density * float(cfg.X_H) / hydrogen_mass_g
            column = np.asarray(
                column_file["data"][:, :, iz : iz + slab_nz], dtype=float
            )

            valid = (
                np.isfinite(temperature)
                & np.isfinite(n_h)
                & np.isfinite(column)
                & (temperature > 0.0)
                & (n_h > 0.0)
                & (column > 0.0)
            )
            low = valid & (temperature >= t_min) & (temperature < t_split)
            counts["all_cells"] += int(temperature.size)
            counts["finite_positive_cells"] += int(np.count_nonzero(valid))
            counts["temperature_below_10K_not_sampled"] += int(
                np.count_nonzero(valid & (temperature < t_min))
            )
            counts["low_branch_10K_le_T_lt_3000K"] += int(np.count_nonzero(low))
            if not np.any(low):
                continue

            selected_nh = n_h[low]
            selected_column = column[low]
            coordinates = (
                np.log10(selected_nh),
                np.log10(selected_column),
                np.log10(temperature[low]),
            )
            outside_nh = (coordinates[0] < log_nh[0]) | (
                coordinates[0] > log_nh[-1]
            )
            outside_column = (coordinates[1] < log_column[0]) | (
                coordinates[1] > log_column[-1]
            )
            counts["nH_outside_table"] += int(np.count_nonzero(outside_nh))
            counts["NH_outside_table"] += int(np.count_nonzero(outside_column))
            if outside_nh.any() or outside_column.any():
                raise ValueError("low-temperature simulation cell is outside table axes")

            log_coefficient, failure_weight = interpolate_log_coefficient(
                (log_nh, log_column, log_temperature),
                raw_log,
                failure_mask,
                coordinates,
            )
            touched = failure_weight > TOUCH_EPS
            counts["touches_failure_node"] += int(np.count_nonzero(touched))
            if touched.any():
                raise ValueError("simulation cell touches a failed Cloudy node")
            low_luminosity += float(
                np.sum(
                    np.power(10.0, log_coefficient)
                    * np.square(selected_nh)
                    * cell_volume_cm3,
                    dtype=np.float64,
                )
            )

    hot_result = json.loads(args.hot_result.read_text())
    hot_luminosity = float(hot_result["total_halpha_luminosity_erg_s"])
    combined = low_luminosity + hot_luminosity
    result = {
        "dataset": str(args.dataset.resolve()),
        "raw_low_temperature_cloudy_output": str(args.input_dir.resolve()),
        "temperature_definition": "low: 10 <= T_QUOKKA < 3000 K; hot: T_QUOKKA >= 3000 K",
        "interpolation": "trilinear in log10(epsilon_Halpha/n_H^2)",
        "counts": counts,
        "low_temperature_halpha_luminosity_Lsun": low_luminosity / lsun_erg_s,
        "hot_temperature_halpha_luminosity_Lsun": hot_luminosity / lsun_erg_s,
        "combined_halpha_luminosity_Lsun": combined / lsun_erg_s,
        "low_fraction_of_combined": low_luminosity / combined if combined else 0.0,
        "hot_fraction_of_combined": hot_luminosity / combined if combined else 0.0,
        "elapsed_minutes": (time.perf_counter() - start) / 60.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
