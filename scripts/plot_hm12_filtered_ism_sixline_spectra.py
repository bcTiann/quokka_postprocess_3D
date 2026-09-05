#!/usr/bin/env python3
"""Compute spectra from the seven-field Cloudy Jeans lookup table.

Cells are separated by T_QUOKKA.  The lookup and thermal temperature is
T_DESPOTIC below 3000 K and T_QUOKKA otherwise.  LOS y and LOS z use their
matching velocity component and projected domain area.  The output is at
R=infinity in cgs surface-luminosity-density-per-velocity units. The simulation
column selects the HM2012 attenuation field and is clipped to the tabulated
1e18--1e21 cm^-2 interval; density and temperature are never clipped. Raw
Cloudy failures are never filled, and execution aborts if a simulation stencil
touches one.
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
import h5py
import numpy as np
import yt
from yt.units.physical_constants import kb, mh
from yt.units.yt_array import YTArray

from plot_cii_cloudy_tdsp_split_spectra import (
    COLUMN_FIELD,
    TDSP_FIELD,
)
from plot_expanded_four_model_spectra import DVDR_FIELD
from plot_halpha_huang_figure2_losz_check import _open_caches
from plot_cloudy_line_physics_ablation_spectra import (
    N_CHANNELS,
    REGIME_SPLIT_K,
    accumulate_velocity_spectra,
)
from quokka2s.cloudy_sixline_lookup import CloudySixLineLookup
from quokka2s.pipeline.prep import config as cfg
from quokka2s.pipeline.cache import cache_root_for_dataset, field_cache_path
from quokka2s.line_regimes import electron_fraction_from_mean_molecular_weight
from quokka2s.pipeline.prep.physics_fields import (
    DVDR_FLOOR,
    _HI_emissivity_from_number_density,
    _clip_to_table_domain,
    _table_emissivity,
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


LINE_KEYS = (
    "cii",
    "halpha",
    "hi21",
    "ciii_977",
    "ciii_1907",
    "ciii_1909",
    "civ_1548",
    "civ_1551",
)
CO_LINE_KEYS = ("co10", "co21")
LINE_TITLES = {
    "cii": r"C II 158 $\mu$m",
    "halpha": r"H$\alpha$",
    "hi21": "H I 21 cm",
    "ciii_977": r"C III 977.020 $\AA$",
    "ciii_1907": r"C III] 1906.68 $\AA$",
    "ciii_1909": r"C III] 1908.73 $\AA$",
    "civ_1548": r"C IV 1548.19 $\AA$",
    "civ_1551": r"C IV 1550.78 $\AA$",
}
CO_LINE_TITLES = {
    "co10": "CO(1-0)",
    "co21": "CO(2-1)",
}
REGIME_KEYS = ("T_QUOKKA_lt_3000K", "T_QUOKKA_ge_3000K")
NEW_KEY = "cloudy_hm2012_attenuation_grid_eightline_jeans"
NEW_LABEL = "Cloudy HM2012 attenuation grid + filtered Black/ISM"
REFERENCE_LABELS = {"cii": "DESPOTIC", "halpha": "pipeline", "hi21": "pipeline"}


def _projected_area_cm2(
    los: str,
    dimensions: tuple[int, int, int],
    cell_width_cm: np.ndarray,
) -> float:
    nx, ny, nz = dimensions
    if los == "y":
        return float(nx * nz * cell_width_cm[0] * cell_width_cm[2])
    if los == "z":
        return float(nx * ny * cell_width_cm[0] * cell_width_cm[1])
    raise ValueError(f"unsupported LOS: {los!r}")


def _default_velocity_range_kms(los: str) -> float:
    return 50.0 if los == "y" else 200.0


def _recompute_dvdr_slab(
    ds,
    dimensions: tuple[int, int, int],
    iz: int,
    local_nz: int,
) -> np.ndarray:
    """Return |div(v)|/3 for one z slab using a one-cell z halo."""
    nx, ny, nz = dimensions
    load_start = max(0, iz - 1)
    load_stop = min(nz, iz + local_nz + 1)
    left_edge = ds.domain_left_edge.copy()
    left_edge[2] += load_start * ds.domain_width[2] / ds.domain_dimensions[2]
    if load_stop == nz:
        left_edge[2] = (
            ds.domain_right_edge[2]
            - (load_stop - load_start)
            * ds.domain_width[2] / ds.domain_dimensions[2]
        )
    grid = ds.covering_grid(
        level=ds.max_level,
        left_edge=left_edge,
        dims=(nx, ny, load_stop - load_start),
    )
    vx = np.asarray(grid[("gas", "velocity_x")].to("cm/s"), dtype=float)
    vy = np.asarray(grid[("gas", "velocity_y")].to("cm/s"), dtype=float)
    vz = np.asarray(grid[("gas", "velocity_z")].to("cm/s"), dtype=float)
    del grid
    widths = np.asarray(
        ds.domain_width.to("cm") / ds.domain_dimensions, dtype=float
    )
    divergence = (
        np.gradient(vx, widths[0], axis=0)
        + np.gradient(vy, widths[1], axis=1)
        + np.gradient(vz, widths[2], axis=2)
    )
    core_start = iz - load_start
    core = np.abs(
        divergence[:, :, core_start:core_start + local_nz]
    ) / 3.0
    return np.maximum(core, DVDR_FLOOR).reshape(-1)


def _plot_comparison(
    path: Path,
    velocity: np.ndarray,
    curves: np.ndarray,
    labels: tuple[str, ...],
    title: str,
) -> None:
    colors = ("#0072B2", "#9467BD", "#D55E00", "#009E73", "#000000")
    styles = ("--", "-", "-", "-", "-")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    shared_max = float(np.nanmax(curves))
    for branch, axis in enumerate(axes):
        for index, label in enumerate(labels):
            axis.plot(
                velocity, curves[index, branch], color=colors[index],
                linestyle=styles[index], linewidth=1.7,
                drawstyle="steps-mid", label=label,
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
        axis.legend(fontsize=7.8, frameon=False)
        axis.ticklabel_format(style="sci", axis="y", scilimits=(0, 0),
                              useMathText=True)
        axis.tick_params(axis="y", labelleft=True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _plot_cloudy_carbon(
    path: Path, velocity: np.ndarray, curves: np.ndarray, title: str, label: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    shared_max = float(np.nanmax(curves))
    for branch, axis in enumerate(axes):
        axis.plot(
            velocity,
            curves[branch],
            color="#D55E00",
            linewidth=1.7,
            drawstyle="steps-mid",
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
        axis.legend(frameon=False)
        axis.ticklabel_format(style="sci", axis="y", scilimits=(0, 0),
                              useMathText=True)
        axis.tick_params(axis="y", labelleft=True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _preflight_cloudy_sampling(
    ds,
    *,
    dimensions: tuple[int, int, int],
    slab_starts: list[int],
    slab_nz: int,
    column_file,
    tdsp_file,
    dvdr_file,
    despotic_lookup: TableLookup,
    recompute_tdsp: bool,
    recompute_dvdr: bool,
    cloudy_lookup: CloudySixLineLookup,
    hydrogen_mass_g: float,
) -> dict[str, object]:
    """Scan all lookup stencils before any spectrum accumulation begins."""
    nx, ny, nz = dimensions
    report: dict[str, object] = {
        "cells_scanned": 0,
        "invalid_nonpositive_or_nonfinite_inputs": 0,
        "density_or_temperature_out_of_bounds": 0,
        "NH_below_1e18_clipped": 0,
        "NH_above_1e21_clipped": 0,
        "failure_touched_union_cells": 0,
        "failure_touched_by_line": {key: 0 for key in LINE_KEYS},
        "maximum_failure_weight": 0.0,
    }
    for slab_number, iz in enumerate(slab_starts, start=1):
        local_nz = min(slab_nz, nz - iz)
        left_edge = ds.domain_left_edge.copy()
        left_edge[2] += iz * (ds.domain_width[2] / ds.domain_dimensions[2])
        grid = ds.covering_grid(
            level=ds.max_level, left_edge=left_edge, dims=(nx, ny, local_nz)
        )
        density = np.asarray(
            grid[("gas", "density")].to("g/cm**3"), dtype=float
        ).reshape(-1)
        tq = np.asarray(grid[("boxlib", "temperature")], dtype=float).reshape(-1)
        del grid
        n_h = density * float(cfg.X_H) / hydrogen_mass_g
        column = np.asarray(
            column_file["data"][:, :, iz:iz + local_nz], dtype=float
        ).reshape(-1)
        if recompute_tdsp:
            if recompute_dvdr:
                dvdr = _recompute_dvdr_slab(
                    ds, dimensions, iz, local_nz
                )
            else:
                dvdr = np.asarray(
                    dvdr_file["data"][:, :, iz:iz + local_nz], dtype=float
                ).reshape(-1)
            safe = _clip_to_table_domain(
                despotic_lookup, n_h, column, dvdr
            )
            tdsp = despotic_lookup.temperature(*safe)
        else:
            tdsp = np.asarray(
                tdsp_file["data"][:, :, iz:iz + local_nz], dtype=float
            ).reshape(-1)
        lookup_t = np.where(tq < REGIME_SPLIT_K, tdsp, tq)
        report["cells_scanned"] += int(tq.size)

        valid = (
            np.isfinite(n_h)
            & np.isfinite(column)
            & np.isfinite(lookup_t)
            & (n_h > 0.0)
            & (column > 0.0)
            & (lookup_t > 0.0)
        )
        report["invalid_nonpositive_or_nonfinite_inputs"] += int(
            np.count_nonzero(~valid)
        )
        log_nh = np.full(n_h.shape, np.nan)
        log_t = np.full(lookup_t.shape, np.nan)
        log_nh[valid] = np.log10(n_h[valid])
        log_t[valid] = np.log10(lookup_t[valid])
        outside = valid & (
            (log_nh < cloudy_lookup.log_nH[0])
            | (log_nh > cloudy_lookup.log_nH[-1])
            | (log_t < cloudy_lookup.log_T[0])
            | (log_t > cloudy_lookup.log_T[-1])
        )
        report["density_or_temperature_out_of_bounds"] += int(
            np.count_nonzero(outside)
        )
        sampled = valid & ~outside
        if np.any(sampled):
            diagnostics = cloudy_lookup.diagnose(
                lookup_t[sampled], n_h[sampled], column[sampled]
            )
            report["NH_below_1e18_clipped"] += int(
                np.count_nonzero(diagnostics.attenuation_column_below_table)
            )
            report["NH_above_1e21_clipped"] += int(
                np.count_nonzero(diagnostics.attenuation_column_above_table)
            )
            touched = diagnostics.failure_touched
            report["failure_touched_union_cells"] += int(
                np.count_nonzero(np.any(touched, axis=0))
            )
            for line_index, line_key in enumerate(LINE_KEYS):
                report["failure_touched_by_line"][line_key] += int(
                    np.count_nonzero(touched[line_index])
                )
            report["maximum_failure_weight"] = max(
                float(report["maximum_failure_weight"]),
                diagnostics.maximum_failure_weight,
            )
        print(
            f"[preflight {slab_number:02d}/{len(slab_starts):02d}] "
            f"failure cells={report['failure_touched_union_cells']}",
            flush=True,
        )
    return report


def _open_unkeyed_field_cache(
    path: Path,
    field: tuple[str, str],
    dimensions: tuple[int, int, int],
) -> h5py.File:
    """Open a user-selected cache after structural validation only.

    This is used solely by the old/new DESPOTIC-table comparison.  The column
    and velocity-gradient fields are simulation-derived and do not depend on
    the DESPOTIC table, although the historical shared cache key included its
    path and modification time.
    """
    handle = h5py.File(path, "r")
    actual_field = (
        str(handle.attrs.get("field_type", "")),
        str(handle.attrs.get("field_name", "")),
    )
    if actual_field != field:
        handle.close()
        raise ValueError(f"cache field mismatch in {path}: {actual_field}")
    if tuple(handle["data"].shape) != dimensions:
        handle.close()
        raise ValueError(f"cache shape mismatch in {path}")
    return handle


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    stem = "cloudy_hm2012_attgrid_ism_nh21_cmb_cr_defaultabund_eightline_jeans"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cloudy-table",
        type=Path,
        default=root / f"data/{stem}_7x10x21.npz",
    )
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument("--despotic-table", type=Path,
                        default=Path(cfg.DESPOTIC_TABLE_PATH))
    parser.add_argument("--column-cache", type=Path, default=None)
    parser.add_argument("--dvdr-cache", type=Path, default=None)
    parser.add_argument(
        "--recompute-dvdr",
        action="store_true",
        help=(
            "recompute |div(v)|/3 from the QUOKKA velocities with the current "
            "floor instead of reading the historical dV/dr cache"
        ),
    )
    parser.add_argument(
        "--recompute-tdsp",
        action="store_true",
        help=(
            "interpolate T_DESPOTIC from --despotic-table in each slab; "
            "read the selected column and dV/dr caches without table-key checks"
        ),
    )
    parser.add_argument("--los", choices=("y", "z"), default="y")
    parser.add_argument("--velocity-range-kms", type=float, default=None)
    parser.add_argument("--channels", type=int, default=N_CHANNELS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--slab-nz", type=int, default=32)
    parser.add_argument("--cell-chunk", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--state-key", default=NEW_KEY)
    parser.add_argument("--cloudy-label", default=NEW_LABEL)
    parser.add_argument("--filename-tag", default="hm2012_attgrid_filteredISM")
    parser.add_argument("--max-slabs", type=int, default=None,
                        help="development smoke-test limit; omit for production")
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="write the spectrum bundle and report without standalone figures",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.velocity_range_kms is None:
        args.velocity_range_kms = _default_velocity_range_kms(args.los)
    if args.velocity_range_kms <= 0.0:
        raise ValueError("--velocity-range-kms must be positive")
    if args.channels <= 0:
        raise ValueError("--channels must be positive")
    if args.recompute_dvdr and not args.recompute_tdsp:
        raise ValueError("--recompute-dvdr requires --recompute-tdsp")
    if args.output_dir is None:
        directory = "hm2012_attenuation_grid_filtered_black_ism_eightline"
        if args.los == "z":
            directory += "_LOSz"
        args.output_dir = Path(cfg.OUTPUT_DIR) / directory
    for name in ("cloudy_table", "dataset", "despotic_table", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    for name in ("column_cache", "dvdr_cache"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    los_token = f"LOS{args.los}"
    spectra_path = (
        args.output_dir /
        f"{args.filename_tag}_eightline_Tsplit_Rinf_{los_token}.npz"
    )
    report_path = (
        args.output_dir /
        f"{args.filename_tag}_eightline_Tsplit_Rinf_{los_token}.json"
    )
    preflight_path = (
        args.output_dir /
        f"{args.filename_tag}_cloudy_sampling_preflight_{los_token}.json"
    )
    if (spectra_path.exists() or report_path.exists()) and not args.force:
        raise FileExistsError("outputs exist; pass --force")

    cloudy_lookup = CloudySixLineLookup(args.cloudy_table)
    if cloudy_lookup.line_keys != LINE_KEYS:
        raise ValueError(
            f"unexpected Cloudy line order {cloudy_lookup.line_keys}: "
            f"{args.cloudy_table}"
        )
    radiation_description = str(
        np.asarray(cloudy_lookup.metadata["radiation_field"]).item()
    )
    ds = yt.load(str(args.dataset))
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    nx, ny, nz = dimensions
    if args.recompute_tdsp:
        cache_root = cache_root_for_dataset(args.dataset)
        column_path = args.column_cache or field_cache_path(
            cache_root, COLUMN_FIELD
        )
        cache_handles = {
            COLUMN_FIELD: _open_unkeyed_field_cache(
                column_path, COLUMN_FIELD, dimensions
            ),
        }
        cache_paths = {COLUMN_FIELD[1]: str(column_path)}
        if not args.recompute_dvdr:
            dvdr_path = args.dvdr_cache or field_cache_path(
                cache_root, DVDR_FIELD
            )
            cache_handles[DVDR_FIELD] = _open_unkeyed_field_cache(
                dvdr_path, DVDR_FIELD, dimensions
            )
            cache_paths[DVDR_FIELD[1]] = str(dvdr_path)
    else:
        cache_handles, cache_paths = _open_caches(
            args.dataset, args.despotic_table, dimensions
        )
    column_file = cache_handles[COLUMN_FIELD]
    dvdr_file = (
        None if args.recompute_dvdr else cache_handles[DVDR_FIELD]
    )
    tdsp_file = None if args.recompute_tdsp else cache_handles[TDSP_FIELD]
    column_path = Path(cache_paths[COLUMN_FIELD[1]])
    dvdr_path = (
        None if args.recompute_dvdr else Path(cache_paths[DVDR_FIELD[1]])
    )
    tdsp_path = (
        None if args.recompute_tdsp else Path(cache_paths[TDSP_FIELD[1]])
    )
    despotic_lookup = TableLookup(load_table(args.despotic_table))
    cell_width_cm = np.asarray(
        ds.domain_width.to("cm") / ds.domain_dimensions, dtype=float
    )
    cell_volume_cm3 = float(np.prod(cell_width_cm))
    projected_area_cm2 = _projected_area_cm2(args.los, dimensions, cell_width_cm)
    hydrogen_mass_g = float(mh.to_value("g"))
    boltzmann_cgs = float(kb.to_value("erg/K"))
    c_kms = float(SPEED_OF_LIGHT_CGS.to_value("cm/s")) / 1.0e5
    amu_g = 1.66053906660e-24
    masses = np.asarray(
        (12.01, 1.00794, 1.00794, 12.01, 12.01, 12.01, 12.01, 12.01)
    ) * amu_g
    co_mass = (12.01 + 15.999) * amu_g
    velocity_edges = np.linspace(
        -args.velocity_range_kms, args.velocity_range_kms, args.channels + 1
    )
    velocity = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])
    accumulated = np.zeros((len(LINE_KEYS), 2, args.channels))
    reference_accumulated = np.zeros((3, 2, args.channels))
    co_accumulated = np.zeros((len(CO_LINE_KEYS), 2, args.channels))
    input_luminosity = np.zeros((len(LINE_KEYS), 2))
    reference_input_luminosity = np.zeros((3, 2))
    co_input_luminosity = np.zeros((len(CO_LINE_KEYS), 2))
    counts = {
        "all_cells": 0,
        "T_QUOKKA_lt_3000_cells": 0,
        "T_QUOKKA_ge_3000_cells": 0,
        "NH_below_1e18_clipped": 0,
        "NH_above_1e21_clipped": 0,
        "failure_touches": 0,
        "density_or_temperature_out_of_bounds": 0,
        "despotic_dvdr_below_table": 0,
        "despotic_dvdr_above_table": 0,
    }
    started = time.perf_counter()
    slab_starts = list(range(0, nz, args.slab_nz))
    if args.max_slabs is not None:
        if args.max_slabs <= 0:
            raise ValueError("--max-slabs must be positive")
        slab_starts = slab_starts[:args.max_slabs]
    total_slabs = len(slab_starts)
    completed_full_domain = total_slabs == (nz + args.slab_nz - 1) // args.slab_nz
    try:
        preflight = _preflight_cloudy_sampling(
            ds,
            dimensions=dimensions,
            slab_starts=slab_starts,
            slab_nz=args.slab_nz,
            column_file=column_file,
            tdsp_file=tdsp_file,
            dvdr_file=dvdr_file,
            despotic_lookup=despotic_lookup,
            recompute_tdsp=args.recompute_tdsp,
            recompute_dvdr=args.recompute_dvdr,
            cloudy_lookup=cloudy_lookup,
            hydrogen_mass_g=hydrogen_mass_g,
        )
        preflight.update({
            "dataset": str(args.dataset),
            "cloudy_table": str(args.cloudy_table),
            "completed_full_domain": completed_full_domain,
            "temperature_policy": (
                "T_DESPOTIC where T_QUOKKA < 3000 K; T_QUOKKA otherwise"
            ),
            "NH_policy": (
                "clip below 1e18 and above 1e21 cm^-2; no extrapolation"
            ),
        })
        preflight_path.write_text(json.dumps(preflight, indent=2) + "\n")
        counts["NH_below_1e18_clipped"] = int(
            preflight["NH_below_1e18_clipped"]
        )
        counts["NH_above_1e21_clipped"] = int(
            preflight["NH_above_1e21_clipped"]
        )
        counts["failure_touches"] = int(preflight["failure_touched_union_cells"])
        counts["density_or_temperature_out_of_bounds"] = int(
            preflight["density_or_temperature_out_of_bounds"]
        )
        if int(preflight["invalid_nonpositive_or_nonfinite_inputs"]) > 0:
            raise RuntimeError(f"invalid Cloudy lookup inputs; see {preflight_path}")
        if int(preflight["density_or_temperature_out_of_bounds"]) > 0:
            raise RuntimeError(
                f"Cloudy density/temperature inputs are out of bounds; see {preflight_path}"
            )
        if int(preflight["failure_touched_union_cells"]) > 0:
            raise RuntimeError(
                f"simulation touches Cloudy failure nodes; see {preflight_path}"
            )

        for slab_number, iz in enumerate(slab_starts, start=1):
            local_nz = min(args.slab_nz, nz - iz)
            left_edge = ds.domain_left_edge.copy()
            left_edge[2] += iz * (ds.domain_width[2] / ds.domain_dimensions[2])
            grid = ds.covering_grid(level=ds.max_level, left_edge=left_edge,
                                    dims=(nx, ny, local_nz))
            density = np.asarray(grid[("gas", "density")].to("g/cm**3"),
                                 dtype=float).reshape(-1)
            tq = np.asarray(grid[("boxlib", "temperature")], dtype=float).reshape(-1)
            velocity_los = np.asarray(
                grid[("gas", f"velocity_{args.los}")].to("km/s"),
                dtype=float,
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
            n_h = density * float(cfg.X_H) / hydrogen_mass_g
            column = np.asarray(column_file["data"][:, :, iz:iz + local_nz],
                                dtype=float).reshape(-1)
            if args.recompute_dvdr:
                dvdr = _recompute_dvdr_slab(
                    ds, dimensions, iz, local_nz
                )
            else:
                dvdr = np.asarray(
                    dvdr_file["data"][:, :, iz:iz + local_nz], dtype=float
                ).reshape(-1)
            if args.recompute_tdsp:
                safe = _clip_to_table_domain(
                    despotic_lookup, n_h, column, dvdr
                )
                tdsp = despotic_lookup.temperature(*safe)
            else:
                tdsp = np.asarray(
                    tdsp_file["data"][:, :, iz:iz + local_nz], dtype=float
                ).reshape(-1)
            low = tq < REGIME_SPLIT_K
            lookup_t = np.where(low, tdsp, tq)
            counts["all_cells"] += int(tq.size)
            counts["T_QUOKKA_lt_3000_cells"] += int(np.count_nonzero(low))
            counts["T_QUOKKA_ge_3000_cells"] += int(np.count_nonzero(~low))
            counts["despotic_dvdr_below_table"] += int(np.count_nonzero(
                dvdr < despotic_lookup.table.dVdr_values[0]
            ))
            counts["despotic_dvdr_above_table"] += int(np.count_nonzero(
                dvdr > despotic_lookup.table.dVdr_values[-1]
            ))

            log_nh = np.log10(n_h)
            log_t = np.log10(lookup_t)

            # Recompute the independent comparison curves on the same cells,
            # LOS velocity axis, projected area, temperature split, and
            # thermal-width policy as the new Cloudy spectra.  This avoids
            # importing historical LOS-y curves with incompatible provenance.
            safe = _clip_to_table_domain(despotic_lookup, n_h, column, dvdr)
            number_densities = despotic_lookup.number_densities(
                ("e-", "H+", "H"), *safe
            )
            n_e_despotic = np.nan_to_num(number_densities["e-"], nan=0.0)
            n_hp_despotic = np.nan_to_num(number_densities["H+"], nan=0.0)
            n_hi_despotic = np.nan_to_num(number_densities["H"], nan=0.0)
            internal_energy = total_energy - kinetic_energy
            x_e = electron_fraction_from_mean_molecular_weight(
                internal_energy,
                density,
                tq,
                hydrogen_mass_g=hydrogen_mass_g,
                boltzmann_erg_K=boltzmann_cgs,
            )
            n_e_quokka = x_e * n_h
            n_hp_quokka = np.minimum(x_e, 1.0) * n_h
            n_hi_quokka = np.where(x_e <= 1.0, (1.0 - x_e) * n_h, 0.0)
            cii_despotic = _table_emissivity(
                despotic_lookup, "C+", n_h, column, dvdr
            )
            co_emissivity = np.column_stack((
                _table_emissivity(
                    despotic_lookup, "CO", n_h, column, dvdr
                ),
                _table_emissivity(
                    despotic_lookup, "CO21", n_h, column, dvdr
                ),
            ))
            halpha_photon_energy = float(((h * c) / lambda_Halpha).in_cgs().value)

            for branch, selected in enumerate((low, ~low)):
                if not np.any(selected):
                    continue
                if branch == 0:
                    n_e_reference = n_e_despotic[selected]
                    n_hp_reference = n_hp_despotic[selected]
                    n_hi_reference = n_hi_despotic[selected]
                else:
                    n_e_reference = n_e_quokka[selected]
                    n_hp_reference = n_hp_quokka[selected]
                    n_hi_reference = n_hi_quokka[selected]
                reference_epsilon = np.column_stack((
                    cii_despotic[selected],
                    halpha_photon_energy
                    * effective_halpha_recombination_coefficient(
                        lookup_t[selected]
                    )
                    * n_e_reference
                    * n_hp_reference,
                    _HI_emissivity_from_number_density(n_hi_reference),
                ))
                reference_luminosity = reference_epsilon * cell_volume_cm3
                reference_input_luminosity[:, branch] += np.sum(
                    reference_luminosity, axis=0
                )
                for mass in np.unique(masses[:3]):
                    line_indices = np.flatnonzero(masses[:3] == mass)
                    thermal = np.sqrt(
                        boltzmann_cgs * lookup_t[selected] / mass
                    ) / 1.0e5
                    thermal *= 1.0 - velocity_los[selected] / c_kms
                    reference_accumulated[line_indices, branch] += (
                        accumulate_velocity_spectra(
                            velocity_los[selected],
                            thermal,
                            reference_luminosity[:, line_indices],
                            velocity_edges,
                            cell_chunk=args.cell_chunk,
                            workers=args.workers,
                        ).T
                    )

                # CO always uses the DESPOTIC equilibrium temperature and
                # emissivity; T_QUOKKA only selects the displayed branch.
                co_luminosity = (
                    co_emissivity[selected] * cell_volume_cm3
                )
                co_input_luminosity[:, branch] += np.sum(
                    co_luminosity, axis=0
                )
                co_thermal = np.sqrt(
                    boltzmann_cgs * tdsp[selected] / co_mass
                ) / 1.0e5
                co_thermal *= 1.0 - velocity_los[selected] / c_kms
                co_accumulated[:, branch] += accumulate_velocity_spectra(
                    velocity_los[selected],
                    co_thermal,
                    co_luminosity,
                    velocity_edges,
                    cell_chunk=args.cell_chunk,
                    workers=args.workers,
                ).T

            outside = (
                (log_nh < cloudy_lookup.log_nH[0])
                | (log_nh > cloudy_lookup.log_nH[-1])
                | (log_t < cloudy_lookup.log_T[0])
                | (log_t > cloudy_lookup.log_T[-1])
            )
            counts["density_or_temperature_out_of_bounds"] += int(
                np.count_nonzero(outside)
            )
            if np.any(outside):
                raise RuntimeError(
                    "simulation density or lookup temperature leaves the Cloudy "
                    f"table in slab {iz}:{iz + local_nz}"
                )
            sampled = cloudy_lookup.sample(lookup_t, n_h, column)
            coefficients = sampled.emissivity_per_nH2
            n_h2_volume = np.square(n_h) * cell_volume_cm3
            for branch, selected in enumerate((low, ~low)):
                if not np.any(selected):
                    continue
                for mass in np.unique(masses):
                    line_indices = np.flatnonzero(masses == mass)
                    luminosity = (
                        coefficients[line_indices][:, selected].T
                        * n_h2_volume[selected, None]
                    )
                    input_luminosity[line_indices, branch] += np.sum(
                        luminosity, axis=0
                    )
                    thermal = np.sqrt(
                        boltzmann_cgs * lookup_t[selected] / mass
                    ) / 1.0e5
                    thermal *= 1.0 - velocity_los[selected] / c_kms
                    line_spectra = accumulate_velocity_spectra(
                        velocity_los[selected], thermal, luminosity,
                        velocity_edges, cell_chunk=args.cell_chunk,
                        workers=args.workers,
                    ).T
                    accumulated[line_indices, branch] += line_spectra

            elapsed = time.perf_counter() - started
            rate = slab_number / elapsed
            eta = (total_slabs - slab_number) / rate if rate > 0.0 else np.nan
            print(f"[{slab_number:02d}/{total_slabs:02d}] "
                  f"elapsed={elapsed / 60.0:.1f} min ETA={eta / 60.0:.1f} min",
                  flush=True)
    finally:
        for handle in cache_handles.values():
            handle.close()

    spectra = YTArray(
        accumulated / projected_area_cm2, "erg/s/cm**2/(km/s)"
    ).to(DSIGMA_DV_UNIT).d
    reference_spectra = YTArray(
        reference_accumulated / projected_area_cm2,
        "erg/s/cm**2/(km/s)",
    ).to(DSIGMA_DV_UNIT).d
    co_spectra = YTArray(
        co_accumulated / projected_area_cm2,
        "erg/s/cm**2/(km/s)",
    ).to(DSIGMA_DV_UNIT).d
    delta_v_kms = float(velocity_edges[1] - velocity_edges[0])
    captured_luminosity = np.sum(accumulated, axis=-1) * delta_v_kms
    reference_captured_luminosity = (
        np.sum(reference_accumulated, axis=-1) * delta_v_kms
    )
    co_captured_luminosity = np.sum(co_accumulated, axis=-1) * delta_v_kms
    capture_fraction = np.divide(
        captured_luminosity,
        input_luminosity,
        out=np.ones_like(captured_luminosity),
        where=input_luminosity != 0.0,
    )
    reference_capture_fraction = np.divide(
        reference_captured_luminosity,
        reference_input_luminosity,
        out=np.ones_like(reference_captured_luminosity),
        where=reference_input_luminosity != 0.0,
    )
    co_capture_fraction = np.divide(
        co_captured_luminosity,
        co_input_luminosity,
        out=np.ones_like(co_captured_luminosity),
        where=co_input_luminosity != 0.0,
    )
    np.savez_compressed(
        spectra_path, velocity_kms=velocity, dsigma_dv=spectra,
        reference_dsigma_dv=reference_spectra,
        co_dsigma_dv=co_spectra,
        line_keys=np.asarray(LINE_KEYS),
        regime_keys=np.asarray(REGIME_KEYS), dsigma_dv_units=np.asarray(DSIGMA_DV_UNIT),
        co_line_keys=np.asarray(CO_LINE_KEYS),
        state=np.asarray(args.state_key),
        cloudy_table=np.asarray(str(args.cloudy_table)),
        los=np.asarray(args.los),
        projected_area_cm2=np.asarray(projected_area_cm2),
        velocity_range_kms=np.asarray(args.velocity_range_kms),
        input_luminosity_erg_s=input_luminosity,
        captured_luminosity_erg_s=captured_luminosity,
        capture_fraction=capture_fraction,
        reference_input_luminosity_erg_s=reference_input_luminosity,
        reference_captured_luminosity_erg_s=reference_captured_luminosity,
        reference_capture_fraction=reference_capture_fraction,
        co_input_luminosity_erg_s=co_input_luminosity,
        co_captured_luminosity_erg_s=co_captured_luminosity,
        co_capture_fraction=co_capture_fraction,
        completed_full_domain=np.asarray(completed_full_domain),
    )

    figures = {}
    table_title = r"$(N_H,n_H,T)$ Jeans-length table"
    if not args.skip_figures:
        for line_index, line in enumerate(LINE_KEYS[:3]):
            curves = np.concatenate((
                reference_spectra[line_index][None],
                spectra[line_index][None],
            ))
            labels = (REFERENCE_LABELS[line], args.cloudy_label)
            output = args.output_dir / (
                f"{line}_{args.filename_tag}_Tsplit_Rinf_{los_token}.png"
            )
            _plot_comparison(
                output, velocity, curves, labels,
                f"{LINE_TITLES[line]} {table_title}, LOS {args.los}, "
                r"$R=\infty$",
            )
            figures[line] = str(output)

        for line_index, line in enumerate(LINE_KEYS[3:], start=3):
            output = args.output_dir / (
                f"{line}_{args.filename_tag}_Tsplit_Rinf_{los_token}.png"
            )
            _plot_cloudy_carbon(
                output, velocity, spectra[line_index],
                f"{LINE_TITLES[line]}, LOS {args.los}, " + r"$R=\infty$",
                args.cloudy_label,
            )
            figures[line] = str(output)

        for line_index, line in enumerate(CO_LINE_KEYS):
            output = args.output_dir / (
                f"{line}_{args.filename_tag}_Tsplit_Rinf_{los_token}.png"
            )
            _plot_comparison(
                output,
                velocity,
                co_spectra[line_index][None],
                ("DESPOTIC",),
                f"{CO_LINE_TITLES[line]}, LOS {args.los}, " + r"$R=\infty$",
            )
            figures[line] = str(output)

    report = {
        "dataset": str(args.dataset),
        "radiation": radiation_description,
        "cloudy_table": str(args.cloudy_table),
        "external_grackle_hm12_used": False,
        "los": args.los,
        "velocity_field": f"gas/velocity_{args.los}",
        "projected_area_cm2": projected_area_cm2,
        "projected_plane": "x-z" if args.los == "y" else "x-y",
        "column_density_definition": (
            "harmonic mean of +z and -z cumulative columns"
            if getattr(cfg, "COLUMN_DENSITY_DIRECTIONS", "z") == "z"
            else "six-direction aggregate"
        ),
        "velocity_range_kms": [-args.velocity_range_kms, args.velocity_range_kms],
        "velocity_channels": args.channels,
        "temperature_policy": (
            "split by T_QUOKKA; T_DESPOTIC lookup/thermal width below 3000 K, "
            "T_QUOKKA otherwise"
        ),
        "NH_attenuation_lookup_policy": (
            "z+/- harmonic-mean simulation NH; clip below 1e18 to 1e18 and "
            "above 1e21 to 1e21; interpolate inside; never extrapolate"
        ),
        "cloudy_geometry": "Jeans length capped at 100 pc",
        "reference_policy": {
            "cii": "DESPOTIC emissivity recomputed on the same cells",
            "halpha": (
                "DESPOTIC e-/H+ below 3000 K; QUOKKA mu-derived e-/H+ otherwise"
            ),
            "hi21": (
                "DESPOTIC n_HI below 3000 K; QUOKKA mu-derived n_HI otherwise"
            ),
            "thermal_width": (
                "T_DESPOTIC below 3000 K; T_QUOKKA otherwise for all curves"
            ),
        },
        "failure_policy": "raw failures retained; abort if stencil weight > 1e-12",
        "sampling_preflight": str(preflight_path),
        "line_keys": list(LINE_KEYS),
        "co_line_keys": list(CO_LINE_KEYS),
        "counts": counts,
        "completed_full_domain": completed_full_domain,
        "column_cache": str(column_path),
        "temperature_despotic_cache": (
            None if tdsp_path is None else str(tdsp_path)
        ),
        "temperature_despotic_policy": (
            "interpolated slab-by-slab from the selected DESPOTIC table"
            if args.recompute_tdsp else "read from validated field cache"
        ),
        "velocity_gradient_cache": (
            None if dvdr_path is None else str(dvdr_path)
        ),
        "velocity_gradient_policy": (
            "recomputed as max(abs(div(v))/3, DVDR_FLOOR) with z halos"
            if args.recompute_dvdr else "read from selected field cache"
        ),
        "dvdr_cache": None if dvdr_path is None else str(dvdr_path),
        "input_luminosity_erg_s": input_luminosity.tolist(),
        "captured_luminosity_erg_s": captured_luminosity.tolist(),
        "capture_fraction": capture_fraction.tolist(),
        "reference_input_luminosity_erg_s": reference_input_luminosity.tolist(),
        "reference_captured_luminosity_erg_s": (
            reference_captured_luminosity.tolist()
        ),
        "reference_capture_fraction": reference_capture_fraction.tolist(),
        "co_input_luminosity_erg_s": co_input_luminosity.tolist(),
        "co_captured_luminosity_erg_s": co_captured_luminosity.tolist(),
        "co_capture_fraction": co_capture_fraction.tolist(),
        "spectra": str(spectra_path),
        "figures": figures,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
