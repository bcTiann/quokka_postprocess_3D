#!/usr/bin/env python3
"""Compute six HM2012 + filtered Black/ISM spectra for column and Jeans grids.

Cells are separated by T_QUOKKA.  The lookup and thermal temperature is
T_DESPOTIC below 3000 K and T_QUOKKA otherwise.  LOS y and LOS z use their
matching velocity component and projected domain area.  The output is at
R=infinity in cgs surface-luminosity-density-per-velocity units.  Raw Cloudy
failures are never filled; execution aborts if a simulation stencil touches
one.
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
    COLUMN_FIELD,
    TDSP_FIELD,
)
from plot_expanded_four_model_spectra import DVDR_FIELD
from plot_halpha_huang_figure2_losz_check import _open_caches
from plot_cloudy_line_physics_ablation_spectra import (
    N_CHANNELS,
    REGIME_SPLIT_K,
    TOUCH_EPS,
    _brackets,
    accumulate_velocity_spectra,
)
from quokka2s.pipeline.prep import config as cfg
from quokka2s.line_regimes import electron_fraction_from_mean_molecular_weight
from quokka2s.pipeline.prep.physics_fields import (
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


LINE_KEYS = ("cii", "halpha", "hi21", "ciii_977", "ciii_1907", "ciii_1909")
LINE_TITLES = {
    "cii": r"C II 158 $\mu$m",
    "halpha": r"H$\alpha$",
    "hi21": "H I 21 cm",
    "ciii_977": r"C III 977.020 $\AA$",
    "ciii_1907": r"C III] 1906.68 $\AA$",
    "ciii_1909": r"C III] 1908.73 $\AA$",
}
REGIME_KEYS = ("T_QUOKKA_lt_3000K", "T_QUOKKA_ge_3000K")
GEOMETRIES = ("column", "jeans")
NEW_KEY = "cloudy_native_hm2012_filtered_ism"
NEW_LABEL = "Cloudy native HM2012 + filtered Black/ISM"
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


def _load(path: Path, geometry: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        table = {name: np.asarray(source[name]) for name in source.files}
    keys = tuple(str(value) for value in table["line_keys"].tolist())
    if keys != LINE_KEYS:
        raise ValueError(f"unexpected line order {keys}: {path}")
    expected = (
        (len(LINE_KEYS), table["log_nH"].size, table["log_NH"].size,
         table["log_T"].size)
        if geometry == "column"
        else (len(LINE_KEYS), table["log_nH"].size, table["log_T"].size)
    )
    for name in ("log_emissivity_per_nH2", "emissivity_per_nH2",
                 "failure_mask", "zero_mask"):
        if table[name].shape != expected:
            raise ValueError(f"unexpected {name} shape {table[name].shape}: {path}")
    return table


def _interpolate(table: dict[str, np.ndarray], brackets) -> np.ndarray:
    raw_log = np.asarray(table["log_emissivity_per_nH2"], dtype=float)
    coefficient = np.asarray(table["emissivity_per_nH2"], dtype=float)
    failure = np.asarray(table["failure_mask"], dtype=bool)
    zero = np.asarray(table["zero_mask"], dtype=bool)
    n_cells = brackets[0][2].size
    linear_sum = np.zeros((len(LINE_KEYS), n_cells))
    log_sum = np.zeros_like(linear_sum)
    zero_support = np.zeros_like(linear_sum, dtype=bool)
    failure_weight = np.zeros_like(linear_sum)

    def recurse(axis_number: int, indices: list[np.ndarray], weight: np.ndarray):
        nonlocal linear_sum, log_sum, zero_support, failure_weight
        if axis_number == len(brackets):
            local_index = (slice(None), *indices)
            local_log = raw_log[local_index]
            local_weight = weight[None, :]
            linear_sum += coefficient[local_index] * local_weight
            log_sum += np.where(np.isfinite(local_log), local_log, 0.0) * local_weight
            zero_support |= zero[local_index] & (local_weight > TOUCH_EPS)
            failure_weight += failure[local_index] * local_weight
            return
        lower, upper, fraction = brackets[axis_number]
        recurse(axis_number + 1, indices + [lower], weight * (1.0 - fraction))
        recurse(axis_number + 1, indices + [upper], weight * fraction)

    recurse(0, [], np.ones(n_cells))
    touched = failure_weight > TOUCH_EPS
    if np.any(touched):
        details = {
            LINE_KEYS[index]: int(np.count_nonzero(touched[index]))
            for index in range(len(LINE_KEYS)) if np.any(touched[index])
        }
        raise RuntimeError(f"simulation touches raw Cloudy failures: {details}")
    return np.where(zero_support, linear_sum, np.power(10.0, log_sum))


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


def _plot_ciii(
    path: Path, velocity: np.ndarray, curves: np.ndarray, title: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    shared_max = float(np.nanmax(curves))
    for branch, axis in enumerate(axes):
        axis.plot(velocity, curves[0, branch], color="#0072B2", linewidth=1.7,
                  drawstyle="steps-mid", label=r"$(N_H,n_H,T)$")
        axis.plot(velocity, curves[1, branch], color="#D55E00", linewidth=1.7,
                  drawstyle="steps-mid", label=r"$(n_H,T)$ Jeans length")
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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    stem = "cloudy_hm2012_native_plus_filtered_ism_defaultabund_sixline"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--column-table", type=Path,
                        default=root / f"data/{stem}_column_10x10x21.npz")
    parser.add_argument("--jeans-table", type=Path,
                        default=root / f"data/{stem}_jeans_10x21.npz")
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument("--despotic-table", type=Path,
                        default=Path(cfg.DESPOTIC_TABLE_PATH))
    parser.add_argument("--los", choices=("y", "z"), default="y")
    parser.add_argument("--velocity-range-kms", type=float, default=None)
    parser.add_argument("--channels", type=int, default=N_CHANNELS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--slab-nz", type=int, default=32)
    parser.add_argument("--cell-chunk", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--state-key", default=NEW_KEY)
    parser.add_argument("--cloudy-label", default=NEW_LABEL)
    parser.add_argument("--filename-tag", default="nativeHM2012_filteredISM")
    parser.add_argument("--max-slabs", type=int, default=None,
                        help="development smoke-test limit; omit for production")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.velocity_range_kms is None:
        args.velocity_range_kms = _default_velocity_range_kms(args.los)
    if args.velocity_range_kms <= 0.0:
        raise ValueError("--velocity-range-kms must be positive")
    if args.channels <= 0:
        raise ValueError("--channels must be positive")
    if args.output_dir is None:
        directory = "native_hm12_filtered_black_ism_sixline"
        if args.los == "z":
            directory += "_LOSz"
        args.output_dir = Path(cfg.OUTPUT_DIR) / directory
    for name in ("column_table", "jeans_table", "dataset", "despotic_table",
                 "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    los_token = f"LOS{args.los}"
    spectra_path = (
        args.output_dir /
        f"{args.filename_tag}_sixline_Tsplit_Rinf_{los_token}.npz"
    )
    report_path = (
        args.output_dir /
        f"{args.filename_tag}_sixline_Tsplit_Rinf_{los_token}.json"
    )
    if (spectra_path.exists() or report_path.exists()) and not args.force:
        raise FileExistsError("outputs exist; pass --force")

    tables = {
        "column": _load(args.column_table, "column"),
        "jeans": _load(args.jeans_table, "jeans"),
    }
    radiation_descriptions = {
        str(np.asarray(table["radiation_field"]).item())
        for table in tables.values()
    }
    if len(radiation_descriptions) != 1:
        raise ValueError("column and Jeans tables describe different radiation fields")
    radiation_description = radiation_descriptions.pop()
    ds = yt.load(str(args.dataset))
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    nx, ny, nz = dimensions
    cache_handles, cache_paths = _open_caches(
        args.dataset, args.despotic_table, dimensions
    )
    column_file = cache_handles[COLUMN_FIELD]
    tdsp_file = cache_handles[TDSP_FIELD]
    dvdr_file = cache_handles[DVDR_FIELD]
    column_path = Path(cache_paths[COLUMN_FIELD[1]])
    tdsp_path = Path(cache_paths[TDSP_FIELD[1]])
    dvdr_path = Path(cache_paths[DVDR_FIELD[1]])
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
    masses = np.asarray((12.01, 1.00794, 1.00794, 12.01, 12.01, 12.01)) * amu_g
    velocity_edges = np.linspace(
        -args.velocity_range_kms, args.velocity_range_kms, args.channels + 1
    )
    velocity = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])
    accumulated = np.zeros((len(LINE_KEYS), len(GEOMETRIES), 2, args.channels))
    reference_accumulated = np.zeros((3, 2, args.channels))
    input_luminosity = np.zeros((len(LINE_KEYS), len(GEOMETRIES), 2))
    reference_input_luminosity = np.zeros((3, 2))
    counts = {"all_cells": 0, "T_QUOKKA_lt_3000_cells": 0,
              "T_QUOKKA_ge_3000_cells": 0, "failure_touches": 0,
              "axis_out_of_bounds": 0}
    started = time.perf_counter()
    slab_starts = list(range(0, nz, args.slab_nz))
    if args.max_slabs is not None:
        if args.max_slabs <= 0:
            raise ValueError("--max-slabs must be positive")
        slab_starts = slab_starts[:args.max_slabs]
    total_slabs = len(slab_starts)
    completed_full_domain = total_slabs == (nz + args.slab_nz - 1) // args.slab_nz
    try:
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
            tdsp = np.asarray(tdsp_file["data"][:, :, iz:iz + local_nz],
                              dtype=float).reshape(-1)
            dvdr = np.asarray(dvdr_file["data"][:, :, iz:iz + local_nz],
                              dtype=float).reshape(-1)
            low = tq < REGIME_SPLIT_K
            lookup_t = np.where(low, tdsp, tq)
            counts["all_cells"] += int(tq.size)
            counts["T_QUOKKA_lt_3000_cells"] += int(np.count_nonzero(low))
            counts["T_QUOKKA_ge_3000_cells"] += int(np.count_nonzero(~low))

            log_nh = np.log10(n_h)
            log_column = np.log10(column)
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

            for geometry_index, geometry in enumerate(GEOMETRIES):
                table = tables[geometry]
                axes = [np.asarray(table["log_nH"], dtype=float)]
                coordinates = [log_nh]
                if geometry == "column":
                    axes.append(np.asarray(table["log_NH"], dtype=float))
                    coordinates.append(log_column)
                axes.append(np.asarray(table["log_T"], dtype=float))
                coordinates.append(log_t)
                outside = np.zeros(tq.size, dtype=bool)
                for axis, coordinate in zip(axes, coordinates):
                    outside |= (coordinate < axis[0]) | (coordinate > axis[-1])
                counts["axis_out_of_bounds"] += int(np.count_nonzero(outside))
                if np.any(outside):
                    raise RuntimeError(
                        f"simulation leaves {geometry} table in slab {iz}:{iz + local_nz}"
                    )
                brackets = tuple(
                    _brackets(axis, coordinate)
                    for axis, coordinate in zip(axes, coordinates)
                )
                coefficients = _interpolate(table, brackets)
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
                        input_luminosity[
                            line_indices, geometry_index, branch
                        ] += np.sum(luminosity, axis=0)
                        thermal = np.sqrt(
                            boltzmann_cgs * lookup_t[selected] / mass
                        ) / 1.0e5
                        thermal *= 1.0 - velocity_los[selected] / c_kms
                        spectra = accumulate_velocity_spectra(
                            velocity_los[selected], thermal, luminosity,
                            velocity_edges, cell_chunk=args.cell_chunk,
                            workers=args.workers,
                        ).T
                        accumulated[line_indices, geometry_index, branch] += spectra

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
    delta_v_kms = float(velocity_edges[1] - velocity_edges[0])
    captured_luminosity = np.sum(accumulated, axis=-1) * delta_v_kms
    reference_captured_luminosity = (
        np.sum(reference_accumulated, axis=-1) * delta_v_kms
    )
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
    np.savez_compressed(
        spectra_path, velocity_kms=velocity, dsigma_dv=spectra,
        reference_dsigma_dv=reference_spectra,
        line_keys=np.asarray(LINE_KEYS), geometry_keys=np.asarray(GEOMETRIES),
        regime_keys=np.asarray(REGIME_KEYS), dsigma_dv_units=np.asarray(DSIGMA_DV_UNIT),
        state=np.asarray(args.state_key),
        los=np.asarray(args.los),
        projected_area_cm2=np.asarray(projected_area_cm2),
        velocity_range_kms=np.asarray(args.velocity_range_kms),
        input_luminosity_erg_s=input_luminosity,
        captured_luminosity_erg_s=captured_luminosity,
        capture_fraction=capture_fraction,
        reference_input_luminosity_erg_s=reference_input_luminosity,
        reference_captured_luminosity_erg_s=reference_captured_luminosity,
        reference_capture_fraction=reference_capture_fraction,
        completed_full_domain=np.asarray(completed_full_domain),
    )

    figures = {}
    title_geometry = {
        "column": r"$(N_H,n_H,T)$", "jeans": r"$(n_H,T)$",
    }
    for line_index, line in enumerate(LINE_KEYS[:3]):
        for geometry_index, geometry in enumerate(GEOMETRIES):
            curves = np.concatenate((
                reference_spectra[line_index][None],
                spectra[line_index, geometry_index][None],
            ))
            labels = (REFERENCE_LABELS[line], args.cloudy_label)
            output = args.output_dir / (
                f"{line}_{args.filename_tag}_{geometry}_"
                f"Tsplit_Rinf_{los_token}.png"
            )
            _plot_comparison(
                output, velocity, curves, labels,
                f"{LINE_TITLES[line]} {title_geometry[geometry]}, "
                f"LOS {args.los}, "
                r"$R=\infty$",
            )
            figures[f"{line}_{geometry}"] = str(output)

    for line_index, line in enumerate(LINE_KEYS[3:], start=3):
        output = args.output_dir / (
            f"{line}_column_vs_jeans_Tsplit_Rinf_{los_token}.png"
        )
        _plot_ciii(
            output, velocity, spectra[line_index],
            f"{LINE_TITLES[line]}, LOS {args.los}, " + r"$R=\infty$",
        )
        figures[line] = str(output)

    report = {
        "dataset": str(args.dataset),
        "radiation": radiation_description,
        "external_grackle_hm12_used": False,
        "los": args.los,
        "velocity_field": f"gas/velocity_{args.los}",
        "projected_area_cm2": projected_area_cm2,
        "projected_plane": "x-z" if args.los == "y" else "x-y",
        "velocity_range_kms": [-args.velocity_range_kms, args.velocity_range_kms],
        "velocity_channels": args.channels,
        "temperature_policy": (
            "split by T_QUOKKA; T_DESPOTIC lookup/thermal width below 3000 K, "
            "T_QUOKKA otherwise"
        ),
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
        "geometry_keys": list(GEOMETRIES),
        "line_keys": list(LINE_KEYS),
        "counts": counts,
        "completed_full_domain": completed_full_domain,
        "column_cache": str(column_path),
        "temperature_despotic_cache": str(tdsp_path),
        "dvdr_cache": str(dvdr_path),
        "input_luminosity_erg_s": input_luminosity.tolist(),
        "captured_luminosity_erg_s": captured_luminosity.tolist(),
        "capture_fraction": capture_fraction.tolist(),
        "reference_input_luminosity_erg_s": reference_input_luminosity.tolist(),
        "reference_captured_luminosity_erg_s": (
            reference_captured_luminosity.tolist()
        ),
        "reference_capture_fraction": reference_capture_fraction.tolist(),
        "spectra": str(spectra_path),
        "figures": figures,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
