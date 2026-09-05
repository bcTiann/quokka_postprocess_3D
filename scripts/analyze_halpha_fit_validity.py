#!/usr/bin/env python3
"""Audit where the production H-alpha calculation uses the Draine fit.

The diagnostic reproduces the pipeline's two-temperature-regime inputs:

* T_QUOKKA < 3000 K: T_DESPOTIC and DESPOTIC n_e/n_H+;
* T_QUOKKA >= 3000 K: T_QUOKKA and the QUOKKA-mu n_e/n_H+ inference.

It reports both cell-count and H-alpha-luminosity fractions inside the stated
validity range of Draine (2011) Eq. 14.8: 1e3 < T < 3e4 K.  The density check
uses n_e <= 1e6 cm^-3, the upper density through which Draine says the fit
remains accurate to within a few percent.
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
from matplotlib.colors import LogNorm
from yt.units.physical_constants import kb, mh

from plot_halpha_huang_figure2_losz_check import _open_caches
from plot_expanded_four_model_spectra import COLUMN_FIELD, DVDR_FIELD, TDSP_FIELD
from quokka2s.line_regimes import electron_fraction_from_mean_molecular_weight
from quokka2s.pipeline.prep import config as cfg
from quokka2s.pipeline.prep.physics_fields import (
    _clip_to_table_domain,
    c,
    effective_halpha_recombination_coefficient,
    h,
    lambda_Halpha,
)
from quokka2s.tables import load_table
from quokka2s.tables.lookup import TableLookup


T_MIN_K = 1.0e3
T_MAX_K = 3.0e4
NE_MAX_CM3 = 1.0e6
REGIME_SPLIT_K = 3.0e3


def _fraction(part: float, whole: float) -> float:
    return float(part / whole) if whole != 0.0 else float("nan")


def _weighted_quantile_from_hist(
    edges: np.ndarray, weights: np.ndarray, quantile: float,
) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return float("nan")
    cumulative = np.cumsum(weights)
    index = int(np.searchsorted(cumulative, quantile * total, side="left"))
    index = min(max(index, 0), weights.size - 1)
    before = float(cumulative[index - 1]) if index else 0.0
    within = float(weights[index])
    fraction = 0.0 if within <= 0.0 else (quantile * total - before) / within
    return float(edges[index] + fraction * (edges[index + 1] - edges[index]))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument(
        "--despotic-table", type=Path, default=Path(cfg.DESPOTIC_TABLE_PATH),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(cfg.OUTPUT_DIR) / "halpha_draine_fit_validity",
    )
    parser.add_argument("--slab-nz", type=int, default=32)
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.despotic_table = args.despotic_table.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds = yt.load(str(args.dataset))
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    nx, ny, nz = dimensions
    handles, cache_paths = _open_caches(
        args.dataset, args.despotic_table, dimensions,
    )
    lookup = TableLookup(load_table(args.despotic_table))

    widths_cm = np.asarray(
        ds.domain_width.to("cm") / ds.domain_dimensions, dtype=float,
    )
    cell_volume_cm3 = float(np.prod(widths_cm))
    hydrogen_mass_g = float(mh.to_value("g"))
    boltzmann_cgs = float(kb.to_value("erg/K"))
    photon_energy_erg = float(((h * c) / lambda_Halpha).in_cgs().value)

    log_t_edges = np.linspace(0.0, 9.5, 191)
    log_ne_edges = np.linspace(-20.0, 10.0, 241)
    cell_hist = np.zeros((log_t_edges.size - 1, log_ne_edges.size - 1))
    luminosity_hist = np.zeros_like(cell_hist)
    t_lum_hist = np.zeros(log_t_edges.size - 1)
    ne_lum_hist = np.zeros(log_ne_edges.size - 1)

    category_names = (
        "all_finite_cells",
        "positive_emissivity_cells",
        "temperature_valid",
        "temperature_below_1000K",
        "temperature_above_30000K",
        "electron_density_valid",
        "electron_density_above_1e6cm-3",
        "temperature_and_density_valid",
        "nonpositive_ne_or_nHp",
    )
    branch_names = ("T_QUOKKA_lt_3000K", "T_QUOKKA_ge_3000K", "all")
    counts = {
        branch: {category: 0 for category in category_names}
        for branch in branch_names
    }
    luminosities = {
        branch: {category: 0.0 for category in category_names}
        for branch in branch_names
    }

    started = time.perf_counter()
    n_slabs = (nz + args.slab_nz - 1) // args.slab_nz
    try:
        for slab_number, iz in enumerate(range(0, nz, args.slab_nz), start=1):
            local_nz = min(args.slab_nz, nz - iz)
            left_edge = ds.domain_left_edge.copy()
            left_edge[2] += iz * (
                ds.domain_width[2] / ds.domain_dimensions[2]
            )
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
            total_energy = np.asarray(
                grid[("gas", "total_energy_density")].to("erg/cm**3"),
                dtype=float,
            ).reshape(-1)
            kinetic_energy = np.asarray(
                grid[("gas", "kinetic_energy_density")].to("erg/cm**3"),
                dtype=float,
            ).reshape(-1)
            del grid

            column = np.asarray(
                handles[COLUMN_FIELD]["data"][:, :, iz:iz + local_nz],
                dtype=float,
            ).reshape(-1)
            dvdr = np.asarray(
                handles[DVDR_FIELD]["data"][:, :, iz:iz + local_nz],
                dtype=float,
            ).reshape(-1)
            temperature_dsp = np.asarray(
                handles[TDSP_FIELD]["data"][:, :, iz:iz + local_nz],
                dtype=float,
            ).reshape(-1)

            n_h = density * float(cfg.X_H) / hydrogen_mass_g
            safe = _clip_to_table_domain(lookup, n_h, column, dvdr)
            despotic_densities = lookup.number_densities(("e-", "H+"), *safe)
            n_e_dsp = np.nan_to_num(despotic_densities["e-"], nan=0.0)
            n_hp_dsp = np.nan_to_num(despotic_densities["H+"], nan=0.0)

            internal_energy = total_energy - kinetic_energy
            x_e = electron_fraction_from_mean_molecular_weight(
                internal_energy,
                density,
                temperature_qk,
                hydrogen_mass_g=hydrogen_mass_g,
                boltzmann_erg_K=boltzmann_cgs,
            )
            n_e_hot = x_e * n_h
            n_hp_hot = np.minimum(x_e, 1.0) * n_h

            low = temperature_qk < REGIME_SPLIT_K
            temperature = np.where(low, temperature_dsp, temperature_qk)
            n_e = np.where(low, n_e_dsp, n_e_hot)
            n_hp = np.where(low, n_hp_dsp, n_hp_hot)
            emissivity = (
                photon_energy_erg
                * effective_halpha_recombination_coefficient(temperature)
                * n_e
                * n_hp
            )
            luminosity = emissivity * cell_volume_cm3

            finite = (
                np.isfinite(temperature)
                & np.isfinite(n_e)
                & np.isfinite(n_hp)
                & np.isfinite(luminosity)
            )
            positive = finite & (temperature > 0.0) & (n_e > 0.0) & (n_hp > 0.0) & (luminosity > 0.0)
            t_valid = positive & (temperature > T_MIN_K) & (temperature < T_MAX_K)
            t_low = positive & (temperature <= T_MIN_K)
            t_high = positive & (temperature >= T_MAX_K)
            ne_valid = positive & (n_e <= NE_MAX_CM3)
            ne_high = positive & (n_e > NE_MAX_CM3)
            both_valid = t_valid & ne_valid
            nonpositive = finite & ((n_e <= 0.0) | (n_hp <= 0.0))

            categories = {
                "all_finite_cells": finite,
                "positive_emissivity_cells": positive,
                "temperature_valid": t_valid,
                "temperature_below_1000K": t_low,
                "temperature_above_30000K": t_high,
                "electron_density_valid": ne_valid,
                "electron_density_above_1e6cm-3": ne_high,
                "temperature_and_density_valid": both_valid,
                "nonpositive_ne_or_nHp": nonpositive,
            }
            for branch_name, branch_mask in (
                ("T_QUOKKA_lt_3000K", low),
                ("T_QUOKKA_ge_3000K", ~low),
                ("all", np.ones_like(low, dtype=bool)),
            ):
                for category, mask in categories.items():
                    selected = branch_mask & mask
                    counts[branch_name][category] += int(np.count_nonzero(selected))
                    luminosities[branch_name][category] += float(
                        np.sum(luminosity[selected])
                    )

            if np.any(positive):
                log_t = np.log10(temperature[positive])
                log_ne = np.log10(n_e[positive])
                selected_luminosity = luminosity[positive]
                cell_hist += np.histogram2d(
                    log_t, log_ne, bins=(log_t_edges, log_ne_edges),
                )[0]
                luminosity_hist += np.histogram2d(
                    log_t, log_ne, bins=(log_t_edges, log_ne_edges),
                    weights=selected_luminosity,
                )[0]
                t_lum_hist += np.histogram(
                    log_t, bins=log_t_edges, weights=selected_luminosity,
                )[0]
                ne_lum_hist += np.histogram(
                    log_ne, bins=log_ne_edges, weights=selected_luminosity,
                )[0]

            elapsed = time.perf_counter() - started
            rate = slab_number / elapsed
            eta = (n_slabs - slab_number) / rate if rate > 0.0 else np.nan
            print(
                f"[{slab_number:02d}/{n_slabs:02d}] "
                f"elapsed={elapsed / 60.0:.2f} min ETA={eta / 60.0:.2f} min",
                flush=True,
            )
    finally:
        for handle in handles.values():
            handle.close()

    fractions = {}
    for branch in branch_names:
        positive_count = counts[branch]["positive_emissivity_cells"]
        positive_luminosity = luminosities[branch]["positive_emissivity_cells"]
        fractions[branch] = {}
        for category in category_names:
            fractions[branch][category] = {
                "per_positive_emissivity_cell": _fraction(
                    counts[branch][category], positive_count,
                ),
                "per_positive_Halpha_luminosity": _fraction(
                    luminosities[branch][category], positive_luminosity,
                ),
            }

    quantiles = {}
    for quantile in (0.5, 0.9, 0.99, 0.999):
        label = f"p{100.0 * quantile:g}"
        quantiles[label] = {
            "temperature_K": 10.0 ** _weighted_quantile_from_hist(
                log_t_edges, t_lum_hist, quantile,
            ),
            "electron_density_cm-3": 10.0 ** _weighted_quantile_from_hist(
                log_ne_edges, ne_lum_hist, quantile,
            ),
        }

    report = {
        "dataset": str(args.dataset),
        "despotic_table": str(args.despotic_table),
        "cache_paths": cache_paths,
        "grid_shape": list(dimensions),
        "pipeline_temperature_policy": (
            "T_DESPOTIC if T_QUOKKA < 3000 K; T_QUOKKA otherwise"
        ),
        "pipeline_electron_policy": (
            "DESPOTIC n_e/n_H+ if T_QUOKKA < 3000 K; "
            "QUOKKA-mu n_e/n_H+ otherwise"
        ),
        "draine_temperature_validity_K": [T_MIN_K, T_MAX_K],
        "draine_density_statement_checked": "n_e <= 1e6 cm^-3",
        "counts": counts,
        "Halpha_luminosity_erg_s-1": luminosities,
        "fractions": fractions,
        "Halpha_luminosity_weighted_quantiles": quantiles,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    json_path = args.output_dir / "halpha_draine_fit_validity.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")

    npz_path = args.output_dir / "halpha_draine_fit_validity_histograms.npz"
    np.savez_compressed(
        npz_path,
        log10_temperature_edges=log_t_edges,
        log10_electron_density_edges=log_ne_edges,
        cell_count_histogram=cell_hist,
        Halpha_luminosity_histogram=luminosity_hist,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), sharex=True, sharey=True)
    panels = (
        (cell_hist, "Positive-emissivity cell count"),
        (luminosity_hist, r"H$\alpha$ luminosity [erg s$^{-1}$]"),
    )
    for axis, (histogram, title) in zip(axes, panels):
        positive_values = histogram[histogram > 0.0]
        norm = None
        if positive_values.size:
            norm = LogNorm(vmin=float(positive_values.min()), vmax=float(positive_values.max()))
        image = axis.pcolormesh(
            log_t_edges, log_ne_edges, histogram.T,
            shading="auto", cmap="viridis", norm=norm,
        )
        axis.axvline(np.log10(T_MIN_K), color="white", linestyle="--", linewidth=1.2)
        axis.axvline(np.log10(T_MAX_K), color="white", linestyle="--", linewidth=1.2)
        axis.axhline(np.log10(NE_MAX_CM3), color="white", linestyle=":", linewidth=1.2)
        axis.set_title(title)
        axis.set_xlabel(r"$\log_{10}(T_{\rm used}/{\rm K})$")
        axis.set_ylabel(r"$\log_{10}(n_e/{\rm cm}^{-3})$")
        fig.colorbar(image, ax=axis, pad=0.02)
    fig.suptitle(r"Pipeline H$\alpha$ inputs and Draine (2011) fit range")
    fig.tight_layout()
    png_path = args.output_dir / "halpha_draine_fit_validity.png"
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({
        "report": str(json_path),
        "plot": str(png_path),
        "inside_temperature_and_density_cell_fraction": fractions["all"]["temperature_and_density_valid"]["per_positive_emissivity_cell"],
        "inside_temperature_and_density_Halpha_fraction": fractions["all"]["temperature_and_density_valid"]["per_positive_Halpha_luminosity"],
    }, indent=2))


if __name__ == "__main__":
    main()
