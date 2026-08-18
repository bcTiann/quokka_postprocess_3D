"""Test sigmaNT=0 versus 2 km/s in the production GOW/LVG solver.

Representative inputs are actual T_QUOKKA<3000 K simulation cells selected
at luminosity-weighted T_DESPOTIC quantiles.  Both solves keep the same n_H,
N_H, dV/dr, initial temperature, GOW chemistry, and LVG geometry; only sigmaNT
changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import yt
from yt.units.physical_constants import kb, mh

from quokka2s.pipeline.cache import (
    cache_root_for_dataset,
    compute_cache_key,
    field_cache_key,
    field_cache_path,
)
from quokka2s.pipeline.prep import config as cfg
from quokka2s.tables.solver import solve_gow_lvg_point


FIELDS = {
    "column_density_H": ("gas", "column_density_H"),
    "dVdr_lvg": ("gas", "dVdr_lvg"),
    "temperature_despotic": ("gas", "temperature_despotic"),
    "Cplus_luminosity_despotic": ("gas", "C+_luminosity_despotic"),
}
QUANTILES = np.asarray((0.10, 0.30, 0.50, 0.70, 0.90))
SIGMA_VALUES = (0.0, 2.0e5)


def _open_caches(dataset: Path, despotic_table: Path, shape: tuple[int, ...]):
    base_key = compute_cache_key(
        dataset_path=dataset,
        despotic_table_path=despotic_table,
        downsample_factor=cfg.DOWNSAMPLE_FACTOR,
        column_extension_lateral_kpc=cfg.COLUMN_EXTENSION_LATERAL_KPC,
    )
    root = cache_root_for_dataset(dataset)
    handles = {}
    for label, field in FIELDS.items():
        path = field_cache_path(root, field).resolve()
        handle = h5py.File(path, "r")
        if str(handle.attrs.get("cache_key", "")) != field_cache_key(base_key, field):
            handle.close()
            raise RuntimeError(f"stale field cache: {path}")
        if tuple(handle["data"].shape) != shape:
            handle.close()
            raise ValueError(f"field-cache shape mismatch: {path}")
        handles[label] = handle
    return handles


def _weighted_quantile_indices(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order], dtype=float)
    if cumulative[-1] <= 0.0:
        raise ValueError("non-positive total luminosity weight")
    positions = np.searchsorted(cumulative, QUANTILES * cumulative[-1], side="left")
    return order[np.clip(positions, 0, order.size - 1)]


def _solve(point: dict[str, float], sigma_nt_cms: float) -> dict[str, float | bool]:
    lines, abundances, mu, _cv, _eint, temperature, _energy, failed = solve_gow_lvg_point(
        point["n_H_cm3"], point["N_H_cm2"], point["dVdr_s-1"],
        species=("C+",), abundance_only=("e-", "H+", "H2", "H"),
        Tg_init=100.0, sigma_nt_cms=sigma_nt_cms,
    )
    line = lines["C+"]
    sound_speed_sq = float(kb.to_value("erg/K")) * temperature / (
        mu * float(mh.to_value("g"))
    )
    clumping = np.sqrt(1.0 + 0.75 * sigma_nt_cms**2 / sound_speed_sq)
    return {
        "failed": bool(failed),
        "sigmaNT_cms": float(sigma_nt_cms),
        "temperature_K": float(temperature),
        "mu": float(mu),
        "clumping_factor": float(clumping),
        "Cplus_abundance_per_H": float(abundances["C+"]),
        "Cplus_lumPerH_erg_s-1_H-1": float(line.lumPerH),
        "Cplus_tau": float(line.tau),
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    if not (np.isfinite(numerator) and np.isfinite(denominator)) or denominator == 0.0:
        return None
    return float(numerator / denominator)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument("--despotic-table", type=Path, default=Path(cfg.DESPOTIC_TABLE_PATH))
    parser.add_argument(
        "--output", type=Path,
        default=root / "output/plt0655228_down1_Lext15kpc/despotic_sigmaNT_LVG_sensitivity.json",
    )
    parser.add_argument("--slab-nz", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in ("dataset", "despotic_table", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite: {args.output}")

    ds = yt.load(str(args.dataset))
    ds.force_periodicity()
    shape = tuple(int(value) for value in ds.domain_dimensions)
    nx, ny, nz = shape
    caches = _open_caches(args.dataset, args.despotic_table, shape)
    hydrogen_mass_g = float(mh.to_value("g"))
    collected = {name: [] for name in (
        "n_H_cm3", "N_H_cm2", "dVdr_s-1", "T_QUOKKA_K",
        "T_DESPOTIC_cached_K", "Cplus_luminosity_weight", "flat_index",
    )}
    try:
        for iz in range(0, nz, args.slab_nz):
            local_nz = min(args.slab_nz, nz - iz)
            left_edge = ds.domain_left_edge.copy()
            left_edge[2] += iz * (ds.domain_width[2] / ds.domain_dimensions[2])
            grid = ds.covering_grid(
                level=ds.max_level, left_edge=left_edge,
                dims=(nx, ny, local_nz),
            )
            density = np.asarray(grid[("gas", "density")].to("g/cm**3"), dtype=float)
            temperature_qk = np.asarray(grid[("boxlib", "temperature")], dtype=float)
            del grid
            n_h = density * float(cfg.X_H) / hydrogen_mass_g
            column = np.asarray(
                caches["column_density_H"]["data"][:, :, iz:iz + local_nz], dtype=float,
            )
            dvdr = np.asarray(
                caches["dVdr_lvg"]["data"][:, :, iz:iz + local_nz], dtype=float,
            )
            temperature_dsp = np.asarray(
                caches["temperature_despotic"]["data"][:, :, iz:iz + local_nz], dtype=float,
            )
            luminosity = np.asarray(
                caches["Cplus_luminosity_despotic"]["data"][:, :, iz:iz + local_nz], dtype=float,
            )
            valid = (
                (temperature_qk < 3000.0) & (luminosity > 0.0)
                & np.isfinite(n_h) & np.isfinite(column) & np.isfinite(dvdr)
                & np.isfinite(temperature_dsp) & np.isfinite(luminosity)
                & (n_h > 0.0) & (column > 0.0) & (dvdr > 0.0)
                & (temperature_dsp > 0.0)
            )
            if not np.any(valid):
                continue
            local_flat = np.flatnonzero(valid)
            ix, iy, local_iz = np.unravel_index(local_flat, valid.shape)
            global_flat = (ix * ny + iy) * nz + (iz + local_iz)
            for name, array in (
                ("n_H_cm3", n_h), ("N_H_cm2", column), ("dVdr_s-1", dvdr),
                ("T_QUOKKA_K", temperature_qk),
                ("T_DESPOTIC_cached_K", temperature_dsp),
                ("Cplus_luminosity_weight", luminosity),
            ):
                collected[name].append(np.asarray(array[valid], dtype=float))
            collected["flat_index"].append(global_flat.astype(np.int64))
    finally:
        for handle in caches.values():
            handle.close()

    arrays = {name: np.concatenate(parts) for name, parts in collected.items()}
    selected = _weighted_quantile_indices(
        arrays["T_DESPOTIC_cached_K"], arrays["Cplus_luminosity_weight"],
    )
    reports = []
    for quantile, index in zip(QUANTILES, selected):
        point = {
            name: float(arrays[name][index])
            for name in (
                "n_H_cm3", "N_H_cm2", "dVdr_s-1", "T_QUOKKA_K",
                "T_DESPOTIC_cached_K", "Cplus_luminosity_weight",
            )
        }
        point["flat_index"] = int(arrays["flat_index"][index])
        zero = _solve(point, SIGMA_VALUES[0])
        canonical = _solve(point, SIGMA_VALUES[1])
        reports.append({
            "luminosity_weighted_TDESPOTIC_quantile": float(quantile),
            "simulation_cell": point,
            "sigmaNT_0": zero,
            "sigmaNT_2kms": canonical,
            "ratios_2kms_over_0": {
                "temperature": _ratio(canonical["temperature_K"], zero["temperature_K"]),
                "Cplus_abundance": _ratio(
                    canonical["Cplus_abundance_per_H"], zero["Cplus_abundance_per_H"],
                ),
                "Cplus_lumPerH": _ratio(
                    canonical["Cplus_lumPerH_erg_s-1_H-1"],
                    zero["Cplus_lumPerH_erg_s-1_H-1"],
                ),
                "Cplus_tau": _ratio(canonical["Cplus_tau"], zero["Cplus_tau"]),
            },
        })

    result = {
        "dataset": str(args.dataset),
        "despotic_table": str(args.despotic_table),
        "selection": (
            "actual T_QUOKKA<3000 K cells at luminosity-weighted "
            "T_DESPOTIC quantiles"
        ),
        "eligible_cells": int(arrays["n_H_cm3"].size),
        "fixed_between_solves": "n_H, N_H, dVdr, Tg_init=100 K, GOW, LVG",
        "only_changed": "sigmaNT: 0 versus 2e5 cm/s",
        "points": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
