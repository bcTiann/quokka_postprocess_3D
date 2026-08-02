"""Conservative Sobolev optical-depth upper bound for hot QUOKKA cells.

The bound assumes that every gas-phase carbon nucleus is C+ and that every C+
ion is in the lower fine-structure level.  It therefore needs no additional
Cloudy calculation.  Cell statistics apply to every T_QUOKKA >= 3000 K cell.
Luminosity-weighted statistics use the currently cached production-hybrid C+
field and explicitly record the Cloudy table used to create that cache.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import yt
from yt.units.physical_constants import mh

from quokka2s.pipeline.cache import (
    cache_root_for_dataset,
    compute_cache_key,
    field_cache_key,
    field_cache_path,
)
from quokka2s.pipeline.prep import config as cfg
from quokka2s.pipeline.prep.physics_fields import DVDR_FLOOR


TEMPERATURE_CUTOFF_K = 3000.0
A_UL_S_INV = 2.3e-6
WAVELENGTH_MICRON = 157.636
G_UPPER = 4.0
G_LOWER = 2.0
CARBON_PER_H = float(cfg.A_C)
LSUN_ERG_S = 3.828e33
TAU_THRESHOLDS = (0.1, 1.0)
LOG_TAU_EDGES = np.linspace(-20.0, 12.0, 32_001)

DENSITY_GRADIENT_FIELD = ('gas', 'dVdr_lvg')
EMISSIVITY_FIELD = ('gas', 'C+_luminosity')


def _tau_coefficient_cm3_s() -> float:
    wavelength_cm = WAVELENGTH_MICRON * 1.0e-4
    return (
        A_UL_S_INV * wavelength_cm**3 / (8.0 * np.pi)
        * (G_UPPER / G_LOWER) * CARBON_PER_H
    )


def _open_cache(
    cache_root: Path,
    base_key: str,
    field: tuple[str, str],
) -> h5py.File:
    path = field_cache_path(cache_root, field)
    if not path.exists():
        raise FileNotFoundError(f'missing field cache: {path}')
    handle = h5py.File(path, 'r')
    expected_key = field_cache_key(base_key, field)
    actual_key = str(handle.attrs.get('cache_key', ''))
    actual_field = (
        str(handle.attrs.get('field_type', '')),
        str(handle.attrs.get('field_name', '')),
    )
    if actual_key != expected_key or actual_field != field:
        handle.close()
        raise RuntimeError(
            f'stale/mismatched cache {path}: expected {expected_key} {field}, '
            f'found {actual_key} {actual_field}'
        )
    return handle


def _histogram_percentiles(
    histogram: np.ndarray,
    percentiles: tuple[float, ...] = (0.5, 0.9, 0.99),
) -> dict[str, float | None]:
    total = float(histogram.sum())
    if not total > 0.0:
        return {f'p{int(q * 100):02d}': None for q in percentiles}
    cumulative = np.cumsum(histogram)
    centers = 0.5 * (LOG_TAU_EDGES[:-1] + LOG_TAU_EDGES[1:])
    output: dict[str, float | None] = {}
    for q in percentiles:
        index = int(np.searchsorted(cumulative, q * total, side='left'))
        index = min(index, centers.size - 1)
        output[f'p{int(q * 100):02d}'] = float(10.0 ** centers[index])
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=cfg.YT_DATASET_PATH)
    parser.add_argument('--despotic-table', default=cfg.DESPOTIC_TABLE_PATH)
    parser.add_argument('--cloudy-table', default=cfg.CLOUDY_CII_TABLE_PATH)
    parser.add_argument('--slab-nz', type=int, default=32)
    parser.add_argument(
        '--output',
        default=str(
            Path(cfg.OUTPUT_DIR) / 'cloudy_cii_sobolev_upper_bound.json'
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.slab_nz <= 0:
        raise ValueError('--slab-nz must be positive')

    dataset_path = Path(args.dataset).resolve()
    cloudy_table_path = Path(args.cloudy_table).resolve()
    with np.load(cloudy_table_path, allow_pickle=False) as cloudy_table:
        cloudy_t_bounds = [
            float(10.0 ** cloudy_table['log_T'][0]),
            float(10.0 ** cloudy_table['log_T'][-1]),
        ]

    base_key = compute_cache_key(
        dataset_path=dataset_path,
        despotic_table_path=args.despotic_table,
        downsample_factor=cfg.DOWNSAMPLE_FACTOR,
        column_extension_lateral_kpc=cfg.COLUMN_EXTENSION_LATERAL_KPC,
    )
    cache_root = cache_root_for_dataset(dataset_path)
    dvdr_file = _open_cache(cache_root, base_key, DENSITY_GRADIENT_FIELD)
    emissivity_file = _open_cache(cache_root, base_key, EMISSIVITY_FIELD)

    ds = yt.load(str(dataset_path))
    if cfg.DOWNSAMPLE_FACTOR != 1:
        dvdr_file.close()
        emissivity_file.close()
        raise NotImplementedError('this diagnostic currently requires downsample=1')
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    if (
        dvdr_file['data'].shape != dimensions
        or emissivity_file['data'].shape != dimensions
    ):
        dvdr_file.close()
        emissivity_file.close()
        raise ValueError('field-cache shape does not match the dataset')

    nx, ny, nz = dimensions
    cell_width = ds.domain_width.to('cm') / ds.domain_dimensions
    cell_volume_cm3 = float(np.prod(np.asarray(cell_width, dtype=float)))
    hydrogen_mass_g = float(mh.to_value('g'))
    coefficient = _tau_coefficient_cm3_s()

    high_count = 0
    emitting_count = 0
    high_above_weighting_tmax = 0
    floor_count = 0
    max_tau = 0.0
    max_emitting_tau = 0.0
    cell_threshold_counts = {str(value): 0 for value in TAU_THRESHOLDS}
    emitting_threshold_counts = {str(value): 0 for value in TAU_THRESHOLDS}
    luminosity_threshold_sums = {str(value): 0.0 for value in TAU_THRESHOLDS}
    luminosity_sum = 0.0
    floor_luminosity_sum = 0.0
    cell_histogram = np.zeros(LOG_TAU_EDGES.size - 1, dtype=np.int64)
    luminosity_histogram = np.zeros(LOG_TAU_EDGES.size - 1, dtype=float)
    start = time.perf_counter()

    try:
        for slab_number, iz in enumerate(range(0, nz, args.slab_nz), start=1):
            slab_nz = min(args.slab_nz, nz - iz)
            left_edge = ds.domain_left_edge.copy()
            left_edge[2] += iz * cell_width[2]
            grid = ds.covering_grid(
                level=ds.max_level,
                left_edge=left_edge,
                dims=(nx, ny, slab_nz),
            )
            temperature = np.asarray(
                grid[('boxlib', 'temperature')], dtype=float,
            )
            density = np.asarray(
                grid[('gas', 'density')].to('g/cm**3'), dtype=float,
            )
            dvdr = np.asarray(
                dvdr_file['data'][:, :, iz:iz + slab_nz], dtype=float,
            )
            emissivity = np.asarray(
                emissivity_file['data'][:, :, iz:iz + slab_nz], dtype=float,
            )

            high = temperature >= TEMPERATURE_CUTOFF_K
            if not high.any():
                continue
            high_count += int(high.sum())
            high_above_weighting_tmax += int(
                (high & (temperature > cloudy_t_bounds[1] * (1.0 + 1e-12))).sum()
            )

            n_h = density[high] * float(cfg.X_H) / hydrogen_mass_g
            high_dvdr = dvdr[high]
            if not (
                np.isfinite(n_h).all()
                and np.isfinite(high_dvdr).all()
                and (n_h > 0.0).all()
                and (high_dvdr > 0.0).all()
            ):
                raise ValueError('non-positive/non-finite nH or dVdr in hot cells')
            tau = coefficient * n_h / high_dvdr
            high_emissivity = emissivity[high]
            if np.any(~np.isfinite(high_emissivity)) or np.any(high_emissivity < 0.0):
                raise ValueError('Cloudy luminosity weights are invalid')

            emitting = high_emissivity > 0.0
            emitting_count += int(emitting.sum())
            luminosity_sum += float(high_emissivity.sum())
            max_tau = max(max_tau, float(tau.max()))
            if emitting.any():
                max_emitting_tau = max(
                    max_emitting_tau, float(tau[emitting].max())
                )

            at_floor = high_dvdr <= DVDR_FLOOR * (1.0 + 1e-12)
            floor_count += int(at_floor.sum())
            floor_luminosity_sum += float(high_emissivity[at_floor].sum())
            for threshold in TAU_THRESHOLDS:
                key = str(threshold)
                above = tau > threshold
                cell_threshold_counts[key] += int(above.sum())
                emitting_threshold_counts[key] += int((above & emitting).sum())
                luminosity_threshold_sums[key] += float(
                    high_emissivity[above].sum()
                )

            log_tau = np.log10(tau)
            cell_histogram += np.histogram(log_tau, bins=LOG_TAU_EDGES)[0]
            luminosity_histogram += np.histogram(
                log_tau, bins=LOG_TAU_EDGES, weights=high_emissivity,
            )[0]
            print(
                f'[{slab_number}/{(nz + args.slab_nz - 1) // args.slab_nz}] '
                f'z={iz}:{iz + slab_nz} hot={high_count:,}',
                flush=True,
            )
    finally:
        dvdr_file.close()
        emissivity_file.close()

    elapsed = time.perf_counter() - start
    luminosity_lsun = luminosity_sum * cell_volume_cm3 / LSUN_ERG_S
    result = {
        'dataset': str(dataset_path),
        'temperature_selection': 'T_QUOKKA >= 3000 K',
        'method': 'strict upper bound: all gas-phase carbon is C+ in lower level',
        'formula': 'tau_max = coefficient * n_H / dVdr_lvg',
        'tau_coefficient_cm3_s': coefficient,
        'atomic_inputs': {
            'A_ul_s-1': A_UL_S_INV,
            'wavelength_micron': WAVELENGTH_MICRON,
            'g_upper': G_UPPER,
            'g_lower': G_LOWER,
            'carbon_per_H': CARBON_PER_H,
        },
        'gradient_definition': (
            f'abs(divergence(v))/3 floored at {DVDR_FLOOR:.1e} s^-1'
        ),
        'luminosity_weighting': {
            'field_cache': 'gas/C+_luminosity',
            'cloudy_table': str(cloudy_table_path),
            'cloudy_table_temperature_bounds_K': cloudy_t_bounds,
            'warning': (
                'weighted statistics are provisional because this cached '
                'legacy Cloudy table is zero above its maximum temperature'
            ),
        },
        'counts': {
            'hot_cells': high_count,
            'hot_cells_above_weighting_table_Tmax': high_above_weighting_tmax,
            'hot_cells_with_positive_cached_cloudy_emissivity': emitting_count,
            'hot_cells_at_dVdr_floor': floor_count,
            'tau_above_threshold': cell_threshold_counts,
            'positive_emissivity_cells_tau_above_threshold': (
                emitting_threshold_counts
            ),
        },
        'fractions': {
            'cells_tau_above_threshold': {
                key: value / high_count
                for key, value in cell_threshold_counts.items()
            },
            'positive_emissivity_cells_tau_above_threshold': {
                key: value / emitting_count if emitting_count else None
                for key, value in emitting_threshold_counts.items()
            },
            'cached_cloudy_luminosity_tau_above_threshold': {
                key: value / luminosity_sum if luminosity_sum > 0.0 else None
                for key, value in luminosity_threshold_sums.items()
            },
            'hot_cells_at_dVdr_floor': floor_count / high_count,
            'cached_cloudy_luminosity_at_dVdr_floor': (
                floor_luminosity_sum / luminosity_sum
                if luminosity_sum > 0.0 else None
            ),
        },
        'tau_upper_bound_statistics': {
            'all_hot_cells': {
                **_histogram_percentiles(cell_histogram),
                'maximum': max_tau,
            },
            'cached_cloudy_luminosity_weighted': {
                **_histogram_percentiles(luminosity_histogram),
                'maximum_among_positive_emissivity_cells': max_emitting_tau,
            },
        },
        'cached_cloudy_luminosity_Lsun': luminosity_lsun,
        'elapsed_minutes': elapsed / 60.0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Wrote {output}')
    print(json.dumps(result['tau_upper_bound_statistics'], indent=2))
    print(json.dumps(result['fractions'], indent=2))


if __name__ == '__main__':
    main()
