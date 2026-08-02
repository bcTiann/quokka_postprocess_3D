"""Measure how simulation cells depend on filled Cloudy [C II] nodes.

This diagnostic reads the raw QUOKKA density/temperature in z slabs and pairs
them with the cached column-density and hybrid [C II] emissivity fields.  It
does not modify the simulation, lookup table, or field caches.
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

from ...cloudy_cii_lookup import CloudyCIILookup
from ..cache import (
    cache_root_for_dataset,
    compute_cache_key,
    field_cache_key,
    field_cache_path,
)
from . import config as cfg


_COLUMN_FIELD = ('gas', 'column_density_H')
_EMISSIVITY_FIELD = ('gas', 'C+_luminosity')
_TOUCH_EPS = 1.0e-12
_LSUN_ERG_S = 3.828e33


def _open_validated_cache(
    path: Path,
    expected_key: str,
    expected_field: tuple[str, str],
) -> h5py.File:
    if not path.exists():
        raise FileNotFoundError(
            f'Required field cache does not exist: {path}\n'
            'Run the normal C+ pipeline compute first.'
        )
    handle = h5py.File(path, 'r')
    actual_key = str(handle.attrs.get('cache_key', ''))
    actual_field = (
        str(handle.attrs.get('field_type', '')),
        str(handle.attrs.get('field_name', '')),
    )
    if actual_key != expected_key or actual_field != expected_field:
        handle.close()
        raise RuntimeError(
            f'Stale or mismatched field cache: {path}\n'
            f'expected key/field {expected_key} {expected_field}, '
            f'found {actual_key} {actual_field}'
        )
    return handle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Count QUOKKA cells whose Cloudy CII trilinear interpolation '
            'uses one or more originally failed grid nodes.'
        )
    )
    parser.add_argument('--dataset', default=cfg.YT_DATASET_PATH)
    parser.add_argument('--cloudy-table', default=cfg.CLOUDY_CII_TABLE_PATH)
    parser.add_argument('--despotic-table', default=cfg.DESPOTIC_TABLE_PATH)
    parser.add_argument('--slab-nz', type=int, default=32)
    parser.add_argument(
        '--output',
        default=str(Path(cfg.OUTPUT_DIR) / 'cloudy_cii_failure_sampling.json'),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.slab_nz <= 0:
        raise ValueError('--slab-nz must be positive')

    dataset_path = Path(args.dataset).resolve()
    lookup = CloudyCIILookup(args.cloudy_table)
    base_key = compute_cache_key(
        dataset_path=dataset_path,
        despotic_table_path=args.despotic_table,
        downsample_factor=cfg.DOWNSAMPLE_FACTOR,
        column_extension_lateral_kpc=cfg.COLUMN_EXTENSION_LATERAL_KPC,
    )
    cache_root = cache_root_for_dataset(dataset_path)
    column_file = _open_validated_cache(
        field_cache_path(cache_root, _COLUMN_FIELD),
        field_cache_key(base_key, _COLUMN_FIELD),
        _COLUMN_FIELD,
    )
    emissivity_file = _open_validated_cache(
        field_cache_path(cache_root, _EMISSIVITY_FIELD),
        field_cache_key(base_key, _EMISSIVITY_FIELD),
        _EMISSIVITY_FIELD,
    )

    ds = yt.load(str(dataset_path))
    if cfg.DOWNSAMPLE_FACTOR != 1:
        column_file.close()
        emissivity_file.close()
        raise NotImplementedError(
            'This slab diagnostic currently requires DOWNSAMPLE_FACTOR=1.'
        )
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    if column_file['data'].shape != dimensions:
        column_file.close()
        emissivity_file.close()
        raise ValueError(
            f'cache shape {column_file["data"].shape} != dataset {dimensions}'
        )

    nx, ny, nz = dimensions
    cell_width = ds.domain_width.to('cm') / ds.domain_dimensions
    cell_volume_cm3 = float(np.prod(np.asarray(cell_width, dtype=float)))
    t_min, t_max = lookup.temperature_bounds_K
    n_min, n_max = 10.0 ** lookup.log_nH[[0, -1]]
    column_min, column_max = 10.0 ** lookup.log_NH[[0, -1]]
    hydrogen_mass_g = float(mh.to_value('g'))

    counters = {
        'all_cells': 0,
        'cloudy_branch_T_ge_3000': 0,
        'cloudy_table_sampled': 0,
        'temperature_above_table_zero_emissivity': 0,
        'touches_failure_node': 0,
        'failure_weight_ge_0.01': 0,
        'failure_weight_ge_0.10': 0,
        'failure_weight_ge_0.25': 0,
        'failure_weight_ge_0.50': 0,
        'failure_weight_nearly_one': 0,
        'nH_clamped': 0,
        'NH_clamped': 0,
        'either_axis_clamped': 0,
        'touches_failure_and_axis_clamped': 0,
        'hot_outside_planned_full_nH': 0,
        'hot_outside_planned_full_NH': 0,
        'hot_above_planned_full_Tmax': 0,
    }
    hot_ranges = {
        'temperature_K': [np.inf, -np.inf],
        'n_H_cm-3': [np.inf, -np.inf],
        'N_H_cm-2': [np.inf, -np.inf],
    }
    max_failure_weight = 0.0
    sum_failure_weight = 0.0
    emissivity_sum = 0.0
    affected_emissivity_sum = 0.0
    start_time = time.perf_counter()
    n_slabs = (nz + args.slab_nz - 1) // args.slab_nz

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
            n_H = density * float(cfg.X_H) / hydrogen_mass_g
            column = np.asarray(
                column_file['data'][:, :, iz:iz + slab_nz], dtype=float,
            )
            emissivity = np.asarray(
                emissivity_file['data'][:, :, iz:iz + slab_nz], dtype=float,
            )
            del grid, density

            finite_positive = (
                np.isfinite(temperature) & np.isfinite(n_H)
                & np.isfinite(column) & (n_H > 0.0) & (column > 0.0)
            )
            cloudy_branch = finite_positive & (temperature >= 3000.0)
            sampled = cloudy_branch & (temperature >= t_min) & (temperature <= t_max)
            above = cloudy_branch & (temperature > t_max)
            failure_weight = lookup.failure_interpolation_weight(
                temperature, n_H, column,
            )
            touched = sampled & (failure_weight > _TOUCH_EPS)
            n_clamped = sampled & ((n_H < n_min) | (n_H > n_max))
            column_clamped = sampled & (
                (column < column_min) | (column > column_max)
            )
            either_clamped = n_clamped | column_clamped

            counters['all_cells'] += int(temperature.size)
            counters['cloudy_branch_T_ge_3000'] += int(np.count_nonzero(cloudy_branch))
            counters['cloudy_table_sampled'] += int(np.count_nonzero(sampled))
            counters['temperature_above_table_zero_emissivity'] += int(np.count_nonzero(above))
            counters['touches_failure_node'] += int(np.count_nonzero(touched))
            counters['failure_weight_ge_0.01'] += int(np.count_nonzero(sampled & (failure_weight >= 0.01)))
            counters['failure_weight_ge_0.10'] += int(np.count_nonzero(sampled & (failure_weight >= 0.10)))
            counters['failure_weight_ge_0.25'] += int(np.count_nonzero(sampled & (failure_weight >= 0.25)))
            counters['failure_weight_ge_0.50'] += int(np.count_nonzero(sampled & (failure_weight >= 0.50)))
            counters['failure_weight_nearly_one'] += int(np.count_nonzero(sampled & (failure_weight >= 1.0 - 1.0e-10)))
            counters['nH_clamped'] += int(np.count_nonzero(n_clamped))
            counters['NH_clamped'] += int(np.count_nonzero(column_clamped))
            counters['either_axis_clamped'] += int(np.count_nonzero(either_clamped))
            counters['touches_failure_and_axis_clamped'] += int(np.count_nonzero(touched & either_clamped))
            counters['hot_outside_planned_full_nH'] += int(np.count_nonzero(
                cloudy_branch & ((n_H < 1.0e-4) | (n_H > 1.0e6))
            ))
            counters['hot_outside_planned_full_NH'] += int(np.count_nonzero(
                cloudy_branch & ((column < 1.0e15) | (column > 1.0e24))
            ))
            counters['hot_above_planned_full_Tmax'] += int(np.count_nonzero(
                cloudy_branch & (temperature > 1.0e9)
            ))

            if np.any(cloudy_branch):
                for key, values in (
                    ('temperature_K', temperature),
                    ('n_H_cm-3', n_H),
                    ('N_H_cm-2', column),
                ):
                    selected = values[cloudy_branch]
                    hot_ranges[key][0] = min(
                        hot_ranges[key][0], float(np.min(selected)),
                    )
                    hot_ranges[key][1] = max(
                        hot_ranges[key][1], float(np.max(selected)),
                    )

            if np.any(sampled):
                max_failure_weight = max(
                    max_failure_weight, float(np.max(failure_weight[sampled])),
                )
                sum_failure_weight += float(np.sum(failure_weight[sampled]))
                emissivity_sum += float(np.sum(emissivity[sampled]))
                affected_emissivity_sum += float(np.sum(emissivity[touched]))

            elapsed = time.perf_counter() - start_time
            rate = slab_number / elapsed if elapsed > 0.0 else 0.0
            remaining = (n_slabs - slab_number) / rate if rate > 0.0 else np.nan
            print(
                f'[{slab_number:02d}/{n_slabs:02d}] z={iz}:{iz + slab_nz}  '
                f'touched={counters["touches_failure_node"]:,}  '
                f'elapsed={elapsed / 60:.1f} min  '
                f'ETA={remaining / 60:.1f} min',
                flush=True,
            )
    finally:
        column_file.close()
        emissivity_file.close()

    sampled_count = counters['cloudy_table_sampled']
    touched_count = counters['touches_failure_node']
    result = {
        'dataset': str(dataset_path),
        'cloudy_table': str(Path(args.cloudy_table).resolve()),
        'cloudy_grid_shape': list(lookup.failure_mask.shape),
        'original_failure_nodes': int(np.count_nonzero(lookup.failure_mask)),
        'temperature_bounds_K': [float(t_min), float(t_max)],
        'touch_definition': f'trilinear failure weight > {_TOUCH_EPS:g}',
        'counts': counters,
        'fractions': {
            'touched_per_sampled_cell': (
                touched_count / sampled_count if sampled_count else 0.0
            ),
            'axis_clamped_per_sampled_cell': (
                counters['either_axis_clamped'] / sampled_count
                if sampled_count else 0.0
            ),
            'mean_failure_weight_per_sampled_cell': (
                sum_failure_weight / sampled_count if sampled_count else 0.0
            ),
            'affected_cloudy_luminosity': (
                affected_emissivity_sum / emissivity_sum
                if emissivity_sum > 0.0 else 0.0
            ),
        },
        'maximum_failure_weight': max_failure_weight,
        'hot_cell_input_ranges': hot_ranges,
        'planned_full_grid_bounds': {
            'temperature_K': [3000.0, 1.0e9],
            'n_H_cm-3': [1.0e-4, 1.0e6],
            'N_H_cm-2': [1.0e15, 1.0e24],
        },
        'cloudy_luminosity_Lsun': emissivity_sum * cell_volume_cm3 / _LSUN_ERG_S,
        'affected_cloudy_luminosity_Lsun': (
            affected_emissivity_sum * cell_volume_cm3 / _LSUN_ERG_S
        ),
        'elapsed_minutes': (time.perf_counter() - start_time) / 60.0,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')

    print('\nCloudy [C II] failure-node sampling summary')
    print('=' * 52)
    print(f'Original failed table nodes       : {result["original_failure_nodes"]:,}')
    print(f'Cloudy-table-sampled cells        : {sampled_count:,}')
    print(f'Cells touching a failed node      : {touched_count:,} '
          f'({result["fractions"]["touched_per_sampled_cell"]:.6%})')
    print(f'Maximum failure interpolation wt. : {max_failure_weight:.6g}')
    print(f'Cloudy luminosity                 : {result["cloudy_luminosity_Lsun"]:.6g} Lsun')
    print(f'Affected Cloudy luminosity        : '
          f'{result["affected_cloudy_luminosity_Lsun"]:.6g} Lsun '
          f'({result["fractions"]["affected_cloudy_luminosity"]:.6%})')
    print(f'JSON result                       : {output}')


if __name__ == '__main__':
    main()
