#!/usr/bin/env python3
"""Save edge-on and face-on gas maps using the available mixed temperatures.

Build: python tools/figures/build_gas_projection_maps.py --config configs/emission_process.yaml
       --output-dir output/gas_projection_maps --no-plot
Replot: python tools/figures/build_gas_projection_maps.py
        --output-dir output/gas_projection_maps --plot-only
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT / 'src'))

from quokka2s.constants import HYDROGEN_MASS_G
from quokka2s.figures.gas_projections import plot_gas_projection_maps
from quokka2s.physics.settings import X_H
from quokka2s.processing_inputs import load_processing_inputs
from quokka2s.products.gas_projections import MultiviewAccumulator
from quokka2s.run_settings import load_process_config
from quokka2s.snapshot_reader import slab_windows


def accumulate_projection_slab(
    snapshot, emission_calculator, accumulator, x_start, x_stop, query_chunk,
):
    """Read one full-y/z slab, query its batches, then accumulate both views.

    The temporary mixed-temperature and valid arrays have one value per slab
    cell. They are discarded with the slab when this function returns.
    emission_calculator is shared across batches; calculate(cells=...) returns
    each batch's DESPOTIC temperature [K] and QUOKKA temperature branch.
    Hot cells remain available even when DESPOTIC or a line lookup fails.
    """
    slab = snapshot.read_slab(
        x_start=x_start,
        x_stop=x_stop,
    )
    mixed_temperature = np.empty(slab.cell_count)
    valid = np.empty(slab.cell_count, dtype=bool)
    for start in range(0, slab.cell_count, query_chunk):
        stop = min(start + query_chunk, slab.cell_count)
        cells = slab.batch(
            start=start,
            stop=stop,
        )
        emission = emission_calculator.calculate(cells=cells)
        batch_temperature = np.where(
            emission.cold_cells,
            emission.despotic_temperature_K,
            cells.temperature_QUOKKA_K,
        )
        mixed_temperature[start:stop] = batch_temperature
        valid[start:stop] = np.isfinite(batch_temperature) & (batch_temperature > 0.0)
        del cells, emission

    # Restore (slab_nx, Ny, Nz) for the x- and z-axis projection sums.
    density = slab.density_g_cm3.reshape(slab.shape)
    velocity_z = slab.velocity_z_kms.reshape(slab.shape)
    temperature_quokka = slab.temperature_QUOKKA_K.reshape(slab.shape)
    mixed_temperature = mixed_temperature.reshape(slab.shape)
    valid = valid.reshape(slab.shape)
    accumulator.add(
        ix=x_start,
        rho=density,
        vz_kms=velocity_z,
        tq_K=temperature_quokka,
        mixed_K=mixed_temperature,
        valid=valid,
    )

    # Independent cell totals are used to check the projected map integrals.
    cold = temperature_quokka < 3000.0
    counts = {}
    masses = {}
    selections = {
        'all': np.ones(slab.shape, dtype=bool),
        'cold': cold,
        'hot': ~cold,
        'excluded': ~valid,
        'retained': valid,
    }
    for name, selected in selections.items():
        counts[name] = int(np.count_nonzero(selected))
        masses[name] = float(density[selected].sum() * snapshot.cell_volume_cm3)
    counts['excluded_cold'] = int(np.count_nonzero(cold & ~valid))
    return counts, masses


def process_gas_projection_maps(snapshot, emission_calculator, settings):
    """Keep only the accumulated 2D maps while streaming through all x slabs."""
    if snapshot.processing_shape != snapshot.shape:
        raise ValueError(
            'Gas projection maps require full x and y ranges; use the standard '
            'process/plot workflow for region line images, spectra, and gas phases'
        )
    accumulator = MultiviewAccumulator(
        shape=snapshot.shape,
        widths_cm=snapshot.cell_widths.to('cm').value,
    )
    counts = {name: 0 for name in ('all', 'cold', 'hot', 'excluded', 'retained', 'excluded_cold')}
    masses = {name: 0.0 for name in ('all', 'cold', 'hot', 'excluded', 'retained')}
    selected_slabs = list(slab_windows(snapshot.shape[0], settings.slab_nx))
    began = time.monotonic()
    for number, (x_start, x_stop) in enumerate(selected_slabs, start=1):
        slab_counts, slab_masses = accumulate_projection_slab(
            snapshot=snapshot,
            emission_calculator=emission_calculator,
            accumulator=accumulator,
            x_start=x_start,
            x_stop=x_stop,
            query_chunk=settings.query_chunk,
        )
        for name in counts:
            counts[name] += slab_counts[name]
        for name in masses:
            masses[name] += slab_masses[name]
        elapsed = time.monotonic() - began
        fraction = number / len(selected_slabs)
        remaining = elapsed * (1.0 - fraction) / fraction
        print(
            f'Gas projection maps: {100 * fraction:.1f}% '
            f'({number}/{len(selected_slabs)} slabs); '
            f'elapsed {elapsed / 60:.1f} min; ETA ~{remaining / 60:.1f} min',
            flush=True,
        )
    return accumulator, counts, masses


def add_view_extents_and_particles(payload, snapshot):
    """Add kpc extents and edge-view stars within x_mid +/- L_x/40.

    The particle slab therefore has total thickness L_x/20, preserving the
    existing figure's selection. Particles are shown only in the edge view.
    """
    dataset = snapshot.dataset
    left = dataset.domain_left_edge.to('kpc').value
    right = dataset.domain_right_edge.to('kpc').value
    payload['extent_edge_kpc'] = np.array([left[1], right[1], left[2], right[2]])
    payload['extent_face_kpc'] = np.array([left[0], right[0], left[1], right[1]])
    particles = dataset.all_data()
    particle_type = 'StochasticStellarPop_particles'
    positions = []
    for axis in 'xyz':
        position = particles[particle_type, 'particle_position_' + axis].to('kpc')
        positions.append(np.asarray(position))
    x_mid = (left[0] + right[0]) / 2.0
    half_depth = (right[0] - left[0]) / 40.0
    selected = np.abs(positions[0] - x_mid) <= half_depth
    payload['particles_edge_y_kpc'] = positions[1][selected]
    payload['particles_edge_z_kpc'] = positions[2][selected]
    return {
        'type': particle_type,
        'total': len(selected),
        'shown': int(np.count_nonzero(selected)),
        'view': 'edge only',
        'x_slab_kpc': [float(x_mid - half_depth), float(x_mid + half_depth)],
    }


def check_projection_conservation(payload, volume_report, masses):
    """Check both projected views against independently accumulated cell totals."""
    widths = np.asarray(volume_report['cell_widths_cm'])
    validation = {}
    for selection, mass_key in (('all', 'all'), ('valid', 'retained')):
        totals = volume_report['selections'][selection]
        for view, area in (('edge', widths[1] * widths[2]), ('face', widths[0] * widths[1])):
            column_mass = payload[f'{selection}_{view}_sigma_g_cm2']
            mass_from_map = float(column_mass.sum() * area)
            np.testing.assert_allclose(mass_from_map, masses[mass_key], rtol=1e-12)
            validation[f'{selection}_{view}_mass_g'] = mass_from_map
            for field, total_key in (
                ('vz_kms', 'momentum_z_g_kms'),
                ('T_quokka_K', 'T_quokka_mass_g_K'),
                ('T_mixed_K', 'T_mixed_mass_g_K'),
            ):
                average = payload[f'{selection}_{view}_{field}']
                numerator = np.where(column_mass > 0.0, average, 0.0) * column_mass * area
                summed = float(numerator.sum())
                np.testing.assert_allclose(
                    summed,
                    totals[total_key],
                    rtol=1e-12,
                    atol=float(np.abs(numerator).sum() * 1e-14),
                )
                validation[f'{selection}_{view}_{total_key}'] = summed
    return validation


def save_projection_data(payload, report, output_dir):
    """Save native-resolution map arrays and their calculation summary."""
    np.savez_compressed(output_dir / 'multiview_maps.npz', **payload)
    (output_dir / 'multiview_report.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )


def render_projection_figures(payload, report, stem):
    """Draw the saved maps and write their display ranges beside the figures."""
    display = plot_gas_projection_maps(payload=payload, report=report, stem=stem)
    stem.with_name(stem.name + '_display.json').write_text(
        json.dumps(display, indent=2) + '\n',
        encoding='utf-8',
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/emission_process.yaml')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--figure-stem', type=Path)
    parser.add_argument('--plot-only', action='store_true', help='Render saved multiview_maps.npz only')
    parser.add_argument('--no-plot', action='store_true', help='Save maps without rendering figures')
    args = parser.parse_args()
    if args.plot_only and args.no_plot:
        parser.error('--plot-only and --no-plot cannot be used together')
    output_dir = args.output_dir.resolve()
    figure_stem = args.figure_stem or output_dir / 'multiview_figure_particles'
    if args.plot_only:
        report = json.loads((output_dir / 'multiview_report.json').read_text())
        with np.load(output_dir / 'multiview_maps.npz', allow_pickle=False) as data:
            payload = {key: np.array(data[key]) for key in data.files}
    else:
        settings = load_process_config(args.config)
        settings.output_dir = output_dir
        if settings.max_slabs is not None:
            parser.error('Gas projections require the full snapshot; remove max_slabs from the config')
        snapshot, emission_calculator = load_processing_inputs(settings)
        accumulator, counts, masses = process_gas_projection_maps(
            snapshot=snapshot,
            emission_calculator=emission_calculator,
            settings=settings,
        )
        payload = accumulator.payload()
        particle_report = add_view_extents_and_particles(payload=payload, snapshot=snapshot)
        volume_report = accumulator.report()
        validation = check_projection_conservation(
            payload=payload,
            volume_report=volume_report,
            masses=masses,
        )
        report = {
            'status': 'completed',
            'completed_at': datetime.now(timezone.utc).isoformat(),
            'dataset': str(settings.dataset),
            'despotic_table': str(settings.despotic_table),
            'cloudy_table': str(settings.cloudy_table),
            'full_snapshot': True,
            'shape': list(snapshot.shape),
            'counts': counts,
            'mass_g': masses,
            'X_H': X_H,
            'hydrogen_mass_g': HYDROGEN_MASS_G,
            'temperature': 'DESPOTIC for T_QUOKKA < 3000 K; QUOKKA otherwise, then density weighted',
            'display_selection': 'Mixed temperature available: DESPOTIC for cold cells; QUOKKA for hot cells',
            'density': 'Central plane slices at x index nx//2 and z index nz//2',
            'column': 'Integral nH dl along x (edge) or z (face)',
            'velocity': 'Density-weighted original vz along each view',
            'particles': particle_report,
            'accumulator': volume_report,
            'validation': validation,
        }
        save_projection_data(payload=payload, report=report, output_dir=output_dir)
    if not args.no_plot:
        render_projection_figures(payload=payload, report=report, stem=figure_stem)


if __name__ == '__main__':
    main()
