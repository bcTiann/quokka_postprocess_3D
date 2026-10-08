#!/usr/bin/env python3
"""Save the five Figure 1 fields at one x index, using the current process setup.

Build: python tools/figures/build_table_input_slice.py --config configs/emission_process.yaml
       --output-dir output/figure1 --slice-index 216
Replot: python tools/figures/build_table_input_slice.py --output-dir output/figure1
        --plot-only
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT / 'src'))

from quokka2s.figures.table_input_slices import plot_table_input_slice
from quokka2s.paths import resolve_path
from quokka2s.products.table_input_slices import prepare_slice_plot_data
from quokka2s.run_settings import DEFAULT_PROCESS_CONFIG, load_process_config


def calculate_slice_fields(snapshot, despotic_reader, slice_index, query_chunk):
    """Return five CGS/K fields and the mixed-temperature display mask at one x layer.

    snapshot.read_slab() handles neighbours and periodic x/y derivatives.
    The returned field arrays have shape (Ny, Nz), normally (256, 2048).
    DESPOTIC temperature comes from the reader's temperature-only query.
    The saved valid mask means mixed temperature is available: cold cells require T_DESPOTIC,
    hot cells use T_QUOKKA regardless of DESPOTIC or line-emissivity failures.
    """
    from quokka2s.physics.gas_fields import HYDROGEN_MASS_G
    from quokka2s.physics.gas_fields import mixed_gas_temperature_K
    from quokka2s.physics.settings import X_H, EMISSION_TEMPERATURE_BOUNDARY_K

    if snapshot.processing_shape != snapshot.shape:
        raise ValueError(
            'Table-input slices require full x and y ranges; use the standard '
            'process/plot workflow for region line images, spectra, and gas phases'
        )
    if not 0 <= slice_index < snapshot.shape[0]:
        raise ValueError(f'slice_index must be in [0, {snapshot.shape[0]})')
    slab = snapshot.read_slab(
        x_start=slice_index,
        x_stop=slice_index + 1,
    )
    temperature_despotic = np.empty(slab.cell_count)
    valid = np.empty(slab.cell_count, dtype=bool)
    for cells in slab.iter_batches(batch_size=query_chunk):
        start = cells.batch_start
        stop = start + cells.cell_count
        batch_temperature_despotic = despotic_reader.read_temperature(cells=cells)
        temperature_despotic[start:stop] = batch_temperature_despotic
        cold_cells = cells.temperature_QUOKKA_K < EMISSION_TEMPERATURE_BOUNDARY_K
        mixed_temperature = mixed_gas_temperature_K(
            cold_cells=cold_cells,
            temperature_despotic_K=batch_temperature_despotic,
            temperature_quokka_K=cells.temperature_QUOKKA_K,
        )
        valid[start:stop] = np.isfinite(mixed_temperature) & (mixed_temperature > 0.0)

    # Every saved 2D array keeps y first and z second; plotting transposes it.
    shape_yz = snapshot.shape[1:]
    left = snapshot.dataset.domain_left_edge.to('kpc').value
    right = snapshot.dataset.domain_right_edge.to('kpc').value
    payload = {
        'nH_slice': (slab.density_g_cm3 * X_H / HYDROGEN_MASS_G).reshape(shape_yz),
        'NH_slice': slab.shielding_NH_cm2.reshape(shape_yz),
        'dVdr_slice': slab.velocity_gradient_s.reshape(shape_yz),
        'T_qk_slice': slab.temperature_QUOKKA_K.reshape(shape_yz),
        'T_dsp_slice': temperature_despotic.reshape(shape_yz),
        'valid': valid.reshape(shape_yz),
        'extent_kpc': np.array([left[1], right[1], left[2], right[2]]),
    }
    return payload


def save_slice_data(payload, snapshot, settings, slice_index):
    """Save raw fields, prepared numerical panels, and a compact report."""
    from quokka2s.physics.gas_fields import HYDROGEN_MASS_G
    from quokka2s.physics.settings import X_H

    np.savez_compressed(settings.output_dir / 'slice_data.npz', **payload)
    report = {
        'status': 'completed',
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'dataset': str(settings.dataset),
        'despotic_table': str(settings.despotic_table),
        'cloudy_table': str(settings.cloudy_table),
        'snapshot_shape': list(snapshot.shape),
        'slice_axis': 'x',
        'slice_index': slice_index,
        'cells': int(payload['valid'].size),
        'valid_cells': int(np.count_nonzero(payload['valid'])),
        'excluded_cells': int(np.count_nonzero(~payload['valid'])),
        'X_H': X_H,
        'hydrogen_mass_g': HYDROGEN_MASS_G,
        'NH_definition': 'Harmonic mean of the inclusive +z and -z hydrogen columns',
        'display_mask': 'Mixed temperature available: DESPOTIC for cold cells; QUOKKA for hot cells',
    }
    (settings.output_dir / 'slice_report.json').write_text(
        json.dumps(report, indent=2) + '\n',
        encoding='utf-8',
    )
    return report


def render_slice_figures(payload, report, output_dir):
    """Render the title-free paper PDF/PNG and the titled PNG from saved arrays."""
    index = int(report['slice_index'])
    stem = f'multi_field_slices_idx{index:04d}'
    for show_title, filename in ((True, stem + '_full.png'), (False, stem + '.png')):
        plot_table_input_slice(
            payload=payload,
            extent_kpc=payload['extent_kpc'],
            output_path=output_dir / filename,
            slice_index=index,
            dataset_name=report['dataset'],
            show_title=show_title,
            save_pdf=not show_title,
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', type=Path, default=ROOT / DEFAULT_PROCESS_CONFIG)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--slice-index', type=int, default=216)
    parser.add_argument('--plot-only', action='store_true', help='Render saved slice_data.npz only')
    parser.add_argument('--no-plot', action='store_true', help='Save arrays without rendering figures')
    args = parser.parse_args()
    if args.plot_only and args.no_plot:
        parser.error('--plot-only and --no-plot cannot be used together')
    output_dir = resolve_path(args.output_dir)
    if args.plot_only:
        report = json.loads((output_dir / 'slice_report.json').read_text())
        with np.load(output_dir / 'slice_data.npz', allow_pickle=False) as data:
            payload = {key: np.array(data[key]) for key in data.files}
        render_slice_figures(payload=payload, report=report, output_dir=output_dir)
        return

    from quokka2s.processing_inputs import load_processing_inputs

    settings = load_process_config(args.config)
    settings = replace(settings, output_dir=output_dir)
    snapshot, emission_calculator = load_processing_inputs(settings)
    payload = calculate_slice_fields(
        snapshot=snapshot,
        despotic_reader=emission_calculator.despotic_reader,
        slice_index=args.slice_index,
        query_chunk=settings.query_chunk,
    )
    payload = prepare_slice_plot_data(payload=payload)
    report = save_slice_data(
        payload=payload,
        snapshot=snapshot,
        settings=settings,
        slice_index=args.slice_index,
    )
    if not args.no_plot:
        render_slice_figures(payload=payload, report=report, output_dir=output_dir)


if __name__ == '__main__':
    main()
