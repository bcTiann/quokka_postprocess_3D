#!/usr/bin/env python3
"""Save gas-mass and line-luminosity phase bins from the current process setup.

Build: python tools/figures/build_emission_phase_histograms.py
       --config configs/emission_process.yaml --output-dir output/emission_phase --no-plot
Replot: python tools/figures/build_emission_phase_histograms.py
        --output-dir output/emission_phase --plot-only
"""
from __future__ import annotations

import argparse
from dataclasses import replace
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

from quokka2s.figures.emission_phase_histograms import plot_panels
from quokka2s.paths import resolve_path
from quokka2s.run_settings import DEFAULT_PROCESS_CONFIG, load_process_config


def accumulate_phase_batch(histograms, cells, emission_calculator):
    """Query one batch once, update all bins, and return cell/mass totals.

    The fourteen saved panels include gas mass and every separate line.
    Each panel selects only the quantities it needs; no shared emission mask
    removes gas or a different line's available luminosity.
    Intrinsic epsilon [erg/s/cm^3] times cell volume [cm^3] supplies each
    line-luminosity weight. Temperature choices come from the emission result.
    emission_calculator is the shared CellEmissionCalculator from input loading.
    """
    from quokka2s.physics.gas_fields import mixed_gas_temperature_K
    from quokka2s.products.emission_phase_histograms import accumulate_emission_phase_histograms

    emission = emission_calculator.calculate(cells=cells)
    accumulate_emission_phase_histograms(
        histograms=histograms,
        cells=cells,
        emission=emission,
    )
    mass = cells.density_g_cm3 * cells.cell_volume_cm3
    despotic_temperature = emission.despotic_temperature_K
    mixed_temperature = mixed_gas_temperature_K(
        cold_cells=emission.cold_cells,
        temperature_despotic_K=despotic_temperature,
        temperature_quokka_K=cells.temperature_QUOKKA_K,
    )
    missing_despotic_temperature = ~np.isfinite(despotic_temperature) | (despotic_temperature <= 0.0)
    missing_mixed_temperature = ~np.isfinite(mixed_temperature) | (mixed_temperature <= 0.0)
    return {
        'processed_cells': cells.cell_count,
        'total_mass_g': float(np.sum(mass, dtype=np.float64)),
        'missing_despotic_temperature_cells': int(np.count_nonzero(missing_despotic_temperature)),
        'missing_mixed_temperature_cells': int(np.count_nonzero(missing_mixed_temperature)),
    }


def process_phase_histograms(snapshot, emission_calculator, settings):
    """Read one slab at a time and retain only small accumulated 0.2-dex bins."""
    from quokka2s.products.emission_phase_histograms import PANELS, DexHistogram
    from quokka2s.snapshot_reader import slab_windows

    if snapshot.processing_shape != snapshot.shape:
        raise ValueError(
            'Emission phase histograms require full x and y ranges; use the standard '
            'process/plot workflow for region line images, spectra, and gas phases'
        )
    histograms = {key: DexHistogram(step=0.2) for key, _, _ in PANELS}
    totals = {
        'processed_cells': 0,
        'total_mass_g': 0.0,
        'missing_despotic_temperature_cells': 0,
        'missing_mixed_temperature_cells': 0,
    }
    selected_slabs = list(slab_windows(
        x_start=0,
        x_stop=snapshot.shape[0],
        slab_nx=settings.slab_nx,
    ))
    if settings.max_slabs is not None:
        selected_slabs = selected_slabs[:settings.max_slabs]
    began = time.monotonic()
    for slab_number, (x_start, x_stop) in enumerate(selected_slabs, start=1):
        slab = snapshot.read_slab(
            x_start=x_start,
            x_stop=x_stop,
        )
        for cells in slab.iter_batches(batch_size=settings.query_chunk):
            batch_totals = accumulate_phase_batch(
                histograms=histograms,
                cells=cells,
                emission_calculator=emission_calculator,
            )
            for key in totals:
                totals[key] += batch_totals[key]
            del cells
        del slab
        elapsed = time.monotonic() - began
        fraction = slab_number / len(selected_slabs)
        remaining = elapsed * (1.0 - fraction) / fraction
        print(
            f'Phase histograms: {100 * fraction:.1f}% '
            f'({slab_number}/{len(selected_slabs)} slabs); '
            f'elapsed {elapsed / 60:.1f} min; ETA ~{remaining / 60:.1f} min',
            flush=True,
        )
    return histograms, totals


def save_phase_histograms(histograms, totals, snapshot, settings):
    """Save numerical bins for every panel and record the processed selection."""
    from quokka2s.products.phase_histogram_preparation import (
        DISPLAY_PANEL_KEYS,
        add_phase_histogram_display_fields,
    )

    panels = {key: histogram.result() for key, histogram in histograms.items()}
    add_phase_histogram_display_fields(panels=panels)
    display_panel_keys = np.asarray(DISPLAY_PANEL_KEYS)
    payload = {
        'panel_keys': np.asarray(tuple(panels)),
        'display_panel_keys': display_panel_keys,
    }
    for key, panel in panels.items():
        for field, value in panel.items():
            payload[f'{key}__{field}'] = value
    np.savez_compressed(settings.output_dir / 'phase_histograms.npz', **payload)
    report = {
        'status': 'completed',
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'dataset': str(settings.dataset),
        'despotic_table': str(settings.despotic_table),
        'cloudy_table': str(settings.cloudy_table),
        'snapshot_shape': list(snapshot.shape),
        'full_snapshot': totals['processed_cells'] == snapshot.cell_count,
        'mass_selection': {
            'QUOKKA_temperature_and_NH': 'all cells',
            'DESPOTIC_temperature': 'available DESPOTIC temperature',
            'mixed_temperature': 'available DESPOTIC temperature for cold cells; QUOKKA for hot cells',
        },
        'line_selection': 'each line requires its own emissivity and thermal temperature',
        'panel_counts': {key: histogram.count for key, histogram in histograms.items()},
        'bin_width_dex': 0.2,
        'line_emission': 'intrinsic',
        **totals,
    }
    (settings.output_dir / 'phase_histograms.json').write_text(
        json.dumps(report, indent=2) + '\n',
        encoding='utf-8',
    )
    return panels, display_panel_keys


def load_saved_phase_histograms(output_dir):
    """Read only the small saved histogram arrays; no snapshot or table lookup."""
    with np.load(output_dir / 'phase_histograms.npz', allow_pickle=False) as data:
        display_panel_keys = np.array(data['display_panel_keys'])
        panels = {}
        for key in data['panel_keys']:
            panels[key] = {}
        for name in data.files:
            if '__' in name:
                key, field = name.split('__', maxsplit=1)
                panels[key][field] = np.array(data[name])
    return panels, display_panel_keys


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', type=Path, default=ROOT / DEFAULT_PROCESS_CONFIG)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-slabs', type=int, help='Limit this diagnostic run to its first N slabs')
    parser.add_argument('--plot-only', action='store_true', help='Render saved phase_histograms.npz only')
    parser.add_argument('--no-plot', action='store_true', help='Save bins without rendering figures')
    args = parser.parse_args()
    if args.plot_only and args.no_plot:
        parser.error('--plot-only and --no-plot cannot be used together')
    if args.max_slabs is not None and args.max_slabs <= 0:
        parser.error('--max-slabs must be positive')
    output_dir = resolve_path(args.output_dir)
    if args.plot_only:
        panels, display_panel_keys = load_saved_phase_histograms(output_dir)
    else:
        from quokka2s.processing_inputs import load_processing_inputs

        settings = load_process_config(args.config)
        settings = replace(
            settings,
            output_dir=output_dir,
            max_slabs=(settings.max_slabs if args.max_slabs is None else args.max_slabs),
        )
        snapshot, emission_calculator = load_processing_inputs(settings)
        histograms, totals = process_phase_histograms(
            snapshot=snapshot,
            emission_calculator=emission_calculator,
            settings=settings,
        )
        panels, display_panel_keys = save_phase_histograms(
            histograms=histograms,
            totals=totals,
            snapshot=snapshot,
            settings=settings,
        )
    if not args.no_plot:
        plot_panels(
            panels=panels,
            display_panel_keys=display_panel_keys,
            png=output_dir / 'phase_histograms_10panel.png',
            pdf=output_dir / 'phase_histograms_10panel.pdf',
        )


if __name__ == '__main__':
    main()
