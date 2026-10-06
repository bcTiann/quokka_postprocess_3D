"""Measure all-cell DESPOTIC input extrema directly from the snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from unyt import unyt_array

from quokka2s.physics import gas_fields, settings
from quokka2s.despotic.snapshot_domain import AXIS_NAMES
from quokka2s.file_provenance import file_sha256
from quokka2s.snapshot_reader import Snapshot, slab_windows


ROOT = Path(__file__).resolve().parents[2]


def parse_arguments(argv=None):
    """Read snapshot/output paths and the maximum x width of each slab."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dataset',
        type=Path,
        default=ROOT / 'inputs/snapshots/plt0655228',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='JSON destination (default: output/<dataset>/table_build/snapshot_domain.json)',
    )
    parser.add_argument('--slab-nx', type=int, default=16)
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = ROOT / 'output' / args.dataset.name / 'table_build/snapshot_domain.json'
    if args.output.exists() or args.slab_nx < 1:
        raise ValueError('Output must be new and slab size positive')
    return args


def update_axis_extrema(stats, name, values):
    """Count all supplied cells and update extrema using positive finite values."""
    valid = np.isfinite(values) & (values > 0.0)
    record = stats[name]
    record['count'] += int(values.size)
    record['invalid_count'] += int(values.size - np.count_nonzero(valid))
    if np.any(valid):
        record['minimum'] = min(record['minimum'], float(values[valid].min()))
        record['maximum'] = max(record['maximum'], float(values[valid].max()))


def measure_snapshot_axes(snapshot, slab_nx):
    """Scan all native cells once; retain only one slab and three axis statistics."""
    stats = {}
    for name in AXIS_NAMES:
        stats[name] = dict(
            minimum=float('inf'),
            maximum=-float('inf'),
            invalid_count=0,
            count=0,
        )

    for x_start, x_stop in slab_windows(
        x_start=0,
        x_stop=snapshot.shape[0],
        slab_nx=slab_nx,
    ):
        slab = snapshot.read_slab(x_start=x_start, x_stop=x_stop)
        cells = slab.batch(start=0, stop=slab.cell_count)

        # Restore density units for exactly the scanner's original nH conversion.
        density = unyt_array(cells.density_g_cm3, 'g/cm**3')
        nH = gas_fields.hydrogen_number_density(density).to_value('cm**-3')
        update_axis_extrema(stats, 'nH', nH)
        update_axis_extrema(stats, 'NH', cells.shielding_NH_cm2)
        update_axis_extrema(stats, 'dVdr', cells.velocity_gradient_s)
        del density, nH, cells, slab

        percentage = 100.0 * x_stop / snapshot.shape[0]
        print(f'Scanned {percentage:.1f}% ({x_stop}/{snapshot.shape[0]} x planes)', flush=True)
    return stats


def main(argv=None):
    args = parse_arguments(argv)

    import yt

    dataset = yt.load(str(args.dataset.resolve()))
    if dataset.max_level != 0 or settings.COLUMN_DENSITY_DIRECTIONS != 'z':
        raise ValueError('Scanner requires a full-resolution uniform snapshot and z columns')
    snapshot = Snapshot(dataset=dataset)
    stats = measure_snapshot_axes(snapshot=snapshot, slab_nx=args.slab_nx)

    result = {
        'dataset': str(args.dataset.resolve()),
        'shape': list(snapshot.shape),
        'total_cells': snapshot.cell_count,
        'selection': 'all simulation cells',
        'X_H': float(settings.X_H),
        'column_mean': settings.COLUMN_DENSITY_MEAN,
        'column_directions': settings.COLUMN_DENSITY_DIRECTIONS,
        'source': 'fresh snapshot fields; full z columns; periodic x/y velocity differences',
        # Source hashes identify this diagnostic run; they are not lookup requirements.
        'physics_source_sha256': file_sha256(gas_fields.__file__),
        'snapshot_reader_source_sha256': file_sha256(ROOT / 'src/quokka2s/snapshot_reader.py'),
        'axes': stats,
        'units': {'nH': 'cm^-3', 'NH': 'cm^-2', 'dVdr': 's^-1'},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
