"""Measure all-cell DESPOTIC input extrema directly from the snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yt

from quokka2s.pipeline.prep import config as cfg
from quokka2s.pipeline.prep import physics_fields as physics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--slab-nx', type=int, default=16)
    args = parser.parse_args()
    if args.output.exists() or args.slab_nx < 1:
        raise ValueError('Output must be new and slab size positive')
    ds = yt.load(str(args.dataset.resolve()))
    if (ds.max_level != 0 or cfg.DOWNSAMPLE_FACTOR != 1
            or cfg.COLUMN_DENSITY_DIRECTIONS != 'z'):
        raise ValueError('Scanner requires a full-resolution uniform snapshot and z columns')
    shape = tuple(int(x) for x in ds.domain_dimensions)
    width = ds.domain_width / ds.domain_dimensions
    stats = {name: dict(minimum=float('inf'), maximum=-float('inf'),
                        invalid_count=0, count=0) for name in ('nH', 'NH', 'dVdr')}
    for ix in range(0, shape[0], args.slab_nx):
        end = min(ix+args.slab_nx, shape[0])
        # One x ghost cell on each interior side reproduces the full-cube
        # central differences. y/z are complete, including full z columns.
        lo, hi = max(0, ix-1), min(shape[0], end+1)
        edge = ds.domain_left_edge.copy()
        edge[0] += lo * width[0]
        grid = ds.covering_grid(level=0, left_edge=edge, dims=(hi-lo, *shape[1:]))
        core = slice(ix-lo, end-lo)
        for name, function, unit in (
            ('nH', physics._number_density_H, 'cm**-3'),
            ('NH', physics._column_density_H, 'cm**-2'),
            ('dVdr', physics._dVdr_lvg, 's**-1'),
        ):
            data = np.asarray(function(None, grid).to(unit), dtype=float)[core]
            valid = np.isfinite(data) & (data > 0.0)
            s = stats[name]
            s['count'] += int(data.size)
            s['invalid_count'] += int(data.size - np.count_nonzero(valid))
            if np.any(valid):
                s['minimum'] = min(s['minimum'], float(data[valid].min()))
                s['maximum'] = max(s['maximum'], float(data[valid].max()))
            del data, valid
        del grid
        print(f'Scanned {end}/{shape[0]} x planes', flush=True)
    result = dict(dataset=str(args.dataset.resolve()), shape=list(shape),
                  total_cells=int(np.prod(shape)), selection='all simulation cells',
                  X_H=float(cfg.X_H), column_mean=cfg.COLUMN_DENSITY_MEAN,
                  column_directions=cfg.COLUMN_DENSITY_DIRECTIONS,
                  source='fresh snapshot fields; full z columns; x halo for velocity gradient',
                  physics_source_sha256=hashlib.sha256(Path(physics.__file__).read_bytes()).hexdigest(),
                  axes=stats, units={'nH':'cm^-3', 'NH':'cm^-2', 'dVdr':'s^-1'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
