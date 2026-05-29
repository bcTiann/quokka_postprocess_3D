#!/usr/bin/env python
"""Decisive test: are the 'stripes' a slice-orientation effect?

On an x=const slice, N_x (slice-normal) shows 2D structure while N_y,N_z (in-plane)
look striped. If we instead take a z=const slice (x-y plane), the prediction is
that N_z becomes the structured one and N_x,N_y become striped — proving the
striping is inherent to viewing a directional-column field on a 2D slice, not a bug.

Top row: x=const slice (y-z plane).  Bottom row: z=const slice (x-y plane).
Columns: N_x+, N_y+, N_z+.  L_ext=0 (no floor) to match the panel in question.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

_SRC = Path(__file__).resolve().parents[1] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yt  # noqa
import matplotlib; matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt  # noqa
from matplotlib.colors import Normalize  # noqa

from quokka2s.pipeline.prep import config as cfg                       # noqa
from quokka2s.pipeline.prep import physics_fields as phys              # noqa
from quokka2s.analysis import along_sight_cumulation                   # noqa
from quokka2s.data_handling import YTDataProvider, make_downsampled_dataset  # noqa
from quokka2s.pipeline.cache import compute_cache_key, cache_root_for_dataset  # noqa


def main():
    cfg.COLUMN_EXTENSION_LATERAL_KPC = 0.0   # no floor, isolate the in-box columns
    ds = yt.load(cfg.YT_DATASET_PATH)
    if cfg.DOWNSAMPLE_FACTOR > 1:
        ds = make_downsampled_dataset(ds, cfg.DOWNSAMPLE_FACTOR)
    phys.add_all_fields(ds)
    key = compute_cache_key(dataset_path=cfg.YT_DATASET_PATH, despotic_table_path=cfg.DESPOTIC_TABLE_PATH,
                            downsample_factor=cfg.DOWNSAMPLE_FACTOR, column_extension_lateral_kpc=0.0)
    p = YTDataProvider(ds, cache_root=cache_root_for_dataset(cfg.YT_DATASET_PATH), cache_key=key)

    nH = np.asarray(p.get_slab_z(('gas', 'number_density_H'))[0].in_cgs())
    d = {a: float(p.get_slab_z(('boxlib', f'd{a}'))[0].in_cgs().value.flat[0]) for a in 'xyz'}
    nx, ny, nz = nH.shape

    # the three "+" columns, full 3D
    Nx = along_sight_cumulation(nH * d['x'], axis='x', sign='+')
    Ny = along_sight_cumulation(nH * d['y'], axis='y', sign='+')
    Nz = along_sight_cumulation(nH * d['z'], axis='z', sign='+')

    xi, zi = nx // 2, nz // 2
    # x=const slice (y-z plane): take [xi,:,:] -> (ny,nz)
    xslice = {'N_x+': Nx[xi, :, :], 'N_y+': Ny[xi, :, :], 'N_z+': Nz[xi, :, :]}
    # z=const slice (x-y plane): take [:,:,zi] -> (nx,ny)
    zslice = {'N_x+': Nx[:, :, zi], 'N_y+': Ny[:, :, zi], 'N_z+': Nz[:, :, zi]}

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for j, nm in enumerate(['N_x+', 'N_y+', 'N_z+']):
        a = xslice[nm]
        with np.errstate(divide='ignore'):
            la = np.log10(np.where(a > 0, a, np.nan))
        im = axes[0][j].imshow(la.T, origin='lower', aspect='auto', cmap='cividis',
                               norm=Normalize(np.nanmin(la), np.nanmax(la)))
        fig.colorbar(im, ax=axes[0][j], fraction=0.046, pad=0.02)
        axes[0][j].set_title(f'{nm}  on x=const slice (y-z plane)', fontsize=9)
        axes[0][j].set_xlabel('y idx'); axes[0][j].set_ylabel('z idx')

        b = zslice[nm]
        with np.errstate(divide='ignore'):
            lb = np.log10(np.where(b > 0, b, np.nan))
        im = axes[1][j].imshow(lb.T, origin='lower', aspect='auto', cmap='cividis',
                               norm=Normalize(np.nanmin(lb), np.nanmax(lb)))
        fig.colorbar(im, ax=axes[1][j], fraction=0.046, pad=0.02)
        axes[1][j].set_title(f'{nm}  on z=const slice (x-y plane)', fontsize=9)
        axes[1][j].set_xlabel('x idx'); axes[1][j].set_ylabel('y idx')

    fig.suptitle('Same 3 directional columns on two slice orientations (L_ext=0).\n'
                 'Structured panel = the SLICE-NORMAL direction (top: N_x; bottom: N_z) → '
                 'stripes are a slice-orientation effect, not a bug.', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path(cfg._OUTPUT_ROOT) / 'NH_slice_normal_test.png'
    fig.savefig(str(out), dpi=160, bbox_inches='tight'); plt.close(fig)
    print(f'[out] {out}')

    # numeric: variance along each in-plane axis tells which way it is striped
    def axis_var(arr2d):
        la = np.log10(np.where(arr2d > 0, arr2d, np.nan))
        return np.nanmedian(np.nanstd(la, axis=0)), np.nanmedian(np.nanstd(la, axis=1))
    print('x=const slice (axes: dim0=y, dim1=z):  per-quantity (std over y, std over z) in dex')
    for nm in ['N_x+', 'N_y+', 'N_z+']:
        sy, sz = axis_var(xslice[nm]); print(f'  {nm}: std_along_y={sy:.2f}  std_along_z={sz:.2f}')
    print('z=const slice (axes: dim0=x, dim1=y):  per-quantity (std over x, std over y) in dex')
    for nm in ['N_x+', 'N_y+', 'N_z+']:
        sx, syy = axis_var(zslice[nm]); print(f'  {nm}: std_along_x={sx:.2f}  std_along_y={syy:.2f}')


if __name__ == '__main__':
    main()
