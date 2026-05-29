#!/usr/bin/env python
"""For L_ext = 0 and 9 kpc, midplane-x slices of the 6 directional columns
(N_x+, N_x-, N_y+, N_y-, N_z+, N_z-) plus the harmonic and arithmetic means.
2 rows (L_ext) x 8 cols = 16 panels. Full-data-range colourbars, shared per
column across the two rows.
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
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa
from yt.units import kpc  # noqa

from quokka2s.pipeline.prep import config as cfg                       # noqa
from quokka2s.pipeline.prep import physics_fields as phys              # noqa
from quokka2s.analysis import along_sight_cumulation                   # noqa
from quokka2s.data_handling import YTDataProvider, make_downsampled_dataset  # noqa
from quokka2s.pipeline.cache import compute_cache_key, cache_root_for_dataset  # noqa

L_EXTS = [0.0, 9.0]
DIRS = [("x", "+", True), ("x", "-", True), ("y", "+", True),
        ("y", "-", True), ("z", "+", False), ("z", "-", False)]
COL_LABELS = [r'$N_{x+}$', r'$N_{x-}$', r'$N_{y+}$', r'$N_{y-}$',
              r'$N_{z+}$', r'$N_{z-}$', r'$N_{\rm harm}$', r'$N_{\rm arith}$',
              r'$N_{\min}$', r'$N_{\max}$']
NCOL = len(COL_LABELS)


def main():
    ds = yt.load(cfg.YT_DATASET_PATH)
    if cfg.DOWNSAMPLE_FACTOR > 1:
        ds = make_downsampled_dataset(ds, cfg.DOWNSAMPLE_FACTOR)
    phys.add_all_fields(ds)
    key = compute_cache_key(dataset_path=cfg.YT_DATASET_PATH, despotic_table_path=cfg.DESPOTIC_TABLE_PATH,
                            downsample_factor=cfg.DOWNSAMPLE_FACTOR,
                            column_extension_lateral_kpc=cfg.COLUMN_EXTENSION_LATERAL_KPC)
    p = YTDataProvider(ds, cache_root=cache_root_for_dataset(cfg.YT_DATASET_PATH), cache_key=key)

    nH = np.asarray(p.get_slab_z(('gas', 'number_density_H'))[0].in_cgs())
    d = {a: float(p.get_slab_z(('boxlib', f'd{a}'))[0].in_cgs().value.flat[0]) for a in 'xyz'}
    ext = [float(v.in_units('kpc').value) for v in p.get_slab_z(('gas', 'density'))[1]['x']]
    xi = nH.shape[0] // 2
    n_bar_z = nH.mean(axis=(0, 1))                       # <n_H>(z)

    # in-box directional columns (L_ext-independent), sliced at x=center
    inbox = []
    for a, s, _lat in DIRS:
        inbox.append(along_sight_cumulation(nH * d[a], axis=a, sign=s)[xi, :, :].copy())

    # build the 8 panels per L_ext
    panels = {}   # L -> list of 8 (ny,nz) arrays
    for L in L_EXTS:
        floor = float((L * kpc).in_units('cm').value) * n_bar_z[None, :] if L > 0 else 0.0
        Nk = []
        for (a, s, lat), ib in zip(DIRS, inbox):
            Nk.append(ib + floor if (lat and L > 0) else ib.copy())
        stk = np.stack(Nk)
        Nharm = 6.0 / np.sum(1.0 / stk, axis=0)
        Narith = np.mean(stk, axis=0)
        Nmin = np.min(stk, axis=0)      # thinnest / least-shielded direction
        Nmax = np.max(stk, axis=0)      # thickest / most-shielded direction
        panels[L] = Nk + [Nharm, Narith, Nmin, Nmax]

    # shared full range per column (over both L_ext)
    vlim = []
    for j in range(NCOL):
        allv = np.concatenate([np.log10(panels[L][j][panels[L][j] > 0]) for L in L_EXTS])
        vlim.append((float(allv.min()), float(allv.max())))

    nrow, ncol = len(L_EXTS), NCOL
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.25 * ncol, 10.5 * nrow), sharey=True, squeeze=False,
                             gridspec_kw={'wspace': 0.08, 'hspace': 0.04})
    for ri, L in enumerate(L_EXTS):
        for j in range(NCOL):
            ax = axes[ri][j]
            with np.errstate(divide='ignore'):
                logd = np.log10(np.where(panels[L][j] > 0, panels[L][j], np.nan))
            im = ax.imshow(logd.T, origin='lower', extent=ext, aspect='auto', cmap='cividis',
                           norm=Normalize(*vlim[j]))
            ax.tick_params(labelsize=7)
            if ri == 0:
                cax = make_axes_locatable(ax).append_axes('top', size='2%', pad=0.45)
                cb = fig.colorbar(im, cax=cax, orientation='horizontal')
                cb.ax.tick_params(labelsize=7, top=True, bottom=False, labeltop=True, labelbottom=False)
                cax.set_title(COL_LABELS[j] + r' $[\log_{10}\,{\rm cm}^{-2}]$', fontsize=9, pad=3)
            if ri == nrow - 1:
                ax.set_xlabel('y [kpc]', fontsize=8)
            if j == 0:
                ax.set_ylabel(f'$L_{{\\rm ext}}={L:g}$ kpc\nz [kpc]', fontsize=10)
    fig.suptitle('Directional columns N_k + harmonic/arithmetic means — midplane-x slice  '
                 '(rows = L_ext; full-range colourbars shared per column)', fontsize=12, y=0.995)

    out = Path(cfg._OUTPUT_ROOT)
    png = out / 'NH_directions_L0_L9_slices.png'
    fig.savefig(str(png), dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'[out] {png}')
    for j in range(NCOL):
        print(f'  {COL_LABELS[j]:14s} shared range log10 = [{vlim[j][0]:.2f}, {vlim[j][1]:.2f}]')


if __name__ == '__main__':
    main()
