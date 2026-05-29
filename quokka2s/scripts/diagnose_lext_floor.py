#!/usr/bin/env python
"""Decompose a lateral column (N_x+) into the in-box cumulative part vs the
L_ext floor, to show: the in-box part varies cell-to-cell, but the L_ext add-on
is a z-only constant (uses <n_H>(z)), so where it dominates the lateral column
loses its horizontal variation.
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
from yt.units import kpc  # noqa

from quokka2s.pipeline.prep import config as cfg                       # noqa
from quokka2s.pipeline.prep import physics_fields as phys              # noqa
from quokka2s.analysis import along_sight_cumulation                   # noqa
from quokka2s.data_handling import YTDataProvider, make_downsampled_dataset  # noqa
from quokka2s.pipeline.cache import compute_cache_key, cache_root_for_dataset  # noqa


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
    dx = float(p.get_slab_z(('boxlib', 'dx'))[0].in_cgs().value.flat[0])
    ext = [float(v.in_units('kpc').value) for v in p.get_slab_z(('gas', 'density'))[1]['x']]
    xi = nH.shape[0] // 2
    L_ext = float(cfg.COLUMN_EXTENSION_LATERAL_KPC)
    L_cm = float((L_ext * kpc).in_units('cm').value)

    inbox = along_sight_cumulation(nH * dx, axis='x', sign='+')[xi, :, :]   # (ny,nz) in-box only
    floor_z = L_cm * nH.mean(axis=(0, 1))                                   # (nz,) z-only
    floor = np.broadcast_to(floor_z[None, :], inbox.shape)                  # (ny,nz)
    total = inbox + floor

    print(f'x-slice {xi}, L_ext={L_ext} kpc  (lateral column N_x+ decomposition)')
    print(f'  in-box  N_x+: median log10={np.nanmedian(np.log10(inbox)):.2f}  '
          f'[{np.log10(inbox.min()):.2f},{np.log10(inbox.max()):.2f}]')
    print(f'  L_ext floor : median log10={np.nanmedian(np.log10(floor)):.2f}  '
          f'[{np.log10(floor.min()):.2f},{np.log10(floor.max()):.2f}]   '
          f'(constant in y at each z)')
    print(f'  floor / inbox: median={np.nanmedian(floor/inbox):.1f}x   '
          f'frac(floor>inbox)={np.mean(floor>inbox):.2f}  '
          f'-> where >1, the lateral column is floor-dominated (z-bands, no horizontal variation)')
    # quantify horizontal variation washed out: at each z, std over y of total vs inbox
    cv_inbox = np.nanmedian(np.nanstd(np.log10(inbox), axis=0))
    cv_total = np.nanmedian(np.nanstd(np.log10(total), axis=0))
    print(f'  horizontal (per-z) scatter of log10 column:  in-box={cv_inbox:.3f} dex  -> total={cv_total:.3f} dex '
          f'(L_ext flattens it)')

    fig, axes = plt.subplots(1, 4, figsize=(2.7 * 4, 11), sharey=True, gridspec_kw={'wspace': 0.1})
    lo = np.floor(min(np.nanpercentile(np.log10(inbox), 1), np.nanpercentile(np.log10(total), 1)))
    hi = np.ceil(max(np.nanpercentile(np.log10(inbox), 99), np.nanpercentile(np.log10(total), 99)))
    panels = [
        (np.log10(inbox), r'in-box $N_{x+}$ (no $L_{\rm ext}$)' '\n(varies cell-to-cell)', 'cividis', lo, hi),
        (np.log10(floor), r'$L_{\rm ext}\langle n_H\rangle(z)$ floor' '\n(z-bands, const in y)', 'cividis', lo, hi),
        (np.log10(total), r'total $N_{x+}$' '\n(floor-dominated → bands)', 'cividis', lo, hi),
        (np.log10(floor / inbox), r'$\log_{10}$(floor / in-box)' '\n(>0 → floor wins)', 'RdBu_r', -2, 2),
    ]
    for ax, (data, title, cmap, vlo, vhi) in zip(axes, panels):
        im = ax.imshow(data.T, origin='lower', extent=ext, aspect='auto', cmap=cmap, norm=Normalize(vlo, vhi))
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        ax.set_title(title, fontsize=8.5); ax.set_xlabel('y [kpc]', fontsize=8); ax.tick_params(labelsize=7)
    axes[0].set_ylabel('z [kpc]', fontsize=10)
    fig.suptitle(f'Lateral column $N_{{x+}}$ = in-box cumulative + $L_{{\\rm ext}}$ floor   '
                 f'(x-slice {xi}, L_ext={L_ext:g} kpc)', fontsize=11, y=0.93)
    out = Path(cfg.OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    png = out / 'NH_lext_floor_decomposition.png'
    fig.savefig(str(png), dpi=160, bbox_inches='tight'); plt.close(fig)
    print(f'[out] {png}')


if __name__ == '__main__':
    main()
