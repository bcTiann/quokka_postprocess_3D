#!/usr/bin/env python
"""Decompose column_density_H on the midplane-x slice into its 6 directional
columns N_k, the harmonic mean, and the per-cell dominant (minimum) direction.

This verifies WHY the N_H slice looks the way it does (which direction drives
the harmonic mean, where the L_ext lateral floor sits, etc.).
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
from matplotlib.colors import Normalize, BoundaryNorm  # noqa
from yt.units import kpc  # noqa

from quokka2s.pipeline.prep import config as cfg                       # noqa
from quokka2s.pipeline.prep import physics_fields as phys              # noqa
from quokka2s.analysis import along_sight_cumulation                   # noqa
from quokka2s.data_handling import YTDataProvider, make_downsampled_dataset  # noqa
from quokka2s.pipeline.cache import compute_cache_key, cache_root_for_dataset  # noqa

m_H = 1.6726219e-24
XSLICE = None   # None → center x


def main():
    ds = yt.load(cfg.YT_DATASET_PATH)
    if cfg.DOWNSAMPLE_FACTOR > 1:
        ds = make_downsampled_dataset(ds, cfg.DOWNSAMPLE_FACTOR)
    phys.add_all_fields(ds)
    key = compute_cache_key(dataset_path=cfg.YT_DATASET_PATH, despotic_table_path=cfg.DESPOTIC_TABLE_PATH,
                            downsample_factor=cfg.DOWNSAMPLE_FACTOR,
                            column_extension_lateral_kpc=cfg.COLUMN_EXTENSION_LATERAL_KPC)
    p = YTDataProvider(ds, cache_root=cache_root_for_dataset(cfg.YT_DATASET_PATH), cache_key=key)

    nH = np.asarray(p.get_slab_z(('gas', 'number_density_H'))[0].in_cgs())     # (nx,ny,nz)
    dx = float(p.get_slab_z(('boxlib', 'dx'))[0].in_cgs().value.flat[0])
    dy = float(p.get_slab_z(('boxlib', 'dy'))[0].in_cgs().value.flat[0])
    dz = float(p.get_slab_z(('boxlib', 'dz'))[0].in_cgs().value.flat[0])
    nx = nH.shape[0]
    xi = nx // 2 if XSLICE is None else XSLICE

    # lateral L_ext floor  N_ext(z) = L_ext * <n_H>(z)
    L_ext = float(cfg.COLUMN_EXTENSION_LATERAL_KPC)
    if L_ext > 0:
        L_cm = float((L_ext * kpc).in_units('cm').value)
        n_bar_z = nH.mean(axis=(0, 1))                  # (nz,)
        N_ext = (L_cm * n_bar_z)[None, None, :]
    else:
        N_ext = 0.0

    dirs = [("x", "+", dx, True), ("x", "-", dx, True),
            ("y", "+", dy, True), ("y", "-", dy, True),
            ("z", "+", dz, False), ("z", "-", dz, False)]
    names = [f'$N_{{{a}{s}}}$' + ('  (+$L_{ext}$)' if lat else '') for a, s, _, lat in dirs]

    Nk_slices = []
    inv_sum = None
    for a, s, d, lat in dirs:
        N = along_sight_cumulation(nH * d, axis=a, sign=s)
        if lat and L_ext > 0:
            N = N + N_ext
        inv_sum = 1.0 / N if inv_sum is None else inv_sum + 1.0 / N
        Nk_slices.append(N[xi, :, :].copy())            # (ny,nz)
        del N
    Neff = (6.0 / inv_sum)[xi, :, :]                    # harmonic mean slice
    rho_sl = nH[xi, :, :] * m_H / cfg.X_H               # density slice for reference

    # which direction is the MINIMUM (dominates the harmonic mean) per cell
    stack = np.stack(Nk_slices, axis=0)                 # (6,ny,nz)
    argmin = np.argmin(stack, axis=0)                   # 0..5

    # report
    print(f'x-slice index {xi}/{nx};  L_ext={L_ext} kpc')
    print(f'Neff slice: median log10 = {np.nanmedian(np.log10(Neff)):.2f}  '
          f'[{np.log10(np.nanmin(Neff)):.2f}, {np.log10(np.nanmax(Neff)):.2f}]')
    frac = [np.mean(argmin == k) for k in range(6)]
    print('dominant (minimum) direction share on this slice:')
    for k, (a, s, _, lat) in enumerate(dirs):
        print(f'  N_{a}{s}: {100*frac[k]:5.1f}%')
    # cross-check vs the pipeline field
    field_sl = np.asarray(p.get_slab_z(('gas', 'column_density_H'))[0].in_cgs())[xi, :, :]
    print(f'max rel diff vs pipeline column_density_H: '
          f'{np.nanmax(np.abs(Neff-field_sl)/field_sl):.2e}')

    # plot
    ext = None
    extd = p.get_slab_z(('gas', 'density'))[1]['x']
    ext = [float(v.in_units('kpc').value) for v in extd]
    fig, axes = plt.subplots(1, 9, figsize=(2.3 * 9, 11), sharey=True,
                             gridspec_kw={'wspace': 0.08})
    lo = np.floor(min(np.nanpercentile(np.log10(s[s > 0]), 1) for s in Nk_slices + [Neff]))
    hi = np.ceil(max(np.nanpercentile(np.log10(s[s > 0]), 99) for s in Nk_slices + [Neff]))
    panels = [(np.log10(rho_sl), r'$\log_{10}\rho$', 'inferno', None, None)]
    for sl, nm in zip(Nk_slices, names):
        panels.append((np.log10(sl), nm, 'cividis', lo, hi))
    panels.append((np.log10(Neff), r'$N_{\rm eff}=6/\Sigma(1/N_k)$', 'cividis', lo, hi))
    # argmin panel
    panels.append((argmin.astype(float), 'dominant (min) dir', 'tab10', -0.5, 5.5))

    for ax, (data, title, cmap, vlo, vhi) in zip(axes, panels):
        if 'dominant' in title:
            im = ax.imshow(data.T, origin='lower', extent=ext, aspect='auto', cmap='tab10',
                           norm=BoundaryNorm(np.arange(-0.5, 6.5), 256, extend='neither'))
            cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, ticks=range(6))
            cb.ax.set_yticklabels([f'{a}{s}' for a, s, _, _ in dirs], fontsize=7)
        else:
            norm = Normalize(vlo, vhi) if vlo is not None else None
            im = ax.imshow(data.T, origin='lower', extent=ext, aspect='auto', cmap=cmap, norm=norm)
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('y [kpc]', fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel('z [kpc]', fontsize=10)
    fig.suptitle(f'column_density_H decomposition — x-slice {xi}, L_ext={L_ext:g} kpc '
                 f'(panels: rho | 6 directional N_k | harmonic mean | which dir is min)',
                 fontsize=11, y=0.93)

    out = Path(cfg.OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    png = out / 'NH_slice_decomposition.png'
    fig.savefig(str(png), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'[out] {png}')


if __name__ == '__main__':
    main()
