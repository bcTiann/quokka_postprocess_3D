#!/usr/bin/env python
"""Compare HARMONIC vs ARITHMETIC mean of the 6 directional columns N_k, on the
midplane-x slice. Harmonic is min-dominated (current pipeline); arithmetic is
max-dominated (so with L_ext the lateral columns drive it).
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


def main():
    _lext = os.environ.get('PHASE_LEXT')
    if _lext is not None:
        cfg.COLUMN_EXTENSION_LATERAL_KPC = float(_lext)
        print(f'[override] L_ext = {cfg.COLUMN_EXTENSION_LATERAL_KPC} kpc')

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
    dy = float(p.get_slab_z(('boxlib', 'dy'))[0].in_cgs().value.flat[0])
    dz = float(p.get_slab_z(('boxlib', 'dz'))[0].in_cgs().value.flat[0])
    ext = [float(v.in_units('kpc').value) for v in p.get_slab_z(('gas', 'density'))[1]['x']]
    xi = nH.shape[0] // 2

    L_ext = float(cfg.COLUMN_EXTENSION_LATERAL_KPC)
    N_ext = (float((L_ext * kpc).in_units('cm').value) * nH.mean(axis=(0, 1)))[None, None, :] if L_ext > 0 else 0.0

    dirs = [("x", "+", dx, True), ("x", "-", dx, True), ("y", "+", dy, True),
            ("y", "-", dy, True), ("z", "+", dz, False), ("z", "-", dz, False)]
    Nk = []
    for a, s, d, lat in dirs:
        N = along_sight_cumulation(nH * d, axis=a, sign=s)
        if lat and L_ext > 0:
            N = N + N_ext
        Nk.append(N[xi, :, :].copy()); del N
    stack = np.stack(Nk, axis=0)                  # (6,ny,nz)

    N_harm  = 6.0 / np.sum(1.0 / stack, axis=0)
    N_arith = np.mean(stack, axis=0)
    argmin  = np.argmin(stack, axis=0)            # harmonic-dominant dir
    argmax  = np.argmax(stack, axis=0)            # arithmetic-dominant dir
    rho_sl  = nH[xi, :, :] * m_H / cfg.X_H

    lab = [f'{a}{s}' for a, s, _, _ in dirs]
    print(f'x-slice {xi}, L_ext={L_ext} kpc')
    print(f'  HARMONIC : median log10 = {np.nanmedian(np.log10(N_harm)):.2f}  '
          f'[{np.log10(N_harm.min()):.2f}, {np.log10(N_harm.max()):.2f}]')
    print(f'  ARITHMETIC: median log10 = {np.nanmedian(np.log10(N_arith)):.2f}  '
          f'[{np.log10(N_arith.min()):.2f}, {np.log10(N_arith.max()):.2f}]')
    print(f'  median log10(arith/harm) = {np.nanmedian(np.log10(N_arith/N_harm)):.2f} dex')
    print('  harmonic dominant (min) dir shares:', {lab[k]: round(100*np.mean(argmin == k), 1) for k in range(6)})
    print('  arithmetic dominant (max) dir shares:', {lab[k]: round(100*np.mean(argmax == k), 1) for k in range(6)})

    fig, axes = plt.subplots(1, 5, figsize=(2.6 * 5, 11), sharey=True, gridspec_kw={'wspace': 0.1})
    lo = np.floor(min(np.nanpercentile(np.log10(N_harm), 1), np.nanpercentile(np.log10(N_arith), 1)))
    hi = np.ceil(max(np.nanpercentile(np.log10(N_harm), 99), np.nanpercentile(np.log10(N_arith), 99)))
    plots = [
        (np.log10(rho_sl), r'$\log_{10}\rho$', 'inferno', None, None, None),
        (np.log10(N_harm), r'$N_{\rm harm}=6/\Sigma(1/N_k)$ (current)', 'cividis', lo, hi, None),
        (np.log10(N_arith), r'$N_{\rm arith}=\frac{1}{6}\Sigma N_k$', 'cividis', lo, hi, None),
        (np.log10(N_arith / N_harm), r'$\log_{10}(N_{\rm arith}/N_{\rm harm})$', 'viridis', None, None, None),
        (argmax.astype(float), 'arithmetic dominant (max) dir', 'tab10', -0.5, 5.5, lab),
    ]
    for ax, (data, title, cmap, vlo, vhi, ticklab) in zip(axes, plots):
        if ticklab is not None:
            im = ax.imshow(data.T, origin='lower', extent=ext, aspect='auto', cmap='tab10',
                           norm=BoundaryNorm(np.arange(-0.5, 6.5), 256))
            cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, ticks=range(6))
            cb.ax.set_yticklabels(ticklab, fontsize=7)
        else:
            norm = Normalize(vlo, vhi) if vlo is not None else None
            im = ax.imshow(data.T, origin='lower', extent=ext, aspect='auto', cmap=cmap, norm=norm)
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        ax.set_title(title, fontsize=8.5); ax.set_xlabel('y [kpc]', fontsize=8); ax.tick_params(labelsize=7)
    axes[0].set_ylabel('z [kpc]', fontsize=10)
    fig.suptitle(f'N_H: harmonic vs arithmetic mean of the 6 directional columns  '
                 f'(x-slice {xi}, L_ext={L_ext:g} kpc)', fontsize=11, y=0.93)

    lext_tag = f"_Lext{cfg.COLUMN_EXTENSION_LATERAL_KPC:g}kpc"
    out = Path(f"{cfg._OUTPUT_ROOT}/{cfg._DATASET_BASENAME}_down{cfg.DOWNSAMPLE_FACTOR}{lext_tag}/")
    out.mkdir(parents=True, exist_ok=True)
    png = out / 'NH_harmonic_vs_arithmetic.png'
    fig.savefig(str(png), dpi=160, bbox_inches='tight')
    plt.close(fig); print(f'[out] {png}')


if __name__ == '__main__':
    main()
