#!/usr/bin/env python
"""1-D traces of the 6 directional columns N_k and the harmonic mean ALONG z,
at a few fixed y (x-slice = center). Reveals whether the vertical 'stripes' are
genuinely flat along z and which direction drives them.
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
from yt.units import kpc  # noqa

from quokka2s.pipeline.prep import config as cfg                       # noqa
from quokka2s.pipeline.prep import physics_fields as phys              # noqa
from quokka2s.analysis import along_sight_cumulation                   # noqa
from quokka2s.data_handling import YTDataProvider, make_downsampled_dataset  # noqa
from quokka2s.pipeline.cache import compute_cache_key, cache_root_for_dataset  # noqa


def main():
    _lext = os.environ.get('PHASE_LEXT')
    if _lext is not None:
        cfg.COLUMN_EXTENSION_LATERAL_KPC = float(_lext)
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
    zext = [float(v.in_units('kpc').value) for v in p.get_slab_z(('gas', 'density'))[1]['x']]
    z = np.linspace(zext[2], zext[3], nH.shape[2])
    xi = nH.shape[0] // 2
    L_ext = float(cfg.COLUMN_EXTENSION_LATERAL_KPC)
    N_ext = (float((L_ext * kpc).in_units('cm').value) * nH.mean(axis=(0, 1)))[None, None, :] if L_ext > 0 else 0.0

    dirs = [("x", "+", dx, True), ("x", "-", dx, True), ("y", "+", dy, True),
            ("y", "-", dy, True), ("z", "+", dz, False), ("z", "-", dz, False)]
    lab = [f'N_{a}{s}' for a, s, _, _ in dirs]
    Nk = []
    for a, s, d, lat in dirs:
        N = along_sight_cumulation(nH * d, axis=a, sign=s)
        if lat and L_ext > 0:
            N = N + N_ext
        Nk.append(N[xi, :, :].copy()); del N
    Nh = 6.0 / np.sum(1.0 / np.stack(Nk), axis=0)        # (ny,nz)
    nH_sl = nH[xi, :, :]

    # pick y: brightest stripe (max column-summed N_h), a median, a faint
    colsum = np.nansum(Nh, axis=1)
    ys = {'brightest y': int(np.argmax(colsum)),
          'median y':    int(np.argsort(colsum)[len(colsum)//2]),
          'faint y':     int(np.argsort(colsum)[len(colsum)//10])}

    fig, axes = plt.subplots(1, len(ys), figsize=(6.2 * len(ys), 5.5), sharey=True)
    for ax, (name, j) in zip(axes, ys.items()):
        for k in range(6):
            ax.plot(z, Nk[k][j, :], lw=1.0, alpha=0.8, label=lab[k])
        ax.plot(z, Nh[j, :], 'k-', lw=2.2, label='N_harmonic')
        ax2 = ax.twinx()
        ax2.plot(z, nH_sl[j, :], color='0.6', lw=0.8, ls=':', label='n_H (local)')
        ax2.set_yscale('log'); ax2.set_ylabel('n_H [cm$^{-3}$]', color='0.5', fontsize=8)
        ax.set_yscale('log')
        ax.set_xlabel('z [kpc]', fontsize=10)
        ax.set_title(f'{name} (idx {j})', fontsize=10)
        # report flatness of N_harmonic along z
        v = np.log10(Nh[j, :]); v = v[np.isfinite(v)]
        ax.text(0.02, 0.02, f'N_harm log10 spread along z: {v.max()-v.min():.2f} dex',
                transform=ax.transAxes, fontsize=8, va='bottom',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    axes[0].set_ylabel('column density N_k [cm$^{-2}$]', fontsize=10)
    axes[0].legend(fontsize=7, loc='upper center', ncol=2)
    fig.suptitle(f'Directional columns vs z at fixed y (x-slice {xi}, L_ext={L_ext:g} kpc)', fontsize=12)
    out = Path(f"{cfg._OUTPUT_ROOT}/{cfg._DATASET_BASENAME}_down{cfg.DOWNSAMPLE_FACTOR}_Lext{L_ext:g}kpc/")
    out.mkdir(parents=True, exist_ok=True)
    png = out / 'NH_column_traces.png'
    fig.savefig(str(png), dpi=160, bbox_inches='tight'); plt.close(fig)

    # numeric report
    print(f'x-slice {xi}, L_ext={L_ext} kpc')
    for name, j in ys.items():
        dom = np.argmin(np.stack([Nk[k][j, :] for k in range(6)]), axis=0)
        share = {lab[k]: round(100*np.mean(dom == k), 0) for k in range(6) if np.any(dom == k)}
        v = np.log10(Nh[j, :])
        print(f'  {name} (idx {j}): N_harm along z spread = {np.nanmax(v)-np.nanmin(v):.2f} dex; '
              f'dominant dir along z = {share}')
    print(f'[out] {png}')


if __name__ == '__main__':
    main()
