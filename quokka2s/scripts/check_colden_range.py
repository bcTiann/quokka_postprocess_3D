"""One-off: recompute column_density_H with a lateral extension and check
whether the new distribution still fits inside the DESPOTIC table's
col_density_values range [1e15, 1e24] cm^-2.

Run with:  conda run -n yt-env python -m quokka2s.scripts.check_colden_range
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yt
from yt.units import kpc, mh

PROJECT_SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(PROJECT_SRC))

from quokka2s.analysis import along_sight_cumulation                 # noqa: E402
from quokka2s.data_handling import YTDataProvider, make_downsampled_dataset  # noqa: E402
from quokka2s.pipeline.prep import config as cfg                     # noqa: E402
from quokka2s.tables import load_table                               # noqa: E402


L_EXT_KPC = 9.0


def main() -> None:
    print(f"dataset  = {cfg.YT_DATASET_PATH}")
    print(f"down     = {cfg.DOWNSAMPLE_FACTOR}")
    print(f"L_ext    = {L_EXT_KPC} kpc / side (lateral only)")
    print()

    ds = yt.load(cfg.YT_DATASET_PATH)
    if cfg.DOWNSAMPLE_FACTOR > 1:
        ds = make_downsampled_dataset(ds, cfg.DOWNSAMPLE_FACTOR)
    provider = YTDataProvider(ds)

    density_3d, _ = provider.get_slab_z(('gas', 'density'))
    dx_3d, _ = provider.get_slab_z(('boxlib', 'dx'))
    dy_3d, _ = provider.get_slab_z(('boxlib', 'dy'))
    dz_3d, _ = provider.get_slab_z(('boxlib', 'dz'))

    n_H_3d = ((density_3d.in_cgs() * cfg.X_H) / mh.in_cgs()).to('cm**-3').value
    dx_v = dx_3d.in_cgs().value
    dy_v = dy_3d.in_cgs().value
    dz_v = dz_3d.in_cgs().value
    print(f"grid shape = {n_H_3d.shape}    (x, y, z)")

    n_bar_z = n_H_3d.mean(axis=(0, 1))   # (Nz,)
    print(f"n_bar(z): min={n_bar_z.min():.3e}  median={np.median(n_bar_z):.3e}  "
          f"max={n_bar_z.max():.3e}  cm^-3")

    L_ext_cm = float((L_EXT_KPC * kpc).in_cgs().value)
    N_ext_lat_z = L_ext_cm * n_bar_z     # (Nz,)
    print(f"N_ext_lat(z): min={N_ext_lat_z.min():.3e}  median={np.median(N_ext_lat_z):.3e}  "
          f"max={N_ext_lat_z.max():.3e}  cm^-2")
    print()

    inv_old = None
    inv_new = None
    for axis, sign, dxyz, lateral in (
        ("x", "+", dx_v, True),
        ("x", "-", dx_v, True),
        ("y", "+", dy_v, True),
        ("y", "-", dy_v, True),
        ("z", "+", dz_v, False),
        ("z", "-", dz_v, False),
    ):
        N = along_sight_cumulation(n_H_3d * dxyz, axis=axis, sign=sign)
        inc_old = 1.0 / N
        inv_old = inc_old if inv_old is None else inv_old + inc_old
        if lateral:
            N = N + N_ext_lat_z[None, None, :]
        inc_new = 1.0 / N
        inv_new = inc_new if inv_new is None else inv_new + inc_new
        del N, inc_old, inc_new

    colden_old = 6.0 / inv_old
    colden_new = 6.0 / inv_new

    table = load_table(cfg.DESPOTIC_TABLE_PATH)
    col_min = float(table.col_density_values.min())
    col_max = float(table.col_density_values.max())
    print(f"DESPOTIC table colDen range: [{col_min:.3e}, {col_max:.3e}] cm^-2")
    print()

    pcts = (0.0, 0.01, 0.1, 1, 5, 50, 95, 99, 99.9, 99.99, 100.0)
    for label, arr in (('OLD (no extension)', colden_old),
                       ('NEW (9 kpc lateral) ', colden_new)):
        finite = np.isfinite(arr) & (arr > 0)
        a = arr[finite]
        print(f"--- {label} ---")
        for p in pcts:
            v = a.min() if p == 0.0 else (a.max() if p == 100.0 else np.percentile(a, p))
            print(f"  p{p:>6.2f}  = {v:.3e} cm^-2")
        n_below = int((arr < col_min).sum())
        n_above = int((arr > col_max).sum())
        ntot = arr.size
        print(f"  below table min ({col_min:.0e}):  {n_below}/{ntot}  "
              f"= {100*n_below/ntot:.4f}%")
        print(f"  above table max ({col_max:.0e}):  {n_above}/{ntot}  "
              f"= {100*n_above/ntot:.4f}%")
        print()


if __name__ == '__main__':
    main()
