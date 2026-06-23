#!/usr/bin/env python
"""Conservative bad-cell fill: 3D log-linear interp ONLY inside the convex
hull of valid table cells.  No nearest-neighbour extrapolation — cells
outside the convex hull keep NaN.

Bad = (failure_mask | Tg is NaN | Tg > 1e6 K).  These are zeroed out across
every per-cell array (tg_final, mu, cv, Eint, all species abundance / line
fields, all energy:: fields), then the resulting NaN cells are filled via
``scipy.interpolate.griddata(..., method='linear')`` which returns NaN for
points outside the convex hull of the valid data — i.e. it interpolates
but never extrapolates.

Usage::

    python fill_table_convex_hull_only.py <src_npz> <dst_npz>
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata


def fill_in_hull(arr: np.ndarray, log_axes: tuple[np.ndarray, ...]) -> np.ndarray:
    """Fill NaN cells via 3D linear interp; cells outside convex hull stay NaN."""
    finite = np.isfinite(arr)
    if not finite.any() or finite.all():
        return arr
    # Interpolate in log10 if all valid values are strictly positive (Tg,
    # abundances, line emission rates).  Otherwise interpolate in raw space
    # (some energy:: fields like PsiGD are signed).
    use_log = (arr[finite] > 0).all()
    work = np.where(arr > 0, np.log10(arr), np.nan) if use_log else arr.copy()
    pts = np.array(np.meshgrid(*log_axes, indexing='ij')).reshape(3, -1).T
    vals = work.ravel()
    mask = np.isfinite(vals)
    fill_pts = pts[~mask]
    if len(fill_pts) == 0:
        return arr
    filled = griddata(pts[mask], vals[mask], fill_pts, method='linear')
    # NOTE: NO nearest-neighbour fallback — cells outside the convex hull of
    # valid data stay NaN.  That's the whole point of option B.
    out = vals.copy()
    out[~mask] = filled
    out = out.reshape(arr.shape)
    return 10 ** out if use_log else out


def main(src_path: Path, dst_path: Path) -> None:
    print(f"[src] {src_path}")
    print(f"[dst] {dst_path}")
    dst_path.parent.mkdir(exist_ok=True, parents=True)
    raw = np.load(src_path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}

    Tg = data['tg_final']
    garbage = np.isfinite(Tg) & (Tg > 1e6)
    nan = np.isnan(Tg)
    fm = data['failure_mask']
    bad = fm | nan | garbage
    print(f"  failure_mask: {int(fm.sum())}  NaN: {int(nan.sum())}  "
          f"garbage (Tg>1e6): {int(garbage.sum())}  combined bad: {int(bad.sum())}")

    # Zero-out all per-cell arrays at bad locations so they're treated as NaN
    # by the fill (failure_mask is updated to include garbage so analytics can
    # still flag those cells later).
    data['failure_mask'] = fm | garbage
    per_cell_shape = Tg.shape

    for fld in ['tg_final', 'mu_values', 'cv_values', 'Eint_values']:
        if fld in data:
            a = data[fld].astype(float)
            a[bad] = np.nan
            data[fld] = a
    for k in list(data.keys()):
        if k.startswith('energy::') or k.endswith(
                ('_abundance', '_lumPerH', '_freq', '_intIntensity',
                 '_intTB', '_tau', '_tauDust')):
            a = data[k].astype(float)
            if a.shape == per_cell_shape:
                a[bad] = np.nan
                data[k] = a

    log_axes = (np.log10(data['nH_values']),
                np.log10(data['col_density_values']),
                np.log10(data['dVdr_values']))

    # Collect every shape-matching per-cell array, then fill each.
    fields_to_fill = ['tg_final', 'mu_values', 'cv_values', 'Eint_values']
    for k in list(data.keys()):
        if k.startswith('energy::') or k.endswith(
                ('_abundance', '_lumPerH', '_intIntensity', '_intTB',
                 '_tau', '_tauDust')):
            if data[k].shape == per_cell_shape:
                fields_to_fill.append(k)
    print(f"  fields to fill: {len(fields_to_fill)}")

    nan_before = int(np.isnan(data['tg_final']).sum())
    t0 = time.time()
    for fld in fields_to_fill:
        data[fld] = fill_in_hull(data[fld], log_axes)
    print(f"  filled in {time.time() - t0:.1f}s")

    nan_after = int(np.isnan(data['tg_final']).sum())
    Tg_new = data['tg_final']
    fin_now = np.isfinite(Tg_new) & (Tg_new > 0)
    print(f"  tg_final NaN: {nan_before} → {nan_after}")
    print(f"  ↑ cells still NaN after fill (outside convex hull → not extrapolated)")
    print(f"  tg_final range: {float(np.nanmin(Tg_new)):.2f} – {float(np.nanmax(Tg_new)):.2e} K "
          f"({int(fin_now.sum())} finite / {Tg_new.size})")

    np.savez_compressed(dst_path, **data)
    print(f"[saved] {dst_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
