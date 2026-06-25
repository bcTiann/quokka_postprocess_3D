#!/usr/bin/env python
"""Where do the 3D DESPOTIC table's failed cells sit, and does the simulation
actually use them?

  - failure_mask is (nH, NH, dVdr); failures are per (nH, NH) (broadcast over
    dVdr), so collapse to a 35x35 grid.
  - bin the simulation cells onto the table grid points (nearest, via midpoint
    edges) → sim_count[i,j].
  - a failed grid point matters only if sim cells land on it or its neighbours
    (linear interpolation uses the surrounding grid points).

Run:
    /opt/homebrew/Caskroom/miniconda/base/envs/yt-env/bin/python \
        scripts/analyze_table_failures.py
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np

_SRC = Path(__file__).resolve().parents[1] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yt  # noqa
import matplotlib; matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt  # noqa
from matplotlib.colors import LogNorm  # noqa
from scipy.ndimage import binary_dilation  # noqa

from quokka2s.pipeline.prep import config as cfg                       # noqa
from quokka2s.pipeline.prep import physics_fields as phys              # noqa
from quokka2s.data_handling import YTDataProvider, make_downsampled_dataset  # noqa
from quokka2s.pipeline.cache import compute_cache_key, cache_root_for_dataset  # noqa
from quokka2s.tables import load_table                                 # noqa


def _log_edges(vals):
    lv = np.log10(vals); d = np.diff(lv)
    e = np.empty(len(vals) + 1)
    e[1:-1] = lv[:-1] + d / 2; e[0] = lv[0] - d[0] / 2; e[-1] = lv[-1] + d[-1] / 2
    return e


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--table', default=None,
                    help='path to despotic_table.npz (default cfg.DESPOTIC_TABLE_PATH)')
    ap.add_argument('--out-tag', default=None,
                    help='tag suffix for output PNG name (default: table dir basename)')
    args = ap.parse_args()

    t0 = time.time()
    table_path = args.table or cfg.DESPOTIC_TABLE_PATH
    print(f'[table] {table_path}')
    table = load_table(table_path)
    nH_v, col_v = table.nH_values, table.col_density_values
    fm = table.failure_mask
    if fm is None:
        fm = np.zeros(table.tg_final.shape, dtype=bool)

    # "Bad" = failed OR NaN OR garbage (Tg > 1e6 K, see audit doc).
    nan_mask = np.isnan(table.tg_final)
    garbage_mask = np.isfinite(table.tg_final) & (table.tg_final > 1e6)
    bad3d = fm | nan_mask | garbage_mask
    fail2d = bad3d.any(axis=2) if bad3d.ndim == 3 else bad3d   # (nH, NH); collapsed
    # Also report per-(nH,NH) what fraction of dVdr indices are bad
    if bad3d.ndim == 3:
        per_nhNH_frac = bad3d.mean(axis=2)
        bad_all_dV = (bad3d.all(axis=2)).sum()
        print(f'cells where ALL 35 dVdr indices are bad: {int(bad_all_dV)} / {fail2d.size}')
        print(f'cells where ANY  dVdr index is bad:      {int(fail2d.sum())} / {fail2d.size}')
        print(f'  fm-only: {int(fm.sum())}, NaN-only: {int(nan_mask.sum())}, garbage-only: {int(garbage_mask.sum())}')

    nfail = int(fail2d.sum()); ntot = fail2d.size
    print(f'table: {nH_v.size} x {col_v.size} = {ntot} (nH,NH) grid points;  '
          f'failed = {nfail}  ({100*nfail/ntot:.2f}%)')
    if nfail:
        fi, fj = np.where(fail2d)
        print(f'failed nH range : {nH_v[fi].min():.2e} .. {nH_v[fi].max():.2e} cm^-3')
        print(f'failed N_H range: {col_v[fj].min():.2e} .. {col_v[fj].max():.2e} cm^-2')

    # --- simulation coverage (current L_ext) ---
    ds = yt.load(cfg.YT_DATASET_PATH)
    if cfg.DOWNSAMPLE_FACTOR > 1:
        ds = make_downsampled_dataset(ds, cfg.DOWNSAMPLE_FACTOR)
    phys.add_all_fields(ds)
    # Cache key still uses cfg.DESPOTIC_TABLE_PATH (the canonical path) — that's
    # what the on-disk intermediates were keyed on; passing args.table here would
    # only force a fresh recompute if checking a non-canonical table.
    key = compute_cache_key(dataset_path=cfg.YT_DATASET_PATH,
                            despotic_table_path=cfg.DESPOTIC_TABLE_PATH,
                            downsample_factor=cfg.DOWNSAMPLE_FACTOR,
                            column_extension_lateral_kpc=cfg.COLUMN_EXTENSION_LATERAL_KPC)
    p = YTDataProvider(ds, cache_root=cache_root_for_dataset(cfg.YT_DATASET_PATH), cache_key=key)
    nH_u, _ = p.get_slab_z(('gas', 'number_density_H'))
    col_u, _ = p.get_slab_z(('gas', 'column_density_H'))
    n_H = np.asarray(nH_u.in_cgs()).ravel()
    colden = np.asarray(col_u.in_cgs()).ravel()

    with np.errstate(divide='ignore', invalid='ignore'):
        log_nH = np.log10(np.where(n_H > 0, n_H, np.nan))
        log_NH = np.log10(np.where(colden > 0, colden, np.nan))
    ok = np.isfinite(log_nH) & np.isfinite(log_NH)

    nH_edges, col_edges = _log_edges(nH_v), _log_edges(col_v)
    # cells get CLIPPED to the table range in the pipeline → clip here too so
    # out-of-range cells pile onto the edge grid points (as they do in practice).
    lo_nH, hi_nH = nH_edges[0], nH_edges[-1]
    lo_c,  hi_c  = col_edges[0], col_edges[-1]
    x = np.clip(log_NH[ok], lo_c + 1e-9, hi_c - 1e-9)
    y = np.clip(log_nH[ok], lo_nH + 1e-9, hi_nH - 1e-9)
    sim_count, _, _ = np.histogram2d(y, x, bins=[nH_edges, col_edges])   # (nH, NH)

    used        = (sim_count > 0)
    used_dilate = binary_dilation(used, iterations=1)   # interp-reach: ±1 grid pt
    fail_used   = int((fail2d & used).sum())
    fail_near   = int((fail2d & used_dilate).sum())
    print(f'\nsim cells: {int(ok.sum())} valid')
    print(f'failed grid points with a sim cell ON them            : {fail_used}/{nfail}')
    print(f'failed grid points with a sim cell in their ±1 neighbourhood: {fail_near}/{nfail}')
    print(f'failed grid points in EMPTY region (never used)       : {nfail - fail_near}/{nfail}')

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    sc_masked = np.where(sim_count > 0, sim_count, np.nan)
    im = ax.pcolormesh(10 ** col_edges, 10 ** nH_edges, sc_masked, cmap='Greys',
                       norm=LogNorm(vmin=1, vmax=np.nanmax(sc_masked)), shading='flat')
    fig.colorbar(im, ax=ax, label='sim cells per table grid cell', fraction=0.046, pad=0.02)
    if nfail:
        ax.scatter(col_v[fj], nH_v[fi], marker='x', s=55, c='red', linewidths=1.4,
                   label=f'failed grid point ({nfail})', zorder=5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(10 ** col_edges[0], 10 ** col_edges[-1])
    ax.set_ylim(10 ** nH_edges[0], 10 ** nH_edges[-1])
    ax.set_xlabel(r'Column Density $N_{\rm H}$ [cm$^{-2}$]', fontsize=12)
    ax.set_ylabel(r'$n_{\rm H}$ [cm$^{-3}$]', fontsize=12)
    ax.set_title('3D table failures (red x) vs simulation coverage (grey)\n'
                 f'{os.path.basename(cfg.DESPOTIC_TABLE_PATH.rstrip("/").split("/")[-2])}  '
                 f'L_ext={cfg.COLUMN_EXTENSION_LATERAL_KPC:g} kpc;  '
                 f'{nfail}/{ntot} failed, {fail_near} near sim cells', fontsize=10)
    if nfail:
        ax.legend(loc='lower right', framealpha=0.9)
    out = Path(cfg.OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    tag = args.out_tag or Path(table_path).parent.name
    png = out / f'table_failures_vs_coverage_{tag}.png'
    fig.savefig(str(png), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[out] {png}')
    print(f'[done] {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
