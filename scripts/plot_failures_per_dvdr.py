#!/usr/bin/env python
"""Per-dVdr-slice view of the 3D table: simulation cell coverage (grey) + failed
grid points (red x), one panel per dVdr table index (35 total), PLUS per-axis
out-of-range (clipping) statistics.

Run:
    /opt/homebrew/Caskroom/miniconda/base/envs/yt-env/bin/python \
        scripts/plot_failures_per_dvdr.py
"""
from __future__ import annotations
import os, sys, time, math
from pathlib import Path
import numpy as np

_SRC = Path(__file__).resolve().parents[1] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yt  # noqa
import matplotlib; matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt  # noqa
from matplotlib.colors import LogNorm  # noqa

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


def _range_stats(name, data, lo, hi, unit):
    fin = data[np.isfinite(data) & (data > 0)]
    below = np.mean(fin < lo); above = np.mean(fin > hi)
    print(f'  {name:6s} table[{lo:.2e},{hi:.2e}] {unit}:  '
          f'data[{fin.min():.2e},{fin.max():.2e}]  '
          f'below={100*below:.3f}%  above={100*above:.3f}%  '
          f'(clipped total {100*(below+above):.3f}%)')


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--table', default=None,
                    help='path to despotic_table.npz (default cfg.DESPOTIC_TABLE_PATH)')
    ap.add_argument('--out-tag', default=None,
                    help='tag suffix for output PNG name (default: table parent dir name)')
    args = ap.parse_args()

    t0 = time.time()
    table_path = args.table or cfg.DESPOTIC_TABLE_PATH
    print(f'[table] {table_path}')
    table = load_table(table_path)
    nH_v, col_v, dv_v = table.nH_values, table.col_density_values, table.dVdr_values

    # 3D bad-cell mask = failure_mask ∪ (Tg is NaN) ∪ (Tg > 1e6 garbage)
    fm_orig = table.failure_mask if table.failure_mask is not None \
              else np.zeros(table.tg_final.shape, dtype=bool)
    nan_mask = np.isnan(table.tg_final)
    garbage_mask = np.isfinite(table.tg_final) & (table.tg_final > 1e6)
    fm = fm_orig | nan_mask | garbage_mask                  # (nH, NH, dVdr)
    n_dv = len(dv_v)
    print(f'table grid: nH={nH_v.size}, NH={col_v.size}, dVdr={n_dv}')
    print(f'  failure_mask: {int(fm_orig.sum())}  NaN: {int(nan_mask.sum())}  '
          f'garbage(Tg>1e6): {int(garbage_mask.sum())}  combined bad: {int(fm.sum())}')

    ds = yt.load(cfg.YT_DATASET_PATH)
    if cfg.DOWNSAMPLE_FACTOR > 1:
        ds = make_downsampled_dataset(ds, cfg.DOWNSAMPLE_FACTOR)
    phys.add_all_fields(ds)
    key = compute_cache_key(dataset_path=cfg.YT_DATASET_PATH,
                            despotic_table_path=cfg.DESPOTIC_TABLE_PATH,
                            downsample_factor=cfg.DOWNSAMPLE_FACTOR,
                            column_extension_lateral_kpc=cfg.COLUMN_EXTENSION_LATERAL_KPC)
    p = YTDataProvider(ds, cache_root=cache_root_for_dataset(cfg.YT_DATASET_PATH), cache_key=key)
    n_H    = np.asarray(p.get_slab_z(('gas', 'number_density_H'))[0].in_cgs()).ravel()
    colden = np.asarray(p.get_slab_z(('gas', 'column_density_H'))[0].in_cgs()).ravel()
    dvdr   = np.asarray(p.get_slab_z(('gas', 'dVdr_lvg'))[0].in_cgs()).ravel()

    print(f'\n=== per-axis out-of-range (relative to table edges) ===')
    _range_stats('nH',   n_H,    nH_v.min(),  nH_v.max(),  'cm^-3')
    _range_stats('NH',   colden, col_v.min(), col_v.max(), 'cm^-2')
    _range_stats('dVdr', dvdr,   dv_v.min(),  dv_v.max(),  's^-1')

    # log + clip to table range (mirror the pipeline's np.clip before interp)
    nH_e, col_e, dv_e = _log_edges(nH_v), _log_edges(col_v), _log_edges(dv_v)
    with np.errstate(divide='ignore', invalid='ignore'):
        lnH, lNH, ldv = np.log10(n_H), np.log10(colden), np.log10(dvdr)
    ok = np.isfinite(lnH) & np.isfinite(lNH) & np.isfinite(ldv)
    lnH, lNH, ldv = lnH[ok], lNH[ok], ldv[ok]
    lnH = np.clip(lnH, nH_e[0] + 1e-9, nH_e[-1] - 1e-9)
    lNH = np.clip(lNH, col_e[0] + 1e-9, col_e[-1] - 1e-9)
    ldv = np.clip(ldv, dv_e[0] + 1e-9, dv_e[-1] - 1e-9)

    # assign each cell to a dVdr table bin
    dv_bin = np.clip(np.digitize(ldv, dv_e) - 1, 0, n_dv - 1)

    # global colour scale across all dVdr slices
    vmax = 1
    counts = []
    for d in range(n_dv):
        sel = dv_bin == d
        H, _, _ = np.histogram2d(lnH[sel], lNH[sel], bins=[nH_e, col_e])
        counts.append(H)
        vmax = max(vmax, H.max())

    ncol = 7; nrow = math.ceil(n_dv / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.5 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    norm = LogNorm(vmin=1, vmax=vmax)
    im = None
    for d in range(nrow * ncol):
        ax = axes[d // ncol][d % ncol]
        if d >= n_dv:
            ax.set_visible(False); continue
        H = np.where(counts[d] > 0, counts[d], np.nan)
        im = ax.pcolormesh(10 ** col_e, 10 ** nH_e, H, cmap='Greys', norm=norm, shading='flat')
        if fm is not None:
            fi, fj = np.where(fm[:, :, d])
            if fi.size:
                ax.scatter(col_v[fj], nH_v[fi], marker='x', s=18, c='red', linewidths=0.8, zorder=5)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(f'dVdr={dv_v[d]:.1e}  (n={int(counts[d].sum())})', fontsize=7)
        ax.tick_params(labelsize=6)
    fig.supxlabel(r'Column Density $N_{\rm H}$ [cm$^{-2}$]', fontsize=11)
    fig.supylabel(r'$n_{\rm H}$ [cm$^{-3}$]', fontsize=11)
    nfail = int(fm.sum()) if fm is not None else 0
    fig.suptitle(f'Per-dVdr-slice sim coverage (grey) + table failures (red x)   '
                 f'L_ext={cfg.COLUMN_EXTENSION_LATERAL_KPC:g} kpc;  total failed={nfail}',
                 fontsize=11, y=1.005)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, label='sim cells / table cell')

    out = Path(cfg.OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    tag = args.out_tag or Path(table_path).parent.name
    png = out / f'failures_and_coverage_per_dvdr_{tag}.png'
    fig.savefig(str(png), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[out] {png}')
    # dVdr occupancy summary
    occ = np.array([c.sum() for c in counts])
    print(f'[dVdr] slices with sim cells: {int((occ>0).sum())}/{n_dv}; '
          f'busiest dVdr={dv_v[int(np.argmax(occ))]:.2e} ({int(occ.max())} cells)')
    print(f'[done] {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
