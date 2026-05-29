#!/usr/bin/env python
"""Where do the simulation cells sit in the DESPOTIC table's (N_H, n_H) plane?

Same axes as view_table.py's T_g heatmap:
    x = log10 N_H  (column_density_H, 6-dir harmonic + L_ext) [cm^-2]
    y = log10 n_H  (number_density_H)                          [cm^-3]
colour = log10( number of cells per bin )   (set WEIGHT='mass' for mass instead)

Axis ranges are pinned to the table's nH / N_H span so this overlays the
table-view heatmap one-to-one (shows which table region the data actually uses).

Set config.py DOWNSAMPLE_FACTOR=1; L_ext from cfg.COLUMN_EXTENSION_LATERAL_KPC
(or PHASE_LEXT env). Run:
    /opt/homebrew/Caskroom/miniconda/base/envs/yt-env/bin/python \
        quokka2s/scripts/plot_cells_in_nH_NH_plane.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yt  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from quokka2s.pipeline.prep import config as cfg                       # noqa: E402
from quokka2s.pipeline.prep import physics_fields as phys              # noqa: E402
from quokka2s.data_handling import YTDataProvider, make_downsampled_dataset  # noqa: E402
from quokka2s.pipeline.cache import compute_cache_key, cache_root_for_dataset  # noqa: E402
from quokka2s.tables import load_table                                 # noqa: E402

N_BINS = 80
WEIGHT = 'count'   # 'count' = number of cells per bin; 'mass' = Σ cell mass
M_SUN_G = 1.98892e33


def main():
    t0 = time.time()
    _lext = os.environ.get('PHASE_LEXT')
    if _lext is not None:
        cfg.COLUMN_EXTENSION_LATERAL_KPC = float(_lext)
        print(f'[override] L_ext = {cfg.COLUMN_EXTENSION_LATERAL_KPC} kpc')

    ds = yt.load(cfg.YT_DATASET_PATH)
    if cfg.DOWNSAMPLE_FACTOR > 1:
        ds = make_downsampled_dataset(ds, cfg.DOWNSAMPLE_FACTOR)
    phys.add_all_fields(ds)
    key = compute_cache_key(
        dataset_path                 = cfg.YT_DATASET_PATH,
        despotic_table_path          = cfg.DESPOTIC_TABLE_PATH,
        downsample_factor            = cfg.DOWNSAMPLE_FACTOR,
        column_extension_lateral_kpc = cfg.COLUMN_EXTENSION_LATERAL_KPC,
    )
    p = YTDataProvider(ds, cache_root=cache_root_for_dataset(cfg.YT_DATASET_PATH), cache_key=key)

    print(f'[load] down={cfg.DOWNSAMPLE_FACTOR} L_ext={cfg.COLUMN_EXTENSION_LATERAL_KPC} kpc')
    nH_u,     _ = p.get_slab_z(('gas', 'number_density_H'))
    colden_u, _ = p.get_slab_z(('gas', 'column_density_H'))
    n_H    = np.asarray(nH_u.in_cgs()).ravel()
    colden = np.asarray(colden_u.in_cgs()).ravel()

    if WEIGHT == 'mass':
        rho_u, _ = p.get_slab_z(('gas', 'density'))
        dx_u, _ = p.get_slab_z(('boxlib', 'dx'))
        dy_u, _ = p.get_slab_z(('boxlib', 'dy'))
        dz_u, _ = p.get_slab_z(('boxlib', 'dz'))
        dV = (np.asarray(dx_u.in_cgs()) * np.asarray(dy_u.in_cgs()) * np.asarray(dz_u.in_cgs())).ravel()
        w = np.asarray(rho_u.in_cgs()).ravel() * dV / M_SUN_G
        wlabel = r'$\log_{10}\,\left(\sum M_{\rm cell}\right)$ per bin  [M$_\odot$]'
    else:
        w = None
        wlabel = r'$\log_{10}\,(\#\,\mathrm{cells})$ per bin'
    print(f'[load] done in {time.time()-t0:.1f}s  ({n_H.size} cells)')

    # Pin the plane to the TABLE's nH / N_H span so it overlays view_table.
    table = load_table(cfg.DESPOTIC_TABLE_PATH)
    nH_lo, nH_hi   = table.nH_values.min(),         table.nH_values.max()
    col_lo, col_hi = table.col_density_values.min(), table.col_density_values.max()

    with np.errstate(divide='ignore', invalid='ignore'):
        log_nH = np.log10(np.where(n_H > 0, n_H, np.nan))
        log_NH = np.log10(np.where(colden > 0, colden, np.nan))
    ok = np.isfinite(log_nH) & np.isfinite(log_NH)
    # fraction outside the table footprint (these get clipped to edges in the pipeline)
    inside = ok & (n_H >= nH_lo) & (n_H <= nH_hi) & (colden >= col_lo) & (colden <= col_hi)
    print(f'[range] {100*np.mean(inside[ok]):.2f}% of cells inside the table footprint; '
          f'{100*(1-np.mean(inside[ok])):.2f}% outside (clipped in pipeline)')

    x = log_NH[ok]; y = log_nH[ok]
    wv = None if w is None else w[ok]
    x_edges = np.linspace(np.log10(col_lo), np.log10(col_hi), N_BINS + 1)
    y_edges = np.linspace(np.log10(nH_lo),  np.log10(nH_hi),  N_BINS + 1)
    H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=wv)

    with np.errstate(divide='ignore'):
        logH = np.log10(np.where(H > 0, H, np.nan))
    vmax = float(np.nanmax(logH))
    vmin = vmax - 6.0

    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    im = ax.pcolormesh(10 ** x_edges, 10 ** y_edges, logH.T, cmap='viridis',
                       norm=Normalize(vmin, vmax), shading='flat')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(wlabel, fontsize=11)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(col_lo, col_hi); ax.set_ylim(nH_lo, nH_hi)
    ax.set_xlabel(r'Column Density $N_{\rm H}$ [cm$^{-2}$]', fontsize=12)
    ax.set_ylabel(r'$n_{\rm H}$ [cm$^{-3}$]', fontsize=12)
    ax.set_title('Where the simulation cells sit in the table plane '
                 f'({WEIGHT})\n{os.path.basename(cfg.YT_DATASET_PATH)}  '
                 f'(down={cfg.DOWNSAMPLE_FACTOR}, $L_{{\\rm ext}}$={cfg.COLUMN_EXTENSION_LATERAL_KPC:g} kpc)',
                 fontsize=10)

    # rebuild OUTPUT_DIR for the (possibly overridden) L_ext so 0/9/99 don't
    # overwrite each other (cfg.OUTPUT_DIR was frozen at import time).
    lext_tag = f"_Lext{cfg.COLUMN_EXTENSION_LATERAL_KPC:g}kpc"
    geom = "_sphere" if os.environ.get("DESPOTIC_GEOM", "LVG").lower() == "sphere" else ""
    out_dir = Path(f"{cfg._OUTPUT_ROOT}/{cfg._DATASET_BASENAME}_down{cfg.DOWNSAMPLE_FACTOR}{lext_tag}{geom}/")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f'cells_in_nH_NH_plane_{WEIGHT}.png'
    fig.savefig(str(png), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[out] {png}')
    print(f'[done] {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
