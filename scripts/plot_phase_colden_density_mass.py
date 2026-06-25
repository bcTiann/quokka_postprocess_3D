#!/usr/bin/env python
"""Phase plot in the (density, column density) plane, weighted by MASS.

x = log10 ρ [g cm⁻³]      (volume density)
y = log10 N_H [cm⁻²]      (6-direction harmonic-mean column density, incl. L_ext)
colour = log10( Σ cell mass in the bin )   [M_sun]

Unlike PhasePlotTask / PhaseColdenTask (which live on the ρ–T plane and use N_H
only as a colour), here N_H is the y-axis. Shows where the gas mass sits in the
(ρ, N_H) plane.

Set config.py to DOWNSAMPLE_FACTOR=1; L_ext comes from
cfg.COLUMN_EXTENSION_LATERAL_KPC, optionally overridden by PHASE_LEXT env var.

Run:
    /opt/homebrew/Caskroom/miniconda/base/envs/yt-env/bin/python \
        scripts/plot_phase_colden_density_mass.py
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

N_BINS    = 70          # bins per axis
COLOR_DEX = 7.0         # colour spans this many dex down from the most massive bin
M_SUN_G   = 1.98892e33


def main():
    t0 = time.time()
    _lext = os.environ.get('PHASE_LEXT')
    if _lext is not None:
        cfg.COLUMN_EXTENSION_LATERAL_KPC = float(_lext)
        print(f'[override] COLUMN_EXTENSION_LATERAL_KPC = {cfg.COLUMN_EXTENSION_LATERAL_KPC}')

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
    rho_u,    _ = p.get_slab_z(('gas', 'density'))
    colden_u, _ = p.get_slab_z(('gas', 'column_density_H'))   # cache hit
    dx_u,     _ = p.get_slab_z(('boxlib', 'dx'))
    dy_u,     _ = p.get_slab_z(('boxlib', 'dy'))
    dz_u,     _ = p.get_slab_z(('boxlib', 'dz'))

    rho    = np.asarray(rho_u.in_cgs()).ravel()
    colden = np.asarray(colden_u.in_cgs()).ravel()
    dV     = (np.asarray(dx_u.in_cgs()) * np.asarray(dy_u.in_cgs()) *
              np.asarray(dz_u.in_cgs())).ravel()
    mass_msun = rho * dV / M_SUN_G        # cell mass [M_sun]
    print(f'[load] done in {time.time()-t0:.1f}s  ({rho.size} cells)')

    with np.errstate(divide='ignore', invalid='ignore'):
        log_rho = np.log10(np.where(rho > 0, rho, np.nan))
        log_NH  = np.log10(np.where(colden > 0, colden, np.nan))
    ok = np.isfinite(log_rho) & np.isfinite(log_NH) & (mass_msun > 0)

    x = log_rho[ok]; y = log_NH[ok]; w = mass_msun[ok]
    x_edges = np.linspace(x.min(), x.max() + 1e-6, N_BINS + 1)
    y_edges = np.linspace(y.min(), y.max() + 1e-6, N_BINS + 1)
    H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=w)   # Σ mass per bin
    total_mass = w.sum()

    with np.errstate(divide='ignore'):
        logH = np.log10(np.where(H > 0, H, np.nan))
    vmax = float(np.nanmax(logH))
    vmin = vmax - COLOR_DEX

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.pcolormesh(x_edges, y_edges, logH.T, cmap='inferno',
                       norm=Normalize(vmin=vmin, vmax=vmax), shading='flat')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r'$\log_{10}\,\left(\sum M_{\rm cell}\right)$ per bin  [M$_\odot$]', fontsize=11)

    ax.set_xlabel(r'$\log_{10}\,\rho$ [g cm$^{-3}$]', fontsize=12)
    ax.set_ylabel(r'$\log_{10}\,N_{\rm H}$ (6-dir harmonic, +$L_{\rm ext}$) [cm$^{-2}$]', fontsize=12)
    ax.set_title('Mass-weighted phase plot: density vs column density\n'
                 f'{os.path.basename(cfg.YT_DATASET_PATH)}  '
                 f'(down={cfg.DOWNSAMPLE_FACTOR}, $L_{{\\rm ext}}$={cfg.COLUMN_EXTENSION_LATERAL_KPC:g} kpc; '
                 f'total mass {total_mass:.2e} M$_\\odot$)', fontsize=10)
    ax.grid(True, ls=':', lw=0.5, alpha=0.3)

    out_dir = Path(cfg.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / 'phase_density_colden_mass.png'
    fig.savefig(str(png), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[out] {png}')
    print(f'[done] {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
