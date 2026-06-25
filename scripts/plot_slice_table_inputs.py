#!/usr/bin/env python
"""5-panel midplane-x slice of the table's inputs + outputs:
   density | column_density_H | dVdr_lvg | T_QUOKKA | T_DESPOTIC
plus an annotation of the CONSTANT DESPOTIC build parameters (the table's only
variables are nH, N_H, dVdr; everything else — radiation, dust, etc. — is fixed).

Run:
    /opt/homebrew/Caskroom/miniconda/base/envs/yt-env/bin/python \
        scripts/plot_slice_table_inputs.py
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
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa

from quokka2s.pipeline.prep import config as cfg                       # noqa
from quokka2s.pipeline.prep import physics_fields as phys              # noqa
from quokka2s.data_handling import YTDataProvider, make_downsampled_dataset  # noqa
from quokka2s.pipeline.cache import compute_cache_key, cache_root_for_dataset  # noqa

SLICE_AXIS = 'x'
AX_IDX = {'x': 0, 'y': 1, 'z': 2}[SLICE_AXIS]

# (field, label, cmap, fixed_log_vmin, fixed_log_vmax)  None=percentile range
PANELS = [
    (('gas', 'number_density_H'),    r'$\log_{10}n_{\rm H}$ [cm$^{-3}$]',              'inferno', None, None),
    (('gas', 'column_density_H'),    r'$\log_{10}N_{\rm H}$ (6-dir+$L_{\rm ext}$) [cm$^{-2}$]', 'cividis', None, None),
    (('gas', 'dVdr_lvg'),            r'$\log_{10}|\nabla\!\cdot v|/3$ [s$^{-1}$]',     'plasma',  None, None),
    (('gas', 'temperature_quokka'),  r'$\log_{10}T_{\rm QUOKKA}$ [K]',                 'turbo',   2.0, 8.0),
    (('gas', 'temperature_despotic'),r'$\log_{10}T_{\rm DESPOTIC}$ [K]',               'turbo',   2.0, 8.0),
]

# Constant DESPOTIC build parameters (from tables/solver.py — fixed for every
# table cell; the only varied axes are nH, N_H, dVdr).
BUILD_PARAMS = (
    r'$\bf{DESPOTIC\ build\ constants}$ (fixed ∀ cell; table axes = $n_{\rm H},\,N_{\rm H},\,dV\!/dr$):'
    '\n'
    r'network = GOW (Gong, Ostriker & Wolfire 2017)   escape = LVG   evolveTemp = iterateDust'
    '\n'
    r'radiation:  $\chi_{\rm ISRF}=1.0$ (Draine/solar nbhd)   CR $\zeta=2{\times}10^{-17}\,$s$^{-1}$   '
    r'$T_{\rm CMB}=2.73\,$K   $T_{\rm rad,dust}=0$'
    '\n'
    r'dust:  $Z_d=1.0$   $\sigma_{\rm PE}=10^{-21}$   $\sigma_{\rm ISRF}=3{\times}10^{-22}$   '
    r'$\sigma_{10}=2{\times}10^{-25}\,$cm$^2$   $\beta=2.0$   $\alpha_{\rm GD}=3.2{\times}10^{-34}$'
    '\n'
    r'gas:  $\sigma_{\rm NT}=2\,$km/s   $x_{\rm He}=0.1$   ($x_{o\rm H2},x_{p\rm H2}$ init $=0.1,0.4$)   '
    r'$X_{\rm H}=0.74$   $A_V/N_{\rm H}=4{\times}10^{-22}\,$mag cm$^2$'
)


def main():
    ds = yt.load(cfg.YT_DATASET_PATH)
    if cfg.DOWNSAMPLE_FACTOR > 1:
        ds = make_downsampled_dataset(ds, cfg.DOWNSAMPLE_FACTOR)
    phys.add_all_fields(ds)
    key = compute_cache_key(dataset_path=cfg.YT_DATASET_PATH, despotic_table_path=cfg.DESPOTIC_TABLE_PATH,
                            downsample_factor=cfg.DOWNSAMPLE_FACTOR,
                            column_extension_lateral_kpc=cfg.COLUMN_EXTENSION_LATERAL_KPC)
    p = YTDataProvider(ds, cache_root=cache_root_for_dataset(cfg.YT_DATASET_PATH), cache_key=key)

    extent = None
    slices = []
    for field, label, cmap, vlo, vhi in PANELS:
        arr_u, ext = p.get_slab_z(field)
        arr = np.asarray(arr_u.in_cgs())
        if extent is None:
            extent = [float(v.in_units('kpc').value) for v in ext[SLICE_AXIS]]
        idx = arr.shape[AX_IDX] // 2
        sl = [slice(None)] * 3; sl[AX_IDX] = idx
        slices.append(np.asarray(arr[tuple(sl)]).copy())   # (ny, nz)
        del arr_u, arr

    n = len(PANELS)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n, 12), sharey=True,
                             gridspec_kw={'wspace': 0.08, 'top': 0.84, 'bottom': 0.16})
    ext_plot = [extent[0], extent[1], extent[2], extent[3]]
    for ax, data2d, (field, label, cmap, vlo, vhi) in zip(axes, slices, PANELS):
        with np.errstate(divide='ignore', invalid='ignore'):
            pos = data2d > 0
            logd = np.where(pos, np.log10(np.where(pos, data2d, 1.0)), np.nan)
        if vlo is None:
            vlo = float(np.nanmin(logd)); vhi = float(np.nanmax(logd))   # full data range (no percentile clip)
        im = ax.imshow(logd.T, origin='lower', extent=ext_plot, aspect='auto',
                       cmap=cmap, norm=Normalize(vlo, vhi))
        ax.tick_params(labelsize=8)
        cax = make_axes_locatable(ax).append_axes('top', size='2.5%', pad=0.5)
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.ax.tick_params(labelsize=7, top=True, bottom=False, labeltop=True, labelbottom=False)
        cax.set_title(label, fontsize=8.5, pad=4)
        ax.set_xlabel('y [kpc]', fontsize=9)
    axes[0].set_ylabel('z [kpc]', fontsize=10)

    fig.suptitle(f'Table inputs & outputs — midplane-x slice   '
                 f'{os.path.basename(cfg.YT_DATASET_PATH)} '
                 f'(down={cfg.DOWNSAMPLE_FACTOR}, $L_{{\\rm ext}}$={cfg.COLUMN_EXTENSION_LATERAL_KPC:g} kpc)',
                 fontsize=12, y=0.90)
    fig.text(0.5, 0.045, BUILD_PARAMS, ha='center', va='top', fontsize=8.5,
             family='monospace',
             bbox=dict(boxstyle='round', fc='#f5f5f5', ec='0.6', alpha=0.95))

    out = Path(cfg.OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    png = out / 'slice_table_inputs.png'
    fig.savefig(str(png), dpi=190, bbox_inches='tight')
    plt.close(fig)
    print(f'[out] {png}')


if __name__ == '__main__':
    main()
