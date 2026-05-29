#!/usr/bin/env python
"""Re-plot the column_density_H midplane-x slice with (left) the p0.5-p99.5
percentile colour range I used before vs (right) the FULL data range, to show
the apparent 'constant vertical stripes' are colourbar saturation, not real."""
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

from quokka2s.pipeline.prep import config as cfg                       # noqa
from quokka2s.pipeline.prep import physics_fields as phys              # noqa
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

    arr_u, ed = p.get_slab_z(('gas', 'column_density_H'))
    arr = np.asarray(arr_u.in_cgs())
    ext = [float(v.in_units('kpc').value) for v in ed['x']]
    xi = arr.shape[0] // 2
    sl = arr[xi, :, :]
    with np.errstate(divide='ignore'):
        logd = np.where(sl > 0, np.log10(sl), np.nan)

    p_lo, p_hi = np.nanpercentile(logd, 0.5), np.nanpercentile(logd, 99.5)
    f_lo, f_hi = np.nanmin(logd), np.nanmax(logd)
    print(f'L_ext={cfg.COLUMN_EXTENSION_LATERAL_KPC:g} kpc  x-slice {xi}')
    print(f'  percentile range p0.5-p99.5 : [{p_lo:.2f}, {p_hi:.2f}]  ({p_hi-p_lo:.2f} dex)')
    print(f'  FULL data range             : [{f_lo:.2f}, {f_hi:.2f}]  ({f_hi-f_lo:.2f} dex)')

    fig, axes = plt.subplots(1, 2, figsize=(8, 12), sharey=True, gridspec_kw={'wspace': 0.25})
    for ax, (vlo, vhi, ttl) in zip(axes, [
            (p_lo, p_hi, f'percentile p0.5–p99.5\n[{p_lo:.1f}, {p_hi:.1f}] ({p_hi-p_lo:.1f} dex)'),
            (f_lo, f_hi, f'FULL range\n[{f_lo:.1f}, {f_hi:.1f}] ({f_hi-f_lo:.1f} dex)')]):
        im = ax.imshow(logd.T, origin='lower', extent=ext, aspect='auto', cmap='cividis',
                       norm=Normalize(vlo, vhi))
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, label=r'$\log_{10}N_{\rm H}$ [cm$^{-2}$]')
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel('y [kpc]', fontsize=10)
    axes[0].set_ylabel('z [kpc]', fontsize=11)
    fig.suptitle(f'column_density_H midplane-x slice — colourbar comparison '
                 f'(L_ext={cfg.COLUMN_EXTENSION_LATERAL_KPC:g} kpc)\n'
                 'full range reveals the midplane peak → edge fall-off (not flat stripes)',
                 fontsize=11, y=0.95)
    out = Path(f"{cfg._OUTPUT_ROOT}/{cfg._DATASET_BASENAME}_down{cfg.DOWNSAMPLE_FACTOR}"
               f"_Lext{cfg.COLUMN_EXTENSION_LATERAL_KPC:g}kpc/")
    out.mkdir(parents=True, exist_ok=True)
    png = out / 'NH_slice_colorbar_compare.png'
    fig.savefig(str(png), dpi=170, bbox_inches='tight'); plt.close(fig)
    print(f'[out] {png}')


if __name__ == '__main__':
    main()
