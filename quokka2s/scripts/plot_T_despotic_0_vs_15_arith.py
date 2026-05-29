#!/usr/bin/env python
"""T_DESPOTIC L_ext = 0 vs 15 kpc, with ARITHMETIC-mean column density.
Per slice: [T_DSP@0, T_DSP@15, plain ratio T_DSP@0 / T_DSP@15], 4 slices.
Reads the (arithmetic) TemperatureSlicesTask caches. Full-range T colourbars.
"""
import glob, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = Path('/Users/baochen/quokka_postprocessing/output')
DIRS = {0: 'plt0655228_down1_Lext0kpc', 15: 'plt0655228_down1_Lext15kpc'}


def load(L):
    cand = sorted(glob.glob(str(ROOT / DIRS[L] / 'task_intermediates' / 'TemperatureSlicesTask_*.h5')),
                  key=os.path.getmtime)
    with h5py.File(cand[-1], 'r') as f:
        return f['T_md_slices'][:], f['slice_indices'][:], f['extent'][:]


T0, sidx, extent = load(0)
T15, sidx15, _ = load(15)
assert np.array_equal(sidx, sidx15)
n = T0.shape[0]
ext = [extent[0], extent[1], extent[2], extent[3]]

both = np.concatenate([T0[np.isfinite(T0) & (T0 > 0)], T15[np.isfinite(T15) & (T15 > 0)]])
tnorm = Normalize(np.log10(both.min()), np.log10(both.max()))     # full range
with np.errstate(all='ignore'):
    ratios = [T0[i] / T15[i] for i in range(n)]
allr = np.concatenate([r[np.isfinite(r) & (r > 0)] for r in ratios])
rlo, rhi = float(allr.min()), float(allr.max())
rnorm = TwoSlopeNorm(1.0, vmin=min(rlo, 0.999), vmax=max(rhi, 1.001))

plot_order = []
for i in range(n):
    plot_order.append(('T', T0[i],  rf'$\log_{{10}}T_{{\rm DSP}}$  $L=0$',      int(sidx[i])))
    plot_order.append(('T', T15[i], rf'$\log_{{10}}T_{{\rm DSP}}$  $L=15$ kpc', int(sidx[i])))
    plot_order.append(('R', ratios[i], r'$T_{\rm DSP}^{0}/T_{\rm DSP}^{15}$',   int(sidx[i])))

npan = len(plot_order)
fig, axes = plt.subplots(1, npan, figsize=(2.4 * npan, 12), sharey=True,
                         gridspec_kw={'wspace': 0.08, 'top': 0.86, 'bottom': 0.06})
for ax, (kind, arr, label, si) in zip(axes, plot_order):
    if kind == 'T':
        with np.errstate(all='ignore'):
            d = np.log10(np.where(arr > 0, arr, np.nan))
        im = ax.imshow(d.T, origin='lower', extent=ext, aspect='auto', cmap='turbo', norm=tnorm)
    else:
        im = ax.imshow(arr.T, origin='lower', extent=ext, aspect='auto', cmap='RdBu_r', norm=rnorm)
    ax.tick_params(labelsize=7)
    cax = make_axes_locatable(ax).append_axes('top', size='2.5%', pad=0.5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=7, top=True, bottom=False, labeltop=True, labelbottom=False)
    cb.ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
    cax.set_title(f'{label}\nx idx {si}', fontsize=8, pad=3)
axes[0].set_ylabel('z [kpc]', fontsize=10)
for ax in axes:
    ax.set_xlabel('y [kpc]', fontsize=8)
fig.suptitle('T_DESPOTIC  L_ext 0 vs 15 kpc  (ARITHMETIC-mean column density, full-range T)\n'
             'ratio>1 = cooler at L_ext=15', fontsize=12, y=0.93)

OUT = ROOT / 'plt0655228_down1_LextDiff_0kpc_vs_9kpc'
OUT.mkdir(parents=True, exist_ok=True)
out = OUT / 'T_DESPOTIC_arith_0_vs_15.png'
fig.savefig(str(out), dpi=200, bbox_inches='tight'); plt.close(fig)
print(f'[out] {out}')
for i in range(n):
    r = ratios[i][np.isfinite(ratios[i])]
    print(f'  idx {int(sidx[i])}: T0/T15 median={np.median(r):.3f}  max={np.max(r):.3f}  '
          f'frac cooler(>1.02)={np.mean(r>1.02):.2f}')
