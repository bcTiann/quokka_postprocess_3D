"""T_DESPOTIC L_ext=0 vs 9 kpc, multi-slice — laid out like TemperatureSlicesTask.

Single row of 3 x n_slices panels, interleaved per slice index:
    [ T_DSP@L0, T_DSP@L9, log10(T_DSP^0/T_DSP^9),   (idx0)
      T_DSP@L0, T_DSP@L9, log10(...),               (idx1)  ... ]

Mirrors temperature_slices.py: figsize (2.4*n_panels, 12), aspect='auto',
top horizontal colourbars, shared turbo log-T scale across all T panels,
shared symmetric RdBu_r scale across all ratio panels.

Reads the cached TemperatureSlicesTask intermediates (T_DESPOTIC at 4 slices)
for both L_ext — no recompute.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

# Fixed log10-T colour range, matching the multi_field_slices T panels
# (100 K – 1e8 K) so the two figures are directly comparable.
T_LOG_VMIN, T_LOG_VMAX = 2.0, 8.0
# Shared PLAIN-ratio (T_DSP^0 / T_DSP^compare) colour range across the 0-vs-9
# and 0-vs-99 figures; 0.5–2.0 ⇔ ±0.3 dex, centered (white) at 1.0.
RATIO_VMIN, RATIO_VCENTER, RATIO_VMAX = 0.5, 1.0, 2.0

ROOT = Path("/Users/baochen/quokka_postprocessing/output")
SRC = {
    0: ROOT / "plt0655228_down1_Lext0kpc/task_intermediates/TemperatureSlicesTask_c6efc720.h5",
    9: ROOT / "plt0655228_down1_Lext9kpc/task_intermediates/TemperatureSlicesTask_dab74771.h5",
}
RATIO_CLIP_DEX = 1.0
SLICE_AXIS = 'x'


def load(L):
    with h5py.File(SRC[L], "r") as f:
        return f["T_md_slices"][:], f["slice_indices"][:], f["extent"][:]


T0, sidx, extent = load(0)
T9, sidx9, _ = load(9)
assert np.array_equal(sidx, sidx9)
n_slices = T0.shape[0]
ext_plot = [extent[0], extent[1], extent[2], extent[3]]

# Pass A: build log panels + gather ranges (mirror temperature_slices).
t_panels, r_panels = [], []
for i in range(n_slices):
    a0 = T0[i].T   # vertical = long axis
    a9 = T9[i].T
    for arr in (a0, a9):
        pos = arr > 0
        log = np.where(pos, np.log10(np.where(pos, arr, 1.0)), np.nan)
        t_panels.append({'log': log, 'lo': np.nanpercentile(log, 0.5),
                         'hi': np.nanpercentile(log, 99.5)})
    both = (a0 > 0) & (a9 > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(both, a0 / np.where(both, a9, 1.0), np.nan)
    r_panels.append({'log': ratio})

log_vmin_T, log_vmax_T = T_LOG_VMIN, T_LOG_VMAX

# Interleave: [T@0, T@9, ratio] per slice.
plot_order = []
for i in range(n_slices):
    plot_order.append(('T', t_panels[2 * i],     r'$\log_{10}T_{\rm DSP}$  $L_{\rm ext}=0$',      int(sidx[i])))
    plot_order.append(('T', t_panels[2 * i + 1], r'$\log_{10}T_{\rm DSP}$  $L_{\rm ext}=9$ kpc',  int(sidx[i])))
    plot_order.append(('R', r_panels[i],         r'$T_{\rm DSP}^{0}/T_{\rm DSP}^{9}$', int(sidx[i])))

n_panels = len(plot_order)
fig, axes = plt.subplots(1, n_panels, figsize=(2.4 * n_panels, 12), sharey=True,
                         gridspec_kw={'wspace': 0.08, 'top': 0.86, 'bottom': 0.06})

for ax, (kind, st, label, sidx_i) in zip(axes, plot_order):
    if kind == 'T':
        im = ax.imshow(st['log'], origin='lower', extent=ext_plot, aspect='auto',
                       cmap='turbo', norm=Normalize(log_vmin_T, log_vmax_T))
    else:
        im = ax.imshow(st['log'], origin='lower', extent=ext_plot, aspect='auto',
                       cmap='RdBu_r', norm=TwoSlopeNorm(RATIO_VCENTER, vmin=RATIO_VMIN, vmax=RATIO_VMAX))
    ax.tick_params(axis='both', labelsize=8)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='2.5%', pad=0.55)
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=7, top=True, bottom=False, labeltop=True, labelbottom=False)
    # Inset ticks so adjacent colourbars' edge labels don't collide.
    if kind == 'T':
        cbar.set_ticks([2, 4, 6, 8])              # full range 100 K – 1e8 K (matches multi_field)
    else:
        cbar.set_ticks([0.7, 1.0, 1.5])           # plain ratio; inset from the 0.5/2.0 edges
    cax.set_title(f'{label}\n{SLICE_AXIS} idx = {sidx_i}', fontsize=8, pad=4)

plane = {'x': ('y', 'z'), 'y': ('x', 'z'), 'z': ('x', 'y')}[SLICE_AXIS]
axes[0].set_ylabel(f'{plane[1]} [kpc]', fontsize=10)
for ax in axes:
    ax.set_xlabel(f'{plane[0]} [kpc]', fontsize=9)

OUT = ROOT / "plt0655228_down1_LextDiff_0kpc_vs_9kpc"
OUT.mkdir(parents=True, exist_ok=True)
out = OUT / "T_DESPOTIC_lext_slices_0_vs_9.png"
fig.savefig(str(out), dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}  ({n_slices} slices, {n_panels} panels)")
for i in range(n_slices):
    fin = r_panels[i]['log'][np.isfinite(r_panels[i]['log'])]
    print(f"  idx {int(sidx[i])}: T0/T9 median={np.median(fin):.4f}  max={np.max(fin):.3f}")
