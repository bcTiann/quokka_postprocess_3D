"""Is the T_DESPOTIC change with L_ext numerical noise, or physical?

Test 1 (noise floor): L0 vs L0 itself  → exactly 0 (the pipeline is
        deterministic; column_density_H + table interp have no randomness).
Test 2 (scaling):      L0 vs L9  vs  L0 vs L99.  If the difference GROWS with
        L_ext and keeps the same sign (higher L_ext → cooler), it scales with a
        physical parameter and cannot be random numerical noise.

Also produces the L0-vs-L99 multi-slice figure in the TemperatureSlicesTask
layout: per slice [T_DSP@L0, T_DSP@L99, log10(T@0/T@99)].
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

# Fixed log10-T colour range, matching multi_field_slices T panels (100 K – 1e8 K).
T_LOG_VMIN, T_LOG_VMAX = 2.0, 8.0
# Shared PLAIN-ratio (T_DSP^0 / T_DSP^compare) range with the 0-vs-9 figure;
# 0.5–2.0 ⇔ ±0.3 dex, centered (white) at 1.0.
RATIO_VMIN, RATIO_VCENTER, RATIO_VMAX = 0.5, 1.0, 2.0

ROOT = Path("/Users/baochen/quokka_postprocessing/output")
SRC = {
    0:  ROOT / "plt0655228_down1_Lext0kpc/task_intermediates/TemperatureSlicesTask_c6efc720.h5",
    9:  ROOT / "plt0655228_down1_Lext9kpc/task_intermediates/TemperatureSlicesTask_dab74771.h5",
    99: ROOT / "plt0655228_down1_Lext99kpc/task_intermediates/TemperatureSlicesTask_dab74771.h5",
}


def load(L):
    with h5py.File(SRC[L], "r") as f:
        return f["T_md_slices"][:], f["slice_indices"][:], f["extent"][:]


T = {L: load(L)[0] for L in SRC}
_, sidx, extent = load(0)
ext_plot = [extent[0], extent[1], extent[2], extent[3]]
n_slices = T[0].shape[0]

# ── Test: noise floor + scaling ──────────────────────────────────────────────
def stats(Tb, Tc):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log10(Tb) - np.log10(Tc)
    f = r[np.isfinite(r)]
    return np.median(f), np.max(np.abs(f)), np.mean(np.abs(f) > 0.005), np.mean(f < -1e-9)

print(f"{'comparison':16s} {'median':>9s} {'max|Δ|':>8s} {'frac>0.005':>11s} {'frac(Tc>Tb)':>12s}")
for label, base, comp in [("L0 vs L0", 0, 0), ("L0 vs L9", 0, 9), ("L0 vs L99", 0, 99)]:
    m, mx, fr, fneg = stats(T[base], T[comp])
    print(f"{label:16s} {m:+9.4f} {mx:8.4f} {fr:11.3f} {fneg:12.3f}")
print("\n(median/max are log10(T_base/T_comp) over all 4 slices; "
      "frac(Tc>Tb) = fraction where the higher-L_ext run is HOTTER)")

# ── L0-vs-L99 figure (temperature_slices layout) ─────────────────────────────
COMP = 99
T0, T9 = T[0], T[COMP]
t_panels, r_panels = [], []
for i in range(n_slices):
    a0, ac = T0[i].T, T9[i].T
    for arr in (a0, ac):
        pos = arr > 0
        log = np.where(pos, np.log10(np.where(pos, arr, 1.0)), np.nan)
        t_panels.append({"log": log, "lo": np.nanpercentile(log, 0.5), "hi": np.nanpercentile(log, 99.5)})
    both = (a0 > 0) & (ac > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(both, a0 / np.where(both, ac, 1.0), np.nan)
    r_panels.append({"log": ratio})

log_vmin_T, log_vmax_T = T_LOG_VMIN, T_LOG_VMAX

plot_order = []
for i in range(n_slices):
    plot_order.append(("T", t_panels[2 * i],     r"$\log_{10}T_{\rm DSP}$  $L_{\rm ext}=0$",       int(sidx[i])))
    plot_order.append(("T", t_panels[2 * i + 1], r"$\log_{10}T_{\rm DSP}$  $L_{\rm ext}=99$ kpc",  int(sidx[i])))
    plot_order.append(("R", r_panels[i],         r"$T_{\rm DSP}^{0}/T_{\rm DSP}^{99}$", int(sidx[i])))

n_panels = len(plot_order)
fig, axes = plt.subplots(1, n_panels, figsize=(2.4 * n_panels, 12), sharey=True,
                         gridspec_kw={"wspace": 0.08, "top": 0.86, "bottom": 0.06})
for ax, (kind, st, label, sidx_i) in zip(axes, plot_order):
    cmap, norm = (("turbo", Normalize(log_vmin_T, log_vmax_T)) if kind == "T"
                  else ("RdBu_r", TwoSlopeNorm(RATIO_VCENTER, vmin=RATIO_VMIN, vmax=RATIO_VMAX)))
    im = ax.imshow(st["log"], origin="lower", extent=ext_plot, aspect="auto", cmap=cmap, norm=norm)
    ax.tick_params(axis="both", labelsize=8)
    cax = make_axes_locatable(ax).append_axes("top", size="2.5%", pad=0.55)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7, top=True, bottom=False, labeltop=True, labelbottom=False)
    if kind == "T":
        cb.set_ticks([2, 4, 6, 8])                # full range 100 K – 1e8 K (matches multi_field)
    else:
        cb.set_ticks([0.7, 1.0, 1.5])             # plain ratio; inset from the 0.5/2.0 edges
    cax.set_title(f"{label}\nx idx = {sidx_i}", fontsize=8, pad=4)

axes[0].set_ylabel("z [kpc]", fontsize=10)
for ax in axes:
    ax.set_xlabel("y [kpc]", fontsize=9)

OUT = ROOT / "plt0655228_down1_LextDiff_0kpc_vs_9kpc"
out = OUT / "T_DESPOTIC_lext_slices_0_vs_99.png"
fig.savefig(str(out), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")
