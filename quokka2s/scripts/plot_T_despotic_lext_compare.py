"""T_DESPOTIC slice comparison: L_ext = 0 vs 9 kpc (new data, down=1).

Answers the advisor's question "why is the temperature not changing when you
set L_ext = 9 kpc?".  Reads the two runs' CACHED TemperatureSlicesTask
intermediates (no recompute) and plots, per slice:

    row 0:  log10 T_DESPOTIC  @ L_ext = 0 kpc
    row 1:  log10 T_DESPOTIC  @ L_ext = 9 kpc
    row 2:  Δ = log10( T_DSP@9 / T_DSP@0 )      (diverging, ~0 everywhere)

Run:  /opt/homebrew/Caskroom/miniconda/base/envs/yt-env/bin/python \
        quokka2s/scripts/plot_T_despotic_lext_compare.py
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

ROOT = Path("/Users/baochen/quokka_postprocessing/output")
SRC = {
    0: ROOT / "plt0655228_down1_Lext0kpc/task_intermediates/TemperatureSlicesTask_c6efc720.h5",
    9: ROOT / "plt0655228_down1_Lext9kpc/task_intermediates/TemperatureSlicesTask_dab74771.h5",
}
OUT = ROOT / "plt0655228_down1_LextDiff_0kpc_vs_9kpc"
OUT.mkdir(parents=True, exist_ok=True)


def load(L):
    with h5py.File(SRC[L], "r") as f:
        return f["T_md_slices"][:], f["slice_indices"][:], f["extent"][:]


Tmd0, sidx, extent = load(0)
Tmd9, sidx9, _ = load(9)
assert np.array_equal(sidx, sidx9), "slice index mismatch"
n = Tmd0.shape[0]
ext = [extent[0], extent[1], extent[2], extent[3]]  # [y0,y1,z0,z1] kpc

# Shared log10-T colour range across both L_ext (rows 0,1).
both = np.concatenate([Tmd0[np.isfinite(Tmd0) & (Tmd0 > 0)],
                       Tmd9[np.isfinite(Tmd9) & (Tmd9 > 0)]])
tnorm = mcolors.Normalize(vmin=np.log10(np.percentile(both, 0.5)),
                          vmax=np.log10(np.percentile(both, 99.5)))

# Symmetric diff range.
with np.errstate(all="ignore"):
    diffs = [np.log10(Tmd9[i]) - np.log10(Tmd0[i]) for i in range(n)]
allfin = np.concatenate([d[np.isfinite(d)] for d in diffs])
dlim = max(min(np.percentile(np.abs(allfin), 99.5), 0.2), 0.02)

fig, axes = plt.subplots(3, n, figsize=(2.6 * n, 13),
                         gridspec_kw={"wspace": 0.06, "hspace": 0.12})

def show(ax, arr, norm, cmap, log=True):
    a = np.log10(arr) if log else arr
    return ax.imshow(a.T, origin="lower", extent=ext, aspect="auto",
                     cmap=cmap, norm=norm)

im_t = im_d = None
for j in range(n):
    im_t = show(axes[0, j], Tmd0[j], tnorm, "turbo")
    show(axes[1, j], Tmd9[j], tnorm, "turbo")
    im_d = axes[2, j].imshow(diffs[j].T, origin="lower", extent=ext, aspect="auto",
                             cmap="RdBu_r", norm=mcolors.Normalize(-dlim, dlim))
    med = np.nanmedian(diffs[j][np.isfinite(diffs[j])])
    axes[0, j].set_title(f"slice x-idx {int(sidx[j])}", fontsize=10)
    axes[2, j].set_title(f"median Δ = {med:+.4f} dex", fontsize=8)
    for r in range(3):
        axes[r, j].tick_params(labelsize=7)
        if j: axes[r, j].set_yticklabels([])

for r, lab in enumerate([r"$\log_{10}T_{\rm DSP}$  $L_{\rm ext}=0$",
                         r"$\log_{10}T_{\rm DSP}$  $L_{\rm ext}=9$ kpc",
                         r"$\Delta=\log_{10}(T_{\rm DSP}^{9}/T_{\rm DSP}^{0})$"]):
    axes[r, 0].set_ylabel(f"z [kpc]\n{lab}", fontsize=9)
for j in range(n):
    axes[2, j].set_xlabel("y [kpc]", fontsize=8)

fig.colorbar(im_t, ax=axes[0:2, :].ravel().tolist(), location="right",
             fraction=0.015, pad=0.01, label=r"$\log_{10}T_{\rm DESPOTIC}$ [K]")
fig.colorbar(im_d, ax=axes[2, :].ravel().tolist(), location="right",
             fraction=0.015, pad=0.01, label=r"$\Delta\log_{10}T$ [dex]")

fig.suptitle("T_DESPOTIC slices: L_ext = 0 vs 9 kpc  (plt0655228, down=1)\n"
             "Row 3 ≈ 0 everywhere → DESPOTIC equilibrium T is insensitive to the "
             "lateral column extension", fontsize=12)
out = OUT / "temperature_lext_compare.png"
fig.savefig(str(out), dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# Quantitative summary.
print("\nPer-slice T_DESPOTIC change (L9 vs L0):")
for i in range(n):
    d = diffs[i][np.isfinite(diffs[i])]
    print(f"  x-idx {int(sidx[i])}: median={np.median(d):+.4f}  "
          f"p1={np.percentile(d,1):+.3f}  p99={np.percentile(d,99):+.3f}  "
          f"max|Δ|={np.max(np.abs(d)):.3f} dex")
