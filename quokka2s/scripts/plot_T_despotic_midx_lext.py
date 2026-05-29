"""T_DESPOTIC midplane-x slice: L_ext = 0 vs 9 kpc, with log ratio.

Reads the freshly-regenerated MultiFieldSlicesTask caches (slice_idx=None →
mid-x slice) for both L_ext and plots three panels:

    1.  log10 T_DESPOTIC   L_ext = 0 kpc   (slice = mid x)
    2.  log10 T_DESPOTIC   L_ext = 9 kpc   (slice = mid x)
    3.  log10( T_DSP^{L=0} / T_DSP^{L=9} )      (diverging, ~0 everywhere)
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

ROOT = Path("/Users/baochen/quokka_postprocessing/output")
CACHE = "task_intermediates/MultiFieldSlicesTask_2e5afaff.h5"


def load(L):
    with h5py.File(ROOT / f"plt0655228_down1_Lext{L}kpc" / CACHE, "r") as f:
        return f["slices"]["T_dsp_slice"][:], f["extent_kpc"][:]


T0, ext = load(0)
T9, _ = load(9)
ext_plot = [ext[0], ext[1], ext[2], ext[3]]  # [y0,y1,z0,z1] kpc

both = np.concatenate([T0[np.isfinite(T0) & (T0 > 0)], T9[np.isfinite(T9) & (T9 > 0)]])
tnorm = mcolors.Normalize(vmin=np.log10(np.percentile(both, 0.5)),
                          vmax=np.log10(np.percentile(both, 99.5)))

with np.errstate(all="ignore"):
    ratio = np.log10(T0) - np.log10(T9)        # log10(T_DSP@0 / T_DSP@9)
fin = ratio[np.isfinite(ratio)]
dlim = max(min(np.percentile(np.abs(fin), 99.5), 0.3), 0.02)

fig, axes = plt.subplots(1, 3, figsize=(9, 12), sharey=True,
                         gridspec_kw={"wspace": 0.1})

im0 = axes[0].imshow(np.log10(T0).T, origin="lower", extent=ext_plot, aspect="auto",
                     cmap="turbo", norm=tnorm)
axes[0].set_title(r"$\log_{10}T_{\rm DSP}$,  $L_{\rm ext}=0$", fontsize=11)
axes[1].imshow(np.log10(T9).T, origin="lower", extent=ext_plot, aspect="auto",
               cmap="turbo", norm=tnorm)
axes[1].set_title(r"$\log_{10}T_{\rm DSP}$,  $L_{\rm ext}=9$ kpc", fontsize=11)
im2 = axes[2].imshow(ratio.T, origin="lower", extent=ext_plot, aspect="auto",
                     cmap="RdBu_r", norm=mcolors.Normalize(-dlim, dlim))
axes[2].set_title(r"$\log_{10}(T_{\rm DSP}^{0}/T_{\rm DSP}^{9})$", fontsize=11)

axes[0].set_ylabel("z [kpc]", fontsize=10)
for ax in axes:
    ax.set_xlabel("y [kpc]", fontsize=9)
    ax.tick_params(labelsize=8)

fig.colorbar(im0, ax=axes[:2], location="bottom", fraction=0.04, pad=0.06,
             label=r"$\log_{10}T_{\rm DESPOTIC}$ [K]")
fig.colorbar(im2, ax=axes[2], location="bottom", fraction=0.08, pad=0.06,
             label=r"$\Delta\log_{10}T$ [dex]")

med = np.nanmedian(fin)
fig.suptitle("T_DESPOTIC  mid-x slice  (plt0655228, down=1)\n"
             f"L_ext 0 vs 9 kpc — median log ratio = {med:+.4f} dex "
             "(≈0 → T insensitive to L_ext)", fontsize=12, y=0.98)

OUT = ROOT / "plt0655228_down1_LextDiff_0kpc_vs_9kpc"
OUT.mkdir(parents=True, exist_ok=True)
out = OUT / "T_DESPOTIC_midx_0_vs_9.png"
fig.savefig(str(out), dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

f = ratio[np.isfinite(ratio)]
print(f"log10(T@0/T@9): median={np.median(f):+.4f}  p1={np.percentile(f,1):+.3f}  "
      f"p99={np.percentile(f,99):+.3f}  max|Δ|={np.max(np.abs(f)):.3f} dex  "
      f"frac|Δ|>0.02={np.mean(np.abs(f)>0.02):.3f}")
