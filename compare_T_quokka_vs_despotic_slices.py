"""yz-slice comparison: T_QUOKKA (raw boxlib) vs T_DESPOTIC (lookup tg_final).

Run from repo root:
    cd /Users/baochen/quokka_postprocessing
    python compare_T_quokka_vs_despotic_slices.py

Output: ./T_quokka_vs_despotic_yz_slices.png
"""
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "quokka2s/src")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import yt
yt.set_log_level("error")

from quokka2s.pipeline.prep import physics_fields as pf, config as cfg


# ------------------------------------------------------------------
# Load dataset, register derived fields
# ------------------------------------------------------------------
print(f"Loading {cfg.YT_DATASET_PATH} ...")
ds = yt.load(cfg.YT_DATASET_PATH)
ds.force_periodicity()
pf.add_all_fields(ds)

print(f"Reading covering grid {tuple(ds.domain_dimensions)} ...")
grid = ds.covering_grid(level=0,
                        left_edge=ds.domain_left_edge,
                        dims=ds.domain_dimensions)

print("Reading temperature fields (lazy: triggers lookup chain) ...")
T_q = grid[('gas', 'temperature_quokka')].to('K').value      # (128, 128, 512)
T_d = grid[('gas', 'temperature_despotic')].to('K').value


# ------------------------------------------------------------------
# Pick x slices and compute shared color scale
# ------------------------------------------------------------------
X_INDICES = [16, 48, 80, 112]
dx_pc = float((ds.domain_right_edge[0] - ds.domain_left_edge[0])
              .in_units('pc').v) / ds.domain_dimensions[0]
x_pc_values = [round(idx * dx_pc, 1) for idx in X_INDICES]

# Joint min/max for shared LogNorm across both T fields.
T_min_q = T_q[T_q > 0].min() if (T_q > 0).any() else 1.0
T_min_d = T_d[T_d > 0].min() if (T_d > 0).any() else 1.0
T_min = max(min(float(T_min_q), float(T_min_d)), 1.0)
T_max = float(max(T_q.max(), T_d.max()))
print(f"\nGlobal T statistics:")
print(f"  T_QUOKKA  : min={T_q.min():.2e}, max={T_q.max():.2e}, median={np.median(T_q):.2e}")
print(f"  T_DESPOTIC: min={T_d.min():.2e}, max={T_d.max():.2e}, median={np.median(T_d):.2e}")
print(f"  Color range used: [{T_min:.1f}, {T_max:.2e}] K")


# ------------------------------------------------------------------
# Figure: 4 rows (x slices) x 3 cols (T_Q | T_D | log ratio).
# Panels are oriented z-vertical (disk-vertical axis up), y-horizontal.
# ------------------------------------------------------------------
extent = [0, 978, -1956, 1956]   # y_min, y_max, z_min, z_max  (pc)

fig, axes = plt.subplots(4, 3, figsize=(9, 16), constrained_layout=True)

for row, (xi, x_label) in enumerate(zip(X_INDICES, x_pc_values)):
    # Transpose so z (long axis) goes vertically in the panel.
    Tq_slice = T_q[xi, :, :].T                            # (nz=512, ny=128)
    Td_slice = T_d[xi, :, :].T
    ratio = np.log10(np.maximum(Td_slice, 1.0) /
                     np.maximum(Tq_slice, 1.0))

    im_q = axes[row, 0].imshow(Tq_slice, origin='lower', extent=extent,
                               norm=LogNorm(vmin=T_min, vmax=T_max),
                               cmap=cfg.CMAP, aspect='equal')
    im_d = axes[row, 1].imshow(Td_slice, origin='lower', extent=extent,
                               norm=LogNorm(vmin=T_min, vmax=T_max),
                               cmap=cfg.CMAP, aspect='equal')
    im_r = axes[row, 2].imshow(ratio, origin='lower', extent=extent,
                               cmap='RdBu_r', vmin=-1, vmax=1,
                               aspect='equal')

    axes[row, 0].set_ylabel(f"x = {x_label} pc  (idx {xi})\nz [pc]")
    for ax in axes[row]:
        if row == 3:
            ax.set_xlabel("y [pc]")

axes[0, 0].set_title("T_QUOKKA  (raw boxlib)", fontsize=11)
axes[0, 1].set_title("T_DESPOTIC  (LVG self-consistent)", fontsize=11)
axes[0, 2].set_title(r"$\log_{10}(T_{\rm DESPOTIC}/T_{\rm QUOKKA})$", fontsize=11)

fig.colorbar(im_d, ax=axes[:, 0:2].ravel().tolist(), label="T [K]",
             shrink=0.6, pad=0.02)
fig.colorbar(im_r, ax=axes[:, 2].tolist(), label=r"$\log_{10}$ ratio",
             shrink=0.6, pad=0.02)

fig.suptitle(
    "yz-slice comparison: T_QUOKKA vs T_DESPOTIC\n"
    f"dataset = {cfg.YT_DATASET_PATH.split('/')[-1]}, "
    f"table = {cfg.DESPOTIC_TABLE_PATH.split('/')[-2]}",
    fontsize=13,
)

out_path = "T_quokka_vs_despotic_yz_slices.png"
fig.savefig(out_path, dpi=100, bbox_inches='tight')
print(f"\nSaved: {out_path}")
