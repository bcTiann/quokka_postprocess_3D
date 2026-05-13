"""验证 DESPOTIC 表格中 T × Eint(T) 的单调性。

物理保证：T × Eint = E_g / (n_H k_B)，即单位 H 核热内能 / k_B。
热容 Cv > 0 保证物理上严格单调递增。
此脚本验证数值插值是否引入了局部抖动。
"""

import sys
sys.path.insert(0, "/Users/baochen/quokka_postprocessing/quokka2s/src")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from quokka2s.tables import load_table

TABLE_PATH = "/Users/baochen/quokka_postprocessing/output_tables_3D_GOW_T2e7/despotic_table.npz"
OUT_DIR    = "/Users/baochen/quokka_postprocessing"

# ── 1. 加载表格 ──────────────────────────────────────────────────────────────
print("Loading table...")
table = load_table(TABLE_PATH)

T_vals   = table.T_values            # (n_T,)
nH_vals  = table.nH_values           # (n_nH,)
col_vals = table.col_density_values  # (n_col,)
Eint     = table.Eint_values         # (n_nH, n_col, n_T)

print(f"Table shape (nH, col, T): {Eint.shape}")
print(f"T range:   {T_vals.min():.1f} – {T_vals.max():.1f} K  ({len(T_vals)} points)")
print(f"nH range:  {nH_vals.min():.2e} – {nH_vals.max():.2e} cm⁻³  ({len(nH_vals)} points)")
print(f"col range: {col_vals.min():.2e} – {col_vals.max():.2e} cm⁻²  ({len(col_vals)} points)")

# ── 2. 计算 T × Eint ─────────────────────────────────────────────────────────
T_Eint = T_vals[None, None, :] * Eint   # (n_nH, n_col, n_T)

# NaN 统计
n_nan = np.sum(np.isnan(Eint))
print(f"\nNaN in Eint: {n_nan} / {Eint.size}  ({100*n_nan/Eint.size:.2f}%)")

# ── 3. 单调性检验（沿 T 轴差分） ───────────────────────────────────────────
diff = np.diff(T_Eint, axis=2)   # (n_nH, n_col, n_T-1)

# 忽略含 NaN 的点
valid_mask = np.isfinite(diff)
n_valid    = valid_mask.sum()
n_bad      = np.sum((diff <= 0) & valid_mask)

print(f"\n── 单调性检验 ──")
print(f"有效差分点: {n_valid}")
print(f"非单调点  : {n_bad}  ({100*n_bad/n_valid:.4f}%)")

if n_bad == 0:
    print("✅ T × Eint 在所有有效格点上严格单调递增，二分法唯一解有保证。")
else:
    print("⚠️  存在非单调点，以下列出最严重的前 10 个：")
    bad_idx = np.argwhere((diff <= 0) & valid_mask)
    # 按违反程度（diff 值最小）排序
    bad_vals = diff[(diff <= 0) & valid_mask]
    order = np.argsort(bad_vals)
    bad_idx_sorted = bad_idx[order[:10]]
    print(f"  {'i_nH':>5} {'i_col':>6} {'i_T':>5}  nH[cm⁻³]    col[cm⁻²]   T[K]→T+1[K]  diff")
    for idx in bad_idx_sorted:
        i, j, k = idx
        print(f"  {i:>5} {j:>6} {k:>5}  "
              f"{nH_vals[i]:.2e}  {col_vals[j]:.2e}  "
              f"{T_vals[k]:.1f}→{T_vals[k+1]:.1f}  "
              f"{diff[i,j,k]:.3e}")

# ── 4. Eint 的基本统计 ────────────────────────────────────────────────────
finite_eint = Eint[np.isfinite(Eint)]
print(f"\n── Eint 统计 ──")
print(f"min:  {finite_eint.min():.4f}")
print(f"max:  {finite_eint.max():.4f}")
print(f"mean: {finite_eint.mean():.4f}")
print(f"p1:   {np.percentile(finite_eint, 1):.4f}")
print(f"p99:  {np.percentile(finite_eint, 99):.4f}")

# ── 5. 可视化 ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

col_mid = len(col_vals) // 2
nH_mid  = len(nH_vals)  // 2

# (a) T × Eint 曲线：不同 nH，固定 col 中间值
ax = axes[0, 0]
for i in np.linspace(0, len(nH_vals)-1, 6, dtype=int):
    y = T_Eint[i, col_mid, :]
    mask = np.isfinite(y)
    ax.plot(T_vals[mask], y[mask], label=f'nH={nH_vals[i]:.1e}', lw=1.5)
ax.set_xscale('log')
ax.set_xlabel('T [K]')
ax.set_ylabel('T × Eint  [K]')
ax.set_title(f'T×Eint vs T\n(colDen={col_vals[col_mid]:.1e} cm⁻²)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# (b) Eint 本身随 T 的变化
ax = axes[0, 1]
for i in np.linspace(0, len(nH_vals)-1, 6, dtype=int):
    y = Eint[i, col_mid, :]
    mask = np.isfinite(y)
    ax.plot(T_vals[mask], y[mask], label=f'nH={nH_vals[i]:.1e}', lw=1.5)
ax.set_xscale('log')
ax.set_xlabel('T [K]')
ax.set_ylabel('Eint_dimless')
ax.set_title(f'Eint_dimless vs T\n(colDen={col_vals[col_mid]:.1e} cm⁻²)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# (c) diff(T×Eint) 热图：nH × T，固定 col 中间值
ax = axes[1, 0]
d_slice = diff[:, col_mid, :]     # (n_nH, n_T-1)
vmax = np.nanpercentile(np.abs(d_slice[np.isfinite(d_slice)]), 99)
im = ax.imshow(d_slice, aspect='auto', origin='lower',
               norm=mcolors.SymLogNorm(linthresh=1e-3, vmin=-vmax, vmax=vmax),
               cmap='RdBu', extent=[0, len(T_vals)-1, 0, len(nH_vals)-1])
ax.set_xlabel('T index')
ax.set_ylabel('nH index')
ax.set_title(f'diff(T×Eint) 沿 T 轴\n蓝=正(单调) 红=负(违反)')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# (d) T × Eint 曲线：不同 col，固定 nH 中间值
ax = axes[1, 1]
for j in np.linspace(0, len(col_vals)-1, 6, dtype=int):
    y = T_Eint[nH_mid, j, :]
    mask = np.isfinite(y)
    ax.plot(T_vals[mask], y[mask], label=f'col={col_vals[j]:.1e}', lw=1.5)
ax.set_xscale('log')
ax.set_xlabel('T [K]')
ax.set_ylabel('T × Eint  [K]')
ax.set_title(f'T×Eint vs T（不同 colDen）\n(nH={nH_vals[nH_mid]:.1e} cm⁻³)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.suptitle('DESPOTIC Eint 单调性验证', fontsize=14)
plt.tight_layout()
out = f"{OUT_DIR}/eint_monotonicity_check.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")
