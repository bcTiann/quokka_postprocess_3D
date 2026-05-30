#!/usr/bin/env python
"""GOW table v2 (buggy missing-emitter) vs v4 (LVG-consistent + emitter fix)
side-by-side comparison at smallest dVdr.

Top row: log10(T) for v2 (left), v4 (right), and their ratio (log10(v2/v4)).
Shows where the emitter+LVG bug was tilting the table the most.
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

ROOT = Path('/Users/baochen/quokka_postprocessing')
v2 = np.load(ROOT / 'output_tables_3D_GOW_LVG_v2_buggy_emitter/despotic_table.npz', allow_pickle=True)
v4 = np.load(ROOT / 'output_tables_3D_GOW_LVG/despotic_table.npz', allow_pickle=True)

assert np.array_equal(v2['nH_values'], v4['nH_values'])
nH = v2['nH_values']; NH = v2['col_density_values']; dV = v2['dVdr_values']
log_nH = np.log10(nH); log_NH = np.log10(NH)

# Use dVdr index 0 (smallest) where LVG effect is largest, and median (17)
for d_idx, tag in [(0, 'dVdr_min'), (17, 'dVdr_median'), (34, 'dVdr_max')]:
    T2 = v2['tg_final'][:, :, d_idx]
    T4 = v4['tg_final'][:, :, d_idx]
    log_T2 = np.log10(np.where(T2 > 0, T2, np.nan))
    log_T4 = np.log10(np.where(T4 > 0, T4, np.nan))
    log_ratio = log_T2 - log_T4    # = log10(v2 / v4)

    # Shared log10(T) range for the two T panels
    both = np.concatenate([log_T2[np.isfinite(log_T2)].ravel(),
                           log_T4[np.isfinite(log_T4)].ravel()])
    tnorm = Normalize(both.min(), both.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                              gridspec_kw={'wspace': 0.3})

    im0 = axes[0].pcolormesh(log_NH, log_nH, log_T2, cmap='turbo', norm=tnorm, shading='auto')
    axes[0].set_title(f'v2 (buggy)\nlog$_{{10}}$ T$_g$  @  dVdr = {dV[d_idx]:.1e}', fontsize=10)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].pcolormesh(log_NH, log_nH, log_T4, cmap='turbo', norm=tnorm, shading='auto')
    axes[1].set_title(f'v4 (LVG + emitter fix)\nlog$_{{10}}$ T$_g$  (same colorbar)', fontsize=10)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Ratio panel — diverging colormap centred at 0 dex
    rmax = max(abs(np.nanmin(log_ratio)), abs(np.nanmax(log_ratio)), 0.1)
    rnorm = TwoSlopeNorm(0.0, vmin=-rmax, vmax=rmax)
    im2 = axes[2].pcolormesh(log_NH, log_nH, log_ratio, cmap='RdBu_r', norm=rnorm, shading='auto')
    axes[2].set_title(f'log$_{{10}}$(v2 / v4)\nred = bug made T$_g$ too HOT', fontsize=10)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Contour at log10(v2/v4) = 1 (10×) to mark dramatic regions
    cs = axes[2].contour(log_NH, log_nH, log_ratio, levels=[1.0, 2.0],
                          colors='black', linewidths=1.0, linestyles=['--', '-'])
    axes[2].clabel(cs, fmt={1.0: '10×', 2.0: '100×'}, fontsize=8)

    for ax in axes:
        ax.set_xlabel(r'log$_{10}$ N$_H$ [cm$^{-2}$]', fontsize=9)
    axes[0].set_ylabel(r'log$_{10}$ n$_H$ [cm$^{-3}$]', fontsize=9)

    fig.suptitle(f'GOW table T$_g$: pre-fix vs post-fix @ {tag}', fontsize=11, y=1.02)

    out = ROOT / 'output' / 'three_network_compare' / f'GOW_v2_vs_v4_compare_{tag}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[out] {out}")

# Stats summary
T2 = v2['tg_final']; T4 = v4['tg_final']
finite = np.isfinite(T2) & np.isfinite(T4) & (T2 > 0) & (T4 > 0)
ratio = np.where(finite, np.log10(T2/T4), np.nan)
print("\n=== log10(v2 / v4) statistics ===")
print(f"  mean    = {np.nanmean(ratio):+.3f} dex")
print(f"  median  = {np.nanmedian(ratio):+.3f} dex")
print(f"  p10     = {np.nanpercentile(ratio, 10):+.3f}")
print(f"  p90     = {np.nanpercentile(ratio, 90):+.3f}")
print(f"  max     = {np.nanmax(ratio):+.3f}")
print(f"  fraction abs > 0.5 dex (>3.2×): {(np.abs(ratio) > 0.5).mean()*100:.1f}%")
print(f"  fraction abs > 1.0 dex (>10×):  {(np.abs(ratio) > 1.0).mean()*100:.1f}%")
print(f"  fraction abs > 2.0 dex (>100×): {(np.abs(ratio) > 2.0).mean()*100:.1f}%")
