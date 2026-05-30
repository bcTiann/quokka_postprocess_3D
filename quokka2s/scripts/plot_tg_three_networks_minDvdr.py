#!/usr/bin/env python
"""Side-by-side T_g (nH, NH) at smallest dVdr for GOW vs NL99 vs NL99_GC.

Shared log10(T) colorbar across the three panels; black contour at T=200 K
to call out the cold regime.  Uses the canonical (post-computeDerived fix)
3D LVG tables.
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = Path('/Users/baochen/quokka_postprocessing')
TABLES = {
    'GOW':     ROOT / 'output_tables_3D_GOW_LVG/despotic_table.npz',
    'NL99':    ROOT / 'output_tables_3D_NL99_LVG/despotic_table.npz',
    'NL99_GC': ROOT / 'output_tables_3D_NL99_GC_LVG/despotic_table.npz',
}

# Load all three at smallest dVdr.
panels = {}
for name, path in TABLES.items():
    d = np.load(path, allow_pickle=True)
    nH  = d['nH_values']
    NH  = d['col_density_values']
    dV  = d['dVdr_values']
    Tg  = d['tg_final'][:, :, 0]  # smallest dVdr index
    panels[name] = (nH, NH, dV[0], Tg)
    print(f"  {name:8s}: dVdr_min = {dV[0]:.2e} s^-1, "
          f"T range = [{np.nanmin(Tg):.2g}, {np.nanmax(Tg):.2g}] K")

# Shared log10(T) colorbar from full data range across all 3 panels.
all_T = np.concatenate([panels[k][3][np.isfinite(panels[k][3]) & (panels[k][3] > 0)].ravel()
                        for k in TABLES])
vmin, vmax = np.log10(all_T.min()), np.log10(all_T.max())
norm = Normalize(vmin, vmax)
cmap = 'turbo'

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True,
                         gridspec_kw={'wspace': 0.05})
for ax, (name, (nH, NH, dV0, Tg)) in zip(axes, panels.items()):
    with np.errstate(all='ignore'):
        d = np.log10(np.where(Tg > 0, Tg, np.nan))
    # imshow with (NH on x, nH on y); pcolormesh would be more correct for
    # log axes, use that with edge coordinates.
    log_nH = np.log10(nH)
    log_NH = np.log10(NH)
    im = ax.pcolormesh(log_NH, log_nH, d, cmap=cmap, norm=norm, shading='auto')
    # Contours at T = 50, 100, 200 K (matplotlib requires ascending levels).
    contour_levels_K = [50.0, 100.0, 200.0]
    contour_levels = [np.log10(T) for T in contour_levels_K]
    cs = ax.contour(log_NH, log_nH, d, levels=contour_levels,
                    colors='black', linewidths=1.0,
                    linestyles=[':', '--', '-'])
    fmt = {lv: f'{T:g} K' for lv, T in zip(contour_levels, contour_levels_K)}
    ax.clabel(cs, fmt=fmt, fontsize=8, inline=True)
    ax.set_title(f'{name}  (dVdr = {dV0:.1e} s$^{{-1}}$)', fontsize=11)
    ax.set_xlabel(r'$\log_{10}\,N_{\rm H}$  [cm$^{-2}$]', fontsize=10)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel(r'$\log_{10}\,n_{\rm H}$  [cm$^{-3}$]', fontsize=10)

cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label(r'$\log_{10}\,T_g$  [K]', fontsize=10)
cbar.ax.tick_params(labelsize=8)

fig.suptitle('DESPOTIC equilibrium $T_g$ at smallest dVdr — three networks '
             '(shared colorbar; contours: solid 200 K, dashed 100 K, dotted 50 K)',
             fontsize=11, y=1.02)

OUT = ROOT / 'output' / 'three_network_compare'
OUT.mkdir(parents=True, exist_ok=True)
out_path = OUT / 'Tg_three_networks_minDvdr.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"\n[out] {out_path}")

# Also report what fraction of cells are below 200 K in each panel.
print("\nfraction of cells with T < 200 K:")
for name, (_, _, _, Tg) in panels.items():
    m = np.isfinite(Tg) & (Tg > 0)
    frac = (Tg[m] < 200).mean() if m.any() else float('nan')
    print(f"  {name:8s}: {frac*100:.1f}%   (of {m.sum()}/{Tg.size} valid cells)")
