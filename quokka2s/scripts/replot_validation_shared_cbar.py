#!/usr/bin/env python
"""Re-draw the cold+dense DESPOTIC validation scatters for a set of L_ext
values with a SHARED N_H colour bar, so the three panels are directly
comparable by colour.

Reads the existing per-L_ext CSVs (no DESPOTIC re-solve) and overwrites each
PNG with the shared-range version.  Cheap (seconds).

Run:
    /opt/homebrew/Caskroom/miniconda/base/envs/yt-env/bin/python \
        quokka2s/scripts/replot_validation_shared_cbar.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Which runs to pool + redraw (must already have CSVs on disk).
DOWNSAMPLE = 1
L_EXTS     = [0.0, 9.0, 99.0]
LOG_RHO_MIN = -23.0       # only used to re-print the mask line in the title
LOG_TQK_MAX = 4.0

OUT_DIR = Path('/Users/baochen/quokka_postprocessing/output/despotic_validation')

# CSV columns (see validate_despotic_cold_dense.py):
#   0 flat_idx 1 log_rho 2 n_H 3 colDen 4 dVdr
#   5 T_QK 6 T_DSP_table 7 T_DSP_real 8 dex_table_over_real 9 dex_real_over_QK 10 failed
C_COLDEN, C_TQK, C_TTAB, C_TREAL, C_DEXTR, C_FAILED = 3, 5, 6, 7, 8, 10


def _csv(l_ext: float) -> Path:
    return OUT_DIR / f'despotic_validation_cold_dense_d{DOWNSAMPLE}_Lext{l_ext:g}kpc.csv'


def main():
    data = {}
    for L in L_EXTS:
        p = _csv(L)
        if not p.exists():
            print(f'[skip] missing {p.name}')
            continue
        data[L] = np.loadtxt(p, delimiter=',', skiprows=1)
    if not data:
        print('no CSVs found, nothing to do'); return
    Ls = list(data.keys())

    # Global N_H colour range pooled across all runs (shared colour bar).
    all_logN = np.concatenate([
        np.log10(a[:, C_COLDEN][a[:, C_COLDEN] > 0]) for a in data.values()])
    cnorm = Normalize(vmin=float(all_logN.min()), vmax=float(all_logN.max()))

    # global axis range
    lo, hi = np.inf, -np.inf
    for a in data.values():
        for col in (C_TQK, C_TTAB, C_TREAL):
            v = a[:, col][a[:, col] > 0]
            lo = min(lo, np.log10(v).min()); hi = max(hi, np.log10(v).max())
    lo, hi = np.floor(lo), np.ceil(hi)

    n = len(Ls)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.4), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    sc = None
    for ax, L in zip(axes, Ls):
        a = data[L]
        colDen, Tqk, Ttab, Treal = a[:, C_COLDEN], a[:, C_TQK], a[:, C_TTAB], a[:, C_TREAL]
        failed = a[:, C_FAILED].astype(bool)
        ok = (~failed & np.isfinite(Treal) & (Treal > 0)
              & (Tqk > 0) & (Ttab > 0) & (colDen > 0))
        lr = np.log10(Tqk[ok]); lt_tab = np.log10(Ttab[ok]); lt_real = np.log10(Treal[ok])
        cval = np.log10(colDen[ok])

        # table-vs-real interpolation error (the table's precision)
        dex = np.abs(a[:, C_DEXTR][ok])
        med_dex, max_dex = float(np.median(dex)), float(np.max(dex))

        ax.grid(True, ls=':', lw=0.5, alpha=0.35, zorder=0)
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, zorder=1,
                label=r'$T_{\rm DESPOTIC}=T_{\rm QUOKKA}$')
        sc = ax.scatter(lr, lt_real, c=cval, cmap='viridis', norm=cnorm,
                        marker='o', s=22, alpha=0.85, linewidth=0.0, zorder=3,
                        label=r'$T_{\rm DESPOTIC}$ (real-time)')
        ax.scatter(lr, lt_tab, facecolors='none', edgecolors='k',
                   marker='o', s=46, linewidth=0.6, alpha=0.6, zorder=2,
                   label=r'$T_{\rm DESPOTIC}$ (table)')
        # interpolation-error annotation (table vs real-time)
        ax.text(0.04, 0.96,
                'table vs real-time:\n'
                f'median {med_dex:.3f} dex\nmax {max_dex:.3f} dex',
                transform=ax.transAxes, va='top', ha='left', fontsize=8,
                bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.9))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel(r'$\log_{10}\,T_{\rm QUOKKA}$ [K]', fontsize=10)
        ax.set_title(f'$L_{{\\rm ext}}$ = {L:g} kpc   ({int(ok.sum())} cells)', fontsize=10)
        if L == Ls[0]:
            ax.set_ylabel(r'$\log_{10}\,T_{\rm DESPOTIC}$ [K]', fontsize=10)
            ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
    cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label(r'$\log_{10} N_{\rm H}$ [cm$^{-2}$]  (shared)', fontsize=10)
    fig.suptitle('Cold+dense table-T vs real-time DESPOTIC  '
                 '(stratified 200-cell sample, down=1; "table vs real-time" box = '
                 'interpolation error)\n'
                 f'mask: $\\log_{{10}}\\rho$ > {LOG_RHO_MIN:g}  &  '
                 f'$\\log_{{10}}T_{{\\rm QK}}$ < {LOG_TQK_MAX:g}', fontsize=10, y=1.02)

    plots_dir = OUT_DIR / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    png = plots_dir / f'despotic_validation_cold_dense_d{DOWNSAMPLE}_sharedcbar_combined.png'
    fig.savefig(str(png), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[out] {png}')


if __name__ == '__main__':
    main()
