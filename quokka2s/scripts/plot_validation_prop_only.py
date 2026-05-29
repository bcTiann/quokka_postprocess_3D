#!/usr/bin/env python
"""Standalone figure of the PROPORTIONAL (representative) cold+dense validation
sample: 1 row x 3 columns (L_ext = 0 / 9 / 99 kpc).

Reads the existing *_prop.csv files (no DESPOTIC re-solve).

Run:
    /opt/homebrew/Caskroom/miniconda/base/envs/yt-env/bin/python \
        quokka2s/scripts/plot_validation_prop_only.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

DOWNSAMPLE  = 1
L_EXTS      = [0.0, 9.0, 99.0]
METHOD      = 'prop'
LOG_RHO_MIN = -23.0
LOG_TQK_MAX = 4.0

# point sizes
S_REAL, S_TAB, LW_TAB = 14, 30, 0.5

OUT_DIR = Path('/Users/baochen/quokka_postprocessing/output/despotic_validation')
C_COLDEN, C_TQK, C_TTAB, C_TREAL, C_DEXTR, C_FAILED = 3, 5, 6, 7, 8, 10


def _csv(l_ext: float) -> Path:
    return OUT_DIR / f'despotic_validation_cold_dense_d{DOWNSAMPLE}_Lext{l_ext:g}kpc_{METHOD}.csv'


def main():
    data = {}
    for L in L_EXTS:
        p = _csv(L)
        if p.exists():
            data[L] = np.loadtxt(p, delimiter=',', skiprows=1)
        else:
            print(f'[skip] missing {p.name}')
    if not data:
        print('no prop CSVs found'); return
    Ls = list(data.keys())

    # shared colour + axis ranges across panels
    all_logN = np.concatenate([np.log10(a[:, C_COLDEN][a[:, C_COLDEN] > 0]) for a in data.values()])
    cnorm = Normalize(vmin=float(all_logN.min()), vmax=float(all_logN.max()))
    lo, hi = np.inf, -np.inf
    for a in data.values():
        for col in (C_TQK, C_TTAB, C_TREAL):
            v = a[:, col][a[:, col] > 0]
            lo = min(lo, np.log10(v).min()); hi = max(hi, np.log10(v).max())
    lo, hi = np.floor(lo), np.ceil(hi)

    n = len(Ls)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.4), sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    sc = None
    for ci, (ax, L) in enumerate(zip(axes, Ls)):
        a = data[L]
        colDen, Tqk, Ttab, Treal = a[:, C_COLDEN], a[:, C_TQK], a[:, C_TTAB], a[:, C_TREAL]
        failed = a[:, C_FAILED].astype(bool)
        ok = (~failed & np.isfinite(Treal) & (Treal > 0) & (Tqk > 0) & (Ttab > 0) & (colDen > 0))
        lr = np.log10(Tqk[ok]); lt_tab = np.log10(Ttab[ok]); lt_real = np.log10(Treal[ok])
        cval = np.log10(colDen[ok])
        dex = np.abs(a[:, C_DEXTR][ok])
        med_dex, max_dex = float(np.median(dex)), float(np.max(dex))

        ax.grid(True, ls=':', lw=0.5, alpha=0.35, zorder=0)
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, zorder=1,
                label=r'$T_{\rm DESPOTIC}=T_{\rm QUOKKA}$')
        sc = ax.scatter(lr, lt_real, c=cval, cmap='viridis', norm=cnorm,
                        marker='o', s=S_REAL, alpha=0.85, linewidth=0.0, zorder=3,
                        label=r'$T_{\rm DESPOTIC}$ (real-time)')
        ax.scatter(lr, lt_tab, facecolors='none', edgecolors='k',
                   marker='o', s=S_TAB, linewidth=LW_TAB, alpha=0.55, zorder=2,
                   label=r'$T_{\rm DESPOTIC}$ (table)')
        # interpolation-error box in the (empty) lower-right corner
        ax.text(0.97, 0.03,
                'table vs real-time:\n'
                f'median {med_dex:.3f} dex\nmax {max_dex:.3f} dex',
                transform=ax.transAxes, va='bottom', ha='right', fontsize=8,
                bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.9))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel(r'$\log_{10}\,T_{\rm QUOKKA}$ [K]', fontsize=10)
        ax.set_title(f'$L_{{\\rm ext}}$ = {L:g} kpc   ({int(ok.sum())} cells)', fontsize=10)
        if ci == 0:
            # y-label describes HOW the cells were sampled
            ax.set_ylabel('sampling: weighted by # cells per '
                          r'$(\rho,\,T_{\rm QK})$ bin' '\n'
                          r'$\log_{10}\,T_{\rm DESPOTIC}$ [K]', fontsize=10)
            # stacked just above the interpolation-error box (also lower-right)
            ax.legend(fontsize=8.5, loc='lower right',
                      bbox_to_anchor=(0.98, 0.22), framealpha=0.9)

    cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label(r'$\log_{10} N_{\rm H}$ [cm$^{-2}$]  (shared)', fontsize=10)
    fig.suptitle('Cold+dense validation — representative sample '
                 '(each cell drawn $\\propto$ number of cells per '
                 r'$(\rho,\,T_{\rm QK})$ bin; down=1, 400 cells)' '\n'
                 f'mask: $\\log_{{10}}\\rho$ > {LOG_RHO_MIN:g}  &  '
                 f'$\\log_{{10}}T_{{\\rm QK}}$ < {LOG_TQK_MAX:g}', fontsize=10, y=1.02)

    plots_dir = OUT_DIR / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    png = plots_dir / f'despotic_validation_cold_dense_d{DOWNSAMPLE}_prop_only.png'
    fig.savefig(str(png), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[out] {png}')


if __name__ == '__main__':
    main()
