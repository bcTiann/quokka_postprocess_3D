#!/usr/bin/env python
"""How each cold+dense cell's T_DESPOTIC migrates as L_ext grows, relative to
L_ext = 0 kpc — for the REPRESENTATIVE (proportional) sample.

Because (n_H, T_QUOKKA) do not depend on L_ext and the sample is drawn with a
fixed RNG, the SAME 400 cells appear at every L_ext and are matched ROW-BY-ROW
across the three *_prop.csv files (verified: T_QK is bit-identical per row).
Only column_density_H changes with L_ext, so T_DESPOTIC moves.

Two panels (real-time T_DESPOTIC):
  left  — migration scatter: each cell sits at its fixed log10 T_QUOKKA; a
          vertical line shows its T_DESPOTIC moving from L_ext=0 down to 9 kpc
          (blue) and to 99 kpc (red).  T_QK is the same, so the lines are
          vertical; the clump barely moves, the diffuse tail moves a lot.
  right — histogram of  Δ = log10(T_DSP^{L} / T_DSP^{0})  for L=9 and L=99.

Run:
    /opt/homebrew/Caskroom/miniconda/base/envs/yt-env/bin/python \
        quokka2s/scripts/plot_validation_migration.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

DOWNSAMPLE = 1
BASE_L     = 0.0
COMP_LS    = [9.0, 99.0]
LOG_RHO_MIN, LOG_TQK_MAX = -23.0, 4.0

OUT_DIR = Path('/Users/baochen/quokka_postprocessing/output/despotic_validation')
# CSV cols:
C_TQK, C_TTAB, C_TREAL, C_FAILED = 5, 6, 7, 10
C_T = C_TREAL                      # use real-time DESPOTIC

COLOR = {9.0: 'tab:blue', 99.0: 'firebrick'}


def _csv(l_ext: float) -> Path:
    return OUT_DIR / f'despotic_validation_cold_dense_d{DOWNSAMPLE}_Lext{l_ext:g}kpc_prop.csv'


def main():
    A = {}
    for L in [BASE_L] + COMP_LS:
        p = _csv(L)
        if not p.exists():
            print(f'[abort] missing {p.name}'); return
        A[L] = np.loadtxt(p, delimiter=',', skiprows=1)

    # Row-order matching sanity check (T_QK must be identical per row).
    for L in COMP_LS:
        if not np.array_equal(A[BASE_L][:, C_TQK], A[L][:, C_TQK]):
            print(f'[abort] row order differs between L0 and L{L:g}'); return

    Tqk = A[BASE_L][:, C_TQK]
    T0  = A[BASE_L][:, C_T]
    # joint validity: real T>0 and not failed at base AND every compared L_ext
    ok = (A[BASE_L][:, C_FAILED] == 0) & (T0 > 0) & (Tqk > 0)
    for L in COMP_LS:
        ok &= (A[L][:, C_FAILED] == 0) & (A[L][:, C_T] > 0)

    x   = np.log10(Tqk[ok])
    y0  = np.log10(T0[ok])
    yL  = {L: np.log10(A[L][:, C_T][ok]) for L in COMP_LS}
    dL  = {L: yL[L] - y0 for L in COMP_LS}   # Δlog10 T_DSP vs L0

    fig, (axm, axh) = plt.subplots(1, 2, figsize=(13.5, 6.0),
                                   gridspec_kw={'width_ratios': [1.0, 1.0], 'wspace': 0.25})

    # ---------- left: migration scatter ----------
    lo = np.floor(min(x.min(), y0.min(), min(v.min() for v in yL.values())))
    hi = np.ceil(max(x.max(), y0.max(), max(v.max() for v in yL.values())))
    axm.grid(True, ls=':', lw=0.5, alpha=0.35, zorder=0)
    axm.plot([lo, hi], [lo, hi], 'k--', lw=1.0, zorder=1)
    # draw the longer (L99) lines first, then L9 on top
    for L in sorted(COMP_LS, reverse=True):
        segs = [[(xi, y0i), (xi, yLi)] for xi, y0i, yLi in zip(x, y0, yL[L])]
        lc = LineCollection(segs, colors=COLOR[L], linewidths=0.6, alpha=0.55, zorder=2)
        axm.add_collection(lc)
    axm.scatter(x, y0, s=7, c='0.25', alpha=0.5, linewidth=0, zorder=3)  # L0 start points
    axm.set_xlim(lo, hi); axm.set_ylim(lo, hi); axm.set_aspect('equal', 'box')
    axm.set_xlabel(r'$\log_{10}\,T_{\rm QUOKKA}$ [K]', fontsize=11)
    axm.set_ylabel(r'$\log_{10}\,T_{\rm DESPOTIC}$ (real-time) [K]', fontsize=11)
    axm.set_title('migration of $T_{\\rm DESPOTIC}$ with $L_{\\rm ext}$\n'
                  '(line = same cell; $T_{\\rm QUOKKA}$ fixed so lines are vertical)',
                  fontsize=10)
    handles = [Line2D([0], [0], color='0.25', marker='o', ls='none', ms=4, label='$L_{\\rm ext}=0$ (start)'),
               Line2D([0], [0], color=COLOR[9.0],  lw=2, label='$0\\to9$ kpc'),
               Line2D([0], [0], color=COLOR[99.0], lw=2, label='$0\\to99$ kpc'),
               Line2D([0], [0], color='k', ls='--', lw=1, label='$T_{\\rm DSP}=T_{\\rm QK}$')]
    axm.legend(handles=handles, fontsize=8.5, loc='lower right', framealpha=0.9)

    # ---------- right: histogram of the plain ratio T_DSP^L / T_DSP^0 ----------
    rL = {L: np.power(10.0, dL[L]) for L in COMP_LS}     # = T_DSP^L / T_DSP^0
    allv = np.concatenate(list(rL.values()))
    bmin, bmax = np.floor(allv.min() * 20) / 20, np.ceil(allv.max() * 20) / 20
    bins = np.linspace(bmin, bmax, 49)
    axh.axvline(1.0, color='k', lw=1.0, ls='-', alpha=0.6, zorder=1)   # 1.0 = no change
    for L in COMP_LS:
        med = float(np.median(rL[L]))
        frac_cool = float(np.mean(rL[L] < 0.9))          # cooled by more than 10%
        axh.hist(rL[L], bins=bins, histtype='stepfilled', color=COLOR[L], alpha=0.45,
                 edgecolor=COLOR[L], linewidth=1.2, zorder=2,
                 label=f'$0\\to{L:g}$ kpc  (median {med:.2f}×, '
                       f'{frac_cool:.0%} cooled >10%)')
        axh.axvline(med, color=COLOR[L], lw=1.4, ls='--', zorder=3)
    axh.set_yscale('log')
    axh.set_xlabel(r'$T_{\rm DSP}^{L}\,/\,T_{\rm DSP}^{0}$   '
                   '($<1$ = cooler at larger $L_{\\rm ext}$)', fontsize=10)
    axh.set_ylabel('cell count', fontsize=11)
    axh.set_title('how far each cell\'s $T_{\\rm DESPOTIC}$ shifts (relative to $L_{\\rm ext}=0$)',
                  fontsize=10)
    axh.legend(fontsize=8.5, loc='upper left', framealpha=0.9)
    axh.grid(True, ls=':', lw=0.5, alpha=0.35)

    fig.suptitle('Cold+dense representative sample — $T_{\\rm DESPOTIC}$ response to $L_{\\rm ext}$ '
                 '(real-time, down=1, 400 cells)\n'
                 f'mask: $\\log_{{10}}\\rho$ > {LOG_RHO_MIN:g}  &  '
                 f'$\\log_{{10}}T_{{\\rm QK}}$ < {LOG_TQK_MAX:g}', fontsize=11, y=1.02)

    plots_dir = OUT_DIR / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    png = plots_dir / f'despotic_validation_cold_dense_d{DOWNSAMPLE}_prop_migration.png'
    fig.savefig(str(png), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[out] {png}')
    print(f'[info] {int(ok.sum())} cells valid in all of L_ext={[BASE_L]+COMP_LS}')
    for L in COMP_LS:
        d = dL[L]
        print(f'  0->{L:g} kpc:  median {np.median(d):+.3f} dex  '
              f'max |Δ| {np.max(np.abs(d)):.3f} dex  '
              f'frac cooler {np.mean(d < 0):.0%}')


if __name__ == '__main__':
    main()
