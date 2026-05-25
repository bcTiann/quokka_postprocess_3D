#!/usr/bin/env python
"""Draw the phase-cut representative plots in two flavours:

  * HARMONIC only (3 groups: L_ext = 0/9/99 kpc)        → plots/
  * WITH the arithmetic-mean sensitivity variant (4 grp) → arithmetic_mean/

Each flavour produces:
  representative_byMask_2dhist_mass_d1.png   (rows = group, cols = mask)
  representative_byMask_2dhist_count_d1.png
  representative_byMask_fraction_allLext_d1.png  fraction vs phase cut + table
  representative_byMask_NH_hist_allLext_d1.png   N_H distribution (group rows)

Harmonic npz live in despotic_validation/; the arithmetic-mean npz live in
despotic_validation/arithmetic_mean/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

DOWNSAMPLE = 1
OUT_DIR    = Path('/Users/baochen/quokka_postprocessing/output/despotic_validation')
PLOTS_DIR  = OUT_DIR / 'plots'                 # harmonic-only plots
ARITH_DIR  = OUT_DIR / 'arithmetic_mean'       # arith npz + arith-inclusive plots

# (label, short, npz "Ltag", colour). The last one is the arithmetic variant.
GROUP_HARM = [
    (r'$L_{\rm ext}$=0 kpc',  'L=0',  'd1_Lext0kpc',  'tab:blue'),
    (r'$L_{\rm ext}$=9 kpc',  'L=9',  'd1_Lext9kpc',  'tab:orange'),
    (r'$L_{\rm ext}$=99 kpc', 'L=99', 'd1_Lext99kpc', 'tab:green'),
]
GROUP_ARITH = (r'$L_{\rm ext}$=9 kpc (arith. mean)', 'L=9 ar', 'd1_Lext9kpc_arith', 'tab:red')
GROUP_ALL  = GROUP_HARM + [GROUP_ARITH]

# Ordered diffuse -> dense (matches representative_cold_dense.py).
MASKS = [
    ('WIM',            r'$n_{\rm H}\,0.02\!-\!1,\ T_{\rm QK}\,8\!-\!30$kK',  'Warm ionized\n(WIM)'),
    ('WNM',            r'$n_{\rm H}\,0.05\!-\!3,\ T_{\rm QK}\,3\!-\!8$kK',   'Warm neutral\n(WNM)'),
    ('nHgt30_Tlt200',  r'$n_{\rm H}{>}30,\ T_{\rm QK}{<}200$',  'Cool H I (CNM)\n+ molecular'),
    ('nHgt100_Tlt100', r'$n_{\rm H}{>}100,\ T_{\rm QK}{<}100$', 'Diffuse molecular\nH$_2$'),
    ('nHgt1000_Tlt50', r'$n_{\rm H}{>}1000,\ T_{\rm QK}{<}50$', 'Dense molecular\nH$_2$'),
]


def _npz(ltag, tag):
    base = ARITH_DIR if 'arith' in ltag else OUT_DIR
    return base / f'representative_{ltag}_{tag}.npz'


def _draw_2dhist_grid(groups, field, weight_word, outdir, fname):
    grid, allpos = {}, []
    for _lab, _s, ltag, _c in groups:
        for tag, _crit, _ph in MASKS:
            p = _npz(ltag, tag)
            if p.exists():
                d = np.load(p); grid[(ltag, tag)] = d
                allpos.append(d[field][d[field] > 0].ravel())
    if not allpos:
        print('no npz found'); return
    norm = LogNorm(vmin=float(np.concatenate(allpos).min()),
                   vmax=float(np.concatenate(allpos).max()))
    nR, nC = len(groups), len(MASKS)
    fig, axes = plt.subplots(nR, nC, figsize=(4.1 * nC, 3.7 * nR), sharex=True, sharey=True,
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    im = None
    for r, (glab, _s, ltag, _c) in enumerate(groups):
        for c, (tag, crit, phase) in enumerate(MASKS):
            ax = axes[r, c]; d = grid.get((ltag, tag))
            if d is None:
                ax.set_visible(False); continue
            H = d[field]; Te = d['T_edges']; ext = [Te[0], Te[-1], Te[0], Te[-1]]
            im = ax.imshow(np.ma.masked_where(H <= 0, H).T, origin='lower',
                           extent=ext, aspect='equal', cmap='viridis', norm=norm)
            ax.plot([Te[0], Te[-1]], [Te[0], Te[-1]], 'r--', lw=1.1)
            frac = float(d['frac_mass']) if field == 'H_mass' else float(d['frac_cnt'])
            noun = 'mass' if field == 'H_mass' else 'cells'
            ax.text(0.05, 0.95,
                    f'total cells: {int(d["n_mask"])}\n'
                    + r'$T_{\rm DESPOTIC} > T_{\rm QUOKKA}$' + '\n'
                    + f'for {frac:.0%} of {noun}',
                    transform=ax.transAxes, va='top', ha='left', fontsize=7.5,
                    bbox=dict(boxstyle='round', fc='white', alpha=0.75, lw=0))
            if r == 0:
                ax.set_title(f'{crit}\n{phase}', fontsize=9)
            if c == 0:
                ax.set_ylabel(f'{glab}\n' + r'$\log_{10}T_{\rm DESPOTIC}$', fontsize=8.5)
            if r == nR - 1:
                ax.set_xlabel(r'$\log_{10}T_{\rm QUOKKA}$ [K]', fontsize=9)
            ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.04, pad=0.02)
    cb.set_label(f'$\\log_{{10}}$ ({weight_word} per bin)', fontsize=10)
    fig.savefig(str(outdir / fname), dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'[out] {outdir / fname}')


def _draw_fraction(groups, outdir):
    xpos = np.arange(len(MASKS))
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    table = {}
    for glab, short, ltag, col in groups:
        ys, ok = [], True
        for tag, _c, _p in MASKS:
            p = _npz(ltag, tag)
            if not p.exists():
                ok = False; break
            ys.append(float(np.load(p)['frac_mass']))
        if not ok:
            print(f'[skip] {ltag} incomplete'); continue
        ls = '--' if 'arith' in ltag else '-'
        ax.plot(xpos, ys, ls, marker='o', lw=2, ms=7, color=col, label=glab)
        table[short] = ys
    ncells = [int(np.load(_npz('d1_Lext9kpc', tag))['n_mask']) for tag, _c, _p in MASKS]
    ax.set_xticks(xpos)
    ax.set_xticklabels(
        [f'{crit}\n{phase.replace(chr(10), " ")}\n({n} cells)'
         for (_t, crit, phase), n in zip(MASKS, ncells)], fontsize=8)
    ax.set_ylabel(r'by-mass fraction with $T_{\rm DESPOTIC} > T_{\rm QUOKKA}$', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, ls=':', alpha=0.4)
    col_hdr = {'WIM': 'WIM', 'WNM': 'WNM', 'nHgt30_Tlt200': 'CNM',
               'nHgt100_Tlt100': 'mol', 'nHgt1000_Tlt50': 'dense'}
    hdr = ' ' * 8 + ' '.join(f'{col_hdr.get(t, t[:5]):>5}' for t, _c, _p in MASKS)
    lines = ['by-mass fraction (%)', hdr]
    for _l, short, _lt, _c in groups:
        if short in table:
            lines.append(f'{short:<7} ' + ' '.join(f'{v*100:5.1f}' for v in table[short]))
    ax.text(0.03, 0.04, '\n'.join(lines), transform=ax.transAxes, va='bottom',
            ha='left', fontsize=8, family='monospace',
            bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.9))
    out = outdir / f'representative_byMask_fraction_allLext_d{DOWNSAMPLE}.png'
    fig.savefig(str(out), dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'[out] {out}')


def _draw_NH_hist(groups, outdir):
    cols = ['tab:purple', 'tab:cyan', 'tab:blue', 'tab:orange', 'tab:green']
    nR = len(groups)
    fig, axes = plt.subplots(nR, 1, figsize=(7.6, 2.7 * nR), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (glab, _s, ltag, _c) in zip(axes, groups):
        for (tag, _crit, phase), c in zip(MASKS, cols):
            p = _npz(ltag, tag)
            if not p.exists():
                continue
            d = np.load(p); e = d['NH_edges']; ctr = 0.5 * (e[:-1] + e[1:])
            h = d['NH_mass'].astype(float)
            if h.sum() == 0:
                continue
            ax.step(ctr, h / h.sum(), where='mid', lw=1.8, color=c,
                    label=phase.replace(chr(10), ' '))
        ax.set_ylabel('mass fraction', fontsize=9)
        ax.set_title(glab, fontsize=9.5, loc='left')
        ax.legend(fontsize=7, loc='upper right'); ax.grid(True, ls=':', alpha=0.35)
    axes[-1].set_xlabel(r'$\log_{10}\,N_{\rm H}$ [cm$^{-2}$]', fontsize=11)
    out = outdir / f'representative_byMask_NH_hist_allLext_d{DOWNSAMPLE}.png'
    fig.savefig(str(out), dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'[out] {out}')


def _make_set(groups, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    _draw_2dhist_grid(groups, 'H_mass',  'mass',  outdir, f'representative_byMask_2dhist_mass_d{DOWNSAMPLE}.png')
    _draw_2dhist_grid(groups, 'H_count', 'count', outdir, f'representative_byMask_2dhist_count_d{DOWNSAMPLE}.png')
    _draw_fraction(groups, outdir)
    _draw_NH_hist(groups, outdir)


def main():
    print('=== by-mass fraction T_DSP > T_QK (rows = colDen recipe) ===')
    for glab, short, ltag, _c in GROUP_ALL:
        row = [f'{float(np.load(_npz(ltag, tag))["frac_mass"]):.1%}'
               if _npz(ltag, tag).exists() else '  --' for tag, _c2, _p in MASKS]
        print(f'  {short:<8} ' + '   '.join(row))
    print('\n--- harmonic only (3 groups) → plots/ ---')
    _make_set(GROUP_HARM, PLOTS_DIR)
    print('\n--- with arithmetic-mean variant (4 groups) → arithmetic_mean/ ---')
    _make_set(GROUP_ALL, ARITH_DIR)


if __name__ == '__main__':
    main()
