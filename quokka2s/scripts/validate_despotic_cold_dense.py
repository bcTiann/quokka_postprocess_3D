#!/usr/bin/env python
"""Validate the pipeline's table-interpolated T_DESPOTIC against the REAL
DESPOTIC solver, in the cold + dense regime.

Motivation
----------
TrustRegionTask surfaced a surprise: in cold + dense cells the pipeline's
``temperature_despotic`` (a *table interpolation* over (nH, NH, dVdr)) often
comes out HOTTER than ``temperature_quokka`` — opposite to the intuition that
DESPOTIC should be colder there. This script checks whether that is
  (a) wrong inputs, (b) table-interpolation error, or (c) real physics,
by running the real DESPOTIC solver (`calculate_single_despotic_point`,
same GOW network + dust/radiation params that BUILT the table) on a
stratified sample of cold+dense cells and comparing, per cell:

    T_QUOKKA   vs   T_DSP(table interp)   vs   T_DSP(real setChemEq)

Run
---
Set config.py to DOWNSAMPLE_FACTOR=1, COLUMN_EXTENSION_LATERAL_KPC=9.0 first
(so the on-disk d1/L9 field caches are reused — no recompute), then:

    /opt/homebrew/Caskroom/miniconda/base/envs/yt-env/bin/python \
        quokka2s/scripts/validate_despotic_cold_dense.py

Never launch via `conda run` (it silently kills long jobs on this Mac).

Outputs (under output/despotic_validation/):
    despotic_validation_cold_dense.csv   — per-cell numbers
    despotic_validation_cold_dense.png   — scatter T_QK vs {table, real}
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# --- selection / sampling knobs ---
LOG_RHO_MIN   = -23.0      # cold+dense mask: log10(ρ [g/cm³]) >  this
LOG_TQK_MAX   = 4.0        # cold+dense mask: log10(T_QK [K])  <  this
BIN_DEX       = 0.5        # stratified-sampling bin width on (logρ, logT_QK)
N_TARGET      = 200        # number of cells to run real DESPOTIC on
RNG_SEED      = 42

_SRC = Path(__file__).resolve().parents[1] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yt  # noqa: E402

from quokka2s.pipeline.prep import config as cfg                       # noqa: E402
from quokka2s.pipeline.prep import physics_fields as phys              # noqa: E402
from quokka2s.data_handling import YTDataProvider, make_downsampled_dataset  # noqa: E402
from quokka2s.pipeline.cache import compute_cache_key, cache_root_for_dataset  # noqa: E402
from quokka2s.tables.solver import calculate_single_despotic_point     # noqa: E402
from despotic.chemistry import GOW                                     # noqa: E402


def _load_cubes(return_provider=False):
    """Load ρ, n_H, T_QUOKKA, colDen, dVdr as 1-D cgs arrays, exactly as the
    pipeline does (cached derived fields are reused).

    return_provider=True also returns the YTDataProvider (so callers can pull
    the 3-D cubes, e.g. to recompute colDen with a different mean method)."""
    ds = yt.load(cfg.YT_DATASET_PATH)
    if cfg.DOWNSAMPLE_FACTOR > 1:
        ds = make_downsampled_dataset(ds, cfg.DOWNSAMPLE_FACTOR)
    phys.add_all_fields(ds)

    key = compute_cache_key(
        dataset_path                 = cfg.YT_DATASET_PATH,
        despotic_table_path          = cfg.DESPOTIC_TABLE_PATH,
        downsample_factor            = cfg.DOWNSAMPLE_FACTOR,
        column_extension_lateral_kpc = cfg.COLUMN_EXTENSION_LATERAL_KPC,
    )
    provider = YTDataProvider(
        ds,
        cache_root = cache_root_for_dataset(cfg.YT_DATASET_PATH),
        cache_key  = key,
    )

    print(f'[load] down={cfg.DOWNSAMPLE_FACTOR}  L_ext={cfg.COLUMN_EXTENSION_LATERAL_KPC} kpc')
    rho_u,    _ = provider.get_slab_z(('gas', 'density'))
    nH_u,     _ = provider.get_slab_z(('gas', 'number_density_H'))
    Tqk_u,    _ = provider.get_slab_z(('gas', 'temperature_quokka'))
    colden_u, _ = provider.get_slab_z(('gas', 'column_density_H'))   # cache hit
    dvdr_u,   _ = provider.get_slab_z(('gas', 'dVdr_lvg'))           # cache hit

    rho    = np.asarray(rho_u.in_cgs()).ravel()
    n_H    = np.asarray(nH_u.in_cgs()).ravel()
    T_qk   = np.asarray(Tqk_u.in_cgs()).ravel()
    colden = np.asarray(colden_u.in_cgs()).ravel()
    dvdr   = np.asarray(dvdr_u.in_cgs()).ravel()
    if return_provider:
        return rho, n_H, T_qk, colden, dvdr, provider
    return rho, n_H, T_qk, colden, dvdr


def _stratified_sample(log_rho, log_Tqk, mask, rng):
    """Pick ~N_TARGET masked cells, spread across (logρ, logT_QK) 0.5-dex bins."""
    idx = np.flatnonzero(mask)                      # flat cube indices passing mask
    lr  = log_rho[idx]
    lt  = log_Tqk[idx]

    br = np.floor(lr / BIN_DEX).astype(np.int64)
    bt = np.floor(lt / BIN_DEX).astype(np.int64)
    key = br * 1000 + bt                            # bt range ≪ 1000 → no collision
    uniq, inv = np.unique(key, return_inverse=True)
    B = uniq.size
    q = max(1, int(round(N_TARGET / B)))
    print(f'[sample] {idx.size} masked cells in {B} non-empty (ρ,T) bins; '
          f'quota={q}/bin')

    chosen = []
    for bi in range(B):
        members = np.flatnonzero(inv == bi)         # positions within `idx`
        k = min(q, members.size)
        chosen.append(rng.choice(members, size=k, replace=False))
    chosen = np.concatenate(chosen)

    if chosen.size > N_TARGET:
        chosen = rng.choice(chosen, size=N_TARGET, replace=False)
    elif chosen.size < N_TARGET:
        remaining = np.setdiff1d(np.arange(idx.size), chosen)
        if remaining.size:
            extra = rng.choice(remaining,
                               size=min(N_TARGET - chosen.size, remaining.size),
                               replace=False)
            chosen = np.concatenate([chosen, extra])
    return idx[chosen]                              # flat cube indices of the sample


def main():
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)

    rho, n_H, T_qk, colden, dvdr = _load_cubes()
    print(f'[load] done in {time.time() - t0:.1f}s  ({rho.size} cells)')

    with np.errstate(divide='ignore', invalid='ignore'):
        log_rho = np.log10(np.where(rho  > 0, rho,  np.nan))
        log_Tqk = np.log10(np.where(T_qk > 0, T_qk, np.nan))

    mask = (
        np.isfinite(log_rho) & np.isfinite(log_Tqk)
        & (log_rho > LOG_RHO_MIN) & (log_Tqk < LOG_TQK_MAX)
        & (n_H > 0) & (colden > 0) & (dvdr > 0)
    )
    print(f'[mask] cold+dense (logρ>{LOG_RHO_MIN}, logT_QK<{LOG_TQK_MAX}): '
          f'{int(mask.sum())} cells')

    sample = _stratified_sample(log_rho, log_Tqk, mask, rng)
    print(f'[sample] selected {sample.size} cells')

    # Table-interpolated T_DSP for the whole sample at once.
    lookup = phys.ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
    nH_s   = n_H[sample]
    col_s  = colden[sample]
    dv_s   = dvdr[sample]
    Tqk_s  = T_qk[sample]
    T_tab  = lookup.temperature(nH_s, col_s, dv_s)

    # Real DESPOTIC, one cell at a time.
    print(f'[despotic] solving {sample.size} cells with real setChemEq '
          f'(GOW, iterateDust)…')
    T_real = np.full(sample.size, np.nan)
    failed = np.zeros(sample.size, dtype=bool)
    for i in range(sample.size):
        ts = time.time()
        out = calculate_single_despotic_point(
            float(nH_s[i]), float(col_s[i]), [float(dv_s[i])],
            chem_network=GOW, log_failures=False,
        )
        T_real[i] = out[6]      # final_Tg
        failed[i] = bool(out[8])
        print(f'  [{i+1:2d}/{sample.size}] nH={nH_s[i]:.3e} colDen={col_s[i]:.3e} '
              f'T_QK={Tqk_s[i]:.1f}  '
              f'T_tab={T_tab[i]:.1f}  T_real={T_real[i]:.1f}  '
              f'{"FAILED" if failed[i] else f"{time.time()-ts:.1f}s"}')

    # --- write CSV ---  (filename encodes down + L_ext so the two L_ext runs
    #                      don't overwrite each other)
    tag = f'd{cfg.DOWNSAMPLE_FACTOR}_Lext{cfg.COLUMN_EXTENSION_LATERAL_KPC:g}kpc'
    out_dir = Path(cfg._OUTPUT_ROOT) / 'despotic_validation'
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'despotic_validation_cold_dense_{tag}.csv'

    with np.errstate(divide='ignore', invalid='ignore'):
        dex_tab_real = np.log10(T_tab / T_real)
        dex_real_qk  = np.log10(T_real / Tqk_s)

    header = ('flat_idx,log_rho,n_H,colDen,dVdr,'
              'T_QK,T_DSP_table,T_DSP_real,dex_table_over_real,dex_real_over_QK,failed')
    rows = np.column_stack([
        sample.astype(float), log_rho[sample], nH_s, col_s, dv_s,
        Tqk_s, T_tab, T_real, dex_tab_real, dex_real_qk, failed.astype(float),
    ])
    np.savetxt(csv_path, rows, delimiter=',', header=header, comments='',
               fmt='%.6e')
    print(f'[out] {csv_path}')

    # --- scatter plot ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    ok = ~failed & np.isfinite(T_real) & (T_real > 0) & (Tqk_s > 0) & (T_tab > 0)
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    lr = np.log10(Tqk_s[ok]); lt_tab = np.log10(T_tab[ok]); lt_real = np.log10(T_real[ok])
    lo = np.floor(min(lr.min(), lt_tab.min(), lt_real.min()))
    hi = np.ceil(max(lr.max(), lt_tab.max(), lt_real.max()))

    ax.grid(True, ls=':', lw=0.5, alpha=0.35, zorder=0)
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, zorder=1,
            label=r'$T_{\rm DESPOTIC} = T_{\rm QUOKKA}$')

    cval = np.log10(col_s[ok])                      # colour by log10 colDen (N_H)
    cnorm = Normalize(vmin=float(cval.min()), vmax=float(cval.max()))
    # real-time = small filled circle; table = hollow ring on top, so the two
    # estimates per cell are distinguishable even at 200 points.
    sc = ax.scatter(lr, lt_real, c=cval, cmap='viridis', norm=cnorm,
                    marker='o', s=22, alpha=0.85, linewidth=0.0, zorder=3,
                    label=r'$T_{\rm DESPOTIC}$ (real-time)')
    ax.scatter(lr, lt_tab, facecolors='none', edgecolors='k',
               marker='o', s=48, linewidth=0.6, alpha=0.6, zorder=2,
               label=r'$T_{\rm DESPOTIC}$ (table)')
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r'$\log_{10} N_{\rm H}$ [cm$^{-2}$]', fontsize=10)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r'$\log_{10}\,T_{\rm QUOKKA}$ [K]', fontsize=11)
    ax.set_ylabel(r'$\log_{10}\,T_{\rm DESPOTIC}$ [K]', fontsize=11)
    ax.set_title(
        'Cold+dense validation: table vs real DESPOTIC\n'
        f'mask:  $\\log_{{10}}\\rho$ > {LOG_RHO_MIN:g}   &   '
        f'$\\log_{{10}}T_{{\\rm QK}}$ < {LOG_TQK_MAX:g}    '
        f'(stratified {int(ok.sum())} cells)\n'
        f'down={cfg.DOWNSAMPLE_FACTOR},  '
        f'$L_{{\\rm ext}}$ = {cfg.COLUMN_EXTENSION_LATERAL_KPC:g} kpc',
        fontsize=9)
    ax.legend(fontsize=8.5, loc='lower right', framealpha=0.9)
    ax.set_aspect('equal', 'box')
    plots_dir = out_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    png_path = plots_dir / f'despotic_validation_cold_dense_{tag}.png'
    fig.savefig(str(png_path), dpi=200, bbox_inches='tight')
    print(f'[out] {png_path}')

    # --- summary ---
    n_fail = int(failed.sum())
    med_interp_err = float(np.nanmedian(np.abs(dex_tab_real[ok])))
    frac_real_hotter = float(np.mean(T_real[ok] > Tqk_s[ok]))
    print('\n================ summary ================')
    print(f'  cells solved        : {sample.size}  ({n_fail} failed)')
    print(f'  median |log10(table/real)| (interp error) : {med_interp_err:.3f} dex')
    print(f'  fraction T_real > T_QK                    : {frac_real_hotter:.1%}')
    print(f'  total wall time     : {time.time() - t0:.1f}s')
    print('=========================================')


if __name__ == '__main__':
    main()
