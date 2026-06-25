#!/usr/bin/env python
"""Full-population (NO sampling) cold+dense comparison: T_QUOKKA vs
table-interpolated T_DESPOTIC over ALL masked cells.

The table interpolation was already validated against real DESPOTIC
(median |log10(table/real)| = 0.001-0.011 dex by
validate_despotic_cold_dense.py), so here we trust table-T and apply it to
the WHOLE cold+dense population — exact, unbiased, and fast (no real-DESPOTIC
solve).  This answers "in the cold+dense gas that actually dominates (by
mass / by number), is DESPOTIC hotter than QUOKKA?", which the stratified
200-cell sample CANNOT (it over-represents rare ρ-T corners).

Saves per-(downsample, L_ext) results to an npz; plot_representative_cold_dense.py
reads the three npz and draws the side-by-side figures with shared scales.

Run once per L_ext (set config.py first), via the env python directly:
    /opt/homebrew/Caskroom/miniconda/base/envs/yt-env/bin/python \
        scripts/representative_cold_dense.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Five ISM phases (Draine Table 1.3), masked on n_H and T_QUOKKA (both
# independent of L_ext, so the same cells are selected at any L_ext).  Ordered
# diffuse -> dense.  Two warm *bands* (WIM/WNM) + the original three cold *cuts*.
#   (tag, nH_min, nH_max, T_min, T_max, Draine phase)  — None = open end, so the
#   cold cuts reproduce the original (n_H > min) & (T_QK < max) byte-identically.
MASKS = [
    ('WIM',             0.02,   1.0,   8.0e3,  3.0e4, 'Warm ionized (WIM)'),
    ('WNM',             0.05,   3.0,   3.0e3,  8.0e3, 'Warm neutral (WNM)'),
    ('nHgt30_Tlt200',   30.0,   None,  None,   200.0, 'CNM + molecular'),
    ('nHgt100_Tlt100',  100.0,  None,  None,   100.0, '(diffuse+dense) H2'),
    ('nHgt1000_Tlt50',  1000.0, None,  None,    50.0, 'dense H2 cores'),
]
# Fixed bin edges so all npz are directly comparable.  N_H lower edge is 17
# (not 18) so the diffuse warm gas — whose 6-direction column can dip below
# 10^18, especially at L_ext=0 — is not dropped off the histogram.
T_EDGES  = np.round(np.arange(0.5, 4.50 + 1e-9, 0.1), 4)   # log10 T [K] (both axes)
NH_EDGES = np.round(np.arange(17.0, 24.0 + 1e-9, 0.1), 4)  # log10 N_H [cm^-2]

_SRC = Path(__file__).resolve().parents[1] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# reuse the exact field loader the validation script uses
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_despotic_cold_dense import _load_cubes          # noqa: E402
from quokka2s.pipeline.prep import config as cfg               # noqa: E402
from quokka2s.pipeline.prep import physics_fields as phys      # noqa: E402
from quokka2s.analysis import along_sight_cumulation           # noqa: E402
from yt.units import kpc                                       # noqa: E402

OUT_DIR = Path(cfg._OUTPUT_ROOT) / 'despotic_validation'

# colDen direction-combine method: 'harmonic' (default, the pipeline's) or
# 'arithmetic' (sensitivity test — overstates shielding, NOT physically
# preferred; see discussion). Pick via argv: `... representative_cold_dense.py arithmetic`.
MEAN_METHOD = sys.argv[1] if len(sys.argv) > 1 else 'harmonic'
assert MEAN_METHOD in ('harmonic', 'arithmetic'), MEAN_METHOD


def _arithmetic_colden_1d(provider) -> np.ndarray:
    """Re-compute colDen with the ARITHMETIC mean of the 6 directional columns
    (instead of the harmonic mean the pipeline uses).  x/y get the SAME lateral
    L_ext extension as `_column_density_H`; ±z is not extended.  Returns the
    colDen cube raveled to 1-D, cgs (cm^-2)."""
    nH3d = np.asarray(provider.get_slab_z(('gas', 'number_density_H'))[0].in_cgs())
    dx3d = np.asarray(provider.get_slab_z(('boxlib', 'dx'))[0].in_cgs())
    dy3d = np.asarray(provider.get_slab_z(('boxlib', 'dy'))[0].in_cgs())
    dz3d = np.asarray(provider.get_slab_z(('boxlib', 'dz'))[0].in_cgs())

    L = float(cfg.COLUMN_EXTENSION_LATERAL_KPC)
    if L > 0.0:
        L_cm = L * float((1.0 * kpc).in_units('cm').value)
        N_ext = (L_cm * nH3d.mean(axis=(0, 1)))[None, None, :]    # (1,1,nz), cm^-2
    else:
        N_ext = None

    acc = None
    for axis, sign, d3d, lateral in (
        ('x', '+', dx3d, True), ('x', '-', dx3d, True),
        ('y', '+', dy3d, True), ('y', '-', dy3d, True),
        ('z', '+', dz3d, False), ('z', '-', dz3d, False),
    ):
        N = along_sight_cumulation(nH3d * d3d, axis=axis, sign=sign)
        if lateral and N_ext is not None:
            N = N + N_ext
        acc = N if acc is None else acc + N
        del N
    return (acc / 6.0).ravel()


def main():
    t0 = time.time()
    if MEAN_METHOD == 'arithmetic':
        rho, n_H, T_qk, _colden_harm, dvdr, provider = _load_cubes(return_provider=True)
        colden = _arithmetic_colden_1d(provider)
        print(f'[colDen] ARITHMETIC mean (sensitivity test); '
              f'median log10 N_H = {np.nanmedian(np.log10(colden[colden>0])):.2f}')
    else:
        rho, n_H, T_qk, colden, dvdr = _load_cubes()
    print(f'[load] method={MEAN_METHOD}  done in {time.time()-t0:.1f}s  ({rho.size} cells)')

    # table-T over EVERY cell (TableLookup chunks internally → low RAM).
    lookup = phys.ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
    tb = lookup.table
    nH_min, nH_max   = float(tb.nH_values.min()),          float(tb.nH_values.max())
    col_min, col_max = float(tb.col_density_values.min()), float(tb.col_density_values.max())
    dv_min,  dv_max  = float(tb.dVdr_values.min()),        float(tb.dVdr_values.max())

    # Range check: arithmetic-mean colDen can exceed the table N_H ceiling.
    # Report how many cells fall outside each axis (so over-range is explicit,
    # not silently NaN-dropped), then CLIP to the table edge exactly like the
    # pipeline's _temperature_despotic does.
    n_oc = int(np.sum(colden > col_max)); n_on = int(np.sum(n_H > nH_max))
    n_od = int(np.sum(dvdr > dv_max))
    print(f'[range] table colDen max = 10^{np.log10(col_max):.1f};  '
          f'cells over colDen max: {n_oc} ({100*n_oc/colden.size:.4f}%), '
          f'over n_H max: {n_on}, over dVdr max: {n_od}  → clipped to edge')
    colden_oor = colden > col_max          # remember for per-mask reporting
    nH_c  = np.clip(n_H,   nH_min,  nH_max)
    col_c = np.clip(colden, col_min, col_max)
    dv_c  = np.clip(dvdr,  dv_min,  dv_max)
    T_dsp = lookup.temperature(nH_c, col_c, dv_c)
    print(f'[table-T] done in {time.time()-t0:.1f}s')

    with np.errstate(divide='ignore', invalid='ignore'):
        log_Tqk  = np.log10(np.where(T_qk   > 0, T_qk,   np.nan))
        log_Tdsp = np.log10(np.where(T_dsp  > 0, T_dsp,  np.nan))
        log_NH   = np.log10(np.where(colden > 0, colden, np.nan))   # UNclipped (real N_H)

    # cells valid for any mask (finite axes + positive inputs)
    base_ok = (np.isfinite(log_Tqk) & np.isfinite(log_Tdsp)
               & (n_H > 0) & (colden > 0) & (dvdr > 0))

    Ltag = f'd{cfg.DOWNSAMPLE_FACTOR}_Lext{cfg.COLUMN_EXTENSION_LATERAL_KPC:g}kpc'
    # arithmetic-mean variant lives in its own subfolder (it is a sensitivity
    # test, kept separate from the canonical harmonic-mean results).
    save_dir = OUT_DIR / 'arithmetic_mean' if MEAN_METHOD == 'arithmetic' else OUT_DIR
    if MEAN_METHOD == 'arithmetic':
        Ltag += '_arith'
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"mask":<18} {"n_cells":>10} {"frac_count":>11} {"frac_mass":>10} {"colDen>max":>11}')
    print('-' * 64)
    for tag, m_nH_min, m_nH_max, m_T_min, m_T_max, _phase in MASKS:
        mask = base_ok & (n_H > m_nH_min) & (T_qk < m_T_max)
        if m_nH_max is not None:
            mask &= (n_H < m_nH_max)
        if m_T_min is not None:
            mask &= (T_qk > m_T_min)
        n_mask = int(mask.sum())
        if n_mask == 0:
            print(f'{tag:<18} {0:>10}  (empty — skipped)')
            continue
        n_oor_mask = int(np.sum(colden_oor & mask))   # masked cells over table colDen max

        # Mass weight ∝ ρ (cell volume dV is a global constant on the uniform
        # down=1 grid, so it cancels in every mass-weighted ratio/2D map).
        m  = rho[mask]
        x  = log_Tqk[mask]
        y  = log_Tdsp[mask]
        nh = log_NH[mask]

        H_mass,  _, _ = np.histogram2d(x, y, bins=[T_EDGES, T_EDGES], weights=m)
        H_count, _, _ = np.histogram2d(x, y, bins=[T_EDGES, T_EDGES])
        NH_mass,  _ = np.histogram(nh, bins=NH_EDGES, weights=m)
        NH_count, _ = np.histogram(nh, bins=NH_EDGES)

        hotter    = y > x
        frac_cnt  = float(np.mean(hotter))
        frac_mass = float(m[hotter].sum() / m.sum())
        print(f'{tag:<18} {n_mask:>10} {frac_cnt:>10.1%} {frac_mass:>10.1%} {n_oor_mask:>11}')

        npz = save_dir / f'representative_{Ltag}_{tag}.npz'
        np.savez(
            npz,
            T_edges=T_EDGES, NH_edges=NH_EDGES,
            H_mass=H_mass, H_count=H_count,
            NH_mass=NH_mass, NH_count=NH_count,
            frac_cnt=frac_cnt, frac_mass=frac_mass,
            n_mask=n_mask,
            downsample=cfg.DOWNSAMPLE_FACTOR,
            L_ext=cfg.COLUMN_EXTENSION_LATERAL_KPC,
            mask_tag=tag, n_H_min=m_nH_min, n_H_max=(np.nan if m_nH_max is None else m_nH_max),
            T_min_K=(np.nan if m_T_min is None else m_T_min), T_max_K=m_T_max,
        )
    print(f'\n[done] {Ltag}  ({time.time()-t0:.1f}s)   npz → {OUT_DIR}')


if __name__ == '__main__':
    main()
