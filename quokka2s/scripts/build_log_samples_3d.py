#!/usr/bin/env python
"""Build a 3D (log nH, log NH, log dVdr) sample file from cached field
intermediates so TablePlots can overlay only the cells whose dVdr falls in
each table slice's bin.

Pulls n_H, N_H, dVdr from `intermediates/plt0655228/fields/*.h5`.  No new
field computation — uses whatever L_ext is currently in the cache.  Writes
to `log_samples_3d.npy` next to the existing 2D `log_samples.npy`.

Usage:
    python quokka2s/scripts/build_log_samples_3d.py [--out PATH]
"""
from __future__ import annotations
import sys, argparse, time
from pathlib import Path

import numpy as np
import h5py
import astropy.constants as const
import astropy.units as u

# Canonical H mass fraction from config (single source of truth, so n_H here
# matches the pipeline); m_H derived from astropy rather than hardcoded.
sys.path.insert(0, '/Users/baochen/quokka_postprocessing/quokka2s/src')
from quokka2s.pipeline.prep import config as cfg

CACHE = Path('/Users/baochen/quokka_postprocessing/intermediates/plt0655228/fields')
PLT   = Path('/Users/baochen/quokka_postprocessing/plt0655228')

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out', default='/Users/baochen/quokka_postprocessing/log_samples_3d.npy',
                help='destination .npy (shape N x 3 by default, N x 4 with --include-mass)')
ap.add_argument('--stride', type=int, default=1,
                help='stride to subsample cells (default 1 = use all); use 4-8 to slim down')
ap.add_argument('--include-mass', action='store_true',
                help='also emit per-cell mass [g] as a 4th column for mass-weighted contours')
args = ap.parse_args()

t0 = time.time()
print('[load] column_density_H ...', flush=True)
with h5py.File(CACHE / 'field_gas_column_density_H.h5', 'r') as f:
    NH = np.asarray(f['data'])
    key_NH = str(f.attrs.get('cache_key', 'none'))[:16]
print(f'  shape={NH.shape}  key={key_NH}', flush=True)

print('[load] dVdr_lvg ...', flush=True)
with h5py.File(CACHE / 'field_gas_dVdr_lvg.h5', 'r') as f:
    dV = np.asarray(f['data'])
    key_dV = str(f.attrs.get('cache_key', 'none'))[:16]
print(f'  shape={dV.shape}  key={key_dV}', flush=True)

# n_H = density * X_H / m_H — compute from snapshot density.
# 2026-06-11 FIX: previous version used `ds.all_data().reshape(nx, ny, nz)`
# which gives a DIFFERENT cell-ordering than the cached covering_grid that
# NH and dVdr came from.  That misaligned per-cell (nH, NH, dVdr, mass)
# pairings by up to 7+ dex in log_nH, scrambling every downstream contour /
# aggregate plot built off the samples.  Verified by direct comparison
# against the same provider that originally wrote field_gas_*.h5.  Switch
# to ds.covering_grid() which DOES match the cached order.
print('[compute] n_H from covering_grid density ...', flush=True)
import yt
ds = yt.load(str(PLT))
nx, ny, nz = NH.shape
cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
                      dims=ds.domain_dimensions)
rho_3d = cg[('gas', 'density')].in_cgs().value
print(f'  shape={rho_3d.shape}', flush=True)

_m_H_q = const.m_p.to(u.g)
assert _m_H_q.unit == u.g
X_H = cfg.X_H                 # canonical H mass fraction (config)
m_H = float(_m_H_q.value)     # g   (proton ≈ H nucleus)
nH = rho_3d * X_H / m_H
print(f'  nH range: {nH.min():.2e} – {nH.max():.2e}', flush=True)

# Per-cell mass (g) — only built if requested.  dV is uniform at down=1.
if args.include_mass:
    dx = float(cg[('boxlib', 'dx')].in_cgs().value.flat[0])
    dy = float(cg[('boxlib', 'dy')].in_cgs().value.flat[0])
    dz = float(cg[('boxlib', 'dz')].in_cgs().value.flat[0])
    dV_cell = dx * dy * dz   # cm^3
    mass_3d = rho_3d * dV_cell
    print(f'  cell dV = {dV_cell:.3e} cm^3,  mass range: {mass_3d.min():.2e} – {mass_3d.max():.2e} g',
          flush=True)
else:
    mass_3d = None

# Stack flat samples
if args.stride > 1:
    s = args.stride
    nH = nH[::s, ::s, ::s]
    NH = NH[::s, ::s, ::s]
    dV = dV[::s, ::s, ::s]
    if mass_3d is not None:
        mass_3d = mass_3d[::s, ::s, ::s]

# Convert to log10, mask non-positive
with np.errstate(divide='ignore', invalid='ignore'):
    lognH = np.log10(np.where(nH > 0, nH, np.nan)).ravel()
    logNH = np.log10(np.where(NH > 0, NH, np.nan)).ravel()
    logdV = np.log10(np.where(dV > 0, dV, np.nan)).ravel()

ok = np.isfinite(lognH) & np.isfinite(logNH) & np.isfinite(logdV)
if args.include_mass:
    mass_flat = mass_3d.ravel()
    ok = ok & np.isfinite(mass_flat) & (mass_flat > 0)
    samples = np.column_stack([lognH[ok], logNH[ok], logdV[ok], mass_flat[ok]])
else:
    samples = np.column_stack([lognH[ok], logNH[ok], logdV[ok]])
print(f'\nsample stats:')
print(f'  finite cells: {ok.sum():,} / {ok.size:,}')
print(f'  log nH:   {samples[:,0].min():.2f} … {samples[:,0].max():.2f}')
print(f'  log NH:   {samples[:,1].min():.2f} … {samples[:,1].max():.2f}')
print(f'  log dVdr: {samples[:,2].min():.2f} … {samples[:,2].max():.2f}')
if args.include_mass:
    print(f'  mass (g): {samples[:,3].min():.2e} … {samples[:,3].max():.2e}')

np.save(args.out, samples)
print(f'\n[saved] {args.out}  shape={samples.shape}  '
      f'({samples.nbytes/1024**2:.0f} MB)  elapsed {time.time()-t0:.1f}s')
