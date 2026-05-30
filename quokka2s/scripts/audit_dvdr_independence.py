#!/usr/bin/env python
"""Sanity check: is Tg actually independent of dVdr (as solver's two-stage
optimization assumes)?

For each of N representative (nH, NH) cells, run a full setChemEq at EACH of
the 35 dVdr grid values, and compare the resulting Tg to the table's
broadcast value (which is Tg solved at canonical_dvdr = dVdr[17]).

If the two-stage optimization is valid, max |log10(Tg_full / Tg_table)| should
be tiny (< 0.01 dex).  If it's significant, the table is wrong along the
dVdr axis.
"""
from __future__ import annotations
import sys, time
import numpy as np
sys.path.insert(0, '/Users/baochen/quokka_postprocessing/quokka2s/src')
from quokka2s.tables.solver import calculate_single_despotic_point
from despotic.chemistry import GOW

TABLE_PATH = '/Users/baochen/quokka_postprocessing/output_tables_3D_GOW_LVG/despotic_table.npz'

tbl = np.load(TABLE_PATH, allow_pickle=True)
nH_grid = tbl['nH_values']
NH_grid = tbl['col_density_values']
dV_grid = tbl['dVdr_values']
Tg_tbl  = tbl['tg_final']

# 10 representative cells across the physics regimes.
# (log_nH_target, log_NH_target, label)
targets = [
    (-4, 15, 'WIM-like (low nH, low NH)'),
    (-1, 18, 'WNM diffuse'),
    ( 0, 20, 'CNM thin'),
    ( 1, 20, 'CNM mid'),
    ( 2, 21, 'CNM dense'),
    ( 3, 22, 'cold dense / molecular onset'),
    ( 4, 23, 'molecular cloud'),
    ( 5, 23, 'dense core'),
    ( 6, 24, 'very dense'),
    ( 0, 23, 'thin but high NH'),
]

print(f"GOW table @ {TABLE_PATH}")
print(f"dVdr axis: {dV_grid[0]:.2e} -> {dV_grid[-1]:.2e} s^-1, {len(dV_grid)} pts")
print(f"canonical_dvdr (used by table): {dV_grid[len(dV_grid)//2]:.2e}")
print()

results = []
t0 = time.time()
for (lognH, logNH, label) in targets:
    i = int(np.argmin(np.abs(np.log10(nH_grid) - lognH)))
    j = int(np.argmin(np.abs(np.log10(NH_grid) - logNH)))
    nH_val = float(nH_grid[i]); NH_val = float(NH_grid[j])
    Tg_table = float(Tg_tbl[i, j, 0])   # broadcast — any k works

    print(f"\n--- {label}: nH={nH_val:.2e}, NH={NH_val:.2e} (i={i}, j={j}) ---")
    print(f"  table broadcast Tg  = {Tg_table:.4f} K  (solved at canonical dVdr)")

    Tg_per_dvdr = []
    for k, dv in enumerate(dV_grid):
        # Single-element dvdr_grid -> canonical_dvdr is dv itself, so setChemEq
        # runs at THIS dVdr.  Real (nH, NH, dVdr) 3-input solve.
        result = calculate_single_despotic_point(
            nH_val, NH_val, [float(dv)],
            chem_network=GOW,
            escape_geom='LVG',
        )
        Tg_per_dvdr.append(float(result[6]))

    Tg_arr = np.array(Tg_per_dvdr)
    Tg_min, Tg_max = float(Tg_arr.min()), float(Tg_arr.max())
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_log = np.log10(Tg_arr / Tg_table)
    abs_max_dex = float(np.max(np.abs(rel_log)))
    print(f"  full-solve Tg @ all 35 dVdr: min={Tg_min:.4f}, max={Tg_max:.4f} K")
    print(f"  spread along dVdr: {Tg_max - Tg_min:.4f} K  "
          f"(log10(max/min) = {np.log10(Tg_max/Tg_min):.4f} dex)")
    print(f"  max |log10(full / broadcast)| = {abs_max_dex:.4f} dex")
    # Endpoints
    print(f"  Tg @ dVdr_min ({dV_grid[0]:.2e}): {Tg_arr[0]:.4f}")
    print(f"  Tg @ dVdr_mid ({dV_grid[len(dV_grid)//2]:.2e}): {Tg_arr[len(dV_grid)//2]:.4f}")
    print(f"  Tg @ dVdr_max ({dV_grid[-1]:.2e}): {Tg_arr[-1]:.4f}")

    results.append({
        'label': label,
        'nH': nH_val, 'NH': NH_val,
        'Tg_table': Tg_table,
        'Tg_per_dvdr': Tg_arr,
        'spread_dex': float(np.log10(Tg_max/Tg_min)) if Tg_min > 0 else float('nan'),
        'broadcast_err_max_dex': abs_max_dex,
    })

print(f"\n\n=== SUMMARY (wall time {time.time()-t0:.0f}s) ===")
print(f"{'cell':40s}  {'Tg_table':>10s}  {'spread':>10s}  {'broadcast_err_max':>20s}")
for r in results:
    print(f"  {r['label']:38s}  {r['Tg_table']:>10.2f}  "
          f"{r['spread_dex']:>10.4f}  {r['broadcast_err_max_dex']:>20.4f}")

# Save for plotting
np.savez('/Users/baochen/quokka_postprocessing/output/three_network_compare/dvdr_audit.npz',
         dV_grid=dV_grid,
         **{r['label'].replace(' ', '_').replace('/', '_'): r['Tg_per_dvdr']
            for r in results},
         labels=np.array([r['label'] for r in results]))
print(f"\nsaved -> output/three_network_compare/dvdr_audit.npz")
