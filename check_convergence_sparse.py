"""Stress-test DESPOTIC convergence on a sparse grid (subset of the full 35³).

Uses identical inputs to build_table.py (same nH/NH/dVdr ranges, same chemistry
network, same species), but with fewer points per axis so the run finishes in
minutes rather than hours. Reports how many cells failed to converge and
where they cluster in (nH, NH, dVdr) space.

Does NOT save the table — output is summary only.

Usage:
    python check_convergence_sparse.py                  # default 8 points/axis = 512 cells
    python check_convergence_sparse.py --points 10       # 1000 cells
    python check_convergence_sparse.py --points 6 -j 4   # 216 cells, 4 workers
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "quokka2s/src")

from despotic.chemistry import GOW

from quokka2s.tables import LogGrid, build_table
from quokka2s.tables.builder import SpeciesSpec


# ── Match build_table.py exactly ──────────────────────────────────────────────
N_H_RANGE     = (1e-4,  1e6)
COL_DEN_RANGE = (1e15,  1e24)
DVDR_RANGE    = (1e-19, 1e-12)

SPECIES_SPECS = (
    SpeciesSpec("CO",   True),
    SpeciesSpec("C",    True),
    SpeciesSpec("C+",   True),
    SpeciesSpec("HCO+", True),
    SpeciesSpec("O",    True),
    SpeciesSpec("e-",   False),
    SpeciesSpec("H+",   False),
    SpeciesSpec("H2",   False),
    SpeciesSpec("H",    False),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=int, default=8,
                    help="grid points per axis (default 8 → %d cells)" % (8 ** 3))
    ap.add_argument("-j", "--workers", type=int, default=-1,
                    help="joblib workers (-1 = all cores; default -1)")
    args = ap.parse_args()

    n = args.points
    total = n ** 3
    print(f"Sparse convergence test: {n}^3 = {total} cells "
          f"(full table is 35^3 = {35**3} = {35**3/total:.1f}× bigger)")
    print(f"  nH    range: {N_H_RANGE[0]:.0e} .. {N_H_RANGE[1]:.0e}")
    print(f"  NH    range: {COL_DEN_RANGE[0]:.0e} .. {COL_DEN_RANGE[1]:.0e}")
    print(f"  dVdr  range: {DVDR_RANGE[0]:.0e} .. {DVDR_RANGE[1]:.0e}")
    print(f"  species:   {[s.name for s in SPECIES_SPECS]}")
    print(f"  workers:   {args.workers}")
    print()

    nH_grid   = LogGrid(*N_H_RANGE,     num_points=n)
    col_grid  = LogGrid(*COL_DEN_RANGE, num_points=n)
    dVdr_grid = LogGrid(*DVDR_RANGE,    num_points=n)

    t0 = time.perf_counter()
    table = build_table(
        nH_grid, col_grid, dVdr_grid,
        species_specs=SPECIES_SPECS,
        show_progress=True,
        chem_network=GOW,
        full_parallel=False,
        workers=args.workers,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nBuild time: {elapsed/60:.2f} min  ({elapsed/total:.2f} s/cell average)")

    fm = table.failure_mask
    if fm is None:
        print("\nfailure_mask is None — table built without convergence tracking")
        return

    n_fail = int(fm.sum())
    print(f"\n=== CONVERGENCE SUMMARY ===")
    print(f"Non-converged: {n_fail} / {total} = {100*n_fail/total:.3f}%")

    if n_fail == 0:
        print("✓ All cells converged on this sparse grid.")
        return

    # Where do failures cluster?
    nH_vals  = np.asarray(table.nH_values)
    NH_vals  = np.asarray(table.col_density_values)
    dV_vals  = np.asarray(table.dVdr_values)

    print("\nFailure rate vs nH (averaged over NH, dVdr):")
    rate_nH = fm.mean(axis=(1, 2))
    for i in np.argsort(rate_nH)[::-1][:5]:
        if rate_nH[i] > 0:
            print(f"  nH={nH_vals[i]:.2e}   fail_rate={100*rate_nH[i]:.1f}%")

    print("\nFailure rate vs NH (averaged over nH, dVdr):")
    rate_NH = fm.mean(axis=(0, 2))
    for i in np.argsort(rate_NH)[::-1][:5]:
        if rate_NH[i] > 0:
            print(f"  NH={NH_vals[i]:.2e}   fail_rate={100*rate_NH[i]:.1f}%")

    print("\nFailure rate vs dVdr (averaged over nH, NH):")
    rate_dV = fm.mean(axis=(0, 1))
    for i in np.argsort(rate_dV)[::-1][:5]:
        if rate_dV[i] > 0:
            print(f"  dVdr={dV_vals[i]:.2e}   fail_rate={100*rate_dV[i]:.1f}%")

    # T of failed cells
    bad_T = table.tg_final[fm]
    print(f"\nT of failed cells (n={bad_T.size}):")
    print(f"  min    = {np.nanmin(bad_T):.2e} K")
    print(f"  median = {np.nanmedian(bad_T):.2e} K")
    print(f"  max    = {np.nanmax(bad_T):.2e} K")

    # Concrete (nH, NH, dVdr) coordinates of first few failures
    print("\nFirst 10 failed cells (nH, NH, dVdr, final_T):")
    fail_coords = np.argwhere(fm)
    for r, c, d in fail_coords[:10]:
        T_here = table.tg_final[r, c, d]
        print(f"  nH={nH_vals[r]:.2e}  NH={NH_vals[c]:.2e}  "
              f"dVdr={dV_vals[d]:.2e}  T_final={T_here:.2e} K")


if __name__ == "__main__":
    main()
