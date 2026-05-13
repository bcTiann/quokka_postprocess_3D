#!/usr/bin/env python3
"""Generate 2D (nH × N_H) heatmaps of the DESPOTIC table, one set per dVdr bin.

Table is now 3D (nH, NH, dVdr) — T is an output, not an input axis. We slice
along dVdr and plot (nH, NH) heatmaps of tg_final, abundances, and lumPerH.

By default plots a few evenly-spaced dVdr slices to keep the output count
manageable. Pass --all to dump every dVdr index (35 slices × 10 fields = 350
PNGs)."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

from . import load_table
from .plotting import plot_table_overview
from ..pipeline.prep import config as cfg


TOKENS = [
    "tg_final",
    "species:CO:abundance",
    "species:C+:abundance",
    "species:C:abundance",
    "species:HCO+:abundance",
    "species:e-:abundance",
    "species:CO:lumPerH",
    "species:C+:lumPerH",
    "species:C:lumPerH",
    "species:HCO+:lumPerH",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--all', action='store_true',
                    help='plot every dVdr index (default: 5 evenly-spaced)')
    ap.add_argument('-n', '--n-slices', type=int, default=5,
                    help='number of evenly-spaced dVdr slices (default 5)')
    ap.add_argument('-o', '--out-root', default='TablePlots',
                    help='output root directory (default: TablePlots/)')
    args = ap.parse_args()

    table = load_table(cfg.DESPOTIC_TABLE_PATH)

    samples_path = "/Users/baochen/quokka_postprocessing/log_samples.npy"
    if Path(samples_path).exists():
        samples = np.load(samples_path)
    else:
        print(f"Warning: {samples_path} not found. Skipping sampling overlay.")
        samples = None

    n_dvdr = len(table.dVdr_values)
    if args.all:
        indices = list(range(n_dvdr))
    else:
        # n_slices evenly spaced through the dVdr axis (include endpoints)
        indices = sorted(set(
            np.linspace(0, n_dvdr - 1, args.n_slices).astype(int).tolist()
        ))

    print(f"Plotting {len(TOKENS)} fields × {len(indices)} dVdr slices "
          f"= {len(TOKENS) * len(indices)} PNGs into {args.out_root}/")

    for d_idx in indices:
        d_val = table.dVdr_values[d_idx]
        figs = plot_table_overview(
            table,
            fields=TOKENS,
            ncols=3,
            figsize=(14, 10),
            separate=True,
            samples=samples,
            dvdr_idx=d_idx,
        )
        out_dir = Path(f"{args.out_root}/dVdr_{d_val:.2e}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for token, fig in zip(TOKENS, figs):
            fname = token.replace(":", "_") + ".png"
            fig.savefig(out_dir / fname, dpi=200)
            plt.close(fig)
        print(f"  done dVdr = {d_val:.2e} s^-1  →  {out_dir}/")


if __name__ == "__main__":
    main()
