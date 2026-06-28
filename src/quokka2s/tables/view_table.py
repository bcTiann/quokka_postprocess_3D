#!/usr/bin/env python3
"""Generate 2D (nH × N_H) heatmaps of the DESPOTIC table, one set per dVdr bin.

Table is now 3D (nH, NH, dVdr) — T is an output, not an input axis. We slice
along dVdr and plot (nH, NH) heatmaps of tg_final, abundances, and lumPerH.

By default plots a few evenly-spaced dVdr slices to keep the output count
manageable. Pass --all to dump every dVdr index (35 slices × 10 fields = 350
PNGs)."""
from pathlib import Path
import os
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
    ap.add_argument('-o', '--out-root', default=None,
                    help='output root directory; if omitted, auto-derived from '
                         'the table\'s parent dir as TablePlots_<basename> '
                         '(strips leading "output_tables_3D_" if present)')
    ap.add_argument('--table', default=None,
                    help='path to the despotic_table.npz to view '
                         '(default: cfg.DESPOTIC_TABLE_PATH)')
    args = ap.parse_args()

    table_path = args.table or cfg.DESPOTIC_TABLE_PATH
    print(f"[view_table] loading: {table_path}")
    table = load_table(table_path)

    # Auto-derive output dir from the table source so different tables never
    # silently land in the same generic TablePlots/ bin.
    if args.out_root is None:
        src_dir_name = Path(table_path).parent.name
        # strip the boilerplate prefix if present, so the suffix carries the
        # network/geom info (e.g. output_tables_3D_GOW_LVG -> GOW_LVG).
        tag = src_dir_name
        for prefix in ('output_tables_3D_', 'output_tables_4D_', 'output_tables_'):
            if tag.startswith(prefix):
                tag = tag[len(prefix):]
                break
        args.out_root = f'TablePlots_{tag}'
        print(f"[view_table] -o auto-derived: {args.out_root}/")

    # Prefer the 3D samples (per-dVdr slice filter via plotting.py).  Fall back
    # to 2D file (overlay is then identical on every slice).
    samples = None
    _repo = Path(__file__).resolve().parents[3]   # src/quokka2s/tables/view_table.py → repo
    for sp in (os.environ.get("LOG_SAMPLES_3D", str(_repo / "log_samples_3d.npy")),
               os.environ.get("LOG_SAMPLES_2D", str(_repo / "log_samples.npy"))):
        if Path(sp).exists():
            samples = np.load(sp)
            print(f"[view_table] samples loaded from {sp}  shape={samples.shape}")
            break
    if samples is None:
        print("Warning: no log_samples file found. Skipping sampling overlay.")

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
            # Append dVdr label to each subplot's title so each frame
            # carries the dVdr value when scrubbing through an MP4 sweep.
            for ax in fig.axes:
                old_title = ax.get_title()
                if old_title:                       # skip colorbar axes etc.
                    ax.set_title(f"{old_title}  |  frame {d_idx+1:02d}/{n_dvdr}  "
                                 f"dVdr = {d_val:.2e} s$^{{-1}}$")
            fname = token.replace(":", "_") + ".png"
            fig.savefig(out_dir / fname, dpi=200)
            plt.close(fig)
        print(f"  done dVdr = {d_val:.2e} s^-1  →  {out_dir}/")


if __name__ == "__main__":
    main()
