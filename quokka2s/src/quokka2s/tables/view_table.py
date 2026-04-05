#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from quokka2s.tables import load_table
from quokka2s.tables.plotting import plot_table_overview
from quokka2s.pipeline.prep import config as cfg


def main():
    table = load_table(cfg.DESPOTIC_TABLE_PATH)
    samples_path = "/Users/baochen/quokka_postprocessing/log_samples.npy"
    if Path(samples_path).exists():
        samples = np.load(samples_path)
    else:
        print(f"Warning: {samples_path} not found. Skipping sampling overlay.")
        samples = None

    tokens = [
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

    # 为每个温度生成图
    for t_idx, t_val in enumerate(table.T_values):
        figs = plot_table_overview(
            table,
            fields=tokens,
            ncols=3,
            figsize=(14, 10),
            separate=True,
            samples=samples,
            t_idx=t_idx,
        )
        out_dir = Path(f"TablePlots/table_overview_T_{t_val:.0f}K")
        out_dir.mkdir(parents=True, exist_ok=True)
        for token, fig in zip(tokens, figs):
            fname = token.replace(":", "_") + ".png"
            fig.savefig(out_dir / fname, dpi=200)
            plt.close(fig)
        print(f"Generated plots for T = {t_val:.0f} K")


if __name__ == "__main__":
    main()
