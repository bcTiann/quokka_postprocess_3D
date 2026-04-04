#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from quokka2s.tables import load_table
from quokka2s.tables.plotting import plot_table_overview
from quokka2s.pipeline.prep import config as cfg


def main():
    table = load_table(cfg.DESPOTIC_TABLE_PATH)
    samples = np.load("/Users/baochen/quokka_postprocessing/log_samples.npy")

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

    figs = plot_table_overview(
        table,
        fields=tokens,
        ncols=3,
        figsize=(14, 10),
        separate=True,  # 默认每个字段单独一张图
        samples=samples,
    )
    out_dir = Path("plots/table_overview")
    out_dir.mkdir(parents=True, exist_ok=True)
    for token, fig in zip(tokens, figs):
        fname = token.replace(":", "_") + ".png"
        fig.savefig(out_dir / fname, dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
