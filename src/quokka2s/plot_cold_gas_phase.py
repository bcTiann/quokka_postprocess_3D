"""Draw saved cold nH–NH histograms without opening snapshots or tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from quokka2s.figures.cold_gas_phase import plot_cold_gas_phase


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, default=Path("output/cold_gas_phases"))
    parser.add_argument("--weight", choices=("count", "mass"), default="count")
    args = parser.parse_args(argv)
    root = args.products.expanduser().resolve()
    paths = sorted((root / "snapshots").glob("*/*/prepared.npz"))
    paths.extend(sorted((root / "jobs").glob("*/prepared.npz")))
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            plot_cold_gas_phase(
                prepared=saved,
                stem=path.parent / "figures" / ("cold_nH_NH_" + args.weight),
                weight=args.weight,
            )
        print(f"Saved cold phase figure: {path.parent}", flush=True)


if __name__ == "__main__":
    main()
