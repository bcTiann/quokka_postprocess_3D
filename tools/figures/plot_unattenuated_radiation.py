#!/usr/bin/env python3
"""Draw saved unattenuated Cloudy-native HM2012, Black/ISM, and their sum.

Run prepare_radiation_fields.py --recipe unattenuated first. This command
reads the prepared NPZ; it does not parse exports or calculate radiation sums.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quokka2s.figures.radiation_fields import add_eight_ev_marker
from quokka2s.products.radiation_fields import (
    UNATTENUATED_CMB_RADIATION_STEM,
    UNATTENUATED_RADIATION_STEM,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        help="Prepared radiation NPZ; defaults to the selected figure's output stem",
    )
    parser.add_argument(
        "--include-cmb",
        action="store_true",
        help="Select the default HM2012+CMB input filename; labels follow the saved recipe",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "output/radiation_fields",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    default_stem = UNATTENUATED_CMB_RADIATION_STEM if args.include_cmb else UNATTENUATED_RADIATION_STEM
    data_path = args.data or output_dir / (default_stem + ".npz")
    with np.load(data_path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    energy = payload["energy_Ryd"]
    include_cmb = bool(payload["include_cmb"])
    stem = UNATTENUATED_CMB_RADIATION_STEM if include_cmb else UNATTENUATED_RADIATION_STEM
    x_min, x_max = payload["x_limits_Ryd"]
    y_min, y_max = payload["intensity_limits_erg_cm2_s"]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.loglog(
        energy,
        payload["hm2012_plus_ism_positive"],
        color="black",
        linewidth=2.4,
        label="HM2012 + CMB + ISM" if include_cmb else "HM2012 + ISM",
        zorder=1,
    )
    ax.loglog(
        energy,
        payload["hm2012_with_optional_cmb_positive"],
        color="#0072B2",
        linestyle="--",
        linewidth=2.0,
        label=(
            "Cloudy-native unattenuated HM2012 + CMB"
            if include_cmb
            else "Cloudy-native unattenuated HM2012"
        ),
        zorder=3,
    )
    ax.loglog(
        energy,
        payload["ism_unattenuated_positive"],
        color="#D55E00",
        linestyle="--",
        linewidth=2.0,
        label="Cloudy table ISM unattenuated",
        zorder=4,
    )
    add_eight_ev_marker(
        axis=ax,
        marker_Ryd=payload["eight_ev_marker_Ryd"],
        label_line=True,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Photon energy [Ryd]")
    ax.set_ylabel(r"$\nu\,4\pi J_\nu$ [erg cm$^{-2}$ s$^{-1}$]")
    ax.set_title("Unattenuated incident radiation fields")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.17)
    fig.tight_layout()

    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(pdf_path)


if __name__ == "__main__":
    main()
