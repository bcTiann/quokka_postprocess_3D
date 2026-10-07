#!/usr/bin/env python3
"""Draw saved quick-extinguished HM2012 and fixed filtered-ISM components.

Run prepare_radiation_fields.py --recipe components first. This command reads
the prepared NPZ; it does not read Cloudy exports or calculate continuum sums.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quokka2s.figures.radiation_fields import add_eight_ev_marker
from quokka2s.paths import resolve_path
from quokka2s.products.radiation_fields import (
    COMBINED_RADIATION_STEM,
    COMPONENT_RADIATION_STEM,
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
        "--output-dir",
        type=Path,
        default=project_root / "output/radiation_fields",
    )
    parser.add_argument(
        "--components-only",
        action="store_true",
        help="Plot only the HM2012 and attenuated-ISM component panel.",
    )
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = COMPONENT_RADIATION_STEM if args.components_only else COMBINED_RADIATION_STEM
    data_path = resolve_path(args.data or output_dir / (stem + ".npz"))
    with np.load(data_path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    energy = payload["energy_Ryd"]
    column_labels = payload["hm12_column_labels"]
    x_min, x_max = payload["x_limits_Ryd"]
    y_min, y_max = payload["intensity_limits_erg_cm2_s"]

    colors = {
        0: "#6A3D9A",
        19: "#0072B2",
        20: "#56B4E9",
        21: "#009E73",
        22: "#E69F00",
        23: "#D55E00",
    }
    linestyles = {
        0: "-",
        19: "--",
        20: "-.",
        21: ":",
        22: (0, (6, 2)),
        23: (0, (2, 1)),
    }
    markers = {0: "o", 19: "s", 20: "^", 21: "D", 22: "v", 23: "P"}

    if args.components_only:
        fig, component_ax = plt.subplots(figsize=(9.4, 4.0))
        axes = (component_ax,)
        absolute_ax = None
    else:
        fig, axes_array = plt.subplots(
            2,
            1,
            figsize=(9.4, 8.0),
            sharex=True,
            gridspec_kw={"height_ratios": (1.0, 1.0)},
        )
        absolute_ax, component_ax = axes_array
        axes = tuple(axes_array)
        for marker_offset, label in enumerate(column_labels):
            absolute_ax.loglog(
                energy,
                payload[f"combined_{label}HM_positive"],
                color=colors[label],
                linestyle=linestyles[label],
                linewidth=2.3,
                marker=markers[label],
                markersize=3.5,
                markevery=(marker_offset * 24, 180),
                label=f"combined {label}HM",
            )

    component_ax.loglog(
        energy,
        payload["ism_extinguish_nh21_positive"],
        color="black",
        linestyle="--",
        linewidth=2.2,
        label="attenuated ISM",
    )
    for label in column_labels:
        component_ax.loglog(
            energy,
            payload[f"hm12_nh{label}_positive"],
            color=colors[label],
            linestyle="-" if args.components_only else linestyles[label],
            linewidth=1.8,
            label=f"{label}HM",
        )

    for axis in axes:
        add_eight_ev_marker(
            axis=axis,
            marker_Ryd=payload["eight_ev_marker_Ryd"],
            label_line=args.components_only or axis is absolute_ax,
        )
        axis.set_xlim(x_min, x_max)
        axis.grid(True, which="both", alpha=0.17)
    if absolute_ax is not None:
        absolute_ax.set_ylim(y_min, y_max)
        absolute_ax.set_ylabel(r"$\nu\,4\pi J_\nu$ [erg cm$^{-2}$ s$^{-1}$]")
        absolute_ax.set_title("Combined continuum")
        absolute_ax.legend(frameon=False, fontsize=7.8, ncol=2, loc="upper center")
        absolute_ax.text(
            0.99,
            0.02,
            "All curves include attenuated ISM",
            transform=absolute_ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.35",
        )
    component_ax.set_ylim(y_min, y_max)
    component_ax.set_xlabel("Photon energy [Ryd]")
    component_ax.set_ylabel(r"$\nu\,4\pi J_\nu$ [erg cm$^{-2}$ s$^{-1}$]")
    component_ax.set_title("HM2012 and ISM components")
    component_ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="upper center")
    fig.tight_layout()

    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(pdf_path)


if __name__ == "__main__":
    main()
