#!/usr/bin/env python3
"""Plot quick-extinguished HM2012 plus a fixed quick-extinguished ISM field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EV_PER_RYD = 13.605693122994
EIGHT_EV_RYD = 8.0 / EV_PER_RYD
RADIATION_X_MIN_EV = 7.0
RADIATION_X_MIN_RYD = RADIATION_X_MIN_EV / EV_PER_RYD
RADIATION_X_MAX_RYD = 1.0e3
RADIATION_Y_DYNAMIC_RANGE_DEX = 8.0
HM12_COLUMN_LABELS = (0, 19, 20, 21, 22, 23)


def load_incident(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#", usecols=(0, 1))
    if data.ndim != 2 or np.any(np.diff(data[:, 0]) <= 0.0):
        raise ValueError(f"unexpected Cloudy incident continuum: {path}")
    return data


def positive(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.0, values, np.nan)


def add_eight_ev_marker(axis: plt.Axes, *, label_line: bool = False) -> None:
    axis.axvline(EIGHT_EV_RYD, color="0.35", linestyle=":", linewidth=1.2)
    if label_line:
        axis.text(
            EIGHT_EV_RYD,
            0.04,
            "8 eV",
            rotation=90,
            transform=axis.get_xaxis_transform(),
            ha="right",
            va="bottom",
            color="0.35",
            fontsize=8,
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_data_dir = (
        project_root
        / "work/cloudy_cooling_tools_history/examples/grackle"
        / "HM12_NATIVE_ISM_NH21_CMB"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
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

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ism_path = data_dir / "export_ism_filtered.inc"
    ism_data = load_incident(ism_path)
    energy = ism_data[:, 0]
    ism_nh21 = ism_data[:, 1]

    hm12: dict[int, np.ndarray] = {}
    hm12_paths: dict[int, Path] = {}
    for label in HM12_COLUMN_LABELS:
        path = (
            data_dir / "export_hm12_native.inc"
            if label == 0
            else data_dir / f"export_hm12_extinguished_nh{label}.inc"
        )
        data = load_incident(path)
        if not np.array_equal(energy, data[:, 0]):
            raise ValueError(f"Cloudy energy mesh differs for {path}")
        hm12_paths[label] = path
        hm12[label] = data[:, 1]

    combined = {
        label: ism_nh21 + hm12_values
        for label, hm12_values in hm12.items()
    }
    visible = (energy >= RADIATION_X_MIN_RYD) & (
        energy <= RADIATION_X_MAX_RYD
    )
    visible_maximum = max(
        float(values[visible].max()) for values in combined.values()
    )
    y_max = 10.0 ** np.ceil(np.log10(visible_maximum))
    y_min = y_max / 10.0**RADIATION_Y_DYNAMIC_RANGE_DEX

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
        for marker_offset, label in enumerate(HM12_COLUMN_LABELS):
            absolute_ax.loglog(
                energy,
                positive(combined[label]),
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
        positive(ism_nh21),
        color="black",
        linestyle="--",
        linewidth=2.2,
        label="attenuated ISM",
    )
    for label in HM12_COLUMN_LABELS:
        component_ax.loglog(
            energy,
            positive(hm12[label]),
            color=colors[label],
            linestyle="-" if args.components_only else linestyles[label],
            linewidth=1.8,
            label=f"{label}HM",
        )

    for axis in axes:
        add_eight_ev_marker(
            axis,
            label_line=args.components_only or axis is absolute_ax,
        )
        axis.set_xlim(RADIATION_X_MIN_RYD, RADIATION_X_MAX_RYD)
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

    stem = (
        "cloudy_HM2012_NH0_19_23_and_attenuated_ISM_NH21_components"
        if args.components_only
        else "cloudy_combined_and_components_HM2012_NH0_19_23_ISM_NH21"
    )
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    arrays: dict[str, np.ndarray] = {
        "energy_Ryd": energy,
        "ism_extinguish_nh21": ism_nh21,
    }
    for label in HM12_COLUMN_LABELS:
        arrays[f"hm12_nh{label}"] = hm12[label]
        arrays[f"combined_{label}HM"] = combined[label]
    np.savez_compressed(npz_path, **arrays)

    report = {
        "definition": (
            "combined 0HM = extinguished ISM NH=1e21 + unattenuated HM12; "
            "combined kHM = extinguished ISM NH=1e21 + HM12 after "
            "extinguish column=k leak=0"
        ),
        "hm12_column_labels": list(HM12_COLUMN_LABELS),
        "ism_log_column_density_cm-2": 21,
        "attenuation_method": "Cloudy extinguish command (quick-test prescription)",
        "energy_marker": {"eV": 8.0, "Ryd": EIGHT_EV_RYD},
        "inputs": {
            "ism_nh21": str(ism_path),
            "hm12": {str(key): str(value) for key, value in hm12_paths.items()},
        },
        "plot_range": {
            "x_min_eV": RADIATION_X_MIN_EV,
            "x_min_Ryd": RADIATION_X_MIN_RYD,
            "x_max_Ryd": RADIATION_X_MAX_RYD,
            "y_max": y_max,
            "y_min": y_min,
            "y_dynamic_range_dex": RADIATION_Y_DYNAMIC_RANGE_DEX,
        },
        "outputs": {
            "plot_png": str(png_path),
            "plot_pdf": str(pdf_path),
            "npz": str(npz_path),
        },
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
