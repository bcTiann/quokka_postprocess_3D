#!/usr/bin/env python3
"""Compare Cloudy-native unattenuated HM2012 with and without the z=0 CMB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EV_PER_RYD = 13.605693122994
HIGH_ENERGY_MIN_RYD = 7.0 / EV_PER_RYD
HIGH_ENERGY_MAX_RYD = 1.0e3


def load_incident(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#", usecols=(0, 1))
    if data.ndim != 2 or np.any(np.diff(data[:, 0]) <= 0.0):
        raise ValueError(f"unexpected Cloudy incident continuum: {path}")
    return data


def positive(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.0, values, np.nan)


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
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hm12_path = data_dir / "export_hm12_native.inc"
    cmb_path = data_dir / "export_cmb_z0.inc"
    combined_path = data_dir / "export_hm12_native_plus_cmb.inc"
    hm12_data = load_incident(hm12_path)
    cmb_data = load_incident(cmb_path)
    combined_data = load_incident(combined_path)
    if not (
        np.array_equal(hm12_data[:, 0], cmb_data[:, 0])
        and np.array_equal(hm12_data[:, 0], combined_data[:, 0])
    ):
        raise ValueError("HM2012, CMB, and combined Cloudy energy meshes differ")

    energy = hm12_data[:, 0]
    hm12 = hm12_data[:, 1]
    cmb = cmb_data[:, 1]
    combined = combined_data[:, 1]
    independently_summed = hm12 + cmb
    positive_combined = combined > 0.0
    maximum_sum_error_dex = float(
        np.max(
            np.abs(
                np.log10(combined[positive_combined])
                - np.log10(independently_summed[positive_combined])
            )
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    styles = (
        (combined, "black", "-", 2.5, "HM2012 + CMB", 1),
        (hm12, "#0072B2", "--", 2.0, "HM2012", 3),
        (cmb, "#009E73", ":", 2.2, "CMB", 4),
    )
    for ax in axes:
        for values, color, linestyle, linewidth, label, zorder in styles:
            ax.loglog(
                energy,
                positive(values),
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label,
                zorder=zorder,
            )
        ax.set_xlabel("Photon energy [Ryd]")
        ax.grid(True, which="both", alpha=0.17)

    full_positive = combined > 0.0
    full_x_min = float(energy[full_positive].min())
    full_x_max = HIGH_ENERGY_MAX_RYD
    full_visible = full_positive & (energy <= full_x_max)
    full_y_max = 10.0 ** np.ceil(np.log10(combined[full_visible].max()))
    axes[0].set_xlim(full_x_min, full_x_max)
    axes[0].set_ylim(full_y_max / 1.0e10, full_y_max)
    axes[0].set_title("Full incident continuum")
    axes[0].set_ylabel(r"$\nu\,4\pi J_\nu$ [erg cm$^{-2}$ s$^{-1}$]")

    high_visible = (
        (energy >= HIGH_ENERGY_MIN_RYD)
        & (energy <= HIGH_ENERGY_MAX_RYD)
        & (hm12 > 0.0)
    )
    high_y_max = 10.0 ** np.ceil(np.log10(hm12[high_visible].max()))
    axes[1].set_xlim(HIGH_ENERGY_MIN_RYD, HIGH_ENERGY_MAX_RYD)
    axes[1].set_ylim(high_y_max / 1.0e8, high_y_max)
    axes[1].set_title("7 eV to 1000 Ryd")
    axes[1].legend(frameon=False)

    fig.suptitle("Cloudy-native HM2012 with and without the CMB")
    fig.tight_layout()

    stem = "cloudy_unattenuated_HM2012_with_CMB_comparison"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        npz_path,
        energy_Ryd=energy,
        hm2012=hm12,
        cmb_z0=cmb,
        hm2012_plus_cmb=combined,
    )
    report = {
        "cloudy_commands": {
            "hm2012": "table HM12 redshift 0",
            "cmb": "CMB redshift 0",
        },
        "inputs": {
            "hm2012": str(hm12_path),
            "cmb": str(cmb_path),
            "hm2012_plus_cmb": str(combined_path),
        },
        "outputs": {"png": str(png_path), "pdf": str(pdf_path), "npz": str(npz_path)},
        "maximum_cmb_to_hm12_ratio_above_7eV": float(
            np.max(cmb[high_visible] / hm12[high_visible])
        ),
        "maximum_combined_vs_independent_sum_error_dex": maximum_sum_error_dex,
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
