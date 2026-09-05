#!/usr/bin/env python3
"""Plot unattenuated Cloudy-native HM2012, Black/ISM, and their sum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EV_PER_RYD = 13.605693122994
RADIATION_X_MIN_EV = 7.0
RADIATION_X_MIN_RYD = RADIATION_X_MIN_EV / EV_PER_RYD
EIGHT_EV_RYD = 8.0 / EV_PER_RYD
RADIATION_X_MAX_RYD = 1.0e3
RADIATION_Y_DYNAMIC_RANGE_DEX = 8.0


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
        "--include-cmb",
        action="store_true",
        help="Use the Cloudy-exported HM2012+CMB continuum instead of HM2012 alone",
    )
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
    ism_path = data_dir / "export_ism_plain.inc"

    if args.include_cmb:
        hm12_path = data_dir / "export_hm12_native_plus_cmb.inc"

    hm12_data = load_incident(hm12_path)
    ism_data = load_incident(ism_path)
    if not np.array_equal(hm12_data[:, 0], ism_data[:, 0]):
        raise ValueError("HM2012 and ISM Cloudy energy meshes differ")

    energy = hm12_data[:, 0]
    hm12 = hm12_data[:, 1]
    ism = ism_data[:, 1]
    combined = hm12 + ism
    visible = (
        (energy >= RADIATION_X_MIN_RYD)
        & (energy <= RADIATION_X_MAX_RYD)
        & (combined > 0.0)
    )
    if not np.any(visible):
        raise ValueError("no positive radiation values in requested x range")
    visible_maximum = float(combined[visible].max())
    y_max = 10.0 ** np.ceil(np.log10(visible_maximum))
    y_min = y_max / 10.0**RADIATION_Y_DYNAMIC_RANGE_DEX

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.loglog(
        energy,
        positive(combined),
        color="black",
        linewidth=2.4,
        label="HM2012 + CMB + ISM" if args.include_cmb else "HM2012 + ISM",
        zorder=1,
    )
    ax.loglog(
        energy,
        positive(hm12),
        color="#0072B2",
        linestyle="--",
        linewidth=2.0,
        label=(
            "Cloudy-native unattenuated HM2012 + CMB"
            if args.include_cmb
            else "Cloudy-native unattenuated HM2012"
        ),
        zorder=3,
    )
    ax.loglog(
        energy,
        positive(ism),
        color="#D55E00",
        linestyle="--",
        linewidth=2.0,
        label="Cloudy table ISM unattenuated",
        zorder=4,
    )
    ax.axvline(EIGHT_EV_RYD, color="0.35", linestyle=":", linewidth=1.2)
    ax.text(
        EIGHT_EV_RYD,
        0.04,
        "8 eV",
        rotation=90,
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="bottom",
        color="0.35",
        fontsize=8,
    )
    ax.set_xlim(RADIATION_X_MIN_RYD, RADIATION_X_MAX_RYD)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Photon energy [Ryd]")
    ax.set_ylabel(r"$\nu\,4\pi J_\nu$ [erg cm$^{-2}$ s$^{-1}$]")
    ax.set_title("Unattenuated incident radiation fields")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.17)
    fig.tight_layout()

    stem = (
        "cloudy_unattenuated_HM2012_CMB_ISM_sum"
        if args.include_cmb
        else "cloudy_unattenuated_HM2012_ISM_sum"
    )
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
        hm2012_with_optional_cmb=hm12,
        ism_unattenuated=ism,
        hm2012_plus_ism=combined,
    )
    report = {
        "definitions": {
            "hm2012": (
                "Cloudy 17.02 table HM12 redshift 0 plus CMB redshift 0, "
                "unattenuated"
                if args.include_cmb
                else "Cloudy 17.02 table HM12 redshift 0, unattenuated"
            ),
            "ism": "Cloudy 17.02 table ISM (Black 1987), unattenuated",
            "sum": "linear sum of the plotted Cloudy incident continua",
        },
        "inputs": {"hm2012": str(hm12_path), "ism": str(ism_path)},
        "plot_range": {
            "x_min_eV": RADIATION_X_MIN_EV,
            "x_min_Ryd": RADIATION_X_MIN_RYD,
            "x_max_Ryd": RADIATION_X_MAX_RYD,
            "y_max": y_max,
            "y_min": y_min,
            "y_dynamic_range_dex": RADIATION_Y_DYNAMIC_RANGE_DEX,
        },
        "outputs": {
            "png": str(png_path),
            "pdf": str(pdf_path),
            "npz": str(npz_path),
        },
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
