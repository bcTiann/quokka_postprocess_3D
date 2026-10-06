#!/usr/bin/env python3
"""Plot unattenuated Cloudy-native HM2012, Black/ISM, and their sum.

Inputs are the historical exports in runtime/cloudy_eightline/sed, including
the plain ISM and native HM12 continua. The current Jeans-table SED generator
exports attenuated continua to a different bundle; it does not produce these
unattenuated inputs. --data-dir must supply this figure's original exports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quokka2s.cloudy.incident_spectrum import read_incident_spectrum
from quokka2s.figures.radiation_fields import (
    RADIATION_X_MAX_RYD,
    RADIATION_X_MIN_EV,
    RADIATION_X_MIN_RYD,
    RADIATION_Y_DYNAMIC_RANGE_DEX,
    add_eight_ev_marker,
    positive,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    default_data_dir = project_root / "runtime/cloudy_eightline/sed"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Directory containing the exported Cloudy incident-continuum .inc files",
    )
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

    hm12_data = read_incident_spectrum(
        path=hm12_path,
        usecols=(0, 1),
    )
    ism_data = read_incident_spectrum(
        path=ism_path,
        usecols=(0, 1),
    )
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
        positive(values=combined),
        color="black",
        linewidth=2.4,
        label="HM2012 + CMB + ISM" if args.include_cmb else "HM2012 + ISM",
        zorder=1,
    )
    ax.loglog(
        energy,
        positive(values=hm12),
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
        positive(values=ism),
        color="#D55E00",
        linestyle="--",
        linewidth=2.0,
        label="Cloudy table ISM unattenuated",
        zorder=4,
    )
    add_eight_ev_marker(
        axis=ax,
        label_line=True,
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
