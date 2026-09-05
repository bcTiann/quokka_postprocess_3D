#!/usr/bin/env python3
"""Compare HM2012 plus physical-slab and quick-extinguished ISM fields."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EV_PER_RYD = 13.605693122994
X_MIN_EV = 7.0
X_MIN_RYD = X_MIN_EV / EV_PER_RYD
EIGHT_EV_RYD = 8.0 / EV_PER_RYD
LY_ALPHA_EV = 10.2
LY_ALPHA_RYD = LY_ALPHA_EV / EV_PER_RYD
X_MAX_RYD = 1.0e3
Y_DYNAMIC_RANGE_DEX = 8.0


def positive(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.0, values, np.nan)


def load(path: Path, usecols: tuple[int, ...]) -> np.ndarray:
    data = np.loadtxt(path, comments="#", usecols=usecols)
    if data.ndim != 2 or np.any(np.diff(data[:, 0]) <= 0.0):
        raise ValueError(f"unexpected Cloudy continuum file: {path}")
    return data


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    radiation_dir = (
        root
        / "work/cloudy_cooling_tools_history/examples/grackle"
        / "HM12_NATIVE_ISM_NH21_CMB"
    )
    slab_dir = (
        root
        / "work/cloudy_cooling_tools_history/examples/grackle"
        / "ISM_PHYSICAL_SLAB_NH21_NH8"
    )
    output_dir = root / "output/radiation_fields"
    output_dir.mkdir(parents=True, exist_ok=True)

    hm12_path = radiation_dir / "export_hm12_native.inc"
    quick_ism_path = radiation_dir / "export_ism_filtered.inc"
    slab_continuum_path = slab_dir / "ism_slab_nh8_NH21_grains.con"

    hm12_data = load(hm12_path, (0, 1))
    quick_data = load(quick_ism_path, (0, 1))
    # save continuum columns: energy, incident, direct transmitted,
    # outward newly emitted, net transmitted.  Column 5 is the same physical
    # outgoing field saved in Cloudy's table-readable .trn file.
    slab_data = load(slab_continuum_path, (0, 1, 2, 3, 4))
    energy = hm12_data[:, 0]
    if not np.array_equal(energy, quick_data[:, 0]):
        raise ValueError("HM2012 and quick-extinguished ISM meshes differ")
    if not np.array_equal(energy, slab_data[:, 0]):
        raise ValueError("HM2012 and physical-slab continuum meshes differ")

    hm12 = hm12_data[:, 1]
    quick_ism = quick_data[:, 1]
    transmitted_ism = slab_data[:, 4]
    physical_combined = hm12 + transmitted_ism
    quick_combined = hm12 + quick_ism

    visible = (energy >= X_MIN_RYD) & (energy <= X_MAX_RYD)
    maximum = max(
        float(values[visible].max())
        for values in (
            physical_combined,
            hm12,
            transmitted_ism,
            quick_combined,
            quick_ism,
        )
    )
    y_max = 10.0 ** np.ceil(np.log10(maximum))
    y_min = y_max / 10.0**Y_DYNAMIC_RANGE_DEX

    fig, axes = plt.subplots(2, 1, figsize=(9.4, 8.0), sharex=True)
    panels = (
        (
            axes[0],
            physical_combined,
            transmitted_ism,
            "HM2012 + transmitted ISM",
            r"transmitted ISM ($n_{\rm H}=8$ cm$^{-3}$, $N_{\rm H}=10^{21}$ cm$^{-2}$)",
        ),
        (
            axes[1],
            quick_combined,
            quick_ism,
            "HM2012 + extinguished ISM",
            r"extinguished ISM (column=21)",
        ),
    )
    for index, (axis, combined, ism, title, ism_label) in enumerate(panels):
        axis.loglog(
            energy,
            positive(combined),
            color="black",
            linewidth=2.4,
            label="combined",
            zorder=1,
        )
        axis.loglog(
            energy,
            positive(hm12),
            color="#0072B2",
            linestyle="--",
            linewidth=2.0,
            label="HM2012",
            zorder=3,
        )
        axis.loglog(
            energy,
            positive(ism),
            color="#D55E00",
            linestyle=":",
            linewidth=2.2,
            label=ism_label,
            zorder=4,
        )
        axis.axvline(EIGHT_EV_RYD, color="0.35", linestyle=":", linewidth=1.2)
        axis.axvline(
            LY_ALPHA_RYD,
            color="#8C2D04",
            linestyle="--",
            linewidth=1.3,
            zorder=5,
        )
        if index == 0:
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
            axis.text(
                LY_ALPHA_RYD,
                0.04,
                r"Ly$\alpha$ 10.2 eV (0.75 Ryd)",
                rotation=90,
                transform=axis.get_xaxis_transform(),
                ha="right",
                va="bottom",
                color="#8C2D04",
                fontsize=8,
            )
        axis.set_xlim(X_MIN_RYD, X_MAX_RYD)
        axis.set_ylim(y_min, y_max)
        axis.set_ylabel(r"$\nu\,4\pi J_\nu$ [erg cm$^{-2}$ s$^{-1}$]")
        axis.set_title(title)
        axis.legend(frameon=False, fontsize=8.5)
        axis.grid(True, which="both", alpha=0.17)
    axes[1].set_xlabel("Photon energy [Ryd]")
    fig.tight_layout()

    stem = "cloudy_HM2012_physical_transmitted_vs_extinguished_ISM_NH21"
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
        physical_ism_net_transmitted=transmitted_ism,
        hm2012_plus_physical_ism=physical_combined,
        quick_extinguished_ism=quick_ism,
        hm2012_plus_quick_extinguished_ism=quick_combined,
    )
    report = {
        "definitions": {
            "physical_ism": (
                "save continuum column 5 from table ISM propagated through "
                "a converged nH=8 cm^-3, NH=1e21 cm^-2 slab with ISM grains; "
                "direct transmitted plus newly emitted outward radiation"
            ),
            "quick_ism": "table ISM followed by extinguish column=21 leak=0",
            "combined": "linear sum with Cloudy-native unattenuated HM2012",
            "cmb_included": False,
        },
        "inputs": {
            "hm2012": str(hm12_path),
            "physical_slab_continuum": str(slab_continuum_path),
            "physical_slab_transmitted": str(
                slab_dir / "ism_slab_nh8_NH21_grains.trn"
            ),
            "quick_extinguished_ism": str(quick_ism_path),
        },
        "plot_range": {
            "x_min_eV": X_MIN_EV,
            "x_min_Ryd": X_MIN_RYD,
            "x_max_Ryd": X_MAX_RYD,
            "y_min": y_min,
            "y_max": y_max,
            "y_dynamic_range_dex": Y_DYNAMIC_RANGE_DEX,
            "markers": {
                "8_eV_Ryd": EIGHT_EV_RYD,
                "Ly_alpha_eV": LY_ALPHA_EV,
                "Ly_alpha_Ryd": LY_ALPHA_RYD,
            },
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
