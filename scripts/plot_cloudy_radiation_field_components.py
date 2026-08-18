#!/usr/bin/env python3
"""Plot Cloudy's actual HM2012, Draine, and Black/ISM incident continua.

The Black/ISM field is exported twice: without foreground attenuation and
after ``extinguish column=21 leak=0``.  All curves are read from Cloudy's
``save incident continuum`` output, so the plot uses the same internal energy
mesh and normalization as the line-table calculations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_incident(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"unexpected incident-continuum format: {path}")
    if np.any(np.diff(data[:, 0]) <= 0.0):
        raise ValueError(f"energy mesh is not increasing: {path}")
    return data[:, :2]


def export_draine(cloudy_exe: Path, working_dir: Path) -> Path:
    root = "export_draine_plain"
    incident = working_dir / f"{root}.inc"
    input_path = working_dir / f"{root}.in"
    input_path.write_text(
        """title export Cloudy table Draine
table Draine
hden -10
constant temperature 1e4 K
stop zone 1
set dr 0
save incident continuum "export_draine_plain.inc"
"""
    )
    subprocess.run([str(cloudy_exe), "-r", root], cwd=working_dir, check=True)
    if not incident.exists():
        raise FileNotFoundError(incident)
    return incident


def positive(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.0, values, np.nan)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_exports = (
        project_root
        / "work/cloudy_cooling_tools_history/examples/grackle/HM12_ISM_NH21"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cloudy-exe",
        type=Path,
        default=Path("/Users/tianbaochen/cloudy/c17.02/source/cloudy.exe"),
    )
    parser.add_argument("--exports-dir", type=Path, default=default_exports)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "output/radiation_fields",
    )
    args = parser.parse_args()
    cloudy_exe = args.cloudy_exe.resolve()
    exports_dir = args.exports_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "hm2012": exports_dir / "export_hm12_external.inc",
        "black_nh0": exports_dir / "export_ism_plain.inc",
        "black_nh21": exports_dir / "export_ism_filtered.inc",
    }
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}; run scripts/build_hm12_filtered_ism_sed.py first"
            )
    required["draine"] = export_draine(cloudy_exe, exports_dir)

    loaded = {name: load_incident(path) for name, path in required.items()}
    energy = loaded["hm2012"][:, 0]
    for name, data in loaded.items():
        if not np.array_equal(energy, data[:, 0]):
            raise ValueError(f"Cloudy energy mesh differs for {name}")

    curves = {name: data[:, 1] for name, data in loaded.items()}
    labels = {
        "hm2012": "HM2012 z=0 (Grackle input)",
        "draine": "Draine (table Draine)",
        "black_nh0": r"Black/ISM ($N_{H,\mathrm{fg}}=0$)",
        "black_nh21": (
            r"Black/ISM ($N_{H,\mathrm{fg}}=10^{21}\,\mathrm{cm}^{-2}$, leak=0)"
        ),
    }
    colors = {
        "hm2012": "#0072B2",
        "draine": "#E69F00",
        "black_nh0": "#009E73",
        "black_nh21": "#CC79A7",
    }
    linestyles = {
        "hm2012": "-",
        "draine": ":",
        "black_nh0": "-",
        "black_nh21": "--",
    }

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 8.6))
    draw_order = ("hm2012", "black_nh0", "black_nh21", "draine")
    for axis in axes:
        for name in draw_order:
            axis.loglog(
                energy,
                positive(curves[name]),
                label=labels[name],
                color=colors[name],
                linestyle=linestyles[name],
                linewidth=2.6 if name == "draine" else 1.7,
                marker="o" if name == "draine" else None,
                markersize=3.0 if name == "draine" else None,
                markevery=30 if name == "draine" else None,
                zorder=5 if name == "draine" else 2,
            )
        axis.axvline(1.0, color="0.35", linestyle=":", linewidth=1.0)
        axis.axvline(4.0, color="0.55", linestyle=":", linewidth=1.0)
        axis.grid(True, which="both", alpha=0.18)
        axis.set_ylabel(r"$\nu\,4\pi J_\nu$ [erg cm$^{-2}$ s$^{-1}$]")

    axes[0].set_title("Cloudy incident radiation fields")
    axes[0].set_xlim(1.0e-8, 1.0e7)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    axes[0].text(1.05, 0.97, "1 Ryd", transform=axes[0].get_xaxis_transform(),
                 ha="left", va="top", fontsize=8)
    axes[0].text(4.15, 0.97, "4 Ryd", transform=axes[0].get_xaxis_transform(),
                 ha="left", va="top", fontsize=8)
    axes[0].set_xlabel("Photon energy [Ryd]")

    axes[1].set_xlim(1.0e-2, 1.0e2)
    axes[1].set_ylim(1.0e-18, 5.0e-2)
    axes[1].set_title("UV and ionizing-energy zoom")
    axes[1].set_xlabel("Photon energy [Ryd]")
    axes[1].text(1.05, 0.97, "H-ionizing", transform=axes[1].get_xaxis_transform(),
                 ha="left", va="top", fontsize=8)

    fig.tight_layout()
    png = output_dir / "cloudy_HM2012_Draine_BlackISM_NH_comparison.png"
    pdf = output_dir / "cloudy_HM2012_Draine_BlackISM_NH_comparison.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    black_fig, black_axes = plt.subplots(2, 1, figsize=(10.2, 8.6))
    for axis in black_axes:
        for name in ("black_nh0", "black_nh21"):
            axis.loglog(
                energy,
                positive(curves[name]),
                label=labels[name],
                color=colors[name],
                linestyle=linestyles[name],
                linewidth=2.0,
            )
        axis.axvline(1.0, color="0.35", linestyle=":", linewidth=1.0)
        axis.axvline(4.0, color="0.55", linestyle=":", linewidth=1.0)
        axis.set_ylim(1.0e-18, 5.0e-2)
        axis.grid(True, which="both", alpha=0.18)
        axis.set_ylabel(r"$\nu\,4\pi J_\nu$ [erg cm$^{-2}$ s$^{-1}$]")

    black_axes[0].set_title("Black/ISM incident radiation field")
    black_axes[0].set_xlim(1.0e-8, 1.0e7)
    black_axes[0].legend(frameon=False, fontsize=10, loc="lower left")
    black_axes[0].text(
        1.05, 0.97, "1 Ryd", transform=black_axes[0].get_xaxis_transform(),
        ha="left", va="top", fontsize=8,
    )
    black_axes[0].text(
        4.15, 0.97, "4 Ryd", transform=black_axes[0].get_xaxis_transform(),
        ha="left", va="top", fontsize=8,
    )
    black_axes[0].set_xlabel("Photon energy [Ryd]")

    black_axes[1].set_xlim(1.0e-2, 1.0e2)
    black_axes[1].set_title("UV and ionizing-energy zoom")
    black_axes[1].set_xlabel("Photon energy [Ryd]")
    black_axes[1].text(
        1.05, 0.97, "H-ionizing",
        transform=black_axes[1].get_xaxis_transform(),
        ha="left", va="top", fontsize=8,
    )

    black_fig.tight_layout()
    black_png = output_dir / "cloudy_BlackISM_NH0_vs_NH21.png"
    black_pdf = output_dir / "cloudy_BlackISM_NH0_vs_NH21.pdf"
    black_fig.savefig(black_png, dpi=220, bbox_inches="tight")
    black_fig.savefig(black_pdf, bbox_inches="tight")
    plt.close(black_fig)

    hm_fig, hm_axes = plt.subplots(2, 1, figsize=(10.2, 8.6))
    for axis in hm_axes:
        axis.loglog(
            energy,
            positive(curves["hm2012"]),
            label=labels["hm2012"],
            color=colors["hm2012"],
            linewidth=2.0,
        )
        axis.axvline(1.0, color="0.35", linestyle=":", linewidth=1.0)
        axis.axvline(4.0, color="0.55", linestyle=":", linewidth=1.0)
        axis.grid(True, which="both", alpha=0.18)
        axis.set_ylabel(r"$\nu\,4\pi J_\nu$ [erg cm$^{-2}$ s$^{-1}$]")

    hm_axes[0].set_title("HM2012 z=0 incident radiation field")
    hm_axes[0].set_xlim(1.0e-8, 1.0e7)
    hm_axes[0].set_ylim(1.0e-42, 3.0e-4)
    hm_axes[0].legend(frameon=False, fontsize=10, loc="lower left")
    hm_axes[0].text(
        1.05, 0.97, "1 Ryd", transform=hm_axes[0].get_xaxis_transform(),
        ha="left", va="top", fontsize=8,
    )
    hm_axes[0].text(
        4.15, 0.97, "4 Ryd", transform=hm_axes[0].get_xaxis_transform(),
        ha="left", va="top", fontsize=8,
    )
    hm_axes[0].set_xlabel("Photon energy [Ryd]")

    hm_axes[1].set_xlim(1.0e-2, 1.0e2)
    hm_axes[1].set_ylim(1.0e-8, 3.0e-4)
    hm_axes[1].set_title("UV and ionizing-energy zoom")
    hm_axes[1].set_xlabel("Photon energy [Ryd]")
    hm_axes[1].text(
        1.05, 0.97, "H-ionizing",
        transform=hm_axes[1].get_xaxis_transform(),
        ha="left", va="top", fontsize=8,
    )

    hm_fig.tight_layout()
    hm_png = output_dir / "cloudy_HM2012_incident_spectrum.png"
    hm_pdf = output_dir / "cloudy_HM2012_incident_spectrum.pdf"
    hm_fig.savefig(hm_png, dpi=220, bbox_inches="tight")
    hm_fig.savefig(hm_pdf, bbox_inches="tight")
    plt.close(hm_fig)

    npz = output_dir / "cloudy_HM2012_Draine_BlackISM_NH_comparison.npz"
    np.savez_compressed(npz, energy_Ryd=energy, **curves)
    report = {
        "quantity": "nu * 4 pi * J_nu",
        "quantity_units": "erg cm^-2 s^-1",
        "energy_units": "Ryd",
        "definitions": {
            "hm2012": "Grackle external HM2012 z=0 init spectrum",
            "draine": "Cloudy table Draine at its built-in standard intensity",
            "black_nh0": "Cloudy table ISM, no foreground extinguish",
            "black_nh21": (
                "Cloudy table ISM followed by extinguish column=21 leak=0"
            ),
        },
        "energy_points": int(energy.size),
        "inputs": {name: str(path) for name, path in required.items()},
        "outputs": {
            "png": str(png),
            "pdf": str(pdf),
            "black_only_png": str(black_png),
            "black_only_pdf": str(black_pdf),
            "hm2012_only_png": str(hm_png),
            "hm2012_only_pdf": str(hm_pdf),
            "npz": str(npz),
        },
    }
    report_path = output_dir / "cloudy_HM2012_Draine_BlackISM_NH_comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
