"""Replot the cached cold/hot C+ comparisons with one shared linear y-limit."""
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("output/plt0655228_down1_Lext15kpc")
INTERMEDIATE_DIR = OUTPUT_DIR / "task_intermediates"
LOW_CACHE = INTERMEDIATE_DIR / "Build_CplusLowCloudyComparison_c389c117.h5"
HIGH_CACHE = INTERMEDIATE_DIR / "Build_CplusHighModelComparison_8e3ea584.h5"

LOW_OUTPUT = OUTPUT_DIR / "Cplus_TQK_lt3000_DESPOTIC_Cloudy_Rinf_shared_ylim.png"
HIGH_OUTPUT = OUTPUT_DIR / "Cplus_TQK_ge3000_DESPOTIC_Cloudy_Rinf_shared_ylim.png"

COLORS = {"DESPOTIC": "#0072B2", "Cloudy HM2012": "#D55E00"}


def _read(group: h5py.Group, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(group[f"{prefix}/v_axis"]), np.asarray(group[f"{prefix}/dsigma_dv"])


def _draw(
    curves: tuple[tuple[str, np.ndarray, np.ndarray], ...],
    title: str,
    ylim: tuple[float, float],
    output: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 4.9))
    for label, velocity, values in curves:
        axis.plot(
            velocity,
            values,
            color=COLORS[label],
            linewidth=1.6,
            drawstyle="steps-mid",
            label=label,
        )
    axis.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
    axis.set_xlabel(r"Velocity [km s$^{-1}$]")
    axis.set_ylabel(r"$d\Sigma_L/dv$ [L$_\odot$ pc$^{-2}$ (km s$^{-1}$)$^{-1}$]")
    axis.set_ylim(*ylim)
    axis.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    axis.legend(fontsize=8.5, frameon=False)
    axis.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
    axis.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with h5py.File(LOW_CACHE, "r") as low_file, h5py.File(HIGH_CACHE, "r") as high_file:
        low_group = low_file["spectra"]
        high_group = high_file["spectra"]
        low_curves = (
            ("DESPOTIC", *_read(low_group, "CPLUS_DESPOTIC_TQK_LT3000_DIAGNOSTIC")),
            ("Cloudy HM2012", *_read(low_group, "CPLUS_CLOUDY_TQK_LT3000_DIAGNOSTIC")),
        )
        high_curves = (
            ("DESPOTIC", *_read(high_group, "CPLUS_DESPOTIC_TQK_GE3000/y")),
            ("Cloudy HM2012", *_read(high_group, "CPLUS_CLOUDY_TQK_GE3000/y")),
        )

    shared_max = max(float(np.nanmax(values)) for _, _, values in low_curves + high_curves)
    shared_ylim = (0.0, 1.10 * shared_max)

    _draw(
        low_curves,
        r"[C II] 158 $\mu$m: $T_{\rm QUOKKA}<3000\,$K, LOS y, R=$\infty$",
        shared_ylim,
        LOW_OUTPUT,
    )
    _draw(
        high_curves,
        r"[C II] 158 $\mu$m: $T_{\rm QUOKKA}\geq3000\,$K, LOS y, R=$\infty$",
        shared_ylim,
        HIGH_OUTPUT,
    )
    print(f"Shared ylim: {shared_ylim}")
    print(f"Saved: {LOW_OUTPUT}")
    print(f"Saved: {HIGH_OUTPUT}")


if __name__ == "__main__":
    main()
