"""Plot the paired low/high-temperature DESPOTIC/Cloudy C+ spectra."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUTPUT_DIR = Path("output/plt0655228_down1_Lext15kpc")


def read_curve(handle: h5py.File, model: str, *, los: str | None = None):
    prefix = f"spectra/{model}"
    if los is not None:
        prefix += f"/{los}"
    return np.asarray(handle[f"{prefix}/v_axis"]), np.asarray(handle[f"{prefix}/dsigma_dv"])


def draw(curves, title: str, output: Path, ylim=None) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 4.9))
    for label, color, velocity, values in curves:
        axis.plot(
            velocity, values, color=color, linewidth=1.6,
            drawstyle="steps-mid", label=label,
        )
    axis.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
    axis.set_xlabel(r"Velocity [km s$^{-1}$]")
    axis.set_ylabel(r"$d\Sigma_L/dv$ [L$_\odot$ pc$^{-2}$ (km s$^{-1}$)$^{-1}$]")
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    axis.legend(fontsize=8.5, frameon=False)
    axis.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
    axis.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def latest_cache(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"no cache matches {directory / pattern}")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--high-cache", type=Path)
    parser.add_argument("--low-cache", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir
    cache_dir = output_dir / "task_intermediates"
    high_cache = args.high_cache or latest_cache(
        cache_dir, "Build_CplusHighModelComparison_*.h5"
    )
    low_cache = args.low_cache or latest_cache(
        cache_dir, "Build_CplusLowCloudyComparison_*.h5"
    )
    auto_output = output_dir / "Cplus_TQK_ge3000_DESPOTIC_Cloudy_full31_Rinf.png"
    low_shared_output = output_dir / "Cplus_TQK_lt3000_DESPOTIC_Cloudy_Rinf_shared_ylim_simpletitle.png"
    high_shared_output = output_dir / "Cplus_TQK_ge3000_DESPOTIC_Cloudy_full31_Rinf_shared_ylim_simpletitle.png"

    with h5py.File(high_cache, "r") as high:
        d_velocity, despotic = read_curve(high, "CPLUS_DESPOTIC_TQK_GE3000", los="y")
        c_velocity, cloudy = read_curve(high, "CPLUS_CLOUDY_TQK_GE3000", los="y")
    np.testing.assert_allclose(d_velocity, c_velocity)
    high_curves = (
        ("DESPOTIC", "#0072B2", d_velocity, despotic),
        ("Cloudy HM2012", "#D55E00", c_velocity, cloudy),
    )
    high_title = (
        r"[C II] 158 $\mu$m: $T_{\rm QUOKKA}\geq3000\,$K, "
        r"LOS y, R=$\infty$"
    )
    draw(high_curves, high_title, auto_output)

    with h5py.File(low_cache, "r") as low:
        low_velocity, low_despotic = read_curve(
            low, "CPLUS_DESPOTIC_TQK_LT3000_DIAGNOSTIC"
        )
        low_cloudy_velocity, low_cloudy = read_curve(
            low, "CPLUS_CLOUDY_TQK_LT3000_DIAGNOSTIC"
        )
    np.testing.assert_allclose(low_velocity, low_cloudy_velocity)
    low_curves = (
        ("DESPOTIC", "#0072B2", low_velocity, low_despotic),
        ("Cloudy HM2012", "#D55E00", low_cloudy_velocity, low_cloudy),
    )
    shared_max = max(
        float(np.nanmax(values))
        for values in (despotic, cloudy, low_despotic, low_cloudy)
    )
    shared_ylim = (0.0, 1.10 * shared_max)
    draw(
        low_curves,
        r"[C II] 158 $\mu$m: $T_{\rm QUOKKA}<3000\,$K, LOS y, R=$\infty$",
        low_shared_output,
        ylim=shared_ylim,
    )
    draw(high_curves, high_title, high_shared_output, ylim=shared_ylim)
    print(f"High cache: {high_cache}")
    print(f"Low cache: {low_cache}")
    print(f"Saved: {auto_output}")
    print(f"Saved: {low_shared_output}")
    print(f"Saved: {high_shared_output}")
    print(f"High maxima: DESPOTIC={np.nanmax(despotic):.9e}, Cloudy={np.nanmax(cloudy):.9e}")
    print(f"Shared ylim: 0, {1.10 * shared_max:.9e}")


if __name__ == "__main__":
    main()
