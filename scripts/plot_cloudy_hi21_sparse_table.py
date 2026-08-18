"""Plot completed low/high HM2012 Cloudy H I 21-cm scans."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


META_RE = re.compile(r"^#\s*(hden|stop column density)\s+(.+?)\s*$")


def read_run(path: Path) -> tuple[float, float, np.ndarray, np.ndarray]:
    log_nh = log_nh_column = None
    log_temperature: list[float] = []
    log_coefficient: list[float] = []
    for line in path.read_text().splitlines():
        match = META_RE.match(line)
        if match:
            if match.group(1) == "hden":
                log_nh = float(match.group(2))
            else:
                log_nh_column = float(match.group(2))
            continue
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        log_temperature.append(float(fields[0]))
        log_coefficient.append(float(fields[1]) if len(fields) >= 2 else np.nan)
    if log_nh is None or log_nh_column is None:
        raise ValueError(f"missing loop metadata in {path}")
    return (
        log_nh,
        log_nh_column,
        np.asarray(log_temperature),
        np.asarray(log_coefficient),
    )


def load_directory(path: Path) -> dict[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
    result = {}
    for run_file in sorted(path.glob("*_run*.dat")):
        log_nh, log_column, log_t, log_eps = read_run(run_file)
        key = (log_nh, log_column)
        if key in result:
            raise ValueError(f"duplicate grid point {key}")
        result[key] = (log_t, log_eps)
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base = root / "work/cloudy_cooling_tools_history/examples/grackle"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--low-dir", type=Path,
        default=base / "hm_2012_hi21_low_sparse_output",
    )
    parser.add_argument(
        "--high-dir", type=Path,
        default=base / "hm_2012_hi21_high_sparse_output",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "output/plt0655228_down1_Lext15kpc/Cloudy_HI21_sparse_table.png",
    )
    parser.add_argument(
        "--title", default="Cloudy H I 21-cm table — HM2012, z=0",
    )
    args = parser.parse_args()

    low = load_directory(args.low_dir)
    high = load_directory(args.high_dir)
    if low.keys() != high.keys():
        raise ValueError("expected matching low/high scan grids")
    log_nh_axis = sorted({key[0] for key in low})
    log_column_axis = sorted({key[1] for key in low})
    if len(low) != len(log_nh_axis) * len(log_column_axis):
        raise ValueError("scan grid is not Cartesian")

    fig, axes = plt.subplots(
        len(log_nh_axis), len(log_column_axis),
        figsize=(15.0, 12.0), sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)
    low_color = "#0072B2"
    high_color = "#D55E00"
    failure_color = "#CC3311"

    for row, log_nh in enumerate(log_nh_axis):
        for column, log_nh_column in enumerate(log_column_axis):
            axis = axes[row, column]
            key = (log_nh, log_nh_column)
            for values, color in ((low[key], low_color), (high[key], high_color)):
                log_t, log_eps = values
                valid = np.isfinite(log_eps)
                # NaNs deliberately break the guide line across failed nodes.
                axis.plot(log_t, log_eps, color=color, lw=1.25, alpha=0.9)
                axis.scatter(
                    log_t[valid], log_eps[valid], s=24, color=color,
                    edgecolor="white", linewidth=0.45, zorder=3,
                )
                for failed_t in log_t[~valid]:
                    axis.axvline(
                        failed_t, color=failure_color, lw=1.0,
                        ls="--", alpha=0.9,
                    )
                    axis.text(
                        failed_t + 0.035, 0.05, "failure", rotation=90,
                        color=failure_color, fontsize=7,
                        transform=axis.get_xaxis_transform(),
                        va="bottom", ha="left",
                    )

            axis.axvline(np.log10(3000.0), color="0.55", ls=":", lw=0.8)
            axis.grid(True, alpha=0.22, ls="--", lw=0.45)
            axis.text(
                0.04, 0.08, rf"$\log_{{10}} n_{{\rm H}}={log_nh:g}$",
                transform=axis.transAxes, ha="left", va="bottom", fontsize=9,
            )
            if row == 0:
                axis.set_title(
                    rf"$\log_{{10}} N_{{\rm H}}={log_nh_column:g}$",
                    fontsize=11,
                )

    for axis in axes[-1, :]:
        axis.set_xlabel(r"$\log_{10}(T/\mathrm{K})$")
    for axis in axes[:, 0]:
        axis.set_ylabel(
            r"$\log_{10}(\epsilon_{21}/n_{\rm H}^{2})$"
            "\n" r"$[\mathrm{erg\ s^{-1}\ cm^{3}}]$"
        )

    handles = (
        Line2D([0], [0], color=low_color, marker="o", lw=1.25,
               label="10–3000 K nodes"),
        Line2D([0], [0], color=high_color, marker="o", lw=1.25,
               label=r"3000–$10^7$ K nodes"),
        Line2D([0], [0], color=failure_color, ls="--", lw=1.0,
               label="Cloudy failure (no value)"),
        Line2D([0], [0], color="0.55", ls=":", lw=0.8,
               label="3000 K boundary"),
    )
    fig.legend(
        handles=handles, loc="upper center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 0.955), fontsize=9,
    )
    fig.suptitle(args.title, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
