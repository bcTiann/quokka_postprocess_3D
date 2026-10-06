#!/usr/bin/env python3
"""Plot the full Draine MW R_V=3.1 extinction grid and adopted line positions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quokka2s.physics.dust_attenuation import DEFAULT_DRAINE_TABLE, LINE_WAVELENGTH_MICRON, extinction_cross_sections, load_draine_extinction


GROUPS = (
    ("ciii_977", "C III 977 Å", "#D55E00"),
    ("civ_1548", "C IV 1548/1551 Å", "#C28A00"),
    ("ciii_1907", "C III 1907/1909 Å", "#D55E00"),
    ("halpha", r"H$\alpha$", "#C6414B"),
    ("cii", "C II 158 µm", "#7B57A6"),
    ("co21", "CO(2–1)", "#1479A9"),
    ("co10", "CO(1–0)", "#1479A9"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    wavelength, sigma = load_draine_extinction()
    keys = tuple(LINE_WAVELENGTH_MICRON)
    line_sigma = dict(zip(keys, extinction_cross_sections(keys, wavelength, sigma)))
    if wavelength[0] != 1.e-4 or wavelength[-1] != 1.e4 or wavelength.size != 1077:
        raise ValueError("Unexpected Draine table wavelength grid")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator

    plt.rcParams.update({"font.size": 9, "axes.linewidth": .8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (ax, zoom) = plt.subplots(2, 1, figsize=(7.1, 5.6),
                                  gridspec_kw={"height_ratios": [2.5, 1.1]})
    fig.subplots_adjust(left=.12, right=.98, bottom=.10, top=.96, hspace=.42)
    ax.axvspan(wavelength[-1], 3.5e5, color="#ECEFF2", zorder=0)
    ax.plot(wavelength, sigma, color="#232B33", lw=1.5, label=r"Draine $R_V=3.1$ $C_{\rm ext}/H$", zorder=2)
    ax.axvline(wavelength[-1], color="#6D747D", ls="--", lw=1, zorder=1)
    ax.text(1.12e4, 2.5e-28, "1 cm limit", rotation=90,
            va="bottom", ha="left", fontsize=8, color="#545C65")

    for key, label, color in GROUPS:
        ax.scatter(LINE_WAVELENGTH_MICRON[key], line_sigma[key], s=31,
                   facecolor=color, edgecolor="white", lw=.6, zorder=4)
    # Mark each member of the close UV doublets, though their labels are grouped.
    for key, color in (("civ_1551", "#C28A00"),
                       ("ciii_1909", "#D55E00")):
        ax.scatter(LINE_WAVELENGTH_MICRON[key], line_sigma[key], s=21,
                   facecolor=color, edgecolor="white", lw=.5, zorder=4)

    for key, label, color, offset in (
        ("halpha", r"H$\alpha$", "#C6414B", (10, -15)),
        ("cii", "C II 158 µm", "#7B57A6", (7, 11)),
        ("co21", "CO(2–1)", "#1479A9", (-33, 15)),
        ("co10", "CO(1–0)", "#1479A9", (-34, -17)),
    ):
        ax.annotate(label, (LINE_WAVELENGTH_MICRON[key], line_sigma[key]),
                    xytext=offset, textcoords="offset points", color=color,
                    fontsize=8, arrowprops=dict(arrowstyle="-", color=color, lw=.6))

    hi_wavelength = LINE_WAVELENGTH_MICRON["hi21"]
    ax.axvline(hi_wavelength, color="#67717A", ls=":", lw=1.1)
    ax.text(hi_wavelength, 5.e-25, "H I 21 cm: outside table", rotation=90,
            ha="right", va="center", color="#4D5861", fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(wavelength[0], 3.5e5)
    ax.set_ylim(5.e-29, 1.e-20)
    ax.set_xlabel(r"Vacuum wavelength $\lambda$ [$\mu$m]")
    ax.set_ylabel(r"Extinction cross-section $C_{\rm ext}/H$ [cm$^2$ H$^{-1}$]")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.grid(which="major", alpha=.16, lw=.5)
    ax.text(.035, .045, "Tabulated: 1 Å–1 cm", transform=ax.transAxes,
            fontsize=8, color="#4D5861")
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    # The second panel resolves lines compressed together on the full-range axis.
    zoom.plot(wavelength, sigma, color="#232B33", lw=1.25)
    for key, label, color in GROUPS[:4]:
        zoom.scatter(LINE_WAVELENGTH_MICRON[key], line_sigma[key], s=35,
                     color=color, edgecolor="white", lw=.5, zorder=4)
    for key, color in (("civ_1551", "#C28A00"),
                       ("ciii_1909", "#D55E00")):
        zoom.scatter(LINE_WAVELENGTH_MICRON[key], line_sigma[key], s=25,
                     color=color, edgecolor="white", lw=.5, zorder=4)
    for key, label, color, position in (
        ("ciii_977", "C III 977 Å", "#D55E00", (.083, 3.1e-21)),
        ("civ_1548", "C IV 1548/1551 Å", "#A97700", (.108, 5.0e-22)),
        ("ciii_1907", "C III 1907/1909 Å", "#D55E00", (.23, 1.6e-21)),
        ("halpha", r"H$\alpha$", "#C6414B", (.51, 7.0e-22)),
    ):
        zoom.annotate(label, (LINE_WAVELENGTH_MICRON[key], line_sigma[key]),
                      xytext=position, textcoords="data", color=color,
                      fontsize=7.5, arrowprops=dict(arrowstyle="-", color=color, lw=.6))
    zoom.set_xscale("log")
    zoom.set_yscale("log")
    zoom.set_xlim(.08, .85)
    zoom.set_ylim(2.e-22, 4.e-21)
    zoom.set_xlabel(r"Vacuum wavelength $\lambda$ [$\mu$m] (UV and optical detail)")
    zoom.set_ylabel(r"$C_{\rm ext}/H$ [cm$^2$ H$^{-1}$]")
    zoom.tick_params(labelsize=8, length=3)
    zoom.grid(alpha=.16, lw=.5)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "draine_mw31_full_range_and_lines"
    for suffix in ("pdf", "png"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=300,
                    bbox_inches="tight", metadata={"Title": "Draine MW R_V=3.1 extinction range and adopted emission lines"})
    plt.close(fig)
    metadata = {
        "source": str(DEFAULT_DRAINE_TABLE),
        "table_rows": int(wavelength.size),
        "table_wavelength_micron": [float(wavelength[0]), float(wavelength[-1])],
        "line_wavelength_micron": LINE_WAVELENGTH_MICRON,
        "line_sigma_ext_cm2_H": {key: float(value) for key, value in line_sigma.items()},
        "hi21": "Outside the 1 cm table limit; no dust attenuation applied",
    }
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
