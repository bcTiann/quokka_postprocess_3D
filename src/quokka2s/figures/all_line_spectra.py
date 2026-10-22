"""Compare every saved total emission-line profile on a common velocity axis."""
from __future__ import annotations

from pathlib import Path
from textwrap import fill

from quokka2s.emission_results import SpectralResults
from quokka2s.figures.figure_files import save_figure_formats
from quokka2s.figures.line_labels import LINE_TITLES


# Keep a line's colour fixed across dust states and displayed velocity windows.
LINE_COLORS = {
    "halpha": "#d62728",
    "hi21": "#1f77b4",
    "cii": "#ff7f0e",
    "ciii_977": "#2ca02c",
    "ciii_1907": "#9467bd",
    "ciii_1909": "#8c564b",
    "civ_1548": "#e377c2",
    "civ_1551": "#7f7f7f",
    "co10": "#17becf",
    "co21": "#bcbd22",
}
DUST_STATE_TITLES = {
    "intrinsic": "Intrinsic",
    "attenuated": "Dust attenuated",
}


def plot_all_line_spectra(
    spectra: SpectralResults,
    output: Path,
    *,
    velocity_limits_kms: tuple[float, float],
    filename_suffix: str,
    diagnostic_suffix: str,
    titled: bool,
    formats: tuple[str, ...],
    run_label: str = "",
) -> list[Path]:
    """Write intrinsic and attenuated comparisons using saved normalized totals.

    spectra supplies cold+hot total profiles, their original velocity coordinates
    [km/s], and positive-light flags. Every dimensionless curve was divided by
    its own peak over all saved channels during numerical preparation. Selecting
    a narrower display window does not change those values. A line without a
    positive saved profile receives a legend entry but no data curve.

    output is the figure directory. filename_suffix distinguishes display windows,
    for example "_v50"; diagnostic_suffix labels incomplete products if allowed
    by the caller. Returned paths include both dust states in requested formats.
    run_label adds job, snapshot and saved time to titled figures only.
    """
    import matplotlib.pyplot as plt

    paths = []
    header = fill(run_label, width=115) if titled and run_label else ""
    header_height = 0.22 * (header.count("\n") + 1) if header else 0.0
    figure_height = 4.7 + header_height
    for dust_state in ("intrinsic", "attenuated"):
        print(f"  All-line spectrum: {dust_state}", flush=True)
        figure, axis = plt.subplots(figsize=(9.3, figure_height))
        for line_key in spectra.line_keys:
            line_spectrum = spectra.for_line(
                line=line_key,
                dust_state=dust_state,
                regime="total",
            )
            line_label = LINE_TITLES[line_key]
            line_color = LINE_COLORS[line_key]
            if line_spectrum.profile_has_light:
                axis.plot(
                    line_spectrum.velocity_kms,
                    line_spectrum.peak_normalized_profile,
                    color=line_color,
                    lw=1.6,
                    label=line_label,
                )
            else:
                axis.plot(
                    [],
                    [],
                    color=line_color,
                    ls=":",
                    alpha=0.45,
                    label=f"{line_label} (no positive flux)",
                )
        axis.set_xlim(*velocity_limits_kms)
        axis.set_ylim(-0.035, 1.07)
        axis.set_xlabel(r"$v_z$ [km s$^{-1}$]")
        axis.set_ylabel("Peak-normalised line profile")
        axis.axvline(0.0, color="0.65", ls=":", lw=0.8)
        axis.grid(alpha=0.18)
        axis.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=False,
            fontsize=9,
        )
        if titled:
            axis.set_title(f"All emission lines: {DUST_STATE_TITLES[dust_state]}")
        if header:
            figure.suptitle(
                header,
                fontsize=9,
                y=1 - 0.06 / figure_height,
                parse_math=False,
            )
        figure.subplots_adjust(
            left=0.09,
            right=0.70,
            bottom=0.16 * 4.7 / figure_height if header else 0.16,
            top=0.88 * 4.7 / figure_height if header else (0.88 if titled else 0.97),
        )
        paths.extend(save_figure_formats(
            figure=figure,
            stem=output / f"spectrum_all_lines_{dust_state}{filename_suffix}{diagnostic_suffix}",
            formats=formats,
        ))
        plt.close(figure)
    return paths
