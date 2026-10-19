"""Draw cold nH–NH distributions using only saved numerical figure arrays."""
from __future__ import annotations

from pathlib import Path

from quokka2s.figures.figure_files import save_figure_formats


def plot_cold_gas_phase(*, prepared, stem: Path, formats=("png", "pdf"), weight="count"):
    """Draw occupied-bin zoom and full table-range view of one saved histogram.

    prepared : mapping of str to ndarray
        From ColdGasPhaseAccumulator.build_outputs(), normally loaded from NPZ.
        Contains log10 display values/masks, color limits, physical bin edges,
        table outlines, axis limits and English labels. No snapshot/table reads.
    stem : pathlib.Path
        Figure directory and filename without an extension. PNG/PDF go into
        separate format subdirectories through save_figure_formats().
    formats : tuple of str
        Figure file formats, normally ("png", "pdf").
    weight : {"count", "mass"}
        Select saved log10 cold-cell counts or cold-mass fractions per bin.

    Returns list[pathlib.Path]. Normalization, summation, masks and numerical
    display limits were prepared before saving; this function only lays out plots.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if weight not in ("count", "mass"):
        raise ValueError("Cold phase weight must be 'count' or 'mass'")
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, 5.4),
        layout="constrained",
    )
    color_scale = Normalize(*prepared[weight + "_color_limits_log10"])
    for axis, view, title in (
        (axes[0], "zoom", "Cold cell distribution"),
        (axes[1], "full", "Table coverage"),
    ):
        image = axis.pcolormesh(
            prepared["edges_nH_log10"],
            prepared["edges_NH_log10"],
            prepared[weight + "_display_log10"].T,
            cmap="viridis",
            norm=color_scale,
            rasterized=True,
        )
        for prefix, label, color, style in (
            ("original", "Original table", "#d95f02", "--"),
            ("extended", "Extended table", "#1b9e77", "-"),
        ):
            outline = prepared[prefix + "_table_outline_log10"]
            axis.plot(
                outline[:, 0],
                outline[:, 1],
                color=color,
                linestyle=style,
                linewidth=1.8,
                label=label,
            )
        axis.set_xlim(*prepared[view + "_nH_limits_log10"])
        axis.set_ylim(*prepared[view + "_NH_limits_log10"])
        axis.set_xlabel(r"$\log_{10} n_{\rm H}$ [cm$^{-3}$]")
        axis.set_ylabel(r"$\log_{10} N_{\rm H}$ [cm$^{-2}$]")
        axis.set_title(title)
        axis.tick_params(direction="in", top=True, right=True)
        if not bool(prepared["has_cold_cells"]):
            axis.text(
                0.5,
                0.5,
                str(prepared["empty_population_label"]),
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
    axes[1].legend(loc="best", fontsize=9, framealpha=0.9)
    colorbar = figure.colorbar(image, ax=axes, shrink=0.9, pad=0.02)
    colorbar.set_label(str(prepared[weight + "_colorbar_label"]))
    figure.suptitle(str(prepared["title"]), fontsize=14)
    figure.supxlabel(
        str(prepared["population_label"]) + "\n" + str(prepared["sampling_label"]),
        fontsize=9,
    )
    paths = save_figure_formats(
        figure=figure,
        stem=Path(stem),
        formats=tuple(formats),
    )
    plt.close(figure)
    return paths
