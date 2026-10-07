"""Draw prepared DESPOTIC panels and saved contour segments."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


SAMPLE_CONTOUR_STYLES = {
    0.50: dict(color="red", linestyle="solid", linewidth=1.4),
    0.90: dict(color="darkorange", linestyle="dashed", linewidth=1.1),
    0.99: dict(color="gold", linestyle="dotted", linewidth=1.1),
    1.00: dict(color="royalblue", linestyle="dashdot", linewidth=1.1),
}
TEMPERATURE_CONTOUR_STYLES = {
    100.0: "solid",
    200.0: "dashed",
}


def draw_sample_contours(ax, figure_data: Mapping[str, np.ndarray], slice_index: int) -> None:
    """Draw one slice's saved enclosure segments and legend on any field panel."""
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    prefix = f"samples_slice_{slice_index}"
    fractions = figure_data[f"{prefix}_fractions"]
    if fractions.size == 0:
        return
    handles = []
    for level_index, fraction in enumerate(fractions):
        style = SAMPLE_CONTOUR_STYLES[float(fraction)]
        segments = figure_data[f"{prefix}_segments_{level_index}"]
        if segments.size:
            contour = LineCollection(segments, alpha=0.9, **style)
            ax.add_collection(contour)
        handles.append(Line2D(
            [0],
            [0],
            label=str(figure_data[f"{prefix}_labels"][level_index]),
            **style,
        ))
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
        title=str(figure_data[f"{prefix}_legend_title"]),
        title_fontsize=8,
    )


def draw_temperature_contours(ax, figure_data: Mapping[str, np.ndarray], slice_index: int) -> None:
    """Draw saved 100/200 K staircase segments and their prepared label positions."""
    from matplotlib.collections import LineCollection

    prefix = f"temperature_slice_{slice_index}"
    levels = figure_data[f"{prefix}_levels_K"]
    positions = figure_data[f"{prefix}_label_positions"]
    for level_index, threshold in enumerate(levels):
        segments = figure_data[f"{prefix}_segments_{level_index}"]
        contour = LineCollection(
            segments,
            colors="black",
            linewidths=1.0,
            linestyles=TEMPERATURE_CONTOUR_STYLES[float(threshold)],
        )
        ax.add_collection(contour)
        xmid, ymid = positions[level_index]
        ax.text(
            xmid,
            ymid,
            f"{int(threshold)} K",
            fontsize=7,
            color="black",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6),
        )


def draw_table_panel(
    ax,
    figure_data: Mapping[str, np.ndarray],
    slice_index: int,
    field_index: int,
    *,
    cmap: str,
    show_colorbar: bool,
    figure,
) -> None:
    """Draw one saved (NnH, NNH) panel with its mask, limits and contour geometry.

    table_figure_data prepared physical axis edges, positive-value masks and
    shared per-slice overlays. Matplotlib applies the saved logarithmic color
    scale; this renderer does not analyze the table or samples.
    """
    from matplotlib.colors import LogNorm

    values = figure_data["panel_values"][slice_index, field_index]
    hidden = figure_data["panel_is_hidden"][slice_index, field_index]
    masked_values = np.ma.masked_array(values, mask=hidden)
    norm = None
    if figure_data["panel_has_values"][slice_index, field_index]:
        minimum, maximum = figure_data["color_limits"][slice_index, field_index]
        norm = LogNorm(vmin=minimum, vmax=maximum)
    title = str(figure_data["field_titles"][field_index])
    mesh = ax.pcolormesh(
        figure_data["column_edges_cm2"],
        figure_data["density_edges_cm3"],
        masked_values,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("Column Density (cm$^{-2}$)")
    ax.set_ylabel("n$_\\mathrm{H}$ (cm$^{-3}$)")
    draw_sample_contours(
        ax=ax,
        figure_data=figure_data,
        slice_index=slice_index,
    )
    if str(figure_data["field_tokens"][field_index]) == "tg_final":
        draw_temperature_contours(
            ax=ax,
            figure_data=figure_data,
            slice_index=slice_index,
        )
    if show_colorbar:
        colorbar = figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(title)


def plot_table_overview(
    figure_data: Mapping[str, np.ndarray],
    *,
    fields: Sequence[str] | None = None,
    ncols: int = 3,
    figsize: tuple[float, float] = (15, 10),
    cmap: str = "viridis",
    show_colorbar: bool = True,
    separate: bool = False,
    slice_index: int = 0,
) -> plt.Figure | list[plt.Figure]:
    """Draw selected prepared fields for one saved dVdr slice.

    figure_data comes from read_table_figure_data(), never from a table reader.
    fields selects saved field tokens; None draws them in their saved order.
    slice_index refers to selected-slice order in the prepared file, not the
    original table index. separate=True returns one figure per selected field.
    """
    saved_tokens = tuple(str(token) for token in figure_data["field_tokens"])
    selected_tokens = saved_tokens if fields is None else tuple(fields)
    if not selected_tokens:
        raise ValueError("At least one field token must be specified for plotting.")
    if separate:
        figures = []
        for token in selected_tokens:
            figure, axis = plt.subplots(figsize=figsize)
            draw_table_panel(
                ax=axis,
                figure_data=figure_data,
                slice_index=slice_index,
                field_index=saved_tokens.index(token),
                cmap=cmap,
                show_colorbar=show_colorbar,
                figure=figure,
            )
            figures.append(figure)
        return figures
    panel_count = len(selected_tokens)
    ncols = max(1, ncols)
    nrows = math.ceil(panel_count / ncols)
    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False,
    )
    for panel_index, token in enumerate(selected_tokens):
        row, column = divmod(panel_index, ncols)
        draw_table_panel(
            ax=axes[row, column],
            figure_data=figure_data,
            slice_index=slice_index,
            field_index=saved_tokens.index(token),
            cmap=cmap,
            show_colorbar=show_colorbar,
            figure=figure,
        )
    for panel_index in range(panel_count, nrows * ncols):
        row, column = divmod(panel_index, ncols)
        axes[row, column].axis("off")
    figure.tight_layout()
    return figure
