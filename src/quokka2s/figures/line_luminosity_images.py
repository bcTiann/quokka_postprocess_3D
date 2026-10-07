"""Draw saved line luminosity images and sum pixels only for display."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from quokka2s.emission_results import ImageResults
from quokka2s.figures.figure_files import save_figure_formats
from quokka2s.figures.line_labels import LINE_TITLES


def combine_image_pixels_for_display(images: ImageResults, factor: int) -> ImageResults:
    """Sum neighbouring luminosity pixels into a separate display product.

    images holds native (dust, line, Nx, Ny) light [erg/s] and x/y edges [kpc].
    factor counts pixels combined along each axis; 1 returns the original product.
    Example: factor=2 combines four native luminosities by addition, not averaging.
    """
    values = images.line_luminosity_image_erg_s
    nx, ny = values.shape[-2:]
    if type(factor) is not int or factor <= 0:
        raise ValueError("image_downsample_factor must be a positive integer")
    if nx % factor or ny % factor:
        raise ValueError("image_downsample_factor must divide both image dimensions")
    if factor == 1:
        return images
    # Split each x/y axis into coarse pixels and the native pixels inside them.
    # Example: (2, 10, 256, 256) becomes (2, 10, 128, 2, 128, 2).
    grouped_pixels = values.reshape(
        *values.shape[:2],
        nx // factor,
        factor,
        ny // factor,
        factor,
    )
    binned = grouped_pixels.sum(axis=(3, 5))
    return replace(
        images,
        line_luminosity_image_erg_s=binned,
        x_edges_kpc=images.x_edges_kpc[::factor],
        y_edges_kpc=images.y_edges_kpc[::factor],
    )


def plot_line_images(
    images: ImageResults,
    output: Path,
    *,
    diagnostic_suffix: str,
    titled: bool,
    formats: tuple[str, ...],
) -> list[Path]:
    """Draw paired luminosity images from their own saved names and pixel edges.

    images comes from combine_image_pixels_for_display(); light is erg/s per
    display pixel. Each line shares a colour scale across its dust states.
    Returns image paths; HI writes one image because its adopted dust opacity is zero.
    Example: images.for_line(line="halpha", dust_state="attenuated") is one map.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    image_paths = []
    for key in images.line_keys:
        vmax = max(
            float(images.for_line(line=key, dust_state=state).max())
            for state in images.dust_state_keys
        )
        norm = None
        if vmax > 0:
            norm = LogNorm(vmin=vmax / 1e5, vmax=vmax)
        image_dust_states = images.dust_state_keys
        if key == "hi21":
            image_dust_states = ("intrinsic",)
        for dust_state in image_dust_states:
            display_state = "" if key == "hi21" else dust_state
            figure = create_line_luminosity_image(
                luminosity_image_erg_s=images.for_line(line=key, dust_state=dust_state),
                x_edges_kpc=images.x_edges_kpc,
                y_edges_kpc=images.y_edges_kpc,
                color_scale=norm,
                line_key=key,
                dust_state=display_state,
                titled=titled,
            )
            image_stem = f"line_luminosity_{key}"
            if display_state:
                image_stem += f"_{display_state}"
            image_paths.extend(save_figure_formats(
                figure=figure,
                stem=output / (image_stem + diagnostic_suffix),
                formats=formats,
            ))
            plt.close(figure)
    return image_paths


def create_line_luminosity_image(
    luminosity_image_erg_s: np.ndarray,
    x_edges_kpc: np.ndarray,
    y_edges_kpc: np.ndarray,
    color_scale,
    line_key: str,
    dust_state: str,
    titled: bool,
):
    """Draw one line's x-y image using its shared intrinsic/attenuated scale.

    Parameters
    ----------
    luminosity_image_erg_s : numpy.ndarray, shape (nx, ny)
        One dust state's displayed luminosity [erg/s per pixel].
    x_edges_kpc, y_edges_kpc : numpy.ndarray, shapes (nx + 1,), (ny + 1,)
        Pixel boundaries [kpc] from combine_image_pixels_for_display().
    color_scale : matplotlib.colors.LogNorm or None
        The line's paired colour limits; None means both images are zero.
    line_key, dust_state : str
        Transition name and "intrinsic"/"attenuated"; HI uses an empty state.
    titled : bool
        Include the standalone line/dust title.

    Returns
    -------
    matplotlib.figure.Figure
        Figure ready for save_figure_formats(). Plotting transposes (x, y)
        to Matplotlib's row/column convention without changing saved arrays.
    """
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.2, 4.5))
    image = luminosity_image_erg_s.T  # Matplotlib rows=y, columns=x.
    if color_scale is None:
        axis.set_facecolor("white")
        axis.text(
            0.5,
            0.5,
            "No emission",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    else:
        mesh = axis.pcolormesh(
            x_edges_kpc,
            y_edges_kpc,
            np.ma.masked_less_equal(image, 0.0),
            cmap="magma",
            norm=color_scale,
            shading="flat",
            # Rasterize the pixel grid in PDFs; axes and text remain vector graphics.
            rasterized=True,
        )
    axis.set_xlabel("x [kpc]")
    axis.set_ylabel("y [kpc]")
    axis.set_xlim(x_edges_kpc[[0, -1]])
    axis.set_ylim(y_edges_kpc[[0, -1]])
    axis.set_aspect("equal")
    if titled:
        title = LINE_TITLES[line_key]
        if dust_state == "intrinsic":
            title += " — Intrinsic"
        elif dust_state == "attenuated":
            title += " — Dust attenuated"
        axis.set_title(title)
    if color_scale is not None:
        figure.colorbar(mesh, ax=axis, label=r"Pixel luminosity [erg s$^{-1}$]")
    figure.tight_layout()
    return figure
