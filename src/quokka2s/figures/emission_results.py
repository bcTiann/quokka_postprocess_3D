"""Draw saved emission products; display operations never recalculate emission."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from quokka2s.emission_results import (
    EmissionResults,
    GasPhaseResults,
    ImageResults,
    SpectralResults,
    read_emission_results,
)
from quokka2s.figures.line_labels import LINE_TITLES
from quokka2s.figures.figure_files import save_figure_formats
from quokka2s.figures.gas_phase_spectra import LINE_ORDER, plot_phase_spectrum_overlay

SHORT_VELOCITY_LIMITS_KMS = (-50.0, 50.0)


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
    Example: images.for_line("halpha", "attenuated") supplies one displayed map.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    image_paths = []
    for key in images.line_keys:
        vmax = max(float(images.for_line(key, state).max()) for state in images.dust_state_keys)
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


def plot_line_spectra(
    spectra: SpectralResults,
    output: Path,
    *,
    per_projected_area: bool,
    diagnostic_suffix: str,
    titled: bool,
    formats: tuple[str, ...],
) -> list[Path]:
    """Draw saved total profiles on their original velocity channels [km/s].

    spectra owns line/dust names, total profiles [erg/s/(km/s)] and area [cm^2].
    per_projected_area divides only the display curves by that area. Returns
    one figure per line/format; the visible window remains +/-50 km/s.
    Example: spectra.for_line("halpha", "attenuated", "total") supplies Halpha.
    """
    import matplotlib.pyplot as plt

    if per_projected_area:
        area = spectra.projected_area_cm2
        ylabel = r"$d\Sigma_L/dv$ [erg s$^{-1}$ cm$^{-2}$ (km s$^{-1}$)$^{-1}$]"
    else:
        area = 1.0
        ylabel = r"$dL/dv$ [erg s$^{-1}$ (km s$^{-1}$)$^{-1}$]"
    spectrum_paths = []
    for key in spectra.line_keys:
        intrinsic = spectra.for_line(line=key, dust_state="intrinsic", regime="total")
        attenuated = spectra.for_line(line=key, dust_state="attenuated", regime="total")
        figure, axis = plt.subplots(figsize=(6.2, 3.5))
        if key == "hi21":
            axis.plot(intrinsic.velocity_kms, intrinsic.dL_dv_erg_s_per_kms / area,
                      color="#242424", lw=1.8)
        else:
            axis.plot(intrinsic.velocity_kms, intrinsic.dL_dv_erg_s_per_kms / area,
                      color="#242424", lw=1.5, ls="--", label="Intrinsic")
            axis.plot(attenuated.velocity_kms, attenuated.dL_dv_erg_s_per_kms / area,
                      color="#C53D46", lw=1.8, label="Dust attenuated")
            axis.legend(frameon=False, fontsize=9)
        axis.set_xlim(*SHORT_VELOCITY_LIMITS_KMS)
        axis.set_ylim(bottom=0)
        axis.set_xlabel(r"$v_z$ [km s$^{-1}$]")
        axis.set_ylabel(ylabel)
        if titled:
            axis.set_title(LINE_TITLES[key])
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        axis.grid(alpha=0.18)
        figure.tight_layout()
        spectrum_paths.extend(save_figure_formats(
            figure=figure,
            stem=output / f"spectrum_{key}{diagnostic_suffix}",
            formats=formats,
        ))
        plt.close(figure)
    return spectrum_paths


def plot_gas_phase_comparisons(
    spectra: SpectralResults,
    phase: GasPhaseResults,
    output: Path,
    *,
    diagnostic_suffix: str,
    titled: bool,
    formats: tuple[str, ...],
) -> list[Path]:
    """Draw gas histograms beside named dust-attenuated line spectra.

    spectra and phase come from read_emission_results(), retaining their own
    coordinates and saved dispersions. Returns comparison paths in LINE_ORDER.
    Display branch selection and peak normalization belong to the renderer.
    """
    phase_stem = output / f"gas_phase_spectrum{diagnostic_suffix}"
    return plot_phase_spectrum_overlay(
        line_spectra=spectra,
        gas_phases=phase,
        output_stem=phase_stem,
        line_keys=LINE_ORDER,
        figure_style="full" if titled else "latex",
        formats=formats,
    )


def draw_emission_products(
    products: EmissionResults,
    output_dir: str | Path,
    *,
    per_projected_area: bool = True,
    formats: tuple[str, ...] = ("png", "pdf"),
    allow_partial: bool = False,
    titled: bool = False,
    image_downsample_factor: int = 1,
) -> dict[str, list[Path]]:
    """Draw one figure version from already loaded numerical results.

    Parameters
    ----------
    products : EmissionResults
        read_emission_results() output, shared by paper and titled figures.
    output_dir : str or pathlib.Path
        Destination for this figure version.
    per_projected_area : bool
        Display spectra per saved area [erg/s/cm^2/(km/s)] instead of luminosity.
    formats : tuple of str
        png and/or pdf.
    allow_partial, titled : bool
        Permit diagnostic partial results; add standalone titles.
    image_downsample_factor : int
        Number of native pixels summed along each display axis.

    Returns a dictionary of image, spectrum and gas-phase figure paths.
    Example: draw_emission_products(results, "figures", titled=True).
    """
    if not formats or any(extension not in ("png", "pdf") for extension in formats):
        raise ValueError("formats must contain png and/or pdf")
    if not products.processing_complete and not allow_partial:
        raise ValueError("Partial diagnostic products require allow_partial=True")
    diagnostic_suffix = "" if products.processing_complete else "_partial_diagnostic"

    import matplotlib
    matplotlib.use("Agg")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    images = combine_image_pixels_for_display(
        images=products.images,
        factor=image_downsample_factor,
    )
    image_paths = plot_line_images(
        images=images,
        output=output,
        diagnostic_suffix=diagnostic_suffix,
        titled=titled,
        formats=formats,
    )
    spectrum_paths = plot_line_spectra(
        spectra=products.spectra,
        output=output,
        per_projected_area=per_projected_area,
        diagnostic_suffix=diagnostic_suffix,
        titled=titled,
        formats=formats,
    )
    phase_paths = plot_gas_phase_comparisons(
        spectra=products.spectra,
        phase=products.gas_phases,
        output=output,
        diagnostic_suffix=diagnostic_suffix,
        titled=titled,
        formats=formats,
    )
    return {
        "images": image_paths,
        "spectra": spectrum_paths,
        "gas_phases": phase_paths,
    }


def plot_emission_products(
    products_dir: str | Path,
    output_dir: str | Path,
    *,
    per_projected_area: bool = True,
    formats: tuple[str, ...] = ("png", "pdf"),
    allow_partial: bool = False,
    titled: bool = False,
    image_downsample_factor: int = 1,
) -> dict[str, list[Path]]:
    """Read saved results and draw line images, spectra and gas comparisons.

    products_dir contains the three process NPZ files; output_dir receives
    figures. For two figure versions, read_emission_results() once and call
    draw_emission_products() twice. No simulation inputs are needed.
    """
    products = read_emission_results(directory=products_dir)
    return draw_emission_products(
        products=products,
        output_dir=output_dir,
        per_projected_area=per_projected_area,
        formats=formats,
        allow_partial=allow_partial,
        titled=titled,
        image_downsample_factor=image_downsample_factor,
    )
