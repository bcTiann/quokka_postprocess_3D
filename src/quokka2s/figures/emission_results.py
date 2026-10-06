"""Plot saved line-luminosity images, integrated spectra, and gas phases.

The plotting stage reads only the three processed ``.npz`` products. It never
opens a simulation snapshot or recalculates an emissivity or line profile.
Run it with ``python -m quokka2s.plot_emission_results --config configs/emission_plot.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quokka2s.figures.line_labels import LINE_TITLES
from quokka2s.figures.figure_files import save_figure_formats
from quokka2s.products import DUST_STATES
from quokka2s.figures.gas_phase_spectra import (
    GasPhaseVelocityProfiles,
    LINE_ORDER,
    LineSpectraForComparison,
    plot_phase_spectrum_overlay,
)

SHORT_VELOCITY_LIMITS_KMS = (-50.0, 50.0)


def read_product_arrays(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Read selected arrays from a processed NPZ file.

    Parameters
    ----------
    path : pathlib.Path
        Product file written by the process command.
    required : tuple of str
        Field names to load.

    Returns
    -------
    dict of str to numpy.ndarray
        Independent array copies, usable after the file is closed.
    """
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in required}


def load_emission_products(products_dir: str | Path) -> tuple[dict, dict]:
    """Read the saved fields needed to draw images and spectra.

    Parameters
    ----------
    products_dir : str or pathlib.Path
        Directory containing images.npz and spectra.npz from process.

    Returns
    -------
    images, spectra : tuple of dict
        Images (2, L, nx, ny) [erg/s per pixel] and spectra (2, L, 2, V)
        [erg/s/(km/s)], with their plotting coordinates. Axis 0 is intrinsic,
        then attenuated; the spectrum's third axis is cold, then hot.
        Process checks the numerical products before saving them.

    Examples
    --------
    ``images, spectra = load_emission_products(products_dir)``.
    ``images["line_luminosity_image_erg_s"][0]`` contains intrinsic images.
    """
    directory = Path(products_dir)
    images = read_product_arrays(
        path=directory / "images.npz",
        required=(
            "line_keys",
            "line_luminosity_image_erg_s",
            "x_edges_kpc",
            "y_edges_kpc",
            "full_snapshot",
        ),
    )
    spectra = read_product_arrays(
        path=directory / "spectra.npz",
        required=(
            "velocity_kms",
            "dL_dv_erg_s_per_kms",
            "projected_area_cm2",
            "line_centroid_window_kms",
            "line_sigma_full_kms",
        ),
    )
    return images, spectra


def load_phase_velocity_products(products_dir: str | Path) -> dict:
    """Read the saved gas profiles and dispersions used in comparison figures.

    Parameters
    ----------
    products_dir : str or pathlib.Path
        Process output directory containing phase_velocity.npz.

    Returns
    -------
    dict of str to numpy.ndarray
        histogram_mass_g (6, V) [g per channel] and two dispersion arrays
        (6,) [km/s]. Rows are CNM, UNM, WNM, WIM, HIM, then all gas.

    Examples
    --------
    ``phase = load_phase_velocity_products(products_dir)``.
    ``phase["histogram_mass_g"][-1]`` is the all-gas velocity profile.
    """
    return read_product_arrays(
        path=Path(products_dir) / "phase_velocity.npz",
        required=(
            "histogram_mass_g",
            "sigma_about_global_mean_kms",
            "sigma_internal_kms",
        ),
    )


def combine_image_pixels_for_display(
    images: dict,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum neighbouring luminosity pixels for display.

    Parameters
    ----------
    images : dict
        load_emission_products() image output: (2, L, nx, ny) [erg/s] and
        x/y pixel edges [kpc].
    factor : int
        Pixels combined along each axis; 1 retains the original arrays.

    Returns
    -------
    values, x_edges, y_edges : tuple of numpy.ndarray
        Images (2, L, nx/factor, ny/factor) [erg/s per new pixel] and matching
        edges [kpc]. Pixel luminosities are summed, not averaged.

    Examples
    --------
    With loaded images, combine 2 by 2 native pixels::

        values, x_edges, y_edges = combine_image_pixels_for_display(images, 2)
    """
    values = images["line_luminosity_image_erg_s"]
    nx, ny = values.shape[-2:]
    if type(factor) is not int or factor <= 0:
        raise ValueError("image_downsample_factor must be a positive integer")
    if nx % factor or ny % factor:
        raise ValueError("image_downsample_factor must divide both image dimensions")
    if factor == 1:
        return values, images["x_edges_kpc"], images["y_edges_kpc"]
    # Split each x/y axis into coarse pixels and the native pixels inside them.
    # Example: factor=2 changes (2, 10, 256, 256) into (2, 10, 128, 2, 128, 2).
    grouped_pixels = values.reshape(
        *values.shape[:2],
        nx // factor,
        factor,
        ny // factor,
        factor,
    )
    binned = grouped_pixels.sum(axis=(3, 5))
    x_edges = images["x_edges_kpc"][::factor]
    y_edges = images["y_edges_kpc"][::factor]
    return binned, x_edges, y_edges


def plot_line_images(
    keys,
    image_values,
    x_edges,
    y_edges,
    output: Path,
    *,
    diagnostic_suffix: str,
    titled: bool,
    formats: tuple[str, ...],
) -> list[Path]:
    """Draw intrinsic and attenuated images with each line's shared colour scale.

    Parameters
    ----------
    keys : tuple of str, length L
        Saved line order from load_emission_products().
    image_values : numpy.ndarray, shape (2, L, nx, ny)
        Luminosities [erg/s per pixel] from combine_image_pixels_for_display().
        Dust state 0 is intrinsic and state 1 is attenuated.
    x_edges, y_edges : numpy.ndarray, shapes (nx + 1,), (ny + 1,)
        Display pixel boundaries [kpc] from the same function.
    output : pathlib.Path
        Figure destination.
    diagnostic_suffix : str
        Empty for full-snapshot products, otherwise "_partial_diagnostic".
    titled : bool
        Include the line name and dust state in each figure title.
    formats : tuple of str
        Output extensions, normally ("png", "pdf").

    Returns
    -------
    list of pathlib.Path
        Two figures per line, except one for HI where dust extinction is zero.
        Each line uses limits (maximum / 1e5, maximum) across its two images.

    Examples
    --------
    ``image_values[0, keys.index('halpha')]`` supplies the intrinsic Halpha
    image; its x/y axes are transposed only when passed to Matplotlib.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    image_paths = []
    for line_index, key in enumerate(keys):
        pair = image_values[:, line_index]
        vmax = float(pair.max())
        norm = None
        if vmax > 0:
            norm = LogNorm(vmin=vmax / 1e5, vmax=vmax)
        image_dust_states = tuple(enumerate(DUST_STATES))
        if key == "hi21":
            # HI has no dust treatment: write one image without a dust-state suffix.
            image_dust_states = ((0, ""),)
        for dust_index, dust_state in image_dust_states:
            figure = create_line_luminosity_image(
                luminosity_image_erg_s=pair[dust_index],
                x_edges_kpc=x_edges,
                y_edges_kpc=y_edges,
                color_scale=norm,
                line_key=key,
                dust_state=dust_state,
                titled=titled,
            )
            image_stem = f"line_luminosity_{key}"
            if dust_state:
                image_stem += f"_{dust_state}"
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


def prepare_total_line_spectra(
    spectra: dict,
    per_projected_area: bool,
) -> tuple[np.ndarray, str]:
    """Sum cold/hot profiles and prepare the plotted physical units.

    Parameters
    ----------
    spectra : dict of str to numpy.ndarray
        Saved spectra.npz arrays from load_emission_products().
    per_projected_area : bool
        Divide each spectrum by projected_area_cm2 when True.

    Returns
    -------
    total_profiles : numpy.ndarray, shape (2, L, 400)
        Intrinsic and attenuated total spectra [erg/s/(km/s)], or
        [erg/s/cm^2/(km/s)] when normalized by the projected area.
    ylabel : str
        Axis label matching the selected units.
    """
    if per_projected_area:
        area = float(spectra["projected_area_cm2"])
        ylabel = r"$d\Sigma_L/dv$ [erg s$^{-1}$ cm$^{-2}$ (km s$^{-1}$)$^{-1}$]"
    else:
        area = 1.0
        ylabel = r"$dL/dv$ [erg s$^{-1}$ (km s$^{-1}$)$^{-1}$]"
    total_profiles = spectra["dL_dv_erg_s_per_kms"].sum(axis=2) / area
    return total_profiles, ylabel


def plot_line_spectra(
    keys,
    spectra: dict,
    output: Path,
    *,
    per_projected_area: bool,
    diagnostic_suffix: str,
    titled: bool,
    formats: tuple[str, ...],
) -> list[Path]:
    """Draw intrinsic and attenuated spectra summed over cold and hot cells.

    Parameters
    ----------
    keys : tuple of str, length L
        Saved line order from load_emission_products().
    spectra : dict of str to numpy.ndarray
        Saved spectra.npz arrays. Profiles have shape (2, L, 2, 400)
        [erg/s/(km/s)]; channel centres are in km/s.
    output : pathlib.Path
        Figure destination.
    per_projected_area : bool
        Divide by saved projected_area_cm2 when True; dust states are unchanged.
    diagnostic_suffix : str
        Empty for full-snapshot products, otherwise "_partial_diagnostic".
    titled : bool
        Include the line name above each figure.
    formats : tuple of str
        Output extensions, normally ("png", "pdf").

    Returns
    -------
    list of pathlib.Path
        One figure per line and format, showing only +/-50 km/s while keeping
        the full saved spectra unchanged. HI has one curve because its dust
        extinction is zero.

    Examples
    --------
    ``spectra['dL_dv_erg_s_per_kms'][1, keys.index('halpha')].sum(axis=0)``
    is the dust-attenuated Halpha curve before optional area normalization.
    """
    import matplotlib.pyplot as plt

    spectrum_paths = []
    velocity = spectra["velocity_kms"]
    total_profiles, ylabel = prepare_total_line_spectra(
        spectra=spectra,
        per_projected_area=per_projected_area,
    )

    for line_index, key in enumerate(keys):
        figure, axis = plt.subplots(figsize=(6.2, 3.5))
        intrinsic = total_profiles[0, line_index]
        attenuated = total_profiles[1, line_index]
        if key == "hi21":
            axis.plot(velocity, intrinsic, color="#242424", lw=1.8)
        else:
            axis.plot(velocity, intrinsic, color="#242424", lw=1.5, ls="--",
                      label="Intrinsic")
            axis.plot(velocity, attenuated, color="#C53D46", lw=1.8,
                      label="Dust attenuated")
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
    keys,
    spectra: dict,
    phase: dict,
    output: Path,
    *,
    diagnostic_suffix: str,
    titled: bool,
    formats: tuple[str, ...],
) -> list[Path]:
    """Draw gas histograms beside saved dust-attenuated line spectra.

    Parameters
    ----------
    keys : tuple of str, length L
        Saved line order from load_emission_products().
    spectra, phase : dict of str to numpy.ndarray
        Saved spectra.npz and phase_velocity.npz arrays from the loaders.
        Their coordinates, profiles, and dispersion arrays are passed directly
        to the renderer; no duplicate total profiles or nested reports are built.
    output : pathlib.Path
        Figure destination.
    diagnostic_suffix : str
        Empty for a complete snapshot, otherwise "_partial_diagnostic".
    titled : bool
        Include the standalone title and phase-cut footer.
    formats : tuple of str
        Output extensions, normally ("png", "pdf").

    Returns
    -------
    list of pathlib.Path
        Ten line figures per format, in LINE_ORDER.

    Examples
    --------
    The Halpha curve comes from ``spectra["dL_dv_erg_s_per_kms"][1]``:
    dust state 1 is attenuated, and the renderer sums its cold and hot profiles.
    """
    line_spectra = LineSpectraForComparison(
        line_keys=tuple(keys),
        velocity_kms=spectra["velocity_kms"],
        dL_dv_erg_s_per_kms=spectra["dL_dv_erg_s_per_kms"][1],
        line_centroid_window_kms=spectra["line_centroid_window_kms"][1],
        line_sigma_full_kms=spectra["line_sigma_full_kms"][1],
    )
    gas_phases = GasPhaseVelocityProfiles(
        histogram_mass_g=phase["histogram_mass_g"],
        sigma_about_global_mean_kms=phase["sigma_about_global_mean_kms"],
        sigma_internal_kms=phase["sigma_internal_kms"],
    )
    phase_stem = output / f"gas_phase_spectrum{diagnostic_suffix}"
    return plot_phase_spectrum_overlay(
        line_spectra=line_spectra,
        gas_phases=gas_phases,
        output_stem=phase_stem,
        line_keys=LINE_ORDER,
        figure_style="full" if titled else "latex",
        formats=formats,
    )


@dataclass(frozen=True)
class SavedEmissionProducts:
    """The three processed numerical files shared by all figure versions.

    Attributes
    ----------
    images : dict of str to numpy.ndarray
        images.npz from load_emission_products(). Luminosities have shape
        (2, 10, 256, 256) [erg/s per pixel]; x/y edges are in kpc.
    spectra : dict of str to numpy.ndarray
        spectra.npz from load_emission_products(). Profiles have shape
        (2, 10, 2, 400) [erg/s/(km/s)], with saved channel centres and
        full-profile line dispersions [km/s].
    gas_phases : dict of str to numpy.ndarray
        phase_velocity.npz from load_phase_velocity_products(). Histograms
        have shape (6, 400) [g per channel], with dispersions [km/s].

    Examples
    --------
    ``products.images["line_luminosity_image_erg_s"][0, 0]`` is the first
    line's intrinsic x-y image. Axis 0 is intrinsic, then attenuated.
    """

    images: dict[str, np.ndarray]
    spectra: dict[str, np.ndarray]
    gas_phases: dict[str, np.ndarray]


def load_plot_products(products_dir: str | Path) -> SavedEmissionProducts:
    """Read the image, spectrum, and gas-phase products once for plotting.

    Parameters
    ----------
    products_dir : str or pathlib.Path
        Directory containing the three NPZ files written by process.

    Returns
    -------
    SavedEmissionProducts
        Saved arrays; see the class for shapes and units.
        The loaded arrays can be reused for paper and titled figure versions.

    Examples
    --------
    ``products = load_plot_products("output/plt0655228/processed")``.
    Plotting never opens the snapshot or the emission lookup tables.
    """
    images, spectra = load_emission_products(products_dir)
    gas_phases = load_phase_velocity_products(
        products_dir=products_dir,
    )
    return SavedEmissionProducts(
        images=images,
        spectra=spectra,
        gas_phases=gas_phases,
    )


def draw_emission_products(
    products: SavedEmissionProducts,
    output_dir: str | Path,
    *,
    per_projected_area: bool = True,
    formats: tuple[str, ...] = ("png", "pdf"),
    allow_partial: bool = False,
    titled: bool = False,
    image_downsample_factor: int = 1,
) -> dict[str, list[Path]]:
    """Draw one figure version from already loaded numerical products.

    Parameters
    ----------
    products : SavedEmissionProducts
        The result of load_plot_products(); shared by paper and titled figures.
    output_dir : str or pathlib.Path
        Destination for this figure version.
    per_projected_area : bool
        Divide spectra by the saved area when True [erg/s/cm^2/(km/s)].
        Otherwise show luminosity spectra [erg/s/(km/s)]. This changes units,
        not the dust state; gas-phase comparisons always use attenuated lines.
    formats : tuple of str
        Output extensions, normally ("png", "pdf").
    allow_partial, titled : bool
        Permit diagnostic subsets; include standalone titles.
    image_downsample_factor : int
        Pixels summed along each x/y axis for display; 1 retains native pixels.

    Returns
    -------
    dict of str to list of pathlib.Path
        Figure paths grouped as images, spectra, and gas_phases.

    Examples
    --------
    ``draw_emission_products(products, figure_dir, titled=True)`` reuses the
    loaded arrays without reading files again or recalculating emission.
    """
    if not formats or any(extension not in ("png", "pdf") for extension in formats):
        raise ValueError("formats must contain png and/or pdf")
    images = products.images
    spectra = products.spectra
    phase = products.gas_phases
    full_snapshot = bool(images["full_snapshot"])
    if not full_snapshot and not allow_partial:
        raise ValueError("Partial diagnostic products require allow_partial=True")
    diagnostic_suffix = ""
    if not full_snapshot:
        diagnostic_suffix = "_partial_diagnostic"

    import matplotlib
    matplotlib.use("Agg")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    keys = tuple(str(key) for key in images["line_keys"])
    image_values, x_edges, y_edges = combine_image_pixels_for_display(
        images=images,
        factor=image_downsample_factor,
    )

    image_paths = plot_line_images(
        keys=keys,
        image_values=image_values,
        x_edges=x_edges,
        y_edges=y_edges,
        output=output,
        diagnostic_suffix=diagnostic_suffix,
        titled=titled,
        formats=formats,
    )
    spectrum_paths = plot_line_spectra(
        keys=keys,
        spectra=spectra,
        output=output,
        per_projected_area=per_projected_area,
        diagnostic_suffix=diagnostic_suffix,
        titled=titled,
        formats=formats,
    )
    phase_paths = plot_gas_phase_comparisons(
        keys=keys,
        spectra=spectra,
        phase=phase,
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
    """Read saved products and draw line images, spectra, and gas comparisons.

    Parameters
    ----------
    products_dir, output_dir : str or pathlib.Path
        Input directory with the three process NPZ files; figure destination.
    per_projected_area : bool
        Show spectra per projected area when True; see draw_emission_products().
    formats : tuple of str
        Output extensions, "png" and/or "pdf".
    allow_partial, titled : bool
        Permit diagnostic subsets; include standalone figure titles.
    image_downsample_factor : int
        Pixels summed along each image axis for display; default 1.

    Returns
    -------
    dict of str to list of pathlib.Path
        Written paths grouped as images, spectra, and gas_phases. The display
        window is +/-50 km/s; gas-phase comparisons use attenuated line spectra.

    Examples
    --------
    ``paths = plot_emission_products(products_dir, figure_dir, titled=True)``.
    To draw both versions, load_plot_products() once and call
    draw_emission_products() for each destination, as the plot command does.
    """
    products = load_plot_products(products_dir)
    return draw_emission_products(
        products=products,
        output_dir=output_dir,
        per_projected_area=per_projected_area,
        formats=formats,
        allow_partial=allow_partial,
        titled=titled,
        image_downsample_factor=image_downsample_factor,
    )
