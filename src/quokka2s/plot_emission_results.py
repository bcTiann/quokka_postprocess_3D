"""Plot saved numerical products; no snapshot reading or emission calculation."""
from __future__ import annotations

import argparse
from pathlib import Path

from quokka2s.emission_results import EmissionResults, read_emission_results
from quokka2s.figures.gas_phase_spectra import (
    LINE_ORDER,
    plot_phase_spectrum_overlay,
    prepare_gas_phase_profiles,
)
from quokka2s.figures.line_luminosity_images import (
    combine_image_pixels_for_display,
    plot_line_images,
)
from quokka2s.figures.line_spectra import plot_line_spectra
from quokka2s.run_settings import DEFAULT_PLOT_CONFIG, load_plot_config


def draw_emission_products(
    products: EmissionResults,
    output_dir: str | Path,
    *,
    titled_output_dir: str | Path | None = None,
    per_projected_area: bool = True,
    formats: tuple[str, ...] = ("png", "pdf"),
    allow_partial: bool = False,
    image_downsample_factor: int = 1,
) -> dict[str, dict[str, list[Path]]]:
    """Prepare saved products once and write paper and optional titled figures.

    products retains each file's own coordinates and saved moments. The image
    factor sums native pixels along x/y; gas curves use their full saved-channel
    peaks. per_projected_area changes only line-spectrum display units.
    Returns paths grouped by style, then images, spectra and gas_phases.
    Example: titled_output_dir="figures_titled" adds standalone titles.
    """
    if not formats or any(extension not in ("png", "pdf") for extension in formats):
        raise ValueError("formats must contain png and/or pdf")
    if not products.processing_complete and not allow_partial:
        raise ValueError("Partial diagnostic products require allow_partial=True")
    diagnostic_suffix = "" if products.processing_complete else "_partial_diagnostic"
    images = combine_image_pixels_for_display(
        images=products.images,
        factor=image_downsample_factor,
    )
    gas_profiles = prepare_gas_phase_profiles(gas_phases=products.gas_phases)
    # CIII/CIV have zero cold light, so their attenuated totals equal hot profiles.
    attenuated_lines = {
        key: products.spectra.for_line(
            line=key,
            dust_state="attenuated",
            regime="total",
        )
        for key in LINE_ORDER
    }

    import matplotlib
    matplotlib.use("Agg")

    destinations = [("paper", Path(output_dir), False)]
    if titled_output_dir is not None:
        destinations.append(("titled", Path(titled_output_dir), True))
    paths = {}
    for style, output, titled in destinations:
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
        phase_paths = plot_phase_spectrum_overlay(
            line_spectra=attenuated_lines,
            gas_profiles=gas_profiles,
            output_stem=output / f"gas_phase_spectrum{diagnostic_suffix}",
            titled=titled,
            formats=formats,
        )
        paths[style] = {
            "images": image_paths,
            "spectra": spectrum_paths,
            "gas_phases": phase_paths,
        }
    return paths


def main(argv=None) -> None:
    """Plot saved products from the plotting YAML settings.

    argv is a sequence of command arguments, or None for sys.argv[1:].
    From the repository root, the default is configs/emission_plot.yaml;
    --config PATH selects another YAML file."""
    parser = argparse.ArgumentParser(
        prog='quokka2s-plot',
        description=__doc__,
    )
    parser.add_argument(
        '--config',
        default=DEFAULT_PLOT_CONFIG,
        type=Path,
        help='Plotting YAML file (default: configs/emission_plot.yaml)',
    )
    config_path = parser.parse_args(argv).config
    try:
        config = load_plot_config(config_path)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    products = read_emission_results(directory=config.products)
    # raw_luminosity selects erg/s/(km/s) instead of dividing by projected area.
    # It does not select intrinsic versus dust-attenuated emission.
    draw_emission_products(
        products=products,
        output_dir=config.output_dir,
        titled_output_dir=config.titled_output_dir,
        per_projected_area=not config.raw_luminosity,
        allow_partial=config.allow_partial,
        image_downsample_factor=config.image_downsample_factor,
    )


if __name__ == "__main__":
    main()
