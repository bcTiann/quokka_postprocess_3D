"""Plot saved numerical products; no snapshot reading or emission calculation."""
from __future__ import annotations

import argparse
from pathlib import Path

from quokka2s.run_settings import DEFAULT_PLOT_CONFIG, load_plot_config
from quokka2s.emission_results import read_emission_results
from quokka2s.figures.emission_results import draw_emission_products


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
        per_projected_area=not config.raw_luminosity,
        allow_partial=config.allow_partial,
        image_downsample_factor=config.image_downsample_factor,
    )
    if config.titled_output_dir is not None:
        draw_emission_products(
            products=products,
            output_dir=config.titled_output_dir,
            per_projected_area=not config.raw_luminosity,
            allow_partial=config.allow_partial,
            titled=True,
            image_downsample_factor=config.image_downsample_factor,
        )


if __name__ == "__main__":
    main()
