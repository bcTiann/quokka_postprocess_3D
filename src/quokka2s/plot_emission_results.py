"""Plot saved numerical products; no snapshot reading or emission calculation.

Default output is titled PNG figures, with both +/-50 and +/-200 km/s views.
Format and title flags override the corresponding choices in the shared YAML.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from quokka2s.emission_results import EmissionResults, read_emission_results
from quokka2s.figures.all_line_spectra import plot_all_line_spectra
from quokka2s.figures.display_settings import SPECTRUM_VELOCITY_WINDOWS_KMS
from quokka2s.figures.gas_phase_spectra import (
    LINE_ORDER,
    plot_phase_spectrum_overlay,
)
from quokka2s.figures.line_luminosity_images import plot_line_images
from quokka2s.figures.line_spectra import plot_line_spectra
from quokka2s.run_settings import (
    add_plot_arguments,
    add_run_selection_arguments,
    load_plot_config,
)


def draw_emission_products(
    products: EmissionResults,
    output_dir: str | Path,
    *,
    per_projected_area: bool = True,
    formats: tuple[str, ...] = ("png",),
    styles: tuple[str, ...] = ("titled",),
    allow_partial: bool = False,
) -> dict[str, dict[str, list[Path]]]:
    """Draw the selected formats/styles and both saved-spectrum display windows.

    products retains each file's own coordinates and saved moments. Pixel
    resolution and normalized curves are already prepared and saved.
    per_projected_area selects the saved line-spectrum display units.
    output_dir is the untitled figure directory; the titled version uses its
    sibling with "_titled" appended. Images are drawn once per style, while
    line, gas-phase and all-line spectra each use +/-50 and +/-200 km/s.
    Returns paths grouped by style and product type. For example,
    formats=("png", "pdf"), styles=("untitled", "titled") writes four versions.
    """
    if not formats or any(extension not in ("png", "pdf") for extension in formats):
        raise ValueError("formats must contain png and/or pdf")
    if not styles or any(style not in ("untitled", "titled") for style in styles):
        raise ValueError("styles must contain untitled and/or titled")
    if not products.processing_complete and not allow_partial:
        raise ValueError("Partial diagnostic products require allow_partial=True")
    diagnostic_suffix = "" if products.processing_complete else "_partial_diagnostic"
    images = products.images
    # CIII/CIV have zero cold light, so their attenuated totals equal hot profiles.
    attenuated_lines = {
        key: products.spectra.for_line(
            line=key,
            dust_state="attenuated",
            regime="total",
        )
        for key in LINE_ORDER
    }

    print('Plot: preparing Matplotlib (the first run may build its font cache)...', flush=True)
    import matplotlib
    matplotlib.use("Agg")

    base_output_dir = Path(output_dir)
    paths = {}
    for style in styles:
        titled = style == "titled"
        output = base_output_dir
        if titled:
            output = base_output_dir.with_name(f"{base_output_dir.name}_titled")
        print(f'Figures ({style}): {output}', flush=True)
        image_paths = plot_line_images(
            images=images,
            output=output,
            diagnostic_suffix=diagnostic_suffix,
            titled=titled,
            formats=formats,
        )
        spectrum_paths = []
        phase_paths = []
        all_line_paths = []
        for velocity_limits_kms in SPECTRUM_VELOCITY_WINDOWS_KMS:
            filename_suffix = f"_v{velocity_limits_kms[1]:g}"
            print(f'  Spectrum window: {velocity_limits_kms} km/s', flush=True)
            spectrum_paths.extend(plot_line_spectra(
                spectra=products.spectra,
                output=output,
                per_projected_area=per_projected_area,
                diagnostic_suffix=diagnostic_suffix,
                titled=titled,
                formats=formats,
                velocity_limits_kms=velocity_limits_kms,
                filename_suffix=filename_suffix,
            ))
            phase_paths.extend(plot_phase_spectrum_overlay(
                line_spectra=attenuated_lines,
                gas_phases=products.gas_phases,
                output_stem=output / f"gas_phase_spectrum{diagnostic_suffix}",
                titled=titled,
                formats=formats,
                velocity_limits_kms=velocity_limits_kms,
                filename_suffix=filename_suffix,
            ))
            all_line_paths.extend(plot_all_line_spectra(
                spectra=products.spectra,
                output=output,
                diagnostic_suffix=diagnostic_suffix,
                titled=titled,
                formats=formats,
                velocity_limits_kms=velocity_limits_kms,
                filename_suffix=filename_suffix,
            ))
        paths[style] = {
            "images": image_paths,
            "spectra": spectrum_paths,
            "gas_phases": phase_paths,
            "all_line_spectra": all_line_paths,
        }
        file_count = sum(len(files) for files in paths[style].values())
        print(f'Figures ({style}): saved {file_count} files.', flush=True)
    return paths


def main(argv=None) -> None:
    """Plot the job and snapshot selected by the shared emission YAML.

    argv is a sequence of command arguments, or None for sys.argv[1:].
    From the repository root, the default is configs/emission.yaml;
    --config PATH selects another YAML file. --dataset and --x/--y override the
    YAML selection, locating saved products without accessing snapshots or tables.
    """
    parser = argparse.ArgumentParser(
        prog='plot',
        description=__doc__,
    )
    add_run_selection_arguments(parser)
    add_plot_arguments(parser)
    args = parser.parse_args(argv)
    config_path = args.config
    started = time.perf_counter()
    print(f'Plot configuration: {config_path}', flush=True)
    try:
        config = load_plot_config(
            path=config_path,
            dataset=args.dataset,
            x_index_range=args.x_index_range,
            y_index_range=args.y_index_range,
            figure_formats=args.figure_formats,
            figure_styles=args.figure_styles,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f'Plot: reading saved numerical products from {config.products}', flush=True)
    products = read_emission_results(
        directory=config.products,
        image_downsample_factor=config.image_downsample_factor,
    )
    line_count = len(products.images.line_keys)
    print(f'Plot: loaded {line_count} lines. No simulation or table lookup is needed.', flush=True)
    # raw_luminosity selects saved erg/s/(km/s) instead of light per area.
    # It does not select intrinsic versus dust-attenuated emission.
    paths = draw_emission_products(
        products=products,
        output_dir=config.output_dir,
        per_projected_area=not config.raw_luminosity,
        formats=config.figure_formats,
        styles=config.figure_styles,
        allow_partial=config.allow_partial,
    )
    file_count = sum(len(files) for groups in paths.values() for files in groups.values())
    elapsed_seconds = time.perf_counter() - started
    print(f'Plot finished: saved {file_count} files in {elapsed_seconds:.1f} s.', flush=True)


if __name__ == "__main__":
    main()
