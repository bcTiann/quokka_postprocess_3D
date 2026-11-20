"""Plot saved numerical products; no snapshot reading or emission calculation.

Default output is titled PNG figures, with both +/-50 and +/-200 km/s views.
CLI flags override format, title style and worker settings in the shared YAML.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
import multiprocessing
import os
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
    PlotSettings,
    add_plot_arguments,
    add_run_selection_arguments,
    load_plot_config,
)


def format_run_label(products: EmissionResults) -> str:
    """Format saved run identity; Myr is simulation age, not processing time."""
    jobname = products.jobname or "jobname unavailable"
    snapshot_name = products.snapshot_name or "snapshot unavailable"
    if products.simulation_time_myr is None:
        time_label = "time = unavailable"
    else:
        time_label = f"time = {products.simulation_time_myr:.2f} Myr"
    return f"{jobname} | {snapshot_name} | {time_label}"


def draw_emission_products(
    products: EmissionResults,
    output_dir: str | Path,
    *,
    per_projected_area: bool = True,
    formats: tuple[str, ...] = ("png",),
    styles: tuple[str, ...] = ("titled",),
    allow_partial: bool = False,
    plot_workers: int = 1,
) -> dict[str, dict[str, list[Path]]]:
    """Draw the selected formats/styles and both saved-spectrum display windows.

    products retains each file's own coordinates and saved moments. Pixel
    resolution and normalized curves are already prepared and saved.
    per_projected_area selects the saved line-spectrum display units.
    output_dir is the untitled figure directory; the titled version uses its
    sibling with "_titled" appended. Images are drawn once per style, while
    line, gas-phase and all-line spectra each use +/-50 and +/-200 km/s.
    plot_workers limits rendering processes. One worker renders in this process.
    Returns paths grouped by style and product type. For example,
    formats=("png", "pdf"), styles=("untitled", "titled") writes four versions.
    """
    if not formats or any(extension not in ("png", "pdf") for extension in formats):
        raise ValueError("formats must contain png and/or pdf")
    if not styles or any(style not in ("untitled", "titled") for style in styles):
        raise ValueError("styles must contain untitled and/or titled")
    if not products.processing_complete and not allow_partial:
        raise ValueError("Partial diagnostic products require allow_partial=True")
    if type(plot_workers) is not int or plot_workers < 1:
        raise ValueError("plot_workers must be a positive integer")
    diagnostic_suffix = "" if products.processing_complete else "_partial_diagnostic"
    images = products.images
    run_label = format_run_label(products)
    # CIII/CIV have zero cold light, so their attenuated totals equal hot profiles.
    attenuated_lines = {
        key: products.spectra.for_line(
            line=key,
            dust_state="attenuated",
            regime="total",
        )
        for key in LINE_ORDER
    }

    if hasattr(os, "sched_getaffinity"):
        available_cpus = len(os.sched_getaffinity(0))
    else:
        available_cpus = os.cpu_count() or 1
    task_count = len(images.line_keys) + 3 * len(SPECTRUM_VELOCITY_WINDOWS_KMS)
    worker_count = min(plot_workers, available_cpus, task_count)

    format_label = " + ".join(extension.upper() for extension in formats)
    window_label = ", ".join(f"±{upper:g}" for _, upper in SPECTRUM_VELOCITY_WINDOWS_KMS)
    worker_label = "worker" if worker_count == 1 else "workers"
    print(
        f"  Rendering   {worker_count} {worker_label}  |  {format_label}  |  {window_label} km/s",
        flush=True,
    )
    import matplotlib

    render_pool = nullcontext()
    if worker_count == 1:
        matplotlib.use("Agg")
    else:
        # Build the font cache once before workers start using it.
        import matplotlib.font_manager

        render_pool = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=matplotlib.use,
            initargs=("Agg",),
        )

    base_output_dir = Path(output_dir)
    paths = {}
    with render_pool as executor:
        for style in styles:
            style_started = time.perf_counter()
            titled = style == "titled"
            output = base_output_dir
            if titled:
                output = base_output_dir.with_name(f"{base_output_dir.name}_titled")
            print(f"  {style.capitalize():<11} {output}", flush=True)

            image_paths = []
            spectrum_paths = []
            phase_paths = []
            all_line_paths = []
            render_jobs = []
            image_batches = (images,)
            if executor is not None:
                # Send each worker only the image planes it draws.
                image_batches = tuple(images.select_line(line=key) for key in images.line_keys)
            for image_batch in image_batches:
                draw_images = partial(
                    plot_line_images,
                    images=image_batch,
                    output=output,
                    diagnostic_suffix=diagnostic_suffix,
                    titled=titled,
                    formats=formats,
                    run_label=run_label,
                )
                render_jobs.append((image_paths, draw_images))

            for velocity_limits_kms in SPECTRUM_VELOCITY_WINDOWS_KMS:
                filename_suffix = f"_v{velocity_limits_kms[1]:g}"
                draw_spectra = partial(
                    plot_line_spectra,
                    spectra=products.spectra,
                    output=output,
                    per_projected_area=per_projected_area,
                    diagnostic_suffix=diagnostic_suffix,
                    titled=titled,
                    formats=formats,
                    velocity_limits_kms=velocity_limits_kms,
                    filename_suffix=filename_suffix,
                    run_label=run_label,
                )
                render_jobs.append((spectrum_paths, draw_spectra))
                draw_phases = partial(
                    plot_phase_spectrum_overlay,
                    line_spectra=attenuated_lines,
                    gas_phases=products.gas_phases,
                    output_stem=output / f"gas_phase_spectrum{diagnostic_suffix}",
                    titled=titled,
                    formats=formats,
                    velocity_limits_kms=velocity_limits_kms,
                    filename_suffix=filename_suffix,
                    run_label=run_label,
                )
                render_jobs.append((phase_paths, draw_phases))
                draw_all_lines = partial(
                    plot_all_line_spectra,
                    spectra=products.spectra,
                    output=output,
                    diagnostic_suffix=diagnostic_suffix,
                    titled=titled,
                    formats=formats,
                    velocity_limits_kms=velocity_limits_kms,
                    filename_suffix=filename_suffix,
                    run_label=run_label,
                )
                render_jobs.append((all_line_paths, draw_all_lines))

            if executor is None:
                for figure_paths, draw in render_jobs:
                    figure_paths.extend(draw())
            else:
                # Start the ten-figure batches first so short jobs can fill idle workers.
                long_renderers = (plot_line_spectra, plot_phase_spectrum_overlay)
                render_jobs.sort(key=lambda job: job[1].func not in long_renderers)
                submitted = [
                    (figure_paths, executor.submit(draw))
                    for figure_paths, draw in render_jobs
                ]
                for figure_paths, future in submitted:
                    figure_paths.extend(future.result())

            paths[style] = {
                "images": image_paths,
                "spectra": spectrum_paths,
                "gas_phases": phase_paths,
                "all_line_spectra": all_line_paths,
            }
            if len(styles) > 1:
                file_count = sum(len(files) for files in paths[style].values())
                style_elapsed = time.perf_counter() - style_started
                print(f"              {file_count} files in {style_elapsed:.2f} s", flush=True)
    return paths


def plot_one_snapshot(config: PlotSettings) -> int:
    """Read one saved snapshot/region and return the number of figure files.

    config contains resolved product/output paths and display choices only.
    This independent plotting script reads only saved numerical arrays.
    """
    started = time.perf_counter()
    print(f"  Products    {config.products}", flush=True)
    products = read_emission_results(
        directory=config.products,
        image_downsample_factor=config.image_downsample_factor,
    )
    # raw_luminosity selects saved erg/s/(km/s) instead of light per area.
    # It does not select intrinsic versus dust-attenuated emission.
    paths = draw_emission_products(
        products=products,
        output_dir=config.output_dir,
        per_projected_area=not config.raw_luminosity,
        formats=config.figure_formats,
        styles=config.figure_styles,
        allow_partial=config.allow_partial,
        plot_workers=config.plot_workers,
    )
    file_count = sum(len(files) for groups in paths.values() for files in groups.values())
    elapsed_seconds = time.perf_counter() - started
    print(f"  Finished    {file_count} files in {elapsed_seconds:.2f} s", flush=True)
    return file_count


def main(argv=None) -> None:
    """Plot one saved snapshot/region without opening snapshot or table inputs."""
    parser = argparse.ArgumentParser(
        prog="plot_emission_results.py",
        description=__doc__,
    )
    add_run_selection_arguments(parser)
    add_plot_arguments(parser)
    args = parser.parse_args(argv)
    try:
        config = load_plot_config(
            path=args.config,
            dataset=args.snapshot,
            jobname=args.jobname,
            x_index_range=args.x_index_range,
            y_index_range=args.y_index_range,
            figure_formats=args.figure_formats,
            figure_styles=args.figure_styles,
            plot_workers=args.plot_workers,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print("Emission plots", flush=True)
    print(f"  Config      {args.config}", flush=True)
    plot_one_snapshot(config=config)


if __name__ == "__main__":
    main()
