"""Run snapshot reading, shared cell emission, and the three product accumulators.

Read this file for the order of work. The imported modules own the input/output
format, cell physics, image sums, spectral integration, and gas-phase statistics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from quokka2s.physics.cell_emission import CellEmissionCalculator
from quokka2s.run_settings import (
    ProcessSettings,
    add_run_selection_arguments,
    load_process_config,
)
from quokka2s.processing_inputs import (
    load_processing_inputs,
    prepare_processing_snapshot,
)
from quokka2s.result_files import write_emission_results, write_status
from quokka2s.result_metadata import ResultMetadata
from quokka2s.processing_report import (
    build_field_preprocessing_report,
    build_processing_report,
)
from quokka2s.process_reuse import processing_input_identity, reuse_completed_results
from quokka2s.products.emission_products import EmissionProducts
from quokka2s.process_logging import format_elapsed, print_process_summary


def accumulate_batch(cells, emission_calculator: CellEmissionCalculator, products,
                     progress=None):
    """Calculate a batch once, then add its emission to every product.

    cells is a CellBatch of six (N,) arrays plus grid positions/volume.
    emission_calculator holds the shared table readers and dust cross-sections.
    products receives image, spectrum, phase, and diagnostic sums in place.
    progress optionally records a stage label before each existing calculation.
    Direct-template workers may retain owned Gaussian input vectors until the
    next bounded query wave. No CellBatch or LineEmission object is retained."""
    products.failure_context = {
        "first_cell_id": cells.first_cell_id,
        "last_cell_id": cells.last_cell_id,
    }
    if progress is not None:
        progress('DESPOTIC/Cloudy lookup, line emissivity and dust attenuation')
    emission = emission_calculator.calculate(cells=cells)
    if progress is not None:
        progress('accumulating luminosity images')
    products.images.add_batch(cells=cells, emission=emission)
    if progress is not None:
        progress('preparing original-cell Gaussian inputs')
    products.spectra.add_batch(cells=cells, emission=emission)
    if progress is not None:
        progress('accumulating gas-phase velocity statistics')
    products.phases.add_batch(cells=cells, emission=emission)
    if progress is not None:
        progress('recording missingness, luminosity and cell accounting')
    products.record_batch(cells=cells, emission=emission)
    # A template partial also owns its bounded Gaussian queue; not cells/emission.


def process_snapshot(
    config,
    snapshot,
    emission_calculator,
    products,
    began,
    *,
    status_began=None,
    native_column_reader=None,
):
    """Read selected original columns and accumulate one snapshot's products.

    snapshot provides dynamic grid geometry and the selected x/y bounds; every
    selected column retains its full z depth. The prepared reader reads original
    FAB fields into bounded RAM tiles. Finished worker waves merge in input order.
    Return read, derivation and calculation timings; products receives the sums.
    """
    from quokka2s.native_column_process import process_native_columns

    if native_column_reader is None:
        raise ValueError("Processing requires its prepared original-field reader")
    return process_native_columns(
        config=config,
        snapshot=snapshot,
        reader=native_column_reader,
        emission_calculator=emission_calculator,
        products=products,
        accumulate_batch=accumulate_batch,
        began=began,
        status_began=status_began,
    )


def print_run_settings(config, config_path: Path) -> None:
    """Print the selected run, resolved data/output paths, and worker settings.

    config is ProcessSettings from load_process_config(); config_path is the
    supplied YAML filename. This startup summary does not open any input data.
    Example: sigma13.v2/plt0916720 saves under that job/snapshot's processed/.
    """
    print(f'Process {config.jobname}/{config.snapshot_name}', flush=True)
    print(f'  Snapshot: {config.dataset}', flush=True)
    print(f'  Output:   {config.output_dir}', flush=True)
    print(f'  Config:   {config_path}', flush=True)
    print(f'  DESPOTIC: {config.despotic_table}', flush=True)
    print(f'  Cloudy:   {config.cloudy_table}', flush=True)
    print(f'  Dust:     {config.dust_opacity_table}', flush=True)
    print(
        f'  Execution: query_chunk={config.query_chunk:,}; '
        f'chunk_workers={config.chunk_workers}; '
        f'read_workers={config.z_read_workers}; derive_threads={config.z_reader_threads}',
        flush=True,
    )


def run_one_snapshot(config: ProcessSettings, config_path: Path) -> None:
    """Calculate, check and save one snapshot's numerical products without drawing.

    config owns resolved paths and one native x/y selection. All accumulators,
    snapshot fields and lookup readers stay local to this call. Original native
    fields are held in bounded RAM tiles without writing a temporary disk copy.
    """
    from quokka2s.products.gaussian.template_engine import prepare_template_engine

    print_run_settings(config=config, config_path=config_path)
    print("\nGaussian templates: preparing stored bin integrals...", flush=True)
    gaussian_engine, gaussian_table_seconds = prepare_template_engine(config.gaussian_grid)
    print(
        f"  ready in {format_elapsed(gaussian_table_seconds)} "
        f"({gaussian_table_seconds:.3f} s; excluded from process time)",
        flush=True,
    )
    began = time.monotonic()
    input_identity = processing_input_identity(config=config)
    snapshot, emission_calculator = load_processing_inputs(config, compact_log=True)
    input_loading_seconds = time.monotonic() - began
    print('  allocating image, spectrum and gas-phase accumulators...', flush=True)
    products = EmissionProducts(
        gaussian_engine=gaussian_engine,
        snapshot=snapshot,
        line_keys=emission_calculator.line_keys,
    )
    write_status(
        output_dir=config.output_dir,
        counts=products.counts,
        began=began,
        state='running',
    )
    try:
        stage = 'preparing original snapshot reader'
        print('  preparing original snapshot reader...', flush=True)
        with prepare_processing_snapshot(
            snapshot=snapshot,
            read_workers=config.z_read_workers,
        ) as (reader, reader_initialization_seconds):
            stage = 'processing cells'
            calculation_began = time.monotonic()
            processing_timings = process_snapshot(
                config=config,
                snapshot=snapshot,
                emission_calculator=emission_calculator,
                products=products,
                began=calculation_began,
                status_began=began,
                native_column_reader=reader,
            )
            processing_timings['input_loading_seconds'] = input_loading_seconds
            processing_timings['field_reader_initialization_seconds'] = reader_initialization_seconds
            processing_timings['field_source'] = 'native-columns'
            processing_timings['gaussian_table_build_seconds_excluded'] = gaussian_table_seconds
            stage = 'building results'
            print('\nAll cells finished', flush=True)
            print('  Expanding Gaussian nodes into 400-bin spectra...', flush=True)
            output_build_began = time.monotonic()
            outputs = products.build_outputs()
            processing_timings['output_construction_seconds'] = time.monotonic() - output_build_began
            # Include final convolution in the Gaussian process timing summary.
            processing_timings['gaussian'] = products.spectra.timing_summary()
            metadata = ResultMetadata.from_processing_inputs(
                snapshot=snapshot,
                emission_calculator=emission_calculator,
            )
            outputs.add_metadata(metadata=metadata)
            stage = 'checking results'
            print('  Checking luminosity conservation and cell counts...', flush=True)
            checking_began = time.monotonic()
            image_conservation = products.check_outputs(outputs)
            checking_seconds = time.monotonic() - checking_began
            stage = 'building report'
            report = build_processing_report(
                config=config,
                snapshot=snapshot,
                emission_calculator=emission_calculator,
                products=products,
                outputs=outputs,
                metadata=metadata,
                image_conservation=image_conservation,
                began=began,
                processing_timings=processing_timings,
            )
            report['field_preprocessing'] = build_field_preprocessing_report(
                config=config,
                snapshot=snapshot,
                processing_timings=processing_timings,
            )
            if processing_input_identity(config=config) != input_identity:
                raise ValueError('Processing inputs or source files changed during calculation')
            report['input_identity'] = input_identity
            stage = 'saving results'
            print('  Saving numerical products...', flush=True)
            saving_began = time.monotonic()
            write_emission_results(
                directory=config.output_dir,
                outputs=outputs,
                report=report,
            )
            saving_seconds = time.monotonic() - saving_began
        # Original-field readers have closed before completion is recorded.
        write_status(
            output_dir=config.output_dir,
            counts=products.counts,
            began=began,
            state=report["status"],
            processing_complete=outputs.processing_complete,
            full_snapshot=outputs.full_snapshot,
        )
        from quokka2s.native_column_process import peak_process_rss_bytes

        print_process_summary(
            timings=processing_timings,
            process_seconds=time.monotonic() - began,
            checking_seconds=checking_seconds,
            saving_seconds=saving_seconds,
            peak_rss_bytes=peak_process_rss_bytes(),
            gaussian_table_seconds=gaussian_table_seconds,
            status=report['status'],
        )
    except BaseException as exc:
        print(f'Failed while {stage}: {type(exc).__name__}: {exc}', flush=True)
        write_status(
            output_dir=config.output_dir,
            counts=products.counts,
            began=began,
            state='failed',
            error=f'{type(exc).__name__}: {exc}',
            stage=stage,
            failure_context=(products.failure_context if stage == "processing cells" else {}),
        )
        raise


def main(argv=None):
    """Process exactly one explicit snapshot and save its numerical products.

    argv supplies command arguments, or None uses sys.argv[1:]. Dataset selection
    requires an absolute snapshot path, from --snapshot or YAML. This script never
    discovers sibling snapshots or calls plotting code. Matching complete products
    are reused; existing conflicting products are preserved and reported.
    """
    parser = argparse.ArgumentParser(
        prog="process_snapshot.py",
        description=__doc__,
    )
    add_run_selection_arguments(parser)
    args = parser.parse_args(argv)
    try:
        config = load_process_config(
            path=args.config,
            dataset=args.snapshot,
            jobname=args.jobname,
            x_index_range=args.x_index_range,
            y_index_range=args.y_index_range,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    if reuse_completed_results(config=config):
        print(f"Process: reusing completed matching results: {config.output_dir}", flush=True)
        return
    run_one_snapshot(config=config, config_path=args.config)


if __name__ == '__main__':
    main()
