"""Run snapshot reading, shared cell emission, and the three product accumulators.

Read this file for the order of work. The imported modules own the input/output
format, cell physics, image sums, spectral integration, and gas-phase statistics.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

# Independent batches already provide parallelism. Set native-library defaults
# before importing NumPy; explicit user thread settings remain respected.
for thread_setting in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ.setdefault(thread_setting, "1")

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


def accumulate_batch(cells, emission_calculator: CellEmissionCalculator, products):
    """Calculate a batch once, then add its emission to every product.

    cells is a CellBatch of six (N,) arrays plus grid positions/volume.
    emission_calculator holds the shared table readers and dust cross-sections.
    products receives image, spectrum, phase, and diagnostic sums in place.
    Direct-template workers may retain owned Gaussian input vectors until the
    next bounded query wave. No CellBatch or LineEmission object is retained."""
    products.failure_context = {
        "first_cell_id": cells.first_cell_id,
        "last_cell_id": cells.last_cell_id,
    }
    emission = emission_calculator.calculate(cells=cells)
    products.images.add_batch(cells=cells, emission=emission)
    products.spectra.add_batch(cells=cells, emission=emission)
    products.phases.add_batch(cells=cells, emission=emission)
    products.record_batch(cells=cells, emission=emission)
    # A template partial also owns its bounded Gaussian queue; not cells/emission.


def print_run_settings(config, config_path: Path) -> None:
    """Print the selected run, resolved data/output paths, and worker settings.

    config is ProcessSettings from load_process_config(); config_path is the
    supplied YAML filename. This startup summary does not open any input data.
    Example: sigma13.v2/plt0916720 saves under that job/snapshot's processed/.
    """
    print(f'Configuration: {config_path}', flush=True)
    print(f'Processing run: {config.jobname}/{config.snapshot_name}', flush=True)
    print(f'Simulation: {config.dataset}', flush=True)
    print(f'Numerical output: {config.output_dir}', flush=True)
    print(
        f'Execution: query_chunk={config.query_chunk:,}; '
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
    print("Gaussian templates: preparing stored bin integrals before process timing...", flush=True)
    gaussian_engine, gaussian_table_seconds = prepare_template_engine(config.gaussian_grid)
    print(f"Gaussian templates ready in {gaussian_table_seconds:.2f} s (excluded from process time).", flush=True)
    began = time.monotonic()
    input_identity = processing_input_identity(config=config)
    snapshot, emission_calculator = load_processing_inputs(config)
    input_loading_seconds = time.monotonic() - began
    print('Outputs: allocating image, spectrum and gas-phase accumulators...', flush=True)
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
        with prepare_processing_snapshot(
            snapshot=snapshot,
            read_workers=config.z_read_workers,
        ) as (reader, reader_initialization_seconds):
            stage = 'processing cells'
            calculation_began = time.monotonic()
            from quokka2s.native_column_process import process_native_columns

            processing_timings = process_native_columns(
                config=config,
                snapshot=snapshot,
                reader=reader,
                emission_calculator=emission_calculator,
                products=products,
                accumulate_batch=accumulate_batch,
                calculation_began=calculation_began,
                process_began=began,
            )
            processing_timings['input_loading_seconds'] = input_loading_seconds
            processing_timings['field_reader_initialization_seconds'] = reader_initialization_seconds
            processing_timings['field_source'] = 'native-columns'
            processing_timings['gaussian_table_build_seconds_excluded'] = gaussian_table_seconds
            stage = 'building results'
            print('Outputs: expanding completed Gaussian nodes into 400-bin spectra and preparing image/gas-phase results...', flush=True)
            output_build_began = time.monotonic()
            outputs = products.build_outputs()
            processing_timings['output_construction_seconds'] = time.monotonic() - output_build_began
            # Include final convolution in the Gaussian process timing summary.
            processing_timings['gaussian'] = products.spectra.timing_summary()
            print(
                f"Outputs prepared in {processing_timings['output_construction_seconds']:.2f} s; "
                f"final 400-bin expansion {processing_timings['gaussian']['snapshot_corner_finalization_seconds']:.2f} s.",
                flush=True,
            )
            metadata = ResultMetadata.from_processing_inputs(
                snapshot=snapshot,
                emission_calculator=emission_calculator,
            )
            outputs.add_metadata(metadata=metadata)
            stage = 'checking results'
            print('Outputs: checking luminosity conservation and cell accounting...', flush=True)
            checking_began = time.monotonic()
            image_conservation = products.check_outputs(outputs)
            print(f"Output checks passed in {time.monotonic() - checking_began:.2f} s.", flush=True)
            stage = 'building report'
            print('Outputs: preparing the processing report...', flush=True)
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
            print(f'Outputs: saving numerical products and report to: {config.output_dir}', flush=True)
            saving_began = time.monotonic()
            write_emission_results(
                directory=config.output_dir,
                outputs=outputs,
                report=report,
            )
            print(f"Numerical products saved in {time.monotonic() - saving_began:.2f} s.", flush=True)
        # Original-field readers have closed before completion is recorded.
        write_status(
            output_dir=config.output_dir,
            counts=products.counts,
            began=began,
            state=report["status"],
            processing_complete=outputs.processing_complete,
            full_snapshot=outputs.full_snapshot,
        )
        print(f'Run finished: {report["status"]}; complete process {time.monotonic() - began:.2f} s, Gaussian template preparation excluded.', flush=True)
        print(f'Saved numerical results: {config.output_dir}', flush=True)
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
    requires an absolute snapshot path, from --dataset or YAML. This script never
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
            dataset=args.dataset,
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
