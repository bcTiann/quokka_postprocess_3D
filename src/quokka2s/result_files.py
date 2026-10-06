"""Write completed numerical products and lightweight run status.

The writer receives finished arrays and a report. It never reads a snapshot,
queries a table, adds physical metadata, or changes the numerical results.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quokka2s.products.emission_products import EmissionOutputs


def write_status(output_dir, counts, began, state, **extra):
    """Write progress to output_dir/status.json; this is not a checkpoint.

    counts is products.counts; began is the monotonic run start [s].
    state is "running", "failed" or "completed"; extra holds progress/error details.
    Example: write_status(config.output_dir, products.counts, began, "running").
    """
    data = {
        'status': state,
        'counts': counts,
        'elapsed_seconds': time.monotonic() - began,
        **extra,
    }
    temporary = output_dir / 'status.tmp'
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(output_dir / 'status.json')


def write_emission_results(
    *,
    directory: Path,
    outputs: EmissionOutputs,
    report: dict,
) -> None:
    """Write the completed images, spectra, gas phases and processing report.

    Parameters
    ----------
    directory : pathlib.Path
        Existing process output directory, from ProcessSettings.output_dir.
    outputs : EmissionOutputs
        From products.build_outputs(), with metadata added and checks passed.
        Contains the three numerical NPZ payloads; no cells or lookup objects.
    report : dict
        From build_processing_report(); contains run summaries in JSON form.

    Returns
    -------
    None
        Writes three NPZ files and emission_report.json. The caller marks run
        status completed only after this function returns successfully.

    Example
    -------
    write_emission_results(directory=config.output_dir, outputs=outputs, report=report)
    """
    np.savez_compressed(directory / "images.npz", **outputs.image_payload)
    np.savez_compressed(directory / "spectra.npz", **outputs.spectrum_payload)
    np.savez_compressed(directory / "phase_velocity.npz", **outputs.phase_payload)
    report_path = directory / "emission_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
