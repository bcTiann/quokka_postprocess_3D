"""Add snapshot identity and time to legacy reports before offline plotting.

This explicit preparation command opens yt snapshot metadata only, never cell
fields or emission tables. Plot itself continues to use only saved products.
An explicit dataset path allows a relocated snapshot; matching saved names and
geometry are required, plus Header file state when the report recorded it.
These checks cannot prove source identity for legacy reports without file state.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import stat
import tempfile

from quokka2s.file_provenance import file_sha256


def prepare_plot_metadata(
    *, products: Path, dataset: Path, jobname: str | None = None,
) -> dict:
    """Add validated labels to emission_report.json, preserving numerical files.

    products is an absolute processed-product directory and dataset is an
    explicitly chosen absolute snapshot directory containing Header. jobname
    fills a missing saved job; it cannot replace an existing, different job.
    Time is converted from yt's unit-aware current_time to Myr (million years).
    Return the updated report; replace it atomically only after every check.
    """
    products = Path(products).expanduser()
    dataset = Path(dataset).expanduser()
    if not products.is_absolute() or not dataset.is_absolute():
        raise ValueError('--products and --dataset must both be absolute paths')
    report_path = products / 'emission_report.json'
    if report_path.is_symlink():
        raise ValueError(f'Report must not be a symlink: {report_path}')
    original_text = report_path.read_text()
    report = json.loads(original_text)
    configuration = report.get('run_configuration') or {}

    saved_jobs = [
        value for value in (report.get('jobname'), configuration.get('jobname'))
        if value is not None
    ]
    if jobname is not None:
        saved_jobs.append(jobname)
    if not saved_jobs:
        raise ValueError('Saved jobname is missing; supply --jobname explicitly')
    if any(not isinstance(value, str) or not value.strip() for value in saved_jobs):
        raise ValueError('Job names must be nonempty strings')
    if len(set(saved_jobs)) != 1:
        raise ValueError('Jobname conflicts with the saved report or configuration')

    saved_names = [
        value for value in (
            report.get('snapshot_name'), configuration.get('snapshot_name'),
        )
        if value is not None
    ]
    for saved_dataset in (report.get('dataset'), configuration.get('dataset')):
        if saved_dataset:
            saved_names.append(Path(saved_dataset).name)
    if not saved_names or any(name != dataset.name for name in saved_names):
        raise ValueError('Dataset basename does not match the saved snapshot identity')

    header_path = dataset / 'Header'
    header_before = header_path.stat()
    recorded_files = report.get('input_identity', {}).get('snapshot_files', {}).get('files', [])
    for record in recorded_files:
        if record.get('path') == 'Header':
            if (
                record.get('size_bytes') != header_before.st_size
                or record.get('modified_ns') != header_before.st_mtime_ns
            ):
                raise ValueError('Snapshot Header differs from the saved input identity')
    header_sha256 = file_sha256(header_path)

    import yt

    snapshot = yt.load(str(dataset.resolve()))
    shape = [int(value) for value in snapshot.domain_dimensions]
    if snapshot.max_level != 0:
        raise ValueError('Expected a uniform native-resolution snapshot')
    if report.get('snapshot_grid', {}).get('shape') != shape:
        raise ValueError('Snapshot grid shape differs from the saved report')
    simulation_time_myr = float(snapshot.current_time.to('Myr').value)
    if not math.isfinite(simulation_time_myr) or simulation_time_myr < 0:
        raise ValueError('Snapshot simulation time must be finite and nonnegative')
    saved_time = report.get('simulation_time_myr')
    if saved_time is not None and (
        isinstance(saved_time, bool)
        or not isinstance(saved_time, (int, float))
        or not math.isclose(saved_time, simulation_time_myr, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError('Snapshot time conflicts with the saved simulation_time_myr')

    # Do not attach labels from a Header that changed while metadata was read.
    header_after = header_path.stat()
    if (
        header_before.st_size != header_after.st_size
        or header_before.st_mtime_ns != header_after.st_mtime_ns
        or file_sha256(header_path) != header_sha256
    ):
        raise ValueError('Snapshot Header changed during metadata preparation')
    report['jobname'] = saved_jobs[0]
    report['snapshot_name'] = dataset.name
    report['simulation_time_myr'] = simulation_time_myr
    report['plot_metadata_source'] = {
        'dataset': str(dataset.resolve()),
        'header_sha256': header_sha256,
        'time': "yt dataset.current_time converted to Myr (million years)",
    }
    updated_text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if updated_text == original_text:
        return report
    if report_path.read_text() != original_text:
        raise ValueError('Saved report changed during metadata preparation')
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', dir=products, prefix='.emission_report.', suffix='.tmp',
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            destination.write(updated_text)
        os.chmod(temporary_path, stat.S_IMODE(report_path.stat().st_mode))
        os.replace(temporary_path, report_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--products', type=Path, required=True, help='Absolute processed/ directory')
    parser.add_argument('--dataset', type=Path, required=True, help='Absolute matching snapshot directory')
    parser.add_argument('--jobname', help='Supply only when the saved jobname is absent')
    arguments = parser.parse_args(argv)
    try:
        report = prepare_plot_metadata(
            products=arguments.products,
            dataset=arguments.dataset,
            jobname=arguments.jobname,
        )
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(
        f"Prepared title metadata: {report['jobname']} / {report['snapshot_name']}; "
        f"time = {report['simulation_time_myr']:.6g} Myr",
        flush=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
