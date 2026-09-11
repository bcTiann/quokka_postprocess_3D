#!/usr/bin/env python3
"""Extend the seven depth tracks from 19 to 37 direct samples by reusing data.

Only the 126 odd-index states on the new axis are run. They are common held-out
checks for both the original ten-node and candidate nineteen-node interpolants.
No internal zoning or physical settings are changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from quokka2s.tables.abundances import abundance_metadata
from scripts.build_cloudy_sixline_tables import LOG_NH_DENSITY, SED_DIRECTORY_NAME
from scripts.cloudy_model_depth_common import (
    LOG_T, PC_IN_CM, direct_input, reference_log_abundances, sha256,
)
from scripts.validate_cloudy_model_depth_pilot import TRACKS, run_case, write_json

REFINED_LOG_L_PC = np.linspace(-0.25, 2.0, 37)
PHYSICAL_FIELDS = ('log_nH', 'log_T', 'log_NH', 'log_L_model_pc')


def verify_parent(parent: Path, executable: Path, default_abn: Path) -> tuple[dict, list[dict]]:
    """Verify every old input/output before admitting it as a reused node."""
    summary = json.loads((parent / 'summary.json').read_text())
    if summary['status'] != 'completed' or summary['total'] != 133:
        raise ValueError('Parent must be the completed 133-state pilot')
    provenance = summary['provenance']
    current = dict(cloudy_executable_sha256=sha256(executable),
                   default_abn_sha256=sha256(default_abn),
                   common_code_sha256=sha256(Path(__file__).with_name('cloudy_model_depth_common.py')),
                   pilot_code_sha256=sha256(Path(__file__).with_name('validate_cloudy_model_depth_pilot.py')),
                   abundance=abundance_metadata())
    if any(provenance[key] != value for key, value in current.items()):
        raise ValueError('Executable, composition or old pilot implementation changed')
    expected_sed = {'build_report.json'} | {
        f'logNH{column:g}.{extension}' for _, _, column in TRACKS for extension in ('sed', 'out')}
    if set(provenance['sed_sha256']) != expected_sed:
        raise ValueError('Parent SED inventory is incomplete')
    for name, digest in provenance['sed_sha256'].items():
        if sha256(parent / SED_DIRECTORY_NAME / name) != digest:
            raise ValueError(f'Parent SED changed: {name}')
    indexed = {}
    for record in summary['records']:
        key = (record['track'], record['depth_index'])
        if key in indexed:
            raise ValueError(f'Duplicate parent coordinates: {key}')
        track, old_index = key
        if not 0 <= track < len(TRACKS) or not 0 <= old_index < 19:
            raise ValueError(f'Unexpected parent coordinates: {key}')
        density, temperature, column = TRACKS[track]
        expected = dict(log_nH=LOG_NH_DENSITY[density], log_T=float(LOG_T[temperature]),
                        log_NH=column, log_L_model_pc=float(REFINED_LOG_L_PC[2 * old_index]))
        if any(record[name] != value for name, value in expected.items()):
            raise ValueError(f'Parent input coordinates differ: {key}')
        if record['provenance'] != provenance:
            raise ValueError(f'Parent case has inconsistent provenance: {key}')
        case_path = parent / (record['id'] + '.json')
        if json.loads(case_path.read_text()) != record:
            raise ValueError(f'Parent checkpoint differs from summary: {key}')
        required_outputs = {record['id'] + ext for ext in ('.in', '.out')}
        if record['checks']['valid']:
            required_outputs |= {record['id'] + ext for ext in ('.radius', '.physical', '.lines')}
        if not required_outputs <= set(record['output_sha256']):
            raise ValueError(f'Parent output evidence is incomplete: {key}')
        for name, digest in record['output_sha256'].items():
            if sha256(parent / name) != digest:
                raise ValueError(f'Parent output changed: {name}')
        old_input = direct_input(record['id'], log_nH=expected['log_nH'], log_T=expected['log_T'],
                                 log_NH=expected['log_NH'], log_L_pc=expected['log_L_model_pc'])
        if (parent / (record['id'] + '.in')).read_text() != old_input:
            raise ValueError(f'Parent input no longer matches the requested physics: {key}')
        indexed[key] = dict(record, original_depth_index=old_index,
                            original_role=record['role'], role='candidate_19_node',
                            depth_index=2 * old_index, origin='reused_pilot',
                            source_record_path=str(case_path.resolve()))
    if set(indexed) != {(track, index) for track in range(7) for index in range(19)}:
        raise ValueError('Parent coordinate grid is incomplete')
    return summary, [indexed[key] for key in sorted(indexed)]


def new_cases() -> list[dict]:
    return [dict(id=f'track{track:02d}_newdepth{index:02d}', track=track,
                 depth_index=index, role='independent_holdout', origin='new_direct',
                 log_nH=LOG_NH_DENSITY[density], log_T=float(LOG_T[temp]),
                 log_NH=column, log_L_model_pc=float(REFINED_LOG_L_PC[index]))
            for track, (density, temp, column) in enumerate(TRACKS)
            for index in range(1, 37, 2)]


def interpolate_coefficients(left, right, fraction: float) -> np.ndarray:
    """Match table map rounding and the runtime's log/true-zero policy."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    if (left.shape != right.shape or not np.isfinite([left, right]).all()
            or np.any(left < 0) or np.any(right < 0) or not 0 <= fraction <= 1):
        raise ValueError('Invalid interpolation endpoints or fraction')
    logs = []
    rounded = []
    for value in (left, right):
        log = np.zeros_like(value)
        np.log10(value, out=log, where=value > 0)
        log = np.round(log, 4)
        logs.append(log)
        rounded.append(np.where(value > 0, 10.0 ** log, 0.0))
    result = rounded[0] * (1 - fraction) + rounded[1] * fraction
    positive = (left > 0) & (right > 0)
    result[positive] = 10.0 ** (logs[0][positive] * (1-fraction) + logs[1][positive] * fraction)
    return result


def compare_grid(records: list[dict], node_stride: int, evaluation_indices: list[int]) -> list[dict]:
    if node_stride not in (2, 4):
        raise ValueError('Only the approved ten- and nineteen-node candidates are supported')
    indexed = {(r['track'], r['depth_index']): r for r in records}
    if len(indexed) != len(records):
        raise ValueError('Duplicate refined coordinates')
    comparisons = []
    for track in range(len(TRACKS)):
        for index in evaluation_indices:
            if not 0 < index < 36 or index % node_stride == 0:
                raise ValueError('Evaluation coordinates must be independent interior points')
            lower = index // node_stride * node_stride
            upper = lower + node_stride
            fraction = (index - lower) / node_stride
            trio = [indexed.get((track, q)) for q in (lower, index, upper)]
            item = dict(track=track, depth_index=index,
                        log_L_model_pc=float(REFINED_LOG_L_PC[index]),
                        node_count=36 // node_stride + 1,
                        bracket_indices=[lower, upper], fraction=fraction,
                        valid=all(r is not None and r['checks']['valid'] for r in trio))
            if item['valid']:
                left, direct, right = [np.asarray(r['checks']['emissivity_per_nH2']) for r in trio]
                predicted = interpolate_coefficients(left, right, fraction)
                ratio = [float(p/d) if d > 0 else (1.0 if p == 0 else None)
                         for p, d in zip(predicted, direct)]
                signed_dex = [float(np.log10(p) - np.log10(d)) if p>0 and d>0
                              else (0.0 if p == d else None) for p,d in zip(predicted,direct)]
                item.update(direct=direct.tolist(), interpolated=predicted.tolist(), ratio=ratio,
                            absolute_error=np.abs(predicted-direct).tolist(),
                            relative_error=[abs(v-1) if v is not None else None for v in ratio],
                            signed_error_dex=signed_dex,
                            absolute_error_dex=[abs(v) if v is not None else None for v in signed_dex],
                            zero_mismatch=((predicted==0)!=(direct==0)).tolist())
            comparisons.append(item)
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cloudy-exe', type=Path, required=True)
    parser.add_argument('--default-abn', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--timeout-seconds', type=float, default=600)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout_seconds <= 0:
        parser.error('Workers and timeout must be positive')
    parent, base = args.parent_dir.resolve(), args.output_dir.resolve()
    if parent == base:
        raise ValueError('Refinement output must be separate from the frozen parent pilot')
    executable, default_abn = args.cloudy_exe.resolve(), args.default_abn.resolve()
    parent_summary, reused = verify_parent(parent, executable, default_abn)
    base.mkdir(parents=True, exist_ok=True)
    # A live runner owns this OS lock. Its release does not rely on a stale PID
    # or status file, so simultaneous invocations cannot mix Cloudy outputs.
    runner_lock = (base / '.runner.lock').open('a')
    try:
        fcntl.flock(runner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        runner_lock.close()
        raise RuntimeError('A refinement runner is already active in this directory')
    destination = base / SED_DIRECTORY_NAME
    if not destination.exists():
        shutil.copytree(parent / SED_DIRECTORY_NAME, destination)
    old_provenance = parent_summary['provenance']
    if {p.name for p in destination.iterdir()} != set(old_provenance['sed_sha256']):
        raise ValueError('Refinement SED inventory differs from the parent')
    for name, digest in old_provenance['sed_sha256'].items():
        if sha256(destination/name) != digest:
            raise ValueError(f'Refinement SED changed: {name}')
    provenance = dict(old_provenance, parent_summary_sha256=sha256(parent/'summary.json'),
                      refinement_code_sha256=sha256(Path(__file__)))
    expected = reference_log_abundances(default_abn)
    cases = new_cases()
    plan = dict(authorization='User approved the proposed length-only 37-point scan with "ok".',
                log_L_model_pc=REFINED_LOG_L_PC.tolist(), reused=133, new=126,
                comparison='Ten and nineteen nodes evaluated on the same 126 new odd-index holdouts.',
                internal_zoning='Unchanged; the earlier ten-model zoning proposal is not executed.',
                provenance=provenance, cases=cases)
    plan_path = base / 'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError('Existing refinement plan changed; refusing to mix runs')
    else:
        write_json(plan_path, plan)
    start = time.monotonic()
    new_records = []
    status = dict(status='running', pid=os.getpid(), new_total=126, new_completed=0,
                  reused=133, combined_total=259, workers=args.workers)
    write_json(base/'status.json', status)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_case, case, base, executable, expected,
                                   args.timeout_seconds, provenance) for case in cases]
            for future in as_completed(futures):
                record = future.result()
                record = dict(record, source_record_path=str((base/(record['id']+'.json')).resolve()))
                new_records.append(record)
                status.update(new_completed=len(new_records),
                              new_valid=sum(r['checks']['valid'] for r in new_records),
                              elapsed_s=time.monotonic()-start)
                write_json(base/'status.json', status)
                print(f"{len(new_records)}/126 {record['id']} valid={record['checks']['valid']} {record['elapsed_s']:.1f}s", flush=True)
        records = sorted(reused + new_records, key=lambda r:(r['track'],r['depth_index']))
        summary = dict(status='completed', reused=133, new=126, total=len(records),
                       new_valid=sum(r['checks']['valid'] for r in new_records),
                       valid=sum(r['checks']['valid'] for r in records), elapsed_s=time.monotonic()-start,
                       workers=args.workers, log_L_model_pc=REFINED_LOG_L_PC.tolist(),
                       pc_in_cm=PC_IN_CM, provenance=provenance, records=records,
                       comparisons_10=compare_grid(records,4,list(range(1,37,2))),
                       comparisons_19=compare_grid(records,2,list(range(1,37,2))),
                       original_10_midpoint_comparisons=compare_grid(records,4,list(range(2,37,4))),
                       scope='Length-only representative test; 37-node interpolation and full snapshot impact are not validated.')
        write_json(base/'summary.json',summary)
        status.update(status='completed',elapsed_s=summary['elapsed_s'],valid=summary['valid'])
        write_json(base/'status.json',status)
        print(json.dumps(status),flush=True)
    except BaseException as exc:
        status.update(status='failed',error=str(exc),elapsed_s=time.monotonic()-start)
        write_json(base/'status.json',status)
        raise
    finally:
        runner_lock.close()


if __name__ == '__main__':
    main()
