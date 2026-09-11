#!/usr/bin/env python3
"""Run seven fixed-state depth tracks and test nine geometric midpoints each.

This measures depth-interpolation accuracy only. It is not validation of the
coarse density/temperature/radiation axes or authorization to adopt a table.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from quokka2s.tables.abundances import abundance_metadata
from scripts.build_cloudy_sixline_tables import LOG_NH_DENSITY, SED_DIRECTORY_NAME
from scripts.cloudy_model_depth_common import (
    LOG_DEPTH_PC, LOG_T, PC_IN_CM, direct_input, inspect_direct_output,
    reference_log_abundances, sha256,
)

TRACKS = [(7, 1, 21.0), (6, 2, 20.5), (5, 4, 20.0), (4, 8, 19.5),
          (3, 9, 19.0), (4, 10, 18.5), (3, 11, 18.0)]


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run_case(case: dict, base: Path, executable: Path, expected: dict,
             timeout: float, provenance: dict) -> dict:
    root = case['id']
    path = base / root
    text = direct_input(root, log_nH=case['log_nH'], log_T=case['log_T'],
                        log_NH=case['log_NH'], log_L_pc=case['log_L_model_pc'])
    cached = path.with_suffix('.json')
    if cached.exists():
        record = json.loads(cached.read_text())
        if record['provenance'] != provenance or path.with_suffix('.in').read_text() != text:
            raise ValueError(f'Refusing stale pilot checkpoint: {root}')
        if any(sha256(base / name) != digest for name, digest in record['output_sha256'].items()):
            raise ValueError(f'Pilot output changed after checkpoint: {root}')
        return record
    if path.with_suffix('.in').exists():
        raise RuntimeError(f'Incomplete prior case requires inspection before retry: {root}')
    path.with_suffix('.in').write_text(text)
    started = time.monotonic()
    timed_out = False
    with path.with_suffix('.console').open('w') as log:
        try:
            process = subprocess.run([str(executable), '-r', root], cwd=base,
                                     stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out, returncode = True, None
    if path.with_suffix('.out').exists():
        checks = inspect_direct_output(path, log_nH=case['log_nH'], log_T=case['log_T'],
                                       log_L_pc=case['log_L_model_pc'], returncode=returncode,
                                       expected_abundances=expected)
    else:
        checks = {'valid': False, 'issues': ['Cloudy output missing']}
    record = dict(case, elapsed_s=time.monotonic()-started, returncode=returncode,
                  timed_out=timed_out, checks=checks, provenance=provenance,
                  output_sha256={p.name: sha256(p) for p in base.glob(root + '.*')
                                 if p.suffix != '.json'})
    write_json(cached, record)
    return record


def compare_midpoints(records: list[dict]) -> list[dict]:
    comparisons = []
    indexed = {(r['track'], r['depth_index']): r for r in records}
    for track in range(len(TRACKS)):
        for middle in range(1, 19, 2):
            trio = [indexed.get((track, index)) for index in (middle-1, middle, middle+1)]
            if not all(r and r['checks']['valid'] for r in trio):
                comparisons.append(dict(track=track, depth_index=middle, valid=False))
                continue
            left, direct, right = [np.array(r['checks']['emissivity_per_nH2']) for r in trio]
            # Reproduce the map's four-decimal log-coefficient serialization.
            left_log, right_log = np.zeros_like(left), np.zeros_like(right)
            np.log10(left, out=left_log, where=left>0)
            np.log10(right, out=right_log, where=right>0)
            left = np.where(left>0, 10.0 ** np.round(left_log, 4), 0.0)
            right = np.where(right>0, 10.0 ** np.round(right_log, 4), 0.0)
            predicted = (left + right) / 2.0
            positive = (left > 0) & (right > 0)
            predicted[positive] = 10.0 ** ((np.log10(left[positive]) + np.log10(right[positive])) / 2.0)
            relative = [float(abs(p/d-1)) if d>0 else (0.0 if p==0 else None)
                        for p,d in zip(predicted,direct)]
            dex = [float(abs(np.log10(p/d))) if p>0 and d>0 else (0.0 if p==d else None)
                   for p,d in zip(predicted,direct)]
            comparisons.append(dict(track=track, depth_index=middle, valid=True,
                                    log_L_model_pc=trio[1]['log_L_model_pc'],
                                    direct=direct.tolist(), interpolated=predicted.tolist(),
                                    absolute_error=np.abs(predicted-direct).tolist(),
                                    relative_error=relative, absolute_error_dex=dex,
                                    zero_mismatch=((predicted==0)!=(direct==0)).tolist()))
    return comparisons


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cloudy-exe', type=Path, required=True)
    parser.add_argument('--default-abn', type=Path, required=True)
    parser.add_argument('--sed-dir', type=Path, default=root/'runtime/cloudy_eightline/examples/grackle'/SED_DIRECTORY_NAME)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--timeout-seconds', type=float, default=600)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout_seconds <= 0:
        parser.error('Workers and timeout must be positive')
    base = args.output_dir.resolve(); base.mkdir(parents=True, exist_ok=True)
    executable = args.cloudy_exe.resolve()
    source = args.sed_dir.resolve()
    report = json.loads((source/'build_report.json').read_text())
    if report['hm12_log_NH_attenuation'] != [18.,18.5,19.,19.5,20.,20.5,21.]:
        raise ValueError('Unexpected radiation-field grid')
    if report['ism_log_NH_attenuation'] != 21 or report['extinguish_leak'] != 0:
        raise ValueError('Unexpected foreground radiation prescription')
    if any(e['roundtrip']['maximum_relevant_absolute_error_dex']>report['roundtrip_maximum_allowed_error_dex'] for e in report['entries']):
        raise ValueError('SED round-trip validation failed')
    expected_names={'build_report.json'} | {f'logNH{value:g}.{suffix}'
        for value in (18.,18.5,19.,19.5,20.,20.5,21.) for suffix in ('out','sed')}
    if any(not (source/name).is_file() for name in expected_names):
        raise ValueError('Source SED inventory is incomplete')
    destination=base/SED_DIRECTORY_NAME
    if not destination.exists():
        destination.mkdir()
        for name in sorted(expected_names):
            shutil.copy2(source/name, destination/name)
    if {p.name for p in destination.iterdir()} != expected_names:
        raise ValueError('Copied SED inventory differs from the seven-field setup')
    sed_hashes={p.name:sha256(p) for p in destination.iterdir() if p.is_file()}
    if any(sha256(source/name)!=digest for name,digest in sed_hashes.items()):
        raise ValueError('Copied SED differs from source')
    provenance=dict(cloudy_executable_sha256=sha256(executable),
                    default_abn_sha256=sha256(args.default_abn), sed_sha256=sed_hashes,
                    common_code_sha256=sha256(Path(__file__).with_name('cloudy_model_depth_common.py')),
                    pilot_code_sha256=sha256(Path(__file__)), abundance=abundance_metadata())
    expected = reference_log_abundances(args.default_abn)
    depth_axis = np.linspace(LOG_DEPTH_PC[0], LOG_DEPTH_PC[-1], 19)
    cases = [dict(id=f'track{track:02d}_depth{index:02d}', track=track,
                  depth_index=index, role='node' if index%2==0 else 'midpoint',
                  log_nH=LOG_NH_DENSITY[density], log_T=float(LOG_T[temp]),
                  log_NH=column, log_L_model_pc=float(depth))
             for track,(density,temp,column) in enumerate(TRACKS)
             for index,depth in enumerate(depth_axis)]
    start = time.monotonic(); records=[]
    write_json(base/'status.json', dict(status='running', total=len(cases), completed=0))
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_case,c,base,executable,expected,args.timeout_seconds,provenance) for c in cases]
            for future in as_completed(futures):
                record=future.result(); records.append(record)
                write_json(base/'status.json', dict(status='running', total=len(cases), completed=len(records),
                                                   valid=sum(r['checks']['valid'] for r in records),
                                                   elapsed_s=time.monotonic()-start))
                print(f"{len(records)}/{len(cases)} {record['id']} valid={record['checks']['valid']} {record['elapsed_s']:.1f}s", flush=True)
    except BaseException as exc:
        write_json(base/'status.json',dict(status='failed', total=len(cases), completed=len(records),
                                          error=str(exc), elapsed_s=time.monotonic()-start))
        raise
    summary=dict(status='completed', total=len(records), valid=sum(r['checks']['valid'] for r in records),
                 elapsed_s=time.monotonic()-start, workers=args.workers, pc_in_cm=PC_IN_CM,
                 initial_log_L_model_pc=LOG_DEPTH_PC.tolist(), provenance=provenance,
                 records=sorted(records,key=lambda r:r['id']), comparisons=compare_midpoints(records),
                 scope='Depth-only representative pilot; no full-table adoption or global accuracy claim.')
    write_json(base/'summary.json',summary)
    write_json(base/'status.json',{k:summary[k] for k in ('status','total','valid','elapsed_s')})
    print(json.dumps({k:summary[k] for k in ('status','total','valid','elapsed_s')}))


if __name__ == '__main__':
    main()
