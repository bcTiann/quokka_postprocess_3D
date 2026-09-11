#!/usr/bin/env python3
"""Check the cell-query bridge against the accepted full cold-state checkpoint.

Requeries accepted DESPOTIC T/mu, without solving chemistry or rescanning yt.
This validates query construction, not Cloudy interpolation or final spectra.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from quokka2s.cloudy_cell_queries import prepare_cloudy_cell_queries
from quokka2s.tables.io import load_table
from quokka2s.tables.lookup import TableLookup

APPROVED_TABLE_SHA256 = '80e65d7abac34a2c2acd52767a30164420ba2c66bcc3a8ca29cb5191bdce5960'
APPROVED_EXCLUDED_IDS = np.array([48249829, 48774117], dtype=np.int64)
EXPECTED_SHAPE = (256, 256, 2048)
EXPECTED_COLD_COUNT = 479029


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(checkpoint_dir: Path, table_path: Path) -> dict:
    checkpoint_dir, table_path = checkpoint_dir.resolve(), table_path.resolve()
    summary_path = checkpoint_dir/'model_depth_summary.json'
    verification_path = checkpoint_dir/'distribution_verification.json'
    states_path = checkpoint_dir/'cold_model_states.npz'
    summary = json.loads(summary_path.read_text())
    verification = json.loads(verification_path.read_text())
    table_hash = sha256(table_path)
    if table_hash != APPROVED_TABLE_SHA256 or summary['table_sha256'] != table_hash:
        raise ValueError('DESPOTIC table differs from the accepted checkpoint')
    if (Path(summary['dataset']).name != 'plt0655228' or tuple(summary['shape']) != EXPECTED_SHAPE
            or summary['total_cells'] != int(np.prod(EXPECTED_SHAPE))):
        raise ValueError('Exclusions only apply to the approved full snapshot')
    if (verification['status'] != 'passed' or not verification['exactly_the_user_authorized_two_exclusions']
            or verification['actual_invalid_cold_cell_ids'] != APPROVED_EXCLUDED_IDS.tolist()
            or verification['source_table_sha256'] != table_hash):
        raise ValueError('Exclusion verification does not match the accepted policy')
    with np.load(states_path, allow_pickle=False) as source:
        states = {key: np.asarray(source[key]).copy() for key in source.files}
    if (str(states['table_sha256'].item()) != table_hash
            or tuple(states['snapshot_shape']) != EXPECTED_SHAPE):
        raise ValueError('Cold-state checkpoint provenance does not match')
    ids = states['flat_cell_index']
    if (ids.shape != (EXPECTED_COLD_COUNT,) or ids.dtype.kind not in 'iu'
            or np.any(np.diff(ids) <= 0) or ids.min() < 0 or ids.max() >= np.prod(EXPECTED_SHAPE)):
        raise ValueError('Cold cell IDs are incomplete, duplicated or invalid')
    excluded = np.isin(ids, APPROVED_EXCLUDED_IDS)
    if not np.array_equal(ids[~states['valid']], APPROVED_EXCLUDED_IDS):
        raise ValueError('Unexpected invalid cells in the prior checkpoint')
    if np.any(states['T_QUOKKA_K'] >= 3000):
        raise ValueError('Cold checkpoint contains hot cells')
    table = load_table(table_path)
    lookup = TableLookup(table)
    original = tuple(states[k] for k in ('nH', 'NH', 'dVdr'))
    # Same coordinate-only clamp as physics_fields._clip_to_table_domain.
    safe = tuple(np.clip(value, axis.min(), axis.max()) for value, axis in zip(
        original, (table.nH_values, table.col_density_values, table.dVdr_values)))
    temperature = lookup.temperature(*safe)
    mu = lookup.mu(*safe)
    constants = dict(summary['constants'])
    if constants.pop('gamma') != 5/3:
        raise ValueError('The checkpoint does not use the agreed adiabatic index')
    queries = prepare_cloudy_cell_queries(
        states['rho_g_cm3'], states['NH'], states['T_QUOKKA_K'], states['u_erg_cm3'],
        temperature, mu, authorized_excluded=excluded, **constants,
    )
    comparisons = {
        'nH': (queries.n_H_cm3, states['nH']),
        'T_DESPOTIC': (temperature, states['T_DESPOTIC_K']),
        'mu_DESPOTIC': (mu, states['mu_DESPOTIC']),
        'T_used': (queries.state.temperature_K, states['T_used_K']),
        'mu_used': (queries.state.mean_molecular_weight, states['mu_used']),
        'L_J_pc': (queries.state.jeans_length_cm/constants['parsec_cm'], states['L_J_pc']),
        'L_model_pc': (queries.model_depth_pc, states['L_model_pc']),
    }
    max_errors = {}
    for key, (actual, reference) in comparisons.items():
        if not np.allclose(actual, reference, rtol=1e-12, atol=0, equal_nan=True):
            raise ValueError(f'Cell-query construction disagrees with checkpoint: {key}')
        finite = np.isfinite(actual) & np.isfinite(reference) & (reference != 0)
        max_errors[key] = float(np.max(np.abs(actual[finite]/reference[finite]-1), initial=0))
    if not np.array_equal(queries.column_density_H_cm2, states['NH']):
        raise ValueError('Original cell column density changed')
    if not np.array_equal(queries.state.valid, states['valid']):
        raise ValueError('The set of valid paired states changed')
    mass = states['rho_g_cm3']*float(states['cell_volume_cm3'])
    cold_mass = float(mass.sum())
    excluded_mass = float(mass[excluded].sum())
    recorded_excluded_mass = summary['model_depth']['cold_T_QUOKKA_lt3000K']['flags']['invalid_model_state']['mass_g']
    if not np.isclose(excluded_mass, recorded_excluded_mass, rtol=1e-12, atol=0):
        raise ValueError('Excluded mass differs from the accepted full-snapshot report')
    if sha256(table_path) != table_hash:
        raise ValueError('DESPOTIC input changed during validation')
    return dict(
        status='passed', scope='All retained cold cells: accepted DESPOTIC requery and explicit-depth query construction only.',
        cloudy_emissivities_evaluated=False, hot_snapshot_cells_rechecked=False,
        cold_cells=int(ids.size), valid_cold_cells=int(queries.state.valid.sum()),
        excluded_cell_ids=ids[excluded].tolist(), excluded_mass_g=excluded_mass,
        excluded_fraction_of_cold_mass=excluded_mass/cold_mass,
        model_depth_range_pc=[float(queries.model_depth_pc[~excluded].min()), float(queries.model_depth_pc[~excluded].max())],
        maximum_relative_errors=max_errors, original_NH_preserved=True,
        despotic_coordinate_clips={name:int(np.count_nonzero(a!=b)) for name,a,b in zip(('nH','NH','dVdr'),original,safe)},
        source_sha256={str(path):sha256(path) for path in (table_path,summary_path,verification_path,states_path,
            Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'src/quokka2s/cloudy_cell_queries.py')},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--despotic-table', type=Path, required=True)
    parser.add_argument('--output-report', type=Path, required=True)
    args = parser.parse_args()
    if args.output_report.exists():
        raise FileExistsError(args.output_report)
    report = validate_checkpoint(args.checkpoint_dir, args.despotic_table)
    with args.output_report.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k:report[k] for k in ('status','cold_cells','valid_cold_cells','excluded_cell_ids','maximum_relative_errors')},indent=2))


if __name__ == '__main__':
    main()
