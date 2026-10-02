"""Validate artifacts from the historical manifest-based emission workflow.

The current ``quokka2s process`` command reads numerical tables directly.
Older plotting and diagnostic scripts still use an accepted-table manifest and
a Cloudy coverage audit; their checks live here so the active processing path
does not carry that workflow's assumptions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLOUDY_SHA = '4b8576adc6fc06cb0dc784e4fe41a9f7f6d08cede1c88e53e3b5455676705f31'


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _accepted_file(manifest_path, recorded_path, override=None):
    """Find an accepted artifact next to its manifest after a machine transfer."""
    if override is not None:
        return Path(override).resolve()
    recorded = Path(recorded_path)
    adjacent = manifest_path.parent / recorded.name
    if adjacent.is_file():
        return adjacent.resolve()
    return recorded.resolve()


def validate_accepted_inputs(manifest_path, cloudy_path, audit_path, dataset, *,
                             table_override=None, exclusions_override=None,
                             coverage_override=None):
    """Bind accepted numerical artifacts, allowing their paths to move machines.

    The recorded absolute paths are descriptive. File hashes, grid dimensions,
    accepted excluded-cell indices, source hashes, and full-snapshot totals are
    still checked before and during processing.
    """
    manifest_path = Path(manifest_path).resolve()
    cloudy_path = Path(cloudy_path).resolve()
    audit_path = Path(audit_path).resolve()
    dataset = Path(dataset).resolve()
    manifest = json.loads(manifest_path.read_text())
    if 'accepted interpolated DESPOTIC' not in manifest.get('status', ''):
        raise ValueError('An accepted interpolated DESPOTIC manifest is required')
    if dataset.name != Path(manifest['snapshot']).name:
        raise ValueError('Snapshot name differs from the accepted DESPOTIC snapshot')
    table_path = _accepted_file(manifest_path, manifest['table'], table_override)
    exclusion_path = _accepted_file(manifest_path, manifest['excluded_cells_file'], exclusions_override)
    coverage_path = _accepted_file(manifest_path, manifest['coverage_report'], coverage_override)
    for path, expected in ((table_path, manifest['table_sha256']),
                           (exclusion_path, manifest['excluded_cells_sha256']),
                           (cloudy_path, DEFAULT_CLOUDY_SHA)):
        if _sha256(path) != expected:
            raise ValueError(f'Accepted input hash mismatch: {path}')
    shape = tuple(manifest['snapshot_shape'])
    total = int(np.prod(shape))
    if (total != manifest['total_cells'] or
            total - manifest['excluded_cell_count'] != manifest['retained_cells']):
        raise ValueError('Inconsistent accepted cell counts')
    with np.load(exclusion_path, allow_pickle=False) as source:
        ids = np.array(source['flat_cell_index'])
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or
                ids.size != manifest['excluded_cell_count'] or
                np.any(np.diff(ids) <= 0) or np.any(ids < 0) or np.any(ids >= total)):
            raise ValueError('Invalid accepted exclusion IDs')
        if (not np.array_equal(source['snapshot_shape'], shape) or
                str(source['table_sha256']) != manifest['table_sha256'] or
                not np.array_equal(np.ravel_multi_index(source['cell_xyz_index'].T, shape), ids)):
            raise ValueError('Exclusion provenance mismatch')
        excluded_cold = int(np.count_nonzero(source['T_QUOKKA_K'] < 3000))
        if excluded_cold != manifest['excluded_cold_cell_count']:
            raise ValueError('Excluded cold membership differs from acceptance')
    coverage = json.loads(coverage_path.read_text())
    if (coverage['table_sha256'] != manifest['table_sha256'] or
            coverage['total_cells'] != total or tuple(coverage['shape']) != shape):
        raise ValueError('Coverage report provenance mismatch')
    excluded = coverage['groups']['all_cells']['flags']['excluded_after_interpolation']
    if (excluded['cell_count'] != ids.size or not np.isclose(
            excluded['mass_fraction'], manifest['excluded_mass_fraction'], rtol=1e-12, atol=0)):
        raise ValueError('Coverage exclusions differ from acceptance')
    audit = json.loads(audit_path.read_text())
    physics_path = ROOT / 'src/quokka2s/pipeline/prep/physics_fields.py'
    cold_count = coverage['groups']['T_QUOKKA_lt_3000_K']['cell_count']
    audited_hashes = tuple(audit['sources_sha256'].values())
    if (audit.get('status') != 'completed' or audit['full_snapshot_cells'] != total or
            audit['statistics']['hot']['cells'] != total - cold_count or
            DEFAULT_CLOUDY_SHA not in audited_hashes or
            _sha256(physics_path) not in audited_hashes):
        raise ValueError('Default Cloudy audit provenance mismatch')
    for key in ('invalid_temperature', 'outside_nH', 'outside_T', 'any_failed_node',
                'cell_LJ_below_100pc', 'uses_uncapped_old_node', 'corner_depth_diff_gt_0p1percent'):
        if audit['statistics']['hot']['counts'][key]:
            raise ValueError(f'Default Cloudy hot coverage failed: {key}')
    inputs = (manifest_path, table_path, exclusion_path, coverage_path, cloudy_path, audit_path)
    hashes = {str(p.resolve()): _sha256(p) for p in inputs}
    return manifest, coverage, ids, table_path, inputs, hashes


def check_exclusion_queries(ids, excluded_ids, temperature):
    """Reject new invalid queries or exclusions that have become available."""
    excluded = np.isin(ids, excluded_ids, assume_unique=True)
    unavailable = ~np.isfinite(temperature) | (temperature <= 0)
    if not np.array_equal(excluded, unavailable):
        raise ValueError('Fresh DESPOTIC availability differs from the accepted cell exclusions')
    return excluded
