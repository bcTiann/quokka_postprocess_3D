"""Failure and domain coverage for explicit-depth cell queries.

Raw map failures and independent Cloudy output-validation failures remain
separate. Their union determines unavailable interpolation support. No table
artifact or cell state is changed, and no acceptance threshold is imposed.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .cloudy_cell_queries import CloudyCellQueries
from .cloudy_sixline_lookup import CloudySixLineLookup

AXIS_NAMES = ('log_NH_attenuation', 'log_nH', 'log_T', 'log_L_model_pc')


def validated_coverage_lookup(table_path: Path, validation_path: Path):
    """Require the validator and packed table to describe the same raw build."""
    raw = CloudySixLineLookup(table_path)
    if raw.model_depth_bounds_pc is None:
        raise ValueError('Coverage requires an explicit-depth table')
    report = json.loads(Path(validation_path).read_text())
    if (report.get('execution_status') != 'completed' or report.get('global_issues')
            or report.get('axis_order') != ','.join(AXIS_NAMES)):
        raise ValueError('Cloudy validation is incomplete or has global provenance issues')
    if ('manifest_sha256' not in raw.metadata
            or str(raw.metadata['manifest_sha256'].item()) != report.get('manifest_sha256')):
        raise ValueError('Packed table and validation refer to different build manifests')
    for name in AXIS_NAMES:
        if not np.array_equal(getattr(raw, name), report.get('axes', {}).get(name)):
            raise ValueError(f'Packed table and validation axis mismatch: {name}')
    if 'provenance_json' not in raw.metadata:
        raise ValueError('Packed table lacks raw-map provenance')
    provenance = json.loads(str(raw.metadata['provenance_json'].item()))
    packed_maps = provenance.get('maps', [])
    validated_maps = report.get('maps', [])
    map_count = raw.log_NH_attenuation.size*raw.log_nH.size*raw.log_L_model_pc.size
    packed_hashes = {item['path']:item['sha256'] for item in packed_maps}
    validated_hashes = {item['path']:item.get('hashes', {}).get('.dat') for item in validated_maps}
    if (len(packed_maps) != map_count or len(validated_maps) != map_count
            or len(packed_hashes) != map_count or len(validated_hashes) != map_count
            or packed_hashes != validated_hashes or any(value is None for value in validated_hashes.values())):
        raise ValueError('Packed table and validation describe different raw-map contents')
    raw_mask = np.asarray(report['raw_map_failure_mask'], dtype=bool)
    diagnostic_mask = np.asarray(report['diagnostic_failure_mask'], dtype=bool)
    if not np.array_equal(raw_mask, raw.failure_mask):
        raise ValueError('Packed failure flags differ from independently checked maps')
    if (diagnostic_mask.shape != raw.failure_mask.shape[1:]
            or report.get('state_count') != diagnostic_mask.size
            or report.get('invalid_state_count') != int(diagnostic_mask.sum())
            or report.get('valid_state_count') != int((~diagnostic_mask).sum())):
        raise ValueError('Independent validation mask or state totals are incomplete')
    checked = copy.copy(raw)
    # Original coefficients and masks on the raw lookup and on disk are
    # retained unchanged. The checked view also preserves sampler invariants.
    checked.failure_mask = raw.failure_mask | diagnostic_mask[None]
    checked.emissivity_per_nH2 = raw.emissivity_per_nH2.copy()
    checked.emissivity_per_nH2[checked.failure_mask] = 0.
    checked.log_emissivity_per_nH2 = raw.log_emissivity_per_nH2.copy()
    checked.log_emissivity_per_nH2[checked.failure_mask] = np.nan
    checked.zero_mask = (checked.emissivity_per_nH2 == 0) & ~checked.failure_mask
    return raw, checked


def classify_cell_queries(queries: CloudyCellQueries, raw: CloudySixLineLookup,
                          checked: CloudySixLineLookup):
    """Count out-of-domain cells without clipping density, T or model depth."""
    use = ~queries.excluded
    flags = {'authorized_excluded': queries.excluded.copy()}
    for name, value, axis in (
        ('density', queries.n_H_cm3, raw.log_nH),
        ('temperature', queries.state.temperature_K, raw.log_T),
    ):
        with np.errstate(invalid='ignore'):
            coordinate = np.log10(value)
        tolerance = 1e-12 * max(1., abs(axis[0]), abs(axis[-1]))
        flags[f'outside_{name}'] = use & ((coordinate < axis[0]-tolerance) | (coordinate > axis[-1]+tolerance))
    lower, upper = raw.model_depth_bounds_pc
    flags['outside_depth'] = use & ((queries.model_depth_pc < lower-4*np.spacing(lower))
                                   | (queries.model_depth_pc > upper+4*np.spacing(upper)))
    outside = flags['outside_density'] | flags['outside_temperature'] | flags['outside_depth']
    flags['outside_any_physical_axis'] = outside
    nh_lower, nh_upper = raw.attenuation_column_bounds_cm2
    flags['attenuation_query_below_grid'] = use & (queries.column_density_H_cm2 < nh_lower)
    flags['attenuation_query_above_grid'] = use & (queries.column_density_H_cm2 > nh_upper)
    eligible = use & ~outside
    masks = {label:np.zeros((len(raw.line_keys), *use.shape), dtype=bool)
             for label in ('raw_failure', 'unavailable')}
    if eligible.any():
        args = (queries.state.temperature_K[eligible], queries.n_H_cm3[eligible],
                queries.column_density_H_cm2[eligible])
        for label, lookup in (('raw_failure', raw), ('unavailable', checked)):
            masks[label][:, eligible] = lookup.diagnose(
                *args, model_depth_pc=queries.model_depth_pc[eligible]).failure_touched
    for label, mask in masks.items():
        flags[f'{label}_any_line'] = np.any(mask, axis=0)
        for index, line in enumerate(raw.line_keys):
            flags[f'{label}:{line}'] = mask[index]
    flags['query_available_all_lines'] = eligible & ~flags['unavailable_any_line']
    return flags


class CellCoverageTotals:
    """Accumulate absolute counts/mass before computing global fractions."""
    def __init__(self):
        self.groups = {}

    def add(self, flags, cold, mass_g):
        cold, mass = np.asarray(cold, dtype=bool), np.asarray(mass_g, dtype=float)
        if cold.shape != mass.shape or not np.isfinite(mass).all() or np.any(mass <= 0):
            raise ValueError('Coverage requires finite positive cell masses and matching shapes')
        if any(np.asarray(mask).dtype != np.bool_ or np.shape(mask) != mass.shape for mask in flags.values()):
            raise ValueError('Coverage flags must be boolean and match cell masses')
        for group, selected in (('all', np.ones_like(cold)), ('cold', cold), ('hot', ~cold)):
            item = self.groups.setdefault(group, dict(cells=0, mass_g=0., flags={}))
            if item['flags'] and set(item['flags']) != set(flags):
                raise ValueError('Coverage categories changed between chunks')
            item['cells'] += int(selected.sum())
            item['mass_g'] += float(mass[selected].sum())
            for name, flag in flags.items():
                count = item['flags'].setdefault(name, dict(cells=0, mass_g=0.))
                take = selected & flag
                count['cells'] += int(take.sum())
                count['mass_g'] += float(mass[take].sum())

    def result(self):
        groups = copy.deepcopy(self.groups)
        for item in groups.values():
            for value in item['flags'].values():
                value['cell_fraction'] = value['cells']/item['cells'] if item['cells'] else None
                value['mass_fraction'] = value['mass_g']/item['mass_g'] if item['mass_g'] else None
        return groups
