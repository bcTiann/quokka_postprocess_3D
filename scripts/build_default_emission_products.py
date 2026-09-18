#!/usr/bin/env python3
"""Build LOS-z spectra from accepted default-abundance tables and exact exclusions.

This entry point leaves historical products and global pipeline defaults intact.
The legacy Cloudy table is allowed only for hot cells whose model and supporting
table nodes reach the 100 pc cap. Remaining DESPOTIC exclusions apply to all lines.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from quokka2s.cloudy_cell_queries import prepare_cloudy_cell_queries
from quokka2s.cloudy_sixline_lookup import CloudySixLineLookup
from quokka2s.tables.io import load_table
from quokka2s.tables.lookup import TableLookup
from scripts.check_despotic_snapshot_coverage import slab_windows, _validate_scan_provenance, _sha256

DEFAULT_CLOUDY_SHA = '4b8576adc6fc06cb0dc784e4fe41a9f7f6d08cede1c88e53e3b5455676705f31'


def validate_accepted_inputs(manifest_path, cloudy_path, audit_path, dataset):
    """Bind the accepted table, exact exclusion inventory, snapshot and audit."""
    manifest = json.loads(manifest_path.read_text())
    if 'accepted interpolated DESPOTIC' not in manifest.get('status', ''):
        raise ValueError('An accepted interpolated DESPOTIC manifest is required')
    if Path(manifest['snapshot']).resolve() != dataset.resolve():
        raise ValueError('Exclusions belong to another snapshot')
    table_path = Path(manifest['table'])
    exclusion_path = Path(manifest['excluded_cells_file'])
    coverage_path = Path(manifest['coverage_report'])
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
            coverage['total_cells'] != total or tuple(coverage['shape']) != shape or
            Path(coverage['dataset']).resolve() != dataset.resolve()):
        raise ValueError('Coverage report provenance mismatch')
    excluded = coverage['groups']['all_cells']['flags']['excluded_after_interpolation']
    if (excluded['cell_count'] != ids.size or not np.isclose(
            excluded['mass_fraction'], manifest['excluded_mass_fraction'], rtol=1e-12, atol=0)):
        raise ValueError('Coverage exclusions differ from acceptance')
    audit = json.loads(audit_path.read_text())
    physics_path = ROOT / 'src/quokka2s/pipeline/prep/physics_fields.py'
    cold_count = coverage['groups']['T_QUOKKA_lt_3000_K']['cell_count']
    if (audit.get('status') != 'completed' or audit['full_snapshot_cells'] != total or
            audit['statistics']['hot']['cells'] != total - cold_count or
            audit['sources_sha256'].get(str(cloudy_path.resolve())) != DEFAULT_CLOUDY_SHA or
            audit['sources_sha256'].get(str(physics_path)) != _sha256(physics_path)):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('accepted-despotic', 'cloudy-table', 'cloudy-audit', 'dataset', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--slab-nx', type=int, default=8)
    parser.add_argument('--query-chunk', type=int, default=100000)
    parser.add_argument('--velocity-range-kms', type=float, default=200.)
    parser.add_argument('--channels', type=int, default=300)
    parser.add_argument('--spectral-workers', type=int, default=6)
    parser.add_argument('--max-slabs', type=int, help='Diagnostic subset; never labelled full snapshot')
    args = parser.parse_args()
    if min(args.slab_nx, args.query_chunk, args.channels, args.spectral_workers,
           args.velocity_range_kms) <= 0 or (args.max_slabs is not None and args.max_slabs <= 0):
        parser.error('Numerical sizes must be positive')
    if args.output_dir.exists():
        raise FileExistsError('Choose a new output directory')
    manifest, coverage, excluded_ids, table_path, inputs, hashes = validate_accepted_inputs(
        args.accepted_despotic, args.cloudy_table, args.cloudy_audit, args.dataset)
    import yt
    from yt.units import gravitational_constant
    from quokka2s.pipeline.prep import physics_fields as physics, config as cfg
    from quokka2s.adopted_cell_emission import compute_adopted_cell_emission, COLD_OMITTED_LINES
    from quokka2s.adopted_spectral_products import AdoptedSpectralAccumulator, plot_adopted_spectra

    ds = yt.load(str(args.dataset.resolve()))
    shape = tuple(int(x) for x in ds.domain_dimensions)
    if shape != tuple(manifest['snapshot_shape']) or ds.max_level != 0 or cfg.DOWNSAMPLE_FACTOR != 1:
        raise ValueError('Expected the complete accepted uniform snapshot')
    dsp = TableLookup(load_table(table_path))
    if dsp.table.build_metadata['composition']['setup'] != 'cloudy_c17_02_default_gow_default_v2':
        raise ValueError('Expected native default DESPOTIC abundances')
    domain = _validate_scan_provenance(dsp.table, args.dataset, shape, cfg, physics)
    cloudy = CloudySixLineLookup(args.cloudy_table)
    if cloudy.model_depth_bounds_pc is not None:
        raise ValueError('Expected the audited default legacy Jeans table')
    constants = dict(hydrogen_mass_g=float(physics.m_H.to('g').value),
        boltzmann_erg_K=float(physics.kb.to('erg/K').value),
        gravitational_cm3_g_s2=float(gravitational_constant.to('cm**3/g/s**2').value),
        parsec_cm=float(yt.YTQuantity(1, 'pc').to('cm').value))
    widths = ds.domain_width / ds.domain_dimensions
    volume = float(np.prod(widths.to('cm').value))
    keys = tuple(cloudy.line_keys) + ('co10', 'co21')
    accumulator = AdoptedSpectralAccumulator(keys,
        np.linspace(-args.velocity_range_kms, args.velocity_range_kms, args.channels + 1),
        constants['boltzmann_erg_K'], workers=args.spectral_workers, cell_chunk=16384)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    code_files = (Path(__file__), Path(physics.__file__), ROOT/'src/quokka2s/cloudy_cell_queries.py',
        ROOT/'src/quokka2s/cloudy_hot_emission.py', ROOT/'src/quokka2s/adopted_cell_emission.py',
        ROOT/'src/quokka2s/adopted_spectral_products.py', ROOT/'src/quokka2s/cloudy_sixline_lookup.py',
        ROOT/'src/quokka2s/tables/lookup.py', ROOT/'src/quokka2s/tables/model_depth.py')
    code_hashes = {str(p.resolve()): _sha256(p) for p in code_files}
    began = time.monotonic()
    counts = dict(all=0, cold=0, hot=0, excluded=0, excluded_cold=0, retained=0)
    mass = {key: 0. for key in ('all', 'cold', 'hot', 'excluded')}
    luminosity = np.zeros((len(keys), 2))
    clipped = {key: 0 for key in ('nH', 'NH', 'dVdr')}
    foreground_clipped = dict(below=0, above=0)
    failure_context = {}

    def status(state, **extra):
        data = dict(status=state, counts=counts, elapsed_seconds=time.monotonic()-began, **extra)
        temporary = args.output_dir/'status.tmp'
        temporary.write_text(json.dumps(data, indent=2)+'\n')
        temporary.replace(args.output_dir/'status.json')

    status('running')
    try:
        for slab, (ix, end, lo, hi, core) in enumerate(slab_windows(shape[0], args.slab_nx)):
            if args.max_slabs is not None and slab >= args.max_slabs:
                break
            edge = ds.domain_left_edge.copy(); edge[0] += lo*widths[0]
            grid = ds.covering_grid(0, edge, (hi-lo, *shape[1:]))
            bulk = np.asarray(grid.get_field_parameter('bulk_velocity'))
            if not np.isfinite(bulk).all() or np.any(bulk != 0):
                raise ValueError('Unexpected bulk velocity')
            rho = np.array(grid['gas', 'density'].to('g/cm**3')[core]).ravel()
            tq = np.array(grid['boxlib', 'temperature'][core], dtype=float).ravel()
            column = np.array(physics._column_density_H(None, grid).to('cm**-2')[core]).ravel()
            dvdr = np.array(physics._dVdr_lvg(None, grid).to('s**-1')[core]).ravel()
            vz = np.array(grid['gas', 'velocity_z'].to('km/s')[core]).ravel()
            del grid
            for start in range(0, rho.size, args.query_chunk):
                stop = min(start+args.query_chunk, rho.size); sl = slice(start, stop)
                offset = ix*shape[1]*shape[2]+start
                ids = np.arange(offset, offset+stop-start, dtype=np.int64)
                failure_context = dict(first_cell_id=int(ids[0]), last_cell_id=int(ids[-1]))
                nh = rho[sl]*cfg.X_H/constants['hydrogen_mass_g']
                raw = (nh, column[sl], dvdr[sl])
                axes = (dsp.table.nH_values, dsp.table.col_density_values, dsp.table.dVdr_values)
                for name, values, axis in zip(('nH', 'NH', 'dVdr'), raw, axes):
                    if (not np.isfinite(values).all() or np.any(values <= 0) or
                            np.any(values < axis[0]) or np.any(values > axis[-1])):
                        raise ValueError(f'Fresh DESPOTIC {name} outside accepted snapshot domain')
                td = dsp.temperature(*physics._clip_to_table_domain(dsp, *raw))
                excluded = check_exclusion_queries(ids, excluded_ids, td)
                # Legacy energy/mu arguments are unused by the fixed-mu Jeans estimate.
                queries = prepare_cloudy_cell_queries(rho[sl], column[sl], tq[sl], np.nan, td, np.nan,
                    authorized_excluded=excluded, allow_hot_missing_despotic_exclusions=True, **constants)
                emission = compute_adopted_cell_emission(queries, dvdr[sl], dsp, cloudy,
                    allow_capped_legacy_jeans=True)
                cold = tq[sl] < 3000
                for key, flag in emission.despotic_clipped.items():
                    clipped[key] += int(flag.sum())
                hot = ~cold & emission.valid
                foreground_clipped['below'] += int(np.count_nonzero(hot & (column[sl] < 10**cloudy.log_NH_attenuation[0])))
                foreground_clipped['above'] += int(np.count_nonzero(hot & (column[sl] > 10**cloudy.log_NH_attenuation[-1])))
                for branch, selected in enumerate((cold & emission.valid, hot)):
                    luminosity[:, branch] += emission.emissivity_erg_s_cm3[:, selected].sum(axis=1)*volume
                use = emission.valid
                accumulator.add(vz[sl][use], emission.thermal_temperature_K[:, use],
                    emission.emissivity_erg_s_cm3[:, use], volume, cold_mask=cold[use])
                for name, selected in (('all', np.ones(ids.shape, dtype=bool)), ('cold', cold),
                                       ('hot', ~cold), ('excluded', excluded)):
                    counts[name] += int(selected.sum())
                    mass[name] += float(rho[sl][selected].sum()*volume)
                counts['retained'] += int(use.sum())
                counts['excluded_cold'] += int(np.count_nonzero(excluded & cold))
            status('running', completed_slabs=slab+1)
            print(f'Emission: {counts["all"]}/{manifest["total_cells"]} cells; '
                  f'{time.monotonic()-began:.1f} s', flush=True)
        full = counts['all'] == manifest['total_cells']
        if full:
            expected = dict(cold=coverage['groups']['T_QUOKKA_lt_3000_K']['cell_count'],
                excluded=manifest['excluded_cell_count'], excluded_cold=manifest['excluded_cold_cell_count'],
                retained=manifest['retained_cells'])
            if any(counts[key] != value for key, value in expected.items()):
                raise ValueError('Final counts differ from accepted coverage')
            for group, old in (('all', 'all_cells'), ('cold', 'T_QUOKKA_lt_3000_K')):
                if not np.isclose(mass[group], coverage['groups'][old]['total_valid_mass_g'], rtol=1e-12, atol=0):
                    raise ValueError(f'{group} mass differs from accepted snapshot')
            if not np.isclose(mass['excluded']/mass['all'], manifest['excluded_mass_fraction'], rtol=1e-12, atol=0):
                raise ValueError('Excluded mass differs from acceptance')
        for mapping in (hashes, code_hashes):
            for path, digest in mapping.items():
                if _sha256(path) != digest:
                    raise ValueError(f'Input changed during run: {path}')
        if any(luminosity[keys.index(key), 0] != 0 for key in COLD_OMITTED_LINES):
            raise ValueError('Cold CIII/CIV must be exactly zero')
        payload, spectral_report = accumulator.finalize()
        if not np.allclose(payload['input_luminosity_erg_s'], luminosity, rtol=1e-12, atol=0):
            raise ValueError('Spectrum input differs from independent cell luminosities')
        if np.any(payload['captured_luminosity_erg_s'] > luminosity*(1+1e-12)):
            raise ValueError('Spectrum exceeds supplied luminosity')
        area = float((ds.domain_width[0]*ds.domain_width[1]).to('cm**2').value)
        payload.update(projected_area_cm2=np.asarray(area), line_of_sight=np.asarray('z'),
            full_snapshot=np.asarray(full), source_manifest_sha256=np.asarray(hashes[str(args.accepted_despotic.resolve())]))
        np.savez_compressed(args.output_dir/'spectra.npz', **payload)
        plot_adopted_spectra(payload, args.output_dir/'spectra', projected_area_cm2=area,
            title=('Default-abundance LOS-z spectra' if full else 'Partial diagnostic LOS-z spectra'))
        result = dict(status='completed' if full else 'partial diagnostic', full_snapshot=full,
            completed_at=datetime.now(timezone.utc).isoformat(), counts=counts, mass_g=mass,
            line_keys=list(keys), regimes=['cold', 'hot'], luminosity_erg_s=luminosity.tolist(),
            despotic_coordinate_clipped_cells=clipped, cloudy_attenuation_coordinate_clipped_cells=foreground_clipped,
            source_sha256=hashes, code_sha256=code_hashes, snapshot_domain=domain, constants=constants,
            cell_volume_cm3=volume, spectral_report=spectral_report,
            cold_CIII_CIV='Omitted by the adopted prescription; hot contributions only',
            exclusions='Exact shared cell mask from accepted DESPOTIC manifest; excluded entries are not physical zero emission',
            interpretation='Local volume emission with inherited Cloudy escape treatment and DESPOTIC LVG; no extra foreground dust or intercell transfer.',
            cloudy_legacy_cap_rounding_fraction=0.00010448889400005434,
            elapsed_seconds=time.monotonic()-began)
        (args.output_dir/'emission_report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        status(result['status'])
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}', failure_context=failure_context)
        raise


if __name__ == '__main__':
    main()
