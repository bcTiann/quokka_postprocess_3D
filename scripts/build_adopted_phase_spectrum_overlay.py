#!/usr/bin/env python3
"""Refresh gas-phase velocity comparisons with saved accepted LOS-z spectra.

Only gas mass distributions are recomputed. The ten accepted emission spectra
are reused unchanged; phase temperature must be explicitly selected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from quokka2s.emission_processing import validate_accepted_inputs, check_exclusion_queries
from scripts.check_despotic_snapshot_coverage import slab_windows, _validate_scan_provenance, _sha256
from quokka2s.adopted_phase_overlay import accepted_display_profiles, plot_phase_spectrum_overlay, PHASE_ORDER, LINE_ORDER
from quokka2s.adopted_velocity_phases import AdoptedVelocityPhaseAccumulator
from quokka2s.tables.io import load_table
from quokka2s.tables.lookup import TableLookup


def load_spectra(path):
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.array(data[key]) for key in data.files}
    if (not bool(payload['full_snapshot']) or str(payload['line_of_sight']) != 'z'
            or not np.array_equal(payload['velocity_edges_kms'], np.linspace(-200., 200., 301))):
        raise ValueError('Expected the accepted full-snapshot LOS-z, +/-200 km/s, 300-channel spectra')
    accepted_display_profiles(payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {
        'dataset': ROOT/'inputs/snapshots/plt0655228',
        'accepted-despotic': ROOT/'output/despotic_default_parallel_20260918/interpolated/accepted_table.json',
        'cloudy-table': ROOT/'data/cloudy_hm2012_attgrid_ism_nh21_cmb_cr_defaultabund_eightline_jeans_7x10x21.npz',
        'cloudy-audit': ROOT/'output/default_table_reuse_20260918/cloudy_reuse_coverage.json',
        'spectra': ROOT/'output/cloudy_rgi_comparison_20260920/full/spectra.npz',
        'emission-report': ROOT/'output/cloudy_rgi_comparison_20260920/full/emission_report.json',
    }
    for name, path in defaults.items():
        parser.add_argument('--'+name, type=Path, default=path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--figure-stem', type=Path)
    parser.add_argument('--line-keys', nargs='+', choices=LINE_ORDER, default=LINE_ORDER,
                        help='Transitions to render; numerical inputs remain unchanged')
    parser.add_argument('--figure-style', choices=('latex', 'full'), default='latex',
                        help='LaTeX omits titles and explanatory text; full includes them')
    parser.add_argument('--phase-temperature', choices=('mixed', 'quokka'), required=True)
    parser.add_argument('--slab-nx', type=int, default=8)
    parser.add_argument('--query-chunk', type=int, default=100000)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    if min(args.slab_nx, args.query_chunk) <= 0:
        parser.error('Chunk sizes must be positive')
    report_path = args.output_dir/'phase_velocity_report.json'
    bundle_path = args.output_dir/'phase_velocity.npz'
    figure = args.figure_stem or args.output_dir/'phase_spectrum_overlay'
    spectra = load_spectra(args.spectra)
    if args.plot_only:
        report = json.loads(report_path.read_text())
        if report['status'] != 'completed' or report['phase_temperature'] != args.phase_temperature:
            raise ValueError('Plot-only requires completed bins with the same temperature policy')
        if (_sha256(args.spectra) != report['source_sha256'][str(args.spectra.resolve())]
                or _sha256(bundle_path) != report['bundle_sha256']):
            raise ValueError('Plot sources differ from completed computation')
        with np.load(bundle_path, allow_pickle=False) as data:
            phase_payload = {key: np.array(data[key]) for key in data.files}
        plot_phase_spectrum_overlay(spectra, phase_payload, report, figure,
                                   line_keys=args.line_keys, figure_style=args.figure_style)
        return
    if args.output_dir.exists():
        raise FileExistsError('Use a new output directory or --plot-only')
    if not args.no_plot and any(
            figure.with_name(f'{figure.name}_{key}').with_suffix(ext).exists()
            for key in args.line_keys for ext in ('.png', '.pdf')):
        raise FileExistsError('Choose a new figure stem')

    manifest, coverage, excluded_ids, table_path, _, hashes = validate_accepted_inputs(
        args.accepted_despotic, args.cloudy_table, args.cloudy_audit, args.dataset)
    reference = json.loads(args.emission_report.read_text())
    if (reference.get('status') != 'completed' or not reference.get('full_snapshot')
            or reference['source_sha256'] != hashes
            or str(spectra['source_manifest_sha256']) != hashes[str(args.accepted_despotic.resolve())]
            or tuple(spectra['line_keys']) != tuple(reference['line_keys'])):
        raise ValueError('Accepted spectra and phase inputs must have identical provenance')
    np.testing.assert_array_equal(spectra['cell_counts_by_regime'],
        [reference['counts']['cold']-reference['counts']['excluded_cold'],
         reference['counts']['retained']-reference['counts']['cold']+reference['counts']['excluded_cold']])
    np.testing.assert_allclose(
        np.sum(spectra['dL_dv_erg_s_per_kms']*np.diff(spectra['velocity_edges_kms']), axis=-1),
        spectra['captured_luminosity_erg_s'], rtol=1e-12, atol=0)
    for index, key in enumerate(spectra['line_keys']):
        expected = reference['spectral_report']['lines'][str(key)]['total']
        np.testing.assert_allclose(spectra['total_input_luminosity_erg_s'][index],
                                   expected['input_luminosity_erg_s'], rtol=1e-12)
        np.testing.assert_allclose(spectra['total_captured_luminosity_erg_s'][index],
                                   expected['captured_luminosity_erg_s'], rtol=1e-12)
    for path in (args.spectra, args.emission_report):
        hashes[str(path.resolve())] = _sha256(path)

    import yt
    from quokka2s.pipeline.prep import physics_fields as physics, config as cfg
    ds = yt.load(str(args.dataset.resolve()))
    shape = tuple(int(x) for x in ds.domain_dimensions)
    if shape != tuple(manifest['snapshot_shape']) or ds.max_level != 0 or cfg.DOWNSAMPLE_FACTOR != 1:
        raise ValueError('Expected the accepted full-resolution uniform snapshot')
    dsp = TableLookup(load_table(table_path))
    if dsp.table.build_metadata['composition']['setup'] != 'cloudy_c17_02_default_gow_default_v2':
        raise ValueError('Expected accepted default DESPOTIC abundances')
    domain = _validate_scan_provenance(dsp.table, args.dataset, shape, cfg, physics)
    widths = ds.domain_width/ds.domain_dimensions
    volume = float(np.prod(widths.to('cm').value))
    hydrogen_mass = float(physics.m_H.to('g').value)
    accumulator = AdoptedVelocityPhaseAccumulator(spectra['velocity_edges_kms'])
    code_paths = (Path(__file__), Path(physics.__file__), ROOT/'src/quokka2s/pipeline/utils.py',
        ROOT/'src/quokka2s/adopted_velocity_phases.py', ROOT/'src/quokka2s/adopted_phase_overlay.py',
        ROOT/'src/quokka2s/emission_processing.py', ROOT/'scripts/check_despotic_snapshot_coverage.py',
        ROOT/'src/quokka2s/tables/lookup.py')
    code_hashes = {str(p.resolve()): _sha256(p) for p in code_paths}
    counts = dict(all=0, cold=0, hot=0, excluded=0, excluded_cold=0, retained=0)
    masses = {key: 0. for key in ('all', 'cold', 'hot', 'excluded', 'retained')}
    temperature_range = [np.inf, -np.inf]
    began = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def status(state, **extra):
        temporary = args.output_dir/'status.tmp'
        temporary.write_text(json.dumps(dict(status=state, counts=counts,
            elapsed_seconds=time.monotonic()-began, **extra), indent=2)+'\n')
        temporary.replace(args.output_dir/'status.json')

    status('running')
    try:
        for slab, (ix, end, lo, hi, core) in enumerate(slab_windows(shape[0], args.slab_nx)):
            edge = ds.domain_left_edge.copy()
            edge[0] += lo*widths[0]
            grid = ds.covering_grid(0, edge, (hi-lo, *shape[1:]))
            bulk = np.asarray(grid.get_field_parameter('bulk_velocity'))
            if not np.isfinite(bulk).all() or np.any(bulk != 0):
                raise ValueError('Unexpected bulk-velocity subtraction')
            rho = np.array(grid['gas', 'density'].to('g/cm**3')[core]).ravel()
            tq = np.array(grid['boxlib', 'temperature'][core], dtype=float).ravel()
            column = np.array(physics._column_density_H(None, grid).to('cm**-2')[core]).ravel()
            dvdr = np.array(physics._dVdr_lvg(None, grid).to('s**-1')[core]).ravel()
            vz = np.array(grid['gas', 'velocity_z'].to('km/s')[core]).ravel()
            del grid
            for start in range(0, rho.size, args.query_chunk):
                stop = min(start+args.query_chunk, rho.size)
                sl = slice(start, stop)
                offset = ix*shape[1]*shape[2]+start
                ids = np.arange(offset, offset+stop-start, dtype=np.int64)
                raw = (rho[sl]*cfg.X_H/hydrogen_mass, column[sl], dvdr[sl])
                for name, values, axis in zip(('nH','NH','dVdr'), raw,
                    (dsp.table.nH_values, dsp.table.col_density_values, dsp.table.dVdr_values)):
                    if (not np.isfinite(values).all() or np.any(values <= 0)
                            or np.any(values < axis[0]) or np.any(values > axis[-1])):
                        raise ValueError(f'DESPOTIC {name} query outside accepted domain')
                td = dsp.temperature(*raw)
                excluded = check_exclusion_queries(ids, excluded_ids, td)
                valid = ~excluded
                cold = tq[sl] < 3000.
                temperature = np.where(cold, td, tq[sl]) if args.phase_temperature == 'mixed' else tq[sl]
                accumulator.add(vz[sl][valid], temperature[valid], rho[sl][valid], volume)
                if valid.any():
                    temperature_range[0] = min(temperature_range[0], float(temperature[valid].min()))
                    temperature_range[1] = max(temperature_range[1], float(temperature[valid].max()))
                for key, take in (('all', np.ones(ids.size, dtype=bool)), ('cold', cold),
                        ('hot', ~cold), ('excluded', excluded), ('retained', valid)):
                    counts[key] += int(take.sum())
                    masses[key] += float(rho[sl][take].sum()*volume)
                counts['excluded_cold'] += int(np.count_nonzero(cold & excluded))
            status('running', completed_slabs=slab+1)
            print(f'Gas phase velocities: {counts["all"]}/{manifest["total_cells"]} cells; '
                  f'{time.monotonic()-began:.1f} s', flush=True)
            del rho, tq, column, dvdr, vz
            gc.collect()

        if counts != reference['counts']:
            raise ValueError('Phase selection differs from accepted spectra')
        for key, value in reference['mass_g'].items():
            np.testing.assert_allclose(masses[key], value, rtol=1e-12, atol=0)
        statistics = accumulator.report()
        groups = statistics['groups']
        if groups['total']['count'] != counts['retained']:
            raise ValueError('Phase accumulator missed retained cells')
        np.testing.assert_allclose(groups['total']['mass_g'], masses['retained'], rtol=1e-12, atol=0)
        for group in groups.values():
            pieces = [group[key] for key in ('in_window', 'below_window', 'above_window')]
            if sum(part['count'] for part in pieces) != group['count']:
                raise ValueError('Velocity window counts do not conserve cells')
            np.testing.assert_allclose(sum(part['mass_g'] for part in pieces),
                                       group['mass_g'], rtol=1e-12, atol=0)
            np.testing.assert_allclose(group['histogram_mass_g'],
                                       group['in_window']['mass_g'], rtol=1e-12, atol=0)
        phase_payload = dict(velocity_edges_kms=accumulator.velocity_edges_kms,
            phase_keys=np.asarray((*PHASE_ORDER, 'total')),
            histogram_mass_g=np.stack([accumulator.histogram_mass_g[key] for key in (*PHASE_ORDER, 'total')]))
        np.testing.assert_allclose(phase_payload['histogram_mass_g'][:5].sum(axis=0),
                                   phase_payload['histogram_mass_g'][-1], rtol=1e-12, atol=0)
        for mapping in (hashes, code_hashes):
            for path, digest in mapping.items():
                if _sha256(path) != digest:
                    raise ValueError(f'Source changed during scan: {path}')
        np.savez_compressed(bundle_path, **phase_payload)
        _, line_report = accepted_display_profiles(spectra)
        report = dict(status='completed', full_snapshot=True,
            completed_at=datetime.now(timezone.utc).isoformat(), dataset=str(args.dataset.resolve()),
            spectra=str(args.spectra.resolve()), phase_temperature=args.phase_temperature,
            phase_temperature_range_K=temperature_range, counts=counts, mass_g=masses,
            cell_volume_cm3=volume, X_H=float(cfg.X_H), line_of_sight='z',
            velocity_zero='Raw simulation vz; no recentering', velocity_range_kms=[-200.,200.],
            velocity_channels=300, velocity_statistics=statistics, displayed_lines=line_report,
            phase_sigma='All retained velocities about the common retained-gas mass-weighted mean',
            line_sigma='Channel moments within saved velocity window about each line luminosity centroid',
            normalization='Each displayed curve divided by its own peak inside the plotted velocity range',
            gas_broadening='None; raw cell velocities', line_broadening='Unchanged accepted spectra',
            bundle_sha256=_sha256(bundle_path), source_sha256=hashes, code_sha256=code_hashes,
            snapshot_domain=domain, elapsed_seconds=time.monotonic()-began)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        if not args.no_plot:
            plot_phase_spectrum_overlay(spectra, phase_payload, report, figure,
                                       line_keys=args.line_keys, figure_style=args.figure_style)
        status('completed')
        print(f'Completed gas phase comparison: {report_path}', flush=True)
    except Exception as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
