#!/usr/bin/env python3
"""Build phase histograms and render ten selected manuscript panels.

The table provenance, exact exclusions, fresh snapshot queries and emissivities
are shared with quokka2s.emission_processing. Existing axes, 0.2-dex bins,
absolute mass/luminosity weights are preserved. All ten line arrays are saved;
the figure displays one transition per species alongside four gas panels.
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
import matplotlib
matplotlib.use('Agg')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from quokka2s.emission_processing import validate_accepted_inputs, check_exclusion_queries
from scripts.check_despotic_snapshot_coverage import slab_windows, _validate_scan_provenance, _sha256
from quokka2s.adopted_cell_emission import compute_adopted_cell_emission
from quokka2s.cloudy_cell_queries import prepare_cloudy_cell_queries
from quokka2s.cloudy_sixline_lookup import CloudySixLineLookup
from quokka2s.pipeline.tasks.adopted_phase_hist import (
    PANELS, DexHistogram, add_adopted_phase_chunk, plot_panels,
)
from quokka2s.tables.io import load_table
from quokka2s.tables.lookup import TableLookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {
        'dataset': ROOT/'inputs/snapshots/plt0655228',
        'accepted-despotic': ROOT/'output/despotic_default_parallel_20260918/interpolated/accepted_table.json',
        'cloudy-table': ROOT/'data/cloudy_hm2012_attgrid_ism_nh21_cmb_cr_defaultabund_eightline_jeans_7x10x21.npz',
        'cloudy-audit': ROOT/'output/default_table_reuse_20260918/cloudy_reuse_coverage.json',
        'reference-emission-report': ROOT/'output/cloudy_rgi_comparison_20260920/full/emission_report.json',
    }
    for name, path in defaults.items():
        parser.add_argument('--'+name, type=Path, default=path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--figure-stem', type=Path)
    parser.add_argument('--mass-selection', choices=('retained', 'raw-all'), required=True,
                        help='retained: all panels share the emission mask; raw-all: only raw QK/NH mass panels use all cells')
    parser.add_argument('--slab-nx', type=int, default=8)
    parser.add_argument('--query-chunk', type=int, default=100000)
    parser.add_argument('--max-slabs', type=int, help='Diagnostic subset; never labelled full snapshot')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if min(args.slab_nx, args.query_chunk) <= 0 or (args.max_slabs is not None and args.max_slabs <= 0):
        parser.error('Chunk sizes must be positive')
    bundle = args.output_dir/'phase_histograms.npz'
    report_path = args.output_dir/'phase_histograms.json'
    figure = args.figure_stem or args.output_dir/'phase_histograms_10panel'
    if args.plot_only:
        report = json.loads(report_path.read_text())
        if report.get('status') != 'completed' or report['mass_selection'] != args.mass_selection:
            raise ValueError('Plot-only requires a completed run with the same mass selection')
        with np.load(bundle, allow_pickle=False) as data:
            panels = {key: {name: data[f'{key}__{name}'] for name in ('H', 'x_edges', 'y_edges')}
                      for key, _, _ in PANELS}
        figure.parent.mkdir(parents=True, exist_ok=True)
        plot_panels(panels, figure.with_suffix('.png'), figure.with_suffix('.pdf'))
        return
    if args.output_dir.exists():
        raise FileExistsError('Choose a new output directory; --plot-only reuses completed bins')
    if not args.no_plot and any(figure.with_suffix(ext).exists() for ext in ('.pdf', '.png')):
        raise FileExistsError('Choose a new figure stem')
    manifest, coverage, excluded_ids, table_path, inputs, hashes = validate_accepted_inputs(
        args.accepted_despotic, args.cloudy_table, args.cloudy_audit, args.dataset)
    reference = json.loads(args.reference_emission_report.read_text())
    if (reference.get('status') != 'completed' or not reference.get('full_snapshot') or
            reference['source_sha256'] != hashes):
        raise ValueError('Reference spectra must use the same accepted inputs')
    hashes[str(args.reference_emission_report.resolve())] = _sha256(args.reference_emission_report)

    import yt
    from yt.units import gravitational_constant
    from quokka2s.pipeline.prep import physics_fields as physics, config as cfg
    ds = yt.load(str(args.dataset.resolve()))
    shape = tuple(int(x) for x in ds.domain_dimensions)
    if shape != tuple(manifest['snapshot_shape']) or ds.max_level != 0 or cfg.DOWNSAMPLE_FACTOR != 1:
        raise ValueError('Expected the complete accepted uniform snapshot')
    dsp = TableLookup(load_table(table_path))
    if dsp.table.build_metadata['composition']['setup'] != 'cloudy_c17_02_default_gow_default_v2':
        raise ValueError('Expected default DESPOTIC abundances')
    domain = _validate_scan_provenance(dsp.table, args.dataset, shape, cfg, physics)
    cloudy = CloudySixLineLookup(args.cloudy_table)
    if cloudy.model_depth_bounds_pc is not None:
        raise ValueError('Expected the audited default legacy Jeans table')
    constants = dict(hydrogen_mass_g=float(physics.m_H.to('g').value),
        boltzmann_erg_K=float(physics.kb.to('erg/K').value),
        gravitational_cm3_g_s2=float(gravitational_constant.to('cm**3/g/s**2').value),
        parsec_cm=float(yt.YTQuantity(1, 'pc').to('cm').value))
    widths = ds.domain_width/ds.domain_dimensions
    volume = float(np.prod(widths.to('cm').value))
    code_files = (Path(__file__), Path(physics.__file__),
        ROOT/'src/quokka2s/emission_processing.py', ROOT/'scripts/check_despotic_snapshot_coverage.py',
        ROOT/'src/quokka2s/pipeline/tasks/adopted_phase_hist.py',
        ROOT/'src/quokka2s/pipeline/tasks/phase_combined_plot.py',
        ROOT/'src/quokka2s/cloudy_cell_queries.py', ROOT/'src/quokka2s/cloudy_hot_emission.py',
        ROOT/'src/quokka2s/adopted_cell_emission.py', ROOT/'src/quokka2s/cloudy_sixline_lookup.py',
        ROOT/'src/quokka2s/tables/lookup.py', ROOT/'src/quokka2s/tables/model_depth.py')
    code_hashes = {str(p.resolve()): _sha256(p) for p in code_files}
    histograms = {key: DexHistogram(.2) for key, _, _ in PANELS}
    counts = dict(all=0, cold=0, hot=0, excluded=0, excluded_cold=0, retained=0)
    mass = {key: 0. for key in ('all', 'cold', 'hot', 'excluded', 'retained')}
    clipped = {key: 0 for key in ('nH', 'NH', 'dVdr')}
    ranges = {}
    began = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=False)

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
            edge = ds.domain_left_edge.copy()
            edge[0] += lo*widths[0]
            grid = ds.covering_grid(0, edge, (hi-lo, *shape[1:]))
            bulk = np.asarray(grid.get_field_parameter('bulk_velocity'))
            if not np.isfinite(bulk).all() or np.any(bulk != 0):
                raise ValueError('Unexpected bulk velocity')
            rho = np.array(grid['gas', 'density'].to('g/cm**3')[core]).ravel()
            tq = np.array(grid['boxlib', 'temperature'][core], dtype=float).ravel()
            column = np.array(physics._column_density_H(None, grid).to('cm**-2')[core]).ravel()
            dvdr = np.array(physics._dVdr_lvg(None, grid).to('s**-1')[core]).ravel()
            del grid
            for start in range(0, rho.size, args.query_chunk):
                stop = min(start+args.query_chunk, rho.size)
                sl = slice(start, stop)
                offset = ix*shape[1]*shape[2]+start
                ids = np.arange(offset, offset+stop-start, dtype=np.int64)
                nh = rho[sl]*cfg.X_H/constants['hydrogen_mass_g']
                raw = (nh, column[sl], dvdr[sl])
                axes = (dsp.table.nH_values, dsp.table.col_density_values, dsp.table.dVdr_values)
                for name, values, axis in zip(('nH', 'NH', 'dVdr'), raw, axes):
                    if (not np.isfinite(values).all() or np.any(values <= 0) or
                            np.any(values < axis[0]) or np.any(values > axis[-1])):
                        raise ValueError(f'Fresh DESPOTIC {name} outside accepted snapshot domain')
                td = dsp.temperature(*physics._clip_to_table_domain(dsp, *raw))
                excluded = check_exclusion_queries(ids, excluded_ids, td)
                queries = prepare_cloudy_cell_queries(rho[sl], column[sl], tq[sl], np.nan, td, np.nan,
                    authorized_excluded=excluded, allow_hot_missing_despotic_exclusions=True, **constants)
                emission = compute_adopted_cell_emission(queries, dvdr[sl], dsp, cloudy,
                    allow_capped_legacy_jeans=True)
                add_adopted_phase_chunk(histograms, rho[sl], tq[sl], td, column[sl], emission, volume,
                                        raw_mass_all_cells=args.mass_selection == 'raw-all')
                cold = tq[sl] < 3000
                for name, selected in (('all', np.ones(ids.shape, dtype=bool)), ('cold', cold),
                        ('hot', ~cold), ('excluded', excluded), ('retained', emission.valid)):
                    counts[name] += int(selected.sum())
                    mass[name] += float(rho[sl][selected].sum()*volume)
                counts['excluded_cold'] += int(np.count_nonzero(excluded & cold))
                for key, flag in emission.despotic_clipped.items():
                    clipped[key] += int(flag.sum())
                for key, values in (('rho', rho[sl]), ('nH', nh), ('NH', column[sl]),
                        ('T_QUOKKA', tq[sl]), ('T_DESPOTIC', td[emission.valid]), ('dVdr', dvdr[sl])):
                    if values.size:
                        old = ranges.get(key, (np.inf, -np.inf))
                        ranges[key] = (min(old[0], float(values.min())), max(old[1], float(values.max())))
            status('running', completed_slabs=slab+1)
            print(f'Phase histograms: {counts["all"]}/{manifest["total_cells"]} cells; '
                  f'{time.monotonic()-began:.1f} s', flush=True)
            del rho, tq, column, dvdr
            gc.collect()
        full = counts['all'] == manifest['total_cells']
        if full:
            if counts != reference['counts']:
                raise ValueError('Cell membership differs from accepted spectra')
            for key, value in reference['mass_g'].items():
                np.testing.assert_allclose(mass[key], value, rtol=1e-12, atol=0)
        if any(clipped.values()):
            raise ValueError('Unexpected DESPOTIC query-coordinate clipping')
        panels = {key: histogram.result() for key, histogram in histograms.items()}
        validation = {}
        for key, histogram in histograms.items():
            is_mass = key.startswith('mass') or key == 'NH_rho'
            use_all = args.mass_selection == 'raw-all' and key in ('mass_T_QK', 'NH_rho')
            expected_count = counts['all' if use_all else 'retained']
            if histogram.count != expected_count:
                raise ValueError(f'Wrong cell membership for {key}')
            binned = float(histogram.H.sum())
            np.testing.assert_allclose(binned, histogram.total, rtol=1e-12, atol=0)
            check = dict(direct_sum=histogram.total, bin_sum=binned, cells=histogram.count,
                         unit='g' if is_mass else 'erg/s',
                         bin_sum_relative_error=abs(binned/histogram.total-1) if histogram.total else 0.)
            if is_mass:
                np.testing.assert_allclose(binned, mass['all' if use_all else 'retained'], rtol=1e-12, atol=0)
            elif full:
                expected = reference['spectral_report']['lines'][key]['total']['input_luminosity_erg_s']
                np.testing.assert_allclose(binned, expected, rtol=1e-12, atol=0)
                check.update(reference_luminosity_erg_s=expected,
                             reference_relative_error=abs(binned/expected-1) if expected else 0.)
            validation[key] = check
        for mapping in (hashes, code_hashes):
            for path, digest in mapping.items():
                if _sha256(path) != digest:
                    raise ValueError(f'Input changed during run: {path}')
        np.savez_compressed(bundle, **{f'{key}__{name}': value
                            for key, panel in panels.items() for name, value in panel.items()})
        report = dict(status='completed' if full else 'partial diagnostic', full_snapshot=full,
            completed_at=datetime.now(timezone.utc).isoformat(), dataset=str(args.dataset.resolve()),
            despotic_table=str(table_path), cloudy_table=str(args.cloudy_table.resolve()),
            counts=counts, mass_g=mass, mass_selection=args.mass_selection,
            temperature_split_K=3000, X_H=float(cfg.X_H), cell_volume_cm3=volume,
            temperature_axes={key: temp for key, temp, _ in PANELS},
            line_policy='Canonical compute_adopted_cell_emission; all ten individual lines; CIII/CIV cold luminosity is zero',
            column_definition='Fresh inclusive +/-z columns, harmonic mean; no cached column field',
            dvdr='Fresh abs(div(v))/3, x-slab halo, current numerical floor',
            bin_dex=.2, color_dynamic_range_dex=6, velocity_selection=None,
            despotic_coordinate_clipped_cells=clipped, input_ranges=ranges,
            validation=validation, source_sha256=hashes, code_sha256=code_hashes,
            snapshot_domain=domain, reference_emission_report=str(args.reference_emission_report.resolve()),
            elapsed_seconds=time.monotonic()-began)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        if not args.no_plot:
            if not full:
                raise ValueError('Partial runs must use --no-plot')
            figure.parent.mkdir(parents=True, exist_ok=True)
            plot_panels(panels, figure.with_suffix('.png'), figure.with_suffix('.pdf'))
        status(report['status'])
        print(f'Completed phase histogram bundle: {bundle}', flush=True)
    except Exception as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
