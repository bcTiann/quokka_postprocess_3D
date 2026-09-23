#!/usr/bin/env python3
"""Refresh the five-field Figure 1 at x index 216 with accepted DESPOTIC inputs.

Reuse Plot_MultiFieldSlices and its TABLE_INPUT_PANELS preset. Calculate a
full-z, three-x-cell slab to retain shielding rays and central velocity
derivatives; query the accepted table explicitly instead of legacy defaults.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import matplotlib
matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_default_emission_products import validate_accepted_inputs, check_exclusion_queries
from scripts.check_despotic_snapshot_coverage import _validate_scan_provenance, _sha256
from quokka2s.tables.io import load_table
from quokka2s.tables.lookup import TableLookup
from quokka2s.pipeline.tasks.multi_field_slices import Plot_MultiFieldSlices, TABLE_INPUT_PANELS
from quokka2s.pipeline.base import PipelineConfig


def render(payload, report, output):
    fields = [tuple(p) for p in TABLE_INPUT_PANELS]
    p = list(fields[2])
    p[2] = r'$\log_{10}\,\frac{dV}{dr}$ [s$^{-1}$]'
    fields[2] = tuple(p)
    # Preserve the original five physical fields; apply the established common
    # emission mask only for display, after calculating the unmasked rays.
    slices = {key: np.where(payload['valid'], payload[key], np.nan) for key, *_ in fields}
    config = PipelineConfig(dataset_path=report['dataset'], output_dir=output,
                            downsample_factor=1, column_extension_lateral_kpc=0.)
    context = SimpleNamespace(config=config)
    results = dict(slices_by_idx={report['slice_index']: slices},
                   slice_indices=[report['slice_index']], extent_kpc=payload['extent_kpc'])
    for title, filename in ((True, 'multi_field_slices_idx0216_full.png'),
                             (False, 'multi_field_slices_idx0216.png')):
        plotter = Plot_MultiFieldSlices(config, slice_axis='x', slice_idx=report['slice_index'],
            figure_units='kpc', aspect='equal', panels=fields, show_title=title,
            save_pdf=not title, filename=filename, display_factors={'dVdr_slice': 1.})
        plotter.plot(context, results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'output/pdf/figure1_table_inputs_idx0216')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    output = args.output_dir
    bundle = output/'slice_data.npz'
    report_path = output/'slice_report.json'
    if args.plot_only:
        report = json.loads(report_path.read_text())
        if report['status'] != 'completed' or _sha256(bundle) != report['bundle_sha256']:
            raise ValueError('Expected an intact completed slice bundle')
        with np.load(bundle, allow_pickle=False) as data:
            payload = {key: np.array(data[key]) for key in data.files}
        render(payload, report, output)
        return
    if output.exists():
        raise FileExistsError('Use a new output directory or --plot-only')
    dataset = ROOT/'plt0655228'
    manifest_path = ROOT/'output/despotic_default_parallel_20260918/interpolated/accepted_table.json'
    cloudy = ROOT/'data/cloudy_hm2012_attgrid_ism_nh21_cmb_cr_defaultabund_eightline_jeans_7x10x21.npz'
    audit = ROOT/'output/default_table_reuse_20260918/cloudy_reuse_coverage.json'
    manifest, _, excluded_ids, table_path, _, hashes = validate_accepted_inputs(
        manifest_path, cloudy, audit, dataset)
    import yt
    from quokka2s.pipeline.prep import physics_fields as physics, config as cfg
    ds = yt.load(str(dataset))
    shape = tuple(int(v) for v in ds.domain_dimensions)
    if shape != tuple(manifest['snapshot_shape']) or ds.max_level != 0:
        raise ValueError('Expected accepted uniform full-resolution snapshot')
    if cfg.COLUMN_DENSITY_DIRECTIONS != 'z' or cfg.COLUMN_DENSITY_MEAN != 'harmonic':
        raise ValueError('This figure requires +/-z harmonic-mean shielding')
    dsp = TableLookup(load_table(table_path))
    domain = _validate_scan_provenance(dsp.table, dataset, shape, cfg, physics)
    index = 216
    widths = ds.domain_width/ds.domain_dimensions
    edge = ds.domain_left_edge.copy()
    edge[0] += (index-1)*widths[0]
    grid = ds.covering_grid(0, edge, (3, shape[1], shape[2]))
    nH = np.array(physics._number_density_H(None, grid).to('cm**-3')[1])
    NH = np.array(physics._column_density_H(None, grid).to('cm**-2')[1])
    dvdr = np.array(physics._dVdr_lvg(None, grid).to('s**-1')[1])
    tq = np.array(grid['boxlib', 'temperature'][1], dtype=float)
    raw = (nH, NH, dvdr)
    for name, arr, axis in zip(('nH', 'NH', 'dVdr'), raw,
            (dsp.table.nH_values, dsp.table.col_density_values, dsp.table.dVdr_values)):
        if not np.isfinite(arr).all() or np.any(arr < axis[0]) or np.any(arr > axis[-1]):
            raise ValueError(f'{name} query outside accepted table range')
    td = dsp.temperature(*raw)
    ids = np.arange(index*shape[1]*shape[2], (index+1)*shape[1]*shape[2]).reshape(nH.shape)
    excluded = check_exclusion_queries(ids, excluded_ids, td)
    # Independent 2D prefix/suffix integral: the slice retains complete z rays.
    dz = float(widths[2].to('cm').value)
    nminus = np.cumsum(nH*dz, axis=1)
    nplus = np.cumsum((nH*dz)[:, ::-1], axis=1)[:, ::-1]
    direct = 2./(1./nplus+1./nminus)
    np.testing.assert_allclose(NH, direct, rtol=1e-13, atol=0.)
    # Central x derivative must use its two neighbour planes, not a thin-slice
    # one-sided derivative. Check interior y/z derivatives independently too.
    vx, vy, vz = [np.asarray(grid['gas', 'velocity_'+axis].to('cm/s')) for axis in 'xyz']
    dx, dy, dz = widths.to('cm').value
    div = ((vx[2,1:-1,1:-1]-vx[0,1:-1,1:-1])/(2*dx)
           +(vy[1,2:,1:-1]-vy[1,:-2,1:-1])/(2*dy)
           +(vz[1,1:-1,2:]-vz[1,1:-1,:-2])/(2*dz))
    np.testing.assert_allclose(dvdr[1:-1,1:-1], np.maximum(abs(div)/3,physics.DVDR_FLOOR),
                               rtol=1e-13, atol=0.)
    left, right = ds.domain_left_edge.to('kpc').value, ds.domain_right_edge.to('kpc').value
    payload = dict(nH_slice=nH, NH_slice=NH, dVdr_slice=dvdr, T_qk_slice=tq,
                   T_dsp_slice=td, valid=~excluded, NH_plus_z=nplus, NH_minus_z=nminus,
                   extent_kpc=np.array([left[1],right[1],left[2],right[2]]))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(bundle, **payload)
    report = dict(status='completed', completed_at=datetime.now(timezone.utc).isoformat(),
        dataset=str(dataset), snapshot_shape=shape, slice_index=index, slice_axis='x',
        slice_center_kpc=float((ds.domain_left_edge[0]+(index+.5)*widths[0]).to('kpc').value),
        cells=int(nH.size), valid_cells=int((~excluded).sum()), excluded_cells=int(excluded.sum()),
        excluded_cold_cells=int(np.count_nonzero(excluded & (tq < 3000.))),
        X_H=float(cfg.X_H), hydrogen_mass_g=float(physics.m_H.to('g').value),
        NH_definition='2/(1/N_plus_z+1/N_minus_z), inclusive full-z rays, original unmasked gas',
        dvdr_definition='max(abs(div v)/3, numerical floor); central x derivative using halo',
        despotic_temperature='Accepted DESPOTIC table queried at fresh nH,NH,dVdr for every slice cell',
        display_mask='Exact accepted common emission-valid cells, applied after ray integration',
        table=str(table_path), table_sha256=_sha256(table_path),
        ranges={key: [float(np.nanmin(arr)),float(np.nanmax(arr))]
                for key,arr in payload.items() if key.endswith('_slice')},
        validation=dict(NH_max_relative_error=float(np.max(abs(NH-direct)/NH)),
            dVdr_central_difference='passed', exclusions='exact match with accepted cell IDs',
            table_bounds='all query coordinates inside bounds; no clipping'),
        source_sha256=hashes, snapshot_domain=domain, bundle_sha256=_sha256(bundle))
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    render(payload, report, output)
    print(json.dumps({key: report[key] for key in ('cells','valid_cells','excluded_cells','slice_center_kpc','validation')},indent=2))


if __name__ == '__main__':
    main()
