#!/usr/bin/env python3
"""Full-snapshot diagnostic coverage of a completed, independently checked table.

All Cloudy lines are diagnosed in both temperature branches. These counts
describe table availability, not an adopted per-line emission prescription.
No failure is filled, dropped, or accepted by a numerical threshold here.
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
from quokka2s.cloudy_cell_coverage import validated_coverage_lookup, classify_cell_queries, CellCoverageTotals
from quokka2s.tables.io import load_table
from scripts.check_despotic_snapshot_coverage import slab_windows, _validate_scan_provenance
from scripts.validate_cloudy_cell_checkpoint import (
    sha256, APPROVED_TABLE_SHA256, APPROVED_EXCLUDED_IDS, EXPECTED_SHAPE, EXPECTED_COLD_COUNT,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloudy-table', type=Path, required=True)
    parser.add_argument('--cloudy-validation', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-validation', type=Path, required=True)
    parser.add_argument('--despotic-table', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--slab-nx', type=int, default=8)
    parser.add_argument('--query-chunk', type=int, default=100000)
    args = parser.parse_args()
    if min(args.slab_nx,args.query_chunk) <= 0:
        parser.error('slab and query sizes must be positive')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    # Reject incomplete/absent Cloudy products before loading the snapshot.
    raw,checked = validated_coverage_lookup(args.cloudy_table,args.cloudy_validation)
    summary_path = args.checkpoint_dir/'model_depth_summary.json'
    states_path = args.checkpoint_dir/'cold_model_states.npz'
    summary = json.loads(summary_path.read_text())
    bridge_check = json.loads(args.checkpoint_validation.read_text())
    if bridge_check.get('status') != 'passed':
        raise ValueError('Cell-query checkpoint comparison must pass first')
    inputs = (args.cloudy_table,args.cloudy_validation,args.checkpoint_validation,
              args.despotic_table,summary_path,states_path)
    hashes = {str(p.resolve()):sha256(p) for p in inputs}
    for path in (args.despotic_table,summary_path,states_path):
        if bridge_check['source_sha256'].get(str(path.resolve())) != hashes[str(path.resolve())]:
            raise ValueError(f'Accepted cell checkpoint input changed: {path}')
    if hashes[str(args.despotic_table.resolve())] != APPROVED_TABLE_SHA256:
        raise ValueError('DESPOTIC table differs from accepted setup')
    if (args.dataset.resolve() != Path(summary['dataset']).resolve()
            or args.dataset.name != 'plt0655228' or tuple(summary['shape']) != EXPECTED_SHAPE):
        raise ValueError('Cell exclusions apply only to the approved source snapshot')
    with np.load(states_path,allow_pickle=False) as source:
        cold_states = {name:np.array(source[name]) for name in (
            'flat_cell_index','rho_g_cm3','NH','T_QUOKKA_K','u_erg_cm3',
            'T_DESPOTIC_K','mu_DESPOTIC','valid')}
    cold_ids = cold_states['flat_cell_index']
    if (cold_ids.size != EXPECTED_COLD_COUNT or np.any(np.diff(cold_ids)<=0)
            or not np.array_equal(cold_ids[~cold_states['valid']],APPROVED_EXCLUDED_IDS)):
        raise ValueError('Cold state inventory or exclusion IDs changed')
    import yt
    from quokka2s.pipeline.prep import physics_fields as physics, config as cfg
    ds=yt.load(str(args.dataset.resolve()))
    shape=tuple(int(x) for x in ds.domain_dimensions)
    if shape != EXPECTED_SHAPE or ds.max_level != 0 or cfg.DOWNSAMPLE_FACTOR != 1:
        raise ValueError('Full uniform snapshot required')
    domain = _validate_scan_provenance(load_table(args.despotic_table),args.dataset,shape,cfg,physics)
    constants = dict(summary['constants'])
    if constants.pop('gamma') != 5/3:
        raise ValueError('Unexpected gamma in accepted state checkpoint')
    widths=ds.domain_width/ds.domain_dimensions
    volume=float(np.prod(widths.to('cm').value))
    if not np.isclose(volume,summary['cell_volume_cm3'],rtol=1e-14,atol=0):
        raise ValueError('Cell volume changed')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    totals=CellCoverageTotals()
    begin=time.monotonic()
    scanned=cold_scanned=excluded_scanned=0
    first_issue_ids={}
    def status(kind,**extra):
        target=args.output_dir/'status.json'
        temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(status=kind,scanned_cells=scanned,
            total_cells=int(np.prod(shape)),elapsed_seconds=time.monotonic()-begin,**extra),indent=2)+'\n')
        temporary.replace(target)
    status('running')
    try:
        for ix,end,lo,hi,core in slab_windows(shape[0],args.slab_nx):
            edge=ds.domain_left_edge.copy();edge[0]+=lo*widths[0]
            grid=ds.covering_grid(level=0,left_edge=edge,dims=(hi-lo,*shape[1:]))
            bulk=np.asarray(grid.get_field_parameter('bulk_velocity'))
            if not np.isfinite(bulk).all() or np.any(bulk != 0):
                raise ValueError('Nonzero bulk velocity would change internal energy subtraction')
            rho=np.array(grid['gas','density'].to('g/cm**3')[core],copy=True).ravel()
            tq=np.array(grid['boxlib','temperature'][core],dtype=float,copy=True).ravel()
            nhcol=np.array(physics._column_density_H(None,grid).to('cm**-2')[core],copy=True).ravel()
            u=np.array(physics._internal_energy_density(None,grid).to('erg/cm**3')[core],copy=True).ravel()
            del grid
            slab_size=(end-ix)*shape[1]*shape[2]
            if any(v.size!=slab_size for v in (rho,tq,nhcol,u)):
                raise ValueError('Slab core dimensions disagree')
            for start in range(0,slab_size,args.query_chunk):
                stop=min(start+args.query_chunk,slab_size);sl=slice(start,stop)
                offset=ix*shape[1]*shape[2]+start
                ids=np.arange(offset,offset+stop-start,dtype=np.int64)
                cold=tq[sl]<3000
                left,right=np.searchsorted(cold_ids,[offset,offset+stop-start])
                if not np.array_equal(ids[cold],cold_ids[left:right]):
                    raise ValueError('Fresh snapshot cold-cell membership differs from accepted checkpoint')
                for name,actual in (('rho_g_cm3',rho[sl]),('NH',nhcol[sl]),
                                    ('T_QUOKKA_K',tq[sl]),('u_erg_cm3',u[sl])):
                    if not np.allclose(actual[cold],cold_states[name][left:right],rtol=1e-12,atol=0,equal_nan=True):
                        raise ValueError(f'Fresh cold state differs from accepted checkpoint: {name}')
                td=np.full(ids.shape,np.nan);mu=td.copy()
                td[cold]=cold_states['T_DESPOTIC_K'][left:right]
                mu[cold]=cold_states['mu_DESPOTIC'][left:right]
                excluded=np.isin(ids,APPROVED_EXCLUDED_IDS)
                queries=prepare_cloudy_cell_queries(rho[sl],nhcol[sl],tq[sl],u[sl],td,mu,
                    authorized_excluded=excluded,**constants)
                flags=classify_cell_queries(queries,raw,checked)
                totals.add(flags,queries.state.cold_mask,rho[sl]*volume)
                for name,flag in flags.items():
                    if name.startswith(('outside_','unavailable','raw_failure')):
                        prior=first_issue_ids.setdefault(name,[])
                        prior.extend(ids[flag][:max(0,20-len(prior))].tolist())
                scanned+=ids.size;cold_scanned+=int(cold.sum());excluded_scanned+=int(excluded.sum())
            status('running')
            print(f'Cloudy coverage: {scanned}/{np.prod(shape)} cells',flush=True)
        if (scanned != np.prod(shape) or cold_scanned != EXPECTED_COLD_COUNT or excluded_scanned != 2):
            raise ValueError('Full-snapshot or authorized-exclusion counts do not match')
        groups=totals.result()
        for group,original_group in (('all','all'),('cold','cold_T_QUOKKA_lt3000K'),('hot','hot_T_QUOKKA_ge3000K')):
            if not np.isclose(groups[group]['mass_g'],summary['model_depth'][original_group]['mass_g'],rtol=1e-12,atol=0):
                raise ValueError(f'Full-snapshot {group} mass differs from accepted scan')
        for path in inputs:
            if sha256(path) != hashes[str(path.resolve())]:
                raise ValueError(f'Input changed during coverage scan: {path}')
        result=dict(status='completed diagnostic; adoption not decided',completed_at=datetime.now(timezone.utc).isoformat(),
            source_sha256=hashes,dataset=str(args.dataset.resolve()),shape=shape,total_cells=scanned,
            cold_cells=cold_scanned,authorized_excluded_cells=excluded_scanned,cell_volume_cm3=volume,
            snapshot_domain=domain,groups=groups,first_issue_cell_ids=first_issue_ids,
            line_scope='Every stored Cloudy line in both regimes; not an adopted emission-branch selection.',
            unavailable_policy='Raw map failures OR independent per-state validation failures; no numerical fill.',
            denominators='All cells and all gas mass in each group, including authorized exclusions.',
            original_cell_NH_preserved=True,elapsed_seconds=time.monotonic()-begin,
            checker_sources={str(p.resolve()):sha256(p) for p in (Path(__file__),
                Path(__file__).resolve().parents[1]/'src/quokka2s/cloudy_cell_coverage.py',
                Path(__file__).resolve().parents[1]/'src/quokka2s/cloudy_cell_queries.py')})
        with (args.output_dir/'coverage.json').open('x') as handle:
            json.dump(result,handle,indent=2);handle.write('\n')
        status('completed')
    except BaseException as error:
        status('failed',error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':main()
