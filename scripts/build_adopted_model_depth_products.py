#!/usr/bin/env python3
"""Opt-in mixed-branch emission products from the checked fixed-depth table.

Uses the original full snapshot and accepted DESPOTIC checkpoint. Cold CIII
and CIV are explicitly omitted by the adopted prescription. This runner never
fills failed interpolation support or overwrites the historical products.
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
from quokka2s.cloudy_cell_coverage import validated_coverage_lookup
from quokka2s.tables.io import load_table
from quokka2s.tables.lookup import TableLookup
from scripts.check_despotic_snapshot_coverage import slab_windows, _validate_scan_provenance
from scripts.validate_cloudy_cell_checkpoint import (
    sha256, APPROVED_TABLE_SHA256, APPROVED_EXCLUDED_IDS, EXPECTED_SHAPE, EXPECTED_COLD_COUNT,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cloudy-table','cloudy-validation','coverage-report','checkpoint-dir',
                 'checkpoint-validation','despotic-table','dataset','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--slab-nx',type=int,default=8)
    parser.add_argument('--query-chunk',type=int,default=100000)
    parser.add_argument('--emission-only',action='store_true',
                        help='Validate all emissivities and sum luminosities without velocity spectra')
    parser.add_argument('--velocity-range-kms',type=float,default=200.)
    parser.add_argument('--channels',type=int,default=300)
    parser.add_argument('--spectral-workers',type=int,default=6)
    parser.add_argument('--max-slabs',type=int,help='Diagnostic subset only; never labelled full snapshot')
    args=parser.parse_args()
    if min(args.slab_nx,args.query_chunk,args.channels,args.spectral_workers,args.velocity_range_kms)<=0:
        parser.error('Numerical sizes must be positive')
    if args.max_slabs is not None and args.max_slabs<=0:
        parser.error('--max-slabs must be positive')
    if args.output_dir.exists():
        raise FileExistsError('Choose a new output directory')
    _,cloudy=validated_coverage_lookup(args.cloudy_table,args.cloudy_validation)
    coverage=json.loads(args.coverage_report.read_text())
    for path in (args.cloudy_table,args.cloudy_validation,args.despotic_table):
        if coverage['source_sha256'].get(str(path.resolve()))!=sha256(path):
            raise ValueError(f'Coverage input mismatch: {path}')
    if (coverage['total_cells']!=int(np.prod(EXPECTED_SHAPE))
            or coverage['groups']['hot']['flags']['unavailable_any_line']['cells']
            or coverage['groups']['hot']['flags']['outside_any_physical_axis']['cells']):
        raise ValueError('A full passed hot Cloudy coverage check is required')
    summary_path=args.checkpoint_dir/'model_depth_summary.json'
    states_path=args.checkpoint_dir/'cold_model_states.npz'
    summary=json.loads(summary_path.read_text())
    bridge=json.loads(args.checkpoint_validation.read_text())
    if bridge.get('status')!='passed':
        raise ValueError('Cell state checkpoint validation must pass')
    inputs=(args.cloudy_table,args.cloudy_validation,args.coverage_report,args.despotic_table,
            args.checkpoint_validation,summary_path,states_path)
    hashes={str(p.resolve()):sha256(p) for p in inputs}
    for p in (args.despotic_table,summary_path,states_path):
        if bridge['source_sha256'].get(str(p.resolve()))!=hashes[str(p.resolve())]:
            raise ValueError(f'Accepted checkpoint changed: {p}')
    if hashes[str(args.despotic_table.resolve())]!=APPROVED_TABLE_SHA256:
        raise ValueError('Wrong DESPOTIC table')
    if args.dataset.resolve()!=Path(summary['dataset']).resolve():
        raise ValueError('Exclusions only apply to the accepted snapshot')
    with np.load(states_path,allow_pickle=False) as source:
        states={name:np.array(source[name]) for name in (
            'flat_cell_index','rho_g_cm3','NH','dVdr','T_QUOKKA_K','u_erg_cm3',
            'T_DESPOTIC_K','mu_DESPOTIC','valid')}
    cold_ids=states['flat_cell_index']
    if (cold_ids.size!=EXPECTED_COLD_COUNT or np.any(np.diff(cold_ids)<=0)
            or not np.array_equal(cold_ids[~states['valid']],APPROVED_EXCLUDED_IDS)):
        raise ValueError('Cold inventory changed')
    import yt
    from quokka2s.pipeline.prep import physics_fields as physics, config as cfg
    from quokka2s.adopted_cell_emission import compute_adopted_cell_emission
    ds=yt.load(str(args.dataset.resolve()))
    shape=tuple(int(x) for x in ds.domain_dimensions)
    if shape!=EXPECTED_SHAPE or ds.max_level!=0 or cfg.DOWNSAMPLE_FACTOR!=1:
        raise ValueError('Expected the complete uniform snapshot')
    dsp=TableLookup(load_table(args.despotic_table))
    domain=_validate_scan_provenance(dsp.table,args.dataset,shape,cfg,physics)
    constants=dict(summary['constants'])
    if constants.pop('gamma')!=5/3:
        raise ValueError('Unexpected gamma')
    widths=ds.domain_width/ds.domain_dimensions
    volume=float(np.prod(widths.to('cm').value))
    if not np.isclose(volume,summary['cell_volume_cm3'],rtol=1e-14,atol=0):
        raise ValueError('Cell volume changed')
    accumulator=None
    if not args.emission_only:
        from quokka2s.adopted_spectral_products import AdoptedSpectralAccumulator, plot_adopted_spectra
        accumulator=AdoptedSpectralAccumulator(
            tuple(cloudy.line_keys)+('co10','co21'),
            np.linspace(-args.velocity_range_kms,args.velocity_range_kms,args.channels+1),
            constants['boltzmann_erg_K'],workers=args.spectral_workers)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    began=time.monotonic()
    counts=dict(all=0,cold=0,hot=0,excluded=0,retained=0)
    clipped={k:0 for k in ('nH','NH','dVdr')}
    luminosity=np.zeros((len(cloudy.line_keys)+2,2))
    zero_counts=np.zeros_like(luminosity,dtype=np.int64)
    mass={k:0. for k in ('all','cold','hot','excluded')}
    failure_context={}
    def status(state,**extra):
        data=dict(status=state,counts=counts,elapsed_seconds=time.monotonic()-began,**extra)
        temporary=args.output_dir/'status.tmp'
        temporary.write_text(json.dumps(data,indent=2)+'\n')
        temporary.replace(args.output_dir/'status.json')
    status('running')
    try:
        for slab,(ix,end,lo,hi,core) in enumerate(slab_windows(shape[0],args.slab_nx)):
            if args.max_slabs is not None and slab>=args.max_slabs:break
            edge=ds.domain_left_edge.copy();edge[0]+=lo*widths[0]
            grid=ds.covering_grid(0,edge,(hi-lo,*shape[1:]))
            bulk=np.asarray(grid.get_field_parameter('bulk_velocity'))
            if not np.isfinite(bulk).all() or np.any(bulk!=0):
                raise ValueError('Unexpected bulk velocity')
            rho=np.array(grid['gas','density'].to('g/cm**3')[core]).ravel()
            tq=np.array(grid['boxlib','temperature'][core],dtype=float).ravel()
            column=np.array(physics._column_density_H(None,grid).to('cm**-2')[core]).ravel()
            u=np.array(physics._internal_energy_density(None,grid).to('erg/cm**3')[core]).ravel()
            dvdr=np.array(physics._dVdr_lvg(None,grid).to('s**-1')[core]).ravel()
            vz=None if accumulator is None else np.array(grid['gas','velocity_z'].to('km/s')[core]).ravel()
            del grid
            for start in range(0,rho.size,args.query_chunk):
                stop=min(start+args.query_chunk,rho.size);sl=slice(start,stop)
                offset=ix*shape[1]*shape[2]+start
                ids=np.arange(offset,offset+stop-start,dtype=np.int64)
                failure_context=dict(first_cell_id=int(ids[0]),last_cell_id=int(ids[-1]))
                cold=tq[sl]<3000
                left,right=np.searchsorted(cold_ids,[offset,offset+stop-start])
                if not np.array_equal(ids[cold],cold_ids[left:right]):
                    raise ValueError('Cold membership changed')
                for name,values in (('rho_g_cm3',rho[sl]),('NH',column[sl]),('dVdr',dvdr[sl]),
                                    ('T_QUOKKA_K',tq[sl]),('u_erg_cm3',u[sl])):
                    if not np.allclose(values[cold],states[name][left:right],rtol=1e-12,atol=0,equal_nan=True):
                        raise ValueError(f'Cold checkpoint differs: {name}')
                td=np.full(ids.shape,np.nan);mud=td.copy()
                td[cold]=states['T_DESPOTIC_K'][left:right]
                mud[cold]=states['mu_DESPOTIC'][left:right]
                excluded=np.isin(ids,APPROVED_EXCLUDED_IDS)
                queries=prepare_cloudy_cell_queries(rho[sl],column[sl],tq[sl],u[sl],td,mud,
                    authorized_excluded=excluded,**constants)
                emission=compute_adopted_cell_emission(queries,dvdr[sl],dsp,cloudy)
                for key,flag in emission.despotic_clipped.items():clipped[key]+=int(flag.sum())
                for branch,selected in enumerate((cold&emission.valid,~cold&emission.valid)):
                    values=emission.emissivity_erg_s_cm3[:,selected]
                    luminosity[:,branch]+=values.sum(axis=1)*volume
                    zero_counts[:,branch]+=(values==0).sum(axis=1)
                if accumulator is not None:
                    # Excluded cells remain NaN in emission products. Remove
                    # them explicitly before the spectral input validation.
                    use=emission.valid
                    accumulator.add(vz[sl][use],emission.thermal_temperature_K[:,use],
                        emission.emissivity_erg_s_cm3[:,use],volume,cold_mask=cold[use])
                for name,selected in (('all',np.ones(ids.shape,dtype=bool)),('cold',cold),
                                      ('hot',~cold),('excluded',excluded)):
                    counts[name]+=int(selected.sum());mass[name]+=float(rho[sl][selected].sum()*volume)
                counts['retained']+=int(emission.valid.sum())
            status('running')
            print(f'Adopted emission: {counts["all"]}/{np.prod(shape)} cells',flush=True)
        full=counts['all']==int(np.prod(shape))
        if full:
            if counts['cold']!=EXPECTED_COLD_COUNT or counts['excluded']!=2:
                raise ValueError('Final counts disagree with checkpoint')
            for group,old in (('all','all'),('cold','cold_T_QUOKKA_lt3000K'),('hot','hot_T_QUOKKA_ge3000K')):
                if not np.isclose(mass[group],summary['model_depth'][old]['mass_g'],rtol=1e-12,atol=0):
                    raise ValueError(f'{group} mass differs from accepted snapshot')
        for p in inputs:
            if sha256(p)!=hashes[str(p.resolve())]:raise ValueError(f'Input changed: {p}')
        spectral_report=None
        if accumulator is not None:
            payload,spectral_report=accumulator.finalize()
            if not np.allclose(payload['input_luminosity_erg_s'],luminosity,rtol=1e-12,atol=0):
                raise ValueError('Spectral input luminosities differ from the independent cell sums')
            if np.any(payload['captured_luminosity_erg_s']>luminosity*(1+1e-12)):
                raise ValueError('Spectrum contains more luminosity than the cells supplied')
            projected_area=float((ds.domain_width[0]*ds.domain_width[1]).to('cm**2').value)
            payload['projected_area_cm2']=np.asarray(projected_area)
            payload['line_of_sight']=np.asarray('z')
            np.savez_compressed(args.output_dir/'spectra.npz',**payload)
            plot_adopted_spectra(payload,args.output_dir/'spectra',projected_area_cm2=projected_area,
                title='Adopted LOS-z emission: checked Cloudy + DESPOTIC')
        result=dict(status='completed' if full else 'partial diagnostic',full_snapshot=full,
            completed_at=datetime.now(timezone.utc).isoformat(),counts=counts,mass_g=mass,
            line_keys=list(emission.line_keys),regimes=['cold','hot'],
            luminosity_erg_s=luminosity.tolist(),true_zero_cells=zero_counts.tolist(),
            despotic_coordinate_clipped_cells=clipped,source_sha256=hashes,snapshot_domain=domain,
            cell_volume_cm3=volume,spectral_report=spectral_report,
            cold_CIII_CIV='Explicitly omitted by user on 2026-09-12; hot contributions only',
            excluded_cell_ids=APPROVED_EXCLUDED_IDS.tolist(),
            interpretation='Local volume emission with inherited Cloudy escape treatment and DESPOTIC LVG; no foreground dust or intercell transfer added.',
            elapsed_seconds=time.monotonic()-began,
            code_sha256={str(p.resolve()):sha256(p) for p in (
                Path(__file__),ROOT/'src/quokka2s/adopted_cell_emission.py',
                ROOT/'src/quokka2s/cloudy_hot_emission.py',ROOT/'src/quokka2s/cloudy_cell_queries.py',
                ROOT/'src/quokka2s/adopted_spectral_products.py')})
        (args.output_dir/'emission_report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        status('completed' if full else 'partial diagnostic')
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}',failure_context=failure_context)
        raise


if __name__=='__main__':main()
