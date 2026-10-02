#!/usr/bin/env python3
"""Rebuild Figure 1 from the accepted snapshot and DESPOTIC inputs.

The displayed maps use the emission-valid cells, mixed temperature, and true
line-of-sight hydrogen columns. Full-gas maps are cached for numerical checks.
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
from quokka2s.emission_processing import validate_accepted_inputs
from scripts.check_despotic_snapshot_coverage import slab_windows, _validate_scan_provenance, _sha256
from quokka2s.adopted_multiview import MultiviewAccumulator
from quokka2s.tables.io import load_table
from quokka2s.tables.lookup import TableLookup


def plot_maps(payload, report, stem):
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    from matplotlib.ticker import LogLocator, LogFormatterMathtext, MaxNLocator

    factor = report['X_H']/report['hydrogen_mass_g']
    columns = [
        ('rho_g_cm3', r'Density slice', r'$\rho\;[\mathrm{g\,cm^{-3}}]$', 1., 'viridis'),
        ('sigma_g_cm2', r'Hydrogen column', r'$N_{\rm H}^{\rm LOS}\;[\mathrm{cm^{-2}}]$', factor, 'viridis'),
        ('vz_kms', r'Vertical velocity', r'$\langle v_z\rangle_\rho\;[\mathrm{km\,s^{-1}}]$', 1., 'RdBu_r'),
        ('T_mixed_K', r'Temperature', r'$\langle T\rangle_\rho\;[\mathrm{K}]$', 1., 'viridis'),
    ]
    # Equal physical aspect in both views, including the entire 8-kpc height.
    # Keep one shared scale per column and native pixels (no display smoothing).
    fig = plt.figure(figsize=(8.25, 16.4))
    left, right, top = .075, .975, .97
    gap = .062
    panel_width = (right-left-3*gap)/4
    panel_inches = panel_width*fig.get_figwidth()
    edge_height = panel_inches*8/fig.get_figheight()
    face_height = panel_inches/fig.get_figheight()
    edge_bottom = top-edge_height
    face_top = edge_bottom-.037
    face_bottom = face_top-face_height
    norms = {}
    for col, (key, title, label, conversion, cmap) in enumerate(columns):
        values = [payload[f'valid_{view}_{key}']*conversion for view in ('edge', 'face')]
        finite = np.concatenate([arr[np.isfinite(arr)] for arr in values])
        if key == 'vz_kms':
            bound = float(np.max(np.abs(finite)))
            norm = TwoSlopeNorm(vmin=-bound, vcenter=0., vmax=bound)
        else:
            positive = finite[finite > 0]
            norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
        norms[key] = dict(minimum=norm.vmin, maximum=norm.vmax, clipped=False)
        x = left+col*(panel_width+gap)
        for row, (view, arr, y, h) in enumerate(zip(
                ('edge', 'face'), values, (edge_bottom, face_bottom), (edge_height, face_height))):
            ax = fig.add_axes([x, y, panel_width, h])
            extent = payload[f'extent_{view}_kpc']
            im = ax.imshow(arr.T, origin='lower', extent=extent, aspect='equal',
                           interpolation='nearest', cmap=cmap, norm=norm, rasterized=True)
            ax.tick_params(labelsize=8, direction='out', length=3, pad=2)
            ax.set_xlabel(r'$y\;[\mathrm{kpc}]$' if row == 0 else r'$x\;[\mathrm{kpc}]$',
                          fontsize=9, labelpad=3)
            ax.set_xticks([0., .5])
            if row == 0:
                ax.set_yticks(np.arange(-3., 4.))
                ax.set_title(title, fontsize=10, pad=8)
                ax.scatter(payload['particles_edge_y_kpc'], payload['particles_edge_z_kpc'],
                           s=.48, c='#ec2424', marker='o', linewidths=0, alpha=.75,
                           rasterized=True)
            else:
                ax.set_yticks([0., .5])
            if col == 0:
                ax.set_ylabel(r'$z\;[\mathrm{kpc}]$' if row == 0 else r'$y\;[\mathrm{kpc}]$',
                              fontsize=9, labelpad=3)
            else:
                ax.tick_params(labelleft=False)
        cax = fig.add_axes([x-.006, face_bottom-.064, panel_width+.012, .008])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.ax.tick_params(labelsize=7.5, length=3, pad=2)
        if isinstance(norm, LogNorm):
            cb.locator = LogLocator(base=10, numticks=4)
            cb.formatter = LogFormatterMathtext()
            cb.update_ticks()
            cb.ax.minorticks_off()
        else:
            cb.locator = MaxNLocator(nbins=3, symmetric=True)
            cb.update_ticks()
        cb.set_label(label, fontsize=9, labelpad=4)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.pdf', '.png'):
        fig.savefig(stem.with_suffix(ext), dpi=300, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)
    caption = r'''\begin{figure*}
    \centering
    \includegraphics[width=\textwidth,height=0.78\textheight,keepaspectratio]
    {multiview_figure_particles.pdf}
    \caption{Edge-on (top; viewed along $x$) and face-on (bottom;
    viewed along $z$) maps of the simulation, using the same valid cells
    as the emission analysis. From left to right: central density slices,
    line-of-sight hydrogen column density $N_{\rm H}^{\rm LOS}=\int n_{\rm H}\,dl$,
    density-weighted vertical velocity, and density-weighted temperature.
    The temperature is taken from DESPOTIC where $T_{\rm Q}<3000\,\mathrm{K}$
    and from QUOKKA otherwise. Red dots in the upper panels mark stellar-population
    particles within the central $x$ slab of thickness $L_x/20$.}
    \label{fig:multiview_maps}
\end{figure*}
'''
    stem.with_name(stem.name+'_figure.tex').write_text(caption)
    stem.with_name(stem.name+'_display.json').write_text(json.dumps(dict(
        selection='valid', temperature='mixed', column='LOS integral of nH',
        normalization=norms, bundle_sha256=report['bundle_sha256'],
        figure_size_inches=[8.25,16.4], spatial_aspect='equal'), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {
        'dataset': ROOT/'inputs/snapshots/plt0655228',
        'accepted-despotic': ROOT/'output/despotic_default_parallel_20260918/interpolated/accepted_table.json',
        'cloudy-table': ROOT/'data/cloudy_hm2012_attgrid_ism_nh21_cmb_cr_defaultabund_eightline_jeans_7x10x21.npz',
        'cloudy-audit': ROOT/'output/default_table_reuse_20260918/cloudy_reuse_coverage.json',
        'emission-report': ROOT/'output/cloudy_rgi_comparison_20260920/full/emission_report.json',
        'output-dir': ROOT/'output/multiview/2026-09-22_adopted',
        'figure-stem': ROOT/'output/pdf/multiview_figure_particles',
    }
    for name, path in defaults.items():
        parser.add_argument('--'+name, type=Path, default=path)
    parser.add_argument('--slab-nx', type=int, default=8)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    bundle_path = args.output_dir/'multiview_maps.npz'
    report_path = args.output_dir/'multiview_report.json'
    if args.plot_only:
        report = json.loads(report_path.read_text())
        if report['status'] != 'completed' or _sha256(bundle_path) != report['bundle_sha256']:
            raise ValueError('Expected intact completed map bundle')
        with np.load(bundle_path, allow_pickle=False) as source:
            payload = {key: np.array(source[key]) for key in source.files}
        plot_maps(payload, report, args.figure_stem)
        return
    if args.slab_nx < 1:
        parser.error('slab-nx must be positive')
    if args.output_dir.exists():
        raise FileExistsError('Use a new output directory or --plot-only')
    manifest, coverage, excluded_ids, table_path, _, hashes = validate_accepted_inputs(
        args.accepted_despotic, args.cloudy_table, args.cloudy_audit, args.dataset)
    reference = json.loads(args.emission_report.read_text())
    if (reference.get('status') != 'completed' or not reference['full_snapshot']
            or reference['source_sha256'] != hashes or manifest['excluded_cold_cell_count'] != 0):
        raise ValueError('Maps must match accepted emission provenance with no missing cold cells')
    hashes[str(args.emission_report.resolve())] = _sha256(args.emission_report)
    import yt
    from quokka2s.pipeline.prep import physics_fields as physics, config as cfg
    ds = yt.load(str(args.dataset.resolve()))
    shape = tuple(int(x) for x in ds.domain_dimensions)
    if shape != tuple(manifest['snapshot_shape']) or ds.max_level != 0 or cfg.DOWNSAMPLE_FACTOR != 1:
        raise ValueError('Expected the accepted uniform full-resolution snapshot')
    dsp = TableLookup(load_table(table_path))
    domain = _validate_scan_provenance(dsp.table, args.dataset, shape, cfg, physics)
    widths = ds.domain_width/ds.domain_dimensions
    widths_cm = widths.to('cm').value
    volume = float(np.prod(widths_cm))
    hydrogen_mass = float(physics.m_H.to('g').value)
    accumulator = MultiviewAccumulator(shape, widths_cm)
    counts = dict(all=0, cold=0, hot=0, excluded=0, excluded_cold=0, retained=0)
    masses = {key: 0. for key in ('all', 'cold', 'hot', 'excluded', 'retained')}
    code_paths = (Path(__file__), ROOT/'src/quokka2s/adopted_multiview.py',
                  Path(physics.__file__), ROOT/'src/quokka2s/tables/lookup.py')
    code_hashes = {str(p.resolve()): _sha256(p) for p in code_paths}
    began = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def status(state, **extra):
        path = args.output_dir/'status.tmp'
        path.write_text(json.dumps(dict(status=state, counts=counts,
            elapsed_seconds=time.monotonic()-began, **extra), indent=2)+'\n')
        path.replace(args.output_dir/'status.json')

    status('running')
    try:
        for slab, (ix, end, lo, hi, core) in enumerate(slab_windows(shape[0], args.slab_nx)):
            edge = ds.domain_left_edge.copy()
            edge[0] += lo*widths[0]
            grid = ds.covering_grid(0, edge, (hi-lo, *shape[1:]))
            if np.any(np.asarray(grid.get_field_parameter('bulk_velocity')) != 0):
                raise ValueError('Unexpected bulk-velocity subtraction')
            rho = np.array(grid['gas', 'density'].to('g/cm**3')[core])
            tq = np.array(grid['boxlib', 'temperature'][core], dtype=float)
            vz = np.array(grid['gas', 'velocity_z'].to('km/s')[core])
            cold = tq < 3000.
            mixed = tq.copy()
            # The shielding rays include the original gas, before any display mask.
            # Only cold cells require DESPOTIC temperature for this figure.
            if cold.any():
                nh = rho[cold]*cfg.X_H/hydrogen_mass
                column = np.asarray(physics._column_density_H(None, grid).to('cm**-2')[core])[cold]
                dvdr = np.asarray(physics._dVdr_lvg(None, grid).to('s**-1')[core])[cold]
                raw = (nh, column, dvdr)
                for name, values, axis in zip(('nH', 'NH', 'dVdr'), raw,
                        (dsp.table.nH_values, dsp.table.col_density_values, dsp.table.dVdr_values)):
                    if (not np.isfinite(values).all() or np.any(values < axis[0]) or np.any(values > axis[-1])):
                        raise ValueError(f'Cold DESPOTIC {name} outside accepted domain')
                mixed[cold] = dsp.temperature(*raw)
            del grid
            # Exact accepted flat indices, with x-major and z-fastest ordering.
            valid = np.ones(rho.size, dtype=bool)
            first = ix*shape[1]*shape[2]
            local_exclusions = excluded_ids[(excluded_ids >= first) & (excluded_ids < first+rho.size)]-first
            valid[local_exclusions] = False
            valid = valid.reshape(rho.shape)
            if np.any(cold & ~valid) or not np.isfinite(mixed).all() or np.any(mixed <= 0):
                raise ValueError('Unexpected missing mixed temperature or excluded cold cell')
            accumulator.add(ix, rho, vz, tq, mixed, valid)
            for key, take in (('all', np.ones(rho.shape, dtype=bool)), ('cold', cold),
                              ('hot', ~cold), ('excluded', ~valid), ('retained', valid)):
                counts[key] += int(take.sum())
                masses[key] += float(rho[take].sum()*volume)
            status('running', completed_slabs=slab+1)
            print(f'Multiview: {counts["all"]}/{manifest["total_cells"]} cells; '
                  f'{time.monotonic()-began:.1f} s', flush=True)
            del rho, tq, vz, mixed, valid, cold
            gc.collect()
        if counts != reference['counts']:
            raise ValueError('Map selection differs from accepted emission')
        for key, value in reference['mass_g'].items():
            np.testing.assert_allclose(masses[key], value, rtol=1e-12)
        payload = accumulator.payload()
        left = ds.domain_left_edge.to('kpc').value
        right = ds.domain_right_edge.to('kpc').value
        payload['extent_edge_kpc'] = np.array([left[1], right[1], left[2], right[2]])
        payload['extent_face_kpc'] = np.array([left[0], right[0], left[1], right[1]])
        particles = ds.all_data()
        ptype = 'StochasticStellarPop_particles'
        positions = [np.asarray(particles[ptype, 'particle_position_'+axis].to('kpc')) for axis in 'xyz']
        xmid = (left[0]+right[0])/2
        halfdepth = (right[0]-left[0])/40
        take = np.abs(positions[0]-xmid) <= halfdepth
        payload['particles_edge_y_kpc'] = positions[1][take]
        payload['particles_edge_z_kpc'] = positions[2][take]
        # Both independent sightlines must recover the same retained gas mass.
        validation = {}
        volume_totals = accumulator.report()['selections']
        for selection, mass_key in (('all', 'all'), ('valid', 'retained')):
            for view, area in (('edge', widths_cm[1]*widths_cm[2]),
                               ('face', widths_cm[0]*widths_cm[1])):
                mass_from_map = float(payload[f'{selection}_{view}_sigma_g_cm2'].sum()*area)
                np.testing.assert_allclose(mass_from_map, masses[mass_key], rtol=1e-12)
                validation[f'{selection}_{view}_mass_g'] = mass_from_map
                sigma = payload[f'{selection}_{view}_sigma_g_cm2']
                for field, total_key in (('vz_kms', 'momentum_z_g_kms'),
                        ('T_quokka_K', 'T_quokka_mass_g_K'), ('T_mixed_K', 'T_mixed_mass_g_K')):
                    average = payload[f'{selection}_{view}_{field}']
                    numerator = np.where(sigma > 0., average, 0.)*sigma*area
                    summed = float(numerator.sum())
                    np.testing.assert_allclose(summed, volume_totals[selection][total_key],
                                               rtol=1e-12, atol=float(np.abs(numerator).sum()*1e-14))
                    validation[f'{selection}_{view}_{total_key}'] = summed
        for mapping in (hashes, code_hashes):
            for path, digest in mapping.items():
                if _sha256(path) != digest:
                    raise ValueError(f'Source changed during scan: {path}')
        np.savez_compressed(bundle_path, **payload)
        report = dict(status='completed', completed_at=datetime.now(timezone.utc).isoformat(),
            dataset=str(args.dataset.resolve()), full_snapshot=True, shape=shape,
            counts=counts, mass_g=masses, cell_widths_cm=widths_cm.tolist(),
            X_H=float(cfg.X_H), hydrogen_mass_g=hydrogen_mass,
            temperature='DESPOTIC for T_QUOKKA < 3000 K; QUOKKA otherwise, then density weighted',
            display_selection='Exact accepted emission-valid cells',
            density='Central plane slices at x index nx//2 and z index nz//2',
            column='Integral nH dl along x (edge) or z (face); no attenuation clipping',
            velocity='Density-weighted original vz along each view; no bulk subtraction',
            shielding_for_despotic='Original unmasked gas +/-z harmonic mean before display selection',
            particles=dict(type=ptype, total=len(take), shown=int(take.sum()),
                view='edge only', x_slab_kpc=[float(xmid-halfdepth),float(xmid+halfdepth)]),
            accumulator=accumulator.report(), validation=validation, snapshot_domain=domain,
            bundle_sha256=_sha256(bundle_path), source_sha256=hashes, code_sha256=code_hashes,
            elapsed_seconds=time.monotonic()-began)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        plot_maps(payload, report, args.figure_stem)
        status('completed')
        print(f'Completed Figure 1: {args.figure_stem}', flush=True)
    except Exception as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
