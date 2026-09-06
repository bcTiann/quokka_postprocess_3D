#!/usr/bin/env python3
"""Build the manuscript's nine all-cell phase histograms (no C III/C IV).

Run from the repository root. Outputs are new PNG/PDF plus a small histogram
bundle and a provenance/conservation report. --plot-only reuses that bundle.
This does not invoke the historical mu-based high-temperature line fields.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import h5py
import numpy as np
import yt
from yt.units.physical_constants import mh

from quokka2s.cloudy_sixline_lookup import CloudySixLineLookup
from quokka2s.pipeline.prep import config as cfg
from quokka2s.pipeline.prep.physics_fields import (
    _clip_to_table_domain, _HI_emissivity_from_number_density,
    effective_halpha_recombination_coefficient, h, c, lambda_Halpha,
)
from quokka2s.pipeline.tasks.adopted_phase_hist import (
    PANELS, DexHistogram, select_emissivities, plot_panels,
)
from quokka2s.tables import load_table
from quokka2s.tables.lookup import TableLookup
from plot_hm12_filtered_ism_sixline_spectra import _recompute_dvdr_slab

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLOUDY = ROOT / 'data/cloudy_hm2012_attgrid_ism_nh21_cmb_cr_defaultabund_sixline_jeans_7x10x21.npz'


def emissivities(tq, td, nh, column, dvdr, dsp, cloudy):
    """Evaluate only the applicable branch, keeping zero emission distinct from NaN."""
    safe = _clip_to_table_domain(dsp, nh, column, dvdr)
    low = tq < 3000
    cold, hot = {}, {}
    for key, species in (('co10', 'CO'), ('co21', 'CO21')):
        cold[key] = safe[0] * dsp.line_field(species, 'lumPerH', *safe)
    for key in ('cii', 'halpha', 'hi21'):
        cold[key], hot[key] = np.zeros_like(tq), np.zeros_like(tq)
    if low.any():
        cold_safe = tuple(v[low] for v in safe)
        number = dsp.number_densities(('e-', 'H+', 'H'), *cold_safe)
        for key, value in number.items():
            if not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError(f'Invalid DESPOTIC number density: {key}')
        cold['cii'][low] = cold_safe[0] * dsp.line_field('C+', 'lumPerH', *cold_safe)
        photon = float(((h * c) / lambda_Halpha).in_cgs().value)
        cold['halpha'][low] = (photon * effective_halpha_recombination_coefficient(td[low])
                               * number['e-'] * number['H+'])
        cold['hi21'][low] = _HI_emissivity_from_number_density(number['H'])
    if (~low).any():
        sample = cloudy.sample(tq[~low], nh[~low], column[~low])
        for key in hot:
            hot[key][~low] = (sample.emissivity_per_nH2[cloudy.line_keys.index(key)]
                              * nh[~low] ** 2)
    return select_emissivities(tq, cold, hot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument('--despotic-table', type=Path, default=Path(cfg.DESPOTIC_TABLE_PATH))
    parser.add_argument('--cloudy-table', type=Path, default=DEFAULT_CLOUDY)
    parser.add_argument('--column-cache', type=Path,
                        default=ROOT / 'intermediates/plt0655228/fields/field_gas_column_density_H.h5')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'output/phase_histograms/2026-09-06_adopted')
    parser.add_argument('--slab-nz', type=int, default=16)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = args.output_dir / 'phase_histograms.npz'
    report = args.output_dir / 'phase_histograms.json'
    png = args.output_dir / 'phase_histograms_9panel.png'
    pdf = args.output_dir / 'phase_histograms_9panel.pdf'
    if args.plot_only:
        with np.load(bundle) as data:
            panels = {key: {name: data[f'{key}__{name}'] for name in ('H', 'x_edges', 'y_edges')}
                      for key, _, _ in PANELS}
        plot_panels(panels, png, pdf)
        return
    if any(path.exists() for path in (bundle, report, png, pdf)):
        raise FileExistsError('Choose a new output directory; use --plot-only to replot')
    if args.slab_nz < 1:
        raise ValueError('--slab-nz must be positive')
    started = time.monotonic()
    ds = yt.load(str(args.dataset))
    ds.force_periodicity()
    if ds.max_level != 0:
        raise ValueError('This runner expects the current uniform level-0 snapshot')
    dims = tuple(int(v) for v in ds.domain_dimensions)
    width = np.asarray(ds.domain_width.to('cm') / ds.domain_dimensions)
    volume = float(np.prod(width))
    dsp = TableLookup(load_table(args.despotic_table))
    cloudy = CloudySixLineLookup(args.cloudy_table)
    histograms = {key: DexHistogram(.2) for key, _, _ in PANELS}
    counts = {'cells': 0, 'low_temperature_cells': 0}
    counts.update({name: 0 for name in ('dsp_nH_clipped', 'dsp_NH_clipped', 'dsp_dvdr_clipped')})
    ranges = {}
    ray_positions = ((0, 0), (dims[0] // 2, dims[1] // 2), (dims[0]-1, dims[1]-1))
    density_rays = {position: [] for position in ray_positions}
    with h5py.File(args.column_cache, 'r') as column_file:
        if column_file['data'].shape != dims:
            raise ValueError('Column cache shape differs from the snapshot')
        if column_file.attrs.get('field_name') != 'column_density_H':
            raise ValueError('Wrong column cache field')
        for iz in range(0, dims[2], args.slab_nz):
            size = min(args.slab_nz, dims[2] - iz)
            edge = ds.domain_left_edge.copy()
            edge[2] += iz * ds.domain_width[2] / dims[2]
            grid = ds.covering_grid(0, edge, (dims[0], dims[1], size))
            rho = np.asarray(grid['gas', 'density'].to('g/cm**3')).reshape(-1)
            tq = np.asarray(grid['boxlib', 'temperature']).reshape(-1)
            del grid
            nh = rho * float(cfg.X_H) / float(mh.to('g').value)
            for ix, iy in ray_positions:
                density_rays[ix, iy].append(nh.reshape(dims[0], dims[1], size)[ix, iy].copy())
            column = np.asarray(column_file['data'][:, :, iz:iz+size]).reshape(-1)
            dvdr = _recompute_dvdr_slab(ds, dims, iz, size)
            safe = _clip_to_table_domain(dsp, nh, column, dvdr)
            for name, original, clipped in zip(('dsp_nH_clipped', 'dsp_NH_clipped', 'dsp_dvdr_clipped'),
                                                 (nh, column, dvdr), safe):
                counts[name] += int(np.count_nonzero(original != clipped))
            td = dsp.temperature(*safe)
            mixed = np.where(tq < 3000, td, tq)
            for key, value in (('rho', rho), ('NH', column), ('T_QUOKKA', tq),
                               ('T_DESPOTIC', td), ('dvdr', dvdr)):
                if not np.isfinite(value).all() or np.any(value <= 0):
                    raise ValueError(f'Invalid {key}; not dropping simulation cells')
                lo, hi = ranges.get(key, (np.inf, -np.inf))
                ranges[key] = (min(lo, float(value.min())), max(hi, float(value.max())))
            lr, lc = np.log10(rho), np.log10(column)
            mass = rho * volume
            for key, temperature in (('mass_T_QK', tq), ('mass_T_DSP', td), ('mass_T_2R', mixed)):
                histograms[key].add(lr, np.log10(temperature), mass)
            histograms['NH_rho'].add(lc, lr, mass)
            # Small lookup batches bound the temporary eight-corner Cloudy arrays.
            for start in range(0, tq.size, 65536):
                sl = slice(start, start + 65536)
                eps = emissivities(tq[sl], td[sl], nh[sl], column[sl], dvdr[sl], dsp, cloudy)
                for key, value in eps.items():
                    temperature = td[sl] if key.startswith('co') else mixed[sl]
                    histograms[key].add(lr[sl], np.log10(temperature), value * volume)
            counts['cells'] += tq.size
            counts['low_temperature_cells'] += int(np.count_nonzero(tq < 3000))
            if iz == 0 or (iz + size) % 256 == 0 or iz + size == dims[2]:
                print(f'z={iz+size}/{dims[2]}; cells={counts["cells"]}; '
                      f'elapsed={(time.monotonic()-started)/60:.1f} min', flush=True)
            del rho, tq, nh, column, dvdr, td, mixed, safe, mass, lr, lc
            gc.collect()
        # Sample the validation rays from the same full-x/y slabs used above.
        # Tiny off-origin covering grids can map incorrectly in this yt reader.
        ray_errors = []
        for ix, iy in ray_positions:
            n_ray = np.concatenate(density_rays[ix, iy])
            minus = np.cumsum(n_ray) * width[2]
            plus = np.cumsum(n_ray[::-1])[::-1] * width[2]
            expected = 2.0 / (1.0 / minus + 1.0 / plus)
            cached = column_file['data'][ix, iy, :]
            np.testing.assert_allclose(cached, expected, rtol=1e-6)
            ray_errors.append(float(np.max(np.abs(cached / expected - 1))))
    panels = {key: histogram.result() for key, histogram in histograms.items()}
    validation = {}
    for key, histogram in histograms.items():
        binned = float(histogram.H.sum())
        np.testing.assert_allclose(binned, histogram.total, rtol=1e-10, atol=0)
        assert histogram.count == int(np.prod(dims))
        validation[key] = dict(direct_sum=histogram.total, bin_sum=binned,
                              relative_error=abs(binned-histogram.total)/histogram.total
                              if histogram.total else 0.0,
                              cells=histogram.count,
                              unit='g' if key.startswith('mass') or key == 'NH_rho' else 'erg/s')
    np.savez_compressed(bundle, **{f'{key}__{name}': value
                       for key, panel in panels.items() for name, value in panel.items()})
    metadata = dict(dataset=str(args.dataset), despotic_table=str(args.despotic_table),
                    cloudy_table=str(args.cloudy_table), column_cache=str(args.column_cache),
                    column_definition='harmonic mean of +z and -z inclusive cumulative columns',
                    dvdr='recomputed abs(div(v))/3; current simulation numerical floor',
                    temperature_split_K=3000, X_H=float(cfg.X_H), cell_volume_cm3=volume,
                    line_policy='CII: DESPOTIC low / Cloudy high; Halpha,HI: DESPOTIC densities and analytic low / Cloudy high; CO10,CO21: DESPOTIC all cells',
                    mixed_temperature='T_DESPOTIC below T_QUOKKA=3000 K; T_QUOKKA otherwise',
                    cloud_attenuation_column='clip to 1e18..1e21; interpolate within',
                    bin_dex=.2, color_dynamic_range_dex=6, velocity_selection=None,
                    counts=counts, input_ranges=ranges, validation=validation,
                    column_ray_max_relative_errors=ray_errors,
                    elapsed_minutes=(time.monotonic()-started)/60)
    report.write_text(json.dumps(metadata, indent=2) + '\n')
    plot_panels(panels, png, pdf)
    print(f'Completed: {png}\n{pdf}\nConservation: {report}', flush=True)


if __name__ == '__main__':
    main()
