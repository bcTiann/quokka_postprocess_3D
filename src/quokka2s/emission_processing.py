"""Process an accepted QUOKKA snapshot into line images and full-box spectra.

This module performs the native-resolution cell calculation once and saves
numerical products. Plotting reads those products in a separate command.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from quokka2s.cloudy_cell_queries import prepare_cloudy_cell_queries
from quokka2s.cloudy_sixline_lookup import CloudyFailureTouchError, CloudySixLineLookup
from quokka2s.dust_attenuation import (
    LINE_WAVELENGTH_MICRON, attenuate_emissivities,
    extinction_cross_sections, load_draine_extinction,
    observer_side_hydrogen_column,
)
from quokka2s.emission_config import load_process_config
from quokka2s.emission_product_accumulator import (
    LineLuminosityImageAccumulator, VARIANT_KEYS,
)
from quokka2s.tables.io import load_table
from quokka2s.tables.lookup import TableLookup

DEFAULT_CLOUDY_SHA = '4b8576adc6fc06cb0dc784e4fe41a9f7f6d08cede1c88e53e3b5455676705f31'
AXIS_NAMES = ('nH', 'NH', 'dVdr')
VELOCITY_RANGE_KMS = 200.
VELOCITY_CHANNELS = 400


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def slab_windows(nx, slab_nx):
    """Read one x-neighbour halo around each non-overlapping processed slab."""
    if nx < 2 or slab_nx < 1:
        raise ValueError('x dimension must be at least two and slab size positive')
    for ix in range(0, nx, slab_nx):
        end = min(ix + slab_nx, nx)
        lo, hi = max(0, ix - 1), min(nx, end + 1)
        yield ix, end, lo, hi, slice(ix - lo, end - lo)


def _validate_scan_provenance(table, shape, cfg, physics):
    """Check numerical snapshot provenance without comparing machine paths."""
    domain = (table.build_metadata or {}).get('snapshot_domain')
    if not domain or domain.get('selection') != 'all simulation cells':
        raise ValueError('Candidate lacks all-cell snapshot-domain provenance')
    checks = {
        'shape': list(shape), 'total_cells': int(np.prod(shape)),
        'X_H': float(cfg.X_H), 'column_mean': cfg.COLUMN_DENSITY_MEAN,
        'column_directions': cfg.COLUMN_DENSITY_DIRECTIONS,
        'physics_source_sha256': _sha256(physics.__file__),
    }
    for name, expected in checks.items():
        if domain.get(name) != expected:
            raise ValueError(f'Snapshot provenance mismatch for {name}: '
                             f'table={domain.get(name)!r}, current={expected!r}')
    axes = (table.nH_values, table.col_density_values, table.dVdr_values)
    for name, axis in zip(AXIS_NAMES, axes):
        recorded = domain['axes'][name]
        if axis[0] != recorded['minimum'] or axis[-1] != recorded['maximum']:
            raise ValueError(f'Candidate {name} bounds differ from recorded snapshot extrema')
    return domain


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


def combine_spectral_variants(variant_payloads, projected_area_cm2):
    """Stack the two saved spectra and calculate each line's full-box moments."""
    parts = [variant_payloads[name] for name in VARIANT_KEYS]
    keys = tuple(str(key) for key in parts[0]['line_keys'])
    regimes = np.asarray(parts[0]['regime_keys'])
    edges = np.asarray(parts[0]['velocity_edges_kms'], dtype=float)
    centers = np.asarray(parts[0]['velocity_kms'], dtype=float)
    for part in parts[1:]:
        if (tuple(str(key) for key in part['line_keys']) != keys
                or not np.array_equal(part['regime_keys'], regimes)
                or not np.array_equal(part['velocity_edges_kms'], edges)
                or not np.array_equal(part['cell_counts_by_regime'], parts[0]['cell_counts_by_regime'])):
            raise ValueError('Intrinsic and attenuated spectra have inconsistent axes or cells')
    spectra = np.stack([part['dL_dv_erg_s_per_kms'] for part in parts])
    total = spectra.sum(axis=2)
    weights = total * np.diff(edges)[None, None, :]
    captured = weights.sum(axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        centroid = np.divide(np.sum(weights * centers[None, None, :], axis=-1),
                             captured, out=np.full_like(captured, np.nan), where=captured > 0)
        variance = np.divide(np.sum(weights * (centers[None, None, :] - centroid[..., None])**2,
                                    axis=-1), captured,
                             out=np.full_like(captured, np.nan), where=captured > 0)
    return {
        'schema_version': np.asarray(1),
        'line_keys': np.asarray(keys), 'variant_keys': np.asarray(VARIANT_KEYS),
        'regime_keys': regimes.copy(),
        'axis_order': np.asarray('variant,line,regime,velocity_channel'),
        'velocity_edges_kms': edges.copy(), 'velocity_kms': centers.copy(),
        'dL_dv_erg_s_per_kms': spectra,
        'total_dL_dv_erg_s_per_kms': total,
        'input_luminosity_erg_s': np.stack([part['input_luminosity_erg_s'] for part in parts]),
        'captured_luminosity_erg_s': np.stack([part['captured_luminosity_erg_s'] for part in parts]),
        'outside_velocity_luminosity_erg_s': np.stack(
            [part['outside_velocity_luminosity_erg_s'] for part in parts]),
        'line_centroid_kms': centroid, 'line_sigma_kms': np.sqrt(variance),
        'cell_counts_by_regime': parts[0]['cell_counts_by_regime'].copy(),
        'projected_area_cm2': np.asarray(projected_area_cm2),
        'line_of_sight': np.asarray('z'),
    }


def _compute_with_cloudy_failure_exclusions(queries, dvdr, despotic, cloudy,
                                            compute_emission):
    """Omit hot cells that touch any unavailable Cloudy line node, then retry."""
    excluded_cloudy = np.zeros(queries.excluded.shape, dtype=bool)
    try:
        emission = compute_emission(queries, dvdr, despotic, cloudy,
                                    allow_capped_legacy_jeans=True)
    except CloudyFailureTouchError:
        hot_query = ~queries.state.cold_mask & ~queries.excluded
        diagnostic = cloudy.diagnose(
            queries.state.temperature_K[hot_query], queries.n_H_cm3[hot_query],
            queries.column_density_H_cm2[hot_query])
        excluded_cloudy[hot_query] = diagnostic.failure_touched.any(axis=0)
        if not excluded_cloudy.any():
            raise
        queries = replace(queries, excluded=queries.excluded | excluded_cloudy)
        emission = compute_emission(queries, dvdr, despotic, cloudy,
                                    allow_capped_legacy_jeans=True)
    return queries, emission, excluded_cloudy


def main(argv=None):
    parser = argparse.ArgumentParser(prog='quokka2s process', description=__doc__)
    parser.add_argument('--config', required=True, type=Path,
                        help='YAML file containing the processing input and output paths')
    config_path = parser.parse_args(argv).config
    try:
        args = load_process_config(config_path)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if min(args.slab_nx, args.query_chunk, args.spectral_workers) <= 0 or (
            args.max_slabs is not None and args.max_slabs <= 0):
        parser.error('Numerical sizes must be positive')
    if args.query_chunk > 100000:
        parser.error('Query batches must contain at most 100000 cells')
    if args.output_dir.exists():
        raise FileExistsError('Choose a new output directory')
    for name in ('dataset', 'despotic_table', 'cloudy_table', 'dust_opacity_table'):
        if not getattr(args, name).exists():
            raise FileNotFoundError(f'{name}: {getattr(args, name)}')
    inputs = (args.despotic_table, args.cloudy_table, args.dust_opacity_table)
    hashes = {str(path.resolve()): _sha256(path) for path in inputs}
    snapshot_header = args.dataset / 'Header'
    if not snapshot_header.is_file():
        raise FileNotFoundError(f'Snapshot Header: {snapshot_header}')
    snapshot_header_sha = _sha256(snapshot_header)
    import yt
    from yt.units import gravitational_constant
    from quokka2s.pipeline.prep import physics_fields as physics, config as cfg
    from quokka2s.adopted_cell_emission import compute_adopted_cell_emission, COLD_OMITTED_LINES
    from quokka2s.adopted_spectral_products import AdoptedSpectralAccumulator

    ds = yt.load(str(args.dataset.resolve()))
    shape = tuple(int(x) for x in ds.domain_dimensions)
    if ds.max_level != 0 or cfg.DOWNSAMPLE_FACTOR != 1:
        raise ValueError('Expected the complete uniform snapshot at native resolution')
    dsp = TableLookup(load_table(args.despotic_table))
    if dsp.table.build_metadata['composition']['setup'] != 'cloudy_c17_02_default_gow_default_v2':
        raise ValueError('Expected native default DESPOTIC abundances')
    domain = _validate_scan_provenance(dsp.table, shape, cfg, physics)
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
    dust_wavelengths, dust_extinction = load_draine_extinction(args.dust_opacity_table)
    dust_sigma = extinction_cross_sections(keys, dust_wavelengths, dust_extinction)
    velocity_edges = np.linspace(-VELOCITY_RANGE_KMS, VELOCITY_RANGE_KMS,
                                 VELOCITY_CHANNELS + 1)
    spectra = {
        variant: AdoptedSpectralAccumulator(keys, velocity_edges,
            constants['boltzmann_erg_K'], workers=args.spectral_workers, cell_chunk=16384)
        for variant in VARIANT_KEYS
    }
    images = LineLuminosityImageAccumulator(keys, shape[:2])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    code_files = (Path(__file__), Path(cfg.__file__), Path(physics.__file__), ROOT/'src/quokka2s/cloudy_cell_queries.py',
        ROOT/'src/quokka2s/cloudy_hot_emission.py', ROOT/'src/quokka2s/adopted_cell_emission.py',
        ROOT/'src/quokka2s/adopted_spectral_products.py', ROOT/'src/quokka2s/cloudy_sixline_lookup.py',
        ROOT/'src/quokka2s/tables/lookup.py', ROOT/'src/quokka2s/tables/model_depth.py',
        ROOT/'src/quokka2s/dust_attenuation.py',
        ROOT/'src/quokka2s/emission_product_accumulator.py')
    code_hashes = {str(p.resolve()): _sha256(p) for p in code_files}
    total_cells = int(np.prod(shape))
    input_hashes = {name: hashes[str(getattr(args, name).resolve())]
                    for name in ('despotic_table', 'cloudy_table', 'dust_opacity_table')}
    input_fingerprint = hashlib.sha256(json.dumps({
        'snapshot_header_sha256': snapshot_header_sha,
        'shape': shape,
        'input_sha256': input_hashes,
        'code_sha256': sorted(code_hashes.values()),
    }, sort_keys=True).encode()).hexdigest()
    began = time.monotonic()
    counts = dict(all=0, cold=0, hot=0, excluded=0, excluded_cold=0,
                  excluded_despotic=0, excluded_cloudy=0, retained=0)
    mass = {key: 0. for key in ('all', 'cold', 'hot', 'excluded')}
    luminosity = np.zeros((len(keys), 2))
    transmitted_luminosity = np.zeros_like(luminosity)
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
            rho_cube = np.array(grid['gas', 'density'].to('g/cm**3')[core])
            rho = rho_cube.ravel()
            foreground_column = observer_side_hydrogen_column(
                rho_cube*cfg.X_H/constants['hydrogen_mass_g'],
                float(widths[2].to('cm').value)).ravel()
            del rho_cube
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
                        raise ValueError(f'DESPOTIC {name} outside the table domain')
                td = dsp.temperature(*physics._clip_to_table_domain(dsp, *raw))
                excluded_despotic = ~np.isfinite(td) | (td <= 0)
                # Legacy energy/mu arguments are unused by the fixed-mu Jeans estimate.
                queries = prepare_cloudy_cell_queries(rho[sl], column[sl], tq[sl], np.nan, td, np.nan,
                    authorized_excluded=excluded_despotic,
                    allow_hot_missing_despotic_exclusions=True, **constants)
                queries, emission, excluded_cloudy = _compute_with_cloudy_failure_exclusions(
                    queries, dvdr[sl], dsp, cloudy, compute_adopted_cell_emission)
                excluded = queries.excluded
                cold = tq[sl] < 3000
                for key, flag in emission.despotic_clipped.items():
                    clipped[key] += int(flag.sum())
                hot = ~cold & emission.valid
                foreground_clipped['below'] += int(np.count_nonzero(hot & (column[sl] < 10**cloudy.log_NH_attenuation[0])))
                foreground_clipped['above'] += int(np.count_nonzero(hot & (column[sl] > 10**cloudy.log_NH_attenuation[-1])))
                for branch, selected in enumerate((cold & emission.valid, hot)):
                    luminosity[:, branch] += emission.emissivity_erg_s_cm3[:, selected].sum(axis=1)*volume
                supplied_emissivity = emission.emissivity_erg_s_cm3.copy()
                supplied_emissivity[:, emission.valid] = attenuate_emissivities(
                    emission.emissivity_erg_s_cm3[:, emission.valid],
                    foreground_column[sl][emission.valid], dust_sigma)
                for branch, selected in enumerate((cold & emission.valid, hot)):
                    transmitted_luminosity[:, branch] += supplied_emissivity[:, selected].sum(axis=1)*volume
                use = emission.valid
                images.add_slab_batch(x_start=ix, slab_shape=(end-ix, *shape[1:]),
                    batch_start=start, valid_mask=use,
                    intrinsic_emissivity=emission.emissivity_erg_s_cm3,
                    transmitted_emissivity=supplied_emissivity, cell_volume_cm3=volume)
                for variant, emissivity in (
                    ('intrinsic', emission.emissivity_erg_s_cm3),
                    ('attenuated', supplied_emissivity),
                ):
                    spectra[variant].add(vz[sl][use], emission.thermal_temperature_K[:, use],
                        emissivity[:, use], volume, cold_mask=cold[use])
                for name, selected in (('all', np.ones(ids.shape, dtype=bool)), ('cold', cold),
                                       ('hot', ~cold), ('excluded', excluded)):
                    counts[name] += int(selected.sum())
                    mass[name] += float(rho[sl][selected].sum()*volume)
                counts['retained'] += int(use.sum())
                counts['excluded_cold'] += int(np.count_nonzero(excluded & cold))
                counts['excluded_despotic'] += int(np.count_nonzero(excluded_despotic))
                counts['excluded_cloudy'] += int(np.count_nonzero(excluded_cloudy))
            status('running', completed_slabs=slab+1)
            print(f'Emission: {counts["all"]}/{total_cells} cells; '
                  f'{time.monotonic()-began:.1f} s', flush=True)
        full = counts['all'] == total_cells
        if (counts['cold'] + counts['hot'] != counts['all'] or
                counts['retained'] + counts['excluded'] != counts['all'] or
                counts['excluded_despotic'] + counts['excluded_cloudy'] != counts['excluded']):
            raise ValueError('Cell-selection counts are inconsistent')
        if not np.isclose(mass['cold'] + mass['hot'], mass['all'], rtol=1e-12, atol=0):
            raise ValueError('Temperature-regime masses do not sum to the processed mass')
        for mapping in (hashes, code_hashes):
            for path, digest in mapping.items():
                if _sha256(path) != digest:
                    raise ValueError(f'Input changed during run: {path}')
        if _sha256(snapshot_header) != snapshot_header_sha:
            raise ValueError(f'Snapshot Header changed during run: {snapshot_header}')
        if any(luminosity[keys.index(key), 0] != 0 for key in COLD_OMITTED_LINES):
            raise ValueError('Cold CIII/CIV must be exactly zero')
        if np.any(transmitted_luminosity > luminosity*(1+1e-12)):
            raise ValueError('Dust-transmitted luminosity exceeds intrinsic luminosity')
        variant_results = {name: spectra[name].finalize() for name in VARIANT_KEYS}
        spectral_report = {name: result[1] for name, result in variant_results.items()}
        area = float((ds.domain_width[0]*ds.domain_width[1]).to('cm**2').value)
        spectrum_payload = combine_spectral_variants(
            {name: result[0] for name, result in variant_results.items()}, area)
        image_payload = images.finalize()
        expected_luminosity = np.stack((luminosity, transmitted_luminosity))
        if not np.allclose(spectrum_payload['input_luminosity_erg_s'],
                           expected_luminosity, rtol=1e-12, atol=0):
            raise ValueError('Spectrum input differs from independent cell luminosities')
        if not np.allclose(image_payload['total_luminosity_erg_s'],
                           expected_luminosity.sum(axis=-1), rtol=1e-12, atol=0):
            raise ValueError('Image pixels do not sum to cell luminosities')
        if not np.allclose(spectrum_payload['captured_luminosity_erg_s'] +
                           spectrum_payload['outside_velocity_luminosity_erg_s'],
                           expected_luminosity, rtol=1e-12, atol=0):
            raise ValueError('Spectral window and outside luminosity do not sum to cell luminosities')
        if (not np.array_equal(image_payload['line_luminosity_image_erg_s'][0, keys.index('hi21')],
                               image_payload['line_luminosity_image_erg_s'][1, keys.index('hi21')])
                or not np.array_equal(spectrum_payload['dL_dv_erg_s_per_kms'][0, keys.index('hi21')],
                                      spectrum_payload['dL_dv_erg_s_per_kms'][1, keys.index('hi21')])):
            raise ValueError('H I 21 cm must be unchanged by the adopted dust approximation')
        for product in (image_payload, spectrum_payload):
            product.update(full_snapshot=np.asarray(full),
                           input_fingerprint_sha256=np.asarray(input_fingerprint),
                           dust_sigma_ext_cm2_H=dust_sigma.copy(),
                           dust_rest_wavelength_micron=np.asarray(
                               [LINE_WAVELENGTH_MICRON[key] for key in keys]),
                           dust_observer_side=np.asarray('outer -z boundary face'))
        left_kpc = ds.domain_left_edge.to('kpc').value
        right_kpc = ds.domain_right_edge.to('kpc').value
        image_payload.update(
            x_edges_kpc=np.linspace(left_kpc[0], right_kpc[0], shape[0]//2 + 1),
            y_edges_kpc=np.linspace(left_kpc[1], right_kpc[1], shape[1]//2 + 1),
            line_of_sight=np.asarray('z'))
        np.savez_compressed(args.output_dir/'images.npz', **image_payload)
        np.savez_compressed(args.output_dir/'spectra.npz', **spectrum_payload)
        line_sigma = {
            variant: {key: (float(value) if np.isfinite(value) else None)
                      for key, value in zip(keys, spectrum_payload['line_sigma_kms'][index])}
            for index, variant in enumerate(VARIANT_KEYS)
        }
        result = dict(status='completed' if full else 'partial diagnostic', full_snapshot=full,
            completed_at=datetime.now(timezone.utc).isoformat(), counts=counts, mass_g=mass,
            line_keys=list(keys), variants=list(VARIANT_KEYS), regimes=['cold', 'hot'],
            image_shape=list(images.image_shape), velocity_channels=VELOCITY_CHANNELS,
            intrinsic_luminosity_erg_s=luminosity.tolist(),
            transmitted_luminosity_erg_s=transmitted_luminosity.tolist(),
            line_sigma_kms=line_sigma,
            despotic_coordinate_clipped_cells=clipped, cloudy_attenuation_coordinate_clipped_cells=foreground_clipped,
            input_sha256=input_hashes, snapshot_header_sha256=snapshot_header_sha,
            input_fingerprint_sha256=input_fingerprint,
            code_sha256=code_hashes, snapshot_domain=domain, constants=constants,
            dataset=str(args.dataset.resolve()),
            excluded_mass_fraction=mass['excluded']/mass['all'],
            cell_volume_cm3=volume, spectral_report=spectral_report,
            cold_CIII_CIV='Omitted by the adopted prescription; hot contributions only',
            exclusions='Cells with unavailable required table results are omitted from every line product; omitted entries are not zero emission',
            interpretation=('Local volume emission with inherited Cloudy escape treatment and DESPOTIC LVG; '
                'one-sided -z foreground Draine extinction applied before image and spectrum accumulation; '
                'no scattered-in light.'),
            dust_attenuation=dict(model='Draine MW R_V=3.1 total extinction',
                opacity_table=str(args.dust_opacity_table.resolve()),
                opacity_sha256=input_hashes['dust_opacity_table'],
                observer='outer -z boundary face for each (x,y) sightline',
                interpolation='linear in log(wavelength), log(C_ext/H)',
                sigma_ext_cm2_H=dict(zip(keys, dust_sigma.tolist())),
                hi21='dust extinction neglected beyond the 1 cm table limit'),
            elapsed_seconds=time.monotonic()-began)
        (args.output_dir/'emission_report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        status(result['status'])
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}', failure_context=failure_context)
        raise


if __name__ == '__main__':
    main()
