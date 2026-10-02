"""Process a QUOKKA snapshot into line images and full-box spectra.

This module performs the native-resolution cell calculation once and saves
numerical products. Plotting reads those products in a separate command.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from quokka2s.cloudy_cell_queries import prepare_cloudy_cell_queries
from quokka2s.cloudy_sixline_lookup import CloudyFailureTouchError, CloudySixLineLookup
from quokka2s.adopted_cell_emission import COLD_OMITTED_LINES, compute_adopted_cell_emission
from quokka2s.adopted_spectral_products import AdoptedSpectralAccumulator
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


def _compute_with_cloudy_failure_exclusions(queries, dvdr, despotic, cloudy):
    """Omit hot cells that touch any unavailable Cloudy line node, then retry."""
    excluded_cloudy = np.zeros(queries.excluded.shape, dtype=bool)
    try:
        emission = compute_adopted_cell_emission(
            queries, dvdr, despotic, cloudy, allow_capped_legacy_jeans=True)
    except CloudyFailureTouchError:
        hot_query = ~queries.state.cold_mask & ~queries.excluded
        diagnostic = cloudy.diagnose(
            queries.state.temperature_K[hot_query], queries.n_H_cm3[hot_query],
            queries.column_density_H_cm2[hot_query])
        excluded_cloudy[hot_query] = diagnostic.failure_touched.any(axis=0)
        if not excluded_cloudy.any():
            raise
        queries = replace(queries, excluded=queries.excluded | excluded_cloudy)
        emission = compute_adopted_cell_emission(
            queries, dvdr, despotic, cloudy, allow_capped_legacy_jeans=True)
    return queries, emission, excluded_cloudy


@dataclass(frozen=True)
class _Snapshot:
    dataset: object
    shape: tuple[int, int, int]
    cell_widths: object
    cell_volume_cm3: float
    config: object
    physics: object


@dataclass(frozen=True)
class _EmissionSources:
    despotic: TableLookup
    cloudy: CloudySixLineLookup
    constants: dict
    line_keys: tuple[str, ...]
    dust_sigma_cm2_H: np.ndarray


@dataclass(frozen=True)
class _Provenance:
    snapshot_header: Path
    snapshot_header_sha256: str
    snapshot_domain: dict
    input_sha256: dict
    code_sha256: dict
    input_fingerprint_sha256: str


@dataclass
class _Products:
    images: LineLuminosityImageAccumulator
    spectra: dict
    counts: dict
    mass_g: dict
    intrinsic_luminosity_erg_s: np.ndarray
    attenuated_luminosity_erg_s: np.ndarray
    despotic_clipped_cells: dict
    cloudy_column_clipped_cells: dict
    failure_context: dict


@dataclass(frozen=True)
class _SlabArrays:
    """One processed x slab, flattened in x-y-z order for table batches.

    Density is g/cm^3; both columns are H nuclei/cm^2; velocity gradient
    is s^-1; temperatures are K; line-of-sight velocity is km/s.
    """
    density_g_cm3: np.ndarray
    foreground_NH_cm2: np.ndarray
    temperature_QUOKKA_K: np.ndarray
    shielding_NH_cm2: np.ndarray
    velocity_gradient_s: np.ndarray
    velocity_z_kms: np.ndarray


def _load_processing_inputs(args):
    """Open the snapshot and tables and record exactly which inputs were used."""
    if args.output_dir.exists():
        raise FileExistsError('Choose a new output directory')
    for name in ('dataset', 'despotic_table', 'cloudy_table', 'dust_opacity_table'):
        if not getattr(args, name).exists():
            raise FileNotFoundError(f'{name}: {getattr(args, name)}')
    table_paths = (args.despotic_table, args.cloudy_table, args.dust_opacity_table)
    hashes = {str(path.resolve()): _sha256(path) for path in table_paths}
    snapshot_header = args.dataset / 'Header'
    if not snapshot_header.is_file():
        raise FileNotFoundError(f'Snapshot Header: {snapshot_header}')
    snapshot_header_sha = _sha256(snapshot_header)

    import yt
    from yt.units import gravitational_constant
    from quokka2s.pipeline.prep import physics_fields as physics, config as cfg

    ds = yt.load(str(args.dataset.resolve()))
    shape = tuple(int(x) for x in ds.domain_dimensions)
    if ds.max_level != 0 or cfg.DOWNSAMPLE_FACTOR != 1:
        raise ValueError('Expected the complete uniform snapshot at native resolution')
    despotic = TableLookup(load_table(args.despotic_table))
    if despotic.table.build_metadata['composition']['setup'] != 'cloudy_c17_02_default_gow_default_v2':
        raise ValueError('Expected native default DESPOTIC abundances')
    domain = _validate_scan_provenance(despotic.table, shape, cfg, physics)
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
    snapshot = _Snapshot(dataset=ds, shape=shape, cell_widths=widths,
                         cell_volume_cm3=volume, config=cfg, physics=physics)
    sources = _EmissionSources(despotic=despotic, cloudy=cloudy,
                               constants=constants, line_keys=keys,
                               dust_sigma_cm2_H=dust_sigma)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    code_files = (Path(__file__), Path(cfg.__file__), Path(physics.__file__), ROOT/'src/quokka2s/cloudy_cell_queries.py',
        ROOT/'src/quokka2s/cloudy_hot_emission.py', ROOT/'src/quokka2s/adopted_cell_emission.py',
        ROOT/'src/quokka2s/adopted_spectral_products.py', ROOT/'src/quokka2s/cloudy_sixline_lookup.py',
        ROOT/'src/quokka2s/tables/lookup.py', ROOT/'src/quokka2s/tables/model_depth.py',
        ROOT/'src/quokka2s/dust_attenuation.py',
        ROOT/'src/quokka2s/emission_product_accumulator.py')
    code_hashes = {str(path.resolve()): _sha256(path) for path in code_files}
    input_hashes = {name: hashes[str(getattr(args, name).resolve())]
                    for name in ('despotic_table', 'cloudy_table', 'dust_opacity_table')}
    fingerprint = hashlib.sha256(json.dumps({
        'snapshot_header_sha256': snapshot_header_sha,
        'shape': shape,
        'input_sha256': input_hashes,
        'code_sha256': sorted(code_hashes.values()),
    }, sort_keys=True).encode()).hexdigest()
    provenance = _Provenance(
        snapshot_header=snapshot_header,
        snapshot_header_sha256=snapshot_header_sha,
        snapshot_domain=domain,
        input_sha256=input_hashes,
        code_sha256=code_hashes,
        input_fingerprint_sha256=fingerprint,
    )
    return snapshot, sources, provenance


def _new_products(snapshot, sources, spectral_workers):
    """Keep only the accumulating maps and spectra between x slabs."""
    velocity_edges = np.linspace(-VELOCITY_RANGE_KMS, VELOCITY_RANGE_KMS,
                                 VELOCITY_CHANNELS + 1)
    spectra = {
        variant: AdoptedSpectralAccumulator(sources.line_keys, velocity_edges,
            sources.constants['boltzmann_erg_K'], workers=spectral_workers, cell_chunk=16384)
        for variant in VARIANT_KEYS
    }
    images = LineLuminosityImageAccumulator(sources.line_keys, snapshot.shape[:2])
    luminosity = np.zeros((len(sources.line_keys), 2))
    return _Products(
        images=images, spectra=spectra,
        counts=dict(all=0, cold=0, hot=0, excluded=0, excluded_cold=0,
                    excluded_despotic=0, excluded_cloudy=0, retained=0),
        mass_g={key: 0. for key in ('all', 'cold', 'hot', 'excluded')},
        intrinsic_luminosity_erg_s=luminosity,
        attenuated_luminosity_erg_s=np.zeros_like(luminosity),
        despotic_clipped_cells={key: 0 for key in ('nH', 'NH', 'dVdr')},
        cloudy_column_clipped_cells=dict(below=0, above=0),
        failure_context={},
    )


def _write_status(output_dir, counts, began, state, **extra):
    data = dict(status=state, counts=counts, elapsed_seconds=time.monotonic()-began, **extra)
    temporary = output_dir/'status.tmp'
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(output_dir/'status.json')


def _read_slab(snapshot, window, hydrogen_mass_g):
    """Read and derive an x slab with one-cell halos, retaining its core cells."""
    _, _, lo, hi, core = window
    ds, shape, widths = snapshot.dataset, snapshot.shape, snapshot.cell_widths
    edge = ds.domain_left_edge.copy()
    edge[0] += lo*widths[0]
    grid = ds.covering_grid(0, edge, (hi-lo, *shape[1:]))
    bulk = np.asarray(grid.get_field_parameter('bulk_velocity'))
    if not np.isfinite(bulk).all() or np.any(bulk != 0):
        raise ValueError('Unexpected bulk velocity')
    density_cube = np.array(grid['gas', 'density'].to('g/cm**3')[core])
    density = density_cube.ravel()
    foreground_column = observer_side_hydrogen_column(
        density_cube*snapshot.config.X_H/hydrogen_mass_g,
        float(widths[2].to('cm').value)).ravel()
    del density_cube
    temperature = np.array(grid['boxlib', 'temperature'][core], dtype=float).ravel()
    shielding_column = np.array(snapshot.physics._column_density_H(None, grid).to('cm**-2')[core]).ravel()
    velocity_gradient = np.array(snapshot.physics._dVdr_lvg(None, grid).to('s**-1')[core]).ravel()
    velocity_z = np.array(grid['gas', 'velocity_z'].to('km/s')[core]).ravel()
    del grid
    return _SlabArrays(
        density_g_cm3=density,
        foreground_NH_cm2=foreground_column,
        temperature_QUOKKA_K=temperature,
        shielding_NH_cm2=shielding_column,
        velocity_gradient_s=velocity_gradient,
        velocity_z_kms=velocity_z,
    )


def _process_batch(snapshot, sources, products, slab, x_start, x_stop, start, stop):
    """Query emissivities, apply dust, and add this batch to maps and spectra."""
    batch_slice = slice(start, stop)
    shape, cfg, physics = snapshot.shape, snapshot.config, snapshot.physics
    despotic, cloudy = sources.despotic, sources.cloudy
    volume = snapshot.cell_volume_cm3
    offset = x_start*shape[1]*shape[2]+start
    ids = np.arange(offset, offset+stop-start, dtype=np.int64)
    products.failure_context = dict(first_cell_id=int(ids[0]), last_cell_id=int(ids[-1]))
    hydrogen_density = slab.density_g_cm3[batch_slice]*cfg.X_H/sources.constants['hydrogen_mass_g']
    table_coordinates = (hydrogen_density, slab.shielding_NH_cm2[batch_slice],
                         slab.velocity_gradient_s[batch_slice])
    axes = (despotic.table.nH_values, despotic.table.col_density_values,
            despotic.table.dVdr_values)
    for name, values, axis in zip(('nH', 'NH', 'dVdr'), table_coordinates, axes):
        if (not np.isfinite(values).all() or np.any(values <= 0) or
                np.any(values < axis[0]) or np.any(values > axis[-1])):
            raise ValueError(f'DESPOTIC {name} outside the table domain')
    despotic_temperature = despotic.temperature(
        *physics._clip_to_table_domain(despotic, *table_coordinates))
    excluded_despotic = ~np.isfinite(despotic_temperature) | (despotic_temperature <= 0)
    # Legacy energy/mu arguments are unused by the fixed-mu Jeans estimate.
    queries = prepare_cloudy_cell_queries(
        slab.density_g_cm3[batch_slice], slab.shielding_NH_cm2[batch_slice],
        slab.temperature_QUOKKA_K[batch_slice], np.nan, despotic_temperature, np.nan,
        authorized_excluded=excluded_despotic,
        allow_hot_missing_despotic_exclusions=True, **sources.constants)
    queries, emission, excluded_cloudy = _compute_with_cloudy_failure_exclusions(
        queries, slab.velocity_gradient_s[batch_slice], despotic, cloudy)
    excluded = queries.excluded
    cold = slab.temperature_QUOKKA_K[batch_slice] < 3000
    for key, flag in emission.despotic_clipped.items():
        products.despotic_clipped_cells[key] += int(flag.sum())
    hot = ~cold & emission.valid
    products.cloudy_column_clipped_cells['below'] += int(np.count_nonzero(
        hot & (slab.shielding_NH_cm2[batch_slice] < 10**cloudy.log_NH_attenuation[0])))
    products.cloudy_column_clipped_cells['above'] += int(np.count_nonzero(
        hot & (slab.shielding_NH_cm2[batch_slice] > 10**cloudy.log_NH_attenuation[-1])))
    for branch, selected in enumerate((cold & emission.valid, hot)):
        products.intrinsic_luminosity_erg_s[:, branch] += (
            emission.emissivity_erg_s_cm3[:, selected].sum(axis=1)*volume)
    transmitted_emissivity = emission.emissivity_erg_s_cm3.copy()
    transmitted_emissivity[:, emission.valid] = attenuate_emissivities(
        emission.emissivity_erg_s_cm3[:, emission.valid],
        slab.foreground_NH_cm2[batch_slice][emission.valid], sources.dust_sigma_cm2_H)
    for branch, selected in enumerate((cold & emission.valid, hot)):
        products.attenuated_luminosity_erg_s[:, branch] += (
            transmitted_emissivity[:, selected].sum(axis=1)*volume)
    valid = emission.valid
    products.images.add_slab_batch(x_start=x_start, slab_shape=(x_stop-x_start, *shape[1:]),
        batch_start=start, valid_mask=valid,
        intrinsic_emissivity=emission.emissivity_erg_s_cm3,
        transmitted_emissivity=transmitted_emissivity, cell_volume_cm3=volume)
    for variant, emissivity in (
        ('intrinsic', emission.emissivity_erg_s_cm3),
        ('attenuated', transmitted_emissivity),
    ):
        products.spectra[variant].add(
            slab.velocity_z_kms[batch_slice][valid], emission.thermal_temperature_K[:, valid],
            emissivity[:, valid], volume, cold_mask=cold[valid])
    for name, selected in (('all', np.ones(ids.shape, dtype=bool)), ('cold', cold),
                           ('hot', ~cold), ('excluded', excluded)):
        products.counts[name] += int(selected.sum())
        products.mass_g[name] += float(slab.density_g_cm3[batch_slice][selected].sum()*volume)
    products.counts['retained'] += int(valid.sum())
    products.counts['excluded_cold'] += int(np.count_nonzero(excluded & cold))
    products.counts['excluded_despotic'] += int(np.count_nonzero(excluded_despotic))
    products.counts['excluded_cloudy'] += int(np.count_nonzero(excluded_cloudy))


def _process_snapshot(args, snapshot, sources, products, began):
    """Process all x slabs; only one slab's cell arrays live at a time."""
    total_cells = int(np.prod(snapshot.shape))
    for slab_number, window in enumerate(slab_windows(snapshot.shape[0], args.slab_nx)):
        if args.max_slabs is not None and slab_number >= args.max_slabs:
            break
        x_start, x_stop = window[:2]
        slab = _read_slab(snapshot, window, sources.constants['hydrogen_mass_g'])
        for start in range(0, slab.density_g_cm3.size, args.query_chunk):
            stop = min(start+args.query_chunk, slab.density_g_cm3.size)
            _process_batch(snapshot, sources, products, slab, x_start, x_stop, start, stop)
        del slab
        _write_status(args.output_dir, products.counts, began,
                      'running', completed_slabs=slab_number+1)
        print(f'Emission: {products.counts["all"]}/{total_cells} cells; '
              f'{time.monotonic()-began:.1f} s', flush=True)


def _check_unchanged_inputs(args, provenance):
    """Detect input or code changes made while a long snapshot run was active."""
    for name, expected in provenance.input_sha256.items():
        path = getattr(args, name)
        if _sha256(path) != expected:
            raise ValueError(f'Input changed during run: {path}')
    for path, expected in provenance.code_sha256.items():
        if _sha256(path) != expected:
            raise ValueError(f'Input changed during run: {path}')
    if _sha256(provenance.snapshot_header) != provenance.snapshot_header_sha256:
        raise ValueError(f'Snapshot Header changed during run: {provenance.snapshot_header}')


def _validate_and_finalize(snapshot, sources, products):
    """Check cell accounting and make final numerical image/spectrum arrays."""
    counts, mass = products.counts, products.mass_g
    keys = sources.line_keys
    full = counts['all'] == int(np.prod(snapshot.shape))
    if (counts['cold'] + counts['hot'] != counts['all'] or
            counts['retained'] + counts['excluded'] != counts['all'] or
            counts['excluded_despotic'] + counts['excluded_cloudy'] != counts['excluded']):
        raise ValueError('Cell-selection counts are inconsistent')
    if not np.isclose(mass['cold'] + mass['hot'], mass['all'], rtol=1e-12, atol=0):
        raise ValueError('Temperature-regime masses do not sum to the processed mass')
    if any(products.intrinsic_luminosity_erg_s[keys.index(key), 0] != 0
           for key in COLD_OMITTED_LINES):
        raise ValueError('Cold CIII/CIV must be exactly zero')
    if np.any(products.attenuated_luminosity_erg_s >
              products.intrinsic_luminosity_erg_s*(1+1e-12)):
        raise ValueError('Dust-transmitted luminosity exceeds intrinsic luminosity')
    variant_results = {name: products.spectra[name].finalize() for name in VARIANT_KEYS}
    spectral_report = {name: result[1] for name, result in variant_results.items()}
    area = float((snapshot.dataset.domain_width[0]*snapshot.dataset.domain_width[1]).to('cm**2').value)
    spectrum_payload = combine_spectral_variants(
        {name: result[0] for name, result in variant_results.items()}, area)
    image_payload = products.images.finalize()
    expected_luminosity = np.stack((products.intrinsic_luminosity_erg_s,
                                    products.attenuated_luminosity_erg_s))
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
    return full, image_payload, spectrum_payload, spectral_report


def _write_products_and_report(args, snapshot, sources, products, provenance,
                               full, image_payload, spectrum_payload,
                               spectral_report, began):
    """Save the two numerical products and one human-readable run report."""
    keys = sources.line_keys
    for product in (image_payload, spectrum_payload):
        product.update(full_snapshot=np.asarray(full),
                       input_fingerprint_sha256=np.asarray(provenance.input_fingerprint_sha256),
                       dust_sigma_ext_cm2_H=sources.dust_sigma_cm2_H.copy(),
                       dust_rest_wavelength_micron=np.asarray(
                           [LINE_WAVELENGTH_MICRON[key] for key in keys]),
                       dust_observer_side=np.asarray('outer -z boundary face'))
    left_kpc = snapshot.dataset.domain_left_edge.to('kpc').value
    right_kpc = snapshot.dataset.domain_right_edge.to('kpc').value
    image_payload.update(
        x_edges_kpc=np.linspace(left_kpc[0], right_kpc[0], snapshot.shape[0]//2 + 1),
        y_edges_kpc=np.linspace(left_kpc[1], right_kpc[1], snapshot.shape[1]//2 + 1),
        line_of_sight=np.asarray('z'))
    np.savez_compressed(args.output_dir/'images.npz', **image_payload)
    np.savez_compressed(args.output_dir/'spectra.npz', **spectrum_payload)
    line_sigma = {
        variant: {key: (float(value) if np.isfinite(value) else None)
                  for key, value in zip(keys, spectrum_payload['line_sigma_kms'][index])}
        for index, variant in enumerate(VARIANT_KEYS)
    }
    mass = products.mass_g
    result = dict(status='completed' if full else 'partial diagnostic', full_snapshot=full,
        completed_at=datetime.now(timezone.utc).isoformat(), counts=products.counts, mass_g=mass,
        line_keys=list(keys), variants=list(VARIANT_KEYS), regimes=['cold', 'hot'],
        image_shape=list(products.images.image_shape), velocity_channels=VELOCITY_CHANNELS,
        intrinsic_luminosity_erg_s=products.intrinsic_luminosity_erg_s.tolist(),
        transmitted_luminosity_erg_s=products.attenuated_luminosity_erg_s.tolist(),
        line_sigma_kms=line_sigma,
        despotic_coordinate_clipped_cells=products.despotic_clipped_cells,
        cloudy_attenuation_coordinate_clipped_cells=products.cloudy_column_clipped_cells,
        input_sha256=provenance.input_sha256,
        snapshot_header_sha256=provenance.snapshot_header_sha256,
        input_fingerprint_sha256=provenance.input_fingerprint_sha256,
        code_sha256=provenance.code_sha256, snapshot_domain=provenance.snapshot_domain,
        constants=sources.constants,
        dataset=str(args.dataset.resolve()),
        excluded_mass_fraction=mass['excluded']/mass['all'],
        cell_volume_cm3=snapshot.cell_volume_cm3, spectral_report=spectral_report,
        cold_CIII_CIV='Omitted by the adopted prescription; hot contributions only',
        exclusions='Cells with unavailable required table results are omitted from every line product; omitted entries are not zero emission',
        interpretation=('Local volume emission with inherited Cloudy escape treatment and DESPOTIC LVG; '
            'one-sided -z foreground Draine extinction applied before image and spectrum accumulation; '
            'no scattered-in light.'),
        dust_attenuation=dict(model='Draine MW R_V=3.1 total extinction',
            opacity_table=str(args.dust_opacity_table.resolve()),
            opacity_sha256=provenance.input_sha256['dust_opacity_table'],
            observer='outer -z boundary face for each (x,y) sightline',
            interpolation='linear in log(wavelength), log(C_ext/H)',
            sigma_ext_cm2_H=dict(zip(keys, sources.dust_sigma_cm2_H.tolist())),
            hi21='dust extinction neglected beyond the 1 cm table limit'),
        elapsed_seconds=time.monotonic()-began)
    (args.output_dir/'emission_report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    _write_status(args.output_dir, products.counts, began, result['status'])


def main(argv=None):
    parser = argparse.ArgumentParser(prog='quokka2s process', description=__doc__)
    parser.add_argument('--config', required=True, type=Path,
                        help='YAML file containing the processing input and output paths')
    config_path = parser.parse_args(argv).config
    try:
        args = load_process_config(config_path)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    snapshot, sources, provenance = _load_processing_inputs(args)
    products = _new_products(snapshot, sources, args.spectral_workers)
    began = time.monotonic()
    _write_status(args.output_dir, products.counts, began, 'running')
    try:
        _process_snapshot(args, snapshot, sources, products, began)
        _check_unchanged_inputs(args, provenance)
        full, image_payload, spectrum_payload, spectral_report = _validate_and_finalize(
            snapshot, sources, products)
        _write_products_and_report(args, snapshot, sources, products, provenance,
                                   full, image_payload, spectrum_payload,
                                   spectral_report, began)
    except BaseException as exc:
        _write_status(args.output_dir, products.counts, began, 'failed',
                      error=f'{type(exc).__name__}: {exc}',
                      failure_context=products.failure_context)
        raise


if __name__ == '__main__':
    main()
