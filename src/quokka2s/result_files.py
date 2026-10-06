"""Save image, spectrum, gas-phase arrays, and their processing report."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import time

import numpy as np

from quokka2s.constants import (
    HYDROGEN_MASS_G, BOLTZMANN_ERG_K, GRAVITATIONAL_CGS, PARSEC_CM,
    PLANCK_ERG_S, SPEED_OF_LIGHT_CM_S, ATOMIC_MASS_UNIT_G,
)
from quokka2s.physics.dust_attenuation import LINE_WAVELENGTH_MICRON
from quokka2s.products import DUST_STATES
from quokka2s.products.emission_products import VELOCITY_CHANNELS
from quokka2s.physics.cell_emission import CellEmissionCalculator
from quokka2s.snapshot_reader import Snapshot

def write_status(output_dir, counts, began, state, **extra):
    """Write progress to output_dir/status.json; this is not a checkpoint.

    counts is products.counts; began is the monotonic run start [s].
    state is "running", "failed" or "completed"; extra holds progress/error details.
    Example: write_status(config.output_dir, products.counts, began, "running").
    """
    data = {
        'status': state,
        'counts': counts,
        'elapsed_seconds': time.monotonic() - began,
        **extra,
    }
    temporary = output_dir / 'status.tmp'
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(output_dir / 'status.json')


def write_products_and_report(
    config, snapshot, emission_calculator, products, outputs, image_conservation, began,
):
    """Save the three completed NPZ products and JSON report/status.

    config supplies output_dir. snapshot supplies geometry; emission_calculator
    supplies fixed line/dust settings. products holds the independently accumulated
    totals; outputs is EmissionOutputs from products.build_outputs().
    image_conservation is returned by products.check_outputs(outputs);
    began is the monotonic run start [s]. No cell calculations occur here.

    Returns None. Writes images.npz, spectra.npz, phase_velocity.npz,
    emission_report.json and status.json; adds metadata to output payloads in place.
    Example: call after build_outputs() and check_outputs() have succeeded.
    """
    add_output_metadata(
        snapshot=snapshot,
        emission_calculator=emission_calculator,
        outputs=outputs,
    )
    save_product_arrays(output_dir=config.output_dir, outputs=outputs)
    report = build_processing_report(
        config=config,
        snapshot=snapshot,
        emission_calculator=emission_calculator,
        products=products,
        outputs=outputs,
        image_conservation=image_conservation,
        began=began,
    )
    report_path = config.output_dir / 'emission_report.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    write_status(
        output_dir=config.output_dir,
        counts=products.counts,
        began=began,
        state=report['status'],
    )


def add_output_metadata(
    snapshot: Snapshot,
    emission_calculator: CellEmissionCalculator,
    outputs,
):
    """Add geometry and dust metadata to the generated product dictionaries.

    outputs comes from products.build_outputs(). Updates its payloads in place;
    no numerical field values change. Image edges are saved in kpc.
    """
    for product in (outputs.image_payload, outputs.spectrum_payload, outputs.phase_payload):
        product['full_snapshot'] = np.asarray(outputs.full_snapshot)
    add_dust_metadata(
        emission_calculator=emission_calculator,
        image_payload=outputs.image_payload,
        spectrum_payload=outputs.spectrum_payload,
    )
    add_image_geometry(
        snapshot=snapshot,
        image_payload=outputs.image_payload,
    )


def add_dust_metadata(
    emission_calculator: CellEmissionCalculator, image_payload, spectrum_payload,
):
    """Record the adopted foreground cross-sections without changing light arrays.

    emission_calculator is shared by all batches; the two payload dicts come
    from products.build_outputs(). No cell calculations run here.
    Adds wavelengths (10,) [micron], cross-sections (10,) [cm^2/H], and the
    observer position to both products, in emission_calculator.line_keys order.
    """
    wavelength_micron = np.asarray([
        LINE_WAVELENGTH_MICRON[key] for key in emission_calculator.line_keys
    ])
    cross_sections_cm2_H = np.asarray([
        emission_calculator.dust_cross_section_cm2_H[key] for key in emission_calculator.line_keys
    ])
    for product in (image_payload, spectrum_payload):
        product['dust_sigma_ext_cm2_H'] = cross_sections_cm2_H.copy()
        product['dust_rest_wavelength_micron'] = wavelength_micron.copy()
        product['dust_observer_side'] = np.asarray('outer -z boundary face')


def add_image_geometry(snapshot: Snapshot, image_payload):
    """Add native x/y edges in kpc to the image dict from products.build_outputs().

    snapshot comes from open_snapshot(). For (256, 256, 2048), the saved x/y
    edge arrays each have shape (257,); line_of_sight is the scalar string z.
    """
    left_kpc = snapshot.dataset.domain_left_edge.to('kpc').value
    right_kpc = snapshot.dataset.domain_right_edge.to('kpc').value
    image_payload.update(
        x_edges_kpc=np.linspace(left_kpc[0], right_kpc[0], snapshot.shape[0] + 1),
        y_edges_kpc=np.linspace(left_kpc[1], right_kpc[1], snapshot.shape[1] + 1),
        line_of_sight=np.asarray('z'),
    )


def save_product_arrays(output_dir, outputs):
    """Save EmissionOutputs from products.build_outputs() as three compressed NPZs.

    output_dir is config.output_dir, an existing pathlib.Path. Returns None.
    Files are images.npz, spectra.npz and phase_velocity.npz.
    """
    np.savez_compressed(output_dir / 'images.npz', **outputs.image_payload)
    np.savez_compressed(output_dir / 'spectra.npz', **outputs.spectrum_payload)
    np.savez_compressed(output_dir / 'phase_velocity.npz', **outputs.phase_payload)


def build_processing_report(
    config, snapshot, emission_calculator, products, outputs, image_conservation, began,
) -> dict:
    """Describe the saved results, physical settings and numerical checks.

    Inputs come from the process main function; outputs is products.build_outputs().
    Returns the JSON-ready contents of emission_report.json. Elapsed time uses
    began from time.monotonic(); luminosities are erg/s and dispersions km/s.
    """
    from quokka2s.physics import settings as cfg

    full = outputs.full_snapshot
    spectrum_payload = outputs.spectrum_payload
    spectral_report = outputs.spectral_report
    phase_report = outputs.phase_report
    keys = emission_calculator.line_keys
    line_sigma_window = describe_line_values(
        line_keys=keys,
        values_kms=spectrum_payload['line_sigma_window_kms'],
    )
    line_sigma_full = describe_line_values(
        line_keys=keys,
        values_kms=spectrum_payload['line_sigma_full_kms'],
    )
    line_centroid_window = describe_line_values(
        line_keys=keys,
        values_kms=spectrum_payload['line_centroid_window_kms'],
    )
    line_centroid_full = describe_line_values(
        line_keys=keys,
        values_kms=spectrum_payload['line_centroid_full_kms'],
    )

    mass = products.mass_g
    result = {
        # Run status and the saved array organization.
        'status': 'completed' if full else 'partial diagnostic',
        'full_snapshot': full,
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'counts': products.counts,
        'mass_g': mass,
        'line_keys': list(keys),
        'dust_states': list(DUST_STATES),
        'regimes': ['cold', 'hot'],
        'image_shape': list(products.images.native_xy_shape),
        'velocity_channels': VELOCITY_CHANNELS,
        'execution': {
            'slab_nx': config.slab_nx,
            'query_chunk': config.query_chunk,
            'chunk_workers': config.chunk_workers,
            'spectral_workers_per_chunk': config.spectral_workers,
        },
        # Integrated luminosities, dispersions and query clipping counts.
        'intrinsic_luminosity_erg_s': products.intrinsic_luminosity_erg_s.tolist(),
        'transmitted_luminosity_erg_s': products.attenuated_luminosity_erg_s.tolist(),
        'line_sigma_window_kms': line_sigma_window,
        'line_sigma_full_kms': line_sigma_full,
        'line_centroid_window_kms': line_centroid_window,
        'line_centroid_full_kms': line_centroid_full,
        'despotic_coordinate_clipped_cells': products.despotic_coordinate_clipped_cells,
        'cloudy_attenuation_coordinate_clipped_cells': products.cloudy_column_clipped_cells,
        'input_tables': {
            'despotic': str(config.despotic_table.resolve()),
            'cloudy': str(config.cloudy_table.resolve()),
            'dust': str(config.dust_opacity_table.resolve()),
        },
        'snapshot_grid': {
            'shape': list(snapshot.shape),
            'cell_width_cm': snapshot.cell_widths.to('cm').value.tolist(),
        },
        'physical_settings': {
            'X_H': float(cfg.X_H),
            'column_mean': cfg.COLUMN_DENSITY_MEAN,
            'column_directions': cfg.COLUMN_DENSITY_DIRECTIONS,
            'velocity_gradient_boundary_conditions': {
                'x': 'periodic central difference',
                'y': 'periodic central difference',
                'z': 'central inside; first-order one-sided at faces',
            },
        },
        'constants': {
            'hydrogen_mass_g': HYDROGEN_MASS_G,
            'boltzmann_erg_K': BOLTZMANN_ERG_K,
            'gravitational_cm3_g_s2': GRAVITATIONAL_CGS,
            'parsec_cm': PARSEC_CM,
            'planck_erg_s': PLANCK_ERG_S,
            'speed_of_light_cm_s': SPEED_OF_LIGHT_CM_S,
            'atomic_mass_unit_g': ATOMIC_MASS_UNIT_G,
        },
        # Per-line availability and independent product-consistency checks.
        'dataset': str(config.dataset.resolve()),
        'missing_emissivity_cells': products.missing_emissivity_cells,
        'missing_emissivity_mass_fraction': {
            key: value / mass['all']
            for key, value in products.missing_emissivity_mass_g.items()
        },
        'gas_temperature_missing_mass_fraction': mass['gas_temperature_missing'] / mass['all'],
        'cell_volume_cm3': snapshot.cell_volume_cm3,
        'spectral_report': spectral_report,
        'phase_velocity_report': phase_report,
        'image_luminosity_conservation': image_conservation,
        'cold_CIII_CIV': 'Omitted by the adopted prescription; hot contributions only',
        'exclusions': (
            'Each line omits only its own missing emissivities in both images and spectra. '
            'Hot atomic lines do not depend on DESPOTIC; both CO lines require DESPOTIC '
            'at all temperatures. Missing entries are distinct from physical zero emission.'
        ),
        'interpretation': (
            'Local volume emission with inherited Cloudy escape treatment and DESPOTIC LVG; '
            'one-sided -z foreground Draine extinction applied before image and spectrum accumulation; '
            'no scattered-in light.'
        ),
        'dust_attenuation': {
            'model': 'Draine MW R_V=3.1 total extinction',
            'opacity_table': str(config.dust_opacity_table.resolve()),
            'observer': 'outer -z boundary face for each (x,y) sightline',
            'interpolation': 'linear in log(wavelength), log(C_ext/H)',
            'sigma_ext_cm2_H': {
                key: float(emission_calculator.dust_cross_section_cm2_H[key]) for key in keys
            },
            'hi21': 'dust extinction neglected beyond the 1 cm table limit',
        },
        'elapsed_seconds': time.monotonic() - began,
    }
    return result


def describe_line_values(line_keys, values_kms) -> dict:
    """Name the saved (2, Nline) centroids or dispersions for the JSON report.

    line_keys and values_kms come from the spectrum payload. Rows are intrinsic
    then attenuated; a NaN value for a zero-light line becomes None. For example,
    supplying line_sigma_full_kms makes result['attenuated']['halpha'] its
    dispersion over the complete Gaussian emission, in km/s.
    """
    line_values = {}
    for dust_index, dust_state in enumerate(DUST_STATES):
        line_values[dust_state] = {}
        for key, value in zip(line_keys, values_kms[dust_index]):
            line_values[dust_state][key] = float(value) if np.isfinite(value) else None
    return line_values
