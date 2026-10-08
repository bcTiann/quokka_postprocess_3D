"""Describe a completed processing run without writing files.

This report joins numerical summaries with execution and physical settings.
It does not query tables, change product arrays, or perform validation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import time

import numpy as np

from quokka2s.constants import (
    BOLTZMANN_ERG_K,
    GRAVITATIONAL_CGS,
    PARSEC_CM,
    PLANCK_ERG_S,
    SPEED_OF_LIGHT_CM_S,
    ATOMIC_MASS_UNIT_G,
)
from quokka2s.physics.gas_fields import HYDROGEN_MASS_G
from quokka2s.products import DUST_STATES
from quokka2s.products.emission_products import VELOCITY_CHANNELS


def build_processing_report(
    *, config, snapshot, emission_calculator, products, outputs, metadata,
    image_conservation, began,
) -> dict:
    """Describe the saved results, physical settings and numerical checks.

    Inputs come from the process main function; outputs is products.build_outputs().
    metadata is the ResultMetadata record already attached to outputs.
    Returns the JSON-ready contents of emission_report.json. elapsed_seconds
    runs from began to report construction, before saving; final status.json
    includes saving time. Luminosities are erg/s and dispersions km/s.
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
        'status': 'completed' if outputs.processing_complete else 'partial diagnostic',
        'full_snapshot': full,
        'processing_complete': outputs.processing_complete,
        'processing_region': metadata.processing_region,
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
