"""Export the adopted Python table assets for the independent Julia processor.

This reads existing tables only. No chemistry or radiation solver is invoked.
The numerical NPZ contains no Python objects; JSON carries names and provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from quokka2s import constants
from quokka2s.cloudy.lookup import (
    COORDINATE_LOG_BOUNDARY_RTOL,
    TOUCH_EPS,
    CloudyLookup,
)
from quokka2s.despotic.cell_fields import COORDINATE_BOUNDARY_RTOL
from quokka2s.input_paths import (
    DEFAULT_CLOUDY_TABLE,
    DEFAULT_DESPOTIC_TABLE,
    DEFAULT_DRAINE_TABLE,
)
from quokka2s.line_definitions import LINE_DEFINITIONS
from quokka2s.physics import settings
from quokka2s.physics.composition import abundance_metadata
from quokka2s.physics.gas_fields import HYDROGEN_MASS_G
from quokka2s.physics.hydrogen_emissivity import (
    HALPHA_PHOTON_ENERGY_ERG,
    HI21_FREQUENCY_HZ,
    HI21_SPONTANEOUS_RATE_S,
)
from quokka2s.processing_inputs import (
    load_despotic_lookup,
    prepare_line_dust_cross_sections,
)
from quokka2s.products.emission_products import VELOCITY_CHANNELS, VELOCITY_RANGE_KMS
from quokka2s.products.gas_phase_velocity import PHASE_BOUNDS_K, PHASE_ORDER
from quokka2s.products.line_velocity_moments import MINIMUM_GAUSSIAN_WIDTH_KMS


def sha256(path: Path) -> str:
    """Hash an input without retaining its bytes in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_assets(*, output: Path, despotic_path: Path, cloudy_path: Path, dust_path: Path) -> dict:
    """Write a new asset directory, retaining the exact adopted table values."""
    source_paths = {
        "despotic": despotic_path.resolve(strict=True),
        "cloudy": cloudy_path.resolve(strict=True),
        "dust": dust_path.resolve(strict=True),
    }
    source_hashes = {name: sha256(path) for name, path in source_paths.items()}
    despotic = load_despotic_lookup(source_paths["despotic"]).table
    cloudy = CloudyLookup(path=source_paths["cloudy"])
    line_keys = (*cloudy.line_keys, "co10", "co21")
    if len(line_keys) != 10 or set(line_keys) != set(LINE_DEFINITIONS):
        raise ValueError("Expected the adopted eight atomic lines and two CO lines")
    dust = prepare_line_dust_cross_sections(source_paths["dust"], line_keys)

    # Last dimension identifies a field. The first three dimensions are table
    # coordinates, not cell coordinates; NPZ preserves their named axis order.
    despotic_field_keys = [
        "temperature_K",
        "co10_luminosity_per_H",
        "co21_luminosity_per_H",
        "cii_luminosity_per_H",
        "electron_abundance",
        "ionized_hydrogen_abundance",
        "neutral_hydrogen_abundance",
    ]
    despotic_values = np.stack(
        arrays=(
            despotic.tg_final,
            despotic.species_data["CO"].require_line().lumPerH,
            despotic.species_data["CO21"].require_line().lumPerH,
            despotic.species_data["C+"].require_line().lumPerH,
            despotic.species_data["e-"].abundance,
            despotic.species_data["H+"].abundance,
            despotic.species_data["H"].abundance,
        ),
        axis=-1,
    )
    numerical = {
        "despotic_nH_cm3": despotic.nH_values,
        "despotic_NH_cm2": despotic.col_density_values,
        "despotic_dVdr_s": despotic.dVdr_values,
        "despotic_values": despotic_values,
        "cloudy_log_NH_cm2": cloudy.log_NH_attenuation,
        "cloudy_log_nH_cm3": cloudy.log_nH,
        "cloudy_log_T_K": cloudy.log_T,
        "cloudy_emissivity_per_nH2": np.moveaxis(cloudy.emissivity_per_nH2, 0, -1),
        "cloudy_log_emissivity_per_nH2": np.moveaxis(cloudy.log_emissivity_per_nH2, 0, -1),
        "cloudy_failure_mask": np.moveaxis(cloudy.failed_table_nodes, 0, -1),
        "cloudy_zero_mask": np.moveaxis(cloudy.zero_emission_table_nodes, 0, -1),
    }
    metadata = {
        "schema_version": 1,
        "line_keys": list(line_keys),
        "masses_g": [LINE_DEFINITIONS[key].emitter_mass_amu * constants.ATOMIC_MASS_UNIT_G for key in line_keys],
        "rest_wavelength_micron": [LINE_DEFINITIONS[key].rest_wavelength_micron for key in line_keys],
        "dust_cross_sections_cm2_H": [dust[key] for key in line_keys],
        "phase_keys": list(PHASE_ORDER),
        "phase_boundaries_K": list(PHASE_BOUNDS_K),
        "emission_temperature_boundary_K": settings.EMISSION_TEMPERATURE_BOUNDARY_K,
        "minimum_gaussian_width_kms": MINIMUM_GAUSSIAN_WIDTH_KMS,
        "velocity_edges_kms": np.linspace(-VELOCITY_RANGE_KMS, VELOCITY_RANGE_KMS, VELOCITY_CHANNELS + 1).tolist(),
        "constants": {
            "hydrogen_mass_g": HYDROGEN_MASS_G,
            "boltzmann_erg_K": constants.BOLTZMANN_ERG_K,
            "planck_erg_s": constants.PLANCK_ERG_S,
            "speed_of_light_cm_s": constants.SPEED_OF_LIGHT_CM_S,
            "speed_of_light_kms": constants.SPEED_OF_LIGHT_KMS,
            "atomic_mass_unit_g": constants.ATOMIC_MASS_UNIT_G,
            "parsec_cm": constants.PARSEC_CM,
            "X_H": settings.X_H,
            "dvdr_floor_s": settings.SIMULATION_DVDR_MIN_S,
        },
        "analytic_hydrogen": {
            "halpha_photon_energy_erg": HALPHA_PHOTON_ENERGY_ERG,
            "hi21_frequency_Hz": HI21_FREQUENCY_HZ,
            "hi21_spontaneous_rate_s": HI21_SPONTANEOUS_RATE_S,
        },
        "interpolation": {
            "despotic_boundary_rtol": COORDINATE_BOUNDARY_RTOL,
            "cloudy_log_boundary_rtol": COORDINATE_LOG_BOUNDARY_RTOL,
            "cloudy_touch_eps": TOUCH_EPS,
        },
        "table_axes": {
            "despotic": ["nH_cm3", "NH_cm2", "dVdr_s", "field"],
            "cloudy": ["log10_NH_cm2", "log10_nH_cm3", "log10_T_K", "line"],
        },
        "despotic_field_keys": despotic_field_keys,
        "composition": abundance_metadata(),
        "despotic_build_metadata": dict(despotic.build_metadata or {}),
        "sources": {
            name: {"path": str(path), "sha256": source_hashes[name]}
            for name, path in source_paths.items()
        },
    }
    for name, path in source_paths.items():
        if sha256(path) != source_hashes[name]:
            raise RuntimeError(f"Input changed while exporting: {path}")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "tables.npz", **numerical)
    metadata["numerical_assets_sha256"] = sha256(output / "tables.npz")
    (output / "physics.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--despotic", type=Path, default=DEFAULT_DESPOTIC_TABLE)
    parser.add_argument("--cloudy", type=Path, default=DEFAULT_CLOUDY_TABLE)
    parser.add_argument("--dust", type=Path, default=DEFAULT_DRAINE_TABLE)
    arguments = parser.parse_args()
    metadata = export_assets(
        output=arguments.output.resolve(),
        despotic_path=arguments.despotic,
        cloudy_path=arguments.cloudy,
        dust_path=arguments.dust,
    )
    print(json.dumps({"output": str(arguments.output.resolve()), "line_keys": metadata["line_keys"], "sources": metadata["sources"]}, indent=2))


if __name__ == "__main__":
    main()
