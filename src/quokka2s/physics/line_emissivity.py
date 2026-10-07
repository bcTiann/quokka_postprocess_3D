"""Convert already queried per-species fields into volume emissivity.

These functions do not read tables. DESPOTIC luminosities use its clipped
query nH; Cloudy's emissivity_per_nH2 uses the physical cell nH squared.
"""
from __future__ import annotations

import numpy as np

from quokka2s.physics.hydrogen_emissivity import halpha_emissivity, hi21_emissivity

HYDROGEN_CII_LINE_KEYS = ('cii', 'halpha', 'hi21')
CIII_CIV_LINE_KEYS = ('ciii_977', 'ciii_1907', 'ciii_1909', 'civ_1548', 'civ_1551')
ATOMIC_LINE_KEYS = HYDROGEN_CII_LINE_KEYS + CIII_CIV_LINE_KEYS
CO_LINE_KEYS = ('co10', 'co21')


def check_field_values(name, values, *, positive=False):
    """Check a selected field array; return None and do not change its values.

    Zero emissivity or abundance is physical. Temperature must be positive.
    NaN/inf or negative required values raise instead of becoming zero light.
    """
    invalid_sign = values <= 0 if positive else values < 0
    if not np.isfinite(values).all() or np.any(invalid_sign):
        qualifier = 'positive' if positive else 'nonnegative'
        raise ValueError(f'{name} must be finite and {qualifier}')


def calculate_co_emissivities(despotic_fields, selected_cells):
    """Convert the two CO luminosities per H nucleus to volume emissivity.

    Parameters
    ----------
    despotic_fields : DespoticCellFields
        From DespoticCellReader.read_fields(); CO luminosities [erg/s/H] and clipped
        query_hydrogen_density_cm3 [cm^-3] each have shape (B,).
    selected_cells : ndarray of bool, shape (B,)
        Usable DESPOTIC-temperature queries. Both QUOKKA temperature regimes
        use DESPOTIC CO, independently of Cloudy availability.

    Returns
    -------
    tuple of two ndarray, each shape (R,)
        CO10 and CO21 epsilon [erg/s/cm^3], in selected-cell order; R is the
        number of True entries. Each epsilon is query_nH * luminosity_per_H.
    """
    query_hydrogen_density_cm3 = despotic_fields.query_hydrogen_density_cm3[selected_cells]
    co10_luminosity_per_H = despotic_fields.co10_luminosity_per_H[selected_cells]
    co21_luminosity_per_H = despotic_fields.co21_luminosity_per_H[selected_cells]
    check_field_values('DESPOTIC CO lumPerH', co10_luminosity_per_H)
    check_field_values('DESPOTIC CO21 lumPerH', co21_luminosity_per_H)
    co10_emissivity = query_hydrogen_density_cm3 * co10_luminosity_per_H
    co21_emissivity = query_hydrogen_density_cm3 * co21_luminosity_per_H
    return co10_emissivity, co21_emissivity


def calculate_cold_cii_emissivity(despotic_fields, selected_cells):
    """Return cold CII epsilon [erg/s/cm^3], shape (R_cold,).

    despotic_fields is DespoticCellFields from DespoticCellReader.read_fields(); its
    CII luminosity_per_H [erg/s/H] and query nH [cm^-3] have shape (B,).
    selected_cells is the (B,) bool mask selecting cold cells with usable T_D. The result
    multiplies these two fields at the selected original cell positions.
    """
    query_hydrogen_density_cm3 = despotic_fields.query_hydrogen_density_cm3[selected_cells]
    cii_luminosity_per_H = despotic_fields.cii_luminosity_per_H[selected_cells]
    check_field_values('DESPOTIC C+ lumPerH', cii_luminosity_per_H)
    return query_hydrogen_density_cm3 * cii_luminosity_per_H


def calculate_cold_halpha_emissivity(despotic_fields, selected_cells):
    """Return analytic Halpha epsilon [erg/s/cm^3], shape (Ncold,).

    despotic_fields is DespoticCellFields from DespoticCellReader.read_fields(). T_D [K],
    electron and ionized-H densities [cm^-3] each have shape (B,).
    selected_cells is the (B,) bool mask selecting cold cells with usable T_D; the result
    contains only those cells. No interpolation happens here.
    """
    electron_density_cm3 = despotic_fields.electron_density_cm3[selected_cells]
    ionized_hydrogen_density_cm3 = despotic_fields.ionized_hydrogen_density_cm3[selected_cells]
    check_field_values('DESPOTIC e- number density', electron_density_cm3)
    check_field_values('DESPOTIC H+ number density', ionized_hydrogen_density_cm3)
    return halpha_emissivity(
        temperature_K=despotic_fields.temperature_K[selected_cells],
        electron_density_cm3=electron_density_cm3,
        proton_density_cm3=ionized_hydrogen_density_cm3,
    )


def calculate_cold_hi21_emissivity(despotic_fields, selected_cells):
    """Return analytic HI epsilon [erg/s/cm^3], shape (Ncold,).

    despotic_fields is DespoticCellFields from DespoticCellReader.read_fields(); its
    neutral_hydrogen_density_cm3 has shape (B,) [cm^-3]. selected_cells is
    the (B,) bool mask selecting cold cells with usable T_D. The result retains only those
    cells, in their original order.
    """
    neutral_hydrogen_density_cm3 = despotic_fields.neutral_hydrogen_density_cm3[selected_cells]
    check_field_values('DESPOTIC H number density', neutral_hydrogen_density_cm3)
    return hi21_emissivity(
        neutral_hydrogen_density_cm3=neutral_hydrogen_density_cm3,
    )


def calculate_hot_atomic_emissivities(
    *,
    cloudy_fields,
    hydrogen_density_cm3,
    selected_cells,
    line_keys,
):
    """Return hot atomic epsilon by line name [erg/s/cm^3], each (Nhot,).

    cloudy_fields is CloudyCellFields from CloudyCellReader.read_fields(); its
    emissivity_per_nH2 maps each line name to a (B,) array [erg cm^3/s]. nH is
    the physical (B,) density [cm^-3], before any table clipping.
    selected_cells is the (B,) bool mask selecting successful hot Cloudy queries;
    DESPOTIC availability does not affect this selection.
    line_keys names the lines in the current group, e.g. ('cii', 'halpha', 'hi21').
    Example: result['halpha'][0] describes the first selected hot cell.
    """
    physical_hydrogen_density_cm3 = hydrogen_density_cm3[selected_cells]
    hydrogen_density_squared = physical_hydrogen_density_cm3 ** 2
    emissivity = {}
    for line_key in line_keys:
        emissivity_per_nH2 = cloudy_fields.emissivity_per_nH2[line_key]
        line_emissivity = emissivity_per_nH2[selected_cells] * hydrogen_density_squared
        check_field_values('Cloudy volume emissivity', line_emissivity)
        emissivity[line_key] = line_emissivity
    return emissivity
