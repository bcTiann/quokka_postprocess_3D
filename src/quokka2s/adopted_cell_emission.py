"""Explicit adopted line branches for already validated simulation queries.

This opt-in adapter keeps the DESPOTIC query-coordinate clipping and density
normalization of the existing pipeline. It records clipping, never replaces
invalid table values with zeros, and does not select or adopt input tables.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cloudy_cell_queries import CloudyCellQueries
from .cloudy_hot_emission import sample_cloudy_hot_emission
from .cloudy_sixline_lookup import CloudySixLineLookup


ATOMIC_LINE_KEYS = (
    'cii', 'halpha', 'hi21', 'ciii_977', 'ciii_1907', 'ciii_1909',
    'civ_1548', 'civ_1551',
)
COLD_OMITTED_LINES = ('ciii_977', 'ciii_1907', 'ciii_1909', 'civ_1548', 'civ_1551')


@dataclass(frozen=True)
class AdoptedCellEmission:
    """Line-first volume emissivities and thermal temperatures.

    ``valid`` and ``excluded`` retain the approved cell-selection mask.
    Excluded cells have NaN in both line arrays. Cold C III/C IV entries
    are explicitly zero because those branches are omitted by the adopted
    model, not because a calculation failed. ``despotic_clipped`` contains
    full cell-shaped nH/NH/dVdr flags for the queries actually evaluated.
    """

    line_keys: tuple[str, ...]
    emissivity_erg_s_cm3: np.ndarray
    thermal_temperature_K: np.ndarray
    valid: np.ndarray
    excluded: np.ndarray
    despotic_clipped: dict[str, np.ndarray]


def _checked_values(name, value, shape, *, positive=False):
    value = np.asarray(value, dtype=float)
    if value.shape != shape:
        raise ValueError(f'{name} shape {value.shape} differs from query shape {shape}')
    invalid_sign = value <= 0 if positive else value < 0
    if not np.isfinite(value).all() or np.any(invalid_sign):
        qualifier = 'positive' if positive else 'nonnegative'
        raise ValueError(f'{name} must be finite and {qualifier}')
    return value


def compute_adopted_cell_emission(
    queries: CloudyCellQueries, dvdr_s, despotic_lookup,
    checked_cloudy: CloudySixLineLookup, *, allow_capped_legacy_jeans: bool = False,
) -> AdoptedCellEmission:
    """Evaluate the agreed cold/hot atomic branches and DESPOTIC CO.

    Hot atomic lines use the checked four-dimensional Cloudy table, or an
    explicitly permitted legacy Jeans table whose contributing nodes and
    cell model depths all reach the 100 pc cap. Cold
    C II uses DESPOTIC; cold hydrogen lines use DESPOTIC number densities
    with the existing analytic formulas. C III/C IV are omitted below the
    QUOKKA 3000 K split. Both CO lines use DESPOTIC emissivity and thermal
    temperature for every retained cell. No extra exclusions are inferred.

    As in the existing pipeline, DESPOTIC per-H values are multiplied by
    the clipped query nH. All clipping is exposed to the caller, while
    ``queries`` and the supplied dVdr array remain unchanged.
    """
    atomic_keys = tuple(checked_cloudy.line_keys)
    if len(atomic_keys) != len(ATOMIC_LINE_KEYS) or set(atomic_keys) != set(ATOMIC_LINE_KEYS):
        raise ValueError('Adopted emission requires the eight supported Cloudy atomic lines')
    shape = queries.n_H_cm3.shape
    valid = ~queries.excluded
    if np.any(valid & ~queries.state.valid):
        raise ValueError('Unexcluded invalid cell state cannot produce adopted emission')
    dvdr = np.broadcast_to(np.asarray(dvdr_s, dtype=float), shape)
    _checked_values('Retained cell dVdr', dvdr[valid], (int(valid.sum()),), positive=True)
    hot_emission = sample_cloudy_hot_emission(
        queries, checked_cloudy, allow_capped_legacy_jeans=allow_capped_legacy_jeans,
    )
    keys = atomic_keys + ('co10', 'co21')
    epsilon = np.full((len(keys), *shape), np.nan)
    temperature = np.full_like(epsilon, np.nan)
    epsilon[:len(atomic_keys)] = hot_emission.emissivity_erg_s_cm3
    clips = {name: np.zeros(shape, dtype=bool) for name in ('nH', 'NH', 'dVdr')}

    if valid.any():
        from .pipeline.prep.physics_fields import (
            _clip_to_table_domain, _HI_emissivity_from_number_density,
            effective_halpha_recombination_coefficient, h, c, lambda_Halpha,
        )

        original = (queries.n_H_cm3[valid], queries.column_density_H_cm2[valid], dvdr[valid])
        safe = tuple(np.asarray(value, dtype=float) for value in
                     _clip_to_table_domain(despotic_lookup, *original))
        query_shape = original[0].shape
        for name, before, after in zip(clips, original, safe):
            _checked_values(f'DESPOTIC {name} query', after, query_shape, positive=True)
            clips[name][valid] = before != after
        td = _checked_values('DESPOTIC temperature', despotic_lookup.temperature(*safe),
                             query_shape, positive=True)
        local_cold = queries.state.cold_mask[valid]
        cold = valid & queries.state.cold_mask
        if not np.allclose(td[local_cold], queries.state.temperature_K[cold],
                           rtol=1e-12, atol=0):
            raise ValueError('Cold DESPOTIC temperature differs from the accepted paired state')
        temperature[:len(atomic_keys), valid] = queries.state.temperature_K[valid]
        temperature[len(atomic_keys):, valid] = td

        for offset, species in enumerate(('CO', 'CO21'), start=len(atomic_keys)):
            per_h = _checked_values(f'DESPOTIC {species} lumPerH',
                                   despotic_lookup.line_field(species, 'lumPerH', *safe), query_shape)
            epsilon[offset, valid] = safe[0] * per_h

        if local_cold.any():
            cold_safe = tuple(value[local_cold] for value in safe)
            cold_shape = cold_safe[0].shape
            cii_per_h = _checked_values('DESPOTIC C+ lumPerH',
                despotic_lookup.line_field('C+', 'lumPerH', *cold_safe), cold_shape)
            epsilon[keys.index('cii'), cold] = cold_safe[0] * cii_per_h
            number = despotic_lookup.number_densities(('e-', 'H+', 'H'), *cold_safe)
            densities = {name: _checked_values(f'DESPOTIC {name} number density', number[name], cold_shape)
                         for name in ('e-', 'H+', 'H')}
            photon = float(((h*c)/lambda_Halpha).in_cgs().value)
            epsilon[keys.index('halpha'), cold] = (
                photon * effective_halpha_recombination_coefficient(td[local_cold])
                * densities['e-'] * densities['H+'])
            epsilon[keys.index('hi21'), cold] = _HI_emissivity_from_number_density(densities['H'])
            for key in COLD_OMITTED_LINES:
                epsilon[keys.index(key), cold] = 0.
        _checked_values('Adopted volume emissivity', epsilon[:, valid], (len(keys), int(valid.sum())))
        _checked_values('Adopted thermal temperature', temperature[:, valid],
                        (len(keys), int(valid.sum())), positive=True)

    return AdoptedCellEmission(keys, epsilon, temperature, valid.copy(),
                               queries.excluded.copy(), clips)
