"""Opt-in Cloudy emission for the adopted hot atomic-line branch.

The caller supplies queries from the validated cell-state bridge and the
checked lookup returned by ``validated_coverage_lookup``. This module does
not select a table or supply any cold-branch emissivity.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .cloudy_cell_queries import CloudyCellEmission, CloudyCellQueries
from .cloudy_sixline_lookup import CloudySixLineLookup, TOUCH_EPS
from .tables.abundances import QUOKKA_MASS_FRACTIONS


@dataclass(frozen=True)
class CloudyHotEmission(CloudyCellEmission):
    """Full cell-shaped result with a separate hot-branch applicability mask.

    Cold and excluded cells have NaN emissivity because this adapter does
    not evaluate them. ``excluded`` retains only the caller's authorized
    exclusions; an ordinary cold cell is not an excluded simulation cell.
    Attenuation clipping flags describe only the cells actually queried.
    """

    applicable: np.ndarray


def sample_cloudy_hot_emission(
    queries: CloudyCellQueries, lookup: CloudySixLineLookup,
    *, allow_capped_legacy_jeans: bool = False,
) -> CloudyHotEmission:
    """Sample all stored atomic lines only where T_QUOKKA >= 3000 K.

    Failure-touch and physical-domain exceptions propagate from the strict
    sampler. The legacy Jeans table is rejected unless the caller explicitly
    permits its use after validating provenance. In that mode queried hot
    model depths must be 100 pc and contributing table nodes must reach the
    original CIAOLoop cap, 3.086e20 cm (rounded parsec conversion), with both
    the historical and corrected density conversions. No legacy
    table is accepted for arbitrary cell-dependent model depths.
    Original cell columns are passed unchanged;
    the lookup clips only the incident-radiation query coordinate. Returned
    coefficients are converted to volume emissivity with the actual n_H^2.
    """
    legacy = lookup.model_depth_bounds_pc is None
    if legacy and not allow_capped_legacy_jeans:
        raise ValueError("Hot cell emission requires a four-dimensional Cloudy table")
    applicable = ~queries.state.cold_mask & ~queries.excluded
    shape = queries.n_H_cm3.shape
    epsilon = np.full((len(lookup.line_keys), *shape), np.nan)
    below = np.zeros(shape, dtype=bool)
    above = np.zeros(shape, dtype=bool)
    if np.any(applicable):
        depth_arguments = {"model_depth_pc": queries.model_depth_pc[applicable]}
        if legacy:
            _validate_capped_legacy_jeans_support(queries, lookup, applicable)
            depth_arguments = {}
        sampled = lookup.sample(
            queries.state.temperature_K[applicable],
            queries.n_H_cm3[applicable],
            queries.column_density_H_cm2[applicable],
            **depth_arguments,
        )
        epsilon[:, applicable] = (
            sampled.emissivity_per_nH2 * queries.n_H_cm3[applicable] ** 2
        )
        if not np.isfinite(epsilon[:, applicable]).all() or np.any(epsilon[:, applicable] < 0):
            raise ValueError("Cloudy coefficient conversion produced invalid hot volume emissivity")
        below[applicable] = sampled.attenuation_column_below_table
        above[applicable] = sampled.attenuation_column_above_table
    return CloudyHotEmission(
        epsilon, queries.excluded.copy(), below, above, applicable.copy(),
    )


def _validate_capped_legacy_jeans_support(queries, lookup, applicable):
    """Check geometry using the exact original CIAOLoop Jeans constants.

    The external cell depth uses actual simulation rho and a precise parsec;
    the legacy table used rho=nH*mH/0.76, mu=1 and a rounded 100 pc cap.
    Equality of the capped states is authorized only for this small rounding
    difference. Interpolation must not bring in a shorter model under either
    the legacy conversion or the corrected QUOKKA hydrogen mass fraction.
    """
    depth = queries.model_depth_pc[applicable]
    if not np.isfinite(depth).all() or not np.allclose(depth, 100., rtol=1e-12, atol=0):
        raise ValueError("Legacy Jeans hot queries require model depths of 100 pc")
    # Reuse the sampler's domain validation, attenuation clipping and brackets.
    _, _, _, brackets = lookup._prepare_query(
        queries.state.temperature_K[applicable], queries.n_H_cm3[applicable],
        queries.column_density_H_cm2[applicable], None,
    )
    hydrogen_mass = 1.67373522381e-24
    boltzmann = 1.3806488e-16
    gravitational = 6.67384e-8
    rho = 10. ** lookup.log_nH[:, None] * hydrogen_mass / .76
    temperature = 10. ** lookup.log_T[None, :]
    length_cm = np.pi * np.sqrt(
        (5. / 3.) * boltzmann * temperature / (gravitational * hydrogen_mass * rho)
    )
    corrected_length_cm = length_cm * np.sqrt(QUOKKA_MASS_FRACTIONS['X'] / .76)
    capped = (length_cm >= 3.086e20) & (corrected_length_cm >= 3.086e20)
    uncapped_weight = np.zeros(depth.size)
    for choices in product((0, 1), repeat=3):
        indices = [bracket[choice] for bracket, choice in zip(brackets, choices)]
        weight = np.ones(depth.size)
        for (_, _, fraction), choice in zip(brackets, choices):
            weight *= fraction if choice else 1. - fraction
        uncapped_weight += weight * ~capped[indices[1], indices[2]]
    if np.any(uncapped_weight > TOUCH_EPS):
        count = int(np.count_nonzero(uncapped_weight > TOUCH_EPS))
        raise ValueError(
            f"Legacy Jeans interpolation touches uncapped table nodes in {count} hot cells "
            "under the historical or corrected density conversion"
        )
