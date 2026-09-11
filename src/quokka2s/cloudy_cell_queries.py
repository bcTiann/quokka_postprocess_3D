"""Convert paired simulation states into explicit-depth Cloudy queries.

This opt-in interface does not choose emission branches or adopt a table.
DESPOTIC T and mu must come from the same accepted table query. The caller
supplies QUOKKA internal (not total) energy density and a separately
authorized exclusion mask; invalid cells are never excluded automatically.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cloudy_sixline_lookup import CloudySixLineLookup
from .tables.abundances import QUOKKA_MASS_FRACTIONS
from .tables.model_depth import ModelDepthResult, derive_model_depth


@dataclass(frozen=True)
class CloudyCellEmission:
    emissivity_erg_s_cm3: np.ndarray
    excluded: np.ndarray
    attenuation_column_below_table: np.ndarray
    attenuation_column_above_table: np.ndarray


@dataclass(frozen=True)
class CloudyCellQueries:
    n_H_cm3: np.ndarray
    column_density_H_cm2: np.ndarray
    model_depth_pc: np.ndarray
    state: ModelDepthResult
    excluded: np.ndarray

    def sample(self, lookup: CloudySixLineLookup) -> CloudyCellEmission:
        """Evaluate every stored line; excluded entries remain NaN, not zero.

        Failure-touch exceptions and density/temperature/depth domain errors
        from the strict table reader propagate to the caller. Emission-method
        selection belongs to the caller, which should pass only applicable
        cells when evaluating a particular Cloudy branch.
        """
        if lookup.model_depth_bounds_pc is None:
            raise ValueError("Cell-dependent model depth requires a four-dimensional Cloudy table")
        shape = self.n_H_cm3.shape
        use = ~self.excluded
        epsilon = np.full((len(lookup.line_keys), *shape), np.nan)
        below, above = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
        if np.any(use):
            sample = lookup.sample(
                self.state.temperature_K[use], self.n_H_cm3[use],
                self.column_density_H_cm2[use], model_depth_pc=self.model_depth_pc[use],
            )
            epsilon[:, use] = sample.emissivity_per_nH2 * self.n_H_cm3[use] ** 2
            if not np.isfinite(epsilon[:, use]).all() or np.any(epsilon[:, use] < 0):
                raise ValueError("Cloudy coefficient conversion produced invalid volume emissivity")
            below[use] = sample.attenuation_column_below_table
            above[use] = sample.attenuation_column_above_table
        return CloudyCellEmission(epsilon, self.excluded.copy(), below, above)


def prepare_cloudy_cell_queries(
    density_g_cm3, column_density_H_cm2, temperature_quokka_K,
    internal_energy_density_erg_cm3, temperature_despotic_K,
    mean_molecular_weight_despotic, *, hydrogen_mass_g: float,
    boltzmann_erg_K: float, gravitational_cm3_g_s2: float,
    parsec_cm: float, authorized_excluded=None,
) -> CloudyCellQueries:
    """Use shared X, actual rho, paired T/mu, and the agreed 100 pc cap.

    ``authorized_excluded`` is an explicit boolean array supplied by the
    snapshot-specific policy. Only cold cells with unavailable DESPOTIC T/mu
    may be excluded here. The caller must verify the approved snapshot and
    cell IDs before constructing that mask. Other invalid states raise.
    QUOKKA temperature, density and foreground column must always be valid,
    including for excluded cells, so their mass and coverage remain usable.
    """
    arrays = np.broadcast_arrays(*[
        np.asarray(value, dtype=float) for value in (
            density_g_cm3, column_density_H_cm2, temperature_quokka_K,
            internal_energy_density_erg_cm3, temperature_despotic_K,
            mean_molecular_weight_despotic,
        )
    ])
    rho, column, tq, u, td, mud = arrays
    for name, value in (("density", rho), ("column", column), ("QUOKKA temperature", tq)):
        if not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError(f"Nonpositive or nonfinite {name}")
    if authorized_excluded is None:
        excluded = np.zeros(rho.shape, dtype=bool)
    else:
        mask = np.asarray(authorized_excluded)
        if mask.dtype != np.bool_:
            raise ValueError("authorized_excluded must be explicitly boolean")
        excluded = np.broadcast_to(mask, rho.shape).copy()
    paired_valid = np.isfinite(td) & (td > 0) & np.isfinite(mud) & (mud > 0)
    if np.any(excluded & ((tq >= 3000) | paired_valid)):
        raise ValueError("Authorized exclusions must be cold cells with unavailable DESPOTIC T/mu")
    state = derive_model_depth(
        rho, tq, u, td, mud, hydrogen_mass_g=hydrogen_mass_g,
        boltzmann_erg_K=boltzmann_erg_K, gravitational_cm3_g_s2=gravitational_cm3_g_s2,
        parsec_cm=parsec_cm,
    )
    unexpected = ~state.valid & ~excluded
    if np.any(unexpected):
        raise ValueError(f"Unexpected invalid paired state in {np.count_nonzero(unexpected)} cells")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        n_h = rho * QUOKKA_MASS_FRACTIONS["X"] / hydrogen_mass_g
    if not np.isfinite(n_h).all() or np.any(n_h <= 0):
        raise ValueError("Density conversion produced invalid hydrogen-nuclei density")
    return CloudyCellQueries(n_h, column.copy(), state.model_depth_cm / parsec_cm, state, excluded)
