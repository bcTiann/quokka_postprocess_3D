"""Lookup support for CHIANTI [C II] 158 um upper-level populations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class CiiUpperFractionLookup:
    """Interpolate log10(N_u) on a regular log10(T)-log10(n_H) grid.

    Values outside the precomputed table are clamped to its boundary.  The
    table stores the fractional population of CHIANTI C II level 2, not an
    abundance, ion fraction, or emissivity normalization.
    """

    def __init__(self, log10_temperature, log10_hydrogen_density, log10_nu):
        self.log10_temperature = np.asarray(log10_temperature, dtype=float)
        self.log10_hydrogen_density = np.asarray(log10_hydrogen_density, dtype=float)
        self.log10_nu = np.asarray(log10_nu, dtype=float)

        expected = (
            self.log10_temperature.size,
            self.log10_hydrogen_density.size,
        )
        if self.log10_nu.shape != expected:
            raise ValueError(
                f'CHIANTI C II Nu table has shape {self.log10_nu.shape}; '
                f'expected {expected}'
            )
        if not np.all(np.diff(self.log10_temperature) > 0):
            raise ValueError('CHIANTI C II temperature grid must be increasing')
        if not np.all(np.diff(self.log10_hydrogen_density) > 0):
            raise ValueError('CHIANTI C II hydrogen-density grid must be increasing')
        if not np.all(np.isfinite(self.log10_nu)):
            raise ValueError('CHIANTI C II Nu table contains non-finite values')

        self._interpolator = RegularGridInterpolator(
            (self.log10_temperature, self.log10_hydrogen_density),
            self.log10_nu,
            bounds_error=True,
        )

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        hydrogen_mass_fraction: float | None = None,
        helium_mass_fraction: float | None = None,
    ):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f'CHIANTI C II Nu table not found: {path}. '
                'Run scripts/build_cii_chianti_nu_table.py in the quokka environment.'
            )
        with np.load(path) as table:
            expected_metadata = {
                'hydrogen_mass_fraction': hydrogen_mass_fraction,
                'helium_mass_fraction': helium_mass_fraction,
            }
            if 'collider_elements' not in table or str(table['collider_elements']) != 'H+He':
                raise ValueError(
                    'CHIANTI C II table was not built with the required H+He '
                    'collider composition'
                )
            for key, expected in expected_metadata.items():
                if expected is None:
                    continue
                actual = float(table[key])
                if not np.isclose(actual, expected, rtol=0.0, atol=1e-14):
                    raise ValueError(
                        f'CHIANTI C II table {key}={actual} does not match '
                        f'pipeline value {expected}'
                    )
            return cls(
                table['log10_temperature_K'],
                table['log10_hydrogen_density_cm3'],
                table['log10_upper_fraction'],
            )

    def __call__(self, temperature_K, hydrogen_density_cm3):
        temperature, density = np.broadcast_arrays(
            np.asarray(temperature_K, dtype=float),
            np.asarray(hydrogen_density_cm3, dtype=float),
        )
        log_temperature = np.log10(np.maximum(temperature, np.finfo(float).tiny))
        log_density = np.log10(np.maximum(density, np.finfo(float).tiny))

        log_temperature = np.clip(
            log_temperature,
            self.log10_temperature[0],
            self.log10_temperature[-1],
        )
        log_density = np.clip(
            log_density,
            self.log10_hydrogen_density[0],
            self.log10_hydrogen_density[-1],
        )
        points = np.column_stack((log_temperature.ravel(), log_density.ravel()))
        upper_fraction = np.power(10.0, self._interpolator(points))
        upper_fraction = upper_fraction.reshape(temperature.shape)
        # At n_H=0, the CIE-scaled n_e and n_p also vanish exactly.
        return np.where(density > 0.0, upper_fraction, 0.0)
