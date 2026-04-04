from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Sequence

from .models import DespoticTable, SpeciesRecord
from .solver import LINE_RESULT_FIELDS


class TableLookup:
    """Helper for sampling DESPOTIC tables in log10(nH)-log10(Ncol) space."""

    def __init__(self, table: DespoticTable):
        self.table = table
        log_nH = np.log10(table.nH_values)
        log_col = np.log10(table.col_density_values)
        self._axes = (log_nH, log_col)
        self._interpolators: dict[str, RegularGridInterpolator] = {}
        self._species_meta: dict[str, SpeciesRecord] = dict(table.species_data)
        self._register_field("tg_final", table.tg_final)
        for name, record in self._species_meta.items():
            self._register_field(f"species:{name}:abundance", record.abundance)
            if record.is_emitter and record.line is not None:
                for field in LINE_RESULT_FIELDS:
                    values = getattr(record.line, field)
                    self._register_field(f"species:{name}:line:{field}", values)
                # Preserve backward compatibility for lumPerH token users
                self._register_field(f"species:{name}:lumPerH", record.line.lumPerH)
        if table.energy_terms:
            for term, values in table.energy_terms.items():
                self._register_field(f"energy:{term}", values)

    def _register_field(self, token: str, values: np.ndarray) -> None:
        self._interpolators[token] = RegularGridInterpolator(
            self._axes,
            np.asarray(values, dtype=float),
            method="linear",
            bounds_error=False,
        )
    
    def _eval(
        self,
        token: str,
        nH_cgs: np.ndarray,
        colDen_cgs: np.ndarray,
    ) -> np.ndarray:
        if token not in self._interpolators:
            raise KeyError(f"Field '{token}' not registered in TableLookup.")
        log_points = np.column_stack(
            (np.log10(nH_cgs).ravel(), np.log10(colDen_cgs).ravel())
        )
        values = self._interpolators[token](log_points)
        return values.reshape(nH_cgs.shape)
    
    def temperature(self, nH_cgs: np.ndarray, colDen_cgs: np.ndarray) -> np.ndarray:
        """Interpolates the final gas temperature (Tg_final)."""
        return self._eval("tg_final", nH_cgs, colDen_cgs)

    def abundance(
        self,
        species: str, 
        nH_cgs: np.ndarray,
        colDen_cgs: np.ndarray,
    ) -> np.ndarray:
        """Interpolates the abundance of a given chemical species.

        Args:
            species: The string name of the species to query (e.g., "CO", "H2").
            nH_cgs: A numpy array of nH values (in cm^-3, linear space).
            colDen_cgs: A numpy array of column density values (in cm^-2,
                        linear space). Must have the same shape as `nH_cgs`.

        Returns:
            A numpy array of the interpolated species abundances (unitless),
            with the same shape as the input `nH_cgs` array.
        """
        return self._eval(f"species:{species}:abundance", nH_cgs, colDen_cgs)
    
    def field(
        self,
        token: str,
        nH_cgs: np.ndarray,
        colDen_cgs: np.ndarray,
    ) -> np.ndarray:
        """Provides generic access to any registered field by its token.

        This is a "power-user" method to query fields that may not have a
        dedicated helper, such as specific energy terms (e.g., "energy:H2_LTE").

        Args:
            token: The internal string identifier for the field
                (e.g., "tg_final", "energy:H2_LTE").
            nH_cgs: A numpy array of nH values (in cm^-3, linear space).
            colDen_cgs: A numpy array of column density values (in cm^-2,
                        linear space). Must have the same shape as `nH_cgs`.

        Returns:
            A numpy array of the interpolated field values, with the same
            shape as the input `nH_cgs` array.
        """
        return self._eval(token, nH_cgs, colDen_cgs)
    
    def number_densities(
        self,
        species: Sequence[str],
        n_H_cgs: np.ndarray,
        colDen_cgs: np.ndarray,
    ) -> dict[str, np.ndarray]: 
        """Return n_species = n_H * abundances(species)"""
        return {
            sp: n_H_cgs * self.abundance(sp, n_H_cgs, colDen_cgs)
            for sp in species
        }

    def line_field(
        self,
        species: str,
        field_name: str,
        nH_cgs: np.ndarray,
        colDen_cgs: np.ndarray,
    ) -> np.ndarray:
        """Interpolate a specific line property for an emitting species."""
        record = self._species_meta.get(species)
        if record is None or not record.is_emitter or record.line is None:
            raise ValueError(f"Species '{species}' has no line data registered")
        if field_name not in LINE_RESULT_FIELDS:
            raise ValueError(f"Unknown line field '{field_name}'. Expected one of {LINE_RESULT_FIELDS}")
        token = f"species:{species}:line:{field_name}"
        return self._eval(token, nH_cgs, colDen_cgs)

    def species_record(self, species: str) -> SpeciesRecord:
        """Return the cached SpeciesRecord for metadata queries."""
        if species not in self._species_meta:
            available = ", ".join(self._species_meta)
            raise ValueError(f"Species '{species}' not found. Available species: {available}")
        return self._species_meta[species]
