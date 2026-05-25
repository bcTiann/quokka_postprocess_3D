from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Sequence

from .models import DespoticTable, SpeciesRecord
from .solver import LINE_RESULT_FIELDS


class TableLookup:
    """Helper for sampling DESPOTIC tables in log10(nH)-log10(Ncol)-log10(dVdr) space."""

    def __init__(self, table: DespoticTable):
        self.table = table

        log_nH   = np.log10(table.nH_values)
        log_col  = np.log10(table.col_density_values)
        log_dvdr = np.log10(table.dVdr_values)
        self._axes = (log_nH, log_col, log_dvdr)

        self._interpolators: dict[str, RegularGridInterpolator] = {}
        self._species_meta: dict[str, SpeciesRecord] = dict(table.species_data)

        self._register_field("tg_final", table.tg_final)
        self._register_field("mu", table.mu_values)
        self._register_field("cv", table.cv_values)
        self._register_field("Eint", table.Eint_values)

        for name, record in self._species_meta.items():
            self._register_field(f"species:{name}:abundance", record.abundance)
            if record.is_emitter and record.line is not None:
                for field in LINE_RESULT_FIELDS:
                    values = getattr(record.line, field)
                    self._register_field(f"species:{name}:line:{field}", values)
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
            fill_value=np.nan,
        )

    # Chunk size for _eval — process this many cells per scipy call. Tuned
    # so the per-chunk transient (log_points + scipy internal state) stays
    # well under 1 GB at float64.  Lowering further only saves trivial RAM
    # but pays Python-loop overhead.
    _EVAL_CHUNK = 4_000_000

    def _eval(
        self,
        token: str,
        nH_cgs: np.ndarray,
        colDen_cgs: np.ndarray,
        dVdr_cgs: np.ndarray,
    ) -> np.ndarray:
        if token not in self._interpolators:
            raise KeyError(f"Field '{token}' not registered in TableLookup.")

        interp = self._interpolators[token]
        out_shape = nH_cgs.shape
        nH_flat   = nH_cgs.ravel()
        col_flat  = colDen_cgs.ravel()
        dv_flat   = dVdr_cgs.ravel()
        n         = nH_flat.size

        # Chunked interpolation: a 134 M-cell snapshot at down=1 makes a single
        # `np.column_stack(log10(...), ...)` allocate ~3 GB and scipy's
        # RegularGridInterpolator another ~10 GB transient.  Splitting into
        # 4 M-cell chunks keeps each call's working set ~0.1 GB.
        values = np.empty(n, dtype=float)
        for start in range(0, n, self._EVAL_CHUNK):
            end = min(start + self._EVAL_CHUNK, n)
            pts = np.column_stack((
                np.log10(nH_flat[start:end]),
                np.log10(col_flat[start:end]),
                np.log10(dv_flat[start:end]),
            ))
            values[start:end] = interp(pts)
            del pts
        return values.reshape(out_shape)

    def mu(self, nH_cgs: np.ndarray, colDen_cgs: np.ndarray, dVdr_cgs: np.ndarray) -> np.ndarray:
        """Interpolates the mean molecular weight (mu)."""
        return self._eval("mu", nH_cgs, colDen_cgs, dVdr_cgs)

    def cv(self, nH_cgs: np.ndarray, colDen_cgs: np.ndarray, dVdr_cgs: np.ndarray) -> np.ndarray:
        """Interpolates the dimensionless specific heat at constant volume (cv)."""
        return self._eval("cv", nH_cgs, colDen_cgs, dVdr_cgs)

    def Eint(self, nH_cgs: np.ndarray, colDen_cgs: np.ndarray, dVdr_cgs: np.ndarray) -> np.ndarray:
        """Interpolates the dimensionless internal energy per H nucleus (Eint)."""
        return self._eval("Eint", nH_cgs, colDen_cgs, dVdr_cgs)

    def temperature(self, nH_cgs: np.ndarray, colDen_cgs: np.ndarray, dVdr_cgs: np.ndarray) -> np.ndarray:
        """Interpolates the self-consistent gas temperature (Tg_final)."""
        return self._eval("tg_final", nH_cgs, colDen_cgs, dVdr_cgs)

    def abundance(
        self,
        species: str,
        nH_cgs: np.ndarray,
        colDen_cgs: np.ndarray,
        dVdr_cgs: np.ndarray,
    ) -> np.ndarray:
        """Interpolates the abundance of a given chemical species."""
        return self._eval(f"species:{species}:abundance", nH_cgs, colDen_cgs, dVdr_cgs)

    def field(
        self,
        token: str,
        nH_cgs: np.ndarray,
        colDen_cgs: np.ndarray,
        dVdr_cgs: np.ndarray,
    ) -> np.ndarray:
        """Provides generic access to any registered field by its token."""
        return self._eval(token, nH_cgs, colDen_cgs, dVdr_cgs)

    def number_densities(
        self,
        species: Sequence[str],
        n_H_cgs: np.ndarray,
        colDen_cgs: np.ndarray,
        dVdr_cgs: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return n_species = n_H * abundances(species)"""
        return {
            sp: n_H_cgs * self.abundance(sp, n_H_cgs, colDen_cgs, dVdr_cgs)
            for sp in species
        }

    def line_field(
        self,
        species: str,
        field_name: str,
        nH_cgs: np.ndarray,
        colDen_cgs: np.ndarray,
        dVdr_cgs: np.ndarray,
    ) -> np.ndarray:
        """Interpolate a specific line property for an emitting species."""
        record = self._species_meta.get(species)
        if record is None or not record.is_emitter or record.line is None:
            raise ValueError(f"Species '{species}' has no line data registered")
        if field_name not in LINE_RESULT_FIELDS:
            raise ValueError(f"Unknown line field '{field_name}'. Expected one of {LINE_RESULT_FIELDS}")
        token = f"species:{species}:line:{field_name}"
        return self._eval(token, nH_cgs, colDen_cgs, dVdr_cgs)

    def species_record(self, species: str) -> SpeciesRecord:
        """Return the cached SpeciesRecord for metadata queries."""
        if species not in self._species_meta:
            available = ", ".join(self._species_meta)
            raise ValueError(f"Species '{species}' not found. Available species: {available}")
        return self._species_meta[species]
