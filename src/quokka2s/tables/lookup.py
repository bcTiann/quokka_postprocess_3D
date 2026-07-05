"""Trilinear lookup for the canonical 3D GOW/LVG table."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .models import DespoticTable, SpeciesRecord
from .solver import LINE_RESULT_FIELDS


class TableLookup:
    """Sample table fields in log10 ``(nH, N_H, dVdr)`` space."""

    _EVAL_CHUNK = 4_000_000

    def __init__(self, table: DespoticTable):
        self.table = table
        self._axes = tuple(np.log10(values) for values in (
            table.nH_values, table.col_density_values, table.dVdr_values,
        ))
        self._interpolators: dict[str, RegularGridInterpolator] = {}
        self._species_meta: dict[str, SpeciesRecord] = dict(table.species_data)
        for token, values in (
            ("tg_final", table.tg_final), ("mu", table.mu_values),
            ("cv", table.cv_values), ("Eint", table.Eint_values),
        ):
            self._register_field(token, values)
        for name, record in self._species_meta.items():
            self._register_field(f"species:{name}:abundance", record.abundance)
            if record.line is not None:
                for field in LINE_RESULT_FIELDS:
                    self._register_field(f"species:{name}:line:{field}", getattr(record.line, field))
                self._register_field(f"species:{name}:lumPerH", record.line.lumPerH)
        if table.energy_terms:
            for name, values in table.energy_terms.items():
                self._register_field(f"energy:{name}", values)

    def _register_field(self, token: str, values: np.ndarray) -> None:
        self._interpolators[token] = RegularGridInterpolator(
            self._axes, np.asarray(values, dtype=float), method="linear",
            bounds_error=False, fill_value=np.nan,
        )

    def _eval(self, token: str, nH, colDen, dVdr) -> np.ndarray:
        if token not in self._interpolators:
            raise KeyError(f"Field '{token}' is not registered")
        nH_arr, col_arr, dvdr_arr = np.broadcast_arrays(
            np.asarray(nH, dtype=float), np.asarray(colDen, dtype=float), np.asarray(dVdr, dtype=float)
        )
        shape = nH_arr.shape
        flat = (nH_arr.ravel(), col_arr.ravel(), dvdr_arr.ravel())
        values = np.empty(flat[0].size)
        for start in range(0, values.size, self._EVAL_CHUNK):
            end = min(start + self._EVAL_CHUNK, values.size)
            points = np.column_stack(tuple(np.log10(axis[start:end]) for axis in flat))
            values[start:end] = self._interpolators[token](points)
        return values.reshape(shape)

    def mu(self, nH, colDen, dVdr) -> np.ndarray:
        return self._eval("mu", nH, colDen, dVdr)

    def cv(self, nH, colDen, dVdr) -> np.ndarray:
        return self._eval("cv", nH, colDen, dVdr)

    def Eint(self, nH, colDen, dVdr) -> np.ndarray:
        return self._eval("Eint", nH, colDen, dVdr)

    def temperature(self, nH, colDen, dVdr) -> np.ndarray:
        return self._eval("tg_final", nH, colDen, dVdr)

    def abundance(self, species: str, nH, colDen, dVdr) -> np.ndarray:
        return self._eval(f"species:{species}:abundance", nH, colDen, dVdr)

    def field(self, token: str, nH, colDen, dVdr) -> np.ndarray:
        return self._eval(token, nH, colDen, dVdr)

    def number_densities(self, species: Sequence[str], nH, colDen, dVdr) -> dict[str, np.ndarray]:
        return {name: nH * self.abundance(name, nH, colDen, dVdr) for name in species}

    def line_field(self, species: str, field_name: str, nH, colDen, dVdr) -> np.ndarray:
        record = self._species_meta.get(species)
        if record is None or record.line is None:
            raise ValueError(f"Species '{species}' has no line data")
        if field_name not in LINE_RESULT_FIELDS:
            raise ValueError(f"Unknown line field '{field_name}'; expected one of {LINE_RESULT_FIELDS}")
        return self._eval(f"species:{species}:line:{field_name}", nH, colDen, dVdr)

    def species_record(self, species: str) -> SpeciesRecord:
        try:
            return self._species_meta[species]
        except KeyError as exc:
            raise ValueError(f"Species '{species}' not found; available: {', '.join(self._species_meta)}") from exc
