from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Sequence

from .models import DespoticTable, SpeciesRecord
from .solver import LINE_RESULT_FIELDS


class TableLookup:
    """Helper for sampling DESPOTIC tables in log10(nH)-log10(Ncol)-log10(T) space."""

    def __init__(self, table: DespoticTable):
        self.table = table
        
        # 修改 1：座標軸增加 T_values 維度 (3D)
        log_nH = np.log10(table.nH_values)
        log_col = np.log10(table.col_density_values)
        log_T = np.log10(table.T_values)
        self._axes = (log_nH, log_col, log_T)
        
        self._interpolators: dict[str, RegularGridInterpolator] = {}
        self._species_meta: dict[str, SpeciesRecord] = dict(table.species_data)
        
        # 註冊舊有的欄位
        self._register_field("tg_final", table.tg_final)
        
        # 修改 2：註冊新增的物理量 (mu, cv, Eint)
        self._register_field("mu", table.mu_values)
        self._register_field("cv", table.cv_values)
        self._register_field("Eint", table.Eint_values)
        
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
            fill_value=np.nan # 避免超出邊界時報錯，而是回傳 NaN
        )
    
    def _eval(
        self,
        token: str,
        nH_cgs: np.ndarray,
        colDen_cgs: np.ndarray,
        T_K: np.ndarray, # 修改 3：所有 _eval 都必須傳入 T_K
    ) -> np.ndarray:
        if token not in self._interpolators:
            raise KeyError(f"Field '{token}' not registered in TableLookup.")
        
        # 將輸入的 3 個矩陣組合成 (N, 3) 的座標點進行插值
        log_points = np.column_stack(
            (np.log10(nH_cgs).ravel(), np.log10(colDen_cgs).ravel(), np.log10(T_K).ravel())
        )
        values = self._interpolators[token](log_points)
        return values.reshape(nH_cgs.shape)
    
    # 修改 4：新增三個專屬的 Helper 函數來獲取 mu, cv, Eint
    def mu(self, nH_cgs: np.ndarray, colDen_cgs: np.ndarray, T_K: np.ndarray) -> np.ndarray:
        """Interpolates the mean molecular weight (mu)."""
        return self._eval("mu", nH_cgs, colDen_cgs, T_K)
        
    def cv(self, nH_cgs: np.ndarray, colDen_cgs: np.ndarray, T_K: np.ndarray) -> np.ndarray:
        """Interpolates the dimensionless specific heat at constant volume (cv)."""
        return self._eval("cv", nH_cgs, colDen_cgs, T_K)
        
    def Eint(self, nH_cgs: np.ndarray, colDen_cgs: np.ndarray, T_K: np.ndarray) -> np.ndarray:
        """Interpolates the dimensionless internal energy per H nucleus (Eint)."""
        return self._eval("Eint", nH_cgs, colDen_cgs, T_K)

    def temperature(self, nH_cgs: np.ndarray, colDen_cgs: np.ndarray, T_K: np.ndarray) -> np.ndarray:
        """Interpolates the final gas temperature (Tg_final)."""
        return self._eval("tg_final", nH_cgs, colDen_cgs, T_K)

    def abundance(
        self,
        species: str, 
        nH_cgs: np.ndarray,
        colDen_cgs: np.ndarray,
        T_K: np.ndarray, # 記得加上 T_K
    ) -> np.ndarray:
        """Interpolates the abundance of a given chemical species."""
        return self._eval(f"species:{species}:abundance", nH_cgs, colDen_cgs, T_K)
    
    def field(
        self,
        token: str,
        nH_cgs: np.ndarray,
        colDen_cgs: np.ndarray,
        T_K: np.ndarray, # 記得加上 T_K
    ) -> np.ndarray:
        """Provides generic access to any registered field by its token."""
        return self._eval(token, nH_cgs, colDen_cgs, T_K)
    
    def number_densities(
        self,
        species: Sequence[str],
        n_H_cgs: np.ndarray,
        colDen_cgs: np.ndarray,
        T_K: np.ndarray, # 記得加上 T_K
    ) -> dict[str, np.ndarray]: 
        """Return n_species = n_H * abundances(species)"""
        return {
            sp: n_H_cgs * self.abundance(sp, n_H_cgs, colDen_cgs, T_K)
            for sp in species
        }

    def line_field(
        self,
        species: str,
        field_name: str,
        nH_cgs: np.ndarray,
        colDen_cgs: np.ndarray,
        T_K: np.ndarray, # 記得加上 T_K
    ) -> np.ndarray:
        """Interpolate a specific line property for an emitting species."""
        record = self._species_meta.get(species)
        if record is None or not record.is_emitter or record.line is None:
            raise ValueError(f"Species '{species}' has no line data registered")
        if field_name not in LINE_RESULT_FIELDS:
            raise ValueError(f"Unknown line field '{field_name}'. Expected one of {LINE_RESULT_FIELDS}")
        token = f"species:{species}:line:{field_name}"
        return self._eval(token, nH_cgs, colDen_cgs, T_K)

    def species_record(self, species: str) -> SpeciesRecord:
        """Return the cached SpeciesRecord for metadata queries."""
        if species not in self._species_meta:
            available = ", ".join(self._species_meta)
            raise ValueError(f"Species '{species}' not found. Available species: {available}")
        return self._species_meta[species]
