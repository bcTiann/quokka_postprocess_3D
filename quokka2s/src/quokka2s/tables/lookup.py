from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Sequence

from .models import DespoticTable, SpeciesRecord  # DespoticTable4D dropped 2026-06-23 (4D deprecated; used only by wrapped TableLookup4D below)
from .solver import LINE_RESULT_FIELDS

# CGS constants for the μγ bisection (kept local so lookup.py has no yt dep).
_M_H_CGS = 1.6726219e-24   # g   (hydrogen mass, matches yt.units.mh.in_cgs())
_K_B_CGS = 1.380649e-16    # erg/K


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


# ============================================================================
# DEPRECATED 2026-06-23 — kept in-tree for reference (wrap-don't-delete).
# 4D (nH,NH,dVdr,μγ) DESPOTIC table path was retired in cache schema v5
# (2026-06-13); no pipeline task references any 4D table.  To revive:
# restore the 4D exports in tables/__init__.py and un-wrap these defs.
# ============================================================================
r'''
class TableLookup4D:
    """Sampler for the fixed-T 4D table in log10(nH, N_H, dVdr, T) space.

    Mirrors :class:`TableLookup` but with a fourth (temperature) axis.  Adds
    :meth:`temperature_gamma_mu`, the per-cell μγ bisection that inverts
    QUOKKA's specific internal energy into a chemistry-consistent temperature.
    """

    _EVAL_CHUNK = 4_000_000

    def __init__(self, table: DespoticTable4D):
        self.table = table

        log_nH   = np.log10(table.nH_values)
        log_col  = np.log10(table.col_density_values)
        log_dvdr = np.log10(table.dVdr_values)
        log_T    = np.log10(table.T_values)
        self._axes = (log_nH, log_col, log_dvdr, log_T)

        self._interpolators: dict[str, RegularGridInterpolator] = {}
        self._species_meta: dict[str, SpeciesRecord] = dict(table.species_data)

        self._register_field("mu", table.mu_values)
        self._register_field("cv", table.cv_values)
        self._register_field("Eint", table.Eint_values)

        # g(T) = T / [(γ-1)·μ] = T·cv/μ  (monotone in T → unique bisection root).
        with np.errstate(invalid="ignore", divide="ignore"):
            g_values = (table.T_values[None, None, None, :]
                        * table.cv_values / table.mu_values)
        self._register_field("g", g_values)

        for name, record in self._species_meta.items():
            self._register_field(f"species:{name}:abundance", record.abundance)
            if record.is_emitter and record.line is not None:
                for fld in LINE_RESULT_FIELDS:
                    self._register_field(f"species:{name}:line:{fld}", getattr(record.line, fld))
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

    def _eval(self, token: str, nH, colDen, dVdr, T) -> np.ndarray:
        if token not in self._interpolators:
            raise KeyError(f"Field '{token}' not registered in TableLookup4D.")
        interp = self._interpolators[token]
        nH_arr = np.asarray(nH, dtype=float)
        out_shape = nH_arr.shape
        # Broadcast scalars / lower-rank arrays in the ORIGINAL ND-shape space
        # before ravelling.  Previous code tried `broadcast_to(dVdr, flat_shape)`
        # which silently worked when dVdr was a scalar (broadcasts () → (N,))
        # but crashed when callers passed a same-shape ND array (yt now does:
        # _temperature_gamma_mu passes the full 3D dVdr_lvg field).
        nH_flat  = nH_arr.ravel()
        col_flat = np.broadcast_to(np.asarray(colDen, dtype=float), out_shape).ravel()
        dv_flat  = np.broadcast_to(np.asarray(dVdr,   dtype=float), out_shape).ravel()
        T_flat   = np.broadcast_to(np.asarray(T,      dtype=float), out_shape).ravel()
        n = nH_flat.size

        values = np.empty(n, dtype=float)
        for start in range(0, n, self._EVAL_CHUNK):
            end = min(start + self._EVAL_CHUNK, n)
            pts = np.column_stack((
                np.log10(nH_flat[start:end]),
                np.log10(col_flat[start:end]),
                np.log10(dv_flat[start:end]),
                np.log10(T_flat[start:end]),
            ))
            values[start:end] = interp(pts)
            del pts
        return values.reshape(out_shape)

    def mu(self, nH, colDen, dVdr, T) -> np.ndarray:
        return self._eval("mu", nH, colDen, dVdr, T)

    def cv(self, nH, colDen, dVdr, T) -> np.ndarray:
        return self._eval("cv", nH, colDen, dVdr, T)

    def Eint(self, nH, colDen, dVdr, T) -> np.ndarray:
        return self._eval("Eint", nH, colDen, dVdr, T)

    def abundance(self, species, nH, colDen, dVdr, T) -> np.ndarray:
        return self._eval(f"species:{species}:abundance", nH, colDen, dVdr, T)

    def number_densities(self, species: Sequence[str], n_H, colDen, dVdr, T) -> dict[str, np.ndarray]:
        return {sp: n_H * self.abundance(sp, n_H, colDen, dVdr, T) for sp in species}

    def line_field(self, species: str, field_name: str, nH, colDen, dVdr, T) -> np.ndarray:
        record = self._species_meta.get(species)
        if record is None or not record.is_emitter or record.line is None:
            raise ValueError(f"Species '{species}' has no line data registered")
        if field_name not in LINE_RESULT_FIELDS:
            raise ValueError(f"Unknown line field '{field_name}'. Expected one of {LINE_RESULT_FIELDS}")
        return self._eval(f"species:{species}:line:{field_name}", nH, colDen, dVdr, T)

    def field(self, token: str, nH, colDen, dVdr, T) -> np.ndarray:
        return self._eval(token, nH, colDen, dVdr, T)

    def species_record(self, species: str) -> SpeciesRecord:
        if species not in self._species_meta:
            available = ", ".join(self._species_meta)
            raise ValueError(f"Species '{species}' not found. Available species: {available}")
        return self._species_meta[species]

    def temperature_gamma_mu(
        self,
        nH_cgs: np.ndarray,
        colDen_cgs: np.ndarray,
        e_specific_cgs: np.ndarray,
        dVdr_cgs: float | np.ndarray | None = None,
        n_iter: int = 40,
    ) -> np.ndarray:
        """Invert specific internal energy → temperature via the μγ relation.

        Solves   g(T) ≡ T / [(γ(T)−1)·μ(T)] = e_specific · m_H / k_B
        for T at each cell, by bisection in log10(T) over the table's T range.
        μ and γ=(cv+1)/cv come from the table; ``g`` is precomputed and monotone
        in T, so the root is unique.

        ``dVdr_cgs`` should be the per-cell LVG velocity gradient (same array
        shape as nH).  Pre-2026-05-29 the 4D build broadcast μ/cv across dVdr
        so any slice worked; the new builder solves chemistry per-dVdr so this
        argument is now physically meaningful.  If ``None`` (legacy callers),
        fall back to the table's median dVdr with a warning-worthy note in
        the docstring.

        Parameters are flat or N-D arrays in cgs (e_specific in erg/g).
        Returns T in K, clipped to the table's [T_min, T_max].
        """
        nH = np.asarray(nH_cgs, dtype=float)
        col = np.asarray(colDen_cgs, dtype=float)
        target = np.asarray(e_specific_cgs, dtype=float) * (_M_H_CGS / _K_B_CGS)  # K

        if dVdr_cgs is None:
            # Legacy fallback for old call sites — μ/cv are now genuinely
            # dVdr-dependent so callers should pass the cell's real dVdr.
            dVdr_cgs = float(self.table.dVdr_values[len(self.table.dVdr_values) // 2])

        log_T_lo = np.log10(self.table.T_values.min())
        log_T_hi = np.log10(self.table.T_values.max())
        lo = np.full(nH.shape, log_T_lo, dtype=float)
        hi = np.full(nH.shape, log_T_hi, dtype=float)

        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            g_mid = self._eval("g", nH, col, dVdr_cgs, np.power(10.0, mid))
            # g increasing in T: if g(mid) < target, root is higher → raise lo.
            go_up = g_mid < target
            lo = np.where(go_up, mid, lo)
            hi = np.where(go_up, hi, mid)

        T = np.power(10.0, 0.5 * (lo + hi))
        return np.clip(T, self.table.T_values.min(), self.table.T_values.max())
'''
