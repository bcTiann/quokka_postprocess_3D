"""Build_PhaseHist + Build_PhaseHistNHRho — single 2D histograms on the ρ-T plane.

Each task computes EXACTLY ONE 2D histogram and stores it as an intermediate.
``Plot_PhaseCombined`` then reads ALL the sibling Build results and assembles
phase_combined.png.

These are compute-only Build tasks (no plot of their own).  The colorbar UNIT is
NOT hardcoded: each task keeps the weight in yt units long enough to read its
latex unit (mass → g, luminosity → erg/s) and stores ``unit_latex``; the caller
passes only a short ``symbol`` (e.g. ``r'L_{\\rm CO}'``).  Plot_PhaseCombined
builds the label as ``$\\log_{10}\\,{symbol}$ [{unit_latex}]``.

Usage (in `run_pipeline.py`):
    pipeline.register_task(Build_PhaseHist(cfg, 'mass', 'temperature_quokka',
                                           tag='mass_T_QK', symbol=r'M_{\\rm bin}'))
    pipeline.register_task(Build_PhaseHist(cfg, 'CO_luminosity',
                                           'temperature_two_regime',
                                           tag='CO_T_2R', symbol=r'L_{\\rm CO}'))
    ...
    pipeline.register_task(Plot_PhaseCombined(cfg))   # reads all the above
"""
from __future__ import annotations

import gc

import numpy as np

from ..base import BuildTask, PipelinePlotContext


def _aligned_edges(lo: float, hi: float, step: float) -> np.ndarray:
    """Snap (lo, hi) to multiples of `step` and return uniform bin edges.

    Returns array of length n_bins+1 with bins of width ~`step`.
    """
    lo_snap = np.floor(lo / step) * step
    hi_snap = (np.ceil(hi / step) + 0.5) * step
    n_bins  = int(np.round((hi_snap - lo_snap) / step))
    return np.linspace(lo_snap, hi_snap, n_bins + 1)


def _unit_latex(yt_array) -> str:
    """yt/unyt unit → LaTeX string for a colorbar label, with a plain-str
    fallback (e.g. dimensionless → '')."""
    try:
        return yt_array.units.latex_repr or str(yt_array.units)
    except Exception:
        return str(getattr(yt_array, 'units', ''))


# ─── Build_PhaseHist ──────────────────────────────────────────────────────────
class Build_PhaseHist(BuildTask):
    """Compute ONE weighted 2D histogram on (log ρ, log T).

    Parameters
    ----------
    weight_field : str
        Either ``'mass'`` (weight = ρ·dV) or a yt luminosity field name like
        ``'CO_luminosity'`` (weight = ε·dV).
    T_field : str
        yt field for the y axis, e.g. ``'temperature_two_regime'``.
    tag : str
        Short label stored in the result.  Plot_PhaseCombined finds the right
        panel by this tag.  Must be unique across registered instances.
    symbol : str, optional
        LaTeX symbol for the colorbar (e.g. ``r'L_{\\rm CO}'`` / ``r'M_{\\rm bin}'``).
        The unit is derived from the weight field via yt, not hardcoded.
    bin_dex : float
        log10 bin width (default 0.2 dex).
    """

    def __init__(self, config,
                 weight_field: str, T_field: str, tag: str,
                 symbol: str = '',
                 bin_dex: float = 0.2):
        super().__init__(config, name=f'Build_PhaseHist[{tag}]')
        self.weight_field = str(weight_field)
        self.T_field      = str(T_field)
        self.tag          = str(tag)
        self.symbol       = str(symbol)
        self.bin_dex      = float(bin_dex)

    def compute(self, context: PipelinePlotContext) -> dict:
        p = context.provider

        rho_u, _ = p.get_slab_z(('gas', 'density'))
        T_u,   _ = p.get_slab_z(('gas', self.T_field))
        dx_u,  _ = p.get_slab_z(('boxlib', 'dx'))
        dy_u,  _ = p.get_slab_z(('boxlib', 'dy'))
        dz_u,  _ = p.get_slab_z(('boxlib', 'dz'))

        rho_q = rho_u.in_cgs().ravel()                                   # yt, g/cm**3
        dV_q  = (dx_u.in_cgs() * dy_u.in_cgs() * dz_u.in_cgs()).ravel()  # yt, cm**3
        rho = np.asarray(rho_q)
        T   = np.asarray(T_u.in_cgs()).ravel()
        del rho_u, T_u, dx_u, dy_u, dz_u

        with np.errstate(divide='ignore', invalid='ignore'):
            log_rho = np.log10(np.where(rho > 0, rho, np.nan))
            log_T   = np.log10(np.where(T   > 0, T,   np.nan))

        # Weight kept in yt units long enough to read its latex unit, then
        # dropped to a plain array for histogram2d (values unchanged).
        if self.weight_field == 'mass':
            w_q = rho_q * dV_q                                          # mass (g)
            label_unit = 'g'
        else:
            eps_u, _ = p.get_slab_z(('gas', self.weight_field))
            w_q = eps_u.in_cgs().ravel() * dV_q                         # luminosity
            label_unit = 'erg/s'
            del eps_u
        # Histogram uses the cgs VALUES (so H stays bit-identical regardless of
        # how yt labels the unit).  The colorbar LABEL is yt-derived but in the
        # conventional unit ('erg/s'/'g'), so it reads 'erg/s' rather than yt's
        # decomposed base-unit form 'cm²·g/s³'.  `in_units` validates the dims.
        weight = w_q.value
        unit_latex = _unit_latex((1.0 * w_q.units).in_units(label_unit))
        del rho, T, rho_q, dV_q, w_q

        # Bin edges from THIS task's own data range (tight to actual data).
        # Plot task aligns axes across siblings via union of xlim/ylim.
        rho_lo, rho_hi = float(np.nanmin(log_rho)), float(np.nanmax(log_rho))
        T_lo,   T_hi   = float(np.nanmin(log_T)),   float(np.nanmax(log_T))
        x_edges = _aligned_edges(rho_lo, rho_hi, self.bin_dex)
        y_edges = _aligned_edges(T_lo,   T_hi,   self.bin_dex)

        H, _, _ = np.histogram2d(
            log_rho, log_T,
            bins=[x_edges, y_edges],
            weights=weight,
        )

        # Free the loaded slabs so the next Build task doesn't accumulate memory
        # (each covering_grid load is ~4 GB at down=1 → OOM on 16 GB Mac).
        del log_rho, log_T, weight
        p._cached_grid = None
        gc.collect()

        return {
            'H':            H,
            'x_edges':      x_edges,
            'y_edges':      y_edges,
            'tag':          self.tag,
            'weight_field': self.weight_field,
            'T_field':      self.T_field,
            'symbol':       self.symbol,
            'unit_latex':   unit_latex,
        }


# ─── Build_PhaseHistNHRho ─────────────────────────────────────────────────────
class Build_PhaseHistNHRho(BuildTask):
    """Special: mass histogram on (log N_H, log ρ).

    Uses the DESPOTIC table's 35-pt N_H grid for x bins and the same log_ρ bins
    as Build_PhaseHist for y.
    """

    def __init__(self, config, bin_dex: float = 0.2):
        super().__init__(config, name='Build_PhaseHistNHRho')
        self.bin_dex = float(bin_dex)

    def compute(self, context: PipelinePlotContext) -> dict:
        p = context.provider

        rho_u,  _ = p.get_slab_z(('gas', 'density'))
        nh_u,   _ = p.get_slab_z(('gas', 'column_density_H'))
        dx_u,   _ = p.get_slab_z(('boxlib', 'dx'))
        dy_u,   _ = p.get_slab_z(('boxlib', 'dy'))
        dz_u,   _ = p.get_slab_z(('boxlib', 'dz'))

        rho_q = rho_u.in_cgs().ravel()                                   # yt, g/cm**3
        dV_q  = (dx_u.in_cgs() * dy_u.in_cgs() * dz_u.in_cgs()).ravel()  # yt, cm**3
        rho = np.asarray(rho_q)
        nh  = np.asarray(nh_u.in_cgs()).ravel()
        del rho_u, nh_u, dx_u, dy_u, dz_u

        with np.errstate(divide='ignore', invalid='ignore'):
            log_rho = np.log10(np.where(rho > 0, rho, np.nan))
            log_nh  = np.log10(np.where(nh  > 0, nh,  np.nan))
        m_q = rho_q * dV_q                                               # yt, g
        unit_latex = _unit_latex(m_q)
        mass = m_q.value
        del rho, nh, rho_q, dV_q, m_q

        # y (ρ) edges from this task's own data range.
        rho_lo, rho_hi = float(np.nanmin(log_rho)), float(np.nanmax(log_rho))
        y_edges = _aligned_edges(rho_lo, rho_hi, self.bin_dex)

        # x (N_H) uses the DESPOTIC table's 35-pt grid.
        from ...tables.plotting import _log_edges
        table_npz = np.load(self.config.despotic_table_path, allow_pickle=True)
        x_edges = _log_edges(table_npz['col_density_values'])

        valid = (np.isfinite(log_nh) & np.isfinite(log_rho) & np.isfinite(mass))
        H, _, _ = np.histogram2d(
            log_nh[valid], log_rho[valid],
            bins=[x_edges, y_edges],
            weights=mass[valid],
        )

        # Free the loaded slabs (see Build_PhaseHist.compute() for rationale).
        del log_rho, log_nh, mass
        p._cached_grid = None
        gc.collect()

        return {
            'H':            H,
            'x_edges':      x_edges,
            'y_edges':      y_edges,
            'tag':          'NH_rho',
            'weight_field': 'mass',
            'T_field':      None,
            'symbol':       r'M_{\rm bin}',
            'unit_latex':   unit_latex,
        }
