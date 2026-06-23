"""PhaseHistTask + PhaseHistNHRhoTask — single 2D histograms on the ρ-T plane.

Each task computes EXACTLY ONE 2D histogram and stores it as an intermediate.
PhaseCombinedPlotTask then reads ALL the sibling intermediates and assembles
phase_combined.png.

Replaces the old single-shot `PhaseCombinedTask` in `phase_combined.py`
(deprecated 2026-06-19, see banner there).  The old design computed 15
histograms but only used 7 — the split-task design only computes what
each registered instance asks for, and each intermediate is independently
cacheable so changing one field (e.g. C+_luminosity) doesn't bust the
others.

Usage (in `run_pipeline.py`):
    pipeline.register_task(PhaseHistTask(cfg, 'mass', 'temperature_quokka',
                                         tag='mass_T_QK', units_label=...))
    pipeline.register_task(PhaseHistTask(cfg, 'CO_luminosity',
                                         'temperature_two_regime',
                                         tag='CO_T_2R',   units_label=...))
    ...
    pipeline.register_task(PhaseCombinedPlotTask(cfg))   # reads all the above

The fixed log_ρ / log_T ranges below ensure ALL PhaseHistTask instances
produce histograms on the SAME bin grid — so PhaseCombinedPlotTask can
share x/y axes and pool colour ranges cleanly without inter-task negotiation.
"""
from __future__ import annotations

import numpy as np

from ..base import AnalysisTask, PipelinePlotContext


def _aligned_edges(lo: float, hi: float, step: float) -> np.ndarray:
    """Snap (lo, hi) to multiples of `step` and return uniform bin edges.

    Returns array of length n_bins+1 with bins of width ~`step`.
    """
    lo_snap = np.floor(lo / step) * step
    hi_snap = (np.ceil(hi / step) + 0.5) * step
    n_bins  = int(np.round((hi_snap - lo_snap) / step))
    return np.linspace(lo_snap, hi_snap, n_bins + 1)


# ─── PhaseHistTask ──────────────────────────────────────────────────────────
class PhaseHistTask(AnalysisTask):
    """Compute ONE weighted 2D histogram on (log ρ, log T).

    Parameters
    ----------
    weight_field : str
        Either ``'mass'`` (weight = ρ·dV) or a yt luminosity field name
        like ``'CO_luminosity'`` / ``'C+_luminosity'`` (weight = ε·dV).
    T_field : str
        yt field for the y axis, e.g. ``'temperature_quokka'`` /
        ``'temperature_despotic'`` / ``'temperature_two_regime'``.
    tag : str
        Short label stored in the intermediate dict.  PhaseCombinedPlotTask
        finds the right panel by this tag.  Must be unique across
        registered PhaseHistTask instances in one run.
    units_label : str, optional
        LaTeX string for the cbar title (e.g.
        ``r'$\\log_{10}\\,L_{\\rm CO}$ [erg s$^{-1}$]'``).
    bin_dex : float
        log10 bin width (default 0.2 dex).
    """

    def __init__(self, config,
                 weight_field: str, T_field: str, tag: str,
                 units_label: str = '',
                 bin_dex: float = 0.2):
        super().__init__(config, name=f'PhaseHistTask[{tag}]')
        self.weight_field = str(weight_field)
        self.T_field      = str(T_field)
        self.tag          = str(tag)
        self.units_label  = str(units_label)
        self.bin_dex      = float(bin_dex)

    def compute(self, context: PipelinePlotContext) -> dict:
        p = context.provider

        # Load required fields. yt caches these on disk, so re-running this
        # task after a field cache exists is cheap.
        rho_u, _ = p.get_slab_z(('gas', 'density'))
        T_u,   _ = p.get_slab_z(('gas', self.T_field))
        dx_u,  _ = p.get_slab_z(('boxlib', 'dx'))
        dy_u,  _ = p.get_slab_z(('boxlib', 'dy'))
        dz_u,  _ = p.get_slab_z(('boxlib', 'dz'))

        rho = np.asarray(rho_u.in_cgs()).ravel()
        T   = np.asarray(T_u.in_cgs()).ravel()
        dV  = (np.asarray(dx_u.in_cgs()) *
               np.asarray(dy_u.in_cgs()) *
               np.asarray(dz_u.in_cgs())).ravel()
        del rho_u, T_u, dx_u, dy_u, dz_u

        with np.errstate(divide='ignore', invalid='ignore'):
            log_rho = np.log10(np.where(rho > 0, rho, np.nan))
            log_T   = np.log10(np.where(T   > 0, T,   np.nan))

        if self.weight_field == 'mass':
            weight = rho * dV
        else:
            eps_u, _ = p.get_slab_z(('gas', self.weight_field))
            weight = np.asarray(eps_u.in_cgs()).ravel() * dV
            del eps_u
        del rho, T, dV

        # Bin edges from THIS task's own data range — matches the old
        # PhaseCombinedTask convention (tight to actual data).  Plot task
        # handles axis alignment across siblings via union of xlim/ylim,
        # NOT via matching bin edges (matplotlib sharex shares limits, not
        # edges; imshow uses each panel's own extent).
        rho_lo, rho_hi = float(np.nanmin(log_rho)), float(np.nanmax(log_rho))
        T_lo,   T_hi   = float(np.nanmin(log_T)),   float(np.nanmax(log_T))
        x_edges = _aligned_edges(rho_lo, rho_hi, self.bin_dex)
        y_edges = _aligned_edges(T_lo,   T_hi,   self.bin_dex)

        H, _, _ = np.histogram2d(
            log_rho, log_T,
            bins=[x_edges, y_edges],
            weights=weight,
        )

        # Free the loaded slabs so the next task doesn't accumulate memory.
        # Each covering_grid load is ~4 GB at down=1; without this, 9 tasks
        # in a row OOM-kill a 16 GB Mac silently.
        del log_rho, log_T, weight
        p._cached_grid = None
        import gc; gc.collect()

        return {
            'H':            H,
            'x_edges':      x_edges,
            'y_edges':      y_edges,
            'tag':          self.tag,
            'weight_field': self.weight_field,
            'T_field':      self.T_field,
            'units_label':  self.units_label,
        }

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        # Compute-only task — PhaseCombinedPlotTask renders the figure.
        pass


# ─── PhaseHistNHRhoTask ─────────────────────────────────────────────────────
class PhaseHistNHRhoTask(AnalysisTask):
    """Special: mass histogram on (log N_H, log ρ).

    Uses the DESPOTIC table's 35-pt N_H grid for x bins (so the panel
    aligns with `output/table_plots/<TAG>_L<L>_mass/tg_alldvdr_*.png`)
    and the same log_ρ bins as PhaseHistTask for y.
    """

    def __init__(self, config, bin_dex: float = 0.2):
        super().__init__(config, name='PhaseHistNHRhoTask')
        self.bin_dex = float(bin_dex)

    def compute(self, context: PipelinePlotContext) -> dict:
        p = context.provider

        rho_u,  _ = p.get_slab_z(('gas', 'density'))
        nh_u,   _ = p.get_slab_z(('gas', 'column_density_H'))
        dx_u,   _ = p.get_slab_z(('boxlib', 'dx'))
        dy_u,   _ = p.get_slab_z(('boxlib', 'dy'))
        dz_u,   _ = p.get_slab_z(('boxlib', 'dz'))

        rho = np.asarray(rho_u.in_cgs()).ravel()
        nh  = np.asarray(nh_u.in_cgs()).ravel()
        dV  = (np.asarray(dx_u.in_cgs()) *
               np.asarray(dy_u.in_cgs()) *
               np.asarray(dz_u.in_cgs())).ravel()
        del rho_u, nh_u, dx_u, dy_u, dz_u

        with np.errstate(divide='ignore', invalid='ignore'):
            log_rho = np.log10(np.where(rho > 0, rho, np.nan))
            log_nh  = np.log10(np.where(nh  > 0, nh,  np.nan))
        mass = rho * dV
        del rho, nh, dV

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

        # Free the loaded slabs (see PhaseHistTask.compute() for rationale).
        del log_rho, log_nh, mass
        p._cached_grid = None
        import gc; gc.collect()

        return {
            'H':            H,
            'x_edges':      x_edges,
            'y_edges':      y_edges,
            'tag':          'NH_rho',
            'weight_field': 'mass',
            'T_field':      None,
            'units_label':  r'$\log_{10}\,M_{\rm bin}$ [g]',
        }

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        # Compute-only task.
        pass
