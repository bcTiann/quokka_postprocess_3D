"""PhaseColdenTask: (ρ, T) phase plot of mass-weighted mean column density.

PhasePlotTask reduces extensive cell quantities (mass, luminosity) by
*sum* — that is the right thing for those.  Column density is an intensive
per-cell number, so the reduction here is a mass-weighted MEAN:

    ⟨log10 N_H⟩_M  =  Σ_{i ∈ bin} (m_i × log10 N_H,i) / Σ_{i ∈ bin} m_i

Each bin therefore reports the column density actually experienced by a
representative parcel of gas mass at that (ρ, T).

Purpose: sanity check that the self-shielding pipeline is consistent.
Cold + dense cells should sit inside dense clouds and show high ⟨log10 N_H⟩;
diffuse warm/hot cells should show low ⟨log10 N_H⟩ even after the lateral
extension is added.

A horizontal red line on the colour bar marks  N_H = 1 / 4×10⁻²² cm² ≈
2.5×10²¹ cm⁻²  (the A_V = 1 threshold for cfg.A_LAMBDA_OVER_NH; gas above
that is optically thick to UV).

Output: phase_colden.png
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ..base import AnalysisTask, PipelinePlotContext
from ..prep import config as cfg


# (row_key, T-axis label, y_edges_key)
_PANELS = [
    ('T_QUOKKA',   r'$\log_{10}\,T_{\rm QUOKKA}$ [K]',   'y_qk_edges'),
    ('T_DESPOTIC', r'$\log_{10}\,T_{\rm DESPOTIC}$ [K]', 'y_dsp_edges'),
]


class PhaseColdenTask(AnalysisTask):
    """Mass-weighted ⟨log10 N_H⟩ over the (ρ, T) phase plane.

    Companion to PhasePlotTask; reduction here is a mean, not a sum.
    """

    _PHASE_BOUNDARIES_LOG_T = (np.log10(2.0e4), np.log10(1.0e6))

    def __init__(self, config,
                 bin_dex: float = 0.2,
                 filename: str = 'phase_colden.png'):
        super().__init__(config)
        self.bin_dex = float(bin_dex)
        self.filename = filename

    def compute(self, context: PipelinePlotContext) -> dict:
        p = context.provider

        rho_u,    _ = p.get_slab_z(('gas', 'density'))
        T_qk_u,   _ = p.get_slab_z(('gas', 'temperature_quokka'))
        T_dsp_u,  _ = p.get_slab_z(('gas', 'temperature_despotic'))
        colden_u, _ = p.get_slab_z(('gas', 'column_density_H'))
        dx_u,     _ = p.get_slab_z(('boxlib', 'dx'))
        dy_u,     _ = p.get_slab_z(('boxlib', 'dy'))
        dz_u,     _ = p.get_slab_z(('boxlib', 'dz'))

        rho    = np.asarray(rho_u.in_cgs()).ravel()
        T_qk   = np.asarray(T_qk_u.in_cgs()).ravel()
        T_dsp  = np.asarray(T_dsp_u.in_cgs()).ravel()
        colden = np.asarray(colden_u.in_cgs()).ravel()
        dV     = (np.asarray(dx_u.in_cgs()) *
                  np.asarray(dy_u.in_cgs()) *
                  np.asarray(dz_u.in_cgs())).ravel()
        del rho_u, T_qk_u, T_dsp_u, colden_u, dx_u, dy_u, dz_u

        with np.errstate(divide='ignore', invalid='ignore'):
            log_rho    = np.log10(np.where(rho    > 0, rho,    np.nan))
            log_T_qk   = np.log10(np.where(T_qk   > 0, T_qk,   np.nan))
            log_T_dsp  = np.log10(np.where(T_dsp  > 0, T_dsp,  np.nan))
            log_colden = np.log10(np.where(colden > 0, colden, np.nan))

        rho_lo, rho_hi = float(np.nanmin(log_rho)),   float(np.nanmax(log_rho))
        qk_lo,  qk_hi  = float(np.nanmin(log_T_qk)),  float(np.nanmax(log_T_qk))
        dsp_lo, dsp_hi = float(np.nanmin(log_T_dsp)), float(np.nanmax(log_T_dsp))
        y_lo = min(qk_lo, dsp_lo)
        y_hi = max(qk_hi, dsp_hi)

        def _aligned_edges(lo: float, hi: float, step: float) -> np.ndarray:
            lo_snap = np.floor(lo / step) * step
            hi_snap = (np.ceil(hi / step) + 0.5) * step
            n_bins = int(np.round((hi_snap - lo_snap) / step))
            return np.linspace(lo_snap, hi_snap, n_bins + 1)

        x_edges     = _aligned_edges(rho_lo, rho_hi, self.bin_dex)
        y_qk_edges  = _aligned_edges(y_lo,   y_hi,   self.bin_dex)
        y_dsp_edges = y_qk_edges.copy()

        mass = rho * dV
        del rho, dV
        # NaN-safe product: NaN log_colden → NaN contribution (won't enter bin)
        mass_x_logN = mass * log_colden

        mean_log_N: dict[str, np.ndarray] = {}
        for y_data, y_edges, key in (
            (log_T_qk,  y_qk_edges,  'T_QUOKKA'),
            (log_T_dsp, y_dsp_edges, 'T_DESPOTIC'),
        ):
            valid = (np.isfinite(log_rho) & np.isfinite(y_data)
                     & np.isfinite(log_colden) & np.isfinite(mass))
            num, _, _ = np.histogram2d(
                log_rho[valid], y_data[valid],
                bins=[x_edges, y_edges],
                weights=mass_x_logN[valid],
            )
            den, _, _ = np.histogram2d(
                log_rho[valid], y_data[valid],
                bins=[x_edges, y_edges],
                weights=mass[valid],
            )
            with np.errstate(invalid='ignore', divide='ignore'):
                mean_log_N[key] = np.where(den > 0, num / den, np.nan)

        return {
            'mean_log_N':  mean_log_N,
            'x_edges':     x_edges,
            'y_qk_edges':  y_qk_edges,
            'y_dsp_edges': y_dsp_edges,
        }

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        mean_log_N = results['mean_log_N']
        x_edges    = results['x_edges']
        y_edges = {
            'T_QUOKKA':   results['y_qk_edges'],
            'T_DESPOTIC': results['y_dsp_edges'],
        }

        # Shared cbar range = p1–p99 across both panels' finite bins.
        all_vals = np.concatenate([
            mean_log_N[k][np.isfinite(mean_log_N[k])].ravel() for k in mean_log_N
        ])
        if all_vals.size == 0:
            print('PhaseColdenTask: no valid bins, skipping')
            return
        vmin = float(np.nanpercentile(all_vals, 1.0))
        vmax = float(np.nanpercentile(all_vals, 99.0))

        fig, axes = plt.subplots(
            1, 2, figsize=(10.4, 4.4), sharex=True, sharey=True,
            gridspec_kw={'wspace': 0.06},
        )

        im = None
        for ax, (row_key, t_label, _y_key) in zip(axes, _PANELS):
            H  = mean_log_N[row_key]
            ye = y_edges[row_key]
            im = ax.imshow(
                H.T,
                origin='lower',
                extent=[x_edges[0], x_edges[-1], ye[0], ye[-1]],
                aspect='auto',
                cmap='viridis_r',
                norm=Normalize(vmin=vmin, vmax=vmax),
            )
            ax.set_xlabel(r'$\log_{10}\,\rho$ [g cm$^{-3}$]', fontsize=10)
            ax.set_title(t_label, fontsize=10)
            ax.tick_params(axis='both', labelsize=8)
            ax.set_xlim(x_edges[0], x_edges[-1])
            ax.set_ylim(ye[0], ye[-1])
            for log_T_bnd in self._PHASE_BOUNDARIES_LOG_T:
                if ye[0] <= log_T_bnd <= ye[-1]:
                    ax.axhline(log_T_bnd, color='gray', ls='--', lw=0.6, alpha=0.7)
        axes[0].set_ylabel(r'$\log_{10}\,T$ [K]', fontsize=10)

        cbar = fig.colorbar(
            im, ax=axes.tolist(),
            orientation='vertical', fraction=0.03, pad=0.02,
        )
        cbar.set_label(r'$\langle\log_{10}\,N_{\rm H}\rangle_M$  [cm$^{-2}$]',
                       fontsize=10)
        cbar.ax.tick_params(labelsize=8)

        out = context.config.output_dir / self.filename
        fig.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
