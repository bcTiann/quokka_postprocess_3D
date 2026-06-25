from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ..base import AnalysisTask, PipelinePlotContext
from ..utils import make_axis_labels


class TemperatureCompareTask(AnalysisTask):
    """Side-by-side comparison of QUOKKA native, Eint-bisection, and μ/γ-iteration temperatures.

    Produces two figures:
      temperature_compare.png      — 2×3 projection grid
      temperature_scatter.png      — T_quokka vs T_eint scatter
      temperature_scatter_methods.png — T_eint vs T_mu_gamma scatter
    """

    def __init__(self, config, axis: str | None = None, figure_units: str | None = None):
        super().__init__(config)
        self.axis = axis or 'x'
        self.figure_units = figure_units or config.figure_units
        self.xlabel, self.ylabel = make_axis_labels(self.axis, self.figure_units)

    def prepare(self, context: PipelinePlotContext) -> None:
        provider = context.provider
        self._rho,    _ = provider.get_slab_z(('gas', 'density'))
        self._dx,     _ = provider.get_slab_z(('boxlib', 'dx'))
        self._T_qk,   _ = provider.get_slab_z(('gas', 'temperature_quokka'))
        self._T_desp, _ = provider.get_slab_z(('gas', 'temperature_eint'))
        self._T_mg,   _ = provider.get_slab_z(('gas', 'temperature_despotic'))

    def compute(self, context: PipelinePlotContext) -> dict:
        mass = self._rho * self._dx   # mass column per cell [g/cm²]
        mass_sum = np.sum(mass, axis=0)

        def mw_proj(T):
            return (np.sum(T * mass, axis=0) / mass_sum).value

        T_qk_proj   = mw_proj(self._T_qk)
        T_desp_proj = mw_proj(self._T_desp)
        T_mg_proj   = mw_proj(self._T_mg)

        ratio_desp_qk = T_desp_proj / np.where(T_qk_proj > 0, T_qk_proj, np.nan)
        ratio_mg_eint = T_mg_proj   / np.where(T_desp_proj > 0, T_desp_proj, np.nan)
        diff_mg_eint  = T_desp_proj - T_mg_proj   # Eint − μγ

        # cell-by-cell scatter (flatten, subsample)
        T_qk_flat   = self._T_qk.value.ravel()
        T_desp_flat = self._T_desp.value.ravel()
        T_mg_flat   = self._T_mg.value.ravel()
        rng = np.random.default_rng(42)
        idx = rng.choice(len(T_qk_flat), size=min(200_000, len(T_qk_flat)), replace=False)

        return {
            'T_qk_proj':      T_qk_proj,
            'T_desp_proj':    T_desp_proj,
            'T_mg_proj':      T_mg_proj,
            'ratio_desp_qk':  ratio_desp_qk,
            'ratio_mg_eint':  ratio_mg_eint,
            'diff_mg_eint':   diff_mg_eint,
            'T_qk_sample':    T_qk_flat[idx],
            'T_desp_sample':  T_desp_flat[idx],
            'T_mg_sample':    T_mg_flat[idx],
        }

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        self._plot_projections(results)
        self._plot_scatter_qk_vs_eint(results)
        self._plot_scatter_methods(results)

    def _plot_projections(self, results: dict) -> None:
        T_qk      = results['T_qk_proj']
        T_desp    = results['T_desp_proj']
        T_mg      = results['T_mg_proj']
        ratio_dq  = results['ratio_desp_qk']
        ratio_me  = results['ratio_mg_eint']
        diff      = results['diff_mg_eint']

        all_T = np.concatenate([T_qk.ravel(), T_desp.ravel(), T_mg.ravel()])
        t_min = np.nanpercentile(all_T, 1)
        t_max = np.nanpercentile(all_T, 99)
        t_norm = mcolors.LogNorm(vmin=max(t_min, 1.0), vmax=t_max)

        def ratio_norm(ratio_arr):
            r_vals = ratio_arr[np.isfinite(ratio_arr) & (ratio_arr > 0)]
            r_max  = np.nanpercentile(np.abs(np.log10(r_vals)), 98) if r_vals.size else 0.5
            r_max  = max(r_max, 0.1)
            return mcolors.LogNorm(vmin=10**(-r_max), vmax=10**r_max)

        # diff colorscale: symmetric around 0
        d_vals = diff[np.isfinite(diff)]
        d_max  = np.nanpercentile(np.abs(d_vals), 98) if d_vals.size else 1.0
        d_max  = max(d_max, 1.0)
        d_norm = mcolors.SymLogNorm(linthresh=max(d_max * 0.01, 1.0),
                                    vmin=-d_max, vmax=d_max)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        panels = [
            # row 0
            (T_qk,   t_norm,           'inferno', 'T$_{QUOKKA}$ (ideal gas) [K]'),
            (T_desp, t_norm,           'inferno', 'T$_{Eint}$ (bisection) [K]'),
            (ratio_dq, ratio_norm(ratio_dq), 'RdBu_r', 'T$_{Eint}$ / T$_{QUOKKA}$'),
            # row 1
            (T_mg,   t_norm,           'inferno', 'T$_{\\mu\\gamma}$ (μ/γ iteration) [K]'),
            (diff,   d_norm,           'RdBu_r',  'T$_{Eint}$ − T$_{\\mu\\gamma}$ [K]'),
            (ratio_me, ratio_norm(ratio_me), 'RdBu_r', 'T$_{\\mu\\gamma}$ / T$_{Eint}$'),
        ]
        titles = [
            'QUOKKA native', 'DESPOTIC Eint bisection', 'Ratio Eint/QUOKKA',
            'μ/γ iteration', 'Difference Eint − μγ', 'Ratio μγ/Eint',
        ]
        for ax, (data, norm, cmap, label), title in zip(axes.ravel(), panels, titles):
            im = ax.imshow(data.T, origin='lower', norm=norm, cmap=cmap, aspect='auto')
            fig.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
            ax.set_xlabel(self.xlabel, fontsize=10)
            ax.set_ylabel(self.ylabel, fontsize=10)
            ax.set_title(title, fontsize=11)

        fig.suptitle('Mass-weighted Temperature Projection Comparison (3 methods)', fontsize=13)
        plt.tight_layout()
        out = self.config.output_dir / 'temperature_compare.png'
        plt.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')

    def _plot_scatter_qk_vs_eint(self, results: dict) -> None:
        T_qk   = results['T_qk_sample']
        T_desp = results['T_desp_sample']

        valid = (T_qk > 0) & (T_desp > 0)
        T_qk, T_desp = T_qk[valid], T_desp[valid]

        x_edges = np.logspace(np.log10(T_qk.min()),   np.log10(T_qk.max()),   80)
        y_edges = np.logspace(np.log10(T_desp.min()), np.log10(T_desp.max()), 80)
        H, xe, ye = np.histogram2d(T_qk, T_desp, bins=[x_edges, y_edges])

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.pcolormesh(xe, ye, H.T,
                           norm=mcolors.LogNorm(vmin=1, vmax=H.max()),
                           cmap='plasma')
        fig.colorbar(im, ax=ax, label='Cell count')
        lim = (min(xe[0], ye[0]), max(xe[-1], ye[-1]))
        ax.plot(lim, lim, 'w--', lw=1.2, label='1:1')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel('T$_{QUOKKA}$ [K]', fontsize=12)
        ax.set_ylabel('T$_{Eint}$ [K]', fontsize=12)
        ax.set_title('QUOKKA vs Eint bisection\n(200k random sample)', fontsize=11)
        ax.legend(fontsize=10)
        plt.tight_layout()
        out = self.config.output_dir / 'temperature_scatter.png'
        plt.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')

    def _plot_scatter_methods(self, results: dict) -> None:
        T_eint = results['T_desp_sample']
        T_mg   = results['T_mg_sample']

        valid = (T_eint > 0) & (T_mg > 0)
        T_eint, T_mg = T_eint[valid], T_mg[valid]

        all_T = np.concatenate([T_eint, T_mg])
        edges = np.logspace(np.log10(max(all_T.min(), 1.0)), np.log10(all_T.max()), 80)
        H, xe, ye = np.histogram2d(T_eint, T_mg, bins=[edges, edges])

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.pcolormesh(xe, ye, H.T,
                           norm=mcolors.LogNorm(vmin=1, vmax=H.max()),
                           cmap='plasma')
        fig.colorbar(im, ax=ax, label='Cell count')
        lim = (edges[0], edges[-1])
        ax.plot(lim, lim, 'w--', lw=1.2, label='1:1')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel('T$_{Eint}$ bisection [K]', fontsize=12)
        ax.set_ylabel('T$_{\\mu\\gamma}$ iteration [K]', fontsize=12)
        ax.set_title('Method comparison: Eint bisection vs μ/γ iteration\n(200k random sample)', fontsize=11)
        ax.legend(fontsize=10)
        plt.tight_layout()
        out = self.config.output_dir / 'temperature_scatter_methods.png'
        plt.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
