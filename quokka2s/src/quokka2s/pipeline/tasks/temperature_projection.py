"""TemperatureProjectionTask: line-of-sight mass-weighted T (QUOKKA vs DESPOTIC).

Single row, three panels, same visual language as MultiFieldSlicesTask:

    ⟨T_QUOKKA⟩_ρ   |   ⟨T_DESPOTIC⟩_ρ   |   T_DESPOTIC / T_QUOKKA

The first two are ρ-weighted means along ``slice_axis``,
``⟨T⟩_ρ(p1, p2) = Σ_los(ρ T) / Σ_los(ρ)``, suppressing low-density hot
shells (SN bubbles) and reporting the temperature the gas mass actually
holds.  The third panel divides them:

    R(p1, p2) = ⟨T_DESPOTIC⟩_ρ / ⟨T_QUOKKA⟩_ρ

drawn with a diverging log-scaled colour map centred at R = 1 (i.e.
log10 R = 0).  Blue → DESPOTIC colder than QUOKKA (typical in hot
under-shielded gas where chemistry equilibrium pulls T down); red →
DESPOTIC hotter (rare).

The two T panels share a single log10(T) colour range pooled across both.
The ratio panel has its own diverging colour bar.

Output: temperature_projection.png
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..base import AnalysisTask, PipelinePlotContext


_CMAP_T     = 'turbo'
_CMAP_RATIO = 'RdBu_r'

_T_LABEL_QK_MW  = r'$\log_{10}\,\langle T_{\rm QUOKKA}\rangle_\rho$ [K]'
_T_LABEL_DSP_MW = r'$\log_{10}\,\langle T_{\rm DESPOTIC}\rangle_\rho$ [K]'
_T_LABEL_RATIO  = r'$\log_{10}\,(\langle T_{\rm DSP}\rangle_\rho / \langle T_{\rm QK}\rangle_\rho)$'


class TemperatureProjectionTask(AnalysisTask):
    """ρ-weighted T_QK & T_DSP projections + their ratio."""

    def __init__(self, config,
                 slice_axis: str = 'x',
                 figure_units: str = 'kpc',
                 ratio_clip_dex: float = 1.0,
                 filename: str = 'temperature_projection.png'):
        super().__init__(config)
        self.slice_axis     = slice_axis
        self.figure_units   = figure_units
        self.ratio_clip_dex = float(ratio_clip_dex)
        self.filename       = filename
        self._axis_idx      = {'x': 0, 'y': 1, 'z': 2}[slice_axis]

    def compute(self, context: PipelinePlotContext) -> dict:
        p = context.provider
        T_qk_unyt, extent_dict = p.get_slab_z(('gas', 'temperature_quokka'))
        T_md_unyt, _           = p.get_slab_z(('gas', 'temperature_despotic'))
        rho_unyt,  _           = p.get_slab_z(('gas', 'density'))
        T_qk = T_qk_unyt.to('K').value
        T_md = T_md_unyt.to('K').value
        rho  = rho_unyt.in_cgs().value
        del T_qk_unyt, T_md_unyt, rho_unyt

        # Line-of-sight ρ-weighted mean.  Uniform dx cancels in the ratio.
        mass_col  = rho.sum(axis=self._axis_idx)
        safe_mass = np.where(mass_col > 0, mass_col, 1.0)
        T_qk_mw = (rho * T_qk).sum(axis=self._axis_idx) / safe_mass
        T_md_mw = (rho * T_md).sum(axis=self._axis_idx) / safe_mass
        T_qk_mw = np.where(mass_col > 0, T_qk_mw, np.nan)
        T_md_mw = np.where(mass_col > 0, T_md_mw, np.nan)

        extent_unyt  = extent_dict[self.slice_axis]
        extent_units = [float(v.in_units(self.figure_units).value) for v in extent_unyt]

        return {
            'T_qk_mw': T_qk_mw,
            'T_md_mw': T_md_mw,
            'extent':  extent_units,
        }

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        T_qk_mw = results['T_qk_mw']
        T_md_mw = results['T_md_mw']
        extent  = results['extent']
        ext_plot = [extent[0], extent[1], extent[2], extent[3]]

        # log10 of the two T panels for shared scaling.
        with np.errstate(divide='ignore', invalid='ignore'):
            log_qk = np.where(T_qk_mw > 0, np.log10(np.where(T_qk_mw > 0, T_qk_mw, 1.0)), np.nan)
            log_md = np.where(T_md_mw > 0, np.log10(np.where(T_md_mw > 0, T_md_mw, 1.0)), np.nan)

        # log10 ratio: log_md - log_qk where both finite (and >0).
        with np.errstate(invalid='ignore'):
            log_ratio = log_md - log_qk

        finite_T = np.concatenate([log_qk[np.isfinite(log_qk)],
                                   log_md[np.isfinite(log_md)]])
        if finite_T.size == 0:
            print('TemperatureProjectionTask: no positive T, skipping')
            return
        log_vmin_T = float(np.nanpercentile(finite_T, 0.5))
        log_vmax_T = float(np.nanpercentile(finite_T, 99.5))

        # Ratio scale: symmetric around 0, capped at ratio_clip_dex.
        finite_R = log_ratio[np.isfinite(log_ratio)]
        if finite_R.size:
            r_p = max(abs(np.nanpercentile(finite_R, 1.0)),
                      abs(np.nanpercentile(finite_R, 99.0)))
            r_p = min(r_p, self.ratio_clip_dex)
        else:
            r_p = self.ratio_clip_dex
        r_lim = max(r_p, 0.1)

        n_panels = 3
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(2.4 * n_panels, 12),
            sharey=True,
            gridspec_kw={'wspace': 0.10, 'top': 0.86, 'bottom': 0.06},
        )

        panel_specs = (
            (log_qk,    _T_LABEL_QK_MW,  _CMAP_T,     log_vmin_T, log_vmax_T),
            (log_md,    _T_LABEL_DSP_MW, _CMAP_T,     log_vmin_T, log_vmax_T),
            (log_ratio, _T_LABEL_RATIO,  _CMAP_RATIO, -r_lim,     +r_lim),
        )

        for ax, (arr2d, label, cmap, vmin, vmax) in zip(axes, panel_specs):
            im = ax.imshow(
                arr2d.T,
                origin='lower', extent=ext_plot, aspect='auto',
                cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax),
            )
            ax.tick_params(axis='both', labelsize=8)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('top', size='2.5%', pad=0.55)
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=8, top=True, bottom=False,
                                labeltop=True, labelbottom=False)
            cax.set_title(f'{label}\nproj along {self.slice_axis}',
                          fontsize=9, pad=4)

        plane = {'x': ('y', 'z'), 'y': ('x', 'z'), 'z': ('x', 'y')}[self.slice_axis]
        axes[0].set_ylabel(f'{plane[1]} [{self.figure_units}]', fontsize=10)
        for ax in axes:
            ax.set_xlabel(f'{plane[0]} [{self.figure_units}]', fontsize=9)

        out = context.config.output_dir / self.filename
        fig.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
