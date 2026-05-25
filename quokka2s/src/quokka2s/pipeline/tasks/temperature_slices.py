"""TemperatureSlicesTask: T_QUOKKA vs T_DESPOTIC slice triplets.

Companion of TemperatureProjectionTask (line-of-sight projections live there).
Single row of  3 × n_slices  panels, grouped by slice index:

    [ T_QK@idx0, T_DSP@idx0, T_DSP/T_QK @idx0,
      T_QK@idx1, T_DSP@idx1, T_DSP/T_QK @idx1,
      ... ]

So each slice contributes its own (QK, DSP, ratio) triplet at the same
geometric position, letting you read off "who is bigger" at every slice.

All T panels share one log10(T) colour scale (pooled p0.5/p99.5 across
every T panel).  All ratio panels share one symmetric diverging colour
scale (max |log10(ratio)| pooled across every ratio panel, capped at
``ratio_clip_dex``).  Blue → DESPOTIC colder than QUOKKA, red → hotter.

Output: temperature_slices.png
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..base import AnalysisTask, PipelinePlotContext


_CMAP_T     = 'turbo'
_CMAP_RATIO = 'RdBu_r'

_T_LABEL_QK    = r'$\log_{10}\,T_{\rm QUOKKA}$ [K]'
_T_LABEL_DSP   = r'$\log_{10}\,T_{\rm DESPOTIC}$ [K]'
_T_LABEL_RATIO = r'$\log_{10}\,(T_{\rm DSP} / T_{\rm QK})$'


class TemperatureSlicesTask(AnalysisTask):
    """T_QUOKKA / T_DESPOTIC slice triplets (QK, DSP, ratio) per slice index."""

    def __init__(self, config,
                 slice_axis: str = 'x',
                 n_slices: int = 4,
                 ratio_clip_dex: float = 1.0,
                 figure_units: str = 'kpc',
                 filename: str = 'temperature_slices.png'):
        super().__init__(config)
        self.slice_axis     = slice_axis
        self.n_slices       = int(n_slices)
        self.ratio_clip_dex = float(ratio_clip_dex)
        self.figure_units   = figure_units
        self.filename       = filename
        self._axis_idx      = {'x': 0, 'y': 1, 'z': 2}[slice_axis]

    def compute(self, context: PipelinePlotContext) -> dict:
        p = context.provider
        T_qk_unyt, extent_dict = p.get_slab_z(('gas', 'temperature_quokka'))
        T_md_unyt, _           = p.get_slab_z(('gas', 'temperature_despotic'))
        T_qk = T_qk_unyt.to('K').value
        T_md = T_md_unyt.to('K').value
        del T_qk_unyt, T_md_unyt

        n_cells = T_qk.shape[self._axis_idx]
        # n_slices+2 anchors then drop endpoints to avoid boundary cells.
        slice_indices = (np.linspace(0, n_cells - 1, self.n_slices + 2)
                           .astype(int)[1:-1])

        extent_unyt  = extent_dict[self.slice_axis]
        extent_units = [float(v.in_units(self.figure_units).value) for v in extent_unyt]

        T_qk_slices: list[np.ndarray] = []
        T_md_slices: list[np.ndarray] = []
        for idx in slice_indices:
            T_qk_slices.append(np.asarray(self._take_slice(T_qk, int(idx))).copy())
            T_md_slices.append(np.asarray(self._take_slice(T_md, int(idx))).copy())

        return {
            'T_qk_slices':   T_qk_slices,
            'T_md_slices':   T_md_slices,
            'slice_indices': np.asarray(slice_indices, dtype=int),
            'extent':        extent_units,
        }

    def _take_slice(self, cube: np.ndarray, idx: int) -> np.ndarray:
        sl = [slice(None)] * 3
        sl[self._axis_idx] = int(idx)
        return cube[tuple(sl)]

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        T_qk_slices   = results['T_qk_slices']
        T_md_slices   = results['T_md_slices']
        slice_indices = list(results['slice_indices'])
        extent        = results['extent']
        ext_plot = [extent[0], extent[1], extent[2], extent[3]]

        n_slices = len(slice_indices)
        if n_slices == 0:
            print('TemperatureSlicesTask: no slices to draw, skipping')
            return

        # Pass A — build log10 panels (T and ratio) and gather their ranges.
        # T panels: list of (log_data, p_lo, p_hi, label, sidx)
        # R panels: list of (log_ratio, p_lo, p_hi, sidx)
        t_panels: list[dict | None] = []
        r_panels: list[dict | None] = []
        for i, sidx in enumerate(slice_indices):
            qk = T_qk_slices[i].T   # vertical = long axis
            ds = T_md_slices[i].T

            # T_QK panel
            pos_qk = qk > 0
            if pos_qk.any():
                log_qk = np.where(pos_qk, np.log10(np.where(pos_qk, qk, 1.0)), np.nan)
                t_panels.append({
                    'log_data': log_qk,
                    'p_lo': float(np.nanpercentile(log_qk, 0.5)),
                    'p_hi': float(np.nanpercentile(log_qk, 99.5)),
                    'label': _T_LABEL_QK,
                    'sidx':  int(sidx),
                })
            else:
                t_panels.append(None)

            # T_DSP panel
            pos_ds = ds > 0
            if pos_ds.any():
                log_ds = np.where(pos_ds, np.log10(np.where(pos_ds, ds, 1.0)), np.nan)
                t_panels.append({
                    'log_data': log_ds,
                    'p_lo': float(np.nanpercentile(log_ds, 0.5)),
                    'p_hi': float(np.nanpercentile(log_ds, 99.5)),
                    'label': _T_LABEL_DSP,
                    'sidx':  int(sidx),
                })
            else:
                t_panels.append(None)

            # ratio = log10(T_DSP / T_QK), valid only where both > 0
            both_pos = pos_qk & pos_ds
            if both_pos.any():
                with np.errstate(divide='ignore', invalid='ignore'):
                    log_r = np.where(
                        both_pos,
                        np.log10(np.where(both_pos, ds, 1.0))
                        - np.log10(np.where(both_pos, qk, 1.0)),
                        np.nan,
                    )
                finite_r = log_r[np.isfinite(log_r)]
                r_panels.append({
                    'log_data': log_r,
                    'p_lo': float(np.nanpercentile(finite_r, 1.0)) if finite_r.size else 0.0,
                    'p_hi': float(np.nanpercentile(finite_r, 99.0)) if finite_r.size else 0.0,
                    'sidx': int(sidx),
                })
            else:
                r_panels.append(None)

        valid_T = [p for p in t_panels if p is not None]
        if not valid_T:
            print('TemperatureSlicesTask: all T panels empty, skipping')
            return
        log_vmin_T = min(p['p_lo'] for p in valid_T)
        log_vmax_T = max(p['p_hi'] for p in valid_T)

        valid_R = [p for p in r_panels if p is not None]
        if valid_R:
            r_extreme = max(max(abs(p['p_lo']), abs(p['p_hi'])) for p in valid_R)
            r_lim     = max(min(r_extreme, self.ratio_clip_dex), 0.1)
        else:
            r_lim = self.ratio_clip_dex

        # Interleave for plotting:  QK_0, DSP_0, R_0, QK_1, DSP_1, R_1, ...
        plot_order: list[tuple[str, dict | None]] = []
        for i in range(n_slices):
            plot_order.append(('T', t_panels[2 * i]))      # T_QK
            plot_order.append(('T', t_panels[2 * i + 1]))  # T_DSP
            plot_order.append(('R', r_panels[i]))          # ratio

        n_panels = len(plot_order)
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(2.4 * n_panels, 12),
            sharey=True,
            gridspec_kw={'wspace': 0.08, 'top': 0.86, 'bottom': 0.06},
        )

        for ax, (kind, st) in zip(axes, plot_order):
            if st is None:
                ax.set_title('(empty)', fontsize=9)
                continue
            if kind == 'T':
                im = ax.imshow(
                    st['log_data'],
                    origin='lower', extent=ext_plot, aspect='auto',
                    cmap=_CMAP_T, norm=Normalize(vmin=log_vmin_T, vmax=log_vmax_T),
                )
                label = st['label']
            else:  # 'R'
                im = ax.imshow(
                    st['log_data'],
                    origin='lower', extent=ext_plot, aspect='auto',
                    cmap=_CMAP_RATIO, norm=Normalize(vmin=-r_lim, vmax=+r_lim),
                )
                label = _T_LABEL_RATIO

            ax.tick_params(axis='both', labelsize=8)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('top', size='2.5%', pad=0.55)
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=8, top=True, bottom=False,
                                labeltop=True, labelbottom=False)
            cax.set_title(f'{label}\n{self.slice_axis} idx = {st["sidx"]}',
                          fontsize=9, pad=4)

        plane = {'x': ('y', 'z'), 'y': ('x', 'z'), 'z': ('x', 'y')}[self.slice_axis]
        axes[0].set_ylabel(f'{plane[1]} [{self.figure_units}]', fontsize=10)
        for ax in axes:
            ax.set_xlabel(f'{plane[0]} [{self.figure_units}]', fontsize=9)

        out = context.config.output_dir / self.filename
        fig.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
