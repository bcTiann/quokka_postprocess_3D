"""TemperatureSlicesTask: multi-slice T_QUOKKA vs T_DESPOTIC comparison.

For one snapshot, take several slices through the cube along `slice_axis`
and lay them out row-by-row. Three columns per row:
  - T_QUOKKA      (raw ideal-gas temperature, fixed μ, γ)
  - T_DESPOTIC    (μ/γ-iterated table lookup; the field other tasks use)
  - log10(T_DESPOTIC / T_QUOKKA)   (deviation, ±1 dex window)

Same shared LogNorm across columns 0 and 1 so they are directly comparable.
The ratio column uses RdBu_r at ±1 dex.

This is the pipeline analogue of the standalone script at
`compare_T_quokka_vs_despotic_slices.py`. Differences:
  * runs per snapshot automatically, with output in the snapshot's output_dir
  * reuses the shared covering grid via PipelineDataProvider — no extra IO
  * slice axis and indices are configurable

Output: temperature_slices.png
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ..base import AnalysisTask, PipelinePlotContext
from ..prep import config as cfg


class TemperatureSlicesTask(AnalysisTask):
    """Multi-slice T_QUOKKA vs T_DESPOTIC comparison for one plt snapshot."""

    def __init__(self, config,
                 slice_axis: str = 'x',
                 n_slices: int = 12,
                 cols_per_row: int = 6,
                 slice_indices: tuple[int, ...] | None = None,
                 cmap: str | None = None):
        super().__init__(config)
        self.slice_axis     = slice_axis
        self.n_slices       = int(n_slices)
        # cols_per_row controls layout wrap: 12 slices with cols_per_row=6
        # becomes 2 super-rows × 6 cols (instead of a 1×12 ribbon).
        self.cols_per_row   = int(cols_per_row)
        # When None, indices are computed at plot time from the cube shape so
        # the same default works for any downsample factor.
        self._explicit_idx  = tuple(slice_indices) if slice_indices is not None else None
        self.cmap           = cmap or cfg.CMAP

    def prepare(self, context: PipelinePlotContext) -> None:
        p = context.provider
        T_qk, extent_dict = p.get_slab_z(('gas', 'temperature_quokka'))
        T_md, _           = p.get_slab_z(('gas', 'temperature_despotic'))
        self._T_qk     = T_qk.to('K').value
        self._T_md     = T_md.to('K').value
        self._extent_dict = extent_dict

    def compute(self, context: PipelinePlotContext) -> dict:
        extent_unyt = self._extent_dict[self.slice_axis]
        extent_pc   = [float(v.in_units('pc').value) for v in extent_unyt]
        return {
            'T_qk': self._T_qk,
            'T_md': self._T_md,
            'extent_pc': extent_pc,
        }

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        T_q          = results['T_qk']
        T_d          = results['T_md']
        # extent_pc from get_slab_z is [horiz0, horiz1, vert0, vert1] for the
        # slice plane in the original orientation (vert axis is the longer one,
        # e.g. z when slicing along x). We rotate panels here so the longer
        # axis becomes horizontal — extent reorders to [vert0, vert1, horiz0, horiz1].
        ext_in       = results['extent_pc']
        extent       = [ext_in[2], ext_in[3], ext_in[0], ext_in[1]]

        pos_q = T_q[T_q > 0]
        pos_d = T_d[T_d > 0]
        if pos_q.size == 0 or pos_d.size == 0:
            print('TemperatureSlicesTask: no positive temperature cells, skipping')
            return
        Tmin = max(min(float(pos_q.min()), float(pos_d.min())), 1.0)
        Tmax = float(max(T_q.max(), T_d.max()))
        t_norm = LogNorm(vmin=Tmin, vmax=Tmax)

        horiz_label, vert_label = self._plane_axis_labels()
        # After rotation, the long axis goes horizontal — swap labels accordingly.
        plot_horiz, plot_vert = vert_label, horiz_label

        slice_indices = self._resolve_indices(T_q)
        n_slices      = len(slice_indices)
        cpr           = max(1, min(self.cols_per_row, n_slices))
        n_super_rows  = (n_slices + cpr - 1) // cpr     # ceil(n / cpr)
        n_grid_rows   = 3 * n_super_rows                # 3 metrics per super-row

        # Panel aspect is roughly z:y ≈ 4:1 → with aspect='equal' each panel
        # is width/4 tall. Per super-row needs ~1.3 in for 3 metric rows plus
        # tick/title spacing. With cpr=6 + 2 super-rows we get ~21×8 in,
        # much bigger panels than the old 1×12 strip.
        fig, axes = plt.subplots(n_grid_rows, cpr,
                                 figsize=(3.5 * cpr, 4.0 * n_super_rows),
                                 constrained_layout=True)
        # Normalise axes to a 2D ndarray even when cpr==1 or n_grid_rows==3.
        axes = np.atleast_2d(axes)
        if cpr == 1:
            axes = axes.reshape(n_grid_rows, 1)

        im_d = im_r = None

        for s_idx, idx in enumerate(slice_indices):
            super_row = s_idx // cpr
            col       = s_idx %  cpr
            r0        = 3 * super_row

            Tq_slice = self._take_slice(T_q, idx)
            Td_slice = self._take_slice(T_d, idx)
            ratio    = np.log10(np.maximum(Td_slice, 1.0) /
                                np.maximum(Tq_slice, 1.0))

            axes[r0,     col].imshow(Tq_slice, origin='lower', extent=extent,
                                     norm=t_norm, cmap=self.cmap, aspect='equal')
            im_d = axes[r0 + 1, col].imshow(Td_slice, origin='lower', extent=extent,
                                            norm=t_norm, cmap=self.cmap, aspect='equal')
            im_r = axes[r0 + 2, col].imshow(ratio,    origin='lower', extent=extent,
                                            cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

            axes[r0,     col].set_title(f'{self.slice_axis}_idx = {idx}', fontsize=11)
            axes[r0 + 2, col].set_xlabel(f'{plot_horiz} [pc]')

        # Row labels on the leftmost panel of every super-row.
        for sr in range(n_super_rows):
            r0 = 3 * sr
            axes[r0,     0].set_ylabel(f'T$_{{\\rm QUOKKA}}$\n{plot_vert} [pc]', fontsize=10)
            axes[r0 + 1, 0].set_ylabel(f'T$_{{\\rm DESPOTIC}}$\n{plot_vert} [pc]', fontsize=10)
            axes[r0 + 2, 0].set_ylabel(
                r'$\log_{10}(T_{\rm DESP}/T_{\rm QUO})$'
                f'\n{plot_vert} [pc]',
                fontsize=10,
            )

        # Hide any leftover empty subplots in the bottom super-row (when
        # n_slices is not a multiple of cpr).
        used_in_last_row = n_slices - (n_super_rows - 1) * cpr
        if used_in_last_row < cpr:
            r0 = 3 * (n_super_rows - 1)
            for c in range(used_in_last_row, cpr):
                for r in (r0, r0 + 1, r0 + 2):
                    axes[r, c].set_visible(False)

        # Colorbars: one T + one ratio per super-row, placed on the right of
        # that super-row. Sharing a single colorbar across non-contiguous
        # axes lists causes them to overlap, so we give each super-row its own.
        for sr in range(n_super_rows):
            r0 = 3 * sr
            t_axes = [axes[r0,     c] for c in range(cpr) if axes[r0,     c].get_visible()] + \
                     [axes[r0 + 1, c] for c in range(cpr) if axes[r0 + 1, c].get_visible()]
            r_axes = [axes[r0 + 2, c] for c in range(cpr) if axes[r0 + 2, c].get_visible()]
            t_label = 'T [K]'
            r_label = (r'$\log_{10}(T_{\rm DESPOTIC}/T_{\rm QUOKKA})$' '\n'
                       '(red: DESP hotter; blue: cooler)') if sr == n_super_rows - 1 else \
                       r'$\log_{10}(T_{\rm DESP}/T_{\rm QUO})$'
            fig.colorbar(im_d, ax=t_axes, label=t_label, shrink=0.85, pad=0.02)
            fig.colorbar(im_r, ax=r_axes, label=r_label, shrink=0.85, pad=0.02)

        fig.suptitle(
            r'T$_{\rm QUOKKA}$ (raw ideal gas) vs '
            r'T$_{\rm DESPOTIC}$ (table: n$_H$, N$_H$, dV/dr)'
            f'  —  slices along {self.slice_axis}-axis',
            fontsize=12,
        )

        out = self.config.output_dir / 'temperature_slices.png'
        fig.savefig(str(out), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')

    def _take_slice(self, cube: np.ndarray, idx: int) -> np.ndarray:
        if self.slice_axis == 'x':
            return cube[idx, :, :]
        if self.slice_axis == 'y':
            return cube[:, idx, :]
        return cube[:, :, idx]

    def _resolve_indices(self, cube: np.ndarray) -> tuple[int, ...]:
        """Return slice indices: explicit override if given, else `n_slices`
        evenly-spaced positions across the slice axis (skipping the edges)."""
        if self._explicit_idx is not None:
            return self._explicit_idx
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[self.slice_axis]
        n_cells  = cube.shape[axis_idx]
        # n_slices+2 points then strip first/last so we avoid the boundary cell.
        return tuple(np.linspace(0, n_cells - 1, self.n_slices + 2)
                       .astype(int)[1:-1].tolist())

    def _plane_axis_labels(self) -> tuple[str, str]:
        if self.slice_axis == 'x':
            return ('y', 'z')
        if self.slice_axis == 'y':
            return ('x', 'z')
        return ('x', 'y')
