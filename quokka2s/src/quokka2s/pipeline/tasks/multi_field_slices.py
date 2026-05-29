"""MultiFieldSlicesTask: a single slice through the box visualised in 8 panels.

For one snapshot, take one slice (default: x = mid) and plot side-by-side:
    log10 ρ, log10 T_QUOKKA, log10 T_DESPOTIC, log10 N_H,
    log10 I_CO, log10 I_{C+}, log10 I_{Hα}, log10 I_{HI}

The first three are slices through the cube; N_H is a hydrogen-column
projection; the last four are line surface brightnesses (∫ε·dl along the
slice axis, in erg s⁻¹ pc⁻²).

All fields are loaded from disk intermediates (or trivially from yt), so this
is cheap to run after the rest of the pipeline has populated the cache.
Output: multi_field_slices.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..base import AnalysisTask, PipelinePlotContext
from .temperature_lext_diff import _glob_one_taskcache, _load_results
from ..prep import config as cfg

# Averaging method used by column_density_H (for the N_H panel label).
_NH_MEAN = getattr(cfg, 'COLUMN_DENSITY_MEAN', 'harmonic')


# Constants used by the projection modes below.
_CM_PER_PC_SQ = (3.0857e18) ** 2          # cm² / pc²  (multiply g/cm² → g/pc²)
_X_H          = 0.74                      # hydrogen mass fraction
_M_H_GRAMS    = 1.6726e-24                # g per hydrogen nucleus

# (panel_key, field, label, cmap, mode, log10_vmin, log10_vmax, share_group)
#   mode='slice'              → take a slice through the cube (volumetric units kept)
#   mode='projection_erg_pc2' → sum(ε·dx)·(cm/pc)² → erg/s/pc²  (for luminosities)
#   mode='NH_cgs'             → sum(n_H·dx) → cm⁻²
#   share_group=str           → panels in the same group share vmin/vmax so they
#                               are directly visually comparable; None = independent.
_PANELS = [
    ('rho_slice',  ('gas', 'density'),              r'$\log_{10}\,\rho$ [g cm$^{-3}$]',                  'inferno', 'slice',              None, None, None),
    ('T_qk_slice', ('gas', 'temperature_quokka'),   r'$\log_{10}\,T_{\rm QUOKKA}$ [K]',                  'turbo',   'slice',              2.0, 8.0, 'T'),
    ('T_dsp_slice',('gas', 'temperature_despotic'), r'$\log_{10}\,T_{\rm DESPOTIC}$ [K]',                'turbo',   'slice',              2.0, 8.0, 'T'),
    ('NH_slice',   ('gas', 'column_density_H'),     rf'$\log_{{10}}\,N_{{\rm H}}$ (6-dir {_NH_MEAN}, +$L_{{\rm ext}}$) [cm$^{{-2}}$]', 'inferno', 'slice', None, None, None),
    ('CO_proj',    ('gas', 'CO_luminosity'),        r'$\log_{10}\,I_{\rm CO}$ [erg s$^{-1}$ pc$^{-2}$]',      'viridis', 'projection_erg_pc2', None, None, None),
    ('Cp_proj',    ('gas', 'C+_luminosity'),        r'$\log_{10}\,I_{\rm C^+}$ [erg s$^{-1}$ pc$^{-2}$]',     'viridis', 'projection_erg_pc2', None, None, None),
    ('Ha_proj',    ('gas', 'H_alpha_luminosity'),   r'$\log_{10}\,I_{\rm H\alpha}$ [erg s$^{-1}$ pc$^{-2}$]', 'viridis', 'projection_erg_pc2', None, None, None),
    ('HI_proj',    ('gas', 'HI_luminosity'),        r'$\log_{10}\,I_{\rm HI}$ [erg s$^{-1}$ pc$^{-2}$]',      'viridis', 'projection_erg_pc2', None, None, None),
]


class MultiFieldSlicesTask(AnalysisTask):
    """7-panel slice view: density, two temperatures, four luminosities."""

    def __init__(self, config,
                 slice_axis: str = 'x',
                 slice_idx: int | None = None,
                 figure_units: str = 'kpc',
                 share_lext_partners: tuple[float, ...] = (),
                 filename: str = 'multi_field_slices.png'):
        super().__init__(config)
        self.slice_axis = slice_axis           # axis perpendicular to the slice plane
        self.slice_idx  = slice_idx            # None → midplane
        self.figure_units = figure_units
        # L_ext values to pool each panel's colour range over (so the 0/9/99 kpc
        # figures share per-panel vmin/vmax and are directly comparable). Empty
        # = each figure auto-scales on its own (original behaviour).
        self.share_lext_partners = tuple(float(x) for x in share_lext_partners)
        self.filename = filename
        self._axis_idx = {'x': 0, 'y': 1, 'z': 2}[slice_axis]

    def _sibling_dir(self, l_ext: float) -> Path:
        """Sibling output dir for a given L_ext (mirrors config.OUTPUT_DIR naming)."""
        cur = Path(self.config.output_dir)
        name = cur.name
        idx = name.rfind('_Lext')
        if idx < 0:
            return cur.parent / f'{name}_Lext{l_ext:g}kpc'
        geom = '_sphere' if '_sphere' in name[idx:] else ''
        return cur.parent / f'{name[:idx]}_Lext{l_ext:g}kpc{geom}'

    def _sibling_slices(self):
        """Load the `slices` dict from each partner-L_ext MultiFieldSlicesTask
        intermediate (excluding the current run). Used only to pool colour
        ranges, so a missing sibling is just skipped (warn)."""
        if not self.share_lext_partners:
            return []
        cur = float(getattr(self.config, 'column_extension_lateral_kpc', 0.0))
        out = []
        for l_ext in self.share_lext_partners:
            if abs(l_ext - cur) < 1e-9:
                continue
            sib = self._sibling_dir(l_ext)
            path = _glob_one_taskcache(sib, 'MultiFieldSlicesTask')
            if path is None:
                print(f'  [share] sibling L_ext={l_ext:g} kpc not found at '
                      f'{sib.name} — its range is excluded (re-plot after it exists)')
                continue
            out.append(_load_results(path)['slices'])
            print(f'  [share] pooled colour range with L_ext={l_ext:g} kpc')
        return out

    def compute(self, context: PipelinePlotContext) -> dict:
        p = context.provider
        # Cell width along the slice/projection axis, in cm.  At down=1 this
        # is the same for every cell (uniform grid), so a scalar is enough.
        dx_along, _ = p.get_slab_z(('boxlib', f'd{self.slice_axis}'))
        dx_cm = float(np.asarray(dx_along.in_cgs())[0, 0, 0])
        del dx_along

        slices: dict[str, np.ndarray] = {}
        extent_dict_kpc = None
        for panel_key, field, _label, _cmap, mode, *_ in _PANELS:
            arr, extent_dict = p.get_slab_z(field)
            arr_np = arr.in_cgs().value if hasattr(arr, 'in_cgs') else np.asarray(arr)
            if extent_dict_kpc is None:
                extent_dict_kpc = {
                    k: [float(v.in_units(self.figure_units).value) for v in extent_dict[k]]
                    for k in extent_dict
                }
            if mode == 'slice':
                idx = self.slice_idx if self.slice_idx is not None else arr_np.shape[self._axis_idx] // 2
                sl  = [slice(None)] * 3
                sl[self._axis_idx] = idx
                slices[panel_key] = np.asarray(arr_np[tuple(sl)]).copy()
            elif mode == 'projection_erg_pc2':
                # ε [erg/s/cm³] · dx [cm] summed → erg/s/cm² ; × (cm/pc)² → erg/s/pc².
                proj_cgs = arr_np.sum(axis=self._axis_idx) * dx_cm
                slices[panel_key] = (proj_cgs * _CM_PER_PC_SQ).copy()
            elif mode == 'NH_cgs':
                # n_H = ρ · X_H / m_H [cm⁻³] ; ∫n_H dx → cm⁻².
                proj_cgs = arr_np.sum(axis=self._axis_idx) * dx_cm
                slices[panel_key] = (proj_cgs * _X_H / _M_H_GRAMS).copy()
            else:
                raise ValueError(f'unknown panel mode: {mode!r}')
            del arr, arr_np

        return {
            'slices':       slices,
            'extent_kpc':   extent_dict_kpc[self.slice_axis],
            'slice_axis':   self.slice_axis,
            'slice_idx':    self.slice_idx,
        }

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        slices = results['slices']
        extent = results['extent_kpc']
        # extent_dict[slice_axis] from get_slab_z is
        #   [perp1_lo, perp1_hi, perp2_lo, perp2_hi]
        # which for slice_axis='x' is [y_lo, y_hi, z_lo, z_hi].
        # Each 2D slice in `slices[...]` is shape (n_perp1, n_perp2) — for x-slice
        # that's (ny, nz).  We .T it below so vertical = the long axis (z), then
        # imshow's extent is [horizontal_lo, horizontal_hi, vertical_lo, vertical_hi]
        # = [y_lo, y_hi, z_lo, z_hi] = extent unchanged.
        ext_plot = [extent[0], extent[1], extent[2], extent[3]]

        n_panels = len(_PANELS)
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(2.4 * n_panels, 12),
            sharey=True,
            gridspec_kw={'wspace': 0.08, 'top': 0.86, 'bottom': 0.06},
        )

        # Sibling-L_ext slices (for pooling each panel's colour range so the
        # 0/9/99 kpc figures share per-panel vmin/vmax).
        sib_slices = self._sibling_slices()

        def _panel_pctl(arr2d):
            """(min, max) of log10(arr) over positive cells, or None if empty.
            Full data range (no percentile clip) per the project default."""
            pos = arr2d > 0
            if not pos.any():
                return None
            lg = np.log10(arr2d[pos])
            return float(np.nanmin(lg)), float(np.nanmax(lg))

        # Pass A: compute log10(data) + per-panel percentile range. The range is
        # POOLED across the sibling-L_ext figures so the same panel shares
        # vmin/vmax across 0/9/99 kpc.
        panel_state: dict[str, dict | None] = {}
        for panel_key, _field, _label, _cmap, _mode, _vmin, _vmax, _grp in _PANELS:
            data = slices[panel_key].T                        # vertical = long axis
            pos  = data > 0
            if not pos.any():
                panel_state[panel_key] = None
                continue
            # log10 of non-positive cells → NaN (drawn in cmap's bad-value colour).
            log_data = np.where(pos, np.log10(np.where(pos, data, 1.0)), np.nan)
            p_lo = float(np.nanmin(log_data))                 # full data range (no percentile clip)
            p_hi = float(np.nanmax(log_data))
            for sib in sib_slices:                            # pool across L_ext
                if panel_key in sib:
                    sp = _panel_pctl(np.asarray(sib[panel_key]))
                    if sp is not None:
                        p_lo = min(p_lo, sp[0]); p_hi = max(p_hi, sp[1])
            panel_state[panel_key] = {'log_data': log_data, 'p_lo': p_lo, 'p_hi': p_hi}

        # Pass B: pool ranges across panels that share a group, so e.g. the two
        # T panels get the same vmin/vmax and become directly comparable.
        group_ranges: dict[str, dict[str, float]] = {}
        for panel_key, *_, _vmin, _vmax, grp in _PANELS:
            st = panel_state.get(panel_key)
            if st is None or grp is None:
                continue
            g = group_ranges.setdefault(grp, {'lo': +np.inf, 'hi': -np.inf})
            g['lo'] = min(g['lo'], st['p_lo'])
            g['hi'] = max(g['hi'], st['p_hi'])

        # Pass C: render.  Precedence for vmin/vmax:
        #   1) explicit value in _PANELS (e.g. forcing T to [1, 7])
        #   2) shared-group pooled range
        #   3) per-panel percentile fallback
        for ax, (panel_key, _f, label, cmap, _mode, log_vmin, log_vmax, grp) in zip(axes, _PANELS):
            st = panel_state.get(panel_key)
            if st is None:
                ax.set_title(f'{label}\n(empty)', fontsize=9)
                continue
            if log_vmin is None:
                log_vmin = group_ranges[grp]['lo'] if grp in group_ranges else st['p_lo']
            if log_vmax is None:
                log_vmax = group_ranges[grp]['hi'] if grp in group_ranges else st['p_hi']

            im = ax.imshow(
                st['log_data'],
                origin='lower', extent=ext_plot, aspect='auto',
                cmap=cmap, norm=Normalize(vmin=log_vmin, vmax=log_vmax),
            )
            ax.tick_params(axis='both', labelsize=8)

            # Colourbar on top of each panel, with the field label above the
            # colourbar (away from the colourbar's tick labels below).
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('top', size='2.5%', pad=0.55)
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=8, top=True, bottom=False,
                                labeltop=True, labelbottom=False)
            cax.set_title(label, fontsize=9, pad=4)

        # Axis labels on outer axes only.
        plane = {'x': ('y', 'z'), 'y': ('x', 'z'), 'z': ('x', 'y')}[self.slice_axis]
        axes[0].set_ylabel(f'{plane[1]} [{self.figure_units}]', fontsize=10)
        for ax in axes:
            ax.set_xlabel(f'{plane[0]} [{self.figure_units}]', fontsize=9)

        import os
        ds_name = os.path.basename(str(context.config.dataset_path))
        down = getattr(context.config, 'downsample_factor', '?')
        lext = getattr(context.config, 'column_extension_lateral_kpc', 0.0)
        slice_pos = 'midplane' if self.slice_idx is None else f'index {self.slice_idx}'
        fig.suptitle(
            f'{ds_name}   (down={down},  $L_{{\\rm ext}}$ = {lext:g} kpc)\n'
            f'{plane[0]}–{plane[1]} slice at {self.slice_axis} = {slice_pos}',
            fontsize=13, y=1.0,
        )

        out = context.config.output_dir / self.filename
        fig.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
