"""Multi-field slice view, split into Build + Plot tasks.

``Build_MultiFieldSlices`` (compute + store): take N slices through the box and
extract, per slice index, the 2D arrays for each panel (default 9-panel layout):
    log10 ρ, log10 T_QUOKKA, log10 T_DESPOTIC, log10 T_two-regime,
    log10 N_H, log10 ε_CO, log10 ε_{C+}, log10 ε_{Hα}, log10 ε_{HI}
All panels are volumetric slices (no along-sight projection); intermediates stay
in cgs.  Stores the per-index 2D arrays.

``Plot_MultiFieldSlices`` (plot only): renders one PNG per slice index from the
stored arrays.

Output:
    single-slice mode (slice_idx)    →  multi_field_slices.png
    multi-slice mode (slice_indices) →  <subdir>/multi_field_slices_idxNNNN.png × N
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..base import BuildTask, PlotTask, PipelinePlotContext
from ..intermediate_io import _glob_one_taskcache, _load_results
from ..prep import config as cfg

# Averaging method used by column_density_H (for the N_H panel label).
_NH_MEAN = getattr(cfg, 'COLUMN_DENSITY_MEAN', 'harmonic')


# Constants used by the projection modes below.
_CM_PER_PC_SQ = (3.0857e18) ** 2          # cm² / pc²  (multiply g/cm² → g/pc²)
_X_H          = 0.74                      # hydrogen mass fraction
_M_H_GRAMS    = 1.6726e-24                # g per hydrogen nucleus

# Per-panel display-time unit conversions.  Applied to the raw cgs field
# value before log10/imshow.  Intermediates stay in cgs (so caches don't
# need busting).  Add new entries here for any other displayed unit changes.
#   dVdr_slice:  s⁻¹  →  km/s/pc      (1 pc / 1 km in cgs ≈ 3.0857e13)
_DISPLAY_FACTOR = {
    'dVdr_slice': 3.0857e13,
}

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
    # T_two-regime added 2026-06-20: this is the unified T used by ALL hot-branch
    # luminosity fields (Hα, HI 21cm, C+ 158μm), so it's the "T that actually
    # drives the right 4 emission panels".  Shares the T colorbar group with
    # T_QK and T_DSP so they're directly comparable.
    ('T_2R_slice', ('gas', 'temperature_two_regime'), r'$\log_{10}\,T_{\rm two\text{-}regime}$ [K]',     'turbo',   'slice',              2.0, 8.0, 'T'),
    ('NH_slice',   ('gas', 'column_density_H'),     rf'$\log_{{10}}\,N_{{\rm H}}$ (6-dir {_NH_MEAN}, +$L_{{\rm ext}}$) [cm$^{{-2}}$]', 'inferno', 'slice', None, None, None),
    # 2026-06-01: right 4 panels changed from projection_erg_pc2 (sum along
    # slice axis, surface brightness erg/s/pc²) to slice mode (volumetric
    # emissivity erg/s/cm³ at the slice plane), so each slice frame actually
    # varies across slice_idx in multi-slice / MP4-sweep mode.
    ('CO_slice',   ('gas', 'CO_luminosity'),        r'$\log_{10}\,\varepsilon_{\rm CO}$ [erg s$^{-1}$ cm$^{-3}$]',      'viridis', 'slice', None, None, None),
    ('Cp_slice',   ('gas', 'C+_luminosity'),        r'$\log_{10}\,\varepsilon_{\rm C^+}$ [erg s$^{-1}$ cm$^{-3}$]',     'viridis', 'slice', None, None, None),
    ('Ha_slice',   ('gas', 'H_alpha_luminosity'),   r'$\log_{10}\,\varepsilon_{\rm H\alpha}$ [erg s$^{-1}$ cm$^{-3}$]', 'viridis', 'slice', None, None, None),
    ('HI_slice',   ('gas', 'HI_luminosity'),        r'$\log_{10}\,\varepsilon_{\rm HI}$ [erg s$^{-1}$ cm$^{-3}$]',      'viridis', 'slice', None, None, None),
]

# Preset: the 3 table input axes (n_H, N_H, dVdr) + the two temperatures.
# Useful for inspecting where each sim cell sits along the DESPOTIC table's
# input axes vs the resulting Tg.  All slices (no projections).
# Labels are kept short so they don't overflow with narrow aspect='equal' panels.
TABLE_INPUT_PANELS = [
    ('nH_slice',   ('gas', 'number_density_H'),    r'$\log_{10}\,n_{\rm H}\ [\rm cm^{-3}]$',               'inferno', 'slice', None, None, None),
    ('NH_slice',   ('gas', 'column_density_H'),    r'$\log_{10}\,N_{\rm H}\ [\rm cm^{-2}]$',               'cividis', 'slice', None, None, None),
    ('dVdr_slice', ('gas', 'dVdr_lvg'),            r'$\log_{10}\,dV/dr\ [\rm km\,s^{-1}\,pc^{-1}]$',       'plasma',  'slice', None, None, None),
    ('T_qk_slice', ('gas', 'temperature_quokka'),  r'$\log_{10}\,T_{\rm QUOKKA}\ [\rm K]$',                'turbo',   'slice', 2.0, 8.0, 'T'),
    ('T_dsp_slice',('gas', 'temperature_despotic'),r'$\log_{10}\,T_{\rm DESPOTIC}\ [\rm K]$',              'turbo',   'slice', 2.0, 8.0, 'T'),
]


def _make_init(self, config,
               slice_axis: str = 'x',
               slice_idx: int | None = None,
               slice_indices: tuple[int, ...] | None = None,
               figure_units: str = 'kpc',
               share_lext_partners: tuple[float, ...] = (),
               filename: str = 'multi_field_slices.png',
               subdir: str | None = None,
               aspect: str = 'equal',
               panels=None):
    """Shared __init__ body for Build_ and Plot_MultiFieldSlices (identical
    init args so the two are paired)."""
    self.panels = list(panels) if panels is not None else list(_PANELS)
    self.slice_axis = slice_axis           # axis perpendicular to the slice plane
    self.slice_idx  = slice_idx            # None → midplane (legacy single-slice mode)
    self.slice_indices = (tuple(int(i) for i in slice_indices)
                          if slice_indices is not None else None)
    self.figure_units = figure_units
    self.share_lext_partners = tuple(float(x) for x in share_lext_partners)
    self.filename = filename
    self.subdir = subdir
    self.aspect = aspect
    self._axis_idx = {'x': 0, 'y': 1, 'z': 2}[slice_axis]


# ─── Build ──────────────────────────────────────────────────────────────────
class Build_MultiFieldSlices(BuildTask):
    """Compute the per-slice 2D panel arrays; store them."""

    def __init__(self, config, **kwargs):
        super().__init__(config)
        _make_init(self, config, **kwargs)

    def compute(self, context: PipelinePlotContext) -> dict:
        p = context.provider
        # Cell width along the slice/projection axis, in cm.  At down=1 this
        # is the same for every cell (uniform grid), so a scalar is enough.
        dx_along, _ = p.get_slab_z(('boxlib', f'd{self.slice_axis}'))
        dx_cm = float(np.asarray(dx_along.in_cgs())[0, 0, 0])
        del dx_along

        # Determine list of slice indices to render.  Legacy single-slice
        # path: slice_indices=None → use slice_idx (or midplane).
        # Multi-slice path: slice_indices is a tuple.
        # Projections (modes != 'slice') don't depend on slice_idx and are
        # broadcast across every slice frame.
        slice_idxs: list[int] | None = (
            list(self.slice_indices) if self.slice_indices is not None else None
        )
        # slices_by_idx: idx → {panel_key: 2D array}
        slices_by_idx: dict[int, dict[str, np.ndarray]] = {}
        extent_dict_kpc = None

        for panel_key, field, _label, _cmap, mode, *_ in self.panels:
            arr, extent_dict = p.get_slab_z(field)
            arr_np = arr.in_cgs().value if hasattr(arr, 'in_cgs') else np.asarray(arr)
            if extent_dict_kpc is None:
                extent_dict_kpc = {
                    k: [float(v.in_units(self.figure_units).value) for v in extent_dict[k]]
                    for k in extent_dict
                }
                if slice_idxs is None:
                    default = (self.slice_idx if self.slice_idx is not None
                               else arr_np.shape[self._axis_idx] // 2)
                    slice_idxs = [default]
                for idx in slice_idxs:
                    slices_by_idx.setdefault(idx, {})

            if mode == 'slice':
                sl = [slice(None)] * 3
                for idx in slice_idxs:
                    sl[self._axis_idx] = idx
                    slices_by_idx[idx][panel_key] = np.asarray(arr_np[tuple(sl)]).copy()
            elif mode == 'projection_erg_pc2':
                # ε [erg/s/cm³] · dx [cm] summed → erg/s/cm² ; × (cm/pc)² → erg/s/pc².
                proj = (arr_np.sum(axis=self._axis_idx) * dx_cm * _CM_PER_PC_SQ).copy()
                for idx in slice_idxs:
                    slices_by_idx[idx][panel_key] = proj
            elif mode == 'NH_cgs':
                # n_H = ρ · X_H / m_H [cm⁻³] ; ∫n_H dx → cm⁻².
                proj = (arr_np.sum(axis=self._axis_idx) * dx_cm * _X_H / _M_H_GRAMS).copy()
                for idx in slice_idxs:
                    slices_by_idx[idx][panel_key] = proj
            else:
                raise ValueError(f'unknown panel mode: {mode!r}')
            del arr, arr_np

        # Free the provider's in-RAM covering grid (all materialised 3D fields,
        # multi-GB at down=1) before returning.  plot() needs only the small 2D
        # slices held in `slices_by_idx`.
        import gc
        p._cached_grid = None
        gc.collect()

        return {
            # Multi-slice payload.
            'slices_by_idx': slices_by_idx,
            'extent_kpc':    extent_dict_kpc[self.slice_axis],
            'slice_axis':    self.slice_axis,
            'slice_indices': slice_idxs,
            # Back-compat key for any consumer that reads the legacy single
            # slice (e.g. lext_diff tasks).  Always points at the first idx.
            'slices':        slices_by_idx[slice_idxs[0]],
            'slice_idx':     slice_idxs[0],
        }


# ─── Plot ───────────────────────────────────────────────────────────────────
class Plot_MultiFieldSlices(PlotTask):
    """Render the multi-field slice PNG(s) from Build_MultiFieldSlices."""

    def __init__(self, config, **kwargs):
        super().__init__(config)
        _make_init(self, config, **kwargs)

    def _gather_inputs(self, context: PipelinePlotContext) -> dict:
        return self._load_one(context, 'Build_MultiFieldSlices')

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
        """Load the `slices` dict from each partner-L_ext Build_MultiFieldSlices
        intermediate (excluding the current run).  Used only to pool colour
        ranges; a missing sibling is just skipped (warn).  Dormant under L15-only
        (share_lext_partners=())."""
        if not self.share_lext_partners:
            return []
        cur = float(getattr(self.config, 'column_extension_lateral_kpc', 0.0))
        out = []
        for l_ext in self.share_lext_partners:
            if abs(l_ext - cur) < 1e-9:
                continue
            sib = self._sibling_dir(l_ext)
            path = _glob_one_taskcache(sib, 'Build_MultiFieldSlices')
            if path is None:
                print(f'  [share] sibling L_ext={l_ext:g} kpc not found at '
                      f'{sib.name} — its range is excluded (re-plot after it exists)')
                continue
            out.append(_load_results(path)['slices'])
            print(f'  [share] pooled colour range with L_ext={l_ext:g} kpc')
        return out

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        slice_indices = list(results['slice_indices'])
        # When loaded from the HDF5 task-intermediate cache, dict keys come back
        # as strings (h5py group child names).  Normalise to int so the lookup
        # below works for both fresh-compute and cache-load paths.
        raw = results['slices_by_idx']
        slices_by_idx = {int(k): v for k, v in raw.items()}
        extent = results['extent_kpc']
        # If exactly one slice, write a single PNG to the legacy filename in
        # the OUTPUT_DIR root.  If multiple, write one PNG per index inside a
        # subdir (`<output_dir>/<self.subdir>/`).
        is_multi = len(slice_indices) > 1
        if is_multi:
            subdir_name = self.subdir or 'multi_field_slices_per_slice'
            multi_out_dir = context.config.output_dir / subdir_name
            multi_out_dir.mkdir(parents=True, exist_ok=True)
            print(f'  multi-slice output: {multi_out_dir}/')

        # Cross-slice colour pooling: when rendering many slice indices,
        # compute global log10 min/max per panel across ALL indices so every
        # frame in the animation shares the same colorbar (else each frame
        # auto-scales to its own data range and the animation "breathes").
        # T panels with explicit vmin/vmax in _PANELS keep their fixed range.
        cross_slice_ranges: dict[str, tuple[float, float]] = {}
        if is_multi:
            for panel_key, *_ in self.panels:
                conv = _DISPLAY_FACTOR.get(panel_key, 1.0)
                lo, hi = +np.inf, -np.inf
                for idx in slice_indices:
                    arr = slices_by_idx[int(idx)][panel_key] * conv
                    pos = arr > 0
                    if not pos.any():
                        continue
                    lg = np.log10(arr[pos])
                    lo = min(lo, float(np.nanmin(lg)))
                    hi = max(hi, float(np.nanmax(lg)))
                if np.isfinite(lo) and np.isfinite(hi):
                    cross_slice_ranges[panel_key] = (lo, hi)

        for idx in slice_indices:
            idx_int = int(idx)
            self._plot_one(context, slices_by_idx[idx_int], extent, idx_int,
                            multi_out_dir if is_multi else None,
                            cross_slice_ranges=cross_slice_ranges)

    def _plot_one(self, context, slices, extent, slice_idx, multi_out_dir,
                  cross_slice_ranges: dict[str, tuple[float, float]] | None = None):
        # extent_dict[slice_axis] from get_slab_z is
        #   [perp1_lo, perp1_hi, perp2_lo, perp2_hi]
        # which for slice_axis='x' is [y_lo, y_hi, z_lo, z_hi].
        # Each 2D slice in `slices[...]` is shape (n_perp1, n_perp2) — for x-slice
        # that's (ny, nz).  We .T it below so vertical = the long axis (z), then
        # imshow's extent is [horizontal_lo, horizontal_hi, vertical_lo, vertical_hi]
        # = [y_lo, y_hi, z_lo, z_hi] = extent unchanged.
        ext_plot = [extent[0], extent[1], extent[2], extent[3]]

        n_panels = len(self.panels)
        # Layout: aspect='equal' panels are height-constrained at 1:8 phys aspect
        # (for plt0655228 box).  Tuned 2026-06-01 for readability:
        #   - bigger figure (height 16 in)
        #   - tighter wspace (labels are now short enough)
        #   - colorbars hugged closer to panels (top=0.94, less reserve)
        per_panel_w = 2.8 if self.aspect == 'equal' else 2.4
        wspace     = 0.05 if self.aspect == 'equal' else 0.08
        top_margin = 0.93 if self.aspect == 'equal' else 0.86
        bot_margin = 0.04 if self.aspect == 'equal' else 0.06
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(per_panel_w * n_panels, 18),
            sharey=True,
            gridspec_kw={'wspace': wspace, 'top': top_margin, 'bottom': bot_margin},
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
        for panel_key, _field, _label, _cmap, _mode, _vmin, _vmax, _grp in self.panels:
            conv = _DISPLAY_FACTOR.get(panel_key, 1.0)
            data = slices[panel_key].T * conv                 # vertical = long axis; cgs → display unit
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
                    sp = _panel_pctl(np.asarray(sib[panel_key]) * conv)
                    if sp is not None:
                        p_lo = min(p_lo, sp[0]); p_hi = max(p_hi, sp[1])
            if cross_slice_ranges is not None and panel_key in cross_slice_ranges:
                cs_lo, cs_hi = cross_slice_ranges[panel_key]
                p_lo = min(p_lo, cs_lo); p_hi = max(p_hi, cs_hi)
            panel_state[panel_key] = {'log_data': log_data, 'p_lo': p_lo, 'p_hi': p_hi}

        # Pass B: pool ranges across panels that share a group, so e.g. the two
        # T panels get the same vmin/vmax and become directly comparable.
        group_ranges: dict[str, dict[str, float]] = {}
        for panel_key, *_, _vmin, _vmax, grp in self.panels:
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
        for ax, (panel_key, _f, label, cmap, _mode, log_vmin, log_vmax, grp) in zip(axes, self.panels):
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
                origin='lower', extent=ext_plot, aspect=self.aspect,
                cmap=cmap, norm=Normalize(vmin=log_vmin, vmax=log_vmax),
            )
            ax.tick_params(axis='both', labelsize=8)

            # Colourbar on top of each panel, with the field label above the
            # colourbar (away from the colourbar's tick labels below).
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('top', size='2.5%', pad=0.2)
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=7, top=True, bottom=False,
                                labeltop=True, labelbottom=False)
            # T panels: force INTEGER tick labels on the colourbar (2, 4, 6, 8).
            # MaxNLocator alone doesn't always take effect on a horizontal cbar
            # because matplotlib's colorbar replaces ticks during draw; use the
            # explicit cbar.set_ticks API which it does respect.
            if grp == 'T' and log_vmin is not None and log_vmax is not None:
                import math
                lo = int(math.ceil(log_vmin)); hi = int(math.floor(log_vmax))
                # Pick every-2 step if the range is wider than 4, else every 1.
                step = 2 if (hi - lo) > 4 else 1
                cbar.set_ticks(list(range(lo, hi + 1, step)))
            # Smaller title font for narrow equal-aspect panels (8 → 7) so
            # long labels don't spill across into neighbour axes.
            cax.set_title(label, fontsize=8 if self.aspect == 'equal' else 9, pad=2)

        # Axis labels on outer axes only.
        plane = {'x': ('y', 'z'), 'y': ('x', 'z'), 'z': ('x', 'y')}[self.slice_axis]
        axes[0].set_ylabel(f'{plane[1]} [{self.figure_units}]', fontsize=10)
        for ax in axes:
            ax.set_xlabel(f'{plane[0]} [{self.figure_units}]', fontsize=9)

        import os
        ds_name = os.path.basename(str(context.config.dataset_path))
        down = getattr(context.config, 'downsample_factor', '?')
        lext = getattr(context.config, 'column_extension_lateral_kpc', 0.0)
        slice_pos = f'index {slice_idx}'
        fig.suptitle(
            f'{ds_name}   (down={down},  $L_{{\\rm ext}}$ = {lext:g} kpc)\n'
            f'{plane[0]}–{plane[1]} slice at {self.slice_axis} = {slice_pos}',
            fontsize=13, y=0.99,
        )

        # Output path: single slice → legacy filename in OUTPUT_DIR root;
        # multi-slice → numbered file in the subdir created by plot().
        if multi_out_dir is None:
            out = context.config.output_dir / self.filename
        else:
            out = multi_out_dir / f'multi_field_slices_idx{slice_idx:04d}.png'
        fig.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out}')
