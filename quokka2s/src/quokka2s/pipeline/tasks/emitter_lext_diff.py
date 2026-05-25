"""EmitterLextDiffTask: pixel-wise log10 surface-brightness ratios across L_ext.

Cross-run comparison.  Reads two EmitterTask intermediate HDF5 files (one
per L_ext value) from sibling output directories and plots, per (y, z)
pixel of the projection plane,

    log10( X(L_ext=b) / X(L_ext=a) )

for X ∈ {CO_sb, Cplus_sb, Halpha_sb, HI_sb, T_proj}.

Why these are interesting (compared to T_DSP, which the LextDiff plot
shows changes by ≲ 0.05 dex):

  • DESPOTIC equilibrium *T* responds weakly to A_V once we're past the
    photodissociation threshold — most slice pixels see ≲ 10% T change.
  • *Chemistry abundances* (CO especially) respond strongly: self-shielded
    CO can jump 1–2 dex in n(CO)/n(H) across the A_V ≈ 0.5–3 transition,
    and surface brightness follows.  C+ usually moves the *opposite* way
    (more shielding → more carbon locked in CO/C → less C+).
  • H_alpha and HI 21 cm don't depend directly on column density, but
    indirectly through n_e / n_HI from the DESPOTIC table; expect small
    changes there.

Output:
    {OUTPUT_ROOT}/{dataset}_down{N}_LextDiff_{a}kpc_vs_{b}kpc/
        └── emitter_lext_diff.png

Sibling to both Lext{a}kpc/ and Lext{b}kpc/ source dirs.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..base import AnalysisTask, PipelinePlotContext
from ..cache import _read_nested


_CMAP_DIFF = 'RdBu_r'

# (results-dict key, panel label).  Each panel plots
# log10( field(L_ext=b) / field(L_ext=a) ).
_PANELS: list[tuple[str, str]] = [
    ('CO',     r'$\log_{10}\,(L_{\rm CO}^{b}/L_{\rm CO}^{a})$'),
    ('Cplus',  r'$\log_{10}\,(L_{\rm C^+}^{b}/L_{\rm C^+}^{a})$'),
    ('Halpha', r'$\log_{10}\,(L_{\rm H\alpha}^{b}/L_{\rm H\alpha}^{a})$'),
    ('HI',     r'$\log_{10}\,(L_{\rm HI}^{b}/L_{\rm HI}^{a})$'),
    ('T_proj', r'$\log_{10}\,(T_{\rm proj}^{b}/T_{\rm proj}^{a})$'),
]


def _glob_one_taskcache(dirpath: Path, task_class_name: str) -> Path | None:
    cache_dir = dirpath / 'task_intermediates'
    if not cache_dir.exists():
        return None
    candidates = list(cache_dir.glob(f'{task_class_name}_*.h5'))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_results(path: Path) -> dict:
    with h5py.File(path, 'r') as f:
        return _read_nested(f)


class EmitterLextDiffTask(AnalysisTask):
    """log10 ratios of EmitterTask outputs across two L_ext runs."""

    def __init__(self, config,
                 L_ext_baseline: float = 0.0,
                 L_ext_compare:  float = 9.0,
                 ratio_clip_dex: float = 3.0,
                 figure_units:   str   = 'kpc',
                 filename:       str   = 'emitter_lext_diff.png'):
        super().__init__(config)
        self.L_ext_baseline = float(L_ext_baseline)
        self.L_ext_compare  = float(L_ext_compare)
        self.ratio_clip_dex = float(ratio_clip_dex)
        self.figure_units   = figure_units
        self.filename       = filename

    # ── path helpers (same pattern as TemperatureLextDiffTask) ─────────────
    def _sibling_dir(self, l_ext: float) -> Path:
        cur = Path(self.config.output_dir)
        parent = cur.parent
        name = cur.name
        idx = name.rfind('_Lext')
        if idx < 0:
            return parent / f'{name}_Lext{l_ext:g}kpc'
        base = name[:idx]
        tail = name[idx:]
        geom_suffix = '_sphere' if '_sphere' in tail else ''
        return parent / f'{base}_Lext{l_ext:g}kpc{geom_suffix}'

    def _diff_dir(self) -> Path:
        cur = Path(self.config.output_dir)
        parent = cur.parent
        name = cur.name
        idx = name.rfind('_Lext')
        base = name if idx < 0 else name[:idx]
        geom_suffix = ''
        if idx >= 0 and '_sphere' in name[idx:]:
            geom_suffix = '_sphere'
        return parent / (f'{base}_LextDiff_'
                         f'{self.L_ext_baseline:g}kpc_vs_'
                         f'{self.L_ext_compare:g}kpc{geom_suffix}')

    # ── compute / plot ─────────────────────────────────────────────────────
    def compute(self, context: PipelinePlotContext) -> dict:
        dir_a = self._sibling_dir(self.L_ext_baseline)
        dir_b = self._sibling_dir(self.L_ext_compare)

        path_a = _glob_one_taskcache(dir_a, 'EmitterTask')
        path_b = _glob_one_taskcache(dir_b, 'EmitterTask')
        if path_a is None or path_b is None:
            print(f'EmitterLextDiffTask: missing source(s)\n'
                  f'  L={self.L_ext_baseline}: {path_a}\n'
                  f'  L={self.L_ext_compare}:  {path_b}\n'
                  f'  → skipping.  Run the full pipeline at both L_ext values first.')
            return {'_skip': True}

        res_a = _load_results(path_a)
        res_b = _load_results(path_b)

        # Compute per-pixel log10 ratios for each panel.  Hide pixels where
        # either side is non-positive (log undefined) as NaN.
        diffs: dict[str, np.ndarray] = {}
        for key, _label in _PANELS:
            if key not in res_a or key not in res_b:
                print(f'EmitterLextDiffTask: key {key!r} missing in one of the '
                      f'two intermediates, padding with NaNs.')
                shape_guess = next(
                    (np.asarray(v).shape for v in res_a.values()
                     if hasattr(v, '__len__') and np.ndim(v) == 2),
                    (1, 1),
                )
                diffs[key] = np.full(shape_guess, np.nan)
                continue
            a = np.asarray(res_a[key])
            b = np.asarray(res_b[key])
            if a.shape != b.shape:
                print(f'EmitterLextDiffTask: shape mismatch for {key!r}: '
                      f'{a.shape} vs {b.shape}; skipping that panel.')
                diffs[key] = np.full(a.shape, np.nan)
                continue
            with np.errstate(divide='ignore', invalid='ignore'):
                both_pos = (a > 0) & (b > 0)
                d = np.where(
                    both_pos,
                    np.log10(np.where(both_pos, b, 1.0))
                    - np.log10(np.where(both_pos, a, 1.0)),
                    np.nan,
                )
            diffs[key] = d

        # Carry the projection extent forward (any source works — they share
        # the same dataset/downsample).
        extent_raw = res_a.get('extent', None)
        if extent_raw is None:
            extent_units = None
        else:
            # Cached as a list/tuple of yt-unit values; on disk they round-trip
            # as plain floats in cm.  Convert cm → figure_units locally.
            try:
                cm_to_unit = {'kpc': 3.0857e21, 'pc': 3.0857e18, 'cm': 1.0}[self.figure_units]
            except KeyError:
                cm_to_unit = 1.0
            try:
                extent_units = [float(v) / cm_to_unit for v in extent_raw]
            except Exception:
                extent_units = None

        # Per-pixel sanity stats for each species (printed for inspection).
        for key, _ in _PANELS:
            d = diffs.get(key)
            if d is None:
                continue
            finite = d[np.isfinite(d)]
            if finite.size == 0:
                print(f'[{key:>6s}]  all NaN')
                continue
            p_lo = float(np.nanpercentile(finite, 1.0))
            p_hi = float(np.nanpercentile(finite, 99.0))
            med  = float(np.nanmedian(finite))
            print(f'[{key:>6s}]  log10 diff:  '
                  f'p1 = {p_lo:+.3f}   median = {med:+.3f}   '
                  f'p99 = {p_hi:+.3f}  (dex)')

        return {
            'diffs':       diffs,
            'extent':      extent_units,
            '_skip':       False,
        }

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        if results.get('_skip'):
            return
        diffs  = results['diffs']
        extent = results.get('extent', None)

        # Determine plot extent.  Default to pixel indices if cached extent
        # could not be parsed.
        if extent is None or len(extent) != 4:
            sample = next(iter(diffs.values()))
            ny, nz = np.asarray(sample).shape
            ext_plot = [0, ny, 0, nz]
            xlabel = 'pixel'
            ylabel = 'pixel'
        else:
            ext_plot = [extent[0], extent[1], extent[2], extent[3]]
            xlabel = f'y [{self.figure_units}]'
            ylabel = f'z [{self.figure_units}]'

        n_panels = len(_PANELS)
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(2.4 * n_panels, 12),
            sharey=True,
            gridspec_kw={'wspace': 0.08, 'top': 0.86, 'bottom': 0.06},
        )

        # One common symmetric scale per panel — but each panel gets its own
        # vmax because L_CO can swing 3 dex while T_proj barely moves.
        for ax, (key, label) in zip(axes, _PANELS):
            d = np.asarray(diffs[key])
            finite = d[np.isfinite(d)]
            if finite.size:
                extreme = max(abs(np.nanpercentile(finite, 1.0)),
                              abs(np.nanpercentile(finite, 99.0)))
                r_lim = max(min(extreme, self.ratio_clip_dex), 0.02)
            else:
                r_lim = self.ratio_clip_dex
            im = ax.imshow(
                d.T,
                origin='lower', extent=ext_plot, aspect='auto',
                cmap=_CMAP_DIFF, norm=Normalize(vmin=-r_lim, vmax=+r_lim),
            )
            ax.tick_params(axis='both', labelsize=8)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('top', size='2.5%', pad=0.55)
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=8, top=True, bottom=False,
                                labeltop=True, labelbottom=False)
            cax.set_title(
                f'{label}\n'
                f'a = {self.L_ext_baseline:g} kpc,  b = {self.L_ext_compare:g} kpc',
                fontsize=8, pad=4,
            )
            ax.set_xlabel(xlabel, fontsize=9)
        axes[0].set_ylabel(ylabel, fontsize=10)

        out_dir = self._diff_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / self.filename
        fig.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
