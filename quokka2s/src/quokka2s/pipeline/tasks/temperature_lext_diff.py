"""TemperatureLextDiffTask: pixel-by-pixel  log10(T_DSP@L_b / T_DSP@L_a).

Cross-run comparison task.  Reads two TemperatureSlicesTask intermediate
HDF5 files (one per L_ext value) from sibling output directories, divides
their T_DESPOTIC slices, and plots the log10 ratio per pixel.

What the panels show (a = baseline, b = compare):

    diff(y, z) = log10( T_DSP@L_ext=b  /  T_DSP@L_ext=a )

      diff = 0          → pixel unchanged by the extension
      diff < 0  (blue)  → DESPOTIC's equilibrium T dropped under L_ext=b
                          (added column = stronger UV shielding → cooler)
      diff > 0  (red)   → DESPOTIC got hotter (shouldn't normally happen)

Output:   {OUTPUT_ROOT}/{dataset}_down{N}_LextDiff_{a}kpc_vs_{b}kpc/
              ├── temperature_lext_diff.png    (T_DSP diff, the interesting one)
              └── (a T_QK diff sanity check is printed during compute — it
                   must be exactly 0 in every pixel, since QUOKKA's T does
                   not depend on column density)

The output dir is its own sibling at the same level as
{dataset}_down{N}_Lext{a}kpc/ and {dataset}_down{N}_Lext{b}kpc/, so the
diff figures do not clutter either single-L_ext run dir.
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


_CMAP_DIFF  = 'RdBu_r'
_DIFF_LABEL = (r'$\log_{10}\,(T_{\rm DSP}^{L_{\rm ext}=b}'
               r' / T_{\rm DSP}^{L_{\rm ext}=a})$')


def _glob_one_taskcache(dirpath: Path, task_class_name: str) -> Path | None:
    """Pick the (most recently modified) TaskName_*.h5 in task_intermediates/.

    Returns None if the directory or any matching file is missing.
    """
    cache_dir = dirpath / 'task_intermediates'
    if not cache_dir.exists():
        return None
    candidates = list(cache_dir.glob(f'{task_class_name}_*.h5'))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_results(path: Path, expected_key: str | None = None) -> dict:
    """Read a task-intermediate HDF5 file.

    By default the cache key is NOT validated.  This reader is shared with the
    cross-L_ext diff tasks (``TemperatureLextDiffTask`` / ``EmitterLextDiffTask``),
    which deliberately read sibling dirs whose ``cache_key`` DIFFERS — L_ext is
    folded into the key by design, and that difference is the whole comparison.
    Blanket-guarding here would make every diff panel look "stale" and silently
    zero the comparison, so validation stays OPT-IN.

    Same-L_ext callers (the plot-time aggregators reading their own run's
    siblings) pass ``expected_key`` to opt in: a mismatch raises, so a
    schema-stale sibling is caught instead of being silently plotted.  Build the
    value with `_expected_sibling_key()`.
    """
    with h5py.File(path, 'r') as f:
        data = _read_nested(f)
        if expected_key is not None:
            stored = f.attrs.get('cache_key')
            if isinstance(stored, bytes):
                stored = stored.decode()
            if str(stored) != expected_key:
                raise ValueError(
                    f'cache_key mismatch reading {path.name}: expected '
                    f'{expected_key!r}, found {stored!r} — sibling is stale; '
                    f're-run its task before composing this plot.'
                )
    return data


def _expected_sibling_key(config, sibling_filename: str) -> str | None:
    """Level-2 cache_key a sibling intermediate in the SAME output dir should
    carry, for opt-in validation by same-L_ext readers (the aggregators).

    Returns None when L2 caching is disabled (no despotic table) — callers then
    skip validation.  Mirrors ``AnalysisTask._l2_cache_key``.
    """
    table = getattr(config, 'despotic_table_path', None)
    if table is None or not getattr(config, 'cache_enabled', True):
        return None
    from ..cache import compute_cache_key
    return compute_cache_key(
        dataset_path                 = config.dataset_path,
        despotic_table_path          = table,
        downsample_factor            = config.downsample_factor,
        column_extension_lateral_kpc = config.column_extension_lateral_kpc,
    ) + ':' + sibling_filename


class TemperatureLextDiffTask(AnalysisTask):
    """Pixel-wise log10 T_DESPOTIC ratio between two L_ext runs.

    Source: each run's `TemperatureSlicesTask` intermediate, which stores
    T_qk_slices / T_md_slices as lists of 2-D arrays at the same set of
    slice indices.  The two source runs must use the same `slice_axis` and
    `n_slices` (we sanity-check that here).
    """

    def __init__(self, config,
                 L_ext_baseline: float = 0.0,
                 L_ext_compare:  float = 9.0,
                 ratio_clip_dex: float = 1.0,
                 figure_units:   str   = 'kpc',
                 filename:       str   = 'temperature_lext_diff.png'):
        super().__init__(config)
        self.L_ext_baseline = float(L_ext_baseline)
        self.L_ext_compare  = float(L_ext_compare)
        self.ratio_clip_dex = float(ratio_clip_dex)
        self.figure_units   = figure_units
        self.filename       = filename

    # ── path helpers ────────────────────────────────────────────────────
    def _sibling_dir(self, l_ext: float) -> Path:
        """Build the sibling output dir for a given L_ext value.

        Mirrors the OUTPUT_DIR naming in `pipeline/prep/config.py`.
        Resolves relative to config.output_dir's parent.
        """
        cur = Path(self.config.output_dir)
        parent = cur.parent
        # Strip the existing  _Lext{X}kpc  suffix from the current dir name.
        name = cur.name
        # Hunt for the last "_Lext" marker and drop everything from there to
        # the (optional) trailing geometry suffix and the trailing slash.
        idx = name.rfind('_Lext')
        if idx < 0:
            # Fall back: assume current naming convention isn't being followed.
            base = name
        else:
            tail = name[idx:]
            # tail is e.g. "_Lext9kpc" or "_Lext9kpc_sphere"
            geom_suffix = ''
            if '_sphere' in tail:
                geom_suffix = '_sphere'
            base = name[:idx]
            tail = geom_suffix  # carry sphere suffix forward (or '' if LVG)
        new_tag = f'_Lext{l_ext:g}kpc'
        if idx < 0:
            return parent / f'{base}{new_tag}'
        return parent / f'{base}{new_tag}{tail}'

    def _diff_dir(self) -> Path:
        cur = Path(self.config.output_dir)
        parent = cur.parent
        name = cur.name
        idx = name.rfind('_Lext')
        base = name if idx < 0 else name[:idx]
        # carry sphere suffix
        geom_suffix = ''
        if idx >= 0 and '_sphere' in name[idx:]:
            geom_suffix = '_sphere'
        return parent / (f'{base}_LextDiff_'
                         f'{self.L_ext_baseline:g}kpc_vs_'
                         f'{self.L_ext_compare:g}kpc{geom_suffix}')

    # ── compute / plot ─────────────────────────────────────────────────
    def compute(self, context: PipelinePlotContext) -> dict:
        dir_a = self._sibling_dir(self.L_ext_baseline)
        dir_b = self._sibling_dir(self.L_ext_compare)

        path_a = _glob_one_taskcache(dir_a, 'TemperatureSlicesTask')
        path_b = _glob_one_taskcache(dir_b, 'TemperatureSlicesTask')
        if path_a is None or path_b is None:
            print(f'TemperatureLextDiffTask: missing source(s)\n'
                  f'  L={self.L_ext_baseline}: {path_a}\n'
                  f'  L={self.L_ext_compare}:  {path_b}\n'
                  f'  → skipping.  Run the full pipeline at both L_ext values first.')
            return {'_skip': True}

        res_a = _load_results(path_a)
        res_b = _load_results(path_b)

        T_qk_a = res_a['T_qk_slices']
        T_md_a = res_a['T_md_slices']
        T_qk_b = res_b['T_qk_slices']
        T_md_b = res_b['T_md_slices']
        sidx_a = np.asarray(res_a['slice_indices'])
        sidx_b = np.asarray(res_b['slice_indices'])
        extent = list(res_a['extent'])

        if not np.array_equal(sidx_a, sidx_b):
            print(f'TemperatureLextDiffTask: slice_indices mismatch\n'
                  f'  L={self.L_ext_baseline}: {sidx_a.tolist()}\n'
                  f'  L={self.L_ext_compare}:  {sidx_b.tolist()}\n'
                  f'  → skipping.  Use the same n_slices and slice_axis in both runs.')
            return {'_skip': True}
        slice_indices = sidx_a

        # _read_nested unpacks the saved list-of-arrays as a dict {0: arr, 1: arr, ...}
        # whose keys may be str.  Normalise to a true list ordered by integer key.
        def _as_list(obj) -> list[np.ndarray]:
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                items = sorted(obj.items(), key=lambda kv: int(kv[0]))
                return [np.asarray(v) for _, v in items]
            # Fallback: single ndarray with leading slice axis
            arr = np.asarray(obj)
            return list(arr)

        T_qk_a = _as_list(T_qk_a)
        T_md_a = _as_list(T_md_a)
        T_qk_b = _as_list(T_qk_b)
        T_md_b = _as_list(T_md_b)

        diff_T_md: list[np.ndarray] = []
        diff_T_qk_abs_max: list[float] = []
        for i in range(len(slice_indices)):
            a_dsp = np.asarray(T_md_a[i])
            b_dsp = np.asarray(T_md_b[i])
            a_qk  = np.asarray(T_qk_a[i])
            b_qk  = np.asarray(T_qk_b[i])

            with np.errstate(divide='ignore', invalid='ignore'):
                both_pos = (a_dsp > 0) & (b_dsp > 0)
                d = np.where(
                    both_pos,
                    np.log10(np.where(both_pos, b_dsp, 1.0))
                    - np.log10(np.where(both_pos, a_dsp, 1.0)),
                    np.nan,
                )
                qk_pos = (a_qk > 0) & (b_qk > 0)
                qk_diff = np.where(
                    qk_pos,
                    np.log10(np.where(qk_pos, b_qk, 1.0))
                    - np.log10(np.where(qk_pos, a_qk, 1.0)),
                    np.nan,
                )
            diff_T_md.append(d)
            diff_T_qk_abs_max.append(float(np.nanmax(np.abs(qk_diff))) if np.isfinite(qk_diff).any() else 0.0)

        # Sanity: QUOKKA T should NOT depend on L_ext, so its diff is 0 in every pixel.
        max_qk_drift = float(np.nanmax(diff_T_qk_abs_max)) if diff_T_qk_abs_max else 0.0
        print(f'[sanity]  max |log10 T_QK(L={self.L_ext_compare}) - log10 T_QK(L={self.L_ext_baseline})|'
              f'  =  {max_qk_drift:.2e} dex   (expected = 0)')
        if max_qk_drift > 1e-10:
            print('          ⚠ non-zero — T_QUOKKA should be identical between L_ext '
                  'values (it does not depend on column_density_H).  Investigate.')

        return {
            'diff_T_md':      diff_T_md,
            'slice_indices':  slice_indices,
            'extent':         extent,
            'max_qk_drift':   max_qk_drift,
            '_skip':          False,
        }

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        if results.get('_skip'):
            return
        diff_T_md     = results['diff_T_md']
        slice_indices = list(results['slice_indices'])
        extent        = list(results['extent'])
        ext_plot = [extent[0], extent[1], extent[2], extent[3]]

        n_slices = len(slice_indices)
        if n_slices == 0:
            print('TemperatureLextDiffTask: no slices, skipping')
            return

        # Symmetric diverging scale, capped at ratio_clip_dex.
        finite_concat = np.concatenate([
            np.asarray(d)[np.isfinite(np.asarray(d))].ravel() for d in diff_T_md
        ]) if diff_T_md else np.array([0.0])
        if finite_concat.size:
            extreme = max(abs(np.nanpercentile(finite_concat, 1.0)),
                          abs(np.nanpercentile(finite_concat, 99.0)))
            r_lim = max(min(extreme, self.ratio_clip_dex), 0.05)
        else:
            r_lim = self.ratio_clip_dex

        fig, axes = plt.subplots(
            1, n_slices,
            figsize=(2.4 * n_slices, 12),
            sharey=True,
            gridspec_kw={'wspace': 0.08, 'top': 0.86, 'bottom': 0.06},
        )
        if n_slices == 1:
            axes = np.asarray([axes])

        im = None
        for ax, d, sidx in zip(axes, diff_T_md, slice_indices):
            im = ax.imshow(
                np.asarray(d).T,
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
                f'{_DIFF_LABEL}\n'
                f'a = {self.L_ext_baseline:g} kpc,  b = {self.L_ext_compare:g} kpc, '
                f'idx = {int(sidx)}',
                fontsize=8, pad=4,
            )

        # The slice axis comes from config.projection_axis (used for slicing too
        # in TemperatureSlicesTask, which defaults to slice_axis='x').
        slice_axis = getattr(self.config, 'projection_axis', 'x')
        plane = {'x': ('y', 'z'), 'y': ('x', 'z'), 'z': ('x', 'y')}[slice_axis]
        axes[0].set_ylabel(f'{plane[1]} [{self.figure_units}]', fontsize=10)
        for ax in axes:
            ax.set_xlabel(f'{plane[0]} [{self.figure_units}]', fontsize=9)

        # Save to the dedicated diff dir (sibling to both Lext source dirs).
        out_dir = self._diff_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / self.filename
        fig.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
