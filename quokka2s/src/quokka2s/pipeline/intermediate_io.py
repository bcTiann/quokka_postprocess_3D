"""Shared I/O for reading task-intermediate (Level-2) result files.

These helpers were originally defined inside ``tasks/temperature_lext_diff.py``
and imported by several tasks; they now live here so the task layer doesn't
import shared plumbing from a sibling *task* module.  ``temperature_lext_diff``
re-exports the three underscore helpers for its legacy cross-L_ext diff task.

Layering (acyclic): ``cache.py`` (no intra-package module-level imports)
  ← ``intermediate_io.py`` ← ``base.py`` (PlotTask, lazy import).
"""
from __future__ import annotations

import re
from pathlib import Path

import h5py

from .cache import _read_nested, compute_cache_key


# ── h5py value/key coercion (h5py returns bytes for str attrs/group names) ───
def coerce_str(value) -> str:
    """Coerce an h5py-loaded value (possibly bytes) to a native str."""
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def coerce_key(d: dict, key: str):
    """Look up ``key`` in a dict whose keys may be bytes (h5py group names)."""
    if key in d:
        return d[key]
    for k in d.keys():
        if (isinstance(k, bytes) and k.decode() == key) or str(k) == key:
            return d[k]
    raise KeyError(key)


# ── Locating + reading task-intermediate files ───────────────────────────────
def _glob_one_taskcache(dirpath: Path, task_class_name: str) -> Path | None:
    """Pick the (most recently modified) ``{task_class_name}_*.h5`` in
    ``dirpath/task_intermediates/``.  Returns None if the dir or any match is
    missing."""
    cache_dir = Path(dirpath) / 'task_intermediates'
    if not cache_dir.exists():
        return None
    candidates = list(cache_dir.glob(f'{task_class_name}_*.h5'))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_results(path: Path, expected_key: str | None = None) -> dict:
    """Read a task-intermediate HDF5 file.

    By default the cache key is NOT validated.  This reader is shared with the
    cross-L_ext diff tasks (TemperatureLextDiff / EmitterLextDiff), which
    deliberately read sibling dirs whose ``cache_key`` DIFFERS (L_ext is folded
    into the key by design — that difference is the whole comparison).  Blanket-
    guarding here would make every diff panel look "stale" and silently zero the
    comparison, so validation stays OPT-IN.

    Same-L_ext callers (the Plot tasks reading their own run's Build results)
    pass ``expected_key`` to opt in: a mismatch raises, so a schema-stale sibling
    is caught instead of being silently plotted.  Build it with
    `_expected_sibling_key()`.
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
                    f're-run its task (--mode compute) before plotting.'
                )
    return data


def _expected_sibling_key(config, sibling_filename: str) -> str | None:
    """Level-2 cache_key a sibling intermediate in the SAME output dir should
    carry, for opt-in validation by same-L_ext readers (the Plot tasks).

    Returns None when L2 caching is disabled (no despotic table) — callers then
    skip validation.  Mirrors ``AnalysisTask._l2_cache_key``.
    """
    table = getattr(config, 'despotic_table_path', None)
    if table is None or not getattr(config, 'cache_enabled', True):
        return None
    return compute_cache_key(
        dataset_path                 = config.dataset_path,
        despotic_table_path          = table,
        downsample_factor            = config.downsample_factor,
        column_extension_lateral_kpc = config.column_extension_lateral_kpc,
    ) + ':' + sibling_filename


# ── Build-result loaders used by PlotTask ────────────────────────────────────
def _class_prefix(filename: str) -> str:
    """``Build_PhaseHist_1a2b3c4d.h5`` → ``Build_PhaseHist`` (strip ``_<8hex>.h5``)."""
    m = re.match(r'(.+)_[0-9a-f]{8}\.h5$', filename)
    return m.group(1) if m else filename


def load_one_build(output_dir, build_class_name: str, config) -> dict:
    """Load the newest ``{build_class_name}_*.h5`` from ``output_dir``, cache-key
    validated.  Raises (pointing at ``--mode compute``) if missing."""
    path = _glob_one_taskcache(Path(output_dir), build_class_name)
    if path is None:
        raise RuntimeError(
            f'{build_class_name} result not found in '
            f'{output_dir}/task_intermediates/; run it first with --mode compute.'
        )
    return _load_results(path, expected_key=_expected_sibling_key(config, path.name))


def load_all_builds(output_dir, build_class_name: str, config) -> list[dict]:
    """Load EVERY ``{build_class_name}_*.h5`` (multi-instance Build tasks, e.g.
    the 7 ``Build_PhaseHist``).  Asserts each match's class prefix equals
    ``build_class_name`` so a name that is a prefix of another (Build_PhaseHist
    vs Build_PhaseHistNHRho) cannot leak in."""
    cache_dir = Path(output_dir) / 'task_intermediates'
    out: list[dict] = []
    if not cache_dir.exists():
        return out
    for p in sorted(cache_dir.glob(f'{build_class_name}_*.h5')):
        if _class_prefix(p.name) != build_class_name:
            continue
        out.append(_load_results(p, expected_key=_expected_sibling_key(config, p.name)))
    return out
