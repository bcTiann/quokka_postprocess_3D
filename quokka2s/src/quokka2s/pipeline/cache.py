"""Disk-backed caching for the quokka2s pipeline.

Two cache layers (see plan / README):

  Level 1: derived-field cache  ── one HDF5 per expensive yt-derived field,
           hooked into ``YTDataProvider.get_slab_z`` so it is transparent
           to every task.

  Level 2: task-result cache    ── one HDF5 per ``AnalysisTask``, populated
           by ``compute()`` and consumed by ``plot()``. Lets a user iterate
           on plot styling without re-running physics.

Both layers share this module's HDF5 helpers and cache-key hashing.

Storage layout — field intermediates (shared across tasks, derived from the
snapshot + DESPOTIC table only):

    <dataset_dir>/intermediates/<dataset_basename>/fields/
        field_gas_temperature_despotic.h5
        field_gas_CO_luminosity.h5
        ...

Storage layout — task intermediates (one file per task's compute() output,
lets `--mode plot` skip re-running compute):

    <output_dir>/task_intermediates/<TaskClassName>.h5

Each HDF5 file carries a ``cache_key`` attribute. On load, we recompute
the expected key from the current snapshot/config; if it doesn't match,
the cache is considered stale and the caller must recompute.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


# ── Schema version ────────────────────────────────────────────────────────────
# Bump when the on-disk format changes OR when any cached field's *definition*
# changes (e.g. the DESPOTIC table's `tg_final` computation, the chemistry
# network, the column-density direction symmetrisation). All cached files
# stamped with an older value will be silently invalidated and rebuilt.
CACHE_SCHEMA_VERSION = 4  # 2026-05-29: bumped for solver emitter+true-3D-solve rebuild;
                          # earlier tables had a missing-emitter bug that made CNM Tg
                          # systematically too hot by 1–2.5 dex.  All caches keyed on
                          # older table mtimes are now stale.


# ── Fields worth caching to disk ─────────────────────────────────────────────
# Rule of thumb: cache every derived field whose recompute cost > 1 disk read.
# At down=1 a 3D field is ~1 GB, disk read is 1-2 sec, so anything > ~5 sec to
# recompute deserves a slot here.
#
# A field NOT in this set is computed fresh by yt every cold start. That's
# fine for trivial-cost fields (T_quokka alias, Doppler factor, freq, ρ × X_H,
# thermal-width sqrt math).
CACHED_FIELDS: frozenset[tuple[str, str]] = frozenset({
    ('gas', 'temperature_despotic'),
    ('gas', 'CO_luminosity'),
    ('gas', 'C+_luminosity'),
    ('gas', 'H_alpha_luminosity'),
    ('gas', 'HI_luminosity'),
    ('gas', 'column_density_H'),
    ('gas', 'dVdr_lvg'),
})


# ── Cache-key hashing ────────────────────────────────────────────────────────
def _file_mtime(path: str | Path) -> float:
    p = Path(path)
    return p.stat().st_mtime if p.exists() else 0.0


def compute_cache_key(
    dataset_path: str | Path,
    despotic_table_path: str | Path,
    downsample_factor: int,
    column_extension_lateral_kpc: float = 0.0,
) -> str:
    """sha1 of (snapshot + table + downsample + L_ext + schema version).

    Mutating any one of these invalidates all caches keyed by this value.
    Stored as a hex string in HDF5 ``cache_key`` attribute.

    column_extension_lateral_kpc is the lateral colDen extension length used
    by _column_density_H; folding it in lets L_ext=0 vs L_ext=9 runs keep
    independent intermediate caches.
    """
    # Pull every config setting that changes the cached field's *value* so
    # toggling any of them invalidates the cache. Lazy import keeps this
    # module free of the heavy pipeline-prep imports for non-pipeline callers.
    try:
        from .prep import config as _cfg
        _colden_mean    = getattr(_cfg, 'COLUMN_DENSITY_MEAN', 'harmonic')
        _high_t_blend   = bool(getattr(_cfg, 'HIGH_T_4D_BLEND', False))
        _t_qk_high      = float(getattr(_cfg, 'T_QK_HIGH_K', 1.0e4))
        # T_CUTOFF: dict-keyed by species; sort so order doesn't matter.
        _t_cutoff_dict  = getattr(_cfg, 'T_CUTOFF', {})
        _t_cutoff_str   = ';'.join(f'{k}={v:g}' for k, v in sorted(_t_cutoff_dict.items()))
        _t_cutoff_def   = float(getattr(_cfg, 'T_CUTOFF_DEFAULT', 2.0e7))
        _table_4d_path  = str(getattr(_cfg, 'DESPOTIC_TABLE_4D_PATH', ''))
        _table_4d_mtime = _file_mtime(_table_4d_path) if _table_4d_path else 0.0
    except Exception:
        _colden_mean = 'harmonic'
        _high_t_blend = False
        _t_qk_high = 1.0e4
        _t_cutoff_str = ''
        _t_cutoff_def = 2.0e7
        _table_4d_path = ''
        _table_4d_mtime = 0.0
    # DVDR_FLOOR is defined in physics_fields rather than config; import lazily.
    try:
        from .prep.physics_fields import DVDR_FLOOR as _dvdr_floor
    except Exception:
        _dvdr_floor = 1e-18
    h = hashlib.sha1()
    for component in (
        str(Path(dataset_path).resolve()),
        f'{_file_mtime(dataset_path):.0f}',
        str(Path(despotic_table_path).resolve()),
        f'{_file_mtime(despotic_table_path):.0f}',
        f'downsample={int(downsample_factor)}',
        f'L_ext_kpc={float(column_extension_lateral_kpc):g}',
        f'colden_mean={_colden_mean}',
        f'high_t_4d_blend={_high_t_blend}',
        f't_qk_high={_t_qk_high:g}',
        f't_cutoff={_t_cutoff_str}',
        f't_cutoff_default={_t_cutoff_def:g}',
        f'table_4d_path={_table_4d_path}',
        f'table_4d_mtime={_table_4d_mtime:.0f}',
        f'dvdr_floor={float(_dvdr_floor):g}',
        f'schema={CACHE_SCHEMA_VERSION}',
    ):
        h.update(component.encode())
        h.update(b'\x00')
    return h.hexdigest()


# ── HDF5 helpers ─────────────────────────────────────────────────────────────
def _safe_filename(field: tuple[str, str]) -> str:
    """Convert ('gas', 'CO_luminosity') → 'field_gas_CO_luminosity.h5'."""
    sanitised = '_'.join(field).replace('+', 'plus').replace('-', 'minus').replace('/', '_')
    return f'field_{sanitised}.h5'


def field_cache_path(cache_root: Path, field: tuple[str, str]) -> Path:
    return Path(cache_root) / 'fields' / _safe_filename(field)


def save_field_array(
    path: Path,
    data: np.ndarray,
    units: str,
    cache_key: str,
    field_name: tuple[str, str],
) -> None:
    """Write a 3D field array to ``path`` (one field per HDF5 file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with h5py.File(tmp_path, 'w') as f:
        dset = f.create_dataset('data', data=np.asarray(data), compression='gzip',
                                compression_opts=3, shuffle=True)
        dset.attrs['units']      = units
        dset.attrs['shape']      = np.asarray(data).shape
        f.attrs['cache_key']   = cache_key
        f.attrs['field_type']  = field_name[0]
        f.attrs['field_name']  = field_name[1]
        f.attrs['computed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        f.attrs['schema_version'] = CACHE_SCHEMA_VERSION
    os.replace(tmp_path, path)   # atomic on POSIX


def load_field_array(
    path: Path,
    expected_cache_key: str,
) -> tuple[np.ndarray, str] | None:
    """Return ``(array, units_str)`` if cache is valid, else ``None``."""
    if not path.exists():
        return None
    try:
        with h5py.File(path, 'r') as f:
            stored_key = str(f.attrs.get('cache_key', ''))
            if stored_key != expected_cache_key:
                return None
            dset = f['data']
            return np.asarray(dset[...]), str(dset.attrs.get('units', ''))
    except (OSError, KeyError):
        # corrupt or partial file — treat as miss; caller will overwrite.
        return None


# ── Nested-dict (de)serialisation for Level 2 task results ────────────────────
def save_results_dict(path: Path, results: Mapping[str, Any], cache_key: str) -> None:
    """Save an arbitrary nested dict of numpy / scalar / dict values to HDF5.

    Numbers/arrays become datasets. Nested dicts become groups. None values
    become a placeholder attribute. Lists are converted to arrays if all
    elements are numeric, else stored as JSON strings.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with h5py.File(tmp_path, 'w') as f:
        f.attrs['cache_key']      = cache_key
        f.attrs['computed_at']    = time.strftime('%Y-%m-%dT%H:%M:%S')
        f.attrs['schema_version'] = CACHE_SCHEMA_VERSION
        _write_nested(f, results)
    os.replace(tmp_path, path)


def load_results_dict(path: Path, expected_cache_key: str) -> dict | None:
    if not path.exists():
        return None
    try:
        with h5py.File(path, 'r') as f:
            stored_key = str(f.attrs.get('cache_key', ''))
            if stored_key != expected_cache_key:
                return None
            return _read_nested(f)
    except (OSError, KeyError):
        return None


def _write_nested(group: h5py.Group, data: Mapping[str, Any]) -> None:
    import json
    for key, value in data.items():
        skey = str(key)
        if value is None:
            group.attrs[f'__none__/{skey}'] = True
            continue
        if isinstance(value, Mapping):
            sub = group.create_group(skey)
            _write_nested(sub, value)
            continue
        if isinstance(value, (tuple, list)):
            try:
                arr = np.asarray(value)
                if arr.dtype.kind in {'i', 'u', 'f', 'b'}:
                    group.create_dataset(skey, data=arr)
                    continue
            except Exception:
                pass
            group.attrs[f'__json__/{skey}'] = json.dumps(list(value))
            continue
        if isinstance(value, np.ndarray):
            group.create_dataset(skey, data=value, compression='gzip',
                                 compression_opts=3, shuffle=True)
            # Try to preserve units if it's a unyt array
            units = getattr(value, 'units', None)
            if units is not None:
                group[skey].attrs['units'] = str(units)
            continue
        if isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
            group.attrs[skey] = value
            continue
        if isinstance(value, str):
            group.attrs[skey] = value
            continue
        # Fallback: try numpy conversion (might be a unyt scalar etc.)
        try:
            group.create_dataset(skey, data=np.asarray(value))
        except Exception:
            group.attrs[f'__repr__/{skey}'] = repr(value)


def _read_nested(group: h5py.Group) -> dict:
    import json
    out: dict[str, Any] = {}
    # Scan attrs first
    for attr_name, attr_value in group.attrs.items():
        if attr_name in {'cache_key', 'computed_at', 'schema_version'}:
            continue
        if attr_name.startswith('__none__/'):
            out[attr_name[len('__none__/'):]] = None
        elif attr_name.startswith('__json__/'):
            out[attr_name[len('__json__/'):]] = json.loads(attr_value)
        elif attr_name.startswith('__repr__/'):
            out[attr_name[len('__repr__/'):]] = attr_value   # opaque string
        else:
            out[attr_name] = attr_value
    # Scan datasets and sub-groups
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            out[key] = _read_nested(item)
        else:
            out[key] = np.asarray(item[...])
    return out


# ── Convenience ──────────────────────────────────────────────────────────────
def cache_root_for_dataset(dataset_path: str | Path) -> Path:
    """Standard intermediate-data directory: alongside the snapshot, namespaced
    by snapshot name.

    Example:
        dataset_path = /data/run1/plt0042
        → /data/run1/intermediates/plt0042/
    """
    p = Path(dataset_path).resolve()
    return p.parent / 'intermediates' / p.name


def clean_cache(cache_root: Path) -> None:
    """Remove a cache root and all its contents (used by --clean-cache)."""
    import shutil
    if Path(cache_root).exists():
        shutil.rmtree(cache_root)
