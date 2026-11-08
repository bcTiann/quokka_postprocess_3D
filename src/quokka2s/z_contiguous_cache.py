"""Prepare and read complete native fields with contiguous z sightlines.

Each cache contains five float64 XYZ arrays in C order. The physical axes and
native field units stay unchanged; z simply varies fastest in the saved files.
Conversion holds one complete field in RAM, writes it sequentially, then frees
it. A 512 x 512 x 4096 field requires 8 GiB; all five files occupy about40 GiB.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from .native_snapshot_reader import NATIVE_FIELD_NAMES, NativeSnapshotReader


CACHE_SCHEMA = 1
CACHE_LAYOUT = 'XYZ C order; z contiguous; physical coordinates unchanged'


def _check_source_files_unchanged(source_identity: dict) -> None:
    """Check snapshot_file_state() records without reading field payloads."""
    for record in source_identity['files']:
        path = Path(record['resolved_path'])
        state = path.stat()
        if (
            state.st_size != record['size_bytes']
            or state.st_mtime_ns != record['modified_ns']
        ):
            raise ValueError(f'Snapshot file changed during z conversion: {path}')


def z_contiguous_cache_identity(
    *,
    cache_directory: Path,
    source_identity: dict | None = None,
) -> dict:
    """Validate a completed cache and return stable file/manifest identity.

    source_identity, when supplied, is snapshot_file_state()'s files/link record.
    Only headers and file metadata are read, without repeated value validation.
    """
    cache_directory = Path(cache_directory).resolve(strict=True)
    manifest_bytes = (cache_directory / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('schema') != CACHE_SCHEMA or manifest.get('complete') is not True:
        raise ValueError(f'Incomplete or unsupported z cache: {cache_directory}')
    if manifest.get('layout') != CACHE_LAYOUT:
        raise ValueError(f'Unexpected z-cache layout: {cache_directory}')
    if source_identity is not None and manifest['source_identity'] != source_identity:
        raise ValueError(f'Z cache belongs to a different or changed snapshot: {cache_directory}')
    shape = tuple(manifest['shape'])
    if len(shape) != 3 or any(type(size) is not int or size < 1 for size in shape):
        raise ValueError(f'Invalid z-cache shape: {cache_directory}')
    if manifest['stored_dtype'] != np.dtype(np.float64).str:
        raise ValueError(f'Unexpected z-cache dtype: {cache_directory}')
    if set(manifest['fields']) != set(NATIVE_FIELD_NAMES):
        raise ValueError(f'Z cache must contain the five native fields: {cache_directory}')

    field_states = {}
    for name in NATIVE_FIELD_NAMES:
        record = manifest['fields'][name]
        filename = f'{name}.npy'
        if record['filename'] != filename:
            raise ValueError(f'Unexpected z-cache filename for {name}')
        path = (cache_directory / filename).resolve(strict=True)
        if not path.is_relative_to(cache_directory):
            raise ValueError(f'Z-cache field resolves outside its directory: {path}')
        state = path.stat()
        if (
            state.st_size != record['size_bytes']
            or state.st_mtime_ns != record['modified_ns']
        ):
            raise ValueError(f'Z-cache file changed after conversion: {path}')
        mapped = np.load(file=path, mmap_mode='r', allow_pickle=False)
        try:
            if mapped.shape != shape or mapped.dtype != np.dtype(np.float64):
                raise ValueError(f'Z-cache field has the wrong shape/dtype: {path}')
            if not mapped.flags.c_contiguous or mapped.flags.writeable:
                raise ValueError(f'Z-cache field must be read-only and z contiguous: {path}')
        finally:
            mapped._mmap.close()
        field_states[name] = {
            'filename': filename,
            'size_bytes': state.st_size,
            'modified_ns': state.st_mtime_ns,
        }
    metadata_path = cache_directory / 'snapshot_metadata.json'
    metadata_identity = None
    if metadata_path.exists():
        resolved_metadata = metadata_path.resolve(strict=True)
        if not resolved_metadata.is_relative_to(cache_directory):
            raise ValueError('Metadata sidecar must resolve within its prepared cache')
        metadata_bytes = resolved_metadata.read_bytes()
        metadata_identity = {
            'filename': 'snapshot_metadata.json',
            'size_bytes': len(metadata_bytes),
            'sha256': hashlib.sha256(metadata_bytes).hexdigest(),
        }
    return {
        'directory': str(cache_directory),
        'snapshot_metadata': metadata_identity,
        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'schema': CACHE_SCHEMA,
        'source_identity': manifest['source_identity'],
        'shape': manifest['shape'],
        'native_dtype': manifest['native_dtype'],
        'stored_dtype': manifest['stored_dtype'],
        'layout': CACHE_LAYOUT,
        'fields': field_states,
    }


def prepare_z_contiguous_cache(
    *,
    native_reader: NativeSnapshotReader,
    cache_directory: Path,
    source_identity: dict,
) -> dict:
    """Reuse a matching complete cache or build it without changing the source.

    The directory must be outside the input snapshot. Existing caches are never
    overwritten. A failed build removes only its own uniquely named temporary
    directory. The complete manifest is published with all five fields.
    Return directory, created, preprocessing_seconds and stable identity.
    """
    started = time.perf_counter()
    cache_directory = Path(cache_directory).resolve()
    header_records = [record for record in source_identity['files'] if record['path'] == 'Header']
    if len(header_records) != 1:
        raise ValueError('Source identity must describe one native snapshot Header')
    snapshot_directory = Path(header_records[0]['resolved_path']).parent
    if cache_directory.is_relative_to(snapshot_directory) or snapshot_directory.is_relative_to(cache_directory):
        raise ValueError('Z-cache directory must be separate from the input snapshot')
    _check_source_files_unchanged(source_identity=source_identity)
    if cache_directory.exists():
        identity = z_contiguous_cache_identity(
            cache_directory=cache_directory,
            source_identity=source_identity,
        )
        if tuple(identity['shape']) != native_reader._native_shape:
            raise ValueError('Z cache and native grid dimensions differ')
        if identity['native_dtype'] != native_reader._native_dtype.str:
            raise ValueError('Z cache and native field dtypes differ')
        print(f'Inputs: reusing z-contiguous fields: {cache_directory}', flush=True)
        return {
            'directory': str(cache_directory),
            'created': False,
            'preprocessing_seconds': time.perf_counter() - started,
            'identity': identity,
        }

    cache_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(tempfile.mkdtemp(
        prefix=f'.{cache_directory.name}.building-',
        dir=cache_directory.parent,
    ))
    try:
        field_records = {}
        field_count = len(NATIVE_FIELD_NAMES)
        block_count = len(native_reader.blocks)
        for field_number, name in enumerate(NATIVE_FIELD_NAMES, start=1):
            field_started = time.perf_counter()
            last_progress = field_started
            print(f'Z conversion: field {field_number}/{field_count}, {name}', flush=True)
            field_values = np.empty(
                shape=native_reader._native_shape,
                dtype=np.float64,
                order='C',
            )
            for block_number, block in enumerate(native_reader.blocks, start=1):
                selection = tuple(
                    slice(start, start + size)
                    for start, size in zip(block.start, block.shape)
                )
                source_values = native_reader.read_block_field(
                    block=block,
                    field_name=name,
                )
                field_values[selection] = source_values
                del source_values
                now = time.perf_counter()
                if now - last_progress >= 10:
                    print(f'Z conversion: {name}, blocks {block_number}/{block_count}', flush=True)
                    last_progress = now
            path = temporary_directory / f'{name}.npy'
            with path.open(mode='xb') as output:
                np.save(file=output, arr=field_values, allow_pickle=False)
                output.flush()
                os.fsync(output.fileno())
            del field_values
            state = path.stat()
            field_records[name] = {
                'filename': path.name,
                'size_bytes': state.st_size,
                'modified_ns': state.st_mtime_ns,
            }
            elapsed = time.perf_counter() - field_started
            print(f'Z conversion: {name} saved in {elapsed:.1f}s', flush=True)

        _check_source_files_unchanged(source_identity=source_identity)
        manifest = {
            'schema': CACHE_SCHEMA,
            'complete': True,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'source_identity': source_identity,
            'shape': list(native_reader._native_shape),
            'native_dtype': native_reader._native_dtype.str,
            'stored_dtype': np.dtype(np.float64).str,
            'layout': CACHE_LAYOUT,
            'fields': field_records,
        }
        with (temporary_directory / 'manifest.json').open(mode='x') as output:
            json.dump(manifest, output, indent=2, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        if cache_directory.exists():
            raise FileExistsError(f'Z-cache destination appeared during conversion: {cache_directory}')
        temporary_directory.rename(cache_directory)
    finally:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)

    identity = z_contiguous_cache_identity(
        cache_directory=cache_directory,
        source_identity=source_identity,
    )
    elapsed = time.perf_counter() - started
    print(f'Inputs: z-contiguous fields ready in {elapsed:.1f}s: {cache_directory}', flush=True)
    return {
        'directory': str(cache_directory),
        'created': True,
        'preprocessing_seconds': elapsed,
        'identity': identity,
    }


class ZContiguousReader(NativeSnapshotReader):
    """Use read-only z-contiguous files for the unchanged native slab protocol.

    Only raw-value retrieval changes. Unit conversion, periodic x/y neighbours,
    complete z columns and bounded group-cache ownership remain inherited.
    """

    def __init__(self, *, native_reader: NativeSnapshotReader, cache_directory: Path):
        if native_reader._active_slab_field_cache is not None:
            raise ValueError('Install the z reader before entering a field-cache context')
        identity = z_contiguous_cache_identity(cache_directory=cache_directory)
        if tuple(identity['shape']) != native_reader._native_shape:
            raise ValueError('Z cache and native grid dimensions differ')
        if identity['native_dtype'] != native_reader._native_dtype.str:
            raise ValueError('Z cache and native field dtypes differ')
        self.__dict__.update(native_reader.__dict__)
        self._cache_paths = {
            name: Path(identity['directory']) / f'{name}.npy'
            for name in NATIVE_FIELD_NAMES
        }

    def _copy_selected_native_fields(self, *, requested_indices):
        selection = np.ix_(*requested_indices)
        fields = {}
        for name, path in self._cache_paths.items():
            mapped = np.load(file=path, mmap_mode='r', allow_pickle=False)
            try:
                selected = np.ascontiguousarray(mapped[selection], dtype=np.float64)
                if np.shares_memory(selected, mapped):
                    raise AssertionError('Selected values must not retain the source file map')
                fields[name] = selected
            finally:
                mapped._mmap.close()
        return fields
