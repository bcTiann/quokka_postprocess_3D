"""Save and restore native snapshot metadata without caching any cell values.

Create the sidecar while z conversion already owns the yt Snapshot. The timed
process rebuilds unit-aware geometry and native block descriptors from JSON,
then performs every actual field read and physical derivation as before.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from unyt import unyt_array, unyt_quantity
from unyt.unit_object import Unit
from unyt.unit_registry import UnitRegistry

from .native_snapshot_reader import NativeBlock, NativeSnapshotReader, NATIVE_FIELD_NAMES
from .process_reuse import snapshot_file_state
from .snapshot_reader import Snapshot


SCHEMA = 1
KIND = 'uniform-level-zero-native-snapshot-metadata-only'


def unit_array_record(values):
    """Store exact metadata values in their original units, without conversion."""
    array = np.asarray(values.value)
    return {'values': array.tolist(), 'dtype': array.dtype.str, 'shape': list(array.shape),
            'units': str(values.units)}


def restore_unit_array(record, registry):
    """Recreate a 3-vector geometry or scalar time using the original registry."""
    values = np.asarray(record['values'], dtype=np.dtype(record['dtype']))
    if list(values.shape) != record['shape'] or values.size > 3 or not np.isfinite(values).all():
        raise ValueError('Invalid metadata geometry/time values')
    constructor = unyt_quantity if values.shape == () else unyt_array
    return constructor(values, record['units'], registry=registry)


def write_snapshot_metadata(*, snapshot, dataset_path, metadata_path, source_identity=None):
    """Write metadata from an already opened Snapshot; read no field payloads.

    `snapshot` is the object already used by z conversion, including its cached
    native block descriptors. The file is separate from the protected dataset.
    Source-file identity is checked before and after collecting metadata.
    """
    started = time.perf_counter()
    dataset_path = Path(dataset_path).resolve(strict=True)
    metadata_path = Path(metadata_path).resolve()
    if metadata_path.is_relative_to(dataset_path):
        raise ValueError('Metadata sidecar must be outside the read-only snapshot')
    if metadata_path.exists():
        raise FileExistsError('Preserve the existing metadata sidecar')
    before = snapshot_file_state(dataset=dataset_path)
    if source_identity is not None and before != source_identity:
        raise ValueError('Original snapshot changed before metadata preparation')
    native = snapshot.native_reader
    dataset = snapshot.dataset
    if dataset.dataset_type != 'boxlib_native' or dataset.dimensionality != 3 or dataset.max_level != 0:
        raise ValueError('Require the original 3D uniform level-zero snapshot')
    source_files = {record['resolved_path']: record for record in before['files']}
    blocks = []
    for block in native.blocks:
        record = asdict(block)
        record['filename'] = str(Path(block.filename).resolve(strict=True))
        if record['filename'] not in source_files:
            raise ValueError('Native block does not belong to the recorded source')
        blocks.append(record)
    record = {
        'schema': SCHEMA,
        'kind': KIND,
        'complete': True,
        'dataset': str(dataset_path),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_identity': before,
        'domain_dimensions': np.asarray(dataset.domain_dimensions).tolist(),
        'domain_dimensions_dtype': np.asarray(dataset.domain_dimensions).dtype.str,
        'domain_left_edge': unit_array_record(dataset.domain_left_edge),
        'domain_right_edge': unit_array_record(dataset.domain_right_edge),
        'domain_width': unit_array_record(dataset.domain_width),
        'current_time': unit_array_record(dataset.current_time),
        'unit_registry_json': dataset.unit_registry.to_json(),
        'unit_system_name': dataset.unit_system.name,
        'native_dtype': native._native_dtype.str,
        'field_positions': dict(native._field_positions),
        'native_units': {name: str(unit) for name, unit in native._native_units.items()},
        'gas_units': {name: str(unit) for name, unit in native._gas_units.items()},
        'blocks': blocks,
        'contains_cell_payloads_or_derived_fields': False,
    }
    if snapshot_file_state(dataset=dataset_path) != before:
        raise ValueError('Original snapshot changed during metadata preparation')
    payload = (json.dumps(record, indent=2, allow_nan=False) + '\n').encode()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open('xb') as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    return {'path': str(metadata_path), 'bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest(),
            'preparation_seconds': time.perf_counter() - started,
            'metadata_only': True}


class MetadataDataset:
    """The geometry/unit protocol consumed by Snapshot and native readers.

    This is not a yt dataset and intentionally has no field/index API. The
    attached NativeSnapshotReader uses its restored block descriptors directly.
    `arr` reproduces yt's unyt construction with the same serialized registry.
    """

    def __init__(self, record):
        self.unit_registry = UnitRegistry.from_json(record['unit_registry_json'])
        self.domain_dimensions = np.asarray(record['domain_dimensions'], dtype=record['domain_dimensions_dtype'])
        if (
            self.domain_dimensions.shape != (3,)
            or self.domain_dimensions.dtype.kind not in 'iu'
            or np.any(self.domain_dimensions < 1)
        ):
            raise ValueError('Invalid native snapshot dimensions')
        self.domain_dimensions.flags.writeable = False
        for name in ('domain_left_edge', 'domain_right_edge', 'domain_width', 'current_time'):
            values = restore_unit_array(record[name], self.unit_registry)
            values.flags.writeable = False
            setattr(self, name, values)
        if self.domain_width.shape != (3,) or np.any(self.domain_width.to('cm').value <= 0):
            raise ValueError('Native domain widths must be positive lengths')
        if self.domain_left_edge.shape != (3,) or self.domain_right_edge.shape != (3,):
            raise ValueError('Native domain edges must be three-vectors')
        if self.current_time.shape != ():
            raise ValueError('Native simulation time must be scalar')
        self.current_time.to('s')
        self.dataset_type = 'boxlib_native'
        self.dimensionality = 3
        self.max_level = 0
        self.parameter_filename = record['dataset']
        self.unit_system_name = record['unit_system_name']

    def arr(self, input_array, units=None, dtype=None):
        return unyt_array(input_array, units, registry=self.unit_registry, dtype=dtype)

    def quan(self, value, units=None, dtype=None):
        return unyt_quantity(value, units, registry=self.unit_registry, dtype=dtype)


def restore_native_reader(*, dataset, record):
    """Restore only the metadata that the original native constructor caches."""
    reader = NativeSnapshotReader.__new__(NativeSnapshotReader)
    reader._dataset = dataset
    reader._native_shape = tuple(int(value) for value in dataset.domain_dimensions)
    reader._cell_widths = dataset.domain_width / dataset.domain_dimensions
    reader._z_indices = np.arange(reader._native_shape[2])
    reader._active_slab_field_cache = None
    reader._native_dtype = np.dtype(record['native_dtype'])
    if reader._native_dtype.kind != 'f' or reader._native_dtype.itemsize not in (4, 8):
        raise ValueError('Invalid native floating-point dtype')
    if set(record['field_positions']) != set(NATIVE_FIELD_NAMES) or set(record['native_units']) != set(NATIVE_FIELD_NAMES):
        raise ValueError('Sidecar must describe the five original native fields')
    reader._field_positions = dict(record['field_positions'])
    if any(type(position) is not int or position < 0 for position in reader._field_positions.values()):
        raise ValueError('Invalid native field positions')
    reader._native_units = dict(record['native_units'])
    reader._gas_units = dict(record['gas_units'])
    if set(reader._gas_units) != {'density', 'velocity_x', 'velocity_y', 'velocity_z'}:
        raise ValueError('Invalid declared gas units')
    source_files = {item['resolved_path']: item for item in record['source_identity']['files']}
    blocks = []
    covered_cells = 0
    largest_field_position = max(reader._field_positions.values())
    for item in record['blocks']:
        start = tuple(item['start'])
        shape = tuple(item['shape'])
        if len(start) != 3 or len(shape) != 3:
            raise ValueError('Native blocks must be three-dimensional')
        if any(type(value) is not int for value in (*start, *shape)):
            raise ValueError('Native block bounds must be integers')
        if any(lower < 0 or size < 1 or lower + size > total
               for lower, size, total in zip(start, shape, reader._native_shape)):
            raise ValueError('Native block lies outside original grid')
        cell_count = int(np.prod(shape))
        if item['field_bytes'] != cell_count * reader._native_dtype.itemsize:
            raise ValueError('Native block field byte count differs from geometry')
        if item['filename'] not in source_files or item['payload_offset'] < 0:
            raise ValueError('Native block does not match source-file identity')
        required_bytes = item['payload_offset'] + (largest_field_position + 1) * item['field_bytes']
        if required_bytes > source_files[item['filename']]['size_bytes']:
            raise ValueError('Native block extends beyond its recorded source file')
        blocks.append(NativeBlock(filename=item['filename'], start=start, shape=shape,
                                  payload_offset=item['payload_offset'], field_bytes=item['field_bytes']))
        covered_cells += cell_count
    if covered_cells != int(np.prod(reader._native_shape)):
        raise ValueError('Native blocks must cover the complete original grid')
    reader._blocks = tuple(blocks)
    # Validate the same unit dimensions used by actual slab reads. Physical
    # table-domain validation remains in load_emission_calculator unchanged.
    density_unit = Unit(reader._native_units['gasDensity'], registry=dataset.unit_registry)
    gas_density = Unit(reader._gas_units['density'], registry=dataset.unit_registry)
    density_unit.get_conversion_factor(gas_density)
    gas_density.get_conversion_factor(Unit('g/cm**3', registry=dataset.unit_registry))
    for axis in ('x', 'y', 'z'):
        momentum = dataset.arr([1.0], reader._native_units[f'{axis}-GasMomentum'])
        density = dataset.arr([1.0], reader._native_units['gasDensity'])
        (momentum / density).to(reader._gas_units[f'velocity_{axis}']).to('cm/s')
    return reader


def open_snapshot_metadata(*, dataset_path, metadata_path, xy_region=None):
    """Read/validate the sidecar and source identity inside process time."""
    dataset_path = Path(dataset_path).resolve(strict=True)
    payload = Path(metadata_path).read_bytes()
    record = json.loads(payload)
    if record.get('schema') != SCHEMA or record.get('kind') != KIND or record.get('complete') is not True:
        raise ValueError('Unsupported or incomplete snapshot metadata sidecar')
    if record.get('contains_cell_payloads_or_derived_fields') is not False:
        raise ValueError('Sidecar must contain metadata only')
    if record['dataset'] != str(dataset_path):
        raise ValueError('Sidecar belongs to a different snapshot')
    if snapshot_file_state(dataset=dataset_path) != record['source_identity']:
        raise ValueError('Snapshot changed since metadata preparation')
    dataset = MetadataDataset(record)
    reader = restore_native_reader(dataset=dataset, record=record)
    return Snapshot(dataset=dataset, xy_region=xy_region, field_reader=reader)
