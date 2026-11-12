"""Read bounded contiguous raw slabs before NumPy full-column derivation.

Five disk-backed NPY fields stay in original XYZ C order. Each slab eagerly
reads native full-y rows for its core x cells and two periodic x neighbours.
NumPy sees a local x axis with one halo on each side. Unit conversion order,
full z columns and native y-neighbour indexing retain the physical rules.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import time

import numpy as np

from concurrent.futures import ThreadPoolExecutor

from .physics.slab_derive import derive_bounded_columns, estimated_working_bytes
from .native_snapshot_reader import NativeSnapshotReader
from .native_snapshot_reader import NATIVE_FIELD_NAMES
from .physics import settings
from .physics.gas_fields import HYDROGEN_MASS_G
from .snapshot_reader import SlabArrays


ARRAY_NAMES = (
    'density_g_cm3', 'foreground_NH_cm2', 'temperature_QUOKKA_K',
    'shielding_NH_cm2', 'velocity_gradient_s', 'velocity_z_kms',
)


def _conversion_factor(source_unit, target_unit):
    factor, offset = source_unit.get_conversion_factor(target_unit, np.dtype(np.float64))
    if offset:
        raise ValueError('Density and velocity conversions must have zero offsets')
    return float(factor)


def scalar_unit_stages(native_reader):
    """Extract the exact scalar factors used by the original unyt stages.

    Momentum/rho unit simplification may itself multiply the quotient. Keep
    that coefficient distinct from native->gas and gas->cgs conversion factors.
    """
    dataset = native_reader._dataset
    density = dataset.arr([1.0], native_reader._native_units['gasDensity'])
    density_gas = density.to(native_reader._gas_units['density'])
    density_cgs = density_gas.to('g/cm**3')
    density_to_gas = _conversion_factor(density.units, density_gas.units)
    gas_density_to_cgs = _conversion_factor(density_gas.units, density_cgs.units)
    velocity_factors = np.empty((3, 4), dtype=np.float64)
    for axis_index, axis in enumerate(('x', 'y', 'z')):
        momentum = dataset.arr([1.0], native_reader._native_units[f'{axis}-GasMomentum'])
        native_velocity = momentum / density
        gas_velocity = native_velocity.to(native_reader._gas_units[f'velocity_{axis}'])
        cgs_velocity = gas_velocity.to('cm/s')
        kms_velocity = gas_velocity.to('km/s')
        velocity_factors[axis_index, 0] = float(native_velocity.value[0])
        velocity_factors[axis_index, 1] = _conversion_factor(native_velocity.units, gas_velocity.units)
        velocity_factors[axis_index, 2] = _conversion_factor(gas_velocity.units, cgs_velocity.units)
        velocity_factors[axis_index, 3] = _conversion_factor(gas_velocity.units, kms_velocity.units)
    # Unit simplification can leave a factor one ulp away from one even when
    # the printed result already says cm**-2. Reproduce the final .to stage.
    hydrogen_probe = dataset.arr([1.0], 'cm**-3')
    cell_column_probe = hydrogen_probe * native_reader._cell_widths[2].in_cgs()
    inverse_probe = 1.0 / cell_column_probe
    harmonic_probe = 2 / (inverse_probe + inverse_probe)
    canonical_column = harmonic_probe.to('cm**-2')
    shielding_to_cgs = _conversion_factor(harmonic_probe.units, canonical_column.units)
    return density_to_gas, gas_density_to_cgs, velocity_factors, shielding_to_cgs


@dataclass(frozen=True)
class FieldFile:
    """Validated float64 NPY payload location, geometry and file identity."""

    path: Path
    payload_offset: int
    shape: tuple[int, int, int]
    size_bytes: int
    modified_ns: int


def inspect_field_file(path, expected_shape):
    """Read only the NPY header; reject layouts this contiguous reader cannot use."""
    with path.open('rb', buffering=0) as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f'Unsupported converted NPY version: {version}')
        payload_offset = stream.tell()
        state = os.fstat(stream.fileno())
    if tuple(shape) != tuple(expected_shape) or dtype != np.dtype(np.float64) or fortran_order:
        raise ValueError(f'Expected native-shape C-order float64 field: {path}')
    if state.st_size != payload_offset + int(np.prod(shape)) * dtype.itemsize:
        raise ValueError(f'Converted field payload length differs from its NPY header: {path}')
    return FieldFile(path, payload_offset, tuple(shape), state.st_size, state.st_mtime_ns)


def contiguous_x_runs(x_indices):
    """Describe contiguous reads as (destination start, stop, native x start).

    Example: [7, 0, 1, 2] becomes [(0, 1, 7), (1, 4, 0)]. Full-box
    selection retains both duplicate endpoint halos rather than dropping them.
    """
    first = 0
    runs = []
    for position in range(1, len(x_indices)):
        if x_indices[position] != x_indices[position - 1] + 1:
            runs.append((first, position, x_indices[first]))
            first = position
    runs.append((first, len(x_indices), x_indices[first]))
    return runs


class BoundedEagerReader(NativeSnapshotReader):
    """Own at most five compact slab inputs, never five full snapshot arrays."""

    def __init__(self, native_reader, *, cache_directory, reader_threads=8, read_workers=5):
        if not isinstance(reader_threads, int) or reader_threads < 1:
            raise ValueError('Reader thread count must be positive')
        if not isinstance(read_workers, int) or not 1 <= read_workers <= len(NATIVE_FIELD_NAMES):
            raise ValueError('Eager read workers must be between one and five')
        self.__dict__.update(native_reader.__dict__)
        self._active_slab_field_cache = None
        self.fields = None
        self.reader_threads = reader_threads
        self.pool = ThreadPoolExecutor(max_workers=reader_threads, thread_name_prefix='raw-reader')
        self.unit_stages = scalar_unit_stages(native_reader)
        self.derivation_statistics = {
            'calls': 0,
            'seconds': 0.0,
            'cells': 0,
            'method': 'numpy-full-z',
            'parallelism': 'ThreadPoolExecutor',
            'maximum_numpy_working_bytes': 0,
            'working_memory_scope': 'Conservative bounded NumPy blocks; not measured RSS',
        }
        self.cache_directory = Path(cache_directory).resolve(strict=True)
        self.field_files = {}
        for name in NATIVE_FIELD_NAMES:
            path = (self.cache_directory / f'{name}.npy').resolve(strict=True)
            if path.parent != self.cache_directory:
                raise ValueError('Converted fields must resolve inside their own cache directory')
            self.field_files[name] = inspect_field_file(path, self._native_shape)
        self.read_workers = min(read_workers, reader_threads)
        self._closed = False
        self.read_statistics = {
            'slab_reads': 0,
            'read_workers': self.read_workers,
            'payload_read_calls': 0,
            'payload_bytes_read': 0,
            'payload_file_open_calls': 0,
            'payload_file_close_calls': 0,
            'peak_raw_slab_bytes_owned': 0,
            'current_raw_slab_bytes_owned': 0,
            'eager_read_seconds': 0.0,
            'fused_kernel_seconds': 0.0,
            'fields_released_before_return': True,
            'native_shape_unchanged': True,
            'full_snapshot_arrays_allocated': False,
            'mapped_bytes_virtual': 0,
            'group_slab_cache_enabled': False,
            'exception_pool_drains': 0,
        }

    def _read_compact_field(self, name, x_indices, runs):
        """Fill one compact array directly using one to three large readinto runs."""
        field = self.field_files[name]
        shape = (len(x_indices), field.shape[1], field.shape[2])
        values = np.empty(shape, dtype=np.float64, order='C')
        raw = memoryview(values).cast('B')
        row_bytes = field.shape[1] * field.shape[2] * 8
        calls = 0
        transferred = 0
        with field.path.open('rb', buffering=0) as stream:
            state = os.fstat(stream.fileno())
            if state.st_size != field.size_bytes or state.st_mtime_ns != field.modified_ns:
                raise ValueError(f'Converted field changed after header validation: {field.path}')
            for destination_start, destination_stop, source_start in runs:
                stream.seek(field.payload_offset + source_start * row_bytes)
                first_byte = destination_start * row_bytes
                stop_byte = destination_stop * row_bytes
                while first_byte < stop_byte:
                    byte_count = stream.readinto(raw[first_byte:stop_byte])
                    calls += 1
                    if byte_count is None or byte_count <= 0:
                        raise ValueError(f'Short converted field read: {field.path}')
                    first_byte += byte_count
                    transferred += byte_count
        values.flags.writeable = False
        return name, values, calls, transferred

    def _read_fields(self, x_start, x_stop):
        nx = self._native_shape[0]
        x_indices = [(x_start - 1) % nx, *range(x_start, x_stop), x_stop % nx]
        runs = contiguous_x_runs(x_indices)
        began = time.perf_counter()
        self.fields = {}
        records = []
        # Submit a bounded group of fields. Reader threads are reused only
        # after every file read has completed, so I/O and derivation do not nest.
        for first in range(0, len(NATIVE_FIELD_NAMES), self.read_workers):
            names = NATIVE_FIELD_NAMES[first:first + self.read_workers]
            futures = [self.pool.submit(self._read_compact_field, name, x_indices, runs) for name in names]
            for future in futures:
                name, values, calls, transferred = future.result()
                self.fields[name] = values
                records.append((calls, transferred))
        raw_bytes = sum(values.nbytes for values in self.fields.values())
        self.read_statistics['slab_reads'] += 1
        self.read_statistics['payload_read_calls'] += sum(item[0] for item in records)
        self.read_statistics['payload_bytes_read'] += sum(item[1] for item in records)
        self.read_statistics['payload_file_open_calls'] += len(records)
        self.read_statistics['payload_file_close_calls'] += len(records)
        self.read_statistics['peak_raw_slab_bytes_owned'] = max(self.read_statistics['peak_raw_slab_bytes_owned'], raw_bytes)
        self.read_statistics['current_raw_slab_bytes_owned'] = raw_bytes
        self.read_statistics['fields_released_before_return'] = False
        self.read_statistics['eager_read_seconds'] += time.perf_counter() - began

    @contextmanager
    def reuse_slab_fields(self, **unused):
        if self._closed:
            raise RuntimeError('The bounded eager reader is closed')
        yield self

    def derive_slab(self, snapshot, *, x_start, x_stop, y_start, y_stop, on_stage=None):
        """Read compact inputs then derive unchanged six read-only slab outputs."""
        if self._closed or self.fields is not None:
            raise RuntimeError('An eager reader is closed or already deriving a slab')
        nx, ny, nz = self._native_shape
        if not (0 <= x_start < x_stop <= nx and 0 <= y_start < y_stop <= ny and nz >= 2):
            raise ValueError('Invalid full-z slab bounds')
        if on_stage is not None:
            on_stage('field_reading')
        try:
            self._read_fields(x_start, x_stop)
            if on_stage is not None:
                on_stage('fused_derivation')
            began = time.perf_counter()
            selected_shape = (x_stop - x_start, y_stop - y_start, nz)
            output = np.empty((6, *selected_shape), dtype=np.float64, order='C')
            column_count = selected_shape[0] * selected_shape[1]
            task_count = min(self.reader_threads, column_count)
            working_bytes = estimated_working_bytes(
                column_count=column_count,
                column_depth=nz,
                threads=task_count,
            )
            self.derivation_statistics['maximum_numpy_working_bytes'] = max(
                self.derivation_statistics['maximum_numpy_working_bytes'],
                working_bytes,
            )
            density_to_gas, gas_density_to_cgs, velocity_factors, shielding_to_cgs = self.unit_stages
            widths_cgs = tuple(float(width.to('cm').value) for width in self._cell_widths)
            dz_foreground = float(snapshot.cell_widths[2].to('cm').value)
            dz_shielding = float(self._cell_widths[2].in_cgs().value)

            def calculate_columns(task):
                # x=1 is the compact core's first row. The raw array's first
                # and last rows provide its halos; original y remains global.
                return derive_bounded_columns(
                    density_raw=self.fields['gasDensity'],
                    temperature_raw=self.fields['temperature'],
                    momentum_x=self.fields['x-GasMomentum'],
                    momentum_y=self.fields['y-GasMomentum'],
                    momentum_z=self.fields['z-GasMomentum'],
                    output=output,
                    velocity_factors=velocity_factors,
                    density_to_gas=density_to_gas,
                    gas_density_to_cgs=gas_density_to_cgs,
                    hydrogen_fraction=float(settings.X_H),
                    hydrogen_mass_g=float(HYDROGEN_MASS_G),
                    dx_cm=widths_cgs[0],
                    dy_cm=widths_cgs[1],
                    dz_gradient_cm=widths_cgs[2],
                    dz_foreground_cm=dz_foreground,
                    dz_shielding_cm=dz_shielding,
                    shielding_to_cgs=shielding_to_cgs,
                    gradient_floor_s=float(settings.SIMULATION_DVDR_MIN_S),
                    x_start=1,
                    y_start=y_start,
                    first_column=task * column_count // task_count,
                    stop_column=(task + 1) * column_count // task_count,
                )

            invalid_density_counts = list(self.pool.map(calculate_columns, range(task_count)))
            if sum(invalid_density_counts):
                raise ValueError('Expected finite nonnegative hydrogen density in full columns')
            arrays = {name: output[index].ravel() for index, name in enumerate(ARRAY_NAMES)}
            for array in arrays.values():
                array.flags.writeable = False
            slab = SlabArrays(
                **arrays,
                x_start=x_start,
                y_start=y_start,
                shape=selected_shape,
                native_y_size=ny,
                cell_volume_cm3=snapshot.cell_volume_cm3,
            )
            elapsed = time.perf_counter() - began
            self.derivation_statistics['calls'] += 1
            self.derivation_statistics['seconds'] += elapsed
            self.derivation_statistics['cells'] += slab.cell_count
            self.read_statistics['fused_kernel_seconds'] += elapsed
            return slab
        except BaseException:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self._closed = True
            self.read_statistics['exception_pool_drains'] += 1
            raise
        finally:
            self.fields = None
            self.read_statistics['current_raw_slab_bytes_owned'] = 0
            self.read_statistics['fields_released_before_return'] = True

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.fields = None
        self._closed = True
        self.read_statistics['current_raw_slab_bytes_owned'] = 0
        self.read_statistics['fields_released_before_return'] = True
