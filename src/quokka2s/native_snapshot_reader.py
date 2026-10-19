"""Copy requested native QUOKKA cells from AMReX files without a covering grid.

yt still opens the snapshot and supplies its block geometry, file offsets,
field order and units. This reader returns compact copies of a slab's selected
x/y cells and periodic neighbours; every sightline retains the full z depth.
Selections covering all block y/z rows use contiguous temporary field buffers.
Sparse y selections covering every block x cell read contiguous x/y bands per
z plane. Other sparse selections use read-only maps. Source buffers and maps
are discarded after the selected values are copied into compact arrays.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import os
from typing import BinaryIO

import numpy as np
from unyt import unyt_array

from .slab_field_cache import SlabFieldCache


NATIVE_FIELD_NAMES = (
    'gasDensity',
    'temperature',
    'x-GasMomentum',
    'y-GasMomentum',
    'z-GasMomentum',
)


@dataclass(frozen=True)
class NativeBlock:
    """One level-zero block: native cell bounds and its file payload location.

    Each field occupies ``field_bytes`` bytes from ``payload_offset`` onward,
    in yt's field order. Its three-dimensional cells use Fortran order on disk.
    No cell values or open file handles are retained here.
    """

    filename: str
    start: tuple[int, int, int]
    shape: tuple[int, int, int]
    payload_offset: int
    field_bytes: int


class NativeSnapshotReader:
    """Read compact unit-aware slabs from a uniform native QUOKKA plotfile.

    ``dataset`` is the opened yt dataset. Construction caches small block and
    field metadata once; it does not cache the blocks' numerical payloads.
    Only three-dimensional, level-zero BoxLib/AMReX native files are supported,
    because the processing workflow uses one uniform native cell grid.

    Example: ``read_slab_fields(x_start=0, x_stop=8, y_start=32, y_stop=96)``
    returns fields of shape ``(10, 66, Nz)``. Its x indices are
    ``[Nx - 1, 0, ..., 7, 8]`` and its y indices are ``[31, ..., 96]``.
    The caller removes the first and last x/y cells after deriving columns
    and gradients. Full block y/z coverage uses contiguous field buffers;
    sparse y selections covering every block x cell use contiguous row bands.
    Other sparse selections copy from read-only maps. Returned arrays retain
    neither source maps nor temporary field or band buffers.
    """

    def __init__(self, dataset):
        # 1. Record the uniform native geometry used by every slab.
        self._dataset = dataset
        self._native_shape = tuple(int(size) for size in dataset.domain_dimensions)
        self._cell_widths = dataset.domain_width / dataset.domain_dimensions
        self._z_indices = np.arange(self._native_shape[2])
        self._active_slab_field_cache = None

        # 2. Isolate yt's private file metadata access in construction.
        # These are the same dtype, field order and payload offsets used by
        # yt.frontends.amrex.io.IOHandlerBoxlib._read_chunk_data().
        index = dataset.index
        if dataset.dataset_type != 'boxlib_native' or dataset.dimensionality != 3:
            raise ValueError('NativeSnapshotReader requires a 3-D native QUOKKA plotfile')
        if index.max_level != 0:
            raise ValueError('NativeSnapshotReader requires a uniform level-zero grid')
        self._native_dtype = np.dtype(index._dtype)
        if self._native_dtype.kind != 'f' or self._native_dtype.itemsize not in (4, 8):
            raise ValueError('Native AMReX fields must use 32-bit or 64-bit floating point')

        self._field_positions = {}
        for field_name in NATIVE_FIELD_NAMES:
            field_key = ('boxlib', field_name)
            if field_key not in index.field_order:
                raise ValueError(f'Native QUOKKA field is missing: {field_key}')
            self._field_positions[field_name] = index.field_order.index(field_key)

        blocks = []
        block_cell_count = 0
        for grid in index.grids:
            block_start = tuple(int(value) for value in grid.get_global_startindex())
            block_shape = tuple(int(size) for size in grid.ActiveDimensions)
            if any(
                start < 0 or start + size > native_size
                for start, size, native_size in zip(
                    block_start,
                    block_shape,
                    self._native_shape,
                )
            ):
                raise ValueError('Native block bounds must lie inside the original cell grid')
            # Only the short FAB header is needed here. Limit read-ahead on
            # filesystems whose default buffer would also read cell payloads.
            with open(
                file=grid.filename,
                mode='rb',
                buffering=8192,
            ) as source_file:
                payload_offset = grid._get_offset(source_file)
            cell_count = int(np.prod(block_shape))
            field_bytes = cell_count * self._native_dtype.itemsize
            blocks.append(
                NativeBlock(
                    filename=grid.filename,
                    start=block_start,
                    shape=block_shape,
                    payload_offset=payload_offset,
                    field_bytes=field_bytes,
                )
            )
            block_cell_count += cell_count
        if block_cell_count != int(np.prod(self._native_shape)):
            raise ValueError('Native blocks must cover the complete level-zero cell grid')
        self._blocks = tuple(blocks)

        # 3. Cache units rather than consulting field metadata during each read.
        self._native_units = {
            field_name: dataset.field_info[('boxlib', field_name)].units
            for field_name in NATIVE_FIELD_NAMES
        }
        self._gas_units = {
            field_name: dataset.field_info[('gas', field_name)].units
            for field_name in ('density', 'velocity_x', 'velocity_y', 'velocity_z')
        }

    @property
    def blocks(self) -> tuple[NativeBlock, ...]:
        """Immutable block geometry/offset records from yt; no cell payloads."""
        return self._blocks

    def read_block_field(self, *, block: NativeBlock, field_name: str) -> np.ndarray:
        """Read one complete native field, e.g. (256,256,128), in one request.

        Returns float64 XYZ values in native units. The array owns its bytes
        or a promoted copy; no source map/file survives this call. AMReX's
        x-fastest bytes are interpreted with yt's dtype and field offset.
        """
        values = self._read_block_bytes(
            block=block,
            field_name=field_name,
            within_field_offset=0,
            byte_count=block.field_bytes,
        )
        native_zyx = values.reshape(block.shape[::-1])
        return np.asarray(native_zyx.transpose(2, 1, 0), dtype=np.float64)

    def read_block_z_plane(
        self,
        *,
        block: NativeBlock,
        field_name: str,
        local_z: int,
    ) -> np.ndarray:
        """Read one native XY plane, e.g. (256,256), without the whole field.

        local_z is the cell index within block, not the global z coordinate.
        Returns promoted float64 XY values in native units, with no open file.
        """
        if not isinstance(local_z, (int, np.integer)) or not 0 <= local_z < block.shape[2]:
            raise ValueError('local_z must select a cell inside the native block')
        plane_bytes = block.shape[0] * block.shape[1] * self._native_dtype.itemsize
        values = self._read_block_bytes(
            block=block,
            field_name=field_name,
            within_field_offset=local_z * plane_bytes,
            byte_count=plane_bytes,
        )
        return np.asarray(values.reshape(block.shape[1], block.shape[0]).T, dtype=np.float64)

    def _read_block_bytes(
        self,
        *,
        block: NativeBlock,
        field_name: str,
        within_field_offset: int,
        byte_count: int,
    ) -> np.ndarray:
        """Open a source read-only and check its complete requested return."""
        field_position = self._field_positions[field_name]
        offset = block.payload_offset + field_position * block.field_bytes + within_field_offset
        with open(file=block.filename, mode='rb', buffering=0) as source_file:
            payload = os.pread(source_file.fileno(), byte_count, offset)
        if len(payload) != byte_count:
            raise ValueError(f'Short native field read: {block.filename}, {field_name}')
        return np.frombuffer(buffer=payload, dtype=self._native_dtype)

    def attach_density_units(self, *, values: np.ndarray):
        """Native float64 XYZ density -> yt's declared gas-density units."""
        native_density = self._dataset.arr(
            input_array=values,
            units=self._native_units['gasDensity'],
        )
        return native_density.to(self._gas_units['density'])

    @contextmanager
    def reuse_slab_fields(
        self,
        *,
        selected_slabs,
        selected_y_range: tuple[int, int],
        slabs_per_read: int = 4,
    ):
        """Reuse bounded native fields for the actual ordered process windows.

        Global half-open windows include any max_slabs truncation. Each group
        keeps original z and periodic x/y neighbours. Outside this context the
        reader retains no value cache. Empty selections perform no payload reads.
        """
        if self._active_slab_field_cache is not None:
            raise RuntimeError('Native slab field reuse contexts cannot be nested')
        cache = SlabFieldCache(
            native_reader=self,
            selected_slabs=selected_slabs,
            selected_y_range=selected_y_range,
            slabs_per_read=slabs_per_read,
        )
        self._active_slab_field_cache = cache
        try:
            yield
        finally:
            self._active_slab_field_cache = None
            cache.close()

    def read_slab_fields(
        self,
        *,
        x_start: int,
        x_stop: int,
        y_start: int,
        y_stop: int,
    ) -> dict[tuple[str, str], unyt_array]:
        """Read owned halo fields, or their read-only views inside a reuse context.

        Values, shapes and units match the direct native read. Reuse only changes
        field ownership and source request repetition; slab derivation is separate.
        """
        if self._active_slab_field_cache is not None:
            return self._active_slab_field_cache.read_slab_fields(
                x_start=x_start,
                x_stop=x_stop,
                y_start=y_start,
                y_stop=y_stop,
            )
        return self._read_uncached_slab_fields(
            x_start=x_start,
            x_stop=x_stop,
            y_start=y_start,
            y_stop=y_stop,
        )

    def _read_uncached_slab_fields(
        self,
        *,
        x_start: int,
        x_stop: int,
        y_start: int,
        y_stop: int,
    ) -> dict[tuple[str, str], unyt_array]:
        """Return selected cell fields, wrapped x/y neighbours, and cell widths.

        Bounds are global native cell indices with excluded stops, satisfying
        ``0 <= start < stop <= axis_size``. For an ``(nx, ny, Nz)`` core, the
        five field arrays have shape ``(nx + 2, ny + 2, Nz)`` in C order.
        Use ``[1:-1, 1:-1, :]`` to select the core after numerical preparation.

        Keys are ``('gas', 'density')``, ``('boxlib', 'temperature')`` and the
        three ``('gas', 'velocity_x/y/z')`` fields. Density and velocities use
        yt's declared gas units; temperature uses its native declared units.
        ``('boxlib', 'dx/dy/dz')`` are unit-aware scalar cell widths. All field
        values are owned copies; source maps and temporary field or band
        buffers are discarded after each copy.
        """
        for axis_name, start, stop, axis_size in (
            ('x', x_start, x_stop, self._native_shape[0]),
            ('y', y_start, y_stop, self._native_shape[1]),
        ):
            if not 0 <= start < stop <= axis_size:
                raise ValueError(
                    f'{axis_name} bounds must satisfy 0 <= start < stop <= {axis_size}'
                )

        # 1. Select the core and one original-box neighbour on each x/y side.
        # Example: y=0:1 selects [Ny - 1, 0, 1], including the opposite face.
        x_indices = np.arange(start=x_start - 1, stop=x_stop + 1)
        x_indices = x_indices % self._native_shape[0]
        y_indices = np.arange(start=y_start - 1, stop=y_stop + 1)
        y_indices = y_indices % self._native_shape[1]
        requested_indices = (x_indices, y_indices, self._z_indices)

        # 2. Copy five simulation fields only from intersecting native blocks.
        native_fields = self._copy_selected_native_fields(
            requested_indices=requested_indices,
        )

        return self.attach_field_units(native_fields=native_fields)

    def attach_field_units(self, *, native_fields: dict[str, np.ndarray]):
        """Attach units to rho/T/three momenta from either native reading path.

        Inputs are five same-shape float64 XYZ arrays in native units. Returns
        unit-aware density, raw temperature, three velocities and cell widths.
        Both the slab reader and range scanner use this exact conversion order.
        """
        # Reproduce yt's density and momentum-derived velocity unit stages.
        # Divide native momentum by native density first, then normalize to
        # declared gas units. Later cgs/km conversions must see these same
        # values; simplifying the unit path can change floating-point results.
        native_density = self._dataset.arr(
            input_array=native_fields['gasDensity'],
            units=self._native_units['gasDensity'],
        )
        density = native_density.to(self._gas_units['density'])
        temperature = self._dataset.arr(
            input_array=native_fields['temperature'],
            units=self._native_units['temperature'],
        )
        fields = {
            ('gas', 'density'): density,
            ('boxlib', 'temperature'): temperature,
        }
        for axis_name in ('x', 'y', 'z'):
            momentum_name = f'{axis_name}-GasMomentum'
            native_momentum = self._dataset.arr(
                input_array=native_fields[momentum_name],
                units=self._native_units[momentum_name],
            )
            native_velocity = native_momentum / native_density
            velocity_name = f'velocity_{axis_name}'
            fields[('gas', velocity_name)] = native_velocity.to(
                self._gas_units[velocity_name],
            )

        # 4. Supply uniform scalar widths for the existing physics calculations.
        for axis_index, axis_name in enumerate(('x', 'y', 'z')):
            fields[('boxlib', f'd{axis_name}')] = self._cell_widths[axis_index]
        return fields

    def _copy_selected_native_fields(
        self,
        *,
        requested_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Copy requested x/y/z indices from disk into compact float64 arrays.

        Indices may repeat at periodic boundaries; output order follows the
        requested arrays. AMReX payloads use their native dtype and Fortran
        order. Output buffers use float64, matching yt's fluid-selection
        promotion even when the plotfile stores float32 values. Source maps
        and temporary field or band buffers are released after copying cells.
        """
        selected_shape = tuple(indices.size for indices in requested_indices)
        native_fields = {
            field_name: np.empty(shape=selected_shape, dtype=np.float64)
            for field_name in NATIVE_FIELD_NAMES
        }
        for block in self._blocks:
            selection = _select_block_cells(
                requested_indices=requested_indices,
                block=block,
            )
            if selection is None:
                continue
            destination_selection, source_zyx_selection = selection
            read_geometry = _select_block_read_geometry(
                block=block,
                source_zyx_selection=source_zyx_selection,
            )

            # Band reads share one read-only descriptor across the five fields.
            # Other geometries retain their existing map/fromfile operations.
            if read_geometry == 'complete_x_rows':
                source_context = open(block.filename, 'rb')
            else:
                source_context = nullcontext(None)
            with source_context as source_file:
                for field_name in NATIVE_FIELD_NAMES:
                    selected_values_zyx = self._read_selected_block_field(
                        block=block,
                        field_name=field_name,
                        source_zyx_selection=source_zyx_selection,
                        read_geometry=read_geometry,
                        source_file=source_file,
                    )
                    # Read in native x-fastest order, then restore x/y/z.
                    # The transpose changes axis order only, never cell values.
                    selected_values_xyz = selected_values_zyx.transpose(2, 1, 0)
                    native_fields[field_name][destination_selection] = selected_values_xyz
        return native_fields

    def _read_selected_block_field(
        self,
        *,
        block: NativeBlock,
        field_name: str,
        source_zyx_selection: tuple[np.ndarray, np.ndarray, np.ndarray],
        read_geometry: str,
        source_file: BinaryIO | None,
    ) -> np.ndarray:
        """Return selected native values as a C-order (z, y, x) owned array.

        The index tuple comes from _select_block_cells() and retains requested
        periodic duplicates. Values keep the file's f32/f64 dtype and native
        field units. Only complete-x-row geometry supplies an open source_file;
        other geometries use the existing whole-field or mapped read path.
        """
        if source_file is not None:
            return self._read_selected_field_bands(
                source_fd=source_file.fileno(),
                block=block,
                field_name=field_name,
                source_x=source_zyx_selection[2].ravel(),
                source_y=source_zyx_selection[1].ravel(),
                source_z=source_zyx_selection[0].ravel(),
            )

        block_field = self._read_block_field(
            block=block,
            field_name=field_name,
            all_yz_rows_selected=read_geometry == 'all_yz_rows',
        )
        block_field_zyx = block_field.transpose(2, 1, 0)
        return block_field_zyx[source_zyx_selection]

    def _read_selected_field_bands(
        self,
        *,
        source_fd: int,
        block: NativeBlock,
        field_name: str,
        source_x: np.ndarray,
        source_y: np.ndarray,
        source_z: np.ndarray,
    ) -> np.ndarray:
        """Read native values into a C-order selected (z, y, x) buffer.

        Source indices are block-local cell positions in requested output order;
        repeated periodic indices remain repeated. Each read contains one
        contiguous +1 y run and all block x cells from one native z plane.
        Values retain the file's f32/f64 dtype and native field units.
        """
        selected_values_zyx = np.empty(
            shape=(source_z.size, source_y.size, source_x.size),
            dtype=self._native_dtype,
            order='C',
        )
        field_position = self._field_positions[field_name]
        field_offset = block.payload_offset + field_position * block.field_bytes
        native_x_size = block.shape[0]
        native_y_size = block.shape[1]
        item_bytes = self._native_dtype.itemsize

        # Splitting on anything except +1 preserves wraparound and duplicates.
        run_boundaries = np.flatnonzero(np.diff(source_y) != 1) + 1
        run_starts = np.concatenate(([0], run_boundaries))
        run_stops = np.concatenate((run_boundaries, [source_y.size]))
        for destination_z, local_z in enumerate(source_z):
            for run_start, run_stop in zip(run_starts, run_stops):
                run_y_size = int(run_stop - run_start)
                first_local_y = int(source_y[run_start])
                first_cell = (int(local_z) * native_y_size + first_local_y) * native_x_size
                band_offset = field_offset + first_cell * item_bytes
                band_cell_count = run_y_size * native_x_size
                band_bytes = band_cell_count * item_bytes
                band_payload = os.pread(source_fd, band_bytes, band_offset)
                if len(band_payload) != band_bytes:
                    raise OSError(
                        f'Short native field read: {block.filename}, '
                        f'{field_name}, offset={band_offset}, '
                        f'expected={band_bytes}, actual={len(band_payload)}'
                    )
                band_values = np.frombuffer(
                    buffer=band_payload,
                    dtype=self._native_dtype,
                    count=band_cell_count,
                )
                band_values_yx = band_values.reshape((run_y_size, native_x_size))
                selected_values_zyx[destination_z, run_start:run_stop, :] = (
                    band_values_yx[:, source_x]
                )
        return selected_values_zyx

    def _read_block_field(
        self,
        *,
        block: NativeBlock,
        field_name: str,
        all_yz_rows_selected: bool,
    ) -> np.ndarray:
        """Return one native x/y/z field using its cached dtype and file offset.

        Sparse x selections use a read-only map, touching selected pages.
        When every y/z row is needed, even a narrow x slice touches
        pages throughout the field; one contiguous read avoids page-fault
        overhead. That full buffer is discarded after copying selected cells.
        Both paths retain the native dtype and Fortran cell order on disk.
        """
        field_position = self._field_positions[field_name]
        field_offset = block.payload_offset + field_position * block.field_bytes
        if all_yz_rows_selected:
            field_values = np.fromfile(
                file=block.filename,
                dtype=self._native_dtype,
                count=int(np.prod(block.shape)),
                offset=field_offset,
            )
            return field_values.reshape(block.shape, order='F')

        return np.memmap(
            filename=block.filename,
            dtype=self._native_dtype,
            mode='r',
            offset=field_offset,
            shape=block.shape,
            order='F',
        )


def _select_block_read_geometry(
    *,
    block: NativeBlock,
    source_zyx_selection: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> str:
    """Select full-y/z, complete-x-row, or partial-x-row field reading.

    Indices address block-local (z, y, x) cells and may repeat at periodic
    boundaries. Full y/z coverage keeps the existing contiguous-field path.
    Otherwise bands require every native x cell; narrow x keeps mapped reads.
    """
    all_yz_rows_selected = (
        source_zyx_selection[1].size >= block.shape[1]
        and source_zyx_selection[0].size >= block.shape[2]
    )
    if all_yz_rows_selected:
        return 'all_yz_rows'
    source_x = source_zyx_selection[2].ravel()
    all_native_x_selected = np.unique(source_x).size == block.shape[0]
    if all_native_x_selected:
        return 'complete_x_rows'
    return 'partial_x_rows'


def _select_block_cells(
    *,
    requested_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    block: NativeBlock,
):
    """Locate requested cells inside one block, or return None if disjoint.

    The returned index tuples address the compact x/y/z destination and the
    z/y/x view of the block-local source. x is last in that view so copies
    follow the file's x-fastest order. Example: requested x=[255, 0, 1] and
    block x=0:128 gives destination x positions [1, 2] and source x=[0, 1].
    """
    destination_indices = []
    source_indices = []
    for axis_index, global_indices in enumerate(requested_indices):
        block_start = block.start[axis_index]
        block_stop = block_start + block.shape[axis_index]
        inside_block = (global_indices >= block_start) & (global_indices < block_stop)
        destination_positions = np.flatnonzero(inside_block)
        if destination_positions.size == 0:
            return None
        local_indices = global_indices[destination_positions] - block_start
        destination_indices.append(destination_positions)
        source_indices.append(local_indices)
    source_x, source_y, source_z = source_indices
    destination_selection = np.ix_(*destination_indices)
    source_zyx_selection = np.ix_(source_z, source_y, source_x)
    return destination_selection, source_zyx_selection
