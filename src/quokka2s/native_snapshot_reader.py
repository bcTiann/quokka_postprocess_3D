"""Copy requested native QUOKKA cells from AMReX files without a covering grid.

yt still opens the snapshot and supplies its block geometry, file offsets,
field order and units. This reader returns compact copies of a slab's selected
x/y cells and periodic neighbours; every sightline retains the full z depth.
Sparse block selections use read-only maps. Selections covering all block
y/z rows use contiguous temporary field buffers. Both are discarded after
the selected values are copied into compact arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from unyt import unyt_array


NATIVE_FIELD_NAMES = (
    'gasDensity',
    'temperature',
    'x-GasMomentum',
    'y-GasMomentum',
    'z-GasMomentum',
)


@dataclass(frozen=True)
class _NativeBlock:
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
    field metadata once; it does not read the blocks' numerical payloads.
    Only three-dimensional, level-zero BoxLib/AMReX native files are supported,
    because the processing workflow uses one uniform native cell grid.

    Example: ``read_slab_fields(x_start=0, x_stop=8, y_start=32, y_stop=96)``
    returns fields of shape ``(10, 66, Nz)``. Its x indices are
    ``[Nx - 1, 0, ..., 7, 8]`` and its y indices are ``[31, ..., 96]``.
    The caller removes the first and last x/y cells after deriving columns
    and gradients. Sparse selections copy from read-only maps; selections
    covering all block y/z rows copy from contiguous temporary buffers.
    Returned arrays retain neither source maps nor whole-block buffers.
    """

    def __init__(self, dataset):
        # 1. Record the uniform native geometry used by every slab.
        self._dataset = dataset
        self._native_shape = tuple(int(size) for size in dataset.domain_dimensions)
        self._cell_widths = dataset.domain_width / dataset.domain_dimensions
        self._z_indices = np.arange(self._native_shape[2])

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
            with open(grid.filename, 'rb') as source_file:
                payload_offset = grid._get_offset(source_file)
            cell_count = int(np.prod(block_shape))
            field_bytes = cell_count * self._native_dtype.itemsize
            blocks.append(
                _NativeBlock(
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

    def read_slab_fields(
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
        values are owned copies; source maps or temporary whole-block field
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

        # 3. Reproduce yt's density and momentum-derived velocity unit stages.
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
        or contiguous field buffers are released after copying selected cells.
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
            all_yz_rows_selected = (
                source_zyx_selection[1].size >= block.shape[1]
                and source_zyx_selection[0].size >= block.shape[2]
            )

            for field_name in NATIVE_FIELD_NAMES:
                block_field = self._read_block_field(
                    block=block,
                    field_name=field_name,
                    all_yz_rows_selected=all_yz_rows_selected,
                )
                # Disk cells use x-fastest order. Copy in that order to avoid
                # repeatedly visiting distant file pages, then restore x/y/z.
                # Both transposes change axis order only, never cell values.
                block_field_zyx = block_field.transpose(2, 1, 0)
                selected_values_zyx = block_field_zyx[source_zyx_selection]
                selected_values_xyz = selected_values_zyx.transpose(2, 1, 0)
                native_fields[field_name][destination_selection] = selected_values_xyz
                del block_field_zyx
                del block_field
        return native_fields

    def _read_block_field(
        self,
        *,
        block: _NativeBlock,
        field_name: str,
        all_yz_rows_selected: bool,
    ) -> np.ndarray:
        """Return one native x/y/z field using its cached dtype and file offset.

        Small regions use a read-only map, so copying touches only selected
        pages. When every y/z row is needed, even a narrow x slice touches
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


def _select_block_cells(
    *,
    requested_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    block: _NativeBlock,
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
