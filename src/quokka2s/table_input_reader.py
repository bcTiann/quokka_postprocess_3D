"""Read table-input fields in native blocks, retaining exact full-z columns.

A density pass records each selected XY column at its z-block boundaries.
The descending pass processes all selected columns at the same z before
moving down, so their periodic neighbours share already-read native fields.
Only the current z layer and small column carries remain in memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import product

import numpy as np

from quokka2s.native_snapshot_reader import (
    NATIVE_FIELD_NAMES,
    NativeBlock,
    NativeSnapshotReader,
)
from quokka2s.physics.gas_fields import (
    hydrogen_number_density,
    hydrogen_number_density_cm3,
    velocity_gradient,
)
from quokka2s.snapshot_reader import Snapshot


@dataclass(frozen=True)
class NativeBlockColumn:
    """Retained XY rectangle and its native blocks in ascending z order.

    Bounds are global cell indices, with excluded stops. An ROI may cut into
    the native XY block; blocks still describe complete on-disk fields.
    Example: x=0:256, y=0:256, with 32 blocks covering z=0:4096.
    """

    x_start: int
    x_stop: int
    y_start: int
    y_stop: int
    blocks: tuple[NativeBlock, ...]


@dataclass(frozen=True)
class TableInputBatch:
    """Four read-only (B,) views in g/cm^3, K, cm^-2 and s^-1.

    From TableInputLayer.iter_batches(); nH is calculated once on first use.
    There are no luminosities, foreground dust columns or velocity spectra.
    """

    density_g_cm3: np.ndarray
    temperature_QUOKKA_K: np.ndarray
    shielding_NH_cm2: np.ndarray
    velocity_gradient_s: np.ndarray

    @cached_property
    def hydrogen_density_cm3(self) -> np.ndarray:
        values = hydrogen_number_density_cm3(
            density_g_cm3=self.density_g_cm3,
        )
        values.flags.writeable = False
        return values


@dataclass(frozen=True)
class TableInputLayer:
    """Owned four-field XYZ arrays for one retained native z block.

    Fields have identical shape (selected_Nx, selected_Ny, native_block_Nz),
    e.g. (256, 256, 128), in g/cm^3, K, cm^-2 and s^-1. x/y/z_start give
    their global origin. Batches are views; finish them before the next layer.
    """

    density_g_cm3: np.ndarray
    temperature_QUOKKA_K: np.ndarray
    shielding_NH_cm2: np.ndarray
    velocity_gradient_s: np.ndarray
    x_start: int
    y_start: int
    z_start: int

    def __post_init__(self):
        for values in (
            self.density_g_cm3,
            self.temperature_QUOKKA_K,
            self.shielding_NH_cm2,
            self.velocity_gradient_s,
        ):
            values.flags.writeable = False

    def iter_batches(self, *, batch_size: int):
        """Yield consecutive (B,) views, each at most batch_size cells."""
        density = self.density_g_cm3.ravel()
        temperature = self.temperature_QUOKKA_K.ravel()
        shielding_column = self.shielding_NH_cm2.ravel()
        gradient = self.velocity_gradient_s.ravel()
        for start in range(0, density.size, batch_size):
            stop = min(start + batch_size, density.size)
            yield TableInputBatch(
                density_g_cm3=density[start:stop],
                temperature_QUOKKA_K=temperature[start:stop],
                shielding_NH_cm2=shielding_column[start:stop],
                velocity_gradient_s=gradient[start:stop],
            )


@dataclass
class NativeColumnState:
    """Keep one selected XY column's sums and upper velocity faces.

    column contains ascending-z native block metadata. lower_boundaries is
    the density prepass result: unit-aware XY arrays [cm^-2], with None at
    the physical lower edge. upper_column is the descending NH carry in the
    same units; upper_velocity_planes contains native density/z momentum at
    one XY face. Both upper values start as None at the physical upper edge.
    """

    column: NativeBlockColumn
    lower_boundaries: list
    upper_column: object = None
    upper_velocity_planes: dict[str, np.ndarray] | None = None


class NativeZLayerFields:
    """Share read-only native fields among columns at one common z interval.

    native_reader is Snapshot.native_reader. read_field() returns its float64
    XYZ array in native units, e.g. (256,256,128), without unit conversion or
    arithmetic. A (block, field_name) is read once while the layer is active.
    clear() releases every cached array before the next z interval.
    """

    def __init__(self, *, native_reader: NativeSnapshotReader):
        self.native_reader = native_reader
        self.z_start = None
        self.fields = {}

    def begin_layer(self, *, z_start: int) -> None:
        """Begin an empty cache; a reader has only one active layer iterator."""
        if self.z_start is not None or self.fields:
            raise RuntimeError('Finish the current z layer before beginning another')
        self.z_start = z_start

    def read_field(self, *, block: NativeBlock, field_name: str) -> np.ndarray:
        """Read a complete native field once, or reuse its existing array."""
        if block.start[2] != self.z_start:
            raise RuntimeError('Requested a native field outside the current z layer')
        key = (block, field_name)
        if key not in self.fields:
            values = self.native_reader.read_block_field(
                block=block,
                field_name=field_name,
            )
            values.flags.writeable = False
            self.fields[key] = values
        return self.fields[key]

    def clear(self) -> None:
        """Release current native fields; keep no arrays from another z layer."""
        self.fields.clear()
        self.z_start = None


class NativeBlockRangeReader:
    """Own native geometry and the selected XY columns, never a full cube.

    snapshot : Snapshot
        Opened yt dataset with original grid, units and optional XY selection.
        Construction reads block metadata only. Native blocks must form a
        Cartesian lattice; shared axis intervals may have different widths.
        One active iter_layers() call owns the current z-layer field cache.

    Example::

        reader = NativeBlockRangeReader(snapshot=snapshot)
        for layer in reader.iter_layers():
            # Reduce this layer's batches, then release it.
            ...
    """

    def __init__(self, *, snapshot: Snapshot):
        self.snapshot = snapshot
        self.native = snapshot.native_reader
        self.z_layer_fields = NativeZLayerFields(native_reader=self.native)
        self.blocks_by_start = {block.start: block for block in self.native.blocks}
        self.axis_edges = self._read_cartesian_block_edges()
        self.columns = self._select_xy_columns()
        self.widths = {}
        for index, name in enumerate(('x', 'y', 'z')):
            self.widths[name] = snapshot.cell_widths[index]
        self.dz_cgs = self.widths['z'].in_cgs()
        dz_cm = float(self.widths['z'].to_value('cm'))
        if not np.isfinite(dz_cm) or dz_cm <= 0:
            raise ValueError('dz_cm must be finite and positive')

    def _read_cartesian_block_edges(self):
        """Check complete shared block faces before requesting cell payloads."""
        edges = []
        for axis in range(3):
            faces = set()
            for block in self.native.blocks:
                faces.add(block.start[axis])
                faces.add(block.start[axis] + block.shape[axis])
            ordered = sorted(faces)
            if ordered[0] != 0 or ordered[-1] != self.snapshot.shape[axis]:
                raise ValueError('Native block faces must cover the original box')
            edges.append(ordered)
        expected_count = 1
        for faces in edges:
            expected_count *= len(faces) - 1
        if (
            expected_count != len(self.native.blocks)
            or len(self.blocks_by_start) != expected_count
        ):
            raise ValueError('Range scanning requires a Cartesian native block lattice')
        for x, y, z in product(edges[0][:-1], edges[1][:-1], edges[2][:-1]):
            block = self.blocks_by_start.get((x, y, z))
            if block is None:
                raise ValueError('Range scanning requires shared native block faces')
            for axis, start in enumerate((x, y, z)):
                position = edges[axis].index(start)
                expected_size = edges[axis][position + 1] - start
                if block.shape[axis] != expected_size:
                    raise ValueError('Native block sizes do not match the shared faces')
        return tuple(edges)

    def _select_xy_columns(self):
        """Intersect the ROI with disjoint native XY columns; keep every z."""
        selected_x_start, selected_x_stop = self.snapshot.xy_region['x']
        selected_y_start, selected_y_stop = self.snapshot.xy_region['y']
        columns = []
        for native_x_start, native_y_start in product(
            self.axis_edges[0][:-1],
            self.axis_edges[1][:-1],
        ):
            first = self.blocks_by_start[(native_x_start, native_y_start, 0)]
            x_start = max(selected_x_start, native_x_start)
            x_stop = min(selected_x_stop, native_x_start + first.shape[0])
            y_start = max(selected_y_start, native_y_start)
            y_stop = min(selected_y_stop, native_y_start + first.shape[1])
            if x_start >= x_stop or y_start >= y_stop:
                continue
            blocks = []
            for z_start in self.axis_edges[2][:-1]:
                blocks.append(
                    self.blocks_by_start[(native_x_start, native_y_start, z_start)]
                )
            columns.append(
                NativeBlockColumn(
                    x_start=x_start,
                    x_stop=x_stop,
                    y_start=y_start,
                    y_stop=y_stop,
                    blocks=tuple(blocks),
                )
            )
        return tuple(columns)

    def prepare_lower_column_boundaries(self, *, column: NativeBlockColumn):
        """Ascending density pass -> exact incoming XY columns [cm^-2].

        Each array has shape (selected_Nx, selected_Ny). The physical lower
        boundary uses None, not an extra zero. All additions remain cellwise.
        """
        boundaries = []
        incoming_column = None
        for block in column.blocks:
            values = self.native.read_block_field(
                block=block,
                field_name='gasDensity',
            )
            core_xy = self._core_xy_slices(column=column, block=block)
            density = self.native.attach_density_units(values=values[core_xy])
            cell_column = self._cell_hydrogen_columns(density=density)
            boundaries.append(incoming_column)
            cumulative = continue_column_sum(
                cell_column=cell_column,
                incoming_column=incoming_column,
            )
            incoming_column = cumulative[:, :, -1].copy()
            del values, density, cell_column, cumulative
        return boundaries

    def prepare_column_states(self, *, columns) -> list[NativeColumnState]:
        """Run each selected column's ascending density pass in original order.

        columns is a tuple selected from self.columns. Each returned state
        owns its exact lower-boundary NH arrays; no 3-D density cube survives
        this pass. Upper carries remain None until that column is processed.
        """
        states = []
        for column in columns:
            lower_boundaries = self.prepare_lower_column_boundaries(column=column)
            states.append(
                NativeColumnState(
                    column=column,
                    lower_boundaries=lower_boundaries,
                )
            )
        return states

    def iter_layers(self, *, columns=None):
        """Yield owned four-field layers, sharing native fields at each z.

        columns optionally selects a tuple from self.columns, e.g.
        columns=reader.columns[:1] keeps one XY column with its complete z.
        All selected columns are processed from high to low z. Their cellwise
        column sums remain independent; xy neighbours use the original box.
        Finishing, closing or raising clears the current raw-field cache.
        """
        selected_columns = self.columns
        if columns is not None:
            selected_columns = tuple(columns)
        if not selected_columns:
            return
        states = self.prepare_column_states(columns=selected_columns)
        for z_index in reversed(range(len(selected_columns[0].blocks))):
            z_start = selected_columns[0].blocks[z_index].start[2]
            self.z_layer_fields.begin_layer(z_start=z_start)
            try:
                for state in states:
                    layer = self.calculate_layer(
                        state=state,
                        z_index=z_index,
                    )
                    yield layer
                    del layer
            finally:
                self.z_layer_fields.clear()

    def calculate_layer(self, *, state: NativeColumnState, z_index: int) -> TableInputLayer:
        """Read one padded block, derive four inputs, and advance its carries.

        state is from prepare_column_states(); z_index selects its native
        block. Returned arrays own XYZ float64 values in g/cm^3, K, cm^-2 and
        s^-1, while state keeps only the next upper NH/velocity face arrays.
        Unit conversion and every per-cell operation retain their old order.
        """
        column = state.column
        block = column.blocks[z_index]
        lower_block = None
        if z_index > 0:
            lower_block = column.blocks[z_index - 1]
        native_fields, next_upper_planes, core = self._read_layer_with_neighbours(
            column=column,
            block=block,
            lower_block=lower_block,
            upper_velocity_planes=state.upper_velocity_planes,
        )
        fields = self.native.attach_field_units(native_fields=native_fields)
        del native_fields
        shielding_column, next_upper_column = self._shielding_column_for_layer(
            density=fields['gas', 'density'][core],
            lower_column=state.lower_boundaries[z_index],
            upper_column=state.upper_column,
        )
        gradient = velocity_gradient(fields).to('s**-1')
        density = fields['gas', 'density'].to('g/cm**3')
        layer = TableInputLayer(
            density_g_cm3=np.array(density[core], order='C'),
            temperature_QUOKKA_K=np.array(
                fields['boxlib', 'temperature'][core],
                dtype=float,
                order='C',
            ),
            shielding_NH_cm2=shielding_column,
            velocity_gradient_s=np.array(gradient[core], order='C'),
            x_start=column.x_start,
            y_start=column.y_start,
            z_start=block.start[2],
        )
        state.upper_column = next_upper_column
        state.upper_velocity_planes = next_upper_planes
        return layer

    def _cell_hydrogen_columns(self, *, density):
        """Retained density with units -> nH*dz [cm^-2], using process rules."""
        hydrogen_density = hydrogen_number_density(density)
        values = hydrogen_density.to_value('cm**-3')
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('nH_cm3 must be finite and nonnegative')
        return hydrogen_density * self.dz_cgs

    def _shielding_column_for_layer(self, *, density, lower_column, upper_column):
        """Restore both full-z sums and preserve high-z reciprocal first."""
        cell_column = self._cell_hydrogen_columns(density=density)
        reversed_upper = continue_column_sum(
            cell_column=np.flip(cell_column, axis=2),
            incoming_column=upper_column,
        )
        plus_column = np.flip(reversed_upper, axis=2)
        next_upper = plus_column[:, :, 0].copy()
        inverse_column_sum = 1.0 / plus_column
        del reversed_upper, plus_column
        minus_column = continue_column_sum(
            cell_column=cell_column,
            incoming_column=lower_column,
        )
        inverse_column_sum = inverse_column_sum + 1.0 / minus_column
        shielding_column = (2 / inverse_column_sum).to('cm**-2')
        return np.array(shielding_column, order='C'), next_upper

    def _read_layer_with_neighbours(
        self,
        *,
        column,
        block,
        lower_block,
        upper_velocity_planes,
    ):
        """Assemble the core and only the velocity faces used by divergence.

        Raw arrays have shape (Nx + 2, Ny + 2, Nz + z_halos), in native units.
        Unused diagonal padding stays NaN: retained dvx/dx needs only x-face
        vx, dvy/dy only y-face vy, and dvz/dz only the two z-face vz planes.
        """
        loaded_z_start = max(0, block.start[2] - 1)
        loaded_z_stop = min(
            self.snapshot.shape[2],
            block.start[2] + block.shape[2] + 1,
        )
        shape = (
            column.x_stop - column.x_start + 2,
            column.y_stop - column.y_start + 2,
            loaded_z_stop - loaded_z_start,
        )
        raw = {}
        for name in NATIVE_FIELD_NAMES:
            raw[name] = np.full(shape=shape, fill_value=np.nan, dtype=float)
        core_z = slice(
            block.start[2] - loaded_z_start,
            block.start[2] - loaded_z_start + block.shape[2],
        )
        core = (slice(1, -1), slice(1, -1), core_z)
        next_upper_planes = self._read_core_fields(
            column=column,
            block=block,
            raw=raw,
            core=core,
        )
        self._read_x_velocity_faces(
            column=column,
            block=block,
            raw=raw,
            core_z=core_z,
        )
        self._read_y_velocity_faces(
            column=column,
            block=block,
            raw=raw,
            core_z=core_z,
        )
        self._read_z_velocity_planes(
            column=column,
            block=block,
            lower_block=lower_block,
            raw=raw,
            upper_velocity_planes=upper_velocity_planes,
        )
        return raw, next_upper_planes, core

    def _core_xy_slices(self, *, column, block):
        """Map the retained global XY rectangle to block-local slices."""
        return (
            slice(column.x_start - block.start[0], column.x_stop - block.start[0]),
            slice(column.y_start - block.start[1], column.y_stop - block.start[1]),
        )

    def _read_core_fields(self, *, column, block, raw, core):
        """Read each complete core field once; reuse its internal ROI faces."""
        xy = self._core_xy_slices(column=column, block=block)
        next_upper_planes = {}
        for name in NATIVE_FIELD_NAMES:
            values = self.z_layer_fields.read_field(
                block=block,
                field_name=name,
            )
            raw[name][core] = values[xy]
            if name in ('gasDensity', 'z-GasMomentum'):
                next_upper_planes[name] = values[xy][:, :, 0].copy()
            self._copy_faces_inside_core(
                column=column,
                block=block,
                field_name=name,
                values=values,
                destination=raw[name],
                core_z=core[2],
            )
            del values
        return next_upper_planes

    def _copy_faces_inside_core(
        self,
        *,
        column,
        block,
        field_name,
        values,
        destination,
        core_z,
    ):
        """Use the current buffer when an ROI face is inside this native block."""
        core_x, core_y = self._core_xy_slices(column=column, block=block)
        if field_name in ('gasDensity', 'x-GasMomentum'):
            faces = (
                (0, (column.x_start - 1) % self.snapshot.shape[0]),
                (-1, column.x_stop % self.snapshot.shape[0]),
            )
            for target, global_x in faces:
                local_x = global_x - block.start[0]
                if 0 <= local_x < block.shape[0]:
                    destination[target, 1:-1, core_z] = values[local_x, core_y, :]
        if field_name in ('gasDensity', 'y-GasMomentum'):
            faces = (
                (0, (column.y_start - 1) % self.snapshot.shape[1]),
                (-1, column.y_stop % self.snapshot.shape[1]),
            )
            for target, global_y in faces:
                local_y = global_y - block.start[1]
                if 0 <= local_y < block.shape[1]:
                    destination[1:-1, target, core_z] = values[core_x, local_y, :]

    def _block_at_xy_cell(self, *, x, y, z_start):
        """Locate one original-box XY cell in the shared native z layer."""
        x_position = int(np.searchsorted(self.axis_edges[0], x, side='right')) - 1
        y_position = int(np.searchsorted(self.axis_edges[1], y, side='right')) - 1
        return self.blocks_by_start[
            (
                self.axis_edges[0][x_position],
                self.axis_edges[1][y_position],
                z_start,
            )
        ]

    def _read_x_velocity_faces(self, *, column, block, raw, core_z):
        """Read external rho/x-momentum once per distinct x neighbour block."""
        groups = {}
        faces = (
            (0, (column.x_start - 1) % self.snapshot.shape[0]),
            (-1, column.x_stop % self.snapshot.shape[0]),
        )
        for target, global_x in faces:
            neighbour = self._block_at_xy_cell(
                x=global_x,
                y=column.y_start,
                z_start=block.start[2],
            )
            if neighbour == block:
                continue
            if neighbour not in groups:
                groups[neighbour] = []
            groups[neighbour].append((target, global_x - neighbour.start[0]))
        for neighbour, requested_faces in groups.items():
            selected_y = slice(
                column.y_start - neighbour.start[1],
                column.y_stop - neighbour.start[1],
            )
            for name in ('gasDensity', 'x-GasMomentum'):
                values = self.z_layer_fields.read_field(
                    block=neighbour,
                    field_name=name,
                )
                for target, local_x in requested_faces:
                    raw[name][target, 1:-1, core_z] = values[local_x, selected_y, :]
                del values

    def _read_y_velocity_faces(self, *, column, block, raw, core_z):
        """Read external rho/y-momentum once per distinct y neighbour block."""
        groups = {}
        faces = (
            (0, (column.y_start - 1) % self.snapshot.shape[1]),
            (-1, column.y_stop % self.snapshot.shape[1]),
        )
        for target, global_y in faces:
            neighbour = self._block_at_xy_cell(
                x=column.x_start,
                y=global_y,
                z_start=block.start[2],
            )
            if neighbour == block:
                continue
            if neighbour not in groups:
                groups[neighbour] = []
            groups[neighbour].append((target, global_y - neighbour.start[1]))
        for neighbour, requested_faces in groups.items():
            selected_x = slice(
                column.x_start - neighbour.start[0],
                column.x_stop - neighbour.start[0],
            )
            for name in ('gasDensity', 'y-GasMomentum'):
                values = self.z_layer_fields.read_field(
                    block=neighbour,
                    field_name=name,
                )
                for target, local_y in requested_faces:
                    raw[name][1:-1, target, core_z] = values[selected_x, local_y, :]
                del values

    def _read_z_velocity_planes(
        self,
        *,
        column,
        block,
        lower_block,
        raw,
        upper_velocity_planes,
    ):
        """Keep internal z differences central; physical z faces stay one-sided."""
        if lower_block is not None:
            xy = self._core_xy_slices(column=column, block=lower_block)
            for name in ('gasDensity', 'z-GasMomentum'):
                plane = self.native.read_block_z_plane(
                    block=lower_block,
                    field_name=name,
                    local_z=lower_block.shape[2] - 1,
                )
                raw[name][1:-1, 1:-1, 0] = plane[xy]
        if block.start[2] + block.shape[2] < self.snapshot.shape[2]:
            for name, plane in upper_velocity_planes.items():
                raw[name][1:-1, 1:-1, -1] = plane


def continue_column_sum(*, cell_column, incoming_column):
    """Continue exact cellwise z additions in unit-aware XYZ [cm^-2] arrays.

    incoming_column is the saved XY accumulator, or None at the physical edge.
    Prepend it as the first element; adding it after a local sum changes rounding.
    """
    if incoming_column is None:
        return np.cumsum(cell_column, axis=2)
    extended = np.concatenate(
        (incoming_column[:, :, None], cell_column),
        axis=2,
    )
    cumulative = np.cumsum(extended, axis=2)
    return cumulative[:, :, 1:]
