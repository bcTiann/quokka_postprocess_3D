"""Read original full-z FAB columns through complete-field NumPy reads.

Each np.fromfile call reads one complete native FAB field. The first core
supplies its raw neighbour-cache values; complete density/Mx/My fields from
remaining FABs supply the other global periodic faces. Only owned C-order
float64 core arrays and independent read-only faces escape the reader.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from math import prod
from threading import Lock
from time import perf_counter

import numpy as np


FIELD_NAMES = (
    "gasDensity",
    "temperature",
    "x-GasMomentum",
    "y-GasMomentum",
    "z-GasMomentum",
)


@dataclass(frozen=True)
class NativeColumnRawTile:
    """Owned full-z raw core and its true global periodic XY neighbours.

    fields contains the five FIELD_NAMES as writable C-order float64 arrays
    [nx,ny,nz]. x_*_faces has shape [2,ny,nz], with index0 at
    (x_start-1)%global_nx and index1 at x_stop%global_nx. y_*_faces has shape
    [nx,2,nz], with index0 at (y_start-1)%global_ny and index1 at y_stop%global_ny.
    Face arrays own their bytes, are read-only, and never alias core arrays.
    The caller may pop unused momentum fields after calculating the gradient.
    """

    x_start: int
    x_stop: int
    y_start: int
    y_stop: int
    fields: dict[str, np.ndarray]
    x_density_faces: np.ndarray
    x_momentum_faces: np.ndarray
    y_density_faces: np.ndarray
    y_momentum_faces: np.ndarray

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Global half-open (x_start,x_stop,y_start,y_stop)."""
        return self.x_start, self.x_stop, self.y_start, self.y_stop


class NativeColumnReader:
    """Read regular level-zero original FABs without a converted disk copy.

    Input metadata comes from NativeSnapshotReader. Calls to read_tile are
    sequential; up to read_workers complete field reads/copies run concurrently.
    Native source dtype/order are retained until copying into ordinary RAM.
    All source files are opened read-only by np.fromfile and closed per call.
    """

    def __init__(self, field_reader, *, read_workers=5):
        if not isinstance(read_workers, (int, np.integer)) or read_workers < 1:
            raise ValueError("read_workers must be a positive integer")
        self._native_shape = tuple(int(size) for size in field_reader._native_shape)
        self._native_dtype = np.dtype(field_reader._native_dtype)
        self._field_positions = dict(field_reader._field_positions)
        self._blocks = tuple(field_reader.blocks)
        self._read_workers = int(read_workers)
        self._tiles = self._validate_geometry()
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=self._read_workers)
        self._stats_lock = Lock()
        self._read_calls = 0
        self._requested_bytes = 0
        self._received_bytes = 0
        self._phase_wall_seconds = {}
        self._tile_statistics = []
        self._face_cache_ready = False
        self._face_cache_bytes = 0
        self._x_faces = {}
        self._y_faces = {}

    def _validate_geometry(self):
        """Reject missing, overlapping or unequal native block tilings."""
        if len(self._native_shape) != 3 or any(size < 1 for size in self._native_shape):
            raise ValueError("Native columns require a positive three-dimensional domain")
        if self._native_dtype.kind != "f" or self._native_dtype.itemsize not in (4, 8):
            raise ValueError("Native columns require float32 or float64 source fields")
        if any(name not in self._field_positions for name in FIELD_NAMES):
            raise ValueError("Native columns require all five raw fields")
        positions = [self._field_positions[name] for name in FIELD_NAMES]
        if any(not isinstance(position, (int, np.integer)) or position < 0 for position in positions):
            raise ValueError("Native field positions must be nonnegative integers")
        if len(set(positions)) != len(FIELD_NAMES):
            raise ValueError("Native field positions must be distinct")
        if not self._blocks or len({tuple(block.shape) for block in self._blocks}) != 1:
            raise ValueError("Native columns require nonempty equal-shape FABs")
        block_shape = tuple(self._blocks[0].shape)
        if len(block_shape) != 3 or any(size < 1 for size in block_shape):
            raise ValueError("Native FAB shape must have three positive dimensions")
        if any(size % width for size, width in zip(self._native_shape, block_shape)):
            raise ValueError("Native FAB dimensions must divide the complete domain")
        expected_bytes = prod(block_shape) * self._native_dtype.itemsize
        grouped = {}
        starts = set()
        for block in self._blocks:
            start = tuple(block.start)
            if block.field_bytes != expected_bytes or block.payload_offset < 0:
                raise ValueError("Native FAB size or payload offset is inconsistent")
            if len(start) != 3 or any(
                index < 0 or index % width or index + width > size
                for index, width, size in zip(start, block_shape, self._native_shape)
            ):
                raise ValueError("Native FAB must align with the complete regular grid")
            if start in starts:
                raise ValueError("Native FABs overlap at a duplicate grid position")
            starts.add(start)
            x_start, y_start, _ = start
            bounds = (
                x_start,
                x_start + block_shape[0],
                y_start,
                y_start + block_shape[1],
            )
            grouped.setdefault(bounds, []).append(block)
        expected_starts = set(
            product(
                *(
                    range(0, size, width)
                    for size, width in zip(self._native_shape, block_shape)
                )
            )
        )
        if starts != expected_starts:
            raise ValueError("Native FABs must cover the complete domain without gaps")
        return {
            bounds: tuple(sorted(blocks, key=lambda block: block.start[2]))
            for bounds, blocks in sorted(grouped.items())
        }

    @property
    def native_shape(self):
        return self._native_shape

    def iter_tile_bounds(self):
        yield from self._tiles

    def _read_field(self, *, block, field_name):
        """Read one owned native-dtype XYZ field; fail before exposing short data."""
        offset = block.payload_offset + self._field_positions[field_name] * block.field_bytes
        values = None
        try:
            values = np.fromfile(
                file=block.filename,
                dtype=self._native_dtype,
                count=prod(block.shape),
                offset=offset,
            )
        finally:
            with self._stats_lock:
                self._read_calls += 1
                self._requested_bytes += block.field_bytes
                self._received_bytes += 0 if values is None else values.nbytes
        if values.size != prod(block.shape):
            raise ValueError(f"Short native field read: {block.filename}, {field_name}")
        return values.reshape(block.shape, order="F")

    def _copy_core(self, *, block, field_name, destination):
        native_xyz = self._read_field(block=block, field_name=field_name)
        z_start = block.start[2]
        z_stop = z_start + block.shape[2]
        np.copyto(
            destination[:, :, z_start:z_stop],
            native_xyz,
            casting="safe",
        )

    @staticmethod
    def _wait_for_tasks(tasks):
        """Drain all writers before returning a tile or raising their first error."""
        error = None
        for task in tasks:
            try:
                task.result()
            except BaseException as caught:
                if error is None:
                    error = caught
        if error is not None:
            raise error

    def _allocate_faces(self):
        nx, ny, nz = self._native_shape
        x_planes = {
            position % nx
            for bounds in self._tiles
            for position in (bounds[0] - 1, bounds[1])
        }
        y_planes = {
            position % ny
            for bounds in self._tiles
            for position in (bounds[2] - 1, bounds[3])
        }
        self._x_faces = {
            name: {plane: np.empty((ny, nz)) for plane in sorted(x_planes)}
            for name in ("gasDensity", "x-GasMomentum")
        }
        self._y_faces = {
            name: {plane: np.empty((nx, nz)) for plane in sorted(y_planes)}
            for name in ("gasDensity", "y-GasMomentum")
        }
        self._face_cache_bytes = sum(
            values.nbytes
            for axis in (self._x_faces, self._y_faces)
            for cache in axis.values()
            for values in cache.values()
        )

    def _copy_faces(self, *, native_xyz, field_name, start):
        """Copy raw global planes, independent of source layout or later mutation."""
        x_start, y_start, z_start = start
        x_stop = x_start + native_xyz.shape[0]
        y_stop = y_start + native_xyz.shape[1]
        z_stop = z_start + native_xyz.shape[2]
        for plane, destination in self._x_faces.get(field_name, {}).items():
            if x_start <= plane < x_stop:
                destination[y_start:y_stop, z_start:z_stop] = native_xyz[plane - x_start, :, :]
        for plane, destination in self._y_faces.get(field_name, {}).items():
            if y_start <= plane < y_stop:
                destination[x_start:x_stop, z_start:z_stop] = native_xyz[:, plane - y_start, :]

    def _read_faces(self, *, block, field_name):
        native_xyz = self._read_field(block=block, field_name=field_name)
        self._copy_faces(
            native_xyz=native_xyz,
            field_name=field_name,
            start=block.start,
        )

    def _ensure_face_cache(self, *, bounds, fields):
        if self._face_cache_ready:
            return
        started = perf_counter()
        self._allocate_faces()
        first_start = (bounds[0], bounds[2], 0)
        for name in ("gasDensity", "x-GasMomentum", "y-GasMomentum"):
            self._copy_faces(
                native_xyz=fields[name],
                field_name=name,
                start=first_start,
            )
        first_blocks = {block.start for block in self._tiles[bounds]}
        tasks = [
            self._executor.submit(
                self._read_faces,
                block=block,
                field_name=name,
            )
            for block in self._blocks if block.start not in first_blocks
            for name in ("gasDensity", "x-GasMomentum", "y-GasMomentum")
        ]
        self._wait_for_tasks(tasks)
        for axis in (self._x_faces, self._y_faces):
            for cache in axis.values():
                for array in cache.values():
                    array.setflags(write=False)
        self._face_cache_ready = True
        self._record_wall("halo_cache_fill", perf_counter() - started)

    def _copy_tile_faces(self, bounds):
        x_start, x_stop, y_start, y_stop = bounds
        nx, ny, nz = self._native_shape
        x_indices = ((x_start - 1) % nx, x_stop % nx)
        y_indices = ((y_start - 1) % ny, y_stop % ny)
        faces = []
        for name in ("gasDensity", "x-GasMomentum"):
            array = np.empty((2, y_stop - y_start, nz))
            for side, plane in enumerate(x_indices):
                array[side, :, :] = self._x_faces[name][plane][y_start:y_stop, :]
            array.setflags(write=False)
            faces.append(array)
        for name in ("gasDensity", "y-GasMomentum"):
            array = np.empty((x_stop - x_start, 2, nz))
            for side, plane in enumerate(y_indices):
                array[:, side, :] = self._y_faces[name][plane][x_start:x_stop, :]
            array.setflags(write=False)
            faces.append(array)
        return faces

    def _record_wall(self, phase, seconds):
        self._phase_wall_seconds[phase] = self._phase_wall_seconds.get(phase, 0.0) + seconds

    def read_tile(self, bounds):
        """Return one exact full-z native XY tile; no source view escapes."""
        if self._closed:
            raise RuntimeError("NativeColumnReader is closed")
        bounds = tuple(bounds)
        if len(bounds) != 4 or any(not isinstance(value, (int, np.integer)) for value in bounds):
            raise ValueError("Native tile bounds must contain four integer cell indices")
        if bounds not in self._tiles:
            raise ValueError("bounds must exactly match an iter_tile_bounds() column")
        started = perf_counter()
        shape = (
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            self._native_shape[2],
        )
        fields = {
            name: np.empty(shape, dtype=np.float64, order="C")
            for name in FIELD_NAMES
        }
        core_started = perf_counter()
        tasks = [
            self._executor.submit(
                self._copy_core,
                block=block,
                field_name=name,
                destination=fields[name],
            )
            for block in self._tiles[bounds]
            for name in FIELD_NAMES
        ]
        self._wait_for_tasks(tasks)
        self._record_wall("core_read_and_gather", perf_counter() - core_started)
        self._ensure_face_cache(bounds=bounds, fields=fields)
        faces_started = perf_counter()
        faces = self._copy_tile_faces(bounds)
        self._record_wall("returned_face_copies", perf_counter() - faces_started)
        elapsed = perf_counter() - started
        self._record_wall("read_tiles_total", elapsed)
        self._tile_statistics.append({
            "bounds": list(bounds),
            "read_tile_seconds": elapsed,
            "owned_core_bytes": sum(array.nbytes for array in fields.values()),
            "owned_returned_face_bytes": sum(array.nbytes for array in faces),
        })
        return NativeColumnRawTile(
            x_start=bounds[0],
            x_stop=bounds[1],
            y_start=bounds[2],
            y_stop=bounds[3],
            fields=fields,
            x_density_faces=faces[0],
            x_momentum_faces=faces[1],
            y_density_faces=faces[2],
            y_momentum_faces=faces[3],
        )

    @property
    def stats(self):
        """API payload counts, not filesystem traffic; phase clocks include copies."""
        with self._stats_lock:
            counts = {
                "read_calls": self._read_calls,
                "requested_bytes": self._requested_bytes,
                "received_bytes": self._received_bytes,
            }
        return {
            "method": "numpy-fromfile",
            "native_shape_xyz": list(self._native_shape),
            "native_dtype": self._native_dtype.str,
            "native_FAB_count": len(self._blocks),
            "native_XY_tile_count": len(self._tiles),
            "read_workers": self._read_workers,
            "unique_required_field_payload_bytes": (
                prod(self._native_shape)
                * len(FIELD_NAMES)
                * self._native_dtype.itemsize
            ),
            **counts,
            "read_call_unit": "One np.fromfile call reading a complete FAB field",
            "phase_wall_seconds": dict(self._phase_wall_seconds),
            "immutable_face_cache_bytes": self._face_cache_bytes,
            "completed_tile_reads": list(self._tile_statistics),
            "physical_storage_bytes": None,
            "physical_storage_note": "Requested payload bytes do not measure filesystem traffic",
            "source_layout": "Native XYZ Fortran order; x varies fastest",
            "output_layout": "Owned float64 C-order XYZ; z varies fastest",
            "disk_copy_created": False,
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True)
        self._x_faces.clear()
        self._y_faces.clear()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("NativeColumnReader is closed")
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
