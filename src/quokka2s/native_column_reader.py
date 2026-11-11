"""Read complete native XY block columns without creating an on-disk z copy.

NativeSnapshotReader supplies only dtype, field offsets and uniform FAB geometry.
This reader owns read-only file descriptors, a small raw periodic-face cache and
bounded field-read workers. A returned tile owns five writable C-order float64
XYZ arrays in native units; its independent neighbour faces are read-only.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import prod
import os
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .native_snapshot_reader import NativeBlock, NativeSnapshotReader


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
    """Gather one complete native XY block column at a time, in RAM only.

    field_reader is an initialized NativeSnapshotReader. No dataset or field
    conversion method is called here. Only regular, equal-shape, level-zero FAB
    tilings are accepted; unsupported geometry raises instead of using a disk
    cache or a partial-z fallback. read_tile calls are sequential; their internal
    field reads are parallel. Returned tiles remain owned by the caller.

    The first core supplies some raw faces. Filling the remaining face cache
    rereads density/Mx fields and narrow native-x My rows. Counters include those
    extra reads; unique payload, requested bytes and physical storage traffic
    are different quantities. Source files are opened with O_RDONLY only.
    """

    def __init__(self, field_reader: NativeSnapshotReader, *, read_workers: int = 5):
        if not isinstance(read_workers, (int, np.integer)) or read_workers < 1:
            raise ValueError("read_workers must be a positive integer")
        self._native_shape = tuple(int(size) for size in field_reader._native_shape)
        self._native_dtype = np.dtype(field_reader._native_dtype)
        self._field_positions = dict(field_reader._field_positions)
        self._blocks = tuple(field_reader.blocks)
        self._read_workers = int(read_workers)
        self._tiles = self._validate_geometry()
        self._fds = {}
        self._executor = None
        self._closed = False
        self._face_cache_ready = False
        self._x_density_cache = {}
        self._x_momentum_cache = {}
        self._y_density_cache = {}
        self._y_momentum_cache = {}
        self._stats_lock = Lock()
        self._ranges_by_file = {}
        self._read_statistics = {}
        self._gather_seconds = {}
        self._read_intervals = {}
        self._gather_intervals = {}
        self._phase_wall_seconds = {}
        self._tile_statistics = []
        self._face_cache_bytes = 0
        try:
            for filename in sorted({block.filename for block in self._blocks}):
                self._fds[filename] = os.open(filename, os.O_RDONLY)
            self._executor = ThreadPoolExecutor(max_workers=self._read_workers)
        except BaseException:
            self.close()
            raise

    def _validate_geometry(self):
        """Require a complete, nonoverlapping Cartesian grid of uniform FABs."""
        if len(self._native_shape) != 3 or any(size < 1 for size in self._native_shape):
            raise ValueError("Native columns require a positive three-dimensional domain")
        if self._native_dtype.kind != "f" or self._native_dtype.itemsize not in (4, 8):
            raise ValueError("Native columns require float32 or float64 source fields")
        if any(name not in self._field_positions for name in FIELD_NAMES):
            raise ValueError("Native columns require all five density/temperature/momentum fields")
        positions = [self._field_positions[name] for name in FIELD_NAMES]
        if any(not isinstance(position, (int, np.integer)) or position < 0 for position in positions):
            raise ValueError("Native field positions must be nonnegative integers")
        if len(set(positions)) != len(FIELD_NAMES):
            raise ValueError("Native field positions must be distinct")
        if not self._blocks:
            raise ValueError("Native columns require at least one FAB")
        shapes = {tuple(block.shape) for block in self._blocks}
        if len(shapes) != 1:
            raise ValueError("Native columns currently require equal-shape uniform FABs")
        block_shape = shapes.pop()
        if len(block_shape) != 3 or any(size < 1 for size in block_shape):
            raise ValueError("Native FAB shape must have three positive dimensions")
        if any(size % block_size for size, block_size in zip(self._native_shape, block_shape)):
            raise ValueError("Uniform FAB dimensions must divide the complete native domain")
        expected_bytes = prod(block_shape) * self._native_dtype.itemsize
        starts = set()
        grouped = {}
        for block in self._blocks:
            if block.field_bytes != expected_bytes or block.payload_offset < 0:
                raise ValueError("Native FAB field size or payload offset is inconsistent")
            start = tuple(block.start)
            if len(start) != 3 or any(
                index < 0 or index % block_size or index + block_size > native_size
                for index, block_size, native_size in zip(start, block_shape, self._native_shape)
            ):
                raise ValueError("Native FAB must align with the regular level-zero grid")
            if start in starts:
                raise ValueError("Native FABs overlap at a duplicate grid position")
            starts.add(start)
            x_start, y_start, z_start = start
            bounds = (
                x_start,
                x_start + block_shape[0],
                y_start,
                y_start + block_shape[1],
            )
            grouped.setdefault(bounds, []).append(block)
        expected_starts = {
            (x_start, y_start, z_start)
            for x_start in range(0, self._native_shape[0], block_shape[0])
            for y_start in range(0, self._native_shape[1], block_shape[1])
            for z_start in range(0, self._native_shape[2], block_shape[2])
        }
        if starts != expected_starts:
            raise ValueError("Native FABs must cover the complete domain without gaps")
        return {
            bounds: tuple(sorted(blocks, key=lambda block: block.start[2]))
            for bounds, blocks in sorted(grouped.items())
        }

    @property
    def native_shape(self) -> tuple[int, int, int]:
        """Complete physical XYZ grid, unchanged by RAM storage order."""
        return self._native_shape

    def iter_tile_bounds(self):
        """Yield global half-open native XY columns, each with complete z depth."""
        yield from self._tiles

    def _check_open(self):
        if self._closed:
            raise RuntimeError("NativeColumnReader is closed")

    def _record_wall(self, phase, seconds):
        self._phase_wall_seconds[phase] = self._phase_wall_seconds.get(phase, 0.0) + seconds

    def _pread(self, *, block, field_name, within_field_offset, byte_count, phase):
        """Read one requested source range and fail on incomplete payloads."""
        offset = (
            block.payload_offset
            + self._field_positions[field_name] * block.field_bytes
            + within_field_offset
        )
        started = perf_counter()
        payload = os.pread(self._fds[block.filename], byte_count, offset)
        elapsed = perf_counter() - started
        key = phase, field_name
        with self._stats_lock:
            counters = self._read_statistics.setdefault(
                key,
                {"read_calls": 0, "requested_bytes": 0, "received_bytes": 0, "worker_read_seconds": 0.0},
            )
            counters["read_calls"] += 1
            counters["requested_bytes"] += byte_count
            counters["received_bytes"] += len(payload)
            counters["worker_read_seconds"] += elapsed
            self._ranges_by_file.setdefault(block.filename, []).append((offset, offset + byte_count))
            self._read_intervals.setdefault(key, []).append((started, started + elapsed))
        if len(payload) != byte_count:
            raise ValueError(
                f"Short native field read: {block.filename}, {field_name}, "
                f"offset={offset}, requested={byte_count}, received={len(payload)}"
            )
        return payload

    def _record_gather(self, *, phase, field_name, started, seconds):
        key = phase, field_name
        with self._stats_lock:
            self._gather_seconds[key] = self._gather_seconds.get(key, 0.0) + seconds
            self._gather_intervals.setdefault(key, []).append((started, started + seconds))

    def _read_core_field(self, *, block, field_name, destination):
        """Worker owns one temporary field; future returns no payload array."""
        payload = self._pread(
            block=block,
            field_name=field_name,
            within_field_offset=0,
            byte_count=block.field_bytes,
            phase="core_full_fields",
        )
        started = perf_counter()
        native_xyz = np.frombuffer(payload, dtype=self._native_dtype).reshape(block.shape, order="F")
        z_start = block.start[2]
        z_stop = z_start + block.shape[2]
        np.copyto(destination[:, :, z_start:z_stop], native_xyz, casting="safe")
        self._record_gather(
            phase="core_full_fields",
            field_name=field_name,
            started=started,
            seconds=perf_counter() - started,
        )

    def _wait_for_tasks(self, tasks):
        """Drain every bounded worker before exposing buffers or raising errors."""
        error = None
        for task in tasks:
            try:
                task.result()
            except BaseException as caught:
                if error is None:
                    error = caught
        if error is not None:
            raise error

    def _allocate_face_cache(self):
        nx, ny, nz = self._native_shape
        x_planes = set()
        y_planes = set()
        for x_start, x_stop, y_start, y_stop in self._tiles:
            x_planes.add((x_start - 1) % nx)
            x_planes.add(x_stop % nx)
            y_planes.add((y_start - 1) % ny)
            y_planes.add(y_stop % ny)
        x_planes = sorted(x_planes)
        y_planes = sorted(y_planes)
        self._x_density_cache = {index: np.empty((ny, nz)) for index in x_planes}
        self._x_momentum_cache = {index: np.empty((ny, nz)) for index in x_planes}
        self._y_density_cache = {index: np.empty((nx, nz)) for index in y_planes}
        self._y_momentum_cache = {index: np.empty((nx, nz)) for index in y_planes}
        self._face_cache_bytes = sum(
            array.nbytes
            for cache in self._face_caches()
            for array in cache.values()
        )

    def _face_caches(self):
        return (
            self._x_density_cache,
            self._x_momentum_cache,
            self._y_density_cache,
            self._y_momentum_cache,
        )

    def _copy_first_core_faces(self, *, bounds, fields):
        x_start, x_stop, y_start, y_stop = bounds
        for field_name, cache in (
            ("gasDensity", self._x_density_cache),
            ("x-GasMomentum", self._x_momentum_cache),
        ):
            for plane, destination in cache.items():
                if x_start <= plane < x_stop:
                    destination[y_start:y_stop, :] = fields[field_name][plane - x_start, :, :]
        for field_name, cache in (
            ("gasDensity", self._y_density_cache),
            ("y-GasMomentum", self._y_momentum_cache),
        ):
            for plane, destination in cache.items():
                if y_start <= plane < y_stop:
                    destination[x_start:x_stop, :] = fields[field_name][:, plane - y_start, :]

    def _read_full_field_faces(self, *, block, field_name):
        payload = self._pread(
            block=block,
            field_name=field_name,
            within_field_offset=0,
            byte_count=block.field_bytes,
            phase="halo_full_fields",
        )
        started = perf_counter()
        native_xyz = np.frombuffer(payload, dtype=self._native_dtype).reshape(block.shape, order="F")
        x_start, y_start, z_start = block.start
        x_stop = x_start + block.shape[0]
        y_stop = y_start + block.shape[1]
        z_stop = z_start + block.shape[2]
        x_cache = self._x_density_cache if field_name == "gasDensity" else self._x_momentum_cache
        for plane, destination in x_cache.items():
            if x_start <= plane < x_stop:
                destination[y_start:y_stop, z_start:z_stop] = native_xyz[plane - x_start, :, :]
        if field_name == "gasDensity":
            for plane, destination in self._y_density_cache.items():
                if y_start <= plane < y_stop:
                    destination[x_start:x_stop, z_start:z_stop] = native_xyz[:, plane - y_start, :]
        self._record_gather(
            phase="halo_full_fields",
            field_name=field_name,
            started=started,
            seconds=perf_counter() - started,
        )

    def _read_momentum_y_faces(self, *, block):
        """Native x rows are contiguous; other My cells are unnecessary halos."""
        x_start, y_start, z_start = block.start
        block_nx, block_ny, block_nz = block.shape
        x_stop = x_start + block_nx
        row_bytes = block_nx * self._native_dtype.itemsize
        for plane, destination in self._y_momentum_cache.items():
            if not y_start <= plane < y_start + block_ny:
                continue
            local_y = plane - y_start
            for local_z in range(block_nz):
                row_index = local_z * block_ny + local_y
                payload = self._pread(
                    block=block,
                    field_name="y-GasMomentum",
                    within_field_offset=row_index * row_bytes,
                    byte_count=row_bytes,
                    phase="halo_My_rows",
                )
                started = perf_counter()
                values = np.frombuffer(payload, dtype=self._native_dtype)
                destination[x_start:x_stop, z_start + local_z] = values
                self._record_gather(
                    phase="halo_My_rows",
                    field_name="y-GasMomentum",
                    started=started,
                    seconds=perf_counter() - started,
                )

    def _ensure_face_cache(self, *, bounds, fields):
        if self._face_cache_ready:
            return
        started = perf_counter()
        self._allocate_face_cache()
        self._copy_first_core_faces(bounds=bounds, fields=fields)
        first_blocks = {block.start for block in self._tiles[bounds]}
        tasks = []
        for block in self._blocks:
            if block.start in first_blocks:
                continue
            for field_name in ("gasDensity", "x-GasMomentum"):
                tasks.append(self._executor.submit(
                    self._read_full_field_faces,
                    block=block,
                    field_name=field_name,
                ))
            tasks.append(self._executor.submit(self._read_momentum_y_faces, block=block))
        self._wait_for_tasks(tasks)
        for cache in self._face_caches():
            for array in cache.values():
                array.setflags(write=False)
        self._face_cache_ready = True
        self._record_wall("halo_cache_fill", perf_counter() - started)

    def _copy_tile_faces(self, bounds):
        """Return independent faces, safe after caller mutates any core field."""
        x_start, x_stop, y_start, y_stop = bounds
        nx, ny, nz = self._native_shape
        tile_nx = x_stop - x_start
        tile_ny = y_stop - y_start
        x_indices = (x_start - 1) % nx, x_stop % nx
        y_indices = (y_start - 1) % ny, y_stop % ny
        faces = []
        for cache in (self._x_density_cache, self._x_momentum_cache):
            array = np.empty((2, tile_ny, nz))
            for side, plane in enumerate(x_indices):
                array[side, :, :] = cache[plane][y_start:y_stop, :]
            array.setflags(write=False)
            faces.append(array)
        for cache in (self._y_density_cache, self._y_momentum_cache):
            array = np.empty((tile_nx, 2, nz))
            for side, plane in enumerate(y_indices):
                array[:, side, :] = cache[plane][x_start:x_stop, :]
            array.setflags(write=False)
            faces.append(array)
        return faces

    def read_tile(self, bounds: tuple[int, int, int, int]) -> NativeColumnRawTile:
        """Read one exact native XY tile; no clipping or alternate source path."""
        self._check_open()
        bounds = tuple(bounds)
        if len(bounds) != 4 or any(not isinstance(value, (int, np.integer)) for value in bounds):
            raise ValueError("Native tile bounds must contain four integer cell indices")
        if bounds not in self._tiles:
            raise ValueError("bounds must exactly match an iter_tile_bounds() native XY column")
        tile_started = perf_counter()
        x_start, x_stop, y_start, y_stop = bounds
        shape = x_stop - x_start, y_stop - y_start, self._native_shape[2]
        fields = {name: np.empty(shape, dtype=np.float64, order="C") for name in FIELD_NAMES}
        core_started = perf_counter()
        tasks = [
            self._executor.submit(
                self._read_core_field,
                block=block,
                field_name=name,
                destination=fields[name],
            )
            for block in self._tiles[bounds]
            for name in FIELD_NAMES
        ]
        self._wait_for_tasks(tasks)
        core_seconds = perf_counter() - core_started
        self._record_wall("core_read_and_gather", core_seconds)
        self._ensure_face_cache(bounds=bounds, fields=fields)
        faces_started = perf_counter()
        faces = self._copy_tile_faces(bounds)
        faces_seconds = perf_counter() - faces_started
        self._record_wall("returned_face_copies", faces_seconds)
        elapsed = perf_counter() - tile_started
        self._record_wall("read_tiles_total", elapsed)
        self._tile_statistics.append({
            "bounds": list(bounds),
            "core_read_and_gather_seconds": core_seconds,
            "returned_face_copy_seconds": faces_seconds,
            "read_tile_seconds": elapsed,
            "owned_core_bytes": sum(array.nbytes for array in fields.values()),
            "owned_returned_face_bytes": sum(array.nbytes for array in faces),
        })
        return NativeColumnRawTile(
            x_start=x_start,
            x_stop=x_stop,
            y_start=y_start,
            y_stop=y_stop,
            fields=fields,
            x_density_faces=faces[0],
            x_momentum_faces=faces[1],
            y_density_faces=faces[2],
            y_momentum_faces=faces[3],
        )

    @property
    def stats(self) -> dict:
        """JSON-ready I/O accounting, with explicit parallel-worker versus wall time.

        Active wall seconds are interval unions. Read and gather unions overlap
        when different workers perform them concurrently; they are not additive.
        """
        with self._stats_lock:
            counters = {key: dict(value) for key, value in self._read_statistics.items()}
            gather = dict(self._gather_seconds)
            ranges = {name: tuple(values) for name, values in self._ranges_by_file.items()}
            read_intervals = {key: tuple(values) for key, values in self._read_intervals.items()}
            gather_intervals = {key: tuple(values) for key, values in self._gather_intervals.items()}
        unique_requested_bytes = 0
        for intervals in ranges.values():
            end = -1
            for start, stop in sorted(intervals):
                unique_requested_bytes += max(0, stop - max(start, end))
                end = max(end, stop)
        by_phase_and_field = {}
        for (phase, name), values in counters.items():
            values["worker_gather_seconds"] = gather.get((phase, name), 0.0)
            values["read_active_wall_seconds"] = self._active_wall_seconds(read_intervals[(phase, name)])
            values["gather_active_wall_seconds"] = self._active_wall_seconds(gather_intervals.get((phase, name), ()))
            by_phase_and_field.setdefault(phase, {})[name] = values
        requested_bytes = sum(values["requested_bytes"] for values in counters.values())
        return {
            "native_shape_xyz": list(self._native_shape),
            "native_dtype": self._native_dtype.str,
            "native_FAB_count": len(self._blocks),
            "native_XY_tile_count": len(self._tiles),
            "read_workers": self._read_workers,
            "unique_required_field_payload_bytes": prod(self._native_shape) * len(FIELD_NAMES) * self._native_dtype.itemsize,
            "read_calls": sum(values["read_calls"] for values in counters.values()),
            "requested_bytes": requested_bytes,
            "received_bytes": sum(values["received_bytes"] for values in counters.values()),
            "requested_source_range_union_bytes": unique_requested_bytes,
            "repeated_requested_bytes": requested_bytes - unique_requested_bytes,
            "worker_read_seconds_sum": sum(values["worker_read_seconds"] for values in counters.values()),
            "worker_gather_seconds_sum": sum(gather.values()),
            "read_active_wall_seconds": self._active_wall_seconds(
                interval for intervals in read_intervals.values() for interval in intervals
            ),
            "gather_active_wall_seconds": self._active_wall_seconds(
                interval for intervals in gather_intervals.values() for interval in intervals
            ),
            "active_wall_note": "Read and gather interval unions may overlap and must not be summed",
            "phase_wall_seconds": dict(self._phase_wall_seconds),
            "by_phase_and_field": by_phase_and_field,
            "immutable_face_cache_bytes": self._face_cache_bytes,
            "completed_tile_reads": list(self._tile_statistics),
            "physical_storage_bytes": None,
            "physical_storage_note": "API requests and their range union do not measure filesystem storage traffic",
        }

    @staticmethod
    def _active_wall_seconds(intervals):
        seconds = 0.0
        previous_stop = float("-inf")
        for start, stop in sorted(intervals):
            seconds += max(0.0, stop - max(start, previous_stop))
            previous_stop = max(previous_stop, stop)
        return seconds

    def close(self):
        """Release workers, readonly descriptors and cached faces; retain stats."""
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        for source_fd in self._fds.values():
            os.close(source_fd)
        self._fds.clear()
        for cache in self._face_caches():
            cache.clear()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
