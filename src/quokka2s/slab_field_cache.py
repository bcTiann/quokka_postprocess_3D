"""Bounded reuse of native halo fields across ordered emission slab windows."""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .native_snapshot_reader import NativeSnapshotReader


# Five float64, unit-aware XYZ value arrays; this is not a process RSS limit.
MAX_CACHED_SLAB_FIELD_BYTES = 3 * 1024**3


class SlabFieldCache:
    """Reuse native converted fields for adjacent selected x windows.

    Windows and selected_y_range are global half-open original-box indices.
    Every group retains original z and periodic x/y neighbours. Only the owner
    thread may read; batch workers receive the existing copied SlabArrays.
    """

    def __init__(
        self,
        *,
        native_reader: NativeSnapshotReader,
        selected_slabs,
        selected_y_range: tuple[int, int],
        slabs_per_read: int,
    ):
        if type(slabs_per_read) is not int or slabs_per_read < 1:
            raise ValueError('slabs_per_read must be a positive integer')
        self._native_reader = native_reader
        self._windows = tuple(tuple(window) for window in selected_slabs)
        self._y_range = tuple(selected_y_range)
        self._slabs_per_read = slabs_per_read
        self._owner_thread = threading.get_ident()
        self._next_index = 0
        self._cached_fields = None
        self._cached_x_start = None
        self._cached_stop_index = None
        nx, ny, nz = native_reader._native_shape
        if len(self._y_range) != 2 or any(type(value) is not int for value in self._y_range):
            raise ValueError('selected_y_range must contain two integer bounds')
        if not 0 <= self._y_range[0] < self._y_range[1] <= ny:
            raise ValueError('selected_y_range must lie inside the original box')
        for index, window in enumerate(self._windows):
            if len(window) != 2 or any(type(value) is not int for value in window):
                raise ValueError('Each selected slab must contain two integer bounds')
            if not 0 <= window[0] < window[1] <= nx:
                raise ValueError('Each selected slab must lie inside the original box')
            if index and self._windows[index - 1][1] != window[0]:
                raise ValueError('Selected slabs must be ordered and contiguous')
        selected_ny = self._y_range[1] - self._y_range[0]
        self._field_bytes_per_x_layer = 5 * (selected_ny + 2) * nz * 8

    def close(self):
        """Release retained values; existing caller-held views remain valid."""
        self._cached_fields = None
        self._cached_x_start = None
        self._cached_stop_index = None

    def _choose_group_stop_index(self):
        """Keep only whole selected windows whose five arrays fit the limit."""
        stop_index = min(self._next_index + self._slabs_per_read, len(self._windows))
        group_start = self._windows[self._next_index][0]
        while stop_index > self._next_index + 1:
            group_stop = self._windows[stop_index - 1][1]
            cached_bytes = (group_stop - group_start + 2) * self._field_bytes_per_x_layer
            if cached_bytes <= MAX_CACHED_SLAB_FIELD_BYTES:
                break
            stop_index -= 1
        return stop_index

    def read_slab_fields(self, *, x_start, x_stop, y_start, y_stop):
        """Return this planned slab's (nx+2, ny+2, Nz) fields and scalar widths.

        Cached arrays are read-only views with the original native-reader units.
        A single unshared window uses the unchanged direct read and is not kept.
        """
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError('Slab field reuse belongs to its creating thread')
        if self._next_index >= len(self._windows):
            raise ValueError('No selected slab remains in this reuse context')
        if (x_start, x_stop) != self._windows[self._next_index]:
            raise ValueError('Slab reads must follow the actual selected windows in order')
        if (y_start, y_stop) != self._y_range:
            raise ValueError('Slab field reuse must keep its selected y range')

        if self._cached_fields is None:
            self.close()
            stop_index = self._choose_group_stop_index()
            if stop_index == self._next_index + 1:
                # No sharing benefit, including a single slab above the cap.
                fields = self._native_reader._read_uncached_slab_fields(
                    x_start=x_start,
                    x_stop=x_stop,
                    y_start=y_start,
                    y_stop=y_stop,
                )
                self._next_index += 1
                return fields
            group_stop = self._windows[stop_index - 1][1]
            fields = self._native_reader._read_uncached_slab_fields(
                x_start=x_start,
                x_stop=group_stop,
                y_start=y_start,
                y_stop=y_stop,
            )
            actual_cached_bytes = sum(value.nbytes for value in fields.values() if value.ndim == 3)
            if actual_cached_bytes > MAX_CACHED_SLAB_FIELD_BYTES:
                raise MemoryError('Actual slab field cache exceeds its 3 GiB value limit')
            for value in fields.values():
                if value.ndim == 3:
                    value.flags.writeable = False
            self._cached_fields = fields
            self._cached_x_start = x_start
            self._cached_stop_index = stop_index

        local_start = x_start - self._cached_x_start
        local_stop = x_stop - self._cached_x_start + 2
        slab_fields = {}
        for key, value in self._cached_fields.items():
            if value.ndim == 3:
                slab_fields[key] = value[local_start:local_stop, :, :]
            else:
                slab_fields[key] = value
        self._next_index += 1
        if self._next_index == self._cached_stop_index:
            self.close()
        return slab_fields
