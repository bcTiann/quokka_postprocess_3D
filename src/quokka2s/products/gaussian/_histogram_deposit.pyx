# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Indexed brackets and stable integer sorting with exact compact scatter.

Each node/profile retains the caller's corner-major multiply/add order.
Only an int32 node-to-slot index spans the full grid; luminosities and touched
IDs grow with the number of nodes visited by a batch. Returned arrays own data.
"""
from libc.math cimport isfinite
from libc.stdint cimport int32_t, int64_t

import numpy as np
import threading
import time


_thread_local = threading.local()


cdef class AxisIndexer:
    """Read-only piecewise-uniform geometry with exact-neighbor correction.

    Segments begin at supplied node indices; the final interval of a segment
    may be shorter to include its exact endpoint. Initialization validates the
    entire axis once. A guess uses nominal spacing, but final brackets and all
    interpolation weights use the actual original float64 axis values.
    """

    cdef object _axis_owner
    cdef object _starts_owner
    cdef object _inverse_steps_owner
    cdef const double[::1] _axis
    cdef const int64_t[::1] _starts
    cdef const double[::1] _inverse_steps
    cdef readonly Py_ssize_t node_count
    cdef readonly Py_ssize_t segment_count

    def __init__(self, axis, segment_starts, nominal_steps):
        values = np.asarray(axis)
        if values.ndim != 1 or values.dtype != np.dtype(np.float64):
            raise ValueError("An indexed axis must be a one-dimensional float64 array")
        if not values.flags.c_contiguous or values.flags.writeable:
            raise ValueError("An indexed axis must be contiguous and read-only")
        if values.size < 2 or not np.isfinite(values).all() or not np.all(np.diff(values) > 0.0):
            raise ValueError("An indexed axis must be finite and strictly increasing")
        starts = np.asarray(segment_starts, dtype=np.int64).copy()
        steps = np.asarray(nominal_steps, dtype=np.float64).copy()
        if starts.ndim != 1 or steps.shape != starts.shape or starts.size not in (1, 3):
            raise ValueError("Indexed geometry requires one or three matching segments")
        if starts[0] != 0 or not np.all(np.diff(starts) > 0) or starts[starts.size - 1] >= values.size - 1:
            raise ValueError("Segment starts must be increasing interior node indices beginning at zero")
        if not np.isfinite(steps).all() or np.any(steps <= 0.0):
            raise ValueError("Indexed nominal spacings must be finite and positive")
        inverse_steps = np.reciprocal(steps)
        if not np.isfinite(inverse_steps).all():
            raise ValueError("Indexed nominal spacings must have finite reciprocals")
        stops = np.concatenate((starts[1:], np.array([values.size - 1], dtype=np.int64)))
        for start, stop, step in zip(starts, stops, steps):
            interior = values[start:stop]
            expected = values[start] + np.arange(stop - start, dtype=np.float64) * step
            scale = max(1.0, abs(values[start]), abs(values[stop]))
            tolerance = 8.0 * float(np.spacing(scale))
            if np.any(np.abs(interior - expected) > tolerance):
                raise ValueError("The supplied axis is not the declared piecewise-uniform geometry")
            regular_endpoint = values[start] + (stop - start) * step
            if values[stop] > regular_endpoint + tolerance:
                raise ValueError("A final axis interval exceeds its declared nominal spacing")
        starts.flags.writeable = False
        inverse_steps.flags.writeable = False
        self._axis_owner = values
        self._starts_owner = starts
        self._inverse_steps_owner = inverse_steps
        self._axis = values
        self._starts = starts
        self._inverse_steps = inverse_steps
        self.node_count = values.size
        self.segment_count = starts.size

    @property
    def description(self):
        return {
            "nodes": self.node_count,
            "segment_starts": self._starts_owner.tolist(),
            "nominal_steps": np.reciprocal(self._inverse_steps_owner).tolist(),
            "validated_read_only": True,
        }

    cdef inline Py_ssize_t _lower_bracket(self, double value) noexcept nogil:
        """Match bounded searchsorted(right)-1 using actual adjacent nodes."""
        cdef Py_ssize_t segment = 0
        cdef Py_ssize_t start
        cdef Py_ssize_t stop
        cdef Py_ssize_t lower
        cdef double coordinate
        if self.segment_count == 3:
            if value >= self._axis[self._starts[2]]:
                segment = 2
            elif value >= self._axis[self._starts[1]]:
                segment = 1
        start = self._starts[segment]
        if segment + 1 < self.segment_count:
            stop = self._starts[segment + 1]
        else:
            stop = self.node_count - 1
        coordinate = (value - self._axis[start]) * self._inverse_steps[segment]
        if coordinate <= 0.0:
            lower = start
        elif coordinate >= stop - start:
            lower = stop - 1
        else:
            lower = start + <Py_ssize_t>coordinate
        if lower < 0:
            lower = 0
        elif lower >= self.node_count - 1:
            lower = self.node_count - 2
        # Arithmetic only proposes a node; these comparisons define the result.
        while lower > 0 and value < self._axis[lower]:
            lower -= 1
        while lower < self.node_count - 2 and value >= self._axis[lower + 1]:
            lower += 1
        return lower

    def brackets(self, const double[:] values):
        """Diagnostic exact lower indices and weights for an in-domain vector."""
        cdef Py_ssize_t count = values.shape[0]
        cdef Py_ssize_t index
        cdef Py_ssize_t lower
        cdef Py_ssize_t bad_position = -1
        cdef int64_t[::1] lower_values
        cdef double[::1] fractions
        lower_owner = np.empty(count, dtype=np.int64)
        fractions_owner = np.empty(count, dtype=np.float64)
        lower_values = lower_owner
        fractions = fractions_owner
        with nogil:
            for index in range(count):
                if not isfinite(values[index]) or values[index] < self._axis[0] or values[index] > self._axis[self.node_count - 1]:
                    bad_position = index
                    break
                lower = self._lower_bracket(values[index])
                lower_values[index] = lower
                fractions[index] = (values[index] - self._axis[lower]) / (self._axis[lower + 1] - self._axis[lower])
        if bad_position >= 0:
            raise ValueError(f"Invalid indexed coordinate at position {bad_position}")
        return lower_owner, fractions_owner

cdef class Workspace:
    """Reuse a full-grid slot index and batch-sized luminosity storage.

    node_to_slot uses four bytes per grid node, with -1 meaning untouched.
    Compact arrays grow geometrically when a batch visits more nodes than an
    earlier batch. Only touched index entries are cleared after each call.
    """

    cdef readonly Py_ssize_t node_count
    cdef readonly Py_ssize_t max_profiles
    cdef object _weights_owner
    cdef object _node_to_slot_owner
    cdef object _touched_owner
    cdef double[:, ::1] _weights
    cdef int32_t[::1] _node_to_slot
    cdef int64_t[::1] _touched
    cdef readonly Py_ssize_t slot_capacity
    cdef Py_ssize_t _touched_count
    cdef bint _in_use
    cdef object _owner_thread
    cdef AxisIndexer _mu_indexer
    cdef AxisIndexer _sigma_indexer
    cdef object _sort_scratch_owner
    cdef object _sort_counts_owner
    cdef int64_t[::1] _sort_scratch
    cdef int32_t[::1] _sort_counts
    cdef bint _radix_enabled
    cdef bint _measure_sorting
    cdef readonly object sort_statistics

    def __init__(self, Py_ssize_t node_count, Py_ssize_t max_profiles=12, *,
                 AxisIndexer mu_indexer, AxisIndexer sigma_indexer,
                 sort_backend="radix", measure_sorting=False):
        if node_count < 1 or max_profiles < 1:
            raise ValueError("node_count and max_profiles must be positive")
        if node_count > 2147483647:
            raise ValueError("The compact workspace supports at most int32_max grid nodes")
        if node_count != mu_indexer.node_count * sigma_indexer.node_count:
            raise ValueError("Interpolation axes do not match this Workspace")
        self._mu_indexer = mu_indexer
        self._sigma_indexer = sigma_indexer
        self.node_count = node_count
        self.max_profiles = max_profiles
        self._weights_owner = np.empty((0, max_profiles), dtype=np.float64)
        self._node_to_slot_owner = np.full(node_count, -1, dtype=np.int32)
        self._touched_owner = np.empty(0, dtype=np.int64)
        self._weights = self._weights_owner
        self._node_to_slot = self._node_to_slot_owner
        self._touched = self._touched_owner
        self.slot_capacity = 0
        self._touched_count = 0
        self._in_use = False
        self._owner_thread = threading.get_ident()
        if sort_backend not in ("numpy", "radix"):
            raise ValueError("Unknown sorted-node backend")
        # Current grids fit two 12-bit digits. Larger supported int32 grids
        # retain the original exact NumPy integer sort.
        self._radix_enabled = sort_backend == "radix" and node_count <= 16777216
        self._measure_sorting = bool(measure_sorting)
        self._sort_scratch_owner = np.empty(0, dtype=np.int64)
        self._sort_counts_owner = np.empty(8192 if self._radix_enabled else 0, dtype=np.int32)
        self._sort_scratch = self._sort_scratch_owner
        self._sort_counts = self._sort_counts_owner
        self.sort_statistics = {
            "requested_backend": sort_backend,
            "radix_eligible_grid": bool(self._radix_enabled),
            "node_count": int(node_count),
            "measured": bool(measure_sorting),
            "calls": 0,
            "output_nodes": 0,
            "sort_seconds": 0.0,
            "workspace_seconds": 0.0,
            "radix_calls": 0,
            "numpy_calls": 0,
        }

    @property
    def allocated_bytes(self):
        return (self._weights_owner.nbytes + self._node_to_slot_owner.nbytes
                + self._touched_owner.nbytes + self._sort_scratch_owner.nbytes
                + self._sort_counts_owner.nbytes)

    @property
    def pending_count(self):
        return self._touched_count

    def visited_count(self):
        """Small-fixture diagnostic; normal calls do not scan the grid index."""
        return int(np.count_nonzero(self._node_to_slot_owner >= 0))

    cdef void _grow_slots(self, Py_ssize_t profile_count) except *:
        """Grow compact storage without changing existing slots or values.

        Both allocations complete before replacing workspace owners, so a
        MemoryError leaves the old touched list available for final cleanup.
        Only initialized profile columns need copying during an active call.
        """
        cdef Py_ssize_t new_capacity
        cdef object new_weights
        cdef object new_touched
        cdef object new_sort_scratch
        new_capacity = min(self.node_count, max(1024, self.slot_capacity * 2))
        new_weights = np.empty((new_capacity, self.max_profiles), dtype=np.float64)
        new_touched = np.empty(new_capacity, dtype=np.int64)
        new_sort_scratch = np.empty(new_capacity if self._radix_enabled else 0, dtype=np.int64)
        if self._touched_count:
            new_weights[:self._touched_count, :profile_count] = self._weights_owner[:self._touched_count, :profile_count]
            new_touched[:self._touched_count] = self._touched_owner[:self._touched_count]
        self._weights_owner = new_weights
        self._touched_owner = new_touched
        self._weights = self._weights_owner
        self._touched = self._touched_owner
        self._sort_scratch_owner = new_sort_scratch
        self._sort_scratch = self._sort_scratch_owner
        self.slot_capacity = new_capacity

    cdef inline Py_ssize_t _claim_node(
        self,
        int64_t node,
        Py_ssize_t profile_count,
    ) except -1 nogil:
        """Return one compact slot, initializing it on this call's first visit."""
        cdef Py_ssize_t slot = self._node_to_slot[node]
        cdef Py_ssize_t profile
        if slot >= 0:
            return slot
        if self._touched_count == self.slot_capacity:
            with gil:
                self._grow_slots(profile_count)
        slot = self._touched_count
        self._node_to_slot[node] = <int32_t>slot
        self._touched[slot] = node
        self._touched_count += 1
        for profile in range(profile_count):
            self._weights[slot, profile] = 0.0
        return slot

    cdef void _clear(self) noexcept nogil:
        cdef Py_ssize_t index
        for index in range(self._touched_count):
            self._node_to_slot[self._touched[index]] = -1
        self._touched_count = 0

    cdef void _radix_sort_touched(self) noexcept nogil:
        """Sort unique nonnegative IDs below 2**24 using two stable LSD passes.

        Floating-point weights have already been fully accumulated. Only the
        integer touched list moves here; the node-to-slot map remains fixed.
        Every scratch allocation was completed when compact slots grew.
        """
        cdef Py_ssize_t index
        cdef Py_ssize_t bucket
        cdef Py_ssize_t low_total = 0
        cdef Py_ssize_t high_total = 0
        cdef Py_ssize_t count
        cdef Py_ssize_t slot
        cdef int64_t node
        for bucket in range(8192):
            self._sort_counts[bucket] = 0
        for index in range(self._touched_count):
            node = self._touched[index]
            self._sort_counts[node & 4095] += 1
            self._sort_counts[4096 + (node >> 12)] += 1
        for bucket in range(4096):
            count = self._sort_counts[bucket]
            self._sort_counts[bucket] = <int32_t>low_total
            low_total += count
            count = self._sort_counts[4096 + bucket]
            self._sort_counts[4096 + bucket] = <int32_t>high_total
            high_total += count
        for index in range(self._touched_count):
            node = self._touched[index]
            bucket = node & 4095
            slot = self._sort_counts[bucket]
            self._sort_scratch[slot] = node
            self._sort_counts[bucket] += 1
        for index in range(self._touched_count):
            node = self._sort_scratch[index]
            bucket = 4096 + (node >> 12)
            slot = self._sort_counts[bucket]
            self._touched[slot] = node
            self._sort_counts[bucket] += 1

    def scatter_parameters(
        self,
        const int64_t[:] rows,
        const double[:] centers,
        const double[:] log_sigmas,
        const double[:] mu_axis,
        const double[:] log_sigma_axis,
        const double[:, :] luminosity,
    ):
        """Prepare selected cells and return sorted owned node weights.

        rows/centers/log_sigmas each have R entries. The Python caller retains
        the original active/fallback masks and supplies NumPy's logarithms.
        All selected coordinates must be finite and inside their grid axes.
        luminosity is (B,P), with rows selecting the R original batch cells.

        Four corner-major passes match the previous concatenated contributions.
        Only four R-length bracket/coefficient arrays are allocated; no 4R
        node, row or coefficient arrays are formed. Sorting touched IDs before
        gathering preserves the previous sorted output and diagnostic sum order.
        """
        cdef Py_ssize_t selected_count = rows.shape[0]
        cdef Py_ssize_t profile_count = luminosity.shape[1]
        cdef Py_ssize_t row_count = luminosity.shape[0]
        cdef Py_ssize_t mu_count = mu_axis.shape[0]
        cdef Py_ssize_t sigma_count = log_sigma_axis.shape[0]
        cdef Py_ssize_t index
        cdef Py_ssize_t profile
        cdef Py_ssize_t corner
        cdef Py_ssize_t lower_mu
        cdef Py_ssize_t lower_sigma
        cdef Py_ssize_t output_count
        cdef Py_ssize_t slot
        cdef Py_ssize_t bad_position = -1
        cdef int error_kind = 0
        cdef int64_t node
        cdef int64_t row
        cdef double mu_fraction
        cdef double sigma_fraction
        cdef double fraction
        cdef double weighted_light
        cdef object mu_lower_owner
        cdef object sigma_lower_owner
        cdef object mu_fraction_owner
        cdef object sigma_fraction_owner
        cdef int64_t[::1] mu_lower
        cdef int64_t[::1] sigma_lower
        cdef double[::1] mu_fractions
        cdef double[::1] sigma_fractions
        cdef object output_nodes_owner
        cdef object output_weights_owner
        cdef int64_t[::1] output_nodes
        cdef double[:, ::1] output_weights
        cdef double call_started = 0.0
        cdef double sort_started = 0.0
        cdef bint used_radix = False

        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("A Workspace must remain in its creating thread")
        if self._in_use:
            raise RuntimeError("A Workspace cannot be used reentrantly")
        if centers.shape[0] != selected_count or log_sigmas.shape[0] != selected_count:
            raise ValueError("rows, centers and log_sigmas must have equal lengths")
        if profile_count > self.max_profiles:
            raise ValueError("luminosity has more profiles than this Workspace")
        if mu_count < 2 or sigma_count < 2:
            raise ValueError("Both interpolation axes must have at least two nodes")
        if self.node_count % sigma_count or self.node_count // sigma_count != mu_count:
            raise ValueError("Interpolation grid does not match this Workspace")

        if (
            &mu_axis[0] != &self._mu_indexer._axis[0]
            or &log_sigma_axis[0] != &self._sigma_indexer._axis[0]
            or mu_axis.strides[0] != self._mu_indexer._axis.strides[0]
            or log_sigma_axis.strides[0] != self._sigma_indexer._axis.strides[0]
        ):
            raise ValueError("Workspace axes must match its initialized read-only geometry")

        self._in_use = True
        if self._measure_sorting:
            call_started = time.perf_counter()
        try:
            mu_lower_owner = np.empty(selected_count, dtype=np.int64)
            sigma_lower_owner = np.empty(selected_count, dtype=np.int64)
            mu_fraction_owner = np.empty(selected_count, dtype=np.float64)
            sigma_fraction_owner = np.empty(selected_count, dtype=np.float64)
            mu_lower = mu_lower_owner
            sigma_lower = sigma_lower_owner
            mu_fractions = mu_fraction_owner
            sigma_fractions = sigma_fraction_owner
            with nogil:
                # Immutable geometry was fully validated at initialization.
                if error_kind == 0:
                    for index in range(selected_count):
                        row = rows[index]
                        if row < 0 or row >= row_count:
                            error_kind = 3
                            bad_position = index
                            break
                        if not isfinite(centers[index]) or centers[index] < mu_axis[0] or centers[index] > mu_axis[mu_count - 1]:
                            error_kind = 4
                            bad_position = index
                            break
                        if not isfinite(log_sigmas[index]) or log_sigmas[index] < log_sigma_axis[0] or log_sigmas[index] > log_sigma_axis[sigma_count - 1]:
                            error_kind = 5
                            bad_position = index
                            break
                        lower_mu = self._mu_indexer._lower_bracket(centers[index])
                        lower_sigma = self._sigma_indexer._lower_bracket(log_sigmas[index])
                        mu_lower[index] = lower_mu
                        sigma_lower[index] = lower_sigma
                        mu_fractions[index] = (centers[index] - mu_axis[lower_mu]) / (mu_axis[lower_mu + 1] - mu_axis[lower_mu])
                        sigma_fractions[index] = (log_sigmas[index] - log_sigma_axis[lower_sigma]) / (log_sigma_axis[lower_sigma + 1] - log_sigma_axis[lower_sigma])
                if error_kind == 0:
                    for corner in range(4):
                        for index in range(selected_count):
                            mu_fraction = mu_fractions[index]
                            sigma_fraction = sigma_fractions[index]
                            if corner == 0:
                                node = sigma_lower[index] * mu_count + mu_lower[index]
                                fraction = (1.0 - mu_fraction) * (1.0 - sigma_fraction)
                            elif corner == 1:
                                node = sigma_lower[index] * mu_count + mu_lower[index] + 1
                                fraction = mu_fraction * (1.0 - sigma_fraction)
                            elif corner == 2:
                                node = (sigma_lower[index] + 1) * mu_count + mu_lower[index]
                                fraction = (1.0 - mu_fraction) * sigma_fraction
                            else:
                                node = (sigma_lower[index] + 1) * mu_count + mu_lower[index] + 1
                                fraction = mu_fraction * sigma_fraction
                            if fraction == 0.0:
                                continue
                            slot = self._claim_node(node, profile_count)
                            row = rows[index]
                            for profile in range(profile_count):
                                weighted_light = luminosity[row, profile] * fraction
                                self._weights[slot, profile] = self._weights[slot, profile] + weighted_light
            if error_kind:
                if error_kind == 3:
                    raise IndexError(f"Invalid row index at selected position {bad_position}")
                raise ValueError(f"Invalid interpolation axis/coordinate kind {error_kind} at position {bad_position}")

            output_count = self._touched_count
            # The float64 accumulation above is unchanged. Sorting only IDs
            # selects the same final ascending node/weight output as NumPy.
            used_radix = self._radix_enabled and output_count >= 1024
            if self._measure_sorting:
                sort_started = time.perf_counter()
            if used_radix:
                with nogil:
                    self._radix_sort_touched()
            else:
                self._touched_owner[:output_count].sort()
            if self._measure_sorting:
                self.sort_statistics["calls"] += 1
                self.sort_statistics["output_nodes"] += output_count
                self.sort_statistics["sort_seconds"] += time.perf_counter() - sort_started
                self.sort_statistics["radix_calls" if used_radix else "numpy_calls"] += 1
            output_nodes_owner = np.empty(output_count, dtype=np.int64)
            output_weights_owner = np.empty((output_count, profile_count), dtype=np.float64)
            output_nodes = output_nodes_owner
            output_weights = output_weights_owner
            with nogil:
                for index in range(output_count):
                    node = self._touched[index]
                    slot = self._node_to_slot[node]
                    output_nodes[index] = node
                    for profile in range(profile_count):
                        output_weights[index, profile] = self._weights[slot, profile]
            return output_nodes_owner, output_weights_owner
        finally:
            with nogil:
                self._clear()
            self._in_use = False
            if self._measure_sorting:
                self.sort_statistics["workspace_seconds"] += time.perf_counter() - call_started

    def scatter(
        self,
        const int64_t[:] node_ids,
        const int64_t[:] rows,
        const double[:] fractions,
        const double[:, :] luminosity,
    ):
        """Return (unique first-visit IDs, owned C weights[K,P]).

        IDs and rows are int64 vectors, fractions float64, and luminosity is
        (R,P) float64, with input strides/read-only flags retained. The caller
        retains its existing finite/nonnegative preparation policy. Nothing is
        clipped or skipped, including zero coefficients. An invalid ID/row
        raises after clearing every node touched before the error.
        """
        cdef Py_ssize_t item_count = node_ids.shape[0]
        cdef Py_ssize_t profile_count = luminosity.shape[1]
        cdef Py_ssize_t row_count = luminosity.shape[0]
        cdef Py_ssize_t index
        cdef Py_ssize_t profile
        cdef Py_ssize_t output_count
        cdef Py_ssize_t slot
        cdef Py_ssize_t bad_position = -1
        cdef int error_kind = 0
        cdef int64_t node
        cdef int64_t row
        cdef double weighted_light
        cdef object output_nodes_owner
        cdef object output_weights_owner
        cdef int64_t[::1] output_nodes
        cdef double[:, ::1] output_weights

        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("A Workspace must remain in its creating thread")
        if self._in_use:
            raise RuntimeError("A Workspace cannot be used reentrantly")
        if rows.shape[0] != item_count or fractions.shape[0] != item_count:
            raise ValueError("node_ids, rows and fractions must have equal lengths")
        if profile_count > self.max_profiles:
            raise ValueError("luminosity has more profiles than this Workspace")

        self._in_use = True
        try:
            with nogil:
                for index in range(item_count):
                    node = node_ids[index]
                    row = rows[index]
                    if node < 0 or node >= self.node_count:
                        error_kind = 1
                        bad_position = index
                        break
                    if row < 0 or row >= row_count:
                        error_kind = 2
                        bad_position = index
                        break
                    slot = self._claim_node(node, profile_count)
                    for profile in range(profile_count):
                        weighted_light = luminosity[row, profile] * fractions[index]
                        self._weights[slot, profile] = self._weights[slot, profile] + weighted_light
            if error_kind:
                kind = "node ID" if error_kind == 1 else "row index"
                raise IndexError(f"Invalid {kind} at contribution position {bad_position}")

            output_count = self._touched_count
            output_nodes_owner = np.empty(output_count, dtype=np.int64)
            output_weights_owner = np.empty((output_count, profile_count), dtype=np.float64)
            output_nodes = output_nodes_owner
            output_weights = output_weights_owner
            with nogil:
                for index in range(output_count):
                    node = self._touched[index]
                    slot = self._node_to_slot[node]
                    output_nodes[index] = node
                    for profile in range(profile_count):
                        output_weights[index, profile] = self._weights[slot, profile]
            return output_nodes_owner, output_weights_owner
        finally:
            with nogil:
                self._clear()
            self._in_use = False
