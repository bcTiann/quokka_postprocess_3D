# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Sparse fixed-layer Gaussian deposition with compiled integer-key lookup.

Logical keys are (sigma_layer, mu_index). Hash buckets and compact luminosity
rows grow with occupied keys; no allocation depends on an enclosing grid size.
"""
from libc.math cimport exp, floor, isfinite, log
from libc.stdint cimport int64_t, uint64_t

import numpy as np
import threading


cdef inline uint64_t mixed_key(int64_t layer, int64_t index) noexcept nogil:
    cdef uint64_t value = <uint64_t>index
    value ^= (<uint64_t>layer + <uint64_t>0x9e3779b97f4a7c15)
    value ^= value >> 30
    value *= <uint64_t>0xbf58476d1ce4e5b9
    value ^= value >> 27
    value *= <uint64_t>0x94d049bb133111eb
    return value ^ (value >> 31)


cdef class _KeyStorage:
    """Own open-address buckets and stable compact rows in creation order."""

    cdef Py_ssize_t size
    cdef Py_ssize_t capacity
    cdef Py_ssize_t hash_capacity
    cdef Py_ssize_t profiles
    cdef object keys_owner
    cdef object weights_owner
    cdef object buckets_owner
    cdef object hash_owner
    cdef int64_t[:, ::1] keys
    cdef double[:, ::1] weights
    cdef int64_t[::1] buckets
    cdef int64_t[::1] slots
    cdef Py_ssize_t max_occupied
    cdef Py_ssize_t growth_count

    def __init__(self, Py_ssize_t profiles):
        if profiles < 1 or profiles > 40:
            raise ValueError('Compact rows require between one and forty profiles')
        self.profiles = profiles
        self.size = 0
        self.capacity = 0
        self.hash_capacity = 1024
        self.max_occupied = 0
        self.growth_count = 0
        self.keys_owner = np.empty((0, 2), dtype=np.int64)
        self.weights_owner = np.empty((0, profiles), dtype=np.float64)
        self.buckets_owner = np.empty(0, dtype=np.int64)
        self.hash_owner = np.full(self.hash_capacity, -1, dtype=np.int64)
        self.keys = self.keys_owner
        self.weights = self.weights_owner
        self.buckets = self.buckets_owner
        self.slots = self.hash_owner

    cdef void _grow_rows(self) except *:
        cdef Py_ssize_t new_capacity = max(1024, int(self.capacity) * 2)
        new_keys = np.empty((new_capacity, 2), dtype=np.int64)
        new_weights = np.empty((new_capacity, self.profiles), dtype=np.float64)
        new_buckets = np.empty(new_capacity, dtype=np.int64)
        if self.size:
            new_keys[:self.size] = self.keys_owner[:self.size]
            new_weights[:self.size] = self.weights_owner[:self.size]
            new_buckets[:self.size] = self.buckets_owner[:self.size]
        self.keys_owner = new_keys
        self.weights_owner = new_weights
        self.buckets_owner = new_buckets
        self.keys = new_keys
        self.weights = new_weights
        self.buckets = new_buckets
        self.capacity = new_capacity
        self.growth_count += 1

    cdef void _grow_hash(self) except *:
        cdef Py_ssize_t new_capacity = int(self.hash_capacity) * 2
        cdef Py_ssize_t row
        cdef Py_ssize_t bucket
        cdef Py_ssize_t mask = new_capacity - 1
        cdef int64_t[::1] new_slots
        owner = np.full(new_capacity, -1, dtype=np.int64)
        new_slots = owner
        with nogil:
            for row in range(self.size):
                bucket = <Py_ssize_t>(mixed_key(self.keys[row, 0], self.keys[row, 1]) & mask)
                while new_slots[bucket] >= 0:
                    bucket = (bucket + 1) & mask
                new_slots[bucket] = row
                self.buckets[row] = bucket
        self.hash_owner = owner
        self.slots = owner
        self.hash_capacity = new_capacity
        self.growth_count += 1

    cdef inline Py_ssize_t claim(self, int64_t layer, int64_t index) except -1 nogil:
        cdef Py_ssize_t mask = self.hash_capacity - 1
        cdef Py_ssize_t bucket = <Py_ssize_t>(mixed_key(layer, index) & mask)
        cdef Py_ssize_t row
        cdef Py_ssize_t profile
        while self.slots[bucket] >= 0:
            row = self.slots[bucket]
            if self.keys[row, 0] == layer and self.keys[row, 1] == index:
                return row
            bucket = (bucket + 1) & mask
        if self.size == self.capacity:
            with gil:
                self._grow_rows()
        if (self.size + 1) * 10 > self.hash_capacity * 6:
            with gil:
                self._grow_hash()
            mask = self.hash_capacity - 1
            bucket = <Py_ssize_t>(mixed_key(layer, index) & mask)
            while self.slots[bucket] >= 0:
                bucket = (bucket + 1) & mask
        row = self.size
        self.keys[row, 0] = layer
        self.keys[row, 1] = index
        self.buckets[row] = bucket
        self.slots[bucket] = row
        for profile in range(self.profiles):
            self.weights[row, profile] = 0.0
        self.size += 1
        if self.size > self.max_occupied:
            self.max_occupied = self.size
        return row

    cdef void clear(self) noexcept nogil:
        cdef Py_ssize_t row
        for row in range(self.size):
            self.slots[self.buckets[row]] = -1
        self.size = 0

    def sorted_arrays(self, Py_ssize_t profiles, bint drop_zero=True):
        if profiles < 1 or profiles > self.profiles:
            raise ValueError('Requested profile count exceeds compact storage')
        if not np.isfinite(self.weights_owner[:self.size, :profiles]).all():
            raise FloatingPointError('Accumulated node luminosity is nonfinite')
        rows = np.arange(self.size, dtype=np.int64)
        if drop_zero:
            rows = rows[np.any(self.weights_owner[:self.size, :profiles] != 0.0, axis=1)]
        order = np.lexsort((self.keys_owner[rows, 1], self.keys_owner[rows, 0]))
        rows = rows[order]
        return (
            np.ascontiguousarray(self.keys_owner[rows]),
            np.ascontiguousarray(self.weights_owner[rows, :profiles]),
        )

    def statistics(self):
        return {
            'occupied_nodes': int(self.size),
            'maximum_occupied_nodes': int(self.max_occupied),
            'slot_capacity': int(self.capacity),
            'hash_capacity': int(self.hash_capacity),
            'payload_bytes': int(self.weights_owner.nbytes),
            'key_bytes': int(self.keys_owner.nbytes),
            'hash_bytes': int(self.hash_owner.nbytes + self.buckets_owner.nbytes),
            'allocated_bytes': int(self.weights_owner.nbytes + self.keys_owner.nbytes + self.hash_owner.nbytes + self.buckets_owner.nbytes),
            'growth_count': int(self.growth_count),
            'memory_scope': 'Persistent NumPy buffers; not process RSS or resize peak',
            'logical_enclosing_grid_allocated': False,
        }


def _validated_parameters(mu_spacing_ratio, log_sigma_step, sigma_floor, mu_anchor):
    values = [float(mu_spacing_ratio), float(log_sigma_step), float(sigma_floor), float(mu_anchor)]
    if not all(np.isfinite(value) for value in values):
        raise ValueError('Layered coordinate parameters must be finite')
    ratio, step, lower, anchor = values
    if ratio <= 0.0 or step <= 0.0 or lower <= 0.0:
        raise ValueError('Require positive mu ratio, log-sigma step and sigma floor')
    log_floor = float(np.log(lower))
    minimum_spacing = ratio * lower
    if not np.isfinite(minimum_spacing) or minimum_spacing <= 0.0:
        raise ValueError('The minimum mu spacing must be representable')
    if anchor + minimum_spacing <= anchor or anchor - minimum_spacing >= anchor:
        raise ValueError('The minimum mu nodes are indistinguishable at this anchor')
    if log_floor + step <= log_floor:
        raise ValueError('Adjacent logarithmic sigma layers are indistinguishable')
    return ratio, step, lower, anchor, log_floor


cdef inline double sigma_coordinate(
    int64_t layer, double log_floor, double step, double sigma_floor,
) noexcept nogil:
    cdef double coordinate
    cdef double value
    if layer < 0:
        return -1.0
    if layer == 0:
        return sigma_floor
    coordinate = log_floor + layer * step
    if not isfinite(coordinate):
        return -1.0
    value = exp(coordinate)
    if not isfinite(value) or value <= 0.0:
        return -1.0
    return value


cdef inline int mean_bracket(
    double mean, double spacing, double anchor,
    int64_t *index, double *fraction,
) noexcept nogil:
    cdef double coordinate = (mean - anchor) / spacing
    cdef double left
    cdef double right
    cdef double previous_left
    cdef double following_right
    cdef int64_t lower
    if not isfinite(coordinate) or coordinate >= 9223372036854774784.0 or coordinate <= -9223372036854774784.0:
        return 1
    lower = <int64_t>floor(coordinate)
    left = anchor + lower * spacing
    right = anchor + (lower + 1) * spacing
    if not isfinite(left) or not isfinite(right) or right <= left:
        return 1
    while mean < left:
        lower -= 1
        right = left
        left = anchor + lower * spacing
        if not isfinite(left) or right <= left:
            return 1
    while mean >= right:
        lower += 1
        left = right
        right = anchor + (lower + 1) * spacing
        if not isfinite(right) or right <= left:
            return 1
    previous_left = anchor + (lower - 1) * spacing
    following_right = anchor + (lower + 2) * spacing
    if not isfinite(previous_left) or not isfinite(following_right) or previous_left >= left or following_right <= right:
        return 1
    index[0] = lower
    fraction[0] = (mean - left) / (right - left)
    if not isfinite(fraction[0]) or fraction[0] < 0.0 or fraction[0] > 1.0:
        return 1
    return 0


cdef class Workspace:
    """One worker's bounded batch scratch and occupied-node hash table."""

    cdef _KeyStorage nodes
    cdef _KeyStorage layers
    cdef double ratio
    cdef double step
    cdef double lower
    cdef double anchor
    cdef double log_floor
    cdef Py_ssize_t profiles
    cdef object owner_thread
    cdef bint in_use
    cdef Py_ssize_t last_output_nodes
    cdef Py_ssize_t metadata_bytes
    cdef Py_ssize_t calls

    def __init__(self, mu_spacing_ratio=0.1, log_sigma_step=1.0 / 32,
                 sigma_floor=1.0e-5, mu_anchor=0.0, max_profiles=12):
        values = _validated_parameters(mu_spacing_ratio, log_sigma_step, sigma_floor, mu_anchor)
        self.ratio = values[0]
        self.step = values[1]
        self.lower = values[2]
        self.anchor = values[3]
        self.log_floor = values[4]
        if type(max_profiles) is not int or max_profiles < 1 or max_profiles > 40:
            raise ValueError('max_profiles must be an integer from one to forty')
        self.profiles = max_profiles
        self.nodes = _KeyStorage(self.profiles)
        self.layers = _KeyStorage(3)
        self.owner_thread = threading.get_ident()
        self.in_use = False
        self.last_output_nodes = 0
        self.metadata_bytes = 0
        self.calls = 0

    cdef inline Py_ssize_t layer_slot(self, int64_t layer) except -1 nogil:
        cdef Py_ssize_t slot = self.layers.claim(layer, 0)
        cdef double value = self.layers.weights[slot, 0]
        if value == 0.0:
            value = sigma_coordinate(layer, self.log_floor, self.step, self.lower)
            self.layers.weights[slot, 0] = value
            self.layers.weights[slot, 1] = self.ratio * value
        return slot

    def deposit(
        self,
        const double[::1] centers,
        const double[::1] sigmas,
        const double[:, ::1] luminosity,
    ):
        """Return sorted owned keys[U,2] and luminosities[U,P].

        Inactive rows may have unavailable coordinates. Active rows require
        finite coordinates and sigma at least the unchanged physical floor.
        Each call is corner-major, then original row order, then profile order.
        """
        cdef Py_ssize_t count = centers.shape[0]
        cdef Py_ssize_t profiles = luminosity.shape[1]
        cdef Py_ssize_t row
        cdef Py_ssize_t profile
        cdef Py_ssize_t corner
        cdef Py_ssize_t slot
        cdef Py_ssize_t low_layer_slot
        cdef Py_ssize_t high_layer_slot
        cdef int64_t layer
        cdef int64_t index = 0
        cdef double coordinate
        cdef double sigma_low
        cdef double sigma_high
        cdef double spacing_low
        cdef double spacing_high
        cdef double alpha_low = 0.0
        cdef double alpha_high = 0.0
        cdef double beta
        cdef double denominator
        cdef double fraction
        cdef double weighted_light
        cdef bint active
        cdef int error = 0
        cdef Py_ssize_t bad_row = -1
        cdef int64_t[::1] layer_ids
        cdef int64_t[::1] low_ids
        cdef int64_t[::1] high_ids
        cdef double[::1] low_fractions
        cdef double[::1] high_fractions
        cdef double[::1] layer_fractions
        if self.in_use or self.owner_thread != threading.get_ident():
            raise RuntimeError('A layered workspace belongs to one worker and cannot be reentered')
        if sigmas.shape[0] != count or luminosity.shape[0] != count:
            raise ValueError('Gaussian vectors and luminosity rows must have equal length')
        if profiles < 1 or profiles > self.profiles:
            raise ValueError('Luminosity profile count exceeds this workspace')
        self.in_use = True
        try:
            layer_owner = np.full(count, -1, dtype=np.int64)
            low_owner = np.empty(count, dtype=np.int64)
            high_owner = np.empty(count, dtype=np.int64)
            low_fraction_owner = np.empty(count, dtype=np.float64)
            high_fraction_owner = np.empty(count, dtype=np.float64)
            layer_fraction_owner = np.empty(count, dtype=np.float64)
            layer_ids = layer_owner
            low_ids = low_owner
            high_ids = high_owner
            low_fractions = low_fraction_owner
            high_fractions = high_fraction_owner
            layer_fractions = layer_fraction_owner
            self.metadata_bytes = count * 48
            with nogil:
                for row in range(count):
                    active = False
                    for profile in range(profiles):
                        if not isfinite(luminosity[row, profile]) or luminosity[row, profile] < 0.0:
                            error = 1
                            bad_row = row
                            break
                        if luminosity[row, profile] != 0.0:
                            active = True
                    if error:
                        break
                    if not active:
                        continue
                    if not isfinite(centers[row]) or not isfinite(sigmas[row]) or sigmas[row] < self.lower:
                        error = 2
                        bad_row = row
                        break
                    coordinate = (log(sigmas[row]) - self.log_floor) / self.step
                    if not isfinite(coordinate) or coordinate < -1.0 or coordinate >= 9223372036854774784.0:
                        error = 3
                        bad_row = row
                        break
                    layer = <int64_t>floor(coordinate)
                    if layer < 0:
                        layer = 0
                    low_layer_slot = self.layer_slot(layer)
                    high_layer_slot = self.layer_slot(layer + 1)
                    sigma_low = self.layers.weights[low_layer_slot, 0]
                    sigma_high = self.layers.weights[high_layer_slot, 0]
                    if sigma_low <= 0.0 or sigma_high <= sigma_low:
                        error = 3
                        bad_row = row
                        break
                    while sigmas[row] < sigma_low and layer > 0:
                        layer -= 1
                        sigma_high = sigma_low
                        high_layer_slot = low_layer_slot
                        low_layer_slot = self.layer_slot(layer)
                        sigma_low = self.layers.weights[low_layer_slot, 0]
                    while sigmas[row] >= sigma_high:
                        layer += 1
                        sigma_low = sigma_high
                        low_layer_slot = high_layer_slot
                        high_layer_slot = self.layer_slot(layer + 1)
                        sigma_high = self.layers.weights[high_layer_slot, 0]
                        if sigma_high <= sigma_low:
                            error = 3
                            break
                    if error or sigma_low <= 0.0 or not isfinite(sigma_high):
                        error = 3
                        bad_row = row
                        break
                    if sigmas[row] == sigma_low:
                        beta = 0.0
                    else:
                        denominator = self.layers.weights[low_layer_slot, 2]
                        if denominator == 0.0:
                            denominator = log(sigma_high / sigma_low)
                            self.layers.weights[low_layer_slot, 2] = denominator
                        if not isfinite(denominator) or denominator <= 0.0:
                            error = 3
                            bad_row = row
                            break
                        beta = log(sigmas[row] / sigma_low) / denominator
                        if not isfinite(beta) or beta < 0.0 or beta > 1.0:
                            error = 3
                            bad_row = row
                            break
                    spacing_low = self.layers.weights[low_layer_slot, 1]
                    spacing_high = self.layers.weights[high_layer_slot, 1]
                    if not isfinite(spacing_low) or not isfinite(spacing_high) or spacing_low <= 0.0 or spacing_high <= 0.0:
                        error = 4
                        bad_row = row
                        break
                    if mean_bracket(centers[row], spacing_low, self.anchor, &index, &alpha_low):
                        error = 4
                        bad_row = row
                        break
                    low_ids[row] = index
                    if mean_bracket(centers[row], spacing_high, self.anchor, &index, &alpha_high):
                        error = 4
                        bad_row = row
                        break
                    high_ids[row] = index
                    layer_ids[row] = layer
                    low_fractions[row] = alpha_low
                    high_fractions[row] = alpha_high
                    layer_fractions[row] = beta
                if error == 0:
                    for corner in range(4):
                        for row in range(count):
                            layer = layer_ids[row]
                            if layer < 0:
                                continue
                            beta = layer_fractions[row]
                            if corner == 0:
                                index = low_ids[row]
                                fraction = (1.0 - beta) * (1.0 - low_fractions[row])
                            elif corner == 1:
                                index = low_ids[row] + 1
                                fraction = (1.0 - beta) * low_fractions[row]
                            elif corner == 2:
                                layer += 1
                                index = high_ids[row]
                                fraction = beta * (1.0 - high_fractions[row])
                            else:
                                layer += 1
                                index = high_ids[row] + 1
                                fraction = beta * high_fractions[row]
                            if fraction == 0.0:
                                continue
                            slot = self.nodes.claim(layer, index)
                            for profile in range(profiles):
                                weighted_light = luminosity[row, profile] * fraction
                                self.nodes.weights[slot, profile] = self.nodes.weights[slot, profile] + weighted_light
            if error:
                raise ValueError(f'Invalid layered deposition input/coordinate kind {error} at row {bad_row}')
            result = self.nodes.sorted_arrays(profiles)
            self.last_output_nodes = result[0].shape[0]
            self.calls += 1
            return result
        finally:
            with nogil:
                self.nodes.clear()
            self.in_use = False

    def statistics(self):
        record = self.nodes.statistics()
        record['occupied_nodes'] = int(self.last_output_nodes)
        record['pending_nodes'] = int(self.nodes.size)
        record['layer_cache_bytes'] = int(self.layers.statistics()['allocated_bytes'])
        record['layer_cache_occupied'] = int(self.layers.size)
        record['allocated_bytes'] += record['layer_cache_bytes']
        record['last_call_temporary_metadata_bytes'] = int(self.metadata_bytes)
        record['deposit_calls'] = int(self.calls)
        record['coordinate_rule'] = 'Fixed log-sigma layers; each layer uses mu_step=ratio*sigma_layer'
        record['sigma_weight_rule'] = 'log(sigma/sigma_low)/log(sigma_high/sigma_low)'
        record['accumulation_order'] = 'corner, original row, profile'
        return record


cdef class SparseNodeStore:
    """Main-thread node accounts; slots and logical identities stay separate."""

    cdef _KeyStorage storage
    cdef object owner_thread
    cdef bint in_use
    cdef Py_ssize_t merge_calls
    cdef Py_ssize_t increment_rows

    def __init__(self, profile_count=40):
        if type(profile_count) is not int or profile_count < 1 or profile_count > 40:
            raise ValueError('profile_count must be an integer from one to forty')
        self.storage = _KeyStorage(profile_count)
        self.owner_thread = threading.get_ident()
        self.in_use = False
        self.merge_calls = 0
        self.increment_rows = 0

    def merge(
        self,
        const int64_t[:, ::1] keys,
        const int64_t[::1] profile_ids,
        const double[:, ::1] weights,
    ):
        """Add one sorted unique-key increment in its original batch order."""
        cdef Py_ssize_t count = keys.shape[0]
        cdef Py_ssize_t profiles = profile_ids.shape[0]
        cdef Py_ssize_t row
        cdef Py_ssize_t column
        cdef Py_ssize_t previous
        cdef Py_ssize_t slot
        cdef Py_ssize_t target_profile
        cdef int error = 0
        cdef Py_ssize_t bad_row = -1
        if self.in_use or self.owner_thread != threading.get_ident():
            raise RuntimeError('A sparse node store belongs to one main thread')
        if keys.shape[1] != 2 or weights.shape[0] != count or weights.shape[1] != profiles:
            raise ValueError('Sparse increment requires keys[U,2] and weights[U,P]')
        if np.shares_memory(np.asarray(keys), self.storage.keys_owner) or np.shares_memory(np.asarray(weights), self.storage.weights_owner):
            raise ValueError('Sparse increment must not alias the main storage')
        with nogil:
            for column in range(profiles):
                if profile_ids[column] < 0 or profile_ids[column] >= self.storage.profiles:
                    error = 1
                    break
                for previous in range(column):
                    if profile_ids[column] == profile_ids[previous]:
                        error = 1
                        break
            if error == 0:
                for row in range(count):
                    if keys[row, 0] < 0:
                        error = 2
                        bad_row = row
                        break
                    if row > 0 and (keys[row, 0] < keys[row - 1, 0] or (keys[row, 0] == keys[row - 1, 0] and keys[row, 1] <= keys[row - 1, 1])):
                        error = 2
                        bad_row = row
                        break
                    for column in range(profiles):
                        if not isfinite(weights[row, column]) or weights[row, column] < 0.0:
                            error = 3
                            bad_row = row
                            break
                    if error:
                        break
        if error:
            raise ValueError(f'Invalid sorted sparse increment kind {error} at row {bad_row}')
        if count == 0 or profiles == 0:
            return
        self.in_use = True
        try:
            with nogil:
                for row in range(count):
                    slot = self.storage.claim(keys[row, 0], keys[row, 1])
                    for column in range(profiles):
                        target_profile = profile_ids[column]
                        self.storage.weights[slot, target_profile] = self.storage.weights[slot, target_profile] + weights[row, column]
            self.merge_calls += 1
            self.increment_rows += count
        finally:
            self.in_use = False

    def arrays(self):
        """Return owned nonzero accounts sorted by logical (layer,mu) key."""
        if self.in_use:
            raise RuntimeError('Cannot read the sparse store during a merge')
        return self.storage.sorted_arrays(self.storage.profiles)

    def statistics(self):
        record = self.storage.statistics()
        record['merge_calls'] = int(self.merge_calls)
        record['increment_rows'] = int(self.increment_rows)
        record['selected_threads'] = 1
        record['batch_group_order_unchanged'] = True
        record['per_location_arithmetic'] = 'one ordinary float64 addition per increment'
        record['logical_keys'] = '(sigma_layer:int64,mu_index:int64)'
        return record


def coordinates(
    const int64_t[:, ::1] keys,
    *,
    mu_spacing_ratio=0.1,
    log_sigma_step=1.0 / 32,
    sigma_floor=1.0e-5,
    mu_anchor=0.0,
):
    """Decode logical keys into owned centers/sigmas [U], in km/s."""
    values = _validated_parameters(mu_spacing_ratio, log_sigma_step, sigma_floor, mu_anchor)
    cdef double ratio = values[0]
    cdef double step = values[1]
    cdef double lower = values[2]
    cdef double anchor = values[3]
    cdef double log_floor = values[4]
    cdef Py_ssize_t count = keys.shape[0]
    cdef Py_ssize_t row
    cdef double sigma
    cdef double spacing
    cdef double center
    cdef double right
    cdef int error = 0
    cdef double[::1] centers
    cdef double[::1] sigmas
    if keys.shape[1] != 2:
        raise ValueError('Logical keys must have shape (node count,2)')
    centers_owner = np.empty(count, dtype=np.float64)
    sigmas_owner = np.empty(count, dtype=np.float64)
    centers = centers_owner
    sigmas = sigmas_owner
    with nogil:
        for row in range(count):
            sigma = sigma_coordinate(keys[row, 0], log_floor, step, lower)
            spacing = ratio * sigma
            center = anchor + keys[row, 1] * spacing
            if keys[row, 1] == <int64_t>0x7fffffffffffffff:
                error = 1
                break
            right = anchor + (keys[row, 1] + 1) * spacing
            if sigma <= 0.0 or not isfinite(spacing) or spacing <= 0.0 or not isfinite(center) or not isfinite(right) or right <= center:
                error = 1
                break
            centers[row] = center
            sigmas[row] = sigma
    if error:
        raise ValueError('A logical node coordinate is unrepresentable or outside integer capacity')
    return centers_owner, sigmas_owner
