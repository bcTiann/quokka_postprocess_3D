# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Exact dense node-major updates with prefetch and optional row parallelism."""

from cython.parallel cimport prange
from libc.stdint cimport int64_t, uintptr_t

import numpy as np


cdef extern from *:
    """
    #ifdef _OPENMP
    #include <omp.h>
    static int quokka_openmp_version(void) { return _OPENMP; }
    static int quokka_team_size(void) { return omp_get_num_threads(); }
    #else
    static int quokka_openmp_version(void) { return 0; }
    static int quokka_team_size(void) { return 1; }
    #endif
    static inline void quokka_prefetch_write(const void *address) {
    #if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(address, 1, 3);
    #endif
    }
    """
    int quokka_openmp_version() noexcept nogil
    int quokka_team_size() noexcept nogil
    void quokka_prefetch_write(const void *address) noexcept nogil


cdef void add_rows(
    double *target,
    const int64_t *nodes,
    const int64_t *profile_ids,
    const double *weights,
    Py_ssize_t first,
    Py_ssize_t stop,
    Py_ssize_t profile_count,
    Py_ssize_t prefetch_distance,
    const Py_ssize_t *prefetch_offsets,
    Py_ssize_t prefetch_count,
) noexcept nogil:
    """Visit one contiguous row interval; each output location gets one add."""
    cdef Py_ssize_t row
    cdef Py_ssize_t column
    cdef Py_ssize_t future
    cdef Py_ssize_t offset
    cdef double *target_row
    cdef double *future_row
    cdef const double *weight_row
    for row in range(first, stop):
        future = row + prefetch_distance
        if prefetch_distance > 0 and future < stop:
            future_row = target + nodes[future] * 40
            for offset in range(prefetch_count):
                quokka_prefetch_write(future_row + prefetch_offsets[offset])
        target_row = target + nodes[row] * 40
        weight_row = weights + row * profile_count
        for column in range(profile_count):
            target_row[profile_ids[column]] = target_row[profile_ids[column]] + weight_row[column]


def merge_node_major(
    double[:, ::1] target,
    const int64_t[::1] nodes,
    const int64_t[::1] profile_ids,
    const double[:, ::1] weights,
    *,
    int threads=1,
    Py_ssize_t prefetch_distance=32,
    Py_ssize_t parallel_threshold_rows=16384,
):
    """Apply one owned sparse increment to the original C target[Nnode,40].

    Nodes must be strictly increasing, as produced by the current compact
    scatter. Complete validation precedes mutation. Different nodes are
    independent, so parallel rows keep every location's original addition and
    ordered calls preserve original batch/group reduction order. The OpenMP
    region completes before this function returns; no persistent H copy or
    worker-owned target is introduced. Prefetch changes only cache residency.
    """
    cdef Py_ssize_t row
    cdef Py_ssize_t column
    cdef Py_ssize_t block
    cdef Py_ssize_t stop
    cdef Py_ssize_t block_rows = 4096
    cdef Py_ssize_t block_count
    cdef Py_ssize_t bad_position = -1
    cdef Py_ssize_t node_count = target.shape[0]
    cdef Py_ssize_t row_count = nodes.shape[0]
    cdef Py_ssize_t profile_count = profile_ids.shape[0]
    cdef Py_ssize_t prefetch_offsets[40]
    cdef Py_ssize_t prefetch_count = 0
    cdef Py_ssize_t previous_cacheline = -1
    cdef Py_ssize_t cacheline
    cdef Py_ssize_t address_offset
    cdef int error_kind = 0
    cdef int64_t previous_node = -1
    cdef double *target_pointer
    cdef const int64_t *nodes_pointer
    cdef const int64_t *profiles_pointer
    cdef const double *weights_pointer
    if target.shape[1] != 40 or profile_count > 40:
        raise ValueError('The accepted spectral accumulator requires 40 luminosity accounts')
    if weights.shape[0] != row_count or weights.shape[1] != profile_count:
        raise ValueError('Sparse weights must match node/profile dimensions')
    if threads < 1 or prefetch_distance < 0 or parallel_threshold_rows < 1:
        raise ValueError('Require positive thread/threshold settings and nonnegative prefetch distance')
    if (
        np.shares_memory(np.asarray(target), np.asarray(nodes))
        or np.shares_memory(np.asarray(target), np.asarray(profile_ids))
        or np.shares_memory(np.asarray(target), np.asarray(weights))
    ):
        raise ValueError('Sparse merge inputs must not alias target')
    with nogil:
        for row in range(row_count):
            if nodes[row] < 0 or nodes[row] >= node_count:
                error_kind = 1
                bad_position = row
                break
            if nodes[row] <= previous_node:
                error_kind = 2
                bad_position = row
                break
            previous_node = nodes[row]
        if error_kind == 0:
            for column in range(profile_count):
                if profile_ids[column] < 0 or profile_ids[column] >= 40:
                    error_kind = 3
                    bad_position = column
                    break
    if error_kind:
        if error_kind == 2:
            raise ValueError(f'Current sparse nodes must be strictly increasing at row {bad_position}')
        kind = 'node' if error_kind == 1 else 'profile'
        raise IndexError(f'Invalid {kind} at sparse position {bad_position}')
    if row_count == 0 or profile_count == 0:
        return
    target_pointer = &target[0, 0]
    nodes_pointer = &nodes[0]
    profiles_pointer = &profile_ids[0]
    weights_pointer = &weights[0, 0]
    # Forty float64 values occupy exactly five 64-byte cache lines. Every row
    # has the same alignment; avoid redundant hints for adjacent profile IDs.
    address_offset = <uintptr_t>target_pointer % 64
    for column in range(profile_count):
        cacheline = (address_offset + profile_ids[column] * 8) // 64
        if cacheline != previous_cacheline:
            prefetch_offsets[prefetch_count] = profile_ids[column]
            prefetch_count += 1
            previous_cacheline = cacheline
    block_count = (row_count + block_rows - 1) // block_rows
    if threads == 1 or row_count < parallel_threshold_rows:
        with nogil:
            add_rows(target_pointer, nodes_pointer, profiles_pointer, weights_pointer,
                     0, row_count, profile_count, prefetch_distance,
                     prefetch_offsets, prefetch_count)
    else:
        for block in prange(block_count, nogil=True, schedule='static', num_threads=threads):
            stop = min((block + 1) * block_rows, row_count)
            add_rows(target_pointer, nodes_pointer, profiles_pointer, weights_pointer,
                     block * block_rows, stop, profile_count, prefetch_distance,
                     prefetch_offsets, prefetch_count)


def kernel_info(int requested_threads=4):
    """Report compiled OpenMP support and observed size of a short native team."""
    cdef Py_ssize_t row
    cdef int[::1] sizes
    if requested_threads < 1:
        raise ValueError('Require positive threads')
    owner = np.empty(requested_threads, dtype=np.intc)
    sizes = owner
    for row in prange(requested_threads, nogil=True, num_threads=requested_threads):
        sizes[row] = quokka_team_size()
    return {
        'openmp_version': quokka_openmp_version(),
        'requested_threads': requested_threads,
        'observed_team_size': int(owner.max()),
        'target_layout': 'Original dense C (Nnode,40)',
        'floating_point': 'ordinary float64 addition; no reduction, FMA or fast-math',
    }
