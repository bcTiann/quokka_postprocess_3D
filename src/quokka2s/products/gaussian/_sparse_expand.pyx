# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Exact A contraction over disjoint output-channel ranges, never node sums."""
from cython.parallel cimport prange
from libc.math cimport floor, ceil, fma
from libc.stdint cimport int64_t, uint64_t, uint8_t, uintptr_t

import numpy as np


cdef extern from *:
    """
    #ifdef _OPENMP
    #include <omp.h>
    static int quokka_expansion_openmp(void) { return _OPENMP; }
    static int quokka_expansion_team(void) { return omp_get_num_threads(); }
    #else
    static int quokka_expansion_openmp(void) { return 0; }
    static int quokka_expansion_team(void) { return 1; }
    #endif
    """
    int quokka_expansion_openmp() noexcept nogil
    int quokka_expansion_team() noexcept nogil


cdef void expand_channels(
    const uint64_t *metadata,
    const uint8_t *active_profiles,
    const double *weights,
    double *output,
    Py_ssize_t count,
    Py_ssize_t first_channel,
    Py_ssize_t stop_channel,
    bint use_fma,
    int64_t *counts,
) noexcept nogil:
    """Keep ascending nodes/profiles for every owned (channel, profile)."""
    cdef Py_ssize_t node
    cdef Py_ssize_t first_bin
    cdef Py_ssize_t stop_bin
    cdef Py_ssize_t channel
    cdef Py_ssize_t profile
    cdef Py_ssize_t active_count
    cdef Py_ssize_t active_index
    cdef int64_t nonzero_coefficients = 0
    cdef int64_t product_updates = 0
    cdef uint64_t active_mask
    cdef uint64_t profile_bits = ((<uint64_t>1) << 40) - 1
    cdef const double *coefficients
    cdef const double *node_weights
    cdef const uint8_t *node_profiles
    cdef double *channel_output
    cdef double coefficient
    cdef double product
    for node in range(count):
        active_mask = metadata[node * 4 + 1] & profile_bits
        if active_mask == 0:
            continue
        first_bin = <Py_ssize_t>metadata[node * 4 + 2]
        stop_bin = <Py_ssize_t>metadata[node * 4 + 3]
        if first_bin < first_channel:
            first_bin = first_channel
        if stop_bin > stop_channel:
            stop_bin = stop_channel
        if first_bin >= stop_bin:
            continue
        active_count = <Py_ssize_t>(metadata[node * 4 + 1] >> 40)
        coefficients = <const double *><uintptr_t>metadata[node * 4]
        node_weights = weights + node * 40
        node_profiles = active_profiles + node * 40
        for channel in range(first_bin, stop_bin):
            coefficient = coefficients[channel]
            if coefficient == 0.0:
                continue
            nonzero_coefficients += 1
            product_updates += active_count
            channel_output = output + channel * 40
            if use_fma:
                for active_index in range(active_count):
                    profile = node_profiles[active_index]
                    channel_output[profile] = fma(coefficient, node_weights[profile], channel_output[profile])
            else:
                for active_index in range(active_count):
                    profile = node_profiles[active_index]
                    product = coefficient * node_weights[profile]
                    channel_output[profile] = channel_output[profile] + product
    counts[0] = nonzero_coefficients
    counts[1] = product_updates


def expand_block(
    const double[::1] centers,
    const double[::1] sigmas,
    const int64_t[::1] sigma_ids,
    const double[:, ::1] weights,
    const double[:, :, ::1] table,
    double[:, ::1] contribution,
    double v_min,
    double delta_v,
    Py_ssize_t n_max,
    Py_ssize_t fractional_subdivisions,
    double cutoff_sigma,
    bint match_numpy_fma=False,
    int threads=0,
):
    """Match the original block's values and four integer statistics exactly.

    Serial metadata preparation repeats the original alignment/support rules.
    Channel workers receive disjoint 64-byte-aligned rows. They iterate nodes
    in the input order and active profiles in ascending order. No float64 node
    reduction, block/band regrouping or post-hoc initial-value addition occurs.
    If necessary, contribution is copied to aligned scratch before any updates
    and copied back afterward, preserving even nonzero initial values.
    """
    cdef Py_ssize_t count = centers.shape[0]
    cdef Py_ssize_t channels = contribution.shape[0]
    cdef Py_ssize_t profiles = contribution.shape[1]
    cdef Py_ssize_t node
    cdef Py_ssize_t profile
    cdef Py_ssize_t active_count
    cdef Py_ssize_t integer_mu
    cdef Py_ssize_t fraction_id
    cdef Py_ssize_t upper_fraction
    cdef Py_ssize_t offset
    cdef Py_ssize_t sigma_id
    cdef Py_ssize_t first_bin
    cdef Py_ssize_t last_bin
    cdef Py_ssize_t chunk
    cdef Py_ssize_t chunk_count
    cdef Py_ssize_t chunk_channels
    cdef Py_ssize_t chunk_stop = 0
    cdef Py_ssize_t aligned_offset
    cdef Py_ssize_t bins_visited = 0
    cdef Py_ssize_t nonzero_coefficients = 0
    cdef Py_ssize_t product_updates = 0
    cdef Py_ssize_t active_profile_count_sum = 0
    cdef Py_ssize_t invalid_nodes = 0
    cdef uint64_t active_mask
    cdef double coordinate
    cdef double fractional_coordinate
    cdef uint64_t[:, ::1] metadata
    cdef uint8_t[:, ::1] active_profiles
    cdef int64_t[:, ::1] chunk_counts
    cdef double[:, ::1] aligned_output
    cdef double *output_pointer
    cdef const double *weight_pointer
    cdef bint copied_output = False
    if profiles != 40 or weights.shape[1] != profiles:
        raise ValueError('The matched production contraction requires exactly 40 profiles')
    if sigmas.shape[0] != count or sigma_ids.shape[0] != count or weights.shape[0] != count:
        raise ValueError('Node arrays must have matching lengths')
    if delta_v <= 0.0 or fractional_subdivisions < 1 or cutoff_sigma <= 0.0:
        raise ValueError('Invalid template geometry')
    threads = select_threads(threads)
    if count == 0:
        return {'support_bins_visited': 0, 'nonzero_template_coefficients': 0,
                'nonzero_product_updates': 0, 'active_profile_count_sum': 0}
    metadata_owner = np.empty((count, 4), dtype=np.uint64)
    metadata = metadata_owner
    active_profiles_owner = np.empty((count, 40), dtype=np.uint8)
    active_profiles = active_profiles_owner
    with nogil:
        for node in range(count):
            metadata[node, 0] = 0
            metadata[node, 1] = 0
            metadata[node, 2] = 0
            metadata[node, 3] = 0
            # Preserve the source expressions and their exact operation order.
            coordinate = centers[node] / delta_v
            integer_mu = <Py_ssize_t>floor(coordinate)
            fractional_coordinate = (coordinate - integer_mu) * fractional_subdivisions
            fraction_id = <Py_ssize_t>floor(fractional_coordinate)
            if fraction_id >= fractional_subdivisions:
                fraction_id = fractional_subdivisions
            elif coordinate != integer_mu + (<double>fraction_id / fractional_subdivisions):
                upper_fraction = fraction_id + 1
                if coordinate == integer_mu + (<double>upper_fraction / fractional_subdivisions):
                    fraction_id = upper_fraction
                else:
                    invalid_nodes += 1
                    continue
            sigma_id = sigma_ids[node]
            offset = n_max - integer_mu
            if (sigma_id < 0 or sigma_id >= table.shape[0] or fraction_id < 0 or
                fraction_id >= table.shape[1] or offset < 0 or offset + channels > table.shape[2]):
                invalid_nodes += 1
                continue
            active_mask = 0
            active_count = 0
            for profile in range(profiles):
                if weights[node, profile] != 0.0:
                    active_mask |= (<uint64_t>1) << profile
                    active_profiles[node, active_count] = <uint8_t>profile
                    active_count += 1
            active_profile_count_sum += active_count
            if active_count == 0:
                continue
            first_bin = <Py_ssize_t>floor((centers[node] - cutoff_sigma * sigmas[node] - v_min) / delta_v) - 1
            last_bin = <Py_ssize_t>ceil((centers[node] + cutoff_sigma * sigmas[node] - v_min) / delta_v) + 1
            if first_bin < 0:
                first_bin = 0
            if last_bin > channels:
                last_bin = channels
            if last_bin > first_bin:
                bins_visited += last_bin - first_bin
            # Additional clamps encode empty out-of-window intervals safely.
            if first_bin > channels:
                first_bin = channels
            if last_bin < 0:
                last_bin = 0
            metadata[node, 0] = <uint64_t><uintptr_t>&table[sigma_id, fraction_id, offset]
            metadata[node, 1] = active_mask | ((<uint64_t>active_count) << 40)
            metadata[node, 2] = first_bin
            metadata[node, 3] = last_bin
    if channels == 0:
        if invalid_nodes:
            raise ValueError('Sparse A requires exact aligned in-bounds table nodes')
        return {'support_bins_visited': bins_visited, 'nonzero_template_coefficients': 0,
                'nonzero_product_updates': 0, 'active_profile_count_sum': active_profile_count_sum}
    output_pointer = &contribution[0, 0]
    aligned_output = contribution
    if threads > 1 and (<uintptr_t>output_pointer % 64) != 0:
        aligned_owner = np.empty(channels * profiles + 8, dtype=np.float64)
        aligned_offset = (-(<uintptr_t>aligned_owner.ctypes.data) & 63) // 8
        aligned_values = aligned_owner[aligned_offset:aligned_offset + channels * profiles].reshape(channels, profiles)
        np.copyto(aligned_values, np.asarray(contribution))
        aligned_output = aligned_values
        output_pointer = &aligned_output[0, 0]
        copied_output = True
    weight_pointer = &weights[0, 0]
    chunk_count = min(threads, channels)
    chunk_channels = (channels + chunk_count - 1) // chunk_count
    # Each worker writes its pair once; padding keeps integer counters separate.
    chunk_counts_owner = np.empty((chunk_count, 8), dtype=np.int64)
    chunk_counts = chunk_counts_owner
    if threads == 1:
        with nogil:
            expand_channels(&metadata[0, 0], &active_profiles[0, 0], weight_pointer, output_pointer,
                            count, 0, channels, match_numpy_fma, &chunk_counts[0, 0])
    else:
        for chunk in prange(chunk_count, nogil=True, schedule='static', num_threads=threads):
            chunk_stop = min((chunk + 1) * chunk_channels, channels)
            expand_channels(&metadata[0, 0], &active_profiles[0, 0], weight_pointer, output_pointer,
                            count, chunk * chunk_channels, chunk_stop,
                            match_numpy_fma, &chunk_counts[chunk, 0])
    for chunk in range(chunk_count):
        nonzero_coefficients += chunk_counts[chunk, 0]
        product_updates += chunk_counts[chunk, 1]
    if copied_output:
        np.copyto(np.asarray(contribution), np.asarray(aligned_output))
    if invalid_nodes:
        raise ValueError('Sparse A requires exact aligned in-bounds table nodes')
    return {
        'support_bins_visited': bins_visited,
        'nonzero_template_coefficients': nonzero_coefficients,
        'nonzero_product_updates': product_updates,
        'active_profile_count_sum': active_profile_count_sum,
    }


def kernel_info(int requested_threads=8):
    """Expose actual team size so a serial local build cannot claim threading."""
    cdef Py_ssize_t row
    cdef int[::1] sizes
    if requested_threads < 1:
        raise ValueError('Require positive threads')
    owner = np.empty(requested_threads, dtype=np.intc)
    sizes = owner
    for row in prange(requested_threads, nogil=True, num_threads=requested_threads):
        sizes[row] = quokka_expansion_team()
    return {
        'openmp_version': quokka_expansion_openmp(),
        'requested_threads': requested_threads,
        'observed_team_size': int(owner.max()),
        'partition': 'Disjoint complete channel rows; nodes and active profiles retain input order',
        'alignment': '64-byte output rows, copied initial values when needed',
        'preferred_expand_threads': 8,
    }


_automatic_thread_count = None
_verified_thread_counts = {1}


def select_threads(int requested_threads=0):
    """Choose an observed eight-thread team or serial; verify explicit teams.

    Zero requests automatic selection. Positive requests are strict and never
    claim parallel execution when this binary/runtime provides fewer workers.
    Successful checks are cached because every block uses the same native team.
    """
    global _automatic_thread_count
    if requested_threads < 0:
        raise ValueError('Channel expansion threads must be positive, or zero for automatic selection')
    if requested_threads == 0:
        if _automatic_thread_count is None:
            observed = kernel_info(8)
            _automatic_thread_count = 8 if observed['observed_team_size'] == 8 else 1
            _verified_thread_counts.add(_automatic_thread_count)
        return _automatic_thread_count
    if requested_threads not in _verified_thread_counts:
        observed = kernel_info(requested_threads)
        if observed['observed_team_size'] != requested_threads:
            raise RuntimeError(
                'The installed Gaussian expansion binary/runtime cannot create '
                f'the explicitly requested OpenMP team of {requested_threads} threads'
            )
        _verified_thread_counts.add(requested_threads)
    return requested_threads
