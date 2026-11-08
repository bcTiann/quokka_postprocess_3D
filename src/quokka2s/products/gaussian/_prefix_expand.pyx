# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Expand standard-normal prefix probabilities into forty luminosity accounts.

Adjacent velocity channels share one cumulative boundary evaluation. Each
channel keeps the original ascending-node/profile accumulation order. No
node-by-channel matrix or online analytic CDF evaluation is used.
"""
from cython.parallel cimport prange
from libc.math cimport floor, ceil, fma, isfinite, INFINITY
from libc.stdint cimport int64_t, uint64_t, uint8_t, uintptr_t

import numpy as np


cdef extern from *:
    """
    #ifdef _OPENMP
    #include <omp.h>
    static int quokka_prefix_openmp(void) { return _OPENMP; }
    static int quokka_prefix_team(void) { return omp_get_num_threads(); }
    #else
    static int quokka_prefix_openmp(void) { return 0; }
    static int quokka_prefix_team(void) { return 1; }
    #endif
    """
    int quokka_prefix_openmp() noexcept nogil
    int quokka_prefix_team() noexcept nogil


cdef inline double phi_lookup(
    double z, double Z, double inverse_h, const double *values, Py_ssize_t M,
) noexcept nogil:
    """Interpolate cumulative fine-cell masses and clip at prepared endpoints."""
    cdef double position
    cdef double fraction
    cdef Py_ssize_t index
    if z <= -Z:
        return values[0]
    if z >= Z:
        return values[M]
    position = (z + Z) * inverse_h
    index = <Py_ssize_t>position
    if index >= M:
        index = M - 1
    fraction = position - index
    return values[index] + fraction * (values[index + 1] - values[index])


cdef inline double boundary_argument(
    Py_ssize_t channel, double a0, double dz, double mu, double sigma,
    double v_min, double delta_v, bint affine_finite,
) noexcept nogil:
    """Retain old affine-boundary order; direct division handles overflow cases."""
    if affine_finite:
        if channel == 0:
            return a0
        return a0 + channel * dz
    return (v_min + channel * delta_v - mu) / sigma


cdef void expand_channels(
    const uint64_t *metadata, const uint8_t *active_profiles,
    const double *coordinates, const double *boundary_starts,
    const double *centers, const double *sigmas, const double *weights,
    const double *values,
    double *output, Py_ssize_t count, Py_ssize_t chunk,
    Py_ssize_t chunk_count, Py_ssize_t chunk_channels,
    Py_ssize_t first_channel, Py_ssize_t stop_channel,
    double v_min, double delta_v, double Z, double inverse_h,
    Py_ssize_t M, bint use_fma, int64_t *counts,
) noexcept nogil:
    """Own complete channel rows; keep node/profile additions in original order."""
    cdef Py_ssize_t node
    cdef Py_ssize_t first_bin
    cdef Py_ssize_t stop_bin
    cdef Py_ssize_t node_stop
    cdef Py_ssize_t channel
    cdef Py_ssize_t profile
    cdef Py_ssize_t active_count
    cdef Py_ssize_t active_index
    cdef int64_t nonzero_coefficients = 0
    cdef int64_t product_updates = 0
    cdef int64_t phi_evaluations = 0
    cdef double a0
    cdef double dz
    cdef double b
    cdef double previous
    cdef double current
    cdef double coefficient
    cdef double product
    cdef bint affine_finite
    cdef const double *node_weights
    cdef const uint8_t *node_profiles
    cdef double *channel_output
    for node in range(count):
        active_count = <Py_ssize_t>(metadata[node * 3] >> 40)
        if active_count == 0:
            continue
        first_bin = <Py_ssize_t>metadata[node * 3 + 1]
        node_stop = <Py_ssize_t>metadata[node * 3 + 2]
        stop_bin = node_stop
        if first_bin < first_channel:
            first_bin = first_channel
        if stop_bin > stop_channel:
            stop_bin = stop_channel
        if first_bin >= stop_bin:
            continue
        a0 = coordinates[node * 2]
        dz = coordinates[node * 2 + 1]
        affine_finite = isfinite(a0) and isfinite(dz)
        node_weights = weights + node * 40
        node_profiles = active_profiles + node * 40
        previous = boundary_starts[node * chunk_count + chunk]
        for channel in range(first_bin, stop_bin):
            b = boundary_argument(channel + 1, a0, dz, centers[node], sigmas[node],
                                  v_min, delta_v, affine_finite)
            # Reuse the following chunk's boundary rather than query it twice.
            if channel + 1 == stop_channel and chunk + 1 < chunk_count and stop_channel < node_stop:
                current = boundary_starts[node * chunk_count + chunk + 1]
            else:
                current = phi_lookup(b, Z, inverse_h, values, M)
                phi_evaluations += 1
            coefficient = current - previous
            previous = current
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
    counts[2] = phi_evaluations


def expand_block(
    const double[::1] centers,
    const double[::1] sigmas,
    const double[:, ::1] weights,
    const double[::1] values,
    double[:, ::1] contribution,
    double v_min,
    double delta_v,
    double Z,
    double h,
    bint match_numpy_fma=False,
    int threads=0,
):
    """Add one ordered block to contribution[K,40] in place.

    centers/sigmas are occupied histogram-node values in km/s; weights[N,40]
    contain luminosities. values[M+1] are the immutable cumulative fine-cell
    masses on [-Z,Z]. Only O(N*threads) boundary metadata is retained; no N-by-K
    probabilities are allocated. contribution[K,40] is updated in place.
    """
    cdef Py_ssize_t count = centers.shape[0]
    cdef Py_ssize_t channels = contribution.shape[0]
    cdef Py_ssize_t node
    cdef Py_ssize_t profile
    cdef Py_ssize_t active_count
    cdef Py_ssize_t first_bin
    cdef Py_ssize_t stop_bin
    cdef Py_ssize_t chunk
    cdef Py_ssize_t chunk_count
    cdef Py_ssize_t chunk_channels
    cdef Py_ssize_t chunk_stop = 0
    cdef Py_ssize_t chunk_first
    cdef Py_ssize_t chunk_last
    cdef Py_ssize_t aligned_offset
    cdef Py_ssize_t M
    cdef Py_ssize_t bin
    cdef int64_t bins_visited = 0
    cdef int64_t nonzero_coefficients = 0
    cdef int64_t product_updates = 0
    cdef int64_t active_profile_count_sum = 0
    cdef int64_t prepared_phi_evaluations = 0
    cdef int64_t phi_evaluations = 0
    cdef int invalid = 0
    cdef uint64_t active_mask
    cdef double inverse_h
    cdef double inverse_sigma
    cdef double a0
    cdef double dz
    cdef double position
    cdef double argument
    cdef bint affine_finite
    cdef bint copied_output = False
    cdef uint64_t[:, ::1] metadata
    cdef uint8_t[:, ::1] active_profiles
    cdef double[:, ::1] coordinates
    cdef double[:, ::1] boundary_starts
    cdef int64_t[:, ::1] chunk_counts
    cdef double[:, ::1] aligned_output
    cdef double *output_pointer
    if weights.shape[0] != count or sigmas.shape[0] != count:
        raise ValueError('Node arrays must have matching lengths')
    if weights.shape[1] != 40 or contribution.shape[1] != 40:
        raise ValueError('The production contraction requires exactly 40 profiles')
    if not isfinite(v_min) or not isfinite(delta_v) or delta_v <= 0.0:
        raise ValueError('Velocity geometry must be finite with positive spacing')
    if not isfinite(v_min + channels * delta_v):
        raise ValueError('The final velocity edge must be finite')
    if not isfinite(Z) or not isfinite(h) or Z <= 0.0 or h <= 0.0:
        raise ValueError('Standardized geometry must be finite and positive')
    inverse_h = 1.0 / h
    if not isfinite(inverse_h):
        raise ValueError('The standardized spacing must be indexable')
    M = values.shape[0] - 1
    if M < 1 or abs(M * h - 2.0 * Z) > 1e-12 * Z:
        raise ValueError('The standardized table must exactly span [-Z,Z]')
    if not np.isfinite(np.asarray(values)).all():
        raise ValueError('The prefix table must be finite')
    if values[0] != 0.0 or values[M] > 1.0 or np.any(np.diff(np.asarray(values)) < 0.0):
        raise ValueError('Require cumulative masses starting at zero, nondecreasing and at most one')
    for array in (np.asarray(centers), np.asarray(sigmas), np.asarray(weights),
                  np.asarray(values)):
        if np.shares_memory(array, np.asarray(contribution)):
            raise ValueError('Output must not alias any input array')
    threads = select_threads(threads)
    if count == 0 or channels == 0:
        return {'support_bins_visited': 0, 'nonzero_template_coefficients': 0,
                'nonzero_product_updates': 0, 'active_profile_count_sum': 0,
                'phi_boundary_evaluations': 0}
    chunk_count = min(threads, channels)
    chunk_channels = (channels + chunk_count - 1) // chunk_count
    metadata_owner = np.zeros((count, 3), dtype=np.uint64)
    metadata = metadata_owner
    active_profiles_owner = np.empty((count, 40), dtype=np.uint8)
    active_profiles = active_profiles_owner
    coordinates_owner = np.empty((count, 2), dtype=np.float64)
    coordinates = coordinates_owner
    boundary_starts_owner = np.empty((count, chunk_count), dtype=np.float64)
    boundary_starts = boundary_starts_owner
    with nogil:
        for node in range(count):
            if not isfinite(centers[node]) or not isfinite(sigmas[node]) or sigmas[node] <= 0.0:
                invalid += 1
                continue
            active_mask = 0
            active_count = 0
            for profile in range(40):
                if not isfinite(weights[node, profile]) or weights[node, profile] < 0.0:
                    invalid += 1
                if weights[node, profile] != 0.0:
                    active_mask |= (<uint64_t>1) << profile
                    active_profiles[node, active_count] = <uint8_t>profile
                    active_count += 1
            active_profile_count_sum += active_count
            if active_count == 0:
                continue
            position = (centers[node] - Z * sigmas[node] - v_min) / delta_v
            if position <= 0.0:
                first_bin = 0
            elif position >= channels + 1:
                first_bin = channels
            else:
                first_bin = <Py_ssize_t>floor(position) - 1
                if first_bin < 0:
                    first_bin = 0
            position = (centers[node] + Z * sigmas[node] - v_min) / delta_v
            if position >= channels:
                stop_bin = channels
            elif position <= -1.0:
                stop_bin = 0
            else:
                stop_bin = <Py_ssize_t>ceil(position) + 1
                if stop_bin > channels:
                    stop_bin = channels
            if stop_bin < first_bin:
                stop_bin = first_bin
            metadata[node, 0] = active_mask | ((<uint64_t>active_count) << 40)
            metadata[node, 1] = first_bin
            metadata[node, 2] = stop_bin
            bins_visited += stop_bin - first_bin
            inverse_sigma = 1.0 / sigmas[node]
            a0 = (v_min - centers[node]) * inverse_sigma
            dz = delta_v * inverse_sigma
            affine_finite = isfinite(inverse_sigma) and isfinite(a0) and isfinite(dz)
            if not affine_finite:
                a0 = INFINITY
                dz = INFINITY
            coordinates[node, 0] = a0
            coordinates[node, 1] = dz
            for chunk in range(chunk_count):
                chunk_first = max(first_bin, chunk * chunk_channels)
                chunk_last = min(stop_bin, min((chunk + 1) * chunk_channels, channels))
                if chunk_first >= chunk_last:
                    continue
                argument = boundary_argument(chunk_first, a0, dz, centers[node], sigmas[node],
                                             v_min, delta_v, affine_finite)
                boundary_starts[node, chunk] = phi_lookup(argument, Z, inverse_h, &values[0], M)
                prepared_phi_evaluations += 1
    if invalid:
        raise ValueError('Require finite Gaussian parameters, positive sigma and finite nonnegative weights')
    aligned_output = contribution
    output_pointer = &contribution[0, 0]
    if threads > 1 and (<uintptr_t>output_pointer % 64) != 0:
        aligned_owner = np.empty(channels * 40 + 8, dtype=np.float64)
        aligned_offset = (-(<uintptr_t>aligned_owner.ctypes.data) & 63) // 8
        aligned_values = aligned_owner[aligned_offset:aligned_offset + channels * 40].reshape(channels, 40)
        np.copyto(aligned_values, np.asarray(contribution))
        aligned_output = aligned_values
        output_pointer = &aligned_output[0, 0]
        copied_output = True
    chunk_counts_owner = np.empty((chunk_count, 8), dtype=np.int64)
    chunk_counts = chunk_counts_owner
    if threads == 1:
        with nogil:
            expand_channels(&metadata[0, 0], &active_profiles[0, 0], &coordinates[0, 0],
                            &boundary_starts[0, 0], &centers[0], &sigmas[0], &weights[0, 0],
                            &values[0], output_pointer, count, 0,
                            chunk_count, chunk_channels, 0, channels, v_min, delta_v, Z,
                            inverse_h, M, match_numpy_fma, &chunk_counts[0, 0])
    else:
        for chunk in prange(chunk_count, nogil=True, schedule='static', num_threads=threads):
            chunk_stop = min((chunk + 1) * chunk_channels, channels)
            expand_channels(&metadata[0, 0], &active_profiles[0, 0], &coordinates[0, 0],
                            &boundary_starts[0, 0], &centers[0], &sigmas[0], &weights[0, 0],
                            &values[0], output_pointer, count, chunk,
                            chunk_count, chunk_channels, chunk * chunk_channels, chunk_stop,
                            v_min, delta_v, Z, inverse_h, M, match_numpy_fma,
                            &chunk_counts[chunk, 0])
    for chunk in range(chunk_count):
        nonzero_coefficients += chunk_counts[chunk, 0]
        product_updates += chunk_counts[chunk, 1]
        phi_evaluations += chunk_counts[chunk, 2]
    if copied_output:
        np.copyto(np.asarray(contribution), np.asarray(aligned_output))
    return {'support_bins_visited': bins_visited,
            'nonzero_template_coefficients': nonzero_coefficients,
            'nonzero_product_updates': product_updates,
            'active_profile_count_sum': active_profile_count_sum,
            'phi_boundary_evaluations': prepared_phi_evaluations + phi_evaluations}


def kernel_info(int requested_threads=8):
    """Expose actual OpenMP support; never label a serial build as eight threads."""
    cdef Py_ssize_t row
    cdef int[::1] sizes
    if requested_threads < 1:
        raise ValueError('Require positive threads')
    owner = np.empty(requested_threads, dtype=np.intc)
    sizes = owner
    for row in prange(requested_threads, nogil=True, num_threads=requested_threads):
        sizes[row] = quokka_prefix_team()
    return {'openmp_version': quokka_prefix_openmp(),
            'requested_threads': requested_threads,
            'observed_team_size': int(owner.max()),
            'partition': 'Independent channel rows; ascending nodes and active profiles',
            'coefficient_algorithms': 'Standard-normal fine-cell-prefix interpolation; shared channel boundaries',
            'probability_matrix_allocated': False,
            'preferred_expand_threads': 8}


_automatic_thread_count = None
_verified_thread_counts = {1}


def select_threads(int requested_threads=0):
    """Select actual eight-thread support or one; explicit requests are strict."""
    global _automatic_thread_count
    if requested_threads < 0:
        raise ValueError('Thread count must be zero for automatic or positive')
    if requested_threads == 0:
        if _automatic_thread_count is None:
            info = kernel_info(8)
            _automatic_thread_count = 8 if info['observed_team_size'] == 8 else 1
            _verified_thread_counts.add(_automatic_thread_count)
        return _automatic_thread_count
    if requested_threads not in _verified_thread_counts:
        info = kernel_info(requested_threads)
        if info['observed_team_size'] != requested_threads:
            raise RuntimeError('The installed prefix binary cannot create the explicitly requested OpenMP team')
        _verified_thread_counts.add(requested_threads)
    return requested_threads
