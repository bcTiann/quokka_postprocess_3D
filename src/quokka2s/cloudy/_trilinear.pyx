# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Prepared-bracket Cloudy support and emissivity sums in float64.

Table axes are NH,nH,T,line, with contiguous line values. Query brackets keep
the production NumPy geometry. Each query retains the eight-corner order
000,001,...111 and separate weight multiplies, value multiplies and additions.
Only support classification precedes interpolation; unavailable queries never
evaluate either emissivity branch. NumPy performs the later base10 power.
"""
from libc.stdint cimport uint8_t

import numpy as np


def fused_interpolation_sums(
    const Py_ssize_t[:] column_lower,
    const Py_ssize_t[:] density_lower,
    const Py_ssize_t[:] temperature_lower,
    const double[:] column_fraction,
    const double[:] density_fraction,
    const double[:] temperature_fraction,
    const double[:, :, :, ::1] linear_table,
    const double[:, :, :, ::1] logarithmic_table,
    const uint8_t[:, :, :, ::1] zero_table,
    const uint8_t[:, :, :, ::1] failure_table,
    const uint8_t[:] lines_with_zero,
    bint has_failures,
    double touch_epsilon,
):
    """Return (linear sums, log sums, zero support, failed query), all owned.

    The first three returned arrays have shape(Q,L), with contiguous line
    outputs; zero support and failed queries use uint8 flags. failure_table's
    last dimension is1 only when every line's mask was verified identical at
    load, otherwise it isL. Globally zero-free lines skip zero-mask reads.
    """
    cdef Py_ssize_t query_count = column_lower.shape[0]
    cdef Py_ssize_t line_count = linear_table.shape[3]
    cdef Py_ssize_t failure_count = failure_table.shape[3]
    cdef Py_ssize_t query
    cdef Py_ssize_t line
    cdef Py_ssize_t corner
    cdef Py_ssize_t axis
    cdef Py_ssize_t x
    cdef Py_ssize_t y
    cdef Py_ssize_t z
    cdef Py_ssize_t xs[8]
    cdef Py_ssize_t ys[8]
    cdef Py_ssize_t zs[8]
    cdef double weights[8]
    cdef double weight
    cdef double term
    cdef double failure_weight
    cdef double f_column
    cdef double f_density
    cdef double f_temperature
    cdef bint need_linear
    cdef bint need_logarithmic
    cdef bint failed
    cdef bint shared_failure = failure_count == 1

    if density_lower.shape[0] != query_count or temperature_lower.shape[0] != query_count:
        raise ValueError('All prepared index arrays must have the same query count')
    if column_fraction.shape[0] != query_count or density_fraction.shape[0] != query_count or temperature_fraction.shape[0] != query_count:
        raise ValueError('All prepared fraction arrays must match the query count')
    if lines_with_zero.shape[0] != line_count:
        raise ValueError('Zero-line flags must match the table line count')
    if failure_count != 1 and failure_count != line_count:
        raise ValueError('Failure masks must contain one verified shared plane or every line')
    for axis in range(3):
        if linear_table.shape[axis] < 2:
            raise ValueError('Every interpolation axis needs at least two nodes')
        if logarithmic_table.shape[axis] != linear_table.shape[axis] or zero_table.shape[axis] != linear_table.shape[axis] or failure_table.shape[axis] != linear_table.shape[axis]:
            raise ValueError('Packed table axes differ')
    if logarithmic_table.shape[3] != line_count or zero_table.shape[3] != line_count:
        raise ValueError('Packed emissivity and zero-mask line counts differ')
    # Bounds are checked before nogil indexing. Coordinate coverage and the
    # exact endpoint policy remain with prepare_interpolation_coordinates().
    for query in range(query_count):
        if column_lower[query] < 0 or column_lower[query] >= linear_table.shape[0] - 1:
            raise IndexError('Prepared column bracket is outside the table')
        if density_lower[query] < 0 or density_lower[query] >= linear_table.shape[1] - 1:
            raise IndexError('Prepared density bracket is outside the table')
        if temperature_lower[query] < 0 or temperature_lower[query] >= linear_table.shape[2] - 1:
            raise IndexError('Prepared temperature bracket is outside the table')

    linear_owner = np.empty((query_count, line_count), dtype=np.float64)
    logarithmic_owner = np.empty((query_count, line_count), dtype=np.float64)
    zero_owner = np.zeros((query_count, line_count), dtype=np.uint8)
    failed_owner = np.zeros(query_count, dtype=np.uint8)
    failure_owner = np.zeros((query_count, failure_count), dtype=np.float64)
    cdef double[:, ::1] linear_sum = linear_owner
    cdef double[:, ::1] logarithmic_sum = logarithmic_owner
    cdef uint8_t[:, ::1] zero_support = zero_owner
    cdef uint8_t[::1] failed_query = failed_owner
    cdef double[:, ::1] failure_sum = failure_owner

    with nogil:
        for query in range(query_count):
            f_column = column_fraction[query]
            f_density = density_fraction[query]
            f_temperature = temperature_fraction[query]
            for line in range(line_count):
                linear_sum[query, line] = 0.0
                logarithmic_sum[query, line] = 0.0
            for corner in range(8):
                x = column_lower[query] + ((corner >> 2) & 1)
                y = density_lower[query] + ((corner >> 1) & 1)
                z = temperature_lower[query] + (corner & 1)
                xs[corner] = x
                ys[corner] = y
                zs[corner] = z
                weight = 1.0
                if (corner & 4) == 0:
                    weight = weight * (1.0 - f_column)
                else:
                    weight = weight * f_column
                if (corner & 2) == 0:
                    weight = weight * (1.0 - f_density)
                else:
                    weight = weight * f_density
                if (corner & 1) == 0:
                    weight = weight * (1.0 - f_temperature)
                else:
                    weight = weight * f_temperature
                weights[corner] = weight
                if has_failures:
                    if shared_failure:
                        term = failure_table[x, y, z, 0] * weight
                        failure_sum[query, 0] = failure_sum[query, 0] + term
                    else:
                        for line in range(line_count):
                            term = failure_table[x, y, z, line] * weight
                            failure_sum[query, line] = failure_sum[query, line] + term
                if weight > touch_epsilon:
                    for line in range(line_count):
                        if lines_with_zero[line] and zero_table[x, y, z, line]:
                            zero_support[query, line] = 1
            failed = False
            if has_failures:
                for line in range(failure_count):
                    if failure_sum[query, line] > touch_epsilon:
                        failed = True
            if failed:
                failed_query[query] = 1
                continue

            # Match the existing point-wise branch selection: mixed-line
            # points evaluate both sums for every line, then select per line.
            need_linear = False
            need_logarithmic = False
            for line in range(line_count):
                if zero_support[query, line]:
                    need_linear = True
                else:
                    need_logarithmic = True
            for corner in range(8):
                x = xs[corner]
                y = ys[corner]
                z = zs[corner]
                weight = weights[corner]
                for line in range(line_count):
                    if need_linear:
                        term = linear_table[x, y, z, line] * weight
                        linear_sum[query, line] = linear_sum[query, line] + term
                    if need_logarithmic:
                        term = logarithmic_table[x, y, z, line] * weight
                        logarithmic_sum[query, line] = logarithmic_sum[query, line] + term
    return linear_owner, logarithmic_owner, zero_owner, failed_owner
