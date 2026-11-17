# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Fused Float64 table arithmetic; axes, fields and masks stay runtime inputs.

Python loads and validates physical tables. These loops preserve all eight
corner products, including zero times NaN, and never retain borrowed arrays.
"""

from libc.math cimport NAN, isfinite
from libc.stdint cimport int64_t

import numpy as np
from math import prod


cdef inline Py_ssize_t lower_bracket(
    const double[:] axis,
    double coordinate,
) noexcept nogil:
    """Match searchsorted(side='right') followed by endpoint clipping."""
    cdef Py_ssize_t left = 0
    cdef Py_ssize_t right = axis.shape[0]
    cdef Py_ssize_t middle
    while left < right:
        middle = left + (right - left) // 2
        if coordinate < axis[middle]:
            right = middle
        else:
            left = middle + 1
    if left < 1:
        left = 1
    elif left >= axis.shape[0]:
        left = axis.shape[0] - 1
    return left - 1


def interpolate_regular_fields(
    const double[:] axis0 not None,
    const double[:] axis1 not None,
    const double[:] axis2 not None,
    const double[:, :] points not None,
    const double[:, :] fields not None,
):
    """Return [query,field] from a C-order-flattened three-dimensional grid.

    Inputs can be read-only strided views. Fields has shape [N0*N1*N2,F].
    Nonfinite or uncovered queries return NaN for every field. The three axes
    have been checked as strictly increasing by the owning SciPy interpolator.
    """
    cdef Py_ssize_t query_count = points.shape[0]
    cdef Py_ssize_t field_count = fields.shape[1]
    cdef Py_ssize_t n0 = axis0.shape[0]
    cdef Py_ssize_t n1 = axis1.shape[0]
    cdef Py_ssize_t n2 = axis2.shape[0]
    if points.shape[1] != 3 or min(n0, n1, n2) < 2:
        raise ValueError("Require points[Q,3] and three axes with at least two nodes")
    if fields.shape[0] != prod((n0, n1, n2)):
        raise ValueError("The flattened table does not match its three axes")
    output = np.empty((query_count, field_count), dtype=np.float64)
    cdef double[:, :] result = output
    cdef Py_ssize_t query, field, i0, i1, i2, side0, side1, side2, node
    cdef double x0, x1, x2, f0, f1, f2, w0, w1, w2, first_weight, weight
    with nogil:
        for query in range(query_count):
            x0 = points[query, 0]
            x1 = points[query, 1]
            x2 = points[query, 2]
            for field in range(field_count):
                result[query, field] = NAN
            if (
                not isfinite(x0) or not isfinite(x1) or not isfinite(x2)
                or x0 < axis0[0] or x0 > axis0[n0 - 1]
                or x1 < axis1[0] or x1 > axis1[n1 - 1]
                or x2 < axis2[0] or x2 > axis2[n2 - 1]
            ):
                continue
            i0 = lower_bracket(axis0, x0)
            i1 = lower_bracket(axis1, x1)
            i2 = lower_bracket(axis2, x2)
            f0 = (x0 - axis0[i0]) / (axis0[i0 + 1] - axis0[i0])
            f1 = (x1 - axis1[i1]) / (axis1[i1 + 1] - axis1[i1])
            f2 = (x2 - axis2[i2]) / (axis2[i2 + 1] - axis2[i2])
            for field in range(field_count):
                result[query, field] = 0.0
            for side0 in range(2):
                w0 = f0 if side0 else 1.0 - f0
                for side1 in range(2):
                    w1 = f1 if side1 else 1.0 - f1
                    first_weight = w0 * w1
                    for side2 in range(2):
                        w2 = f2 if side2 else 1.0 - f2
                        weight = first_weight * w2
                        node = ((i0 + side0) * n1 + i1 + side1) * n2 + i2 + side2
                        for field in range(field_count):
                            result[query, field] += weight * fields[node, field]
    return output


def regular_grid_brackets(
    const double[:] axis0 not None,
    const double[:] axis1 not None,
    const double[:] axis2 not None,
    const double[:, :] points not None,
):
    """Return three (lower, upper, fraction) vectors for runtime log points.

    This matches searchsorted(side='right') and clipped endpoint brackets.
    The owning lookup has checked the increasing axes and physical inputs.
    """
    if points.shape[1] != 3 or min(axis0.shape[0], axis1.shape[0], axis2.shape[0]) < 2:
        raise ValueError("Require points[Q,3] and three axes with at least two nodes")
    lower = np.empty((3, points.shape[0]), dtype=np.int64)
    upper = np.empty((3, points.shape[0]), dtype=np.int64)
    fraction = np.empty((3, points.shape[0]), dtype=np.float64)
    cdef int64_t[:, :] lower_values = lower
    cdef int64_t[:, :] upper_values = upper
    cdef double[:, :] fraction_values = fraction
    cdef Py_ssize_t query, i0, i1, i2
    with nogil:
        for query in range(points.shape[0]):
            i0 = lower_bracket(axis0, points[query, 0])
            i1 = lower_bracket(axis1, points[query, 1])
            i2 = lower_bracket(axis2, points[query, 2])
            lower_values[0, query] = i0
            lower_values[1, query] = i1
            lower_values[2, query] = i2
            upper_values[0, query] = i0 + 1
            upper_values[1, query] = i1 + 1
            upper_values[2, query] = i2 + 1
            fraction_values[0, query] = (points[query, 0] - axis0[i0]) / (axis0[i0 + 1] - axis0[i0])
            fraction_values[1, query] = (points[query, 1] - axis1[i1]) / (axis1[i1 + 1] - axis1[i1])
            fraction_values[2, query] = (points[query, 2] - axis2[i2]) / (axis2[i2 + 1] - axis2[i2])
    return tuple((lower[axis], upper[axis], fraction[axis]) for axis in range(3))


def cloudy_corner_fields(
    lower_indices,
    fractions,
    grid_shape,
    const double[:, :] linear_fields not None,
    const double[:, :] logarithmic_fields not None,
    const unsigned char[:, :] zero_nodes not None,
    const unsigned char[:, :] failed_nodes not None,
    double touch_epsilon,
):
    """Return log/linear sums, zero support and failed-query flags.

    Field and mask matrices have shape [line,N0*N1*N2]. Masks are uint8 views
    of the live Boolean tables, not cached copies. NumPy performs the final
    power-of-ten operation so its error policy and vector implementation remain.
    """
    if len(lower_indices) != 3 or len(fractions) != 3 or len(grid_shape) != 3:
        raise ValueError("Require three lower-index arrays, fractions and grid dimensions")
    cdef const int64_t[:] lower0 = lower_indices[0]
    cdef const int64_t[:] lower1 = lower_indices[1]
    cdef const int64_t[:] lower2 = lower_indices[2]
    cdef const double[:] fraction0 = fractions[0]
    cdef const double[:] fraction1 = fractions[1]
    cdef const double[:] fraction2 = fractions[2]
    cdef Py_ssize_t query_count = lower0.shape[0]
    cdef Py_ssize_t field_count = linear_fields.shape[0]
    cdef Py_ssize_t n0 = grid_shape[0]
    cdef Py_ssize_t n1 = grid_shape[1]
    cdef Py_ssize_t n2 = grid_shape[2]
    cdef Py_ssize_t node_count = prod((n0, n1, n2))
    if min(n0, n1, n2) < 2:
        raise ValueError("Require three grid axes with at least two nodes")
    if not isfinite(touch_epsilon) or touch_epsilon < 0.0:
        raise ValueError("Require a finite nonnegative node-touch threshold")
    if (
        lower1.shape[0] != query_count or lower2.shape[0] != query_count
        or fraction0.shape[0] != query_count or fraction1.shape[0] != query_count
        or fraction2.shape[0] != query_count
        or linear_fields.shape[1] != node_count
        or logarithmic_fields.shape[0] != field_count
        or logarithmic_fields.shape[1] != node_count
        or zero_nodes.shape[0] != field_count or zero_nodes.shape[1] != node_count
        or failed_nodes.shape[0] != field_count or failed_nodes.shape[1] != node_count
    ):
        raise ValueError("Cloudy brackets, fields and masks have incompatible shapes")
    # One query writes all lines together; keep those output entries adjacent.
    logarithmic_output = np.empty((field_count, query_count), dtype=np.float64, order="F")
    linear_output = np.empty((field_count, query_count), dtype=np.float64, order="F")
    zero_output = np.zeros((field_count, query_count), dtype=np.uint8)
    failed_output = np.zeros(query_count, dtype=np.uint8)
    zero_representatives = np.arange(field_count, dtype=np.intp)
    failure_representatives = np.arange(field_count, dtype=np.intp)
    cdef double[:, :] logarithmic = logarithmic_output
    cdef double[:, :] linear = linear_output
    cdef unsigned char[:, :] zero_support = zero_output
    cdef unsigned char[:] failed = failed_output
    cdef Py_ssize_t[:] zero_first = zero_representatives
    cdef Py_ssize_t[:] failure_first = failure_representatives
    cdef Py_ssize_t query, field, side0, side1, side2, corner, node
    cdef Py_ssize_t previous, representative
    cdef Py_ssize_t i0, i1, i2
    cdef Py_ssize_t nodes[8]
    cdef double weights[8]
    cdef double f0, f1, f2, w0, w1, w2, first_weight, failure_weight
    cdef double logarithmic_sum, linear_sum
    cdef bint need_linear, need_logarithmic
    cdef bint equal_masks
    with nogil:
        # Inspect the live masks once per call. Identical lines share support
        # checks, while their coefficients and interpolation remain separate.
        for field in range(field_count):
            for previous in range(field):
                if zero_first[previous] == previous:
                    equal_masks = True
                    for node in range(node_count):
                        if zero_nodes[field, node] != zero_nodes[previous, node]:
                            equal_masks = False
                            break
                    if equal_masks:
                        zero_first[field] = previous
                        break
            for previous in range(field):
                if failure_first[previous] == previous:
                    equal_masks = True
                    for node in range(node_count):
                        if failed_nodes[field, node] != failed_nodes[previous, node]:
                            equal_masks = False
                            break
                    if equal_masks:
                        failure_first[field] = previous
                        break
        for query in range(query_count):
            i0 = lower0[query]
            i1 = lower1[query]
            i2 = lower2[query]
            if i0 < 0 or i0 >= n0 - 1 or i1 < 0 or i1 >= n1 - 1 or i2 < 0 or i2 >= n2 - 1:
                with gil:
                    raise IndexError("A Cloudy lower bracket lies outside the table")
            f0 = fraction0[query]
            f1 = fraction1[query]
            f2 = fraction2[query]
            corner = 0
            for side0 in range(2):
                w0 = f0 if side0 else 1.0 - f0
                for side1 in range(2):
                    w1 = f1 if side1 else 1.0 - f1
                    first_weight = w0 * w1
                    for side2 in range(2):
                        w2 = f2 if side2 else 1.0 - f2
                        weights[corner] = first_weight * w2
                        nodes[corner] = ((i0 + side0) * n1 + i1 + side1) * n2 + i2 + side2
                        corner += 1
            need_linear = False
            need_logarithmic = False
            for field in range(field_count):
                representative = zero_first[field]
                if representative != field:
                    zero_support[field, query] = zero_support[representative, query]
                else:
                    for corner in range(8):
                        node = nodes[corner]
                        if zero_nodes[field, node] and weights[corner] > touch_epsilon:
                            zero_support[field, query] = 1
                if failure_first[field] == field:
                    failure_weight = 0.0
                    for corner in range(8):
                        node = nodes[corner]
                        # Boolean masks contribute one copy of each weight.
                        failure_weight += (failed_nodes[field, node] != 0) * weights[corner]
                    if failure_weight > touch_epsilon:
                        failed[query] = 1
                if zero_support[field, query]:
                    need_linear = True
                else:
                    need_logarithmic = True
            if failed[query]:
                for field in range(field_count):
                    logarithmic[field, query] = NAN
                    linear[field, query] = NAN
                continue
            for field in range(field_count):
                logarithmic_sum = 0.0
                linear_sum = 0.0
                for corner in range(8):
                    node = nodes[corner]
                    if need_logarithmic:
                        logarithmic_sum += weights[corner] * logarithmic_fields[field, node]
                    if need_linear:
                        linear_sum += weights[corner] * linear_fields[field, node]
                logarithmic[field, query] = logarithmic_sum if need_logarithmic else NAN
                linear[field, query] = linear_sum if need_linear else NAN
    return logarithmic_output, linear_output, zero_output.view(np.bool_), failed_output.view(np.bool_)


from libc.math cimport exp


def scatter_scaled_nonnegative_fields(
    destinations,
    coefficients,
    const double[:] scale not None,
    const int64_t[:] rows not None,
):
    """Validate all compact products, then scatter them into full field arrays.

    No output is written before every product and index has passed. False asks
    Python to repeat its original arithmetic, preserving its warnings/errors.
    Destination order and repeated-index writes follow the supplied row order.
    """
    destinations = tuple(destinations)
    coefficients = tuple(coefficients)
    cdef Py_ssize_t field_count = len(coefficients)
    cdef Py_ssize_t count = rows.shape[0]
    cdef Py_ssize_t field, cell, destination_count
    cdef const double[:] coefficient
    cdef const double[:] existing
    cdef double[:] destination
    cdef double product
    cdef bint invalid
    if len(destinations) != field_count or scale.shape[0] != count:
        raise ValueError("Scaled field counts and compact rows must match")
    # The NumPy caller computes every compact product before any write. Borrow
    # only disjoint inputs; overlapping user arrays retain that original path.
    source_arrays = (*coefficients, np.asarray(scale), np.asarray(rows))
    for destination_array in destinations:
        for source_array in source_arrays:
            if np.shares_memory(destination_array, source_array):
                return False
    for field in range(field_count):
        if coefficients[field] is None or destinations[field] is None:
            raise ValueError("Scaled fields require array inputs")
        coefficient = coefficients[field]
        existing = destinations[field]
        if coefficient.shape[0] != count:
            raise ValueError("A coefficient field does not match compact rows")
        destination_count = existing.shape[0]
        invalid = False
        with nogil:
            for cell in range(count):
                if rows[cell] < 0 or rows[cell] >= destination_count:
                    with gil:
                        raise IndexError("A scaled field row lies outside its destination")
                product = coefficient[cell] * scale[cell]
                if not isfinite(product) or product < 0.0:
                    invalid = True
        if invalid:
            return False
    for field in range(field_count):
        coefficient = coefficients[field]
        destination = destinations[field]
        with nogil:
            for cell in range(count):
                destination[rows[cell]] = coefficient[cell] * scale[cell]
    return True


def attenuate_available_line(
    const double[:] emissivity not None,
    const double[:] foreground_column not None,
    double cross_section,
):
    """Attenuate finite epsilon entries; copy missing NaN/Inf without inspecting N.

    The caller selects the default NumPy error policy. Nonfinite optical depth
    requests the original NumPy path to retain overflow warnings. All physical
    values, including the cross-section, arrive at runtime.
    """
    cdef Py_ssize_t count = emissivity.shape[0]
    if foreground_column.shape[0] != count:
        raise ValueError("Foreground column and emissivity shapes must match")
    if not isfinite(cross_section) or cross_section < 0.0:
        return None
    output = np.empty(count, dtype=np.float64)
    cdef double[:] attenuated = output
    cdef Py_ssize_t cell
    cdef double epsilon, column, optical_depth
    cdef bint fallback = False
    with nogil:
        for cell in range(count):
            epsilon = emissivity[cell]
            if not isfinite(epsilon):
                attenuated[cell] = epsilon
                continue
            column = foreground_column[cell]
            if epsilon < 0.0 or not isfinite(column) or column < 0.0:
                fallback = True
                break
            optical_depth = (-cross_section) * column
            if not isfinite(optical_depth):
                fallback = True
                break
            attenuated[cell] = epsilon * exp(optical_depth)
    if fallback:
        return None
    return output
