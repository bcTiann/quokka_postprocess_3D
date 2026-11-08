# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Evaluate 3D field bundles in SciPy's generic linear corner/addition order."""
from libc.math cimport NAN, isinf, isnan


cdef inline Py_ssize_t lower_interval(
    const double[::1] axis,
    double coordinate,
) noexcept nogil:
    """Locate axis[i] <= x < axis[i+1], including the final endpoint."""
    cdef Py_ssize_t lower = 0
    cdef Py_ssize_t upper = axis.shape[0] - 1
    cdef Py_ssize_t middle
    if coordinate == axis[upper]:
        return upper - 1
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if coordinate >= axis[middle]:
            lower = middle
        else:
            upper = middle
    return lower


cpdef bint fill_linear_3d(
    const double[::1] density_axis,
    const double[::1] column_axis,
    const double[::1] gradient_axis,
    const double[:, ::1] points,
    const double[:, :, :, ::1] table,
    double[:, ::1] output,
):
    """Write QxF values for log-coordinate rows Qx3 and a table NxMxKxF.

    Axes must be increasing, finite and contain at least two nodes. All tables
    may contain independent quiet NaNs. No corner is skipped for zero weight.
    Return False for a non-NaN out-of-bounds row so the caller can repeat the
    entire chunk with SciPy, preserving its floating-point warning behavior.
    Inputs and output are owned by the calling lookup/batch, never retained.
    """
    cdef Py_ssize_t query_count = points.shape[0]
    cdef Py_ssize_t field_count = table.shape[3]
    cdef Py_ssize_t query
    cdef Py_ssize_t field
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t corner_i
    cdef Py_ssize_t corner_j
    cdef Py_ssize_t corner_k
    cdef double x
    cdef double y
    cdef double z
    cdef double dx
    cdef double dy
    cdef double dz
    cdef double wx
    cdef double wy
    cdef double wz
    cdef double weight
    cdef double term

    if points.shape[1] != 3:
        raise ValueError("DESPOTIC points must have three logarithmic coordinates")
    if output.shape[0] != query_count or output.shape[1] != field_count:
        raise ValueError("DESPOTIC output must match the query and field counts")
    if density_axis.shape[0] < 2 or column_axis.shape[0] < 2 or gradient_axis.shape[0] < 2:
        raise ValueError("Native DESPOTIC axes require at least two nodes")
    if table.shape[0] != density_axis.shape[0] or table.shape[1] != column_axis.shape[0] or table.shape[2] != gradient_axis.shape[0]:
        raise ValueError("DESPOTIC table dimensions must match its axes")

    with nogil:
        for query in range(query_count):
            x = points[query, 0]
            y = points[query, 1]
            z = points[query, 2]
            if isinf(x) or isinf(y) or isinf(z):
                return False
            if x < density_axis[0] or x > density_axis[density_axis.shape[0] - 1]:
                return False
            if y < column_axis[0] or y > column_axis[column_axis.shape[0] - 1]:
                return False
            if z < gradient_axis[0] or z > gradient_axis[gradient_axis.shape[0] - 1]:
                return False
            if isnan(x) or isnan(y) or isnan(z):
                for field in range(field_count):
                    output[query, field] = NAN
                continue

            i = lower_interval(density_axis, x)
            j = lower_interval(column_axis, y)
            k = lower_interval(gradient_axis, z)
            dx = (x - density_axis[i]) / (density_axis[i + 1] - density_axis[i])
            dy = (y - column_axis[j]) / (column_axis[j + 1] - column_axis[j])
            dz = (z - gradient_axis[k]) / (gradient_axis[k + 1] - gradient_axis[k])
            for field in range(field_count):
                output[query, field] = 0.0

            # SciPy itertools.product order is x outermost, z innermost.
            # Its weight starts at 1, multiplies x/y/z in that order, then
            # multiplies the table value and adds this term to the running sum.
            for corner_i in range(2):
                wx = dx if corner_i else 1.0 - dx
                for corner_j in range(2):
                    wy = dy if corner_j else 1.0 - dy
                    for corner_k in range(2):
                        wz = dz if corner_k else 1.0 - dz
                        weight = 1.0
                        weight = weight * wx
                        weight = weight * wy
                        weight = weight * wz
                        for field in range(field_count):
                            term = table[i + corner_i, j + corner_j, k + corner_k, field] * weight
                            output[query, field] = output[query, field] + term
    return True
