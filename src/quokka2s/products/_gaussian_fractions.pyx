# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Fill Gaussian channel fractions without retaining a channel-edge matrix.

The Python caller owns all buffers and the luminosity reduction. SciPy's typed
erf is the same real-double implementation used by scipy.special.erf.
"""

from scipy.special.cython_special cimport erf
from libc.math cimport fabs


cdef inline double saturated_erf(double argument) noexcept nogil:
    # At these arguments the pinned SciPy float64 erf is already exactly +/-1.
    # This shortcut preserves its rounded value; no nonzero tail is truncated.
    if argument >= 8.0:
        return 1.0
    if argument <= -8.0:
        return -1.0
    return erf(argument)


def fill_channel_fractions(
    const double[:] edges,
    const double[::1] centers,
    const double[::1] denominator,
    double[::1] previous,
    double[:, ::1] fractions,
):
    """Write dimensionless fractions (K,R) for K channels and R original cells.

    edges is (K+1,) [km/s]; centers and denominator are (R,) [km/s]. The
    denominator is NumPy's already-calculated sqrt(2)*max(sigma, 1e-5 km/s).
    previous is worker-owned (R,) scratch. fractions must be C-contiguous;
    edges may be strided or read-only. No input values or cell order change.
    """
    cdef Py_ssize_t channel_count = fractions.shape[0]
    cdef Py_ssize_t cell_count = fractions.shape[1]
    cdef Py_ssize_t channel
    cdef Py_ssize_t cell
    cdef double edge
    cdef double difference
    cdef double argument
    cdef double current
    cdef double difference_erf

    # Validate every shape before entering the unchecked, GIL-free loop.
    if edges.shape[0] != channel_count + 1:
        raise ValueError("edges must have one more entry than channel rows")
    if centers.shape[0] != cell_count or denominator.shape[0] != cell_count:
        raise ValueError("centers and denominator must match the cell columns")
    if previous.shape[0] != cell_count:
        raise ValueError("previous scratch must match the cell columns")

    with nogil:
        for cell in range(cell_count):
            difference = edges[0] - centers[cell]
            argument = difference / denominator[cell]
            previous[cell] = saturated_erf(argument)
        for channel in range(channel_count):
            edge = edges[channel + 1]
            for cell in range(cell_count):
                difference = edge - centers[cell]
                argument = difference / denominator[cell]
                current = saturated_erf(argument)
                difference_erf = current - previous[cell]
                fractions[channel, cell] = difference_erf * 0.5
                previous[cell] = current


cdef inline double linear_lookup_erf(
    double argument,
    const double[::1] values,
) noexcept nogil:
    """Interpolate the immutable nonnegative erf table at 4096 nodes/unit."""
    cdef double absolute = fabs(argument)
    cdef double coordinate
    cdef double fraction
    cdef double result
    cdef Py_ssize_t index
    if argument != argument:
        return argument
    if absolute >= 8.0:
        return saturated_erf(argument)
    coordinate = absolute * 4096.0
    index = <Py_ssize_t>coordinate
    fraction = coordinate - index
    result = values[index] + fraction * (values[index + 1] - values[index])
    if argument < 0.0:
        return -result
    return result


def fill_channel_fractions_linear_lut(
    const double[:] edges,
    const double[::1] centers,
    const double[::1] inverse_denominator,
    double[::1] previous,
    double[:, ::1] fractions,
    const double[::1] values,
):
    """Approximate channel fractions with a one-dimensional linear erf lookup.

    Shapes match fill_channel_fractions(); inverse_denominator contains
    1/(sqrt(2)*max(sigma, 1e-5 km/s)) [1/(km/s)]. values is the read-only
    32769-node erf table on [0,8], with argument spacing 1/4096.
    Reciprocal multiplication and interpolation change floating-point results;
    the existing +/-8 saturation and caller's luminosity reduction are retained.
    """
    cdef Py_ssize_t channel_count = fractions.shape[0]
    cdef Py_ssize_t cell_count = fractions.shape[1]
    cdef Py_ssize_t channel
    cdef Py_ssize_t cell
    cdef double difference
    cdef double argument
    cdef double edge
    cdef double current
    if edges.shape[0] != channel_count + 1:
        raise ValueError("edges must have one more entry than channel rows")
    if centers.shape[0] != cell_count or inverse_denominator.shape[0] != cell_count:
        raise ValueError("centers and inverse_denominator must match the cell columns")
    if previous.shape[0] != cell_count:
        raise ValueError("previous scratch must match the cell columns")
    if values.shape[0] != 32769:
        raise ValueError("linear erf lookup requires 32769 nodes on [0,8]")
    with nogil:
        for cell in range(cell_count):
            difference = edges[0] - centers[cell]
            argument = difference * inverse_denominator[cell]
            previous[cell] = linear_lookup_erf(argument, values)
        for channel in range(channel_count):
            edge = edges[channel + 1]
            for cell in range(cell_count):
                difference = edge - centers[cell]
                argument = difference * inverse_denominator[cell]
                current = linear_lookup_erf(argument, values)
                fractions[channel, cell] = (current - previous[cell]) * 0.5
                previous[cell] = current
