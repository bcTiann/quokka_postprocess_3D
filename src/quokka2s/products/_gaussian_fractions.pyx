# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Fill exact Gaussian fractions or accumulate tabulated channel luminosity.

The Python caller owns all buffers, cell chunks and ordered chunk merges.
SciPy's typed erf matches scipy.special.erf. The linear lookup path fuses
coefficient evaluation with ascending-cell luminosity accumulation.
"""

from scipy.special.cython_special cimport erf
from libc.math cimport fabs, isfinite


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


cdef void accumulate_dense_lookup_channels(
    const double[:] edges,
    const double[::1] centers,
    const double[::1] inverse_denominator,
    const double[:, :] luminosity,
    double[::1] previous,
    double[:, ::1] channel_luminosity,
    const double[::1] values,
) noexcept nogil:
    """Retain every multiply for nonfinite or negative input luminosities."""
    cdef Py_ssize_t channel_count = channel_luminosity.shape[0]
    cdef Py_ssize_t profile_count = channel_luminosity.shape[1]
    cdef Py_ssize_t cell_count = centers.shape[0]
    cdef Py_ssize_t channel
    cdef Py_ssize_t cell
    cdef Py_ssize_t profile
    cdef double difference
    cdef double argument
    cdef double current
    cdef double edge
    cdef double fraction
    cdef double weighted_luminosity
    for cell in range(cell_count):
        difference = edges[0] - centers[cell]
        argument = difference * inverse_denominator[cell]
        previous[cell] = linear_lookup_erf(argument, values)
    for channel in range(channel_count):
        for profile in range(profile_count):
            channel_luminosity[channel, profile] = 0.0
        edge = edges[channel + 1]
        for cell in range(cell_count):
            difference = edge - centers[cell]
            argument = difference * inverse_denominator[cell]
            current = linear_lookup_erf(argument, values)
            fraction = (current - previous[cell]) * 0.5
            previous[cell] = current
            for profile in range(profile_count):
                weighted_luminosity = fraction * luminosity[cell, profile]
                channel_luminosity[channel, profile] = (
                    channel_luminosity[channel, profile] + weighted_luminosity
                )


def accumulate_lookup_channels(
    const double[:] edges,
    const double[::1] centers,
    const double[::1] inverse_denominator,
    const double[:, :] luminosity,
    double[::1] previous,
    double[:, ::1] channel_luminosity,
    const double[::1] values,
):
    """Write (K,M) channel luminosities without a (K,R) fraction matrix.

    edges is (K+1,) [km/s], centers is (R,) [km/s], inverse_denominator is
    (R,) [1/(km/s)], and luminosity is (R,M) [erg/s], with its C/Fortran
    strides retained. Output is C-contiguous (K,M) [erg/s]; previous is
    caller-owned (R,) terminal-erf scratch. The immutable lookup has 32769
    nodes on [0, 8]. The caller divides by channel width and merges chunks.

    Finite nonnegative luminosities permit skipping bins whose erf endpoints
    are already the same +/-1 plateau. No additional tail cutoff is applied.
    Each retained bin accumulates cells in their original order. Other inputs
    use the dense reduction, including zero multiplies and NaN propagation.
    """
    cdef Py_ssize_t channel_count = channel_luminosity.shape[0]
    cdef Py_ssize_t profile_count = channel_luminosity.shape[1]
    cdef Py_ssize_t cell_count = centers.shape[0]
    cdef Py_ssize_t channel
    cdef Py_ssize_t cell
    cdef Py_ssize_t profile
    cdef Py_ssize_t first_channel
    cdef Py_ssize_t stop_channel
    cdef Py_ssize_t low
    cdef Py_ssize_t high
    cdef Py_ssize_t middle
    cdef double difference
    cdef double argument
    cdef double first_argument
    cdef double last_argument
    cdef double previous_value
    cdef double current
    cdef double fraction
    cdef double weighted_luminosity
    cdef bint safe = True
    if edges.shape[0] != channel_count + 1:
        raise ValueError("edges must have one more entry than output channels")
    if inverse_denominator.shape[0] != cell_count or previous.shape[0] != cell_count:
        raise ValueError("Gaussian widths and scratch must match the cell count")
    if luminosity.shape[0] != cell_count or luminosity.shape[1] != profile_count:
        raise ValueError("Luminosity must match the original cell/profile layout")
    if values.shape[0] != 32769:
        raise ValueError("The production linear lookup requires 32769 nodes")

    with nogil:
        for channel in range(channel_count + 1):
            if not isfinite(edges[channel]):
                safe = False
                break
            if channel > 0 and edges[channel] <= edges[channel - 1]:
                safe = False
                break
        if safe:
            for cell in range(cell_count):
                if not isfinite(centers[cell]) or not isfinite(inverse_denominator[cell]) or inverse_denominator[cell] <= 0.0:
                    safe = False
                    break
                for profile in range(profile_count):
                    if not isfinite(luminosity[cell, profile]) or luminosity[cell, profile] < 0.0:
                        safe = False
                        break
                if not safe:
                    break
        if not safe or channel_count == 0:
            accumulate_dense_lookup_channels(
                edges,
                centers,
                inverse_denominator,
                luminosity,
                previous,
                channel_luminosity,
                values,
            )
        else:
            for channel in range(channel_count):
                for profile in range(profile_count):
                    channel_luminosity[channel, profile] = 0.0
            for cell in range(cell_count):
                difference = edges[0] - centers[cell]
                first_argument = difference * inverse_denominator[cell]
                difference = edges[channel_count] - centers[cell]
                last_argument = difference * inverse_denominator[cell]
                previous[cell] = linear_lookup_erf(last_argument, values)
                if first_argument >= 8.0 or last_argument <= -8.0:
                    continue

                first_channel = 0
                if first_argument <= -8.0:
                    # Use actual boundary arithmetic, retaining one guard bin.
                    low = 0
                    high = channel_count
                    while low < high:
                        middle = (low + high) // 2
                        difference = edges[middle] - centers[cell]
                        argument = difference * inverse_denominator[cell]
                        if argument <= -8.0:
                            low = middle + 1
                        else:
                            high = middle
                    first_channel = low - 2
                    if first_channel < 0:
                        first_channel = 0
                stop_channel = channel_count
                if last_argument >= 8.0:
                    low = 0
                    high = channel_count
                    while low < high:
                        middle = (low + high) // 2
                        difference = edges[middle] - centers[cell]
                        argument = difference * inverse_denominator[cell]
                        if argument < 8.0:
                            low = middle + 1
                        else:
                            high = middle
                    stop_channel = low + 1
                    if stop_channel > channel_count:
                        stop_channel = channel_count

                difference = edges[first_channel] - centers[cell]
                argument = difference * inverse_denominator[cell]
                previous_value = linear_lookup_erf(argument, values)
                for channel in range(first_channel, stop_channel):
                    difference = edges[channel + 1] - centers[cell]
                    argument = difference * inverse_denominator[cell]
                    current = linear_lookup_erf(argument, values)
                    fraction = (current - previous_value) * 0.5
                    previous_value = current
                    for profile in range(profile_count):
                        weighted_luminosity = fraction * luminosity[cell, profile]
                        channel_luminosity[channel, profile] = (
                            channel_luminosity[channel, profile] + weighted_luminosity
                        )
