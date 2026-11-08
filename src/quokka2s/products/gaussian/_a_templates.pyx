# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Shift a fractional-mu/sigma probability template; never query a CDF."""
from libc.math cimport floor
from libc.stdint cimport int64_t
from libc.string cimport memcpy


cdef inline void _sigma_bracket(
    double value,
    const double[::1] sigma_nodes,
    Py_ssize_t* lower,
    Py_ssize_t* upper,
    double* fraction,
) noexcept nogil:
    cdef Py_ssize_t left = 0
    cdef Py_ssize_t right = sigma_nodes.shape[0]
    cdef Py_ssize_t middle
    while left < right:
        middle = left + (right - left) // 2
        if sigma_nodes[middle] < value:
            left = middle + 1
        else:
            right = middle
    if sigma_nodes[left] == value:
        lower[0] = left
        upper[0] = left
        fraction[0] = 0.0
    else:
        lower[0] = left - 1
        upper[0] = left
        fraction[0] = (value - sigma_nodes[left - 1]) / (sigma_nodes[left] - sigma_nodes[left - 1])


def evaluate_into(
    const double[::1] mu,
    const double[::1] sigma,
    const double[::1] sigma_nodes,
    const double[:, :, ::1] table,
    double[:, ::1] output,
    double delta_v,
    Py_ssize_t n_max,
    Py_ssize_t R,
    const int64_t[::1] sigma_id,
    bint has_sigma_id,
):
    """Fill output[N,K] after Python has validated all ranges and shapes."""
    cdef Py_ssize_t count = mu.shape[0]
    cdef Py_ssize_t channels = output.shape[1]
    cdef Py_ssize_t cell
    cdef Py_ssize_t channel
    cdef Py_ssize_t sigma_lower
    cdef Py_ssize_t sigma_upper
    cdef Py_ssize_t fraction_lower
    cdef Py_ssize_t fraction_upper
    cdef Py_ssize_t integer_mu
    cdef Py_ssize_t offset
    cdef double coordinate
    cdef double fractional_mu
    cdef double fractional_coordinate
    cdef double mu_weight
    cdef double sigma_weight
    cdef double lower_value
    cdef double upper_value
    with nogil:
        for cell in range(count):
            coordinate = mu[cell] / delta_v
            integer_mu = <Py_ssize_t>floor(coordinate)
            fractional_mu = coordinate - integer_mu
            fractional_coordinate = fractional_mu * R
            fraction_lower = <Py_ssize_t>floor(fractional_coordinate)
            if fraction_lower >= R:
                fraction_lower = R
                fraction_upper = R
                mu_weight = 0.0
            else:
                fraction_upper = fraction_lower + 1
                # Recognize a represented grid point without a tolerance. This
                # also handles a fractional coordinate rounded just below it.
                if coordinate == integer_mu + (<double>fraction_lower / R):
                    fraction_upper = fraction_lower
                    mu_weight = 0.0
                elif coordinate == integer_mu + (<double>fraction_upper / R):
                    fraction_lower = fraction_upper
                    mu_weight = 0.0
                else:
                    mu_weight = fractional_coordinate - fraction_lower
            offset = n_max - integer_mu
            if has_sigma_id:
                sigma_lower = sigma_id[cell]
                sigma_upper = sigma_lower
                sigma_weight = 0.0
            else:
                _sigma_bracket(sigma[cell], sigma_nodes, &sigma_lower, &sigma_upper, &sigma_weight)
            if sigma_lower == sigma_upper:
                if fraction_lower == fraction_upper:
                    memcpy(&output[cell, 0], &table[sigma_lower, fraction_lower, offset], channels * sizeof(double))
                else:
                    for channel in range(channels):
                        output[cell, channel] = (
                            (1.0 - mu_weight) * table[sigma_lower, fraction_lower, offset + channel]
                            + mu_weight * table[sigma_lower, fraction_upper, offset + channel]
                        )
            elif fraction_lower == fraction_upper:
                for channel in range(channels):
                    output[cell, channel] = (
                        (1.0 - sigma_weight) * table[sigma_lower, fraction_lower, offset + channel]
                        + sigma_weight * table[sigma_upper, fraction_lower, offset + channel]
                    )
            else:
                for channel in range(channels):
                    lower_value = (
                        (1.0 - mu_weight) * table[sigma_lower, fraction_lower, offset + channel]
                        + mu_weight * table[sigma_lower, fraction_upper, offset + channel]
                    )
                    upper_value = (
                        (1.0 - mu_weight) * table[sigma_upper, fraction_lower, offset + channel]
                        + mu_weight * table[sigma_upper, fraction_upper, offset + channel]
                    )
                    output[cell, channel] = (1.0 - sigma_weight) * lower_value + sigma_weight * upper_value
