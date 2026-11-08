# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Validate once and partition active original rows between two sigma bands.

No logarithms, Gaussian interpolation weights, luminosity sums or scatter
updates are calculated here. Returned row vectors retain original cell order.
"""
from libc.math cimport isfinite
from libc.stdint cimport int64_t

import numpy as np


def select_original_rows(
    const double[:] centers,
    const double[:] sigmas,
    const double[:, :] luminosity,
    double mu_min,
    double mu_max,
    double sigma_min,
    double sigma_max,
    double threshold,
):
    """Return narrow rows, broad rows, and a slow-path validation flag.

The caller checks shapes first. A nonzero flag requests the original Python
validator so the established exception type/message remains authoritative.
Input rows with zero light may retain nonfinite Gaussian parameters, exactly
as in the existing pure-2D workflow. Input values are never changed.
"""
    cdef Py_ssize_t count = centers.shape[0]
    cdef Py_ssize_t profiles = luminosity.shape[1]
    cdef Py_ssize_t row
    cdef Py_ssize_t profile
    cdef Py_ssize_t narrow_count = 0
    cdef Py_ssize_t broad_count = 0
    cdef bint active
    cdef int invalid = 0
    cdef double light
    cdef int64_t[::1] narrow
    cdef int64_t[::1] broad
    if sigmas.shape[0] != count or luminosity.shape[0] != count:
        raise ValueError('The row selector requires matching input lengths')
    narrow_owner = np.empty(count, dtype=np.int64)
    broad_owner = np.empty(count, dtype=np.int64)
    narrow = narrow_owner
    broad = broad_owner
    with nogil:
        for row in range(count):
            active = False
            for profile in range(profiles):
                light = luminosity[row, profile]
                if not isfinite(light) or light < 0.0:
                    invalid = 1
                    break
                if light != 0.0:
                    active = True
            if invalid:
                break
            if not active:
                continue
            if not isfinite(centers[row]) or not isfinite(sigmas[row]):
                invalid = 2
                break
            if (centers[row] < mu_min or centers[row] > mu_max
                    or sigmas[row] < sigma_min or sigmas[row] > sigma_max):
                invalid = 3
                break
            if sigmas[row] < threshold:
                narrow[narrow_count] = row
                narrow_count += 1
            else:
                broad[broad_count] = row
                broad_count += 1
    return narrow_owner[:narrow_count], broad_owner[:broad_count], invalid
