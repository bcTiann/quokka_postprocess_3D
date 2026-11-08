# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""Validate emissivity/temperature pairs without selected array copies."""
from libc.math cimport isfinite, isnan


def check_paired_fields(const double[:] emissivity, const double[:] temperature):
    """Check one line's epsilon [erg/s/cm^3] and temperature [K], both (B,).

    NaN emissivity denotes a missing result, so its temperature is ignored.
    Available epsilon must be finite and nonnegative, and temperature finite
    and positive. Emissivity errors take precedence if both fields are invalid.
    Inputs are never changed; success returns None, invalid data raises.
    """
    cdef Py_ssize_t index
    cdef bint invalid_emissivity = False
    cdef bint invalid_temperature = False
    cdef double value
    cdef double gas_temperature
    if temperature.shape[0] != emissivity.shape[0]:
        raise ValueError('Paired field sizes differ')
    with nogil:
        for index in range(emissivity.shape[0]):
            value = emissivity[index]
            if not isnan(value):
                if not isfinite(value) or value < 0.0:
                    invalid_emissivity = True
                gas_temperature = temperature[index]
                if not isfinite(gas_temperature) or gas_temperature <= 0.0:
                    invalid_temperature = True
    if invalid_emissivity:
        raise ValueError('Volume emissivity must be finite and nonnegative')
    if invalid_temperature:
        raise ValueError('Thermal temperature must be finite and positive')
