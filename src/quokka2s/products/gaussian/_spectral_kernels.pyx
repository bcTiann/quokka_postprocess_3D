# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Stable centered moments over borrowed Float64 spectral input arrays.

Each profile selects L>0 rows and uses its first emitting velocity as anchor.
The first pass obtains total light and the mean offset; the second sums the
centered bulk plus thermal variance. Width floors are runtime Python inputs.
"""

from libc.math cimport isfinite, isnan, sqrt
from libc.stdint cimport int64_t

import numpy as np


def validate_branch_fields(
    const int64_t[:] rows,
    intrinsic_fields,
    attenuated_fields,
    temperature_fields,
    double cell_volume_cm3,
    Py_ssize_t original_count,
):
    """Validate selected original fields without allocating gathered vectors.

    Field tuples contain Float64 original-cell vectors. Temperature identities
    share validation only within this call. Emitting flags include multiplication
    underflow, using the original maximum-emissivity times volume criterion.
    """
    cdef Py_ssize_t line_count = len(intrinsic_fields)
    if not isfinite(cell_volume_cm3) or cell_volume_cm3 <= 0.0:
        raise ValueError("Require finite positive cell_volume_cm3")
    if len(attenuated_fields) != line_count or len(temperature_fields) != line_count:
        raise ValueError("Each line requires intrinsic, attenuated and temperature fields")
    output = np.zeros(line_count, dtype=np.uint8)
    cdef unsigned char[:] emitting = output
    cdef const double[:] intrinsic
    cdef const double[:] attenuated
    cdef const double[:] temperature
    cdef const double[:] emissivity
    cdef Py_ssize_t line, selected, source, dust
    cdef double value, maximum, maximum_luminosity
    checked_temperatures = set()
    for line in range(line_count):
        intrinsic = intrinsic_fields[line]
        attenuated = attenuated_fields[line]
        temperature = temperature_fields[line]
        if (
            intrinsic.shape[0] != original_count
            or attenuated.shape[0] != original_count
            or temperature.shape[0] != original_count
        ):
            raise ValueError("Per-line fields must match the original cell shape")
        # NumPy gathers both epsilon arrays before temperature validation.
        if line == 0:
            with nogil:
                for selected in range(rows.shape[0]):
                    source = rows[selected]
                    if source < -original_count or source >= original_count:
                        with gil:
                            raise IndexError("An available cell index is outside the input batch")
        identity = id(temperature_fields[line])
        if identity not in checked_temperatures:
            with nogil:
                for selected in range(rows.shape[0]):
                    source = rows[selected]
                    if source < 0:
                        source += original_count
                    value = temperature[source]
                    if not isfinite(value) or value < 0.0:
                        with gil:
                            raise ValueError("temperatures_K must be finite and nonnegative")
            checked_temperatures.add(identity)
        for dust in range(2):
            emissivity = intrinsic if dust == 0 else attenuated
            maximum = 0.0
            with nogil:
                for selected in range(rows.shape[0]):
                    source = rows[selected]
                    if source < 0:
                        source += original_count
                    value = emissivity[source]
                    if not isfinite(value) or value < 0.0:
                        with gil:
                            raise ValueError("emissivity must be finite and nonnegative")
                    if value > maximum:
                        maximum = value
                maximum_luminosity = maximum * cell_volume_cm3
            if not isfinite(maximum_luminosity):
                raise ValueError("emissivity times volume must remain finite")
            if maximum_luminosity != 0.0:
                emitting[line] = 1
    return output.view(np.bool_)


def selected_temperatures_match(
    const int64_t[:] rows,
    const double[:] first,
    const double[:] second,
):
    """Match array_equal on selected equal-shaped vectors without gathering."""
    cdef Py_ssize_t count = first.shape[0]
    cdef Py_ssize_t selected, source
    cdef bint equal = True
    if second.shape[0] != count:
        raise ValueError("Temperature comparison requires matching source shapes")
    with nogil:
        for selected in range(rows.shape[0]):
            source = rows[selected]
            if source < 0:
                source += count
            if source < 0 or source >= count:
                with gil:
                    raise IndexError("An available cell index is outside the input batch")
            if first[source] != second[source]:
                equal = False
                # Continue checking indices, as NumPy gathers all rows first.
    return bool(equal)


def prepare_width_group_fields(
    const int64_t[:] rows,
    intrinsic_fields,
    attenuated_fields,
    const int64_t[:] shared_lines,
    const double[:] temperature_K,
    const double[:] velocity_kms,
    double cell_volume_cm3,
    double mass_g,
    double boltzmann_erg_K,
    double speed_of_light_kms,
    double velocity_cm_per_km,
):
    """Write final F-order L[N,2*line] and shared thermal width[N] directly.

    The caller validated this branch before grouping. Runtime Python inputs
    define all units and physical constants. No temporary epsilon/temperature
    selections are created. Nonfinite arithmetic requests the NumPy fallback.
    """
    cdef Py_ssize_t count = rows.shape[0]
    cdef Py_ssize_t original_count = temperature_K.shape[0]
    cdef Py_ssize_t group_size = shared_lines.shape[0]
    cdef Py_ssize_t line_count = len(intrinsic_fields)
    if velocity_kms.shape[0] != count or len(attenuated_fields) != line_count:
        raise ValueError("Selected velocities and per-line fields have incompatible shapes")
    if (
        not isfinite(mass_g) or mass_g <= 0.0
        or not isfinite(boltzmann_erg_K) or boltzmann_erg_K <= 0.0
        or not isfinite(speed_of_light_kms) or speed_of_light_kms <= 0.0
        or not isfinite(velocity_cm_per_km) or velocity_cm_per_km <= 0.0
    ):
        return None
    light_output = np.empty((count, 2 * group_size), dtype=np.float64, order="F")
    width_output = np.empty(count, dtype=np.float64)
    cdef double[:, :] luminosity = light_output
    cdef double[:] width = width_output
    cdef const double[:] emissivity
    cdef Py_ssize_t selected, source, column, line, dust
    cdef double value, thermal
    cdef bint fallback = False
    with nogil:
        for selected in range(count):
            source = rows[selected]
            if source < 0:
                source += original_count
            if source < 0 or source >= original_count:
                with gil:
                    raise IndexError("An available cell index is outside the input batch")
            thermal = sqrt(boltzmann_erg_K * temperature_K[source] / mass_g) / velocity_cm_per_km
            thermal *= 1.0 - velocity_kms[selected] / speed_of_light_kms
            width[selected] = thermal
            if not isfinite(thermal):
                fallback = True
    for dust in range(2):
        fields = intrinsic_fields if dust == 0 else attenuated_fields
        for column in range(group_size):
            line = shared_lines[column]
            if line < 0:
                line += line_count
            if line < 0 or line >= line_count:
                raise IndexError("A shared-width line index is outside the branch fields")
            emissivity = fields[line]
            if emissivity.shape[0] != original_count:
                raise ValueError("Per-line fields must match the original cell shape")
            with nogil:
                for selected in range(count):
                    source = rows[selected]
                    if source < 0:
                        source += original_count
                    value = emissivity[source] * cell_volume_cm3
                    luminosity[selected, dust * group_size + column] = value
                    if not isfinite(value):
                        fallback = True
    if fallback:
        return None
    return width_output, light_output


def centered_full_moments(
    const double[:] velocity_kms,
    const double[:] thermal_width_kms,
    const double[:, :] luminosity,
    double minimum_width_kms,
    double absolute_minimum_width_kms,
):
    """Return [profile, (light, centroid, centered second)] or request fallback.

    Arrays may be read-only and arbitrarily strided. Inputs are never changed
    or retained. A nonfinite reduction returns None so the existing NumPy path
    can retain its warnings and nonfinite behavior under the default policy.
    """
    cdef Py_ssize_t cell_count = velocity_kms.shape[0]
    cdef Py_ssize_t profile_count = luminosity.shape[1]
    if (
        thermal_width_kms.shape[0] != cell_count
        or luminosity.shape[0] != cell_count
        or profile_count < 1
        or profile_count > 12
    ):
        raise ValueError("Require matching vectors and one to twelve profiles")
    output = np.zeros((profile_count, 3), dtype=np.float64)
    cdef double[:, :] moments = output
    cdef Py_ssize_t profile, cell, first_emitting
    cdef double light, total, weighted_offset, anchor, offset, mean_offset
    cdef double width, centered_offset, centered_second, centroid
    cdef bint numpy_fallback = False
    with nogil:
        for profile in range(profile_count):
            first_emitting = -1
            for cell in range(cell_count):
                if luminosity[cell, profile] > 0.0:
                    first_emitting = cell
                    break
            if first_emitting < 0:
                continue
            anchor = velocity_kms[first_emitting]
            total = 0.0
            weighted_offset = 0.0
            for cell in range(first_emitting, cell_count):
                light = luminosity[cell, profile]
                if light > 0.0:
                    offset = velocity_kms[cell] - anchor
                    total += light
                    weighted_offset += light * offset
            if not isfinite(total) or total <= 0.0 or not isfinite(weighted_offset):
                numpy_fallback = True
                break
            mean_offset = weighted_offset / total
            centroid = anchor + mean_offset
            centered_second = 0.0
            for cell in range(first_emitting, cell_count):
                light = luminosity[cell, profile]
                if light > 0.0:
                    centered_offset = (velocity_kms[cell] - anchor) - mean_offset
                    width = thermal_width_kms[cell]
                    # Match both np.maximum calls, including NaN propagation.
                    if isnan(minimum_width_kms) or width < minimum_width_kms:
                        width = minimum_width_kms
                    if isnan(absolute_minimum_width_kms) or width < absolute_minimum_width_kms:
                        width = absolute_minimum_width_kms
                    centered_second += light * (
                        centered_offset * centered_offset + width * width
                    )
            if not isfinite(centroid) or not isfinite(centered_second):
                numpy_fallback = True
                break
            moments[profile, 0] = total
            moments[profile, 1] = centroid
            moments[profile, 2] = centered_second
    if numpy_fallback:
        return None
    return output
