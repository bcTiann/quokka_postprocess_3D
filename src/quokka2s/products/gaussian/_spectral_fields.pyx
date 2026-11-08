# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""Prepare final cold/hot spectral inputs without packed full-group copies.

The caller owns the original read-only cell and per-line arrays. Returned
temperature/luminosity matrices have shape (Nline, Nbranch) and Fortran order,
matching the frozen branch arrays and their NumPy axis-1 reduction order.
"""

import numpy as np
cimport numpy as cnp

from libc.math cimport isfinite
from libc.stdint cimport int64_t


def prepare_available_group(
    const double[:] velocity_kms,
    const cnp.npy_bool[:] cold_cells,
    const int64_t[:] available_rows,
    tuple intrinsic_fields,
    tuple attenuated_fields,
    tuple temperature_fields,
    double cell_volume_cm3,
    double speed_of_light_kms,
):
    """Select each branch directly from original (B,) fields in cell order.

    No line may borrow another line's missingness mask. The caller has grouped
    only lines with identical available_rows, and retains that group order.
    All values are checked before the returned luminosities enter a deposit.
    """
    cdef Py_ssize_t cell_count = velocity_kms.shape[0]
    cdef Py_ssize_t line_count = len(intrinsic_fields)
    cdef Py_ssize_t selected_count = available_rows.shape[0]
    cdef Py_ssize_t cold_count = 0
    cdef Py_ssize_t hot_count = 0
    cdef Py_ssize_t cold_position = 0
    cdef Py_ssize_t hot_position = 0
    cdef Py_ssize_t position
    cdef Py_ssize_t row
    cdef Py_ssize_t line_index
    cdef Py_ssize_t dust_index
    cdef int error_kind = 0
    cdef double value
    cdef double luminosity
    cdef const double[:] source
    cdef int64_t[:] cold_rows
    cdef int64_t[:] hot_rows
    cdef double[:] cold_velocity
    cdef double[:] hot_velocity
    cdef double[::1, :] cold_destination
    cdef double[::1, :] hot_destination

    if cold_cells.shape[0] != cell_count:
        raise ValueError("The branch mask must match the original cell shape")
    if len(attenuated_fields) != line_count or len(temperature_fields) != line_count:
        raise ValueError("Each line requires intrinsic, attenuated and temperature fields")
    with nogil:
        for position in range(selected_count):
            row = available_rows[position]
            if row < 0 or row >= cell_count:
                error_kind = 1
                break
            value = velocity_kms[row]
            if not isfinite(value):
                error_kind = 2
                break
            if value <= -speed_of_light_kms or value >= speed_of_light_kms:
                error_kind = 3
                break
            if cold_cells[row]:
                cold_count += 1
            else:
                hot_count += 1
    if error_kind == 1:
        raise IndexError("An available cell index is outside the input batch")
    if error_kind == 2:
        raise ValueError("velocity_kms must be finite")
    if error_kind == 3:
        raise ValueError("The nonrelativistic profile requires abs(velocity) < c")

    cold_rows_owner = np.empty(cold_count, dtype=np.int64)
    hot_rows_owner = np.empty(hot_count, dtype=np.int64)
    cold_velocity_owner = np.empty(cold_count, dtype=np.float64)
    hot_velocity_owner = np.empty(hot_count, dtype=np.float64)
    cold_rows = cold_rows_owner
    hot_rows = hot_rows_owner
    cold_velocity = cold_velocity_owner
    hot_velocity = hot_velocity_owner
    with nogil:
        for position in range(selected_count):
            row = available_rows[position]
            if cold_cells[row]:
                cold_rows[cold_position] = row
                cold_velocity[cold_position] = velocity_kms[row]
                cold_position += 1
            else:
                hot_rows[hot_position] = row
                hot_velocity[hot_position] = velocity_kms[row]
                hot_position += 1

    shape_cold = (line_count, cold_count)
    shape_hot = (line_count, hot_count)
    cold_temperature = np.empty(shape_cold, dtype=np.float64, order="F")
    hot_temperature = np.empty(shape_hot, dtype=np.float64, order="F")
    cold_intrinsic = np.empty(shape_cold, dtype=np.float64, order="F")
    hot_intrinsic = np.empty(shape_hot, dtype=np.float64, order="F")
    cold_attenuated = np.empty(shape_cold, dtype=np.float64, order="F")
    hot_attenuated = np.empty(shape_hot, dtype=np.float64, order="F")

    cold_destination = cold_temperature
    hot_destination = hot_temperature
    for line_index in range(line_count):
        source = temperature_fields[line_index]
        if source.shape[0] != cell_count:
            raise ValueError("Each line temperature must match the original cell shape")
        with nogil:
            for position in range(cold_count):
                value = source[cold_rows[position]]
                if not isfinite(value) or value < 0.0:
                    error_kind = 4
                    break
                cold_destination[line_index, position] = value
            if error_kind == 0:
                for position in range(hot_count):
                    value = source[hot_rows[position]]
                    if not isfinite(value) or value < 0.0:
                        error_kind = 4
                        break
                    hot_destination[line_index, position] = value
        if error_kind:
            raise ValueError("temperatures_K must be finite and nonnegative")

    for dust_index in range(2):
        if dust_index == 0:
            fields = intrinsic_fields
            cold_destination = cold_intrinsic
            hot_destination = hot_intrinsic
        else:
            fields = attenuated_fields
            cold_destination = cold_attenuated
            hot_destination = hot_attenuated
        for line_index in range(line_count):
            source = fields[line_index]
            if source.shape[0] != cell_count:
                raise ValueError("Each line emissivity must match the original cell shape")
            with nogil:
                for position in range(cold_count):
                    value = source[cold_rows[position]]
                    if not isfinite(value) or value < 0.0:
                        error_kind = 5
                        break
                    luminosity = value * cell_volume_cm3
                    if not isfinite(luminosity):
                        error_kind = 6
                        break
                    cold_destination[line_index, position] = luminosity
                if error_kind == 0:
                    for position in range(hot_count):
                        value = source[hot_rows[position]]
                        if not isfinite(value) or value < 0.0:
                            error_kind = 5
                            break
                        luminosity = value * cell_volume_cm3
                        if not isfinite(luminosity):
                            error_kind = 6
                            break
                        hot_destination[line_index, position] = luminosity
            if error_kind == 5:
                raise ValueError("emissivity must be finite and nonnegative")
            if error_kind == 6:
                raise ValueError("emissivity times volume must remain finite")

    return (
        (cold_velocity_owner, cold_temperature, cold_intrinsic, cold_attenuated),
        (hot_velocity_owner, hot_temperature, hot_intrinsic, hot_attenuated),
    )


def pack_width_group(
    const double[:, :] intrinsic_luminosity,
    const double[:, :] attenuated_luminosity,
    const int64_t[:] shared_lines,
):
    """Copy directly to the frozen (Nbranch, 2*Nshared) deposit layout.

    The old expression first copied each selected line matrix, transposed both
    and concatenated them. This allocates only the final matrix. Intrinsic line
    columns still precede attenuated line columns in the supplied line order.
    """
    cdef Py_ssize_t cell_count = intrinsic_luminosity.shape[1]
    cdef Py_ssize_t line_count = intrinsic_luminosity.shape[0]
    cdef Py_ssize_t shared_count = shared_lines.shape[0]
    cdef Py_ssize_t shared_index
    cdef Py_ssize_t line_index
    cdef Py_ssize_t row
    cdef double[:, :] destination

    if attenuated_luminosity.shape[0] != line_count or attenuated_luminosity.shape[1] != cell_count:
        raise ValueError("Both dust states must have matching branch shapes")
    for shared_index in range(shared_count):
        if shared_lines[shared_index] < 0 or shared_lines[shared_index] >= line_count:
            raise IndexError("A shared-width line index is outside the branch matrix")
    # NumPy concatenate chooses C order for degenerate single-column inputs.
    output_order = "F" if shared_count > 1 and cell_count > 1 else "C"
    output = np.empty((cell_count, 2 * shared_count), dtype=np.float64, order=output_order)
    destination = output
    with nogil:
        for shared_index in range(shared_count):
            line_index = shared_lines[shared_index]
            for row in range(cell_count):
                destination[row, shared_index] = intrinsic_luminosity[line_index, row]
                destination[row, shared_count + shared_index] = attenuated_luminosity[line_index, row]
    return output
