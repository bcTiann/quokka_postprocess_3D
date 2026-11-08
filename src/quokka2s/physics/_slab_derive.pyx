# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Fused, independent full-z columns; preserve the frozen scalar operation order."""
from libc.math cimport fabs, isfinite


def fill_columns(
    const double[:, :, ::1] density_raw,
    const double[:, :, ::1] temperature_raw,
    const double[:, :, ::1] momentum_x,
    const double[:, :, ::1] momentum_y,
    const double[:, :, ::1] momentum_z,
    double[:, :, :, ::1] output,
    double[:, ::1] scratch,
    const double[:, ::1] velocity_factors,
    double density_to_gas,
    double gas_density_to_cgs,
    double hydrogen_fraction,
    double hydrogen_mass_g,
    double dx_cm,
    double dy_cm,
    double dz_gradient_cm,
    double dz_foreground_cm,
    double dz_shielding_cm,
    double shielding_to_cgs,
    double gradient_floor_s,
    Py_ssize_t x_start,
    Py_ssize_t y_start,
    Py_ssize_t first_column,
    Py_ssize_t stop_column,
):
    """Fill owned output[6,nx,ny,nz] for disjoint flattened XY columns.

    scratch[2,nz] belongs only to this task: nH then central vz in cm/s.
    Inputs are global read-only XYZ C-order float64 raw native values.
    velocity_factors[axis,:] stores division's unit coefficient, native->gas,
    gas->cgs, and gas->km/s, in the original sequential conversion order.
    """
    cdef Py_ssize_t nx = density_raw.shape[0]
    cdef Py_ssize_t ny = density_raw.shape[1]
    cdef Py_ssize_t nz = density_raw.shape[2]
    cdef Py_ssize_t selected_ny = output.shape[2]
    cdef Py_ssize_t column, local_x, local_y, x, y, xp, xm, yp, ym, k
    cdef double density, nH, native_velocity, gas_velocity
    cdef double reverse_column, forward_column, forward_density, cell_column
    cdef double upper_velocity, lower_velocity, dvx_dx, dvy_dy, dvz_dz
    cdef double divergence, gradient
    cdef double twice_dx = 2.0 * dx_cm
    cdef double twice_dy = 2.0 * dy_cm
    cdef double twice_dz = 2.0 * dz_gradient_cm
    cdef Py_ssize_t invalid_density_count = 0
    with nogil:
        for column in range(first_column, stop_column):
            local_x = column // selected_ny
            local_y = column - local_x * selected_ny
            x = x_start + local_x
            y = y_start + local_y
            xp = 0 if x + 1 == nx else x + 1
            xm = nx - 1 if x == 0 else x - 1
            yp = 0 if y + 1 == ny else y + 1
            ym = ny - 1 if y == 0 else y - 1

            for k in range(nz):
                density = density_raw[x, y, k] * density_to_gas
                density = density * gas_density_to_cgs
                output[0, local_x, local_y, k] = density
                output[2, local_x, local_y, k] = temperature_raw[x, y, k]
                nH = density * hydrogen_fraction
                nH = nH / hydrogen_mass_g
                if not isfinite(nH) or nH < 0.0:
                    invalid_density_count += 1
                scratch[0, k] = nH
                native_velocity = momentum_z[x, y, k] / density_raw[x, y, k]
                native_velocity = native_velocity * velocity_factors[2, 0]
                gas_velocity = native_velocity * velocity_factors[2, 1]
                output[5, local_x, local_y, k] = gas_velocity * velocity_factors[2, 3]
                scratch[1, k] = gas_velocity * velocity_factors[2, 2]

            # Reverse cumsum is independent: never subtract a forward prefix
            # from a total. Store its reciprocal in the eventual NH output.
            reverse_column = 0.0
            for k in range(nz - 1, -1, -1):
                cell_column = scratch[0, k] * dz_shielding_cm
                reverse_column = reverse_column + cell_column
                output[3, local_x, local_y, k] = 1.0 / reverse_column

            forward_column = 0.0
            forward_density = 0.0
            for k in range(nz):
                nH = scratch[0, k]
                forward_density = forward_density + nH
                output[1, local_x, local_y, k] = (forward_density - 0.5 * nH) * dz_foreground_cm
                cell_column = nH * dz_shielding_cm
                forward_column = forward_column + cell_column
                output[3, local_x, local_y, k] = 2.0 / (
                    output[3, local_x, local_y, k] + 1.0 / forward_column
                )
                output[3, local_x, local_y, k] = output[3, local_x, local_y, k] * shielding_to_cgs

                upper_velocity = momentum_x[xp, y, k] / density_raw[xp, y, k]
                upper_velocity = upper_velocity * velocity_factors[0, 0]
                upper_velocity = upper_velocity * velocity_factors[0, 1]
                upper_velocity = upper_velocity * velocity_factors[0, 2]
                lower_velocity = momentum_x[xm, y, k] / density_raw[xm, y, k]
                lower_velocity = lower_velocity * velocity_factors[0, 0]
                lower_velocity = lower_velocity * velocity_factors[0, 1]
                lower_velocity = lower_velocity * velocity_factors[0, 2]
                dvx_dx = (upper_velocity - lower_velocity) / twice_dx
                upper_velocity = momentum_y[x, yp, k] / density_raw[x, yp, k]
                upper_velocity = upper_velocity * velocity_factors[1, 0]
                upper_velocity = upper_velocity * velocity_factors[1, 1]
                upper_velocity = upper_velocity * velocity_factors[1, 2]
                lower_velocity = momentum_y[x, ym, k] / density_raw[x, ym, k]
                lower_velocity = lower_velocity * velocity_factors[1, 0]
                lower_velocity = lower_velocity * velocity_factors[1, 1]
                lower_velocity = lower_velocity * velocity_factors[1, 2]
                dvy_dy = (upper_velocity - lower_velocity) / twice_dy
                if k == 0:
                    dvz_dz = (scratch[1, 1] - scratch[1, 0]) / dz_gradient_cm
                elif k == nz - 1:
                    dvz_dz = (scratch[1, nz - 1] - scratch[1, nz - 2]) / dz_gradient_cm
                else:
                    dvz_dz = (scratch[1, k + 1] - scratch[1, k - 1]) / twice_dz
                divergence = dvx_dx + dvy_dy
                divergence = divergence + dvz_dz
                gradient = fabs(divergence) / 3.0
                if gradient < gradient_floor_s:
                    gradient = gradient_floor_s
                output[4, local_x, local_y, k] = gradient
    return invalid_density_count
