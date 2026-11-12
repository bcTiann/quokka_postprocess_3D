"""Select available spectral rows and own the final NumPy Gaussian queues.

Cell velocities and temperatures use km/s and K. Emissivities [erg/s/cm^3]
become luminosities [erg/s] only after applying the original cell volume.
Unavailable rows are never inspected or borrowed into another line's mask.
"""
from __future__ import annotations

import numpy as np

from ..line_velocity_moments import LineVelocityMoments


def prepare_branch_rows(available_rows, cold_cells, velocity_kms, speed_of_light_kms):
    """Own increasing available row IDs and velocities for cold/hot branches."""
    if cold_cells.shape != velocity_kms.shape:
        raise ValueError("The branch mask must match the original cell shape")
    if np.any(available_rows < 0) or np.any(available_rows >= velocity_kms.size):
        raise IndexError("An available cell index is outside the input batch")
    velocity = velocity_kms[available_rows]
    if not np.isfinite(velocity).all():
        raise ValueError("velocity_kms must be finite")
    if np.any(np.abs(velocity) >= speed_of_light_kms):
        raise ValueError("The nonrelativistic profile requires abs(velocity) < c")
    cold = cold_cells[available_rows]
    return tuple(
        (available_rows[selected], velocity[selected])
        for selected in (cold, ~cold)
    )


def validate_branch_emission(
    rows, intrinsic_fields, attenuated_fields, temperature_fields,
    cell_volume_cm3, original_count,
):
    """Validate only a line's available branch and flag its nonzero luminosity."""
    if not np.isfinite(cell_volume_cm3) or cell_volume_cm3 <= 0.0:
        raise ValueError("Require finite positive cell_volume_cm3")
    if (
        len(intrinsic_fields) != len(attenuated_fields)
        or len(intrinsic_fields) != len(temperature_fields)
    ):
        raise ValueError("Each line requires intrinsic, attenuated and temperature fields")
    emitting = np.zeros(len(intrinsic_fields), dtype=np.bool_)
    for index, fields in enumerate(zip(intrinsic_fields, attenuated_fields, temperature_fields)):
        if any(values.shape != (original_count,) for values in fields):
            raise ValueError("Per-line fields must match the original cell shape")
        intrinsic, attenuated, temperature = (values[rows] for values in fields)
        if not np.isfinite(temperature).all() or np.any(temperature < 0.0):
            raise ValueError("temperatures_K must be finite and nonnegative")
        for emissivity in (intrinsic, attenuated):
            if not np.isfinite(emissivity).all() or np.any(emissivity < 0.0):
                raise ValueError("emissivity must be finite and nonnegative")
            with np.errstate(all="ignore"):
                luminosity = emissivity * cell_volume_cm3
            if not np.isfinite(luminosity).all():
                raise ValueError("emissivity times volume must remain finite")
            emitting[index] |= np.any(luminosity != 0.0)
    return emitting


def selected_temperatures_equal(rows, first_temperature_K, second_temperature_K):
    """Compare temperatures only on this validated available branch."""
    return np.array_equal(first_temperature_K[rows], second_temperature_K[rows])


def prepare_direct_group_luminosity(
    rows, intrinsic_fields, attenuated_fields, shared_lines, cell_volume_cm3,
):
    """Own C-order [cell, 2*shared-line] intrinsic then attenuated luminosity."""
    output = np.empty((rows.size, 2 * shared_lines.size), dtype=np.float64)
    for dust_index, fields in enumerate((intrinsic_fields, attenuated_fields)):
        for column, line_index in enumerate(shared_lines):
            np.multiply(
                fields[line_index][rows],
                cell_volume_cm3,
                out=output[:, dust_index * shared_lines.size + column],
            )
    return output


def calculate_centered_full_moments(velocity_kms, thermal_width_kms, luminosity, minimum_width_kms):
    """Use the shared stable two-pass NumPy moments for each luminosity account.

    The positive-light anchor and mean offset remain separate in the second
    pass, including for a narrow distribution with a large common velocity.
    """
    if (
        not 1 <= luminosity.shape[1] <= 12
        or luminosity.shape[0] != velocity_kms.size
        or thermal_width_kms.shape != velocity_kms.shape
    ):
        raise ValueError("Require matching vectors and one to twelve profiles")
    widths = np.maximum(thermal_width_kms, minimum_width_kms)
    output = np.empty((luminosity.shape[1], 3), dtype=np.float64)
    for profile in range(luminosity.shape[1]):
        moment = LineVelocityMoments.from_cells(
            velocity_kms=velocity_kms,
            thermal_width_kms=widths,
            luminosity_erg_s=luminosity[:, profile],
        )
        output[profile] = (
            moment.luminosity_erg_s,
            moment.centroid_kms,
            moment.centered_second_moment,
        )
    return output


def prepare_available_group(
    velocity_kms, cold_cells, available_rows, intrinsic_fields, attenuated_fields,
    temperature_fields, cell_volume_cm3, speed_of_light_kms,
):
    """Retain [line,cell] Fortran-layout branches for the reference methods."""
    branches = prepare_branch_rows(
        available_rows=available_rows,
        cold_cells=cold_cells,
        velocity_kms=velocity_kms,
        speed_of_light_kms=speed_of_light_kms,
    )
    output = []
    for rows, velocity in branches:
        validate_branch_emission(
            rows=rows,
            intrinsic_fields=intrinsic_fields,
            attenuated_fields=attenuated_fields,
            temperature_fields=temperature_fields,
            cell_volume_cm3=cell_volume_cm3,
            original_count=velocity_kms.size,
        )
        matrices = []
        for fields in (temperature_fields, intrinsic_fields, attenuated_fields):
            matrix = np.empty((len(fields), rows.size), dtype=np.float64, order="F")
            for index, values in enumerate(fields):
                matrix[index] = values[rows]
            if fields is not temperature_fields:
                matrix *= cell_volume_cm3
            matrices.append(matrix)
        output.append((velocity, *matrices))
    return tuple(output)


def pack_width_group(intrinsic_luminosity, attenuated_luminosity, shared_lines, cell_major=False):
    """Own one [cell,account] matrix without two selected-line matrix copies."""
    if intrinsic_luminosity.shape != attenuated_luminosity.shape:
        raise ValueError("Both dust states must have matching branch shapes")
    if (
        np.any(shared_lines < 0)
        or np.any(shared_lines >= intrinsic_luminosity.shape[0])
    ):
        raise IndexError("A shared-width line index is outside the branch matrix")
    multiple_rows_and_lines = shared_lines.size > 1 and intrinsic_luminosity.shape[1] > 1
    order = "F" if not cell_major and multiple_rows_and_lines else "C"
    output = np.empty((intrinsic_luminosity.shape[1], 2 * shared_lines.size), order=order)
    for dust_index, values in enumerate((intrinsic_luminosity, attenuated_luminosity)):
        for column, line_index in enumerate(shared_lines):
            output[:, dust_index * shared_lines.size + column] = values[line_index]
    return output
