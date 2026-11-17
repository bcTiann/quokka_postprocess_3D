"""Select available spectral rows and own the final NumPy Gaussian queues.

Cell velocities and temperatures use km/s and K. Emissivities [erg/s/cm^3]
become luminosities [erg/s] only after applying the original cell volume.
Unavailable rows are never inspected or borrowed into another line's mask.
"""
from __future__ import annotations

import numpy as np

from ..line_velocity_moments import MINIMUM_GAUSSIAN_WIDTH_KMS


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
    checked_temperatures = set()
    for index, fields in enumerate(zip(intrinsic_fields, attenuated_fields, temperature_fields)):
        if any(values.shape != (original_count,) for values in fields):
            raise ValueError("Per-line fields must match the original cell shape")
        intrinsic = fields[0][rows]
        attenuated = fields[1][rows]
        temperature_identity = id(fields[2])
        if temperature_identity not in checked_temperatures:
            temperature = fields[2][rows]
            if not np.isfinite(temperature).all() or np.any(temperature < 0.0):
                raise ValueError("temperatures_K must be finite and nonnegative")
            checked_temperatures.add(temperature_identity)
        for emissivity in (intrinsic, attenuated):
            if not np.isfinite(emissivity).all() or np.any(emissivity < 0.0):
                raise ValueError("emissivity must be finite and nonnegative")
            # Positive volume preserves order, so the largest product proves
            # both finiteness and whether any cell emits after underflow.
            with np.errstate(all="ignore"):
                maximum_luminosity = np.max(emissivity, initial=0.0) * cell_volume_cm3
            if not np.isfinite(maximum_luminosity):
                raise ValueError("emissivity times volume must remain finite")
            emitting[index] |= maximum_luminosity != 0.0
    return emitting


def selected_temperatures_equal(rows, first_temperature_K, second_temperature_K):
    """Compare temperatures only on this validated available branch."""
    if first_temperature_K is second_temperature_K:
        return True
    return np.array_equal(first_temperature_K[rows], second_temperature_K[rows])


def prepare_direct_group_luminosity(
    rows, intrinsic_fields, attenuated_fields, shared_lines, cell_volume_cm3,
):
    """Own column-contiguous [cell, 2*shared-line] luminosities for profile reductions."""
    output = np.empty((rows.size, 2 * shared_lines.size), dtype=np.float64, order="F")
    for dust_index, fields in enumerate((intrinsic_fields, attenuated_fields)):
        for column, line_index in enumerate(shared_lines):
            np.multiply(
                fields[line_index][rows],
                cell_volume_cm3,
                out=output[:, dust_index * shared_lines.size + column],
            )
    return output


def calculate_centered_full_moments(velocity_kms, thermal_width_kms, luminosity, minimum_width_kms):
    """Reuse shared velocity offsets and widths in stable two-pass moments.

    Every profile retains its own positive-light mask and first emitting-cell
    anchor. Profiles emitting in all rows share only identical coordinate work;
    masked profiles keep their original selected coordinates and sum order.
    """
    if (
        not 1 <= luminosity.shape[1] <= 12
        or luminosity.shape[0] != velocity_kms.size
        or thermal_width_kms.shape != velocity_kms.shape
    ):
        raise ValueError("Require matching vectors and one to twelve profiles")
    widths = np.maximum(thermal_width_kms, max(minimum_width_kms, MINIMUM_GAUSSIAN_WIDTH_KMS))
    output = np.zeros((luminosity.shape[1], 3), dtype=np.float64)
    all_row_context = None
    for profile in range(luminosity.shape[1]):
        light = luminosity[:, profile]
        emitting = light > 0.0
        if not np.any(emitting):
            continue
        if np.all(emitting):
            if all_row_context is None:
                anchor = velocity_kms[0]
                offsets = velocity_kms - anchor
                thermal_variance = widths ** 2
                all_row_context = (anchor, offsets, thermal_variance)
            anchor, offsets, thermal_variance = all_row_context
        else:
            light = light[emitting]
            selected_velocity = velocity_kms[emitting]
            anchor = selected_velocity[0]
            offsets = selected_velocity - anchor
            thermal_variance = widths[emitting] ** 2
        total = float(np.sum(light, dtype=np.float64))
        mean_offset = float(np.sum(light * offsets) / total)
        centroid = float(anchor + mean_offset)
        bulk_variance = (offsets - mean_offset) ** 2
        centered_second = float(np.sum(light * (bulk_variance + thermal_variance)))
        output[profile] = (total, centroid, centered_second)
    return output


def prepare_available_group(
    velocity_kms, cold_cells, available_rows, intrinsic_fields, attenuated_fields,
    temperature_fields, cell_volume_cm3, speed_of_light_kms,
):
    """Retain Fortran-layout branch preparation under custom NumPy error policies."""
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


def pack_width_group(intrinsic_luminosity, attenuated_luminosity, shared_lines):
    """Own one [cell,account] matrix without two selected-line matrix copies."""
    if intrinsic_luminosity.shape != attenuated_luminosity.shape:
        raise ValueError("Both dust states must have matching branch shapes")
    if (
        np.any(shared_lines < 0)
        or np.any(shared_lines >= intrinsic_luminosity.shape[0])
    ):
        raise IndexError("A shared-width line index is outside the branch matrix")
    output = np.empty((intrinsic_luminosity.shape[1], 2 * shared_lines.size), order="F")
    for dust_index, values in enumerate((intrinsic_luminosity, attenuated_luminosity)):
        for column, line_index in enumerate(shared_lines):
            output[:, dust_index * shared_lines.size + column] = values[line_index]
    return output
