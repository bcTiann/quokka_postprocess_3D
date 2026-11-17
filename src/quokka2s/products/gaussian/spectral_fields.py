"""Select available spectral rows and own the final NumPy Gaussian queues.

Cell velocities and temperatures use km/s and K. Emissivities [erg/s/cm^3]
become luminosities [erg/s] only after applying the original cell volume.
Unavailable rows are never inspected or borrowed into another line's mask.
"""
from __future__ import annotations

import numpy as np

from ..line_velocity_moments import MINIMUM_GAUSSIAN_WIDTH_KMS
from ._spectral_kernels import (
    centered_full_moments,
    prepare_width_group_fields,
    selected_temperatures_match,
    validate_branch_fields,
)


DEFAULT_NUMPY_ERRORS = {
    "divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn",
}


def native_float64_vectors(values):
    """Identify plain aligned Float64 vectors without converting or copying."""
    return all(
        type(array) is np.ndarray
        and array.ndim == 1
        and array.dtype == np.dtype(np.float64)
        and array.flags.aligned
        for array in values
    )


def native_row_indices(rows):
    """Identify the int64 selected-row views supported by the compiled loops."""
    return (
        type(rows) is np.ndarray
        and rows.ndim == 1
        and rows.dtype == np.dtype(np.int64)
        and rows.flags.aligned
    )


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
    """Validate available branch values and flag nonzero cell luminosity.

    Temperature checks are shared only by identical arrays within this call.
    Multiplication by one positive volume is monotone for finite nonnegative
    epsilon. Its maximum product establishes both finite luminosities and
    whether any cell emits, including underflow. Full luminosity vectors are
    calculated later, directly in their final queues.
    """
    if not np.isfinite(cell_volume_cm3) or cell_volume_cm3 <= 0.0:
        raise ValueError("Require finite positive cell_volume_cm3")
    if (
        len(intrinsic_fields) != len(attenuated_fields)
        or len(intrinsic_fields) != len(temperature_fields)
    ):
        raise ValueError("Each line requires intrinsic, attenuated and temperature fields")
    if (
        np.geterr() == DEFAULT_NUMPY_ERRORS
        and native_row_indices(rows)
        and isinstance(cell_volume_cm3, (float, np.float64))
        and isinstance(original_count, (int, np.integer))
        and native_float64_vectors((*intrinsic_fields, *attenuated_fields, *temperature_fields))
    ):
        return validate_branch_fields(
            rows=rows,
            intrinsic_fields=intrinsic_fields,
            attenuated_fields=attenuated_fields,
            temperature_fields=temperature_fields,
            cell_volume_cm3=cell_volume_cm3,
            original_count=original_count,
        )
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
    if (
        np.geterr() == DEFAULT_NUMPY_ERRORS
        and native_row_indices(rows)
        and native_float64_vectors((first_temperature_K, second_temperature_K))
        and first_temperature_K.shape == second_temperature_K.shape
    ):
        return selected_temperatures_match(
            rows=rows,
            first=first_temperature_K,
            second=second_temperature_K,
        )
    return np.array_equal(first_temperature_K[rows], second_temperature_K[rows])


def prepare_direct_group_fields(
    rows, intrinsic_fields, attenuated_fields, shared_lines, temperature_K,
    velocity_kms, cell_volume_cm3, mass_g, boltzmann_erg_K,
    speed_of_light_kms, velocity_cm_per_km,
):
    """Prepare validated Float64 group's thermal width and final F-order light.

    Source epsilon/T vectors use original-cell order; rows and velocities use
    the selected branch order. All units/constants are runtime inputs. Return
    None for unsupported dtype/error policy or nonfinite arithmetic so the
    caller can retain its existing NumPy preparation and warning behavior.
    """
    if not (
        np.geterr() == DEFAULT_NUMPY_ERRORS
        and native_row_indices(rows)
        and native_row_indices(shared_lines)
        and native_float64_vectors((*intrinsic_fields, *attenuated_fields, temperature_K, velocity_kms))
        and all(
            values.shape == temperature_K.shape
            for values in (*intrinsic_fields, *attenuated_fields)
        )
        and all(
            isinstance(value, (float, np.float64))
            for value in (cell_volume_cm3, mass_g, boltzmann_erg_K, speed_of_light_kms, velocity_cm_per_km)
        )
    ):
        return None
    return prepare_width_group_fields(
        rows=rows,
        intrinsic_fields=intrinsic_fields,
        attenuated_fields=attenuated_fields,
        shared_lines=shared_lines,
        temperature_K=temperature_K,
        velocity_kms=velocity_kms,
        cell_volume_cm3=cell_volume_cm3,
        mass_g=mass_g,
        boltzmann_erg_K=boltzmann_erg_K,
        speed_of_light_kms=speed_of_light_kms,
        velocity_cm_per_km=velocity_cm_per_km,
    )


def prepare_direct_group_luminosity(
    rows, intrinsic_fields, attenuated_fields, shared_lines, cell_volume_cm3,
):
    """Own column-contiguous [cell, account] luminosities for sums and scatter."""
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
    """Use stable two-pass centered moments for each luminosity account.

    Float64 NumPy inputs under the default error policy use a compiled loop;
    other dtypes, array subclasses and custom error policies retain NumPy.
    Both paths use the first positive-light row as an anchor, preserving a
    narrow distribution at a large common velocity. No input array is modified.
    """
    if (
        not 1 <= luminosity.shape[1] <= 12
        or luminosity.shape[0] != velocity_kms.size
        or thermal_width_kms.shape != velocity_kms.shape
    ):
        raise ValueError("Require matching vectors and one to twelve profiles")
    if (
        all(
            type(values) is np.ndarray
            and values.dtype == np.dtype(np.float64)
            and values.flags.aligned
            for values in (velocity_kms, thermal_width_kms, luminosity)
        )
        and velocity_kms.ndim == 1
        and thermal_width_kms.ndim == 1
        and luminosity.ndim == 2
        and isinstance(minimum_width_kms, (float, np.float64))
        and np.geterr() == DEFAULT_NUMPY_ERRORS
    ):
        moments = centered_full_moments(
            velocity_kms=velocity_kms,
            thermal_width_kms=thermal_width_kms,
            luminosity=luminosity,
            minimum_width_kms=minimum_width_kms,
            absolute_minimum_width_kms=MINIMUM_GAUSSIAN_WIDTH_KMS,
        )
        if moments is not None:
            return moments
    # Consecutive NumPy accounts with identical positive masks share offsets
    # and thermal variance; no selection is cached between calls.
    widths = np.maximum(thermal_width_kms, minimum_width_kms)
    output = np.zeros((luminosity.shape[1], 3), dtype=np.float64)
    previous_cells = None
    previous_context = None
    for profile in range(luminosity.shape[1]):
        cell_luminosity = luminosity[:, profile]
        emitting_cells = cell_luminosity > 0.0
        if not np.any(emitting_cells):
            continue
        all_rows_emit = bool(np.all(emitting_cells))
        selected_luminosity = cell_luminosity if all_rows_emit else cell_luminosity[emitting_cells]
        if previous_cells is None or not np.array_equal(previous_cells, emitting_cells):
            velocity = velocity_kms if all_rows_emit else velocity_kms[emitting_cells]
            selected_width = widths if all_rows_emit else widths[emitting_cells]
            thermal_width = np.maximum(selected_width, MINIMUM_GAUSSIAN_WIDTH_KMS)
            previous_context = (
                velocity[0],
                velocity - velocity[0],
                thermal_width ** 2,
            )
            previous_cells = emitting_cells
        velocity_anchor, velocity_offset, thermal_variance = previous_context
        total_luminosity = float(np.sum(selected_luminosity, dtype=np.float64))
        mean_offset = float(np.sum(selected_luminosity * velocity_offset) / total_luminosity)
        centroid = float(velocity_anchor + mean_offset)
        bulk_variance = (velocity_offset - mean_offset) ** 2
        centered_second = float(np.sum(selected_luminosity * (bulk_variance + thermal_variance)))
        output[profile] = (total_luminosity, centroid, centered_second)
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
