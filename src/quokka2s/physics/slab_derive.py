"""Derive full-z physical columns with bounded NumPy working arrays.

Inputs use native snapshot units. Keep each unit conversion as a separate
operation, global XY neighbours and independent forward/reverse z sums.
Each caller task writes disjoint complete columns; no field cache is created.
"""
from __future__ import annotations

import numpy as np


# Keep temporary neighbour/velocity arrays bounded even for a large RAM tile.
MAX_BLOCK_CELLS = 131072
# Reserve twelve float64 values per active cell for neighbour gathers,
# velocities, gradient/cumsum temporaries and small index/mask arrays. This
# is a conservative planning allowance, not measured allocator/RSS traffic.
TEMPORARY_BYTES_PER_CELL = 12 * 8


def estimated_working_bytes(*, column_count, column_depth, threads):
    """Bound concurrent NumPy blocks for preflight and derivation receipts."""
    if min(column_count, column_depth, threads) < 1:
        raise ValueError("NumPy column working estimates require positive geometry")
    task_count = min(threads, column_count)
    columns_per_task = (column_count + task_count - 1) // task_count
    columns_per_block = min(
        columns_per_task,
        max(1, MAX_BLOCK_CELLS // column_depth),
    )
    return task_count * columns_per_block * column_depth * TEMPORARY_BYTES_PER_CELL


def _column_slices(first_column, stop_column, depth):
    columns_per_block = max(1, MAX_BLOCK_CELLS // depth)
    for first in range(first_column, stop_column, columns_per_block):
        yield slice(first, min(first + columns_per_block, stop_column))


def _gas_velocity(momentum, density, factors):
    """Apply quotient, native-unit coefficient and gas conversion in order."""
    velocity = np.divide(momentum, density)
    velocity *= factors[0]
    velocity *= factors[1]
    return velocity


def _velocity_gradient(neighbour_velocity, velocity_z_cgs, widths_cm, floor_s):
    """Return abs(div(v))/3, using central XY and first-order z end faces."""
    gradient = neighbour_velocity(0, 1)
    gradient -= neighbour_velocity(0, -1)
    gradient /= 2.0 * widths_cm[0]
    y_gradient = neighbour_velocity(1, 1)
    y_gradient -= neighbour_velocity(1, -1)
    y_gradient /= 2.0 * widths_cm[1]
    gradient += y_gradient
    gradient += np.gradient(
        velocity_z_cgs,
        widths_cm[2],
        axis=-1,
        edge_order=1,
    )
    np.abs(gradient, out=gradient)
    gradient /= 3.0
    np.maximum(gradient, floor_s, out=gradient)
    return gradient


def _fill_column_densities(
    *, density_cgs, foreground, shielding, hydrogen_fraction, hydrogen_mass_g,
    dz_foreground_cm, dz_shielding_cm, shielding_to_cgs,
):
    """Write foreground and harmonic shielding NH for complete z rows.

    The foreground includes half the emitting cell. Shielding uses two
    independent inclusive cumsums; it never subtracts a prefix from a total.
    Return the count of nonfinite or negative nH values, including cold gas.
    """
    hydrogen_density = density_cgs * hydrogen_fraction
    hydrogen_density /= hydrogen_mass_g
    invalid_count = int(np.count_nonzero(
        ~np.isfinite(hydrogen_density) | (hydrogen_density < 0.0)
    ))
    # The previous scalar prefix started at +0. Adding that identity also
    # preserves its sign for a physical -0 density before NumPy's cumsum.
    np.add(0.0, hydrogen_density, out=hydrogen_density)
    np.cumsum(hydrogen_density, axis=-1, dtype=np.float64, out=foreground)
    foreground -= 0.5 * hydrogen_density
    foreground *= dz_foreground_cm
    hydrogen_density *= dz_shielding_cm
    np.cumsum(
        hydrogen_density[:, ::-1],
        axis=-1,
        dtype=np.float64,
        out=shielding[:, ::-1],
    )
    np.reciprocal(shielding, out=shielding)
    np.cumsum(hydrogen_density, axis=-1, dtype=np.float64, out=hydrogen_density)
    np.reciprocal(hydrogen_density, out=hydrogen_density)
    shielding += hydrogen_density
    np.divide(2.0, shielding, out=shielding)
    shielding *= shielding_to_cgs
    return invalid_count


def derive_tile_velocity_gradient(
    *, density_raw, momentum_x, momentum_y, momentum_z,
    x_density_faces, x_momentum_faces, y_density_faces, y_momentum_faces,
    gradient_output, velocity_factors, dx_cm, dy_cm, dz_gradient_cm,
    gradient_floor_s, first_column, stop_column,
):
    """Use global raw neighbour faces before any tile field is overwritten.

    Core arrays have shape(nx,ny,nz); X faces(2,ny,nz) and Y faces(nx,2,nz)
    contain the original global neighbours, including periodic box boundaries.
    A tile edge is never substituted for a global periodic box edge.
    """
    nx, ny, nz = density_raw.shape
    output_rows = gradient_output.reshape(-1, nz)
    with np.errstate(all="ignore"):
        for selected in _column_slices(first_column, stop_column, nz):
            columns = np.arange(selected.start, selected.stop, dtype=np.int64)
            x = columns // ny
            y = columns % ny

            def neighbour_velocity(axis, side):
                if axis == 0:
                    neighbour_x = x + side
                    outside = (neighbour_x < 0) | (neighbour_x >= nx)
                    np.clip(neighbour_x, 0, nx - 1, out=neighbour_x)
                    momentum = momentum_x[neighbour_x, y]
                    density = density_raw[neighbour_x, y]
                    face = 0 if side == -1 else 1
                    momentum[outside] = x_momentum_faces[face, y[outside]]
                    density[outside] = x_density_faces[face, y[outside]]
                else:
                    neighbour_y = y + side
                    outside = (neighbour_y < 0) | (neighbour_y >= ny)
                    np.clip(neighbour_y, 0, ny - 1, out=neighbour_y)
                    momentum = momentum_y[x, neighbour_y]
                    density = density_raw[x, neighbour_y]
                    face = 0 if side == -1 else 1
                    momentum[outside] = y_momentum_faces[x[outside], face]
                    density[outside] = y_density_faces[x[outside], face]
                velocity = _gas_velocity(momentum, density, velocity_factors[axis])
                velocity *= velocity_factors[axis, 2]
                return velocity

            velocity_z = _gas_velocity(
                momentum_z[x, y], density_raw[x, y], velocity_factors[2]
            )
            velocity_z *= velocity_factors[2, 2]
            gradient = _velocity_gradient(
                neighbour_velocity,
                velocity_z,
                (dx_cm, dy_cm, dz_gradient_cm),
                gradient_floor_s,
            )
            output_rows[selected] = gradient
            del gradient, velocity_z


def convert_tile_columns_inplace(
    *, density, momentum_z, foreground_output, shielding_output,
    velocity_factors, density_to_gas, gas_density_to_cgs,
    hydrogen_fraction, hydrogen_mass_g, dz_foreground_cm, dz_shielding_cm,
    shielding_to_cgs, first_column, stop_column,
):
    """Reuse raw density/Mz storage for cgs density and velocity[km/s].

    Call only after every gradient task has completed its neighbour reads.
    Outputs and task ranges describe disjoint complete full-z columns.
    """
    depth = density.shape[2]
    density_rows = density.reshape(-1, depth)
    velocity_rows = momentum_z.reshape(-1, depth)
    foreground_rows = foreground_output.reshape(-1, depth)
    shielding_rows = shielding_output.reshape(-1, depth)
    invalid_count = 0
    with np.errstate(all="ignore"):
        for selected in _column_slices(first_column, stop_column, depth):
            physical_density = density_rows[selected]
            velocity = velocity_rows[selected]
            np.divide(velocity, physical_density, out=velocity)
            velocity *= velocity_factors[2, 0]
            velocity *= velocity_factors[2, 1]
            velocity *= velocity_factors[2, 3]
            physical_density *= density_to_gas
            physical_density *= gas_density_to_cgs
            invalid_count += _fill_column_densities(
                density_cgs=physical_density,
                foreground=foreground_rows[selected],
                shielding=shielding_rows[selected],
                hydrogen_fraction=hydrogen_fraction,
                hydrogen_mass_g=hydrogen_mass_g,
                dz_foreground_cm=dz_foreground_cm,
                dz_shielding_cm=dz_shielding_cm,
                shielding_to_cgs=shielding_to_cgs,
            )
    return invalid_count


def derive_bounded_columns(
    *, density_raw, temperature_raw, momentum_x, momentum_y, momentum_z,
    output, velocity_factors, density_to_gas, gas_density_to_cgs,
    hydrogen_fraction, hydrogen_mass_g, dx_cm, dy_cm, dz_gradient_cm,
    dz_foreground_cm, dz_shielding_cm, shielding_to_cgs, gradient_floor_s,
    x_start, y_start, first_column, stop_column,
):
    """Fill six selected output fields from a read-only slab with XY halos.

    The bounded reader supplies an x halo on both sides and full native y.
    All z cells remain present; output has shape(6,selected_nx,selected_ny,nz).
    """
    nx, ny, nz = density_raw.shape
    selected_ny = output.shape[2]
    rows = output.reshape(6, -1, nz)
    invalid_count = 0
    with np.errstate(all="ignore"):
        for selected in _column_slices(first_column, stop_column, nz):
            columns = np.arange(selected.start, selected.stop, dtype=np.int64)
            x = x_start + columns // selected_ny
            y = y_start + columns % selected_ny

            def neighbour_velocity(axis, side):
                if axis == 0:
                    neighbour_x = (x + side) % nx
                    velocity = _gas_velocity(
                        momentum_x[neighbour_x, y], density_raw[neighbour_x, y],
                        velocity_factors[0],
                    )
                else:
                    neighbour_y = (y + side) % ny
                    velocity = _gas_velocity(
                        momentum_y[x, neighbour_y], density_raw[x, neighbour_y],
                        velocity_factors[1],
                    )
                velocity *= velocity_factors[axis, 2]
                return velocity

            raw_density = density_raw[x, y]
            rows[0, selected] = raw_density
            rows[0, selected] *= density_to_gas
            rows[0, selected] *= gas_density_to_cgs
            rows[2, selected] = temperature_raw[x, y]
            velocity_z = _gas_velocity(momentum_z[x, y], raw_density, velocity_factors[2])
            rows[5, selected] = velocity_z * velocity_factors[2, 3]
            velocity_z *= velocity_factors[2, 2]
            rows[4, selected] = _velocity_gradient(
                neighbour_velocity,
                velocity_z,
                (dx_cm, dy_cm, dz_gradient_cm),
                gradient_floor_s,
            )
            invalid_count += _fill_column_densities(
                density_cgs=rows[0, selected],
                foreground=rows[1, selected],
                shielding=rows[3, selected],
                hydrogen_fraction=hydrogen_fraction,
                hydrogen_mass_g=hydrogen_mass_g,
                dz_foreground_cm=dz_foreground_cm,
                dz_shielding_cm=dz_shielding_cm,
                shielding_to_cgs=shielding_to_cgs,
            )
            del raw_density, velocity_z
    return invalid_count
