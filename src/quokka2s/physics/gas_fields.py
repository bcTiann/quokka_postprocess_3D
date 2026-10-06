"""Derive hydrogen density, shielding column and velocity gradient from a slab."""

import numpy as np
from unyt import s, unyt_quantity

from ..constants import HYDROGEN_MASS_G
from . import settings
from .settings import SIMULATION_DVDR_MIN_S


# Keep yt/unyt units while deriving slab fields; convert to NumPy afterwards.
HYDROGEN_MASS = unyt_quantity(HYDROGEN_MASS_G, 'g')
VELOCITY_GRADIENT_FLOOR_S = SIMULATION_DVDR_MIN_S


def hydrogen_number_density(density):
    """Return hydrogen-nuclei density [cm^-3] from unit-aware gas density.

    density is a yt/unyt array from the slab grid, normally (10, 256, 2048)
    including two x-neighbour layers. Returns the same shape with units.
    Every chemical and ionization state contributes its hydrogen nuclei.
    """
    density_cgs = density.in_cgs()
    number_density = density_cgs * settings.X_H / HYDROGEN_MASS
    return number_density.to('cm**-3')


def shielding_hydrogen_column(grid):
    """Return the harmonic mean of the +/-z columns [cm^-2] for every cell.

    grid is a yt covering grid spanning all z cells. Each one-sided column
    includes the full emitting cell; this is separate from the dust column.
    Example: a (10, 256, 2048) grid returns the same shape with units.
    """
    hydrogen_density = hydrogen_number_density(grid['gas', 'density'])
    dz = grid['boxlib', 'dz'].in_cgs()
    cell_column = hydrogen_density * dz

    # Sum toward +z first, as in the original calculation. Reverse twice to
    # restore the cell ordering after accumulating from the high-z boundary.
    plus_column = np.flip(
        np.cumsum(np.flip(cell_column, axis=2), axis=2),
        axis=2,
    )
    inverse_column_sum = 1.0 / plus_column
    del plus_column

    minus_column = np.cumsum(cell_column, axis=2)
    inverse_column_sum = inverse_column_sum + 1.0 / minus_column
    return (2 / inverse_column_sum).to('cm**-2')


def velocity_gradient(grid, *, vx_left_cm_s=None, vx_right_cm_s=None):
    """Return abs(div(v))/3 [s^-1] using the slab's neighbouring cells.

    grid supplies three velocities and dx/dy/dz with units. Its y and z axes
    span the full box. x/y are periodic; z uses one-sided end differences.
    Optional vx_left_cm_s/vx_right_cm_s are opposite-face (Ny, Nz) planes from
    read_x_velocity_plane(). Example: x=0 uses the supplied x=255 left neighbour.
    Returns a same-shape unit-aware array; read_slab() then removes x halos.
    """
    vx = grid['gas', 'velocity_x'].in_units('cm/s').v
    vy = grid['gas', 'velocity_y'].in_units('cm/s').v
    vz = grid['gas', 'velocity_z'].in_units('cm/s').v
    dx = float(grid['boxlib', 'dx'].in_units('cm').v.flat[0])
    dy = float(grid['boxlib', 'dy'].in_units('cm').v.flat[0])
    dz = float(grid['boxlib', 'dz'].in_units('cm').v.flat[0])

    # x neighbours are in the loaded grid, except at a periodic box face.
    dvx_dx = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2.0 * dx)
    if vx_left_cm_s is not None:
        dvx_dx[0] = (vx[1] - vx_left_cm_s) / (2.0 * dx)
    if vx_right_cm_s is not None:
        dvx_dx[-1] = (vx_right_cm_s - vx[-2]) / (2.0 * dx)

    # The complete y axis is present, so array wrapping supplies its neighbours.
    dvy_dy = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2.0 * dy)

    # Only the z boundary cells use first-order one-sided differences.
    dvz_dz = np.gradient(vz, dz, axis=2, edge_order=1)
    divergence = dvx_dx + dvy_dy + dvz_dz
    gradient = np.maximum(np.abs(divergence) / 3.0, VELOCITY_GRADIENT_FLOOR_S)
    return gradient / s
