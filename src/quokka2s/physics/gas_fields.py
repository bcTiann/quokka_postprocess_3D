"""Derive hydrogen density, shielding column and velocity gradient from a slab."""

import numpy as np
from unyt import s
from yt.utilities.physical_constants import mh

from . import settings
from .settings import SIMULATION_DVDR_MIN_S


# Mean hydrogen atomic mass supplied by yt [g], used in nH = X_H * rho / m_H.
HYDROGEN_MASS_G = mh.to_value("g")


def hydrogen_number_density_cm3(*, density_g_cm3: np.ndarray) -> np.ndarray:
    """Convert physical gas density [g/cm^3] to hydrogen nuclei [cm^-3].

    Input and output have the same shape, e.g. (B,) for a cell batch or
    (Nx, Ny, Nz) for a slab. This does not clip density for table queries.
    Keep multiplication before division to preserve the adopted arithmetic.
    """
    return density_g_cm3 * settings.X_H / HYDROGEN_MASS_G


def mixed_gas_temperature_K(
    *,
    temperature_quokka_K: np.ndarray,
    temperature_despotic_K: np.ndarray,
    cold_cells: np.ndarray,
) -> np.ndarray:
    """Choose the temperature used for gas-phase mass and velocity products.

    All inputs have shape (B,), in original batch order. cold_cells comes from
    the QUOKKA branch: cold cells use DESPOTIC; hot cells use QUOKKA.
    Returns a (B,) temperature array [K], preserving missing cold values as NaN.
    Example: T_Q=[100, 1e6], T_D=[NaN, NaN] gives [NaN, 1e6].
    """
    return np.where(cold_cells, temperature_despotic_K, temperature_quokka_K)


def hydrogen_number_density(density):
    """Return hydrogen-nuclei density [cm^-3] from unit-aware gas density.

    density is a yt/unyt array from the loaded fields, normally (10, 258, 2048)
    including x/y neighbour layers. Returns the same shape with units.
    Every chemical and ionization state contributes its hydrogen nuclei.
    """
    # Convert before stripping units so invalid density dimensions still raise.
    density_cgs = density.to('g/cm**3')
    # unyt promotes the scalar multiplication to at least float64.
    density_values = density_cgs.v.astype(
        np.result_type(density_cgs.dtype, np.float64),
        copy=False,
    )
    number_density_cm3 = hydrogen_number_density_cm3(
        density_g_cm3=density_values,
    )
    # Reattach cm^-3 using the input's unyt type and unit registry.
    return type(density_cgs)(
        number_density_cm3,
        units='cm**-3',
        registry=density_cgs.units.registry,
    )


def shielding_hydrogen_column(fields):
    """Return the harmonic mean of the +/-z columns [cm^-2] for every cell.

    fields supplies unit-aware density and dz spanning all z cells. Each column
    includes the full emitting cell; this is separate from the dust column.
    Example: (10, 258, 2048) density returns the same shape with units.
    """
    hydrogen_density = hydrogen_number_density(fields['gas', 'density'])
    dz = fields['boxlib', 'dz'].in_cgs()
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


def velocity_gradient(fields):
    """Return abs(div(v))/3 [s^-1] using the slab's neighbouring cells.

    fields supplies three velocities and dx/dy/dz with units. x/y include
    wrapped neighbour layers from NativeSnapshotReader; z spans the full box.
    For example, core x=0 is between loaded x=Nx-1 and x=1. Only the core
    gradients are retained by read_slab(); outer neighbour values are discarded.
    Returns a same-shape unit-aware array; z uses one-sided end differences.
    """
    vx = fields['gas', 'velocity_x'].in_units('cm/s').v
    vy = fields['gas', 'velocity_y'].in_units('cm/s').v
    vz = fields['gas', 'velocity_z'].in_units('cm/s').v
    dx = float(fields['boxlib', 'dx'].in_units('cm').v.flat[0])
    dy = float(fields['boxlib', 'dy'].in_units('cm').v.flat[0])
    dz = float(fields['boxlib', 'dz'].in_units('cm').v.flat[0])

    # Both x/y neighbours are present, including across periodic box faces.
    dvx_dx = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2.0 * dx)
    dvy_dy = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2.0 * dy)

    # Only the z boundary cells use first-order one-sided differences.
    dvz_dz = np.gradient(vz, dz, axis=2, edge_order=1)
    divergence = dvx_dx + dvy_dy + dvz_dz
    gradient = np.maximum(np.abs(divergence) / 3.0, SIMULATION_DVDR_MIN_S)
    return gradient / s
