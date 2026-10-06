"""Read native snapshot slabs and expose unit-converted cell batches."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .constants import HYDROGEN_MASS_G
from .physics.dust_attenuation import observer_side_hydrogen_column
from .physics.settings import X_H


@dataclass(frozen=True)
class Snapshot:
    """An opened yt snapshot, its grid geometry, and the action of reading a slab.

    Parameters
    ----------
    dataset : yt dataset
        Returned by yt.load() in processing_inputs.open_snapshot(). Opening it reads
        snapshot metadata; cell fields are loaded later by read_slab().

    Geometry properties are calculated from this dataset, so callers do not
    repeat its shape, cell widths, or volume in the constructor.
    Example: snapshot.read_slab(x_start=0, x_stop=8) returns the first eight
    x layers with derived columns and velocity gradients already calculated.
    """

    dataset: object

    @cached_property
    def shape(self) -> tuple[int, int, int]:
        """Native cell counts (Nx, Ny, Nz), e.g. (256, 256, 2048)."""
        return tuple(int(n) for n in self.dataset.domain_dimensions)

    @cached_property
    def cell_widths(self):
        """Cell lengths (dx, dy, dz), retaining the dataset's yt/unyt units."""
        return self.dataset.domain_width / self.dataset.domain_dimensions

    @cached_property
    def cell_volume_cm3(self) -> float:
        """Volume dx * dy * dz of one original cell [cm^3]."""
        return float(np.prod(self.cell_widths.to('cm').value))

    @cached_property
    def projected_area_cm2(self) -> float:
        """Full x-y image area, domain_width_x * domain_width_y [cm^2]."""
        area = self.dataset.domain_width[0] * self.dataset.domain_width[1]
        return float(area.to('cm**2').value)

    @cached_property
    def cell_count(self) -> int:
        """Total native cells, e.g. 256 * 256 * 2048 = 134217728."""
        return int(np.prod(self.shape))

    def read_slab(self, x_start: int, x_stop: int) -> SlabArrays:
        """Return processed fields for the requested x layers and the full y-z plane.

        Parameters
        ----------
        x_start, x_stop : int
            Global x bounds, with x_stop excluded. For example, 0:8 selects the
            first eight x layers. Neighbour reads are handled inside this function.

        Returns
        -------
        SlabArrays
            Six read-only arrays of shape (slab_nx * Ny * Nz,), flattened in
            C order. Units are specified by SlabArrays. Foreground NH comes from
            observer_side_hydrogen_column(); shielding NH and dVdr come from
            physics.gas_fields.shielding_hydrogen_column() and velocity_gradient().

        Examples
        --------
        With snapshot from load_processing_inputs()::

            slab = snapshot.read_slab(x_start=0, x_stop=8)
        """
        # Keep neighbour geometry internal: first core 0:8 reads 0:9, while an
        # interior core 8:16 reads 7:17. Periodic neighbours across the box faces
        # are read by slab_velocity_gradient(). Only core cells are returned.
        read_start = max(0, x_start - 1)
        read_stop = min(self.shape[0], x_stop + 1)
        core = slice(x_start - read_start, x_stop - read_start)

        # 1. Open the slab's yt grid, including neighbouring x layers.
        grid = read_slab_grid(
            snapshot=self,
            read_start=read_start,
            read_stop=read_stop,
        )

        # 2. Read the simulation fields and retain only the core cells.
        # Each returned array has shape (8, 256, 2048) for a normal slab.
        # Units: density [g/cm^3], temperature [K], velocity_z [km/s].
        density, temperature, velocity_z = read_simulation_fields(
            grid=grid,
            core=core,
        )

        # 3. Calculate the column to the observer at the outer -z face.
        foreground_column = calculate_dust_foreground_column(
            snapshot=self,
            density_g_cm3=density,
        )

        # 4. Calculate the harmonic mean of the +z and -z columns.
        shielding_column = calculate_shielding_column(
            grid=grid,
            core=core,
        )

        # 5. Calculate dV/dr using neighbouring cells, then retain core.
        # This includes neighbours across the periodic x/y boundaries.
        velocity_gradient = calculate_core_velocity_gradient(
            snapshot=self,
            grid=grid,
            x_start=x_start,
            x_stop=x_stop,
            core=core,
        )

        # The returned NumPy arrays own their data; release the yt grid.
        del grid

        # 6. Flatten each field: (8, 256, 2048) -> (4194304,).
        # z varies fastest, followed by y, then x.
        slab = SlabArrays(
            density_g_cm3=density.ravel(),
            foreground_NH_cm2=foreground_column.ravel(),
            temperature_QUOKKA_K=temperature.ravel(),
            shielding_NH_cm2=shielding_column.ravel(),
            velocity_gradient_s=velocity_gradient.ravel(),
            velocity_z_kms=velocity_z.ravel(),
            x_start=x_start,
            shape=(x_stop - x_start, self.shape[1], self.shape[2]),
            cell_volume_cm3=self.cell_volume_cm3,
        )

        # Later batches may read these arrays but must not modify them.
        for array in (
            slab.density_g_cm3,
            slab.foreground_NH_cm2,
            slab.temperature_QUOKKA_K,
            slab.shielding_NH_cm2,
            slab.velocity_gradient_s,
            slab.velocity_z_kms,
        ):
            array.flags.writeable = False

        return slab


@dataclass(frozen=True)
class CellBatch:
    """Views of consecutive cells, with their location in the native grid.

    Parameters
    ----------
    density_g_cm3, foreground_NH_cm2, temperature_QUOKKA_K : ndarray, shape (B,)
        Views from SlabArrays.batch(): density [g/cm^3], observer-side
        hydrogen column [H nuclei/cm^2], and simulation temperature [K].
    shielding_NH_cm2, velocity_gradient_s, velocity_z_kms : ndarray, shape (B,)
        Shielding column [H nuclei/cm^2], LVG gradient [s^-1], and vz [km/s].
    x_start : int
        First x index of the parent slab in the full grid.
    slab_shape : tuple of three int
        Parent core-slab dimensions (slab_nx, Ny, Nz).
    batch_start : int
        First cell index within the C-order flattened slab; z varies fastest.
    cell_volume_cm3 : float
        Native volume of each cell [cm^3].

    first_cell_id and last_cell_id are derived properties: the inclusive
    cell IDs in the C-order flattened full grid.

    The six input arrays are views, with no averaging or copying. They retain
    their parent arrays; batches from read_slab() are read-only.
    hydrogen_density_cm3 is derived from density on first access and then reused.
    Its new (B,) array belongs to this batch and is released with the batch.
    """
    density_g_cm3: np.ndarray
    foreground_NH_cm2: np.ndarray
    temperature_QUOKKA_K: np.ndarray
    shielding_NH_cm2: np.ndarray
    velocity_gradient_s: np.ndarray
    velocity_z_kms: np.ndarray
    x_start: int
    slab_shape: tuple[int, int, int]
    batch_start: int
    cell_volume_cm3: float

    @cached_property
    def hydrogen_density_cm3(self) -> np.ndarray:
        """Physical nH [cm^-3], computed once from this batch's density.

        density_g_cm3 is the original (B,) rho array [g/cm^3]. The result is
        rho * X_H / HYDROGEN_MASS_G, before any lookup-coordinate clipping.
        Repeated reads return the same read-only array, so both readers and
        emissivity calculations use one nH value per original cell.
        Example: cells.hydrogen_density_cm3[7] is the eighth cell's physical nH.
        """
        hydrogen_density_cm3 = self.density_g_cm3 * X_H / HYDROGEN_MASS_G
        hydrogen_density_cm3.flags.writeable = False
        return hydrogen_density_cm3

    @property
    def cell_count(self) -> int:
        """Return B, the number of cells in this batch, as an int."""
        return self.density_g_cm3.size

    @property
    def first_cell_id(self) -> int:
        """Full-grid ID of the first cell; z varies fastest, then y, then x."""
        cells_per_x_layer = self.slab_shape[1] * self.slab_shape[2]
        return int(self.x_start * cells_per_x_layer + self.batch_start)

    @property
    def last_cell_id(self) -> int:
        """Inclusive full-grid ID of the last cell in this batch."""
        return self.first_cell_id + self.cell_count - 1


@dataclass(frozen=True)
class SlabArrays:
    """Six flattened core-slab arrays, retained until its batches finish.

    Parameters
    ----------
    density_g_cm3, foreground_NH_cm2, temperature_QUOKKA_K : ndarray, shape (S,)
        Density [g/cm^3], observer-side column [H nuclei/cm^2], and QUOKKA
        temperature [K], returned by read_slab().
    shielding_NH_cm2, velocity_gradient_s, velocity_z_kms : ndarray, shape (S,)
        Shielding column [H nuclei/cm^2], LVG gradient [s^-1], and vz [km/s].

    x_start : int
        First global x layer, e.g. 8 for the slab covering x=8:16.
    shape : tuple of three int
        Retained slab dimensions, e.g. (8, 256, 2048), without neighbour layers.
    cell_volume_cm3 : float
        Volume of one original cell [cm^3], copied from Snapshot.

    S = slab_nx * Ny * Nz. Arrays use C order, with z varying fastest.
    read_slab() removes the x halos and marks these arrays read-only.
    batch() returns views, so they remain alive while any batch uses them.
    """
    density_g_cm3: np.ndarray
    foreground_NH_cm2: np.ndarray
    temperature_QUOKKA_K: np.ndarray
    shielding_NH_cm2: np.ndarray
    velocity_gradient_s: np.ndarray
    velocity_z_kms: np.ndarray
    x_start: int
    shape: tuple[int, int, int]
    cell_volume_cm3: float

    @property
    def cell_count(self) -> int:
        """Return S, the number of core-slab cells, as an int."""
        return self.density_g_cm3.size

    @property
    def first_cell_id(self) -> int:
        """Full-grid ID of this slab's first cell, for batch-failure reports."""
        cells_per_x_layer = self.shape[1] * self.shape[2]
        return int(self.x_start * cells_per_x_layer)

    def batch(self, start: int, stop: int) -> CellBatch:
        """Select views and preserve each cell's original grid location.

        Parameters
        ----------
        start, stop : int
            Half-open cell bounds within this flattened slab.

        Returns
        -------
        CellBatch
            Six views of shape (stop - start,), in unchanged C order, with
            their slab location and inclusive full-grid cell IDs.

        Examples
        --------
        With a slab returned by snapshot.read_slab()::

            cells = slab.batch(start=0, stop=100)
        """
        selected = slice(start, stop)
        return CellBatch(
            density_g_cm3=self.density_g_cm3[selected],
            foreground_NH_cm2=self.foreground_NH_cm2[selected],
            temperature_QUOKKA_K=self.temperature_QUOKKA_K[selected],
            shielding_NH_cm2=self.shielding_NH_cm2[selected],
            velocity_gradient_s=self.velocity_gradient_s[selected],
            velocity_z_kms=self.velocity_z_kms[selected],
            x_start=self.x_start,
            slab_shape=self.shape,
            batch_start=start,
            cell_volume_cm3=self.cell_volume_cm3,
        )


def slab_windows(nx, slab_nx):
    """Yield the x layers to process in each slab.

    Parameters
    ----------
    nx : int
        Full x dimension, normally snapshot.shape[0]; at least two cells.
    slab_nx : int
        Maximum number of core x cells in each slab.

    Yields
    ------
    tuple of (int, int)
        (x_start, x_stop), with x_stop excluded. read_slab() determines which
        neighbours to read; callers specify only the cells they want to process.

    Examples
    --------
    >>> list(slab_windows(5, 3))
    [(0, 3), (3, 5)]
    """
    if nx < 2 or slab_nx < 1:
        raise ValueError('x dimension must be at least two and slab size positive')
    for x_start in range(0, nx, slab_nx):
        x_stop = min(x_start + slab_nx, nx)
        yield x_start, x_stop


def read_x_velocity_plane(snapshot: Snapshot, x_index: int) -> np.ndarray:
    """Read one full y-z plane of velocity_x for a periodic x neighbour.

    snapshot comes from load_processing_inputs(); x_index is a global x index.
    Returns a copied NumPy array of shape (Ny, Nz) in cm/s.
    Example: read_x_velocity_plane(snapshot, x_index=255) is x=0's left neighbour.
    """
    ds = snapshot.dataset
    # yt treats a one-cell covering-grid axis as spanning the entire domain.
    # Read two adjacent layers instead, then keep only the requested plane.
    first_x = min(x_index, snapshot.shape[0] - 2)
    edge = ds.domain_left_edge.copy()
    edge[0] += first_x * snapshot.cell_widths[0]
    plane_grid = ds.covering_grid(
        level=0,
        left_edge=edge,
        dims=(2, snapshot.shape[1], snapshot.shape[2]),
    )
    velocity_with_units = plane_grid['gas', 'velocity_x'].to('cm/s')
    plane = np.array(velocity_with_units[x_index - first_x])
    del velocity_with_units, plane_grid
    return plane


def slab_velocity_gradient(snapshot: Snapshot, grid, x_start: int, x_stop: int):
    """Calculate a slab's gradient using neighbours across periodic x/y faces.

    grid is the yt covering grid including x halos; x_start/x_stop are the
    half-open core bounds from slab_windows(), e.g. (0, 8) for the first slab.
    Returns the full loaded-grid gradient with s^-1 units. The caller then
    selects [core] to discard the halo layers.
    """
    vx_left_cm_s = None
    vx_right_cm_s = None
    if x_start == 0:
        vx_left_cm_s = read_x_velocity_plane(snapshot, x_index=snapshot.shape[0] - 1)
    if x_stop == snapshot.shape[0]:
        vx_right_cm_s = read_x_velocity_plane(snapshot, x_index=0)

    from .physics.gas_fields import velocity_gradient

    return velocity_gradient(
        grid,
        vx_left_cm_s=vx_left_cm_s,
        vx_right_cm_s=vx_right_cm_s,
    )


def read_slab_grid(snapshot: Snapshot, read_start: int, read_stop: int):
    """Open a yt covering grid with the slab's neighbouring x layers.

    snapshot comes from load_processing_inputs(). read_start/read_stop are
    half-open global x bounds computed inside read_slab(); y/z span the full box.
    Returns a yt covering grid retaining its field units.
    Example: bounds (0, 9) give shape (9, 256, 2048) for the first slab.
    """
    ds = snapshot.dataset          # The yt dataset opened by yt.load().
    shape = snapshot.shape         # Current full grid: (256, 256, 2048).
    widths = snapshot.cell_widths  # (dx, dy, dz), with length units.

    # Move the box's lower corner to the slab's read boundary.
    edge = ds.domain_left_edge.copy()
    edge[0] += read_start * widths[0]
    # First slab: (9, 256, 2048); interior slabs: (10, 256, 2048).
    read_shape = (read_stop - read_start, shape[1], shape[2])
    grid = ds.covering_grid(
        level=0,          # Root-grid resolution; native for this uniform snapshot.
        left_edge=edge,   # Physical starting corner, with length units.
        dims=read_shape,  # Number of cells to read along (x, y, z).
    )

    # Keep velocities in the simulation frame: no bulk velocity subtraction.
    bulk = np.asarray(grid.get_field_parameter('bulk_velocity'))
    if not np.isfinite(bulk).all() or np.any(bulk != 0):
        raise ValueError('Unexpected bulk velocity')

    return grid


def read_simulation_fields(
    grid,
    core: slice,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copy density, QUOKKA temperature and vz from the retained x layers.

    grid comes from read_slab_grid(); core is its retained x slice.
    Returns three 3D NumPy arrays in g/cm^3, K and km/s, respectively.
    Example: core=slice(0, 8) gives three arrays of shape (8, 256, 2048).
    The copies remain valid after grid is released.
    """
    density_with_units = grid['gas', 'density'].to('g/cm**3')
    density = np.array(density_with_units[core])
    del density_with_units

    # The raw QUOKKA temperature field contains numerical values in kelvin.
    temperature = np.array(grid['boxlib', 'temperature'][core], dtype=float)

    velocity_with_units = grid['gas', 'velocity_z'].to('km/s')
    velocity_z = np.array(velocity_with_units[core])
    del velocity_with_units

    return density, temperature, velocity_z


def calculate_dust_foreground_column(
    snapshot: Snapshot,
    density_g_cm3: np.ndarray,
) -> np.ndarray:
    """Calculate each cell's hydrogen column to the outer -z observer face.

    density_g_cm3 is the 3D density array from read_simulation_fields().
    snapshot supplies dz; physics.settings supplies X_H.
    Returns a same-shape NumPy array [cm^-2].
    Example: input and output both have shape (8, 256, 2048).
    The emitting cell contributes half its width; nearer cells contribute fully.
    """
    from .physics import settings

    hydrogen_density = density_g_cm3 * settings.X_H / HYDROGEN_MASS_G
    dz_cm = float(snapshot.cell_widths[2].to('cm').value)
    foreground_column = observer_side_hydrogen_column(hydrogen_density, dz_cm)
    return foreground_column


def calculate_shielding_column(grid, core: slice) -> np.ndarray:
    """Calculate the +z/-z harmonic-mean hydrogen column for table queries.

    grid comes from read_slab_grid(); core selects the retained x layers.
    Returns a copied 3D NumPy array [cm^-2], normally shape (8, 256, 2048).
    Uses physics.gas_fields.shielding_hydrogen_column() for the +/-z columns.
    """
    from .physics.gas_fields import shielding_hydrogen_column

    shielding_with_units = shielding_hydrogen_column(grid)
    shielding_with_units = shielding_with_units.to('cm**-2')
    shielding_column = np.array(shielding_with_units[core])
    return shielding_column


def calculate_core_velocity_gradient(
    snapshot: Snapshot, grid, x_start: int, x_stop: int, core: slice,
) -> np.ndarray:
    """Calculate dV/dr with neighbours, then copy only the retained x layers.

    grid comes from read_slab_grid(); x_start/x_stop are the requested global
    bounds. core selects those cells within the loaded grid, as set by read_slab().
    Returns a copied 3D NumPy array [s^-1], normally shape (8, 256, 2048).
    Uses slab_velocity_gradient() for periodic x/y and one-sided z box faces.
    """
    # The first/last core slab also needs a neighbour from the opposite x face.
    # Example: x=0 uses x=255 and x=1; x=255 uses x=254 and x=0.
    gradient_with_units = slab_velocity_gradient(
        snapshot=snapshot,
        grid=grid,
        x_start=x_start,
        x_stop=x_stop,
    )
    gradient_with_units = gradient_with_units.to('s**-1')
    velocity_gradient = np.array(gradient_with_units[core])
    return velocity_gradient
