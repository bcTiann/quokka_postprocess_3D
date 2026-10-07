"""Read native snapshot slabs and expose unit-converted cell batches."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .physics.dust_attenuation import observer_side_hydrogen_column
from .physics.gas_fields import hydrogen_number_density_cm3


@dataclass(frozen=True)
class Snapshot:
    """An opened yt snapshot, its grid geometry, and the action of reading a slab.

    Parameters
    ----------
    dataset : yt dataset
        Returned by yt.load() in processing_inputs.open_snapshot(). Opening it reads
        snapshot metadata; cell fields are loaded later by read_slab().
    xy_region : dict, optional
        Native cell-index bounds, e.g. {'x': (64, 128), 'y': (80, 144)}.
        Stops are excluded; omitted axes span the original box. z is always full.

    Geometry properties are calculated from this dataset, so callers do not
    repeat its shape, cell widths, or volume in the constructor.
    Example: snapshot.read_slab(x_start=0, x_stop=8) returns the first eight
    x layers with derived columns and velocity gradients already calculated.
    """

    dataset: object
    xy_region: dict | None = None

    def __post_init__(self):
        """Resolve the selected x/y bounds once, without changing native geometry."""
        if self.shape[0] < 2:
            raise ValueError('Native x dimension must be at least two cells')
        requested = {} if self.xy_region is None else self.xy_region
        if not isinstance(requested, dict) or set(requested) - {'x', 'y'}:
            raise ValueError('xy_region must contain only x and y cell-index ranges')
        normalized = {}
        for axis, size in zip(('x', 'y'), self.shape[:2]):
            bounds = requested.get(axis)
            if bounds is None:
                bounds = (0, size)
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(f'xy_region.{axis} must be a two-integer range')
            start, stop = bounds
            if type(start) is not int or type(stop) is not int:
                raise ValueError(f'xy_region.{axis} must use integer cell indices')
            if not 0 <= start < stop <= size:
                raise ValueError(f'xy_region.{axis} must satisfy 0 <= start < stop <= {size}')
            normalized[axis] = (start, stop)
        object.__setattr__(self, 'xy_region', normalized)

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

    @cached_property
    def processing_shape(self) -> tuple[int, int, int]:
        """Selected native cell counts, e.g. (64, 64, 2048), with full z depth."""
        x_start, x_stop = self.xy_region['x']
        y_start, y_stop = self.xy_region['y']
        return x_stop - x_start, y_stop - y_start, self.shape[2]

    @cached_property
    def processing_cell_count(self) -> int:
        """Number of selected cells before any diagnostic max_slabs limit."""
        return int(np.prod(self.processing_shape))

    @cached_property
    def processing_area_cm2(self) -> float:
        """Selected x-y area [cm^2], retaining original native pixel widths."""
        if self.processing_shape[:2] == self.shape[:2]:
            return self.projected_area_cm2
        nx, ny = self.processing_shape[:2]
        area = nx * self.cell_widths[0] * ny * self.cell_widths[1]
        return float(area.to('cm**2').value)

    @cached_property
    def processing_xy_origin(self) -> tuple[int, int]:
        """Selected image's lower native cell indices, e.g. (64, 80)."""
        return self.xy_region['x'][0], self.xy_region['y'][0]

    def read_slab(self, x_start: int, x_stop: int) -> SlabArrays:
        """Derive fields on original neighbours, then retain selected y rows.

        Parameters
        ----------
        x_start, x_stop : int
            Global x bounds, with x_stop excluded. For example, 0:8 selects the
            first eight x layers. Neighbour reads are handled inside this function.

        Returns
        -------
        SlabArrays
            Six read-only arrays of shape (slab_nx * selected_Ny * Nz,), flattened in
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

        # 6. Crop only after columns and gradients use the original neighbours.
        # Copy even a one-x-layer crop so it cannot retain the full y arrays.
        y_start, y_stop = self.xy_region['y']
        if y_start != 0 or y_stop != self.shape[1]:
            selected_y = slice(y_start, y_stop)
            density = density[:, selected_y, :].copy(order='C')
            temperature = temperature[:, selected_y, :].copy(order='C')
            velocity_z = velocity_z[:, selected_y, :].copy(order='C')
            foreground_column = foreground_column[:, selected_y, :].copy(order='C')
            shielding_column = shielding_column[:, selected_y, :].copy(order='C')
            velocity_gradient = velocity_gradient[:, selected_y, :].copy(order='C')

        # 7. Flatten each field: (8, 64, 2048) -> (1048576,) for y=80:144.
        # z varies fastest, followed by y, then x.
        slab = SlabArrays(
            density_g_cm3=density.ravel(),
            foreground_NH_cm2=foreground_column.ravel(),
            temperature_QUOKKA_K=temperature.ravel(),
            shielding_NH_cm2=shielding_column.ravel(),
            velocity_gradient_s=velocity_gradient.ravel(),
            velocity_z_kms=velocity_z.ravel(),
            x_start=x_start,
            shape=(x_stop - x_start, y_stop - y_start, self.shape[2]),
            cell_volume_cm3=self.cell_volume_cm3,
            y_start=y_start,
            native_y_size=self.shape[1],
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
        Retained core-slab dimensions (slab_nx, selected_Ny, Nz).
    batch_start : int
        First cell index within the C-order flattened slab; z varies fastest.
    cell_volume_cm3 : float
        Native volume of each cell [cm^3].
    y_start : int, optional
        First retained y index in the original grid; zero for the full-y default.
    native_y_size : int, optional
        Original Ny for global IDs; defaults to slab_shape[1] for full-y batches.

    first_cell_id and last_cell_id are derived properties: the inclusive
    endpoint IDs in the C-order flattened full grid. A selected-y batch can
    skip native IDs between its x layers.

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
    y_start: int = 0
    native_y_size: int | None = None

    def __post_init__(self):
        if self.native_y_size is None:
            object.__setattr__(self, 'native_y_size', self.slab_shape[1])

    @cached_property
    def hydrogen_density_cm3(self) -> np.ndarray:
        """Physical nH [cm^-3], computed once from this batch's density.

        density_g_cm3 is the original (B,) rho array [g/cm^3]. The result is
        rho * X_H / HYDROGEN_MASS_G, before any lookup-coordinate clipping.
        Repeated reads return the same read-only array, so both readers and
        emissivity calculations use one nH value per original cell.
        Example: cells.hydrogen_density_cm3[7] is the eighth cell's physical nH.
        """
        hydrogen_density_cm3 = hydrogen_number_density_cm3(
            density_g_cm3=self.density_g_cm3,
        )
        hydrogen_density_cm3.flags.writeable = False
        return hydrogen_density_cm3

    @property
    def cell_count(self) -> int:
        """Return B, the number of cells in this batch, as an int."""
        return self.density_g_cm3.size

    @property
    def first_cell_id(self) -> int:
        """Full-grid ID of the first cell; z varies fastest, then y, then x."""
        return self.cell_id_at(self.batch_start)

    @property
    def last_cell_id(self) -> int:
        """Inclusive full-grid ID of the last cell in this batch."""
        return self.cell_id_at(self.batch_start + self.cell_count - 1)

    def cell_id_at(self, slab_index: int) -> int:
        """Map an index in the compact parent slab to its original full-grid ID."""
        return native_cell_id(
            slab_index=slab_index,
            x_start=self.x_start,
            y_start=self.y_start,
            slab_shape=self.slab_shape,
            native_y_size=self.native_y_size,
        )


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
    y_start : int, optional
        First retained y index in the original grid; normally zero or the region start.
    native_y_size : int, optional
        Original Ny for global IDs; defaults to shape[1] for full-y slabs.

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
    y_start: int = 0
    native_y_size: int | None = None

    def __post_init__(self):
        if self.native_y_size is None:
            object.__setattr__(self, 'native_y_size', self.shape[1])

    @property
    def cell_count(self) -> int:
        """Return S, the number of core-slab cells, as an int."""
        return self.density_g_cm3.size

    @property
    def first_cell_id(self) -> int:
        """Full-grid ID of this slab's first cell, for batch-failure reports."""
        return self.cell_id_at(0)

    def cell_id_at(self, slab_index: int) -> int:
        """Map a compact slab index to its original full-grid ID.

        Example: for shape=(2, 2, 4), x_start=3, y_start=1, native_y_size=5,
        compact indices 7 and 8 have native IDs 71 and 84, respectively.
        """
        return native_cell_id(
            slab_index=slab_index,
            x_start=self.x_start,
            y_start=self.y_start,
            slab_shape=self.shape,
            native_y_size=self.native_y_size,
        )

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
            y_start=self.y_start,
            native_y_size=self.native_y_size,
        )

    def iter_batches(self, *, batch_size: int):
        """Yield consecutive CellBatch views with at most batch_size cells.

        Each batch keeps its slab indices, full-grid IDs and six field views.
        The final batch can be shorter. Consumers calculate and discard these
        views; only the parent slab owns the cell arrays.
        Example: 250 cells with batch_size=100 gives 100, 100 and 50 cells.
        """
        for start in range(0, self.cell_count, batch_size):
            stop = min(start + batch_size, self.cell_count)
            yield self.batch(
                start=start,
                stop=stop,
            )


def native_cell_id(slab_index, x_start, y_start, slab_shape, native_y_size):
    """Map one C-order compact slab index to the original grid, with full z depth."""
    selected_y_size = slab_shape[1]
    nz = slab_shape[2]
    local_x, within_x_layer = divmod(slab_index, selected_y_size * nz)
    local_y, z_index = divmod(within_x_layer, nz)
    x_index = x_start + local_x
    y_index = y_start + local_y
    return int((x_index * native_y_size + y_index) * nz + z_index)


def slab_windows(*, x_start: int, x_stop: int, slab_nx: int):
    """Yield the x layers to process in each slab.

    Parameters
    ----------
    x_start, x_stop : int
        Global x bounds, with x_stop excluded. Full box: 0 and 256;
        selected region example: 64 and 128.
    slab_nx : int
        Maximum number of core x cells in each slab.

    Yields
    ------
    tuple of (int, int)
        (x_start, x_stop), with x_stop excluded. read_slab() determines which
        neighbours to read; callers specify only the cells they want to process.

    Examples
    --------
    >>> list(slab_windows(x_start=0, x_stop=5, slab_nx=3))
    [(0, 3), (3, 5)]
    """
    if x_start < 0 or x_stop <= x_start or slab_nx < 1:
        raise ValueError('Require 0 <= x_start < x_stop and a positive slab size')
    for start in range(x_start, x_stop, slab_nx):
        stop = min(start + slab_nx, x_stop)
        yield start, stop


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
    hydrogen_density = hydrogen_number_density_cm3(
        density_g_cm3=density_g_cm3,
    )
    dz_cm = float(snapshot.cell_widths[2].to('cm').value)
    foreground_column = observer_side_hydrogen_column(
        nH_cm3=hydrogen_density,
        dz_cm=dz_cm,
    )
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
