# data_handling.py
import yt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from unyt import unyt_array

from .utils.axes import axis_index, axis_label


# Primitive fields that the GOW chemistry / temperature derived-field chain
# in physics_fields.py consumes.  All other fields are derived from these.
# Stored as {field_key_on_orig_ds: (output_dict_key_for_uniform_grid, unit_str)}.
#
# Note on namespacing: yt's load_uniform_grid stream frontend doesn't allow
# fields under arbitrary ftypes like 'boxlib' to be looked up downstream
# (the field type is rejected by the field detector even after add_field).
# So we read ('boxlib','temperature') from ds_orig but stash it under the
# stream namespace as 'temperature_raw'; an alias is registered on the new
# ds (see _register_boxlib_aliases below) so consumers like
# `_temperature_quokka` and `_column_density_H` can still read
# ('boxlib','temperature') / ('boxlib','dx|dy|dz') unchanged.
_DOWNSAMPLE_PRIMITIVES: Dict[Tuple[str, str], Tuple[object, str]] = {
    ('gas', 'density'):                 (('gas', 'density'),                'g/cm**3'),
    ('gas', 'velocity_x'):              (('gas', 'velocity_x'),             'cm/s'),
    ('gas', 'velocity_y'):              (('gas', 'velocity_y'),             'cm/s'),
    ('gas', 'velocity_z'):              (('gas', 'velocity_z'),             'cm/s'),
    ('gas', 'total_energy_density'):    (('gas', 'total_energy_density'),   'erg/cm**3'),
    ('gas', 'kinetic_energy_density'):  (('gas', 'kinetic_energy_density'), 'erg/cm**3'),
    ('boxlib', 'temperature'):          ('temperature_raw',                 'dimensionless'),
}


def _register_boxlib_aliases(ds_lo: 'yt.Dataset') -> None:
    """Make ('boxlib', 'temperature' / 'dx' / 'dy' / 'dz') resolvable on a
    stream-frontend dataset, mirroring what yt does automatically for
    real plotfiles.  These are the only ('boxlib', ...) fields read by
    `add_all_fields` in physics_fields.py."""
    if 'boxlib' not in ds_lo.fluid_types:
        ds_lo.fluid_types = ds_lo.fluid_types + ('boxlib',)

    def _temperature_alias(field, data):
        return data[('stream', 'temperature_raw')]

    ds_lo.add_field(
        ('boxlib', 'temperature'),
        function=_temperature_alias,
        sampling_type='cell',
        units='dimensionless',
        force_override=True,
    )

    for axis in ('x', 'y', 'z'):
        idx_field = ('index', f'd{axis}')

        def _make_alias(src=idx_field):
            def _alias(field, data):
                return data[src]
            return _alias

        ds_lo.add_field(
            ('boxlib', f'd{axis}'),
            function=_make_alias(),
            sampling_type='cell',
            units='cm',
            force_override=True,
        )


def make_downsampled_dataset(ds_orig: 'yt.Dataset', factor: int) -> 'yt.Dataset':
    """Block-mean-downsample a yt dataset by an integer factor along each axis.

    Reads the small set of primitive fields actually consumed by the
    derived-field chain slab-by-slab from `ds_orig`, averages each
    factor³ block, and packs the results into a new in-memory uniform-grid
    dataset.  All `physics_fields.add_all_fields` derived fields then
    work on the new dataset unchanged.

    Memory: peak ≈ one slab (nx × ny × slab_nz × 8 bytes) per field.
    For 256×256×2048 with slab_nz=128 that is ~64 MB at a time, well
    below the multi-GB peaks the full-domain derived-field chain hits.

    factor=1 is a fast pass-through (returns ds_orig unchanged).
    """
    if factor == 1:
        return ds_orig

    # yt's RegionSelector rejects any region whose edge sits even one ULP
    # outside the domain.  When we slab-loop along z, the last slab's
    # implicit covering_grid right edge (= slab_left + dims*dx) drifts
    # past domain_right_edge by FP rounding, and yt crashes before
    # reading any data.  Forcing periodicity disables that strict check;
    # we never actually read past the domain (only the slab edges suffer
    # ULP drift, no cell falls in that sliver), so this is functionally
    # a no-op.  yt's error message itself recommends this workaround.
    ds_orig.force_periodicity()

    nx, ny, nz = (int(d) for d in ds_orig.domain_dimensions)
    if any(d % factor != 0 for d in (nx, ny, nz)):
        raise ValueError(
            f"domain_dimensions {(nx, ny, nz)} not divisible by factor={factor}"
        )

    nx_lo, ny_lo, nz_lo = nx // factor, ny // factor, nz // factor
    print(f"[downsample] {nx}x{ny}x{nz} -> {nx_lo}x{ny_lo}x{nz_lo} (factor={factor})")

    out_arrays = {
        src_key: np.empty((nx_lo, ny_lo, nz_lo), dtype=np.float64)
        for src_key in _DOWNSAMPLE_PRIMITIVES
    }

    le = ds_orig.domain_left_edge       # unyt_array, code_length
    re = ds_orig.domain_right_edge

    slab_nz = factor * 64

    for iz in range(0, nz, slab_nz):
        cur_nz = min(slab_nz, nz - iz)

        slab_left  = le.copy()
        slab_right = re.copy()
        slab_left[2]  = le[2] + (re[2] - le[2]) * iz / nz
        slab_right[2] = le[2] + (re[2] - le[2]) * (iz + cur_nz) / nz

        box_region = ds_orig.box(slab_left, slab_right)
        grid = ds_orig.covering_grid(
            level=0,
            left_edge=slab_left,
            dims=(nx, ny, cur_nz),
            data_source=box_region,
        )

        for src_key, (_out_key, unit_str) in _DOWNSAMPLE_PRIMITIVES.items():
            if unit_str == 'dimensionless':
                arr = np.asarray(grid[src_key].d)
            else:
                arr = grid[src_key].in_cgs().value

            cur_nz_lo = cur_nz // factor
            ds_arr = arr.reshape(
                nx_lo, factor, ny_lo, factor, cur_nz_lo, factor,
            ).mean(axis=(1, 3, 5))
            out_arrays[src_key][..., iz // factor : iz // factor + cur_nz_lo] = ds_arr
            del arr, ds_arr

        del grid, box_region
        print(f"[downsample] slab z=[{iz}:{iz + cur_nz}] done")

    data_dict = {
        out_key: (out_arrays[src_key], unit_str)
        for src_key, (out_key, unit_str) in _DOWNSAMPLE_PRIMITIVES.items()
    }

    le_cm = ds_orig.domain_left_edge.in_units('cm').value
    re_cm = ds_orig.domain_right_edge.in_units('cm').value
    bbox = np.array([
        [le_cm[0], re_cm[0]],
        [le_cm[1], re_cm[1]],
        [le_cm[2], re_cm[2]],
    ])

    ds_lo = yt.load_uniform_grid(
        data=data_dict,
        domain_dimensions=(nx_lo, ny_lo, nz_lo),
        bbox=bbox,
        length_unit='cm',
        mass_unit='g',
        time_unit='s',
    )
    _register_boxlib_aliases(ds_lo)
    print(f"[downsample] built downsampled dataset: dims={ds_lo.domain_dimensions.tolist()}")
    return ds_lo


class YTDataProvider:
    def __init__(self, ds):
        self.ds = ds
        self._cached_grid = None  # populated lazily by _get_full_grid()

    def _get_full_grid(self):
        """Return (and lazily create) the full-domain covering_grid.

        Why this exists
        ---------------
        yt caches every derived field *on the covering_grid object*.  If you
        call ``ds.covering_grid(...)`` twice you get two independent objects;
        the second one recomputes every derived field from scratch even if the
        geometry is identical.  In this pipeline the expensive fields are:

          column_density_H  – 6 cumulative-sum passes over the full cube
          temperature       – 25-iteration bisection over ~8 M cells

        Both are dependencies of every luminosity and thermal-width field.
        Without caching, these are recomputed once per ``get_slab_z`` call
        (≈ 10 times across EmitterTask + TripleLineTask combined).

        With this cache, the grid is constructed once; yt's internal cache
        satisfies every subsequent field request on the same object for free.
        """
        if self._cached_grid is None:
            level = self.ds.max_level
            dims = self.ds.domain_dimensions * (2 ** level)
            self._cached_grid = self.ds.covering_grid(
                level=level,
                left_edge=self.ds.domain_left_edge,
                dims=dims,
            )
        return self._cached_grid

    def get_slice(self,
                  field: Tuple[str, str],
                  axis: Union[str, int],
                  coord: Optional[float] = None,
                  resolution: Tuple[int, int] = (800, 800)) -> np.ndarray:
        """
        Get a slice of 2D YTNdarray for the specified field and axis.
        """
        axis_str = axis_label(axis)
        axis_idx = axis_index(axis)

        if coord is None:
            coord = self.ds.domain_center[axis_idx]

        full_domain_width = self.ds.domain_width
        plane_axes = [i for i in range(3) if i != axis_idx]
        
        slice_width = full_domain_width[plane_axes[0]]
        slice_height = full_domain_width[plane_axes[1]]

        slc = self.ds.slice(axis_str, coord=coord)

        frb = slc.to_frb(width=slice_width, height=slice_height, resolution=resolution)
        
        data_with_units = frb[field]
        # numpy_data = np.array(data_with_units)
        # unit_string = str(data_with_units.units)

        # print("="*40)
        # print(f"Slice: field = {field}, axis = {axis_str}, units = {data_with_units.units}")
        # print("="*40)
        
        return data_with_units
    

    def get_grid_data(self,
                    field: Tuple[str, str],
                    level: Optional[int] = None,
                    dims: Tuple[int, int, int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 3D array of the entire dataset in [cgs units].
        Returns:
            - 3D YT Narray of the field data (with units)
            - 
        """
        
        use_cache = (level is None and dims is None)

        if level is None:
            level = self.ds.max_level
        if dims is None:
            dims = self.ds.domain_dimensions * (2**level)

        if use_cache:
            grid = self._get_full_grid()
        else:
            grid = self.ds.covering_grid(level=level, left_edge=self.ds.domain_left_edge, dims=dims)
        data_with_units = grid[field].in_cgs()

        # print(f"Retrieved 3D grid data for field '{field}' with shape {data_with_units.shape}")
        # print(f"units: {data_with_units.units}")

        return data_with_units
    


    def downsample_3d_array(self,
                        data_cube: unyt_array,
                        factor: int
                        ) -> unyt_array:
        """
        Downsamples a 3D unyt_array by an integer factor by averaging blocks,
        preserving the units.

        Parameters:
        - data_cube: The input 3D unyt_array (e.g., shape (128, 128, 128)).
        - factor: The integer factor to downsample by (e.g., 2, 4, 8).
                The dimensions of the data_cube must be divisible by the factor.

        Returns:
        - The downsampled 3D unyt_array with the same units as the input.
        """

        orig_shape = np.array(data_cube.shape)

        if not np.all(orig_shape % factor == 0):
            raise ValueError(f"The shape of the data cube {orig_shape} is notdivisible by the facot {factor}.")
        
        new_shape = (orig_shape // factor).astype(int)

        reshaped_cube = data_cube.reshape(new_shape[0], factor,
                                        new_shape[1], factor,
                                        new_shape[2], factor)
        downsampled = reshaped_cube.mean(axis=(1, 3, 5))

        return downsampled



    def get_cubic_box(self,
                      field: Tuple[str, str],
                      box_width: Optional[unyt_array] = None,
                      center: Optional[unyt_array] = None,
                      level: Optional[int] = None
                      ):
        """
        Extracts a data box.
        """

        if center is None:
            center = self.ds.domain_center
            # print(f"Center not provided. Using domain_center: {center}")

        if level is None:
            level = self.ds.max_level


        if box_width is None:
            min_side_length = self.ds.domain_width.min()
            box_width = self.ds.arr([min_side_length] * 3)
            # print(f"Box width not provided. Defaulting to the largest possible width: {box_width}")
        
        half_width = box_width / 2.0
        left_edge = center - half_width
        right_edge = center + half_width
        print(f"Defining a physical box from {left_edge} to {right_edge}")

        pixel_widths = self.ds.domain_width / self.ds.domain_dimensions

        dims = np.round(box_width / pixel_widths).astype(int) * 2**level
        print(f"Calculated corresponding pixel dims: {dims}")

        box_region = self.ds.region(center, left_edge, right_edge)

        grid = self.ds.covering_grid(
            level=level,
            left_edge=left_edge,
            dims=dims,
            data_source=box_region
        )
        
        data_box = grid[field].in_cgs()
        print(f"Retrieved data box for field '{field}', with shape {data_box.shape}")

        extents = {
            'x': [left_edge[1], right_edge[1], left_edge[2], right_edge[2]],
            'y': [left_edge[0], right_edge[0], left_edge[2], right_edge[2]],
            'z': [left_edge[0], right_edge[0], left_edge[1], right_edge[1]]
        }

        return data_box, extents


    def get_slab_z(self,
                      field: Tuple[str, str],
                      slab_width: Optional[unyt_array] = None,
                      center: Optional[unyt_array] = None,
                      level: Optional[int] = None
                      ):
        """
        Extracts a data slab oriented along the Z-axis.
        The slab covers the full extent of the X and Y dimensions.
        The width of the slab in the Z direction is specified by the user.

        Performance note
        ----------------
        When all three optional arguments are left at their defaults (i.e. the
        full-domain slab), this method reuses ``_get_full_grid()`` instead of
        constructing a new covering_grid.  That means yt's per-grid field
        cache is shared across every default call: expensive derived fields
        such as ``temperature`` and ``column_density_H`` are evaluated at most
        once per pipeline run regardless of how many tasks request them.

        When a non-default slab is requested a fresh covering_grid is created
        as before, so the behaviour for sub-domain slabs is unchanged.
        """
        # Detect whether the caller wants the full domain (all defaults).
        # We check BEFORE applying defaults so the sentinel is unambiguous.
        use_full_grid = (slab_width is None and center is None and level is None)

        if center is None:
            center = self.ds.domain_center
        if level is None:
            level = self.ds.max_level
        if slab_width is None:
            slab_width = self.ds.domain_width[2]

        left_edge_xy = self.ds.domain_left_edge[0:2]
        right_edge_xy = self.ds.domain_right_edge[0:2]

        half_width_z = slab_width / 2.0
        left_edge_z = center[2] - half_width_z
        right_edge_z = center[2] + half_width_z

        left_edge = self.ds.arr([left_edge_xy[0], left_edge_xy[1], left_edge_z])
        right_edge = self.ds.arr([right_edge_xy[0], right_edge_xy[1], right_edge_z])

        pixel_widths = self.ds.domain_width / self.ds.domain_dimensions
        dims_xy = self.ds.domain_dimensions[0:2]
        num_z_pixels = np.round(slab_width / pixel_widths[2]).astype(int)
        dims = np.array([dims_xy[0], dims_xy[1], num_z_pixels]) * 2**level

        if use_full_grid:
            # Reuse the cached full-domain grid.  yt looks up the field in its
            # internal cache on this object; if it was already computed by a
            # previous get_slab_z call the result is returned immediately.
            grid = self._get_full_grid()
        else:
            # Sub-domain or non-default level: construct a fresh grid as before.
            box_region = self.ds.box(left_edge, right_edge)
            grid = self.ds.covering_grid(
                level=level,
                left_edge=left_edge,
                dims=dims,
                data_source=box_region,
            )

        data_slab = grid[field].in_cgs()
        # print(f"Retrieved data slab for field '{field}' with shape {data_slab.shape}")

        extents = {
            'x': [left_edge[1], right_edge[1], left_edge[2], right_edge[2]],
            'y': [left_edge[0], right_edge[0], left_edge[2], right_edge[2]],
            'z': [left_edge[0], right_edge[0], left_edge[1], right_edge[1]]
        }

        return data_slab, extents



    def get_projection(self,
                       field: Tuple[str, str],
                       axis: Union[str, int],
                       weight_field: Optional[Tuple[str, str]] = None,
                       resolution: Tuple[int, int] = (800, 800)) -> np.ndarray:
        """
        Get a projection of 2D Numpy array for the specified field and axis.
        """
        axis_str = axis_label(axis)
        axis_idx = axis_index(axis)
        
        full_domain_width = self.ds.domain_width
        plane_axes = [i for i in range(3) if i != axis_idx]
        
        proj_width = full_domain_width[plane_axes[0]]
        proj_height = full_domain_width[plane_axes[1]]
        
        prj = self.ds.proj(field, axis_str, weight_field=weight_field)

        frb = prj.to_frb(width=proj_width, height=proj_height, resolution=resolution)
        
        data_with_units = frb[field]
        # numpy_data = np.array(data_with_units)
        # unit_string = str(data_with_units.units)
        # print("="*40)
        # print(f"Slice: field = {field}, axis = {axis_str}, units = {data_with_units.units}")
        # print("="*40)
        
        return data_with_units
    
    def get_plot_extent(self, axis: Union[str, int], units: str = 'pc') -> List[float]:
        """
        Get the physical extent of the plot for the specified axis.
        """
        axis_idx = axis_index(axis)
        axes = [i for i in range(3) if i != axis_idx]

        horizon_min, horizen_max = self.ds.domain_left_edge.in_units(units)[axes[0]].value, self.ds.domain_right_edge.in_units(units)[axes[0]].value
        vertical_min, vertical_max = self.ds.domain_left_edge.in_units(units)[axes[1]].value, self.ds.domain_right_edge.in_units(units)[axes[1]].value
        
        return [horizon_min, horizen_max, vertical_min, vertical_max]


    def get_particle_positions(self,
                               axis: Union[str, int],
                               depth: float,
                               coord: Optional[float] = None,
                               ptype: str = 'all',
                               units: str = 'pc'
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get particle positions within a slab for plotting on a slice.

        Args:
            axis: The axis PERPENDICULAR to the slice plane (e.g., 'x' for a y-z plot).
            depth: The thickness of the slab along the given axis.
            coord: The center of the slab along the given axis. Defaults to domain center.
            ptype: The particle type to select (e.g., 'all', 'io').
            units: The units for the returned positions.

        Returns:
            A tuple of two NumPy arrays: (particle_x_coords, particle_y_coords) 
            for the plotting plane.
        """
        axis_idx = axis_index(axis)

        if coord is None:
            coord = self.ds.domain_center[axis_idx].in_units(units).value


        # 1. Define the slab boundaries
        min_coord = self.ds.quan(coord - depth / 2.0, units)
        max_coord = self.ds.quan(coord + depth / 2.0, units)

        # 2. Create the geometric box (the "slab")
        # Start with the full domain edges
        left_edge = self.ds.domain_left_edge.copy()
        right_edge = self.ds.domain_right_edge.copy()

        # Modify the edges along the slice axis to define the slab's thickness
        left_edge[axis_idx] = min_coord
        right_edge[axis_idx] = max_coord
        
        # Create the data object representing only the data within this box
        slab_particles = self.ds.box(left_edge, right_edge)
        
        # -----------------------------------------------

        # Determine the axes for the plot
        plot_axes = [axis_label(i) for i in range(3) if i != axis_idx]

        # Get the particle positions for the plotting axes from the new slab object
        p_x = slab_particles[ptype, f'particle_position_{plot_axes[0]}'].in_units(units)
        p_y = slab_particles[ptype, f'particle_position_{plot_axes[1]}'].in_units(units)

        print("="*40)
        print(f"Particles: Found {len(p_x)} particles of type '{ptype}' in slab.")
        print("="*40)
        
        return p_x, p_y
    

    def get_velocity_field(self,
                           axis: Union[str, int],
                           resolution: Tuple[int, int] = (800, 800),
                           units: str = 'pc',
                           vel_units: str = 'km/s',
                           downsample_factor: int = 25
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets the 2D velocity field on a slice plane for quiver plots.

        Args:
            axis: The axis PERPENDICULAR to the slice plane.
            resolution: The resolution of the underlying grid.
            units: The spatial units for the arrow positions.
            vel_units: The units for the velocity vectors.
            downsample_factor: The factor by which to downsample the vector field to avoid overcrowding.

        Returns:
            A tuple of four 2D NumPy arrays: (X, Y, U, V)
        """
        axis_idx = axis_index(axis)

        # Determine the axes for the plot and corresponding velocity components
        plot_axes_indices = [i for i in range(3) if i != axis_idx]
        plot_axes_str = [axis_label(i) for i in plot_axes_indices]
        
        vel_u_field = ('gas', f'velocity_{plot_axes_str[0]}')
        vel_v_field = ('gas', f'velocity_{plot_axes_str[1]}')

        # get_slice  unyt_array
        u_data = self.get_slice(field=vel_u_field, axis=axis, resolution=resolution)
        v_data = self.get_slice(field=vel_v_field, axis=axis, resolution=resolution)


        extent = self.get_plot_extent(axis=axis, units=units)
        x_coords = np.linspace(extent[0], extent[1], resolution[0]) * yt.units.Unit(units)
        y_coords = np.linspace(extent[2], extent[3], resolution[1]) * yt.units.Unit(units)
        X, Y = np.meshgrid(x_coords, y_coords)

        # downsample
        skip = downsample_factor
        X_down = X[::skip, ::skip]
        Y_down = Y[::skip, ::skip]
        

        U_down = u_data[::skip, ::skip].in_units(vel_units)
        V_down = v_data[::skip, ::skip].in_units(vel_units)
        
        # unyt_array
        return X_down, Y_down, U_down, V_down
