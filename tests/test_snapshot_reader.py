"""Check native slab geometry, zero-copy batches, units, and array lifetime."""
import gc
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from unyt import unyt_array

from quokka2s.constants import HYDROGEN_MASS_G
from quokka2s.physics.settings import X_H
from quokka2s.snapshot_reader import (
    Snapshot, SlabArrays, read_x_velocity_plane, slab_windows,
)


def _snapshot_with_velocity(velocity, spacing):
    """Provide in-domain grid reads with units different from the LVG calculation."""
    shape = velocity['velocity_x'].shape

    class Grid:
        def __init__(self, lo, hi):
            self.region = slice(lo, hi)
            self.shape = (hi - lo, shape[1], shape[2])

        def get_field_parameter(self, name):
            return np.zeros(3)

        def __getitem__(self, field):
            name = field[1]
            if name == 'density':
                return unyt_array(np.ones(self.shape) * 1e-24, 'g/cm**3')
            if name == 'temperature':
                return np.full(self.shape, 100.)
            if name in velocity:
                return unyt_array(velocity[name][self.region], 'cm/s').to('m/s')
            if name in spacing:
                return unyt_array([spacing[name]], 'cm').to('m')
            raise KeyError(field)

    class Dataset:
        domain_left_edge = unyt_array([0., 0., 0.], 'cm')

        def covering_grid(self, level, left_edge, dims):
            lo = int(round(left_edge[0].to_value('cm') / spacing['dx']))
            assert level == 0
            assert dims[1:] == shape[1:]
            assert 0 <= lo < lo + dims[0] <= shape[0]
            return Grid(lo, lo + dims[0])

    dataset = Dataset()
    dataset.domain_dimensions = np.array(shape)
    widths = unyt_array([spacing[name] for name in ('dx', 'dy', 'dz')], 'cm')
    dataset.domain_width = widths * dataset.domain_dimensions
    return Snapshot(dataset)


class SnapshotReaderTests(unittest.TestCase):
    def test_batch_hydrogen_density_is_cached_read_only_and_released_with_batch(self):
        density = np.array([1e-24, 2e-24, 3e-24, 4e-24])
        density.flags.writeable = False
        slab = SlabArrays(
            density_g_cm3=density,
            foreground_NH_cm2=np.zeros(4),
            temperature_QUOKKA_K=np.full(4, 100.0),
            shielding_NH_cm2=np.full(4, 1e19),
            velocity_gradient_s=np.full(4, 1e-15),
            velocity_z_kms=np.zeros(4),
            x_start=0,
            shape=(1, 1, 4),
            cell_volume_cm3=1.0,
        )
        cells = slab.batch(start=1, stop=4)
        self.assertNotIn('hydrogen_density_cm3', vars(cells))

        hydrogen_density = cells.hydrogen_density_cm3
        expected = density[1:4] * X_H / HYDROGEN_MASS_G
        np.testing.assert_array_equal(hydrogen_density, expected)
        self.assertIs(cells.hydrogen_density_cm3, hydrogen_density)
        self.assertEqual(hydrogen_density.shape, (3,))
        self.assertFalse(hydrogen_density.flags.writeable)
        self.assertFalse(np.shares_memory(hydrogen_density, cells.density_g_cm3))
        np.testing.assert_array_equal(cells.density_g_cm3, density[1:4])

        # The cache belongs to this batch, not to the slab or a shared reader.
        cached_reference = weakref.ref(hydrogen_density)
        del hydrogen_density
        del cells
        gc.collect()
        self.assertIsNone(cached_reference())

    def test_snapshot_geometry_is_derived_from_dataset_without_reading_fields(self):
        dataset = SimpleNamespace(
            domain_dimensions=np.array([12, 3, 4]),
            domain_width=unyt_array([.24, .09, .20], 'm'),
        )
        snapshot = Snapshot(dataset)
        expected_widths = dataset.domain_width / dataset.domain_dimensions
        self.assertEqual(snapshot.shape, (12, 3, 4))
        self.assertEqual(snapshot.cell_count, 144)
        self.assertEqual(snapshot.xy_region, {'x': (0, 12), 'y': (0, 3)})
        self.assertEqual(snapshot.processing_shape, snapshot.shape)
        self.assertEqual(snapshot.processing_cell_count, snapshot.cell_count)
        self.assertEqual(snapshot.processing_xy_origin, (0, 0))
        np.testing.assert_array_equal(snapshot.cell_widths, expected_widths)
        self.assertEqual(snapshot.cell_volume_cm3, float(np.prod(expected_widths.to('cm').value)))
        expected_area = dataset.domain_width[0] * dataset.domain_width[1]
        self.assertEqual(snapshot.projected_area_cm2, float(expected_area.to('cm**2').value))
        self.assertEqual(snapshot.processing_area_cm2, snapshot.projected_area_cm2)
        # Geometry is calculated once and reused; no covering_grid() is needed.
        self.assertIs(snapshot.cell_widths, snapshot.cell_widths)
        self.assertFalse(hasattr(snapshot, 'config'))
        self.assertFalse(hasattr(snapshot, 'physics'))

    def test_selected_geometry_preserves_original_box_and_native_cell_volume(self):
        dataset = SimpleNamespace(
            domain_dimensions=np.array([12, 5, 4]),
            domain_width=unyt_array([24., 15., 20.], 'cm'),
        )
        snapshot = Snapshot(
            dataset=dataset,
            xy_region={'x': [2, 7], 'y': [1, 3]},
        )
        self.assertEqual(snapshot.xy_region, {'x': (2, 7), 'y': (1, 3)})
        self.assertEqual(snapshot.shape, (12, 5, 4))
        self.assertEqual(snapshot.cell_count, 240)
        self.assertEqual(snapshot.projected_area_cm2, 360.)
        self.assertEqual(snapshot.processing_shape, (5, 2, 4))
        self.assertEqual(snapshot.processing_cell_count, 40)
        self.assertEqual(snapshot.processing_xy_origin, (2, 1))
        self.assertEqual(snapshot.processing_area_cm2, 60.)
        self.assertEqual(snapshot.cell_volume_cm3, 30.)

        for requested in ({}, {'x': None}, {'y': None}, {'x': None, 'y': None}):
            with self.subTest(requested=requested):
                full = Snapshot(dataset=dataset, xy_region=requested)
                self.assertEqual(full.processing_shape, full.shape)
                self.assertEqual(full.processing_area_cm2, full.projected_area_cm2)
        x_only = Snapshot(dataset=dataset, xy_region={'x': (11, 12)})
        self.assertEqual(x_only.processing_shape, (1, 5, 4))
        self.assertEqual(x_only.xy_region['y'], (0, 5))

    def test_selected_geometry_rejects_noninteger_empty_and_out_of_box_bounds(self):
        dataset = SimpleNamespace(domain_dimensions=np.array([12, 5, 4]))
        invalid_regions = (
            [], {'z': (0, 4)}, {'x': '2:7'}, {'x': (2,)}, {'x': (False, 2)},
            {'y': (1., 3)}, {'x': (-1, 2)}, {'x': (0, 13)}, {'y': (0, 6)},
            {'x': (4, 4)}, {'y': (3, 1)},
        )
        for region in invalid_regions:
            with self.subTest(region=region), self.assertRaises(ValueError):
                Snapshot(dataset=dataset, xy_region=region)

    def test_one_cell_processing_window_does_not_change_native_grid_requirement(self):
        self.assertEqual(list(slab_windows(nx=1, slab_nx=8)), [(0, 1)])
        with self.assertRaises(ValueError):
            Snapshot(dataset=SimpleNamespace(domain_dimensions=np.array([1, 5, 4])))

    def test_opposite_face_plane_uses_native_width_with_real_yt(self):
        """A dims=1 covering grid stretches that axis in yt; read two layers."""
        import yt

        shape = (6, 3, 4)
        velocity = np.arange(np.prod(shape), dtype=float).reshape(shape)
        ds = yt.load_uniform_grid(
            {('gas', 'velocity_x'): (velocity, 'cm/s')},
            domain_dimensions=shape,
            length_unit='cm',
            periodicity=(False, False, False),
        )
        snapshot = Snapshot(ds)
        for x_index in (0, shape[0] - 1):
            with self.subTest(x_index=x_index):
                actual = read_x_velocity_plane(snapshot, x_index)
                np.testing.assert_array_equal(actual, velocity[x_index])

    def assert_slab_gradients(self, snapshot, expected, slab_sizes, atol=1e-14):
        """Compare every retained cell against an independently derived stencil."""
        for slab_nx in slab_sizes:
            with self.subTest(shape=snapshot.shape, slab_nx=slab_nx):
                pieces = []
                for start, stop in slab_windows(snapshot.shape[0], slab_nx):
                    slab = snapshot.read_slab(x_start=start, x_stop=stop)
                    pieces.append(slab.velocity_gradient_s.reshape(
                        stop - start, *snapshot.shape[1:]))
                np.testing.assert_allclose(np.concatenate(pieces), expected,
                                           rtol=1e-13, atol=atol)

    def test_batch_views_preserve_positions_across_y_and_x_boundaries(self):
        arrays = [np.arange(48, dtype=float) + 100 * i for i in range(6)]
        for array in arrays:
            array.flags.writeable = False
        slab = SlabArrays(*arrays, x_start=4, shape=(4, 3, 4), cell_volume_cm3=7.)
        # Starts partway through z, crosses a y row and the next x plane.
        cells = slab.batch(start=6, stop=18)
        self.assertEqual(slab.cell_count, 48)
        self.assertEqual(slab.first_cell_id, 48)
        self.assertEqual(cells.cell_count, 12)
        self.assertEqual(cells.first_cell_id, 54)
        self.assertEqual(cells.last_cell_id, 65)
        self.assertEqual(cells.x_start, 4)
        self.assertEqual(cells.slab_shape, (4, 3, 4))
        self.assertEqual(cells.batch_start, 6)
        self.assertEqual(cells.cell_volume_cm3, 7.)
        for name, full in vars(slab).items():
            if not isinstance(full, np.ndarray):
                continue
            selected = getattr(cells, name)
            np.testing.assert_array_equal(selected, full[6:18])
            self.assertTrue(np.shares_memory(selected, full))
            self.assertFalse(selected.flags.writeable)
        x, y, z = np.unravel_index(
            np.arange(cells.first_cell_id, cells.last_cell_id + 1), (12, 3, 4))
        np.testing.assert_array_equal(x, [4] * 6 + [5] * 6)
        np.testing.assert_array_equal(y, [1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(z, [2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1])

    def test_compact_y_global_ids_skip_unselected_rows_between_x_layers(self):
        shape = (3, 2, 4)
        arrays = [np.arange(np.prod(shape), dtype=float) for _ in range(6)]
        slab = SlabArrays(
            density_g_cm3=arrays[0],
            foreground_NH_cm2=arrays[1],
            temperature_QUOKKA_K=arrays[2],
            shielding_NH_cm2=arrays[3],
            velocity_gradient_s=arrays[4],
            velocity_z_kms=arrays[5],
            x_start=4,
            shape=shape,
            cell_volume_cm3=7.,
            y_start=2,
            native_y_size=7,
        )
        native_ids = np.arange(12 * 7 * 4).reshape(12, 7, 4)[4:7, 2:4, :].ravel()
        actual_ids = [slab.cell_id_at(index) for index in range(slab.cell_count)]
        np.testing.assert_array_equal(actual_ids, native_ids)
        self.assertEqual(slab.first_cell_id, 120)
        self.assertEqual(slab.cell_id_at(7), 127)
        self.assertEqual(slab.cell_id_at(8), 148)

        cells = slab.batch(start=6, stop=18)
        self.assertEqual(cells.first_cell_id, 126)
        self.assertEqual(cells.last_cell_id, 177)
        self.assertEqual(cells.y_start, 2)
        self.assertEqual(cells.native_y_size, 7)
        np.testing.assert_array_equal(
            [cells.cell_id_at(index) for index in range(6, 18)],
            native_ids[6:18],
        )

    def test_batch_views_release_slab_arrays_when_the_batch_is_released(self):
        slab = SlabArrays(
            *(np.full(48, i, dtype=float) for i in range(6)),
            x_start=4,
            shape=(4, 3, 4),
            cell_volume_cm3=1.,
        )
        array_references = [
            weakref.ref(array) for array in vars(slab).values()
            if isinstance(array, np.ndarray)
        ]
        cells = slab.batch(start=6, stop=18)
        del slab
        gc.collect()
        self.assertTrue(all(reference() is not None for reference in array_references))
        del cells
        gc.collect()
        self.assertTrue(all(reference() is None for reference in array_references))

    def test_read_slab_converts_units_and_discards_halos_and_grid(self):
        values = np.arange(1., 73.).reshape(6, 3, 4)

        class Grid:
            def get_field_parameter(self, name):
                return np.zeros(3)

            def __getitem__(self, field):
                if field == ('gas', 'density'):
                    return unyt_array(values * 1e-24, 'g/cm**3').to('kg/m**3')
                if field == ('boxlib', 'temperature'):
                    return values + 100.
                if field == ('gas', 'velocity_z'):
                    return unyt_array(values * 1e5, 'cm/s')
                raise KeyError(field)

        class Dataset:
            domain_left_edge = unyt_array([0., 0., -4.], 'cm')
            domain_dimensions = np.array([12, 3, 4])
            domain_width = unyt_array([24., 9., 20.], 'cm')

            def covering_grid(self, level, left_edge, dims):
                self.last_request = (level, left_edge.copy(), dims)
                grid = Grid()
                self.grid_reference = weakref.ref(grid)
                return grid

        ds = Dataset()
        snapshot = Snapshot(ds)
        with (
            patch('quokka2s.physics.settings.X_H', .5),
            patch(
                'quokka2s.physics.gas_fields.shielding_hydrogen_column',
                new=lambda grid: unyt_array(values, 'cm**-2').to('m**-2'),
            ),
            patch(
                'quokka2s.physics.gas_fields.velocity_gradient',
                new=lambda grid, **neighbors: unyt_array(values, 's**-1').to('yr**-1'),
            ),
        ):
            slab = snapshot.read_slab(x_start=4, x_stop=8)
        self.assertEqual(ds.last_request[0], 0)
        np.testing.assert_array_equal(ds.last_request[1].to_value('cm'), [6., 0., -4.])
        self.assertEqual(ds.last_request[2], (6, 3, 4))
        self.assertIsNone(ds.grid_reference())
        expected = values[1:5].ravel()
        np.testing.assert_allclose(slab.density_g_cm3, expected * 1e-24, rtol=1e-14, atol=0)
        np.testing.assert_array_equal(slab.temperature_QUOKKA_K, expected + 100.)
        np.testing.assert_allclose(slab.shielding_NH_cm2, expected, rtol=1e-14, atol=0)
        np.testing.assert_allclose(slab.velocity_gradient_s, expected, rtol=1e-14, atol=0)
        np.testing.assert_allclose(slab.velocity_z_kms, expected, rtol=1e-14, atol=0)
        expected_column = np.array([16.25, 50., 86.25, 125.]) * 1e-24 / HYDROGEN_MASS_G
        np.testing.assert_allclose(slab.foreground_NH_cm2[:4], expected_column,
                                   rtol=1e-14, atol=0)
        self.assertEqual(slab.cell_count, 48)
        self.assertEqual(slab.x_start, 4)
        self.assertEqual(slab.shape, (4, 3, 4))
        self.assertEqual(slab.cell_volume_cm3, 30.)
        for array in vars(slab).values():
            if isinstance(array, np.ndarray):
                self.assertFalse(array.flags.writeable)

        # Retaining slab arrays must not retain the yt dataset or its wrapper.
        dataset_reference = weakref.ref(ds)
        snapshot_reference = weakref.ref(snapshot)
        del ds, snapshot
        gc.collect()
        self.assertIsNone(dataset_reference())
        self.assertIsNone(snapshot_reference())

    def test_shielding_column_uses_two_full_cell_z_columns(self):
        from quokka2s.physics import settings
        from quokka2s.physics.gas_fields import shielding_hydrogen_column

        # nH=[1, 2, 4] cm^-3 and dz=2 cm give one-sided columns:
        # toward +z: [14, 12, 8]; toward -z: [2, 6, 14] cm^-2.
        nH = np.array([1., 2., 4.]).reshape(1, 1, 3)
        density = nH * HYDROGEN_MASS_G / settings.X_H
        grid = {
            ('gas', 'density'): unyt_array(density, 'g/cm**3').to('kg/m**3'),
            ('boxlib', 'dz'): unyt_array([.02], 'm'),
        }
        actual = shielding_hydrogen_column(grid).to_value('cm**-2')
        expected = 2. / (1. / np.array([14., 12., 8.]) + 1. / np.array([2., 6., 14.]))
        np.testing.assert_allclose(actual.ravel(), expected, rtol=1e-14, atol=0)

    def test_region_slab_fields_equal_original_slice_at_internal_and_box_faces(self):
        shape = (9, 7, 6)
        i, j, k = np.indices(shape, dtype=float)
        velocity = {
            'velocity_x': i**2 + 2*j*k,
            'velocity_y': 3*j**2 + i*k,
            'velocity_z': 5*k**2 + i*j,
        }
        original = _snapshot_with_velocity(
            velocity=velocity,
            spacing={'dx': 2., 'dy': 3., 'dz': 5.},
        )
        regions = (
            {'x': (0, 2), 'y': (0, 3)},
            {'x': (2, 7), 'y': (1, 5)},
            {'x': (8, 9), 'y': (6, 7)},
        )
        for region in regions:
            with self.subTest(region=region):
                selected = Snapshot(dataset=original.dataset, xy_region=region)
                x_start, x_stop = region['x']
                y_start, y_stop = region['y']
                full_slab = original.read_slab(x_start=x_start, x_stop=x_stop)
                region_slab = selected.read_slab(x_start=x_start, x_stop=x_stop)
                self.assertEqual(region_slab.shape, (x_stop - x_start, y_stop - y_start, 6))
                self.assertEqual(region_slab.y_start, y_start)
                self.assertEqual(region_slab.native_y_size, 7)
                self.assertEqual(region_slab.cell_volume_cm3, full_slab.cell_volume_cm3)
                for name, full_values in vars(full_slab).items():
                    if not isinstance(full_values, np.ndarray):
                        continue
                    expected = full_values.reshape(full_slab.shape)[:, y_start:y_stop, :]
                    actual = getattr(region_slab, name)
                    np.testing.assert_array_equal(actual, expected.ravel())
                    self.assertFalse(actual.flags.writeable)

    def test_single_x_region_copies_y_crop_but_full_y_keeps_existing_views(self):
        dataset = SimpleNamespace(
            domain_dimensions=np.array([4, 5, 4]),
            domain_width=unyt_array([8., 15., 20.], 'cm'),
        )
        fields = [np.arange(20., dtype=float).reshape(1, 5, 4) + i for i in range(6)]
        for region in (None, {'y': (2, 4)}):
            with (
                self.subTest(region=region),
                patch('quokka2s.snapshot_reader.read_slab_grid'),
                patch(
                    'quokka2s.snapshot_reader.read_simulation_fields',
                    return_value=(fields[0], fields[2], fields[5]),
                ),
                patch(
                    'quokka2s.snapshot_reader.calculate_dust_foreground_column',
                    return_value=fields[1],
                ),
                patch(
                    'quokka2s.snapshot_reader.calculate_shielding_column',
                    return_value=fields[3],
                ),
                patch(
                    'quokka2s.snapshot_reader.calculate_core_velocity_gradient',
                    return_value=fields[4],
                ),
            ):
                snapshot = Snapshot(dataset=dataset, xy_region=region)
                slab = snapshot.read_slab(x_start=1, x_stop=2)
                arrays = [value for value in vars(slab).values() if isinstance(value, np.ndarray)]
                for array, original in zip(arrays, fields):
                    self.assertEqual(np.shares_memory(array, original), region is None)

    def test_gradient_matches_whole_grid_at_slab_seams_and_box_faces(self):
        """Exercise real LVG differences, including a one-cell final slab."""
        from quokka2s.physics.gas_fields import VELOCITY_GRADIENT_FLOOR_S, velocity_gradient

        shape = (17, 5, 6)
        i, j, k = np.indices(shape, dtype=float)
        velocity = {
            'velocity_x': i**2 + 2*j*k,
            'velocity_y': 3*j**2 + i*k,
            'velocity_z': 5*k**2 + i*j,
        }
        spacing = {'dx': 2., 'dy': 3., 'dz': 5.}

        snapshot = _snapshot_with_velocity(velocity, spacing)
        grid = snapshot.dataset.covering_grid(0, snapshot.dataset.domain_left_edge, shape)
        whole_grid = velocity_gradient(grid).to_value('s**-1')

        # A quadratic gives 2*a*i internally. Periodic x/y endpoints use
        # the opposite box face, whereas z retains one-sided differences.
        dx = 2*i
        dy = 6*j
        dz = 10*k
        dx[0] = (1. - (shape[0] - 1)**2) / 2.
        dx[-1] = -(shape[0] - 2)**2 / 2.
        dy[:, 0] = 3.*(1. - (shape[1] - 1)**2) / 2.
        dy[:, -1] = -3.*(shape[1] - 2)**2 / 2.
        dz[:, :, 0] = 5.
        dz[:, :, -1] = 5*(2*shape[2] - 3.)
        divergence = dx/spacing['dx'] + dy/spacing['dy'] + dz/spacing['dz']
        expected = np.maximum(np.abs(divergence) / 3., VELOCITY_GRADIENT_FLOOR_S)
        np.testing.assert_allclose(whole_grid, expected, rtol=1e-13, atol=1e-14)
        self.assert_slab_gradients(snapshot, expected, (1, 4, 8, 17, 20))

    def test_periodic_two_cell_axes_have_zero_centered_derivative(self):
        """For two periodic cells, the left and right neighbours coincide."""
        from quokka2s.physics.gas_fields import VELOCITY_GRADIENT_FLOOR_S, velocity_gradient

        shape = (2, 2, 3)
        i, j, k = np.indices(shape, dtype=float)
        velocity = {'velocity_x': 7*i, 'velocity_y': 11*j, 'velocity_z': 0*k}
        snapshot = _snapshot_with_velocity(velocity, {'dx': 2., 'dy': 3., 'dz': 5.})
        grid = snapshot.dataset.covering_grid(0, snapshot.dataset.domain_left_edge, shape)
        expected = np.full(shape, VELOCITY_GRADIENT_FLOOR_S)
        np.testing.assert_array_equal(velocity_gradient(grid).to_value('s**-1'), expected)
        self.assert_slab_gradients(snapshot, expected, (1, 2, 3), atol=0)

    def test_smooth_periodic_velocity_matches_discrete_sine_derivative(self):
        from quokka2s.physics.gas_fields import VELOCITY_GRADIENT_FLOOR_S, velocity_gradient

        shape = (9, 7, 4)
        i, j, k = np.indices(shape, dtype=float)
        theta_x, theta_y = 2*np.pi/shape[0], 2*np.pi/shape[1]
        velocity = {
            'velocity_x': 8*np.sin(theta_x*i),
            'velocity_y': 5*np.cos(theta_y*j),
            'velocity_z': .4*k,
        }
        spacing = {'dx': 2., 'dy': 3., 'dz': 5.}
        snapshot = _snapshot_with_velocity(velocity, spacing)
        grid = snapshot.dataset.covering_grid(0, snapshot.dataset.domain_left_edge, shape)
        # Centered differences multiply each sine wave by sin(theta)/spacing.
        divergence = (8*np.sin(theta_x)*np.cos(theta_x*i)/spacing['dx']
                      - 5*np.sin(theta_y)*np.sin(theta_y*j)/spacing['dy']
                      + .4/spacing['dz'])
        expected = np.maximum(np.abs(divergence) / 3., VELOCITY_GRADIENT_FLOOR_S)
        np.testing.assert_allclose(velocity_gradient(grid).to_value('s**-1'), expected,
                                   rtol=1e-13, atol=1e-14)
        self.assert_slab_gradients(snapshot, expected, (1, 4, 9))


if __name__ == '__main__':
    unittest.main()
