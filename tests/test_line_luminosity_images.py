"""Streaming LOS-z images conserve valid native-cell luminosities."""
import unittest

import numpy as np

from quokka2s.products.line_luminosity_images import LineLuminosityImageAccumulator
from quokka2s.physics.cell_emission import BatchEmission, LineEmission
from quokka2s.snapshot_reader import CellBatch


def image_batch(
    line_keys,
    x_start,
    slab_shape,
    batch_start,
    valid_cells,
    intrinsic_emissivity,
    attenuated_emissivity,
    cell_volume_cm3,
):
    """Create the actual named batch contract for image-only test inputs."""
    cell_count = valid_cells.size
    cells = CellBatch(
        density_g_cm3=np.ones(cell_count),
        foreground_NH_cm2=np.ones(cell_count),
        temperature_QUOKKA_K=np.ones(cell_count),
        shielding_NH_cm2=np.ones(cell_count),
        velocity_gradient_s=np.ones(cell_count),
        velocity_z_kms=np.zeros(cell_count),
        x_start=x_start,
        slab_shape=slab_shape,
        batch_start=batch_start,
        cell_volume_cm3=cell_volume_cm3,
    )
    lines = {}
    for row, line_key in enumerate(line_keys):
        lines[line_key] = LineEmission(
            intrinsic_emissivity_erg_s_cm3=np.where(valid_cells, intrinsic_emissivity[row], np.nan),
            attenuated_emissivity_erg_s_cm3=np.where(valid_cells, attenuated_emissivity[row], np.nan),
            temperature_K=np.ones(cell_count),
        )
    emission = BatchEmission(
        lines=lines,
        despotic_temperature_K=np.ones(cell_count),
        cold_cells=np.zeros(cell_count, dtype=bool),
        despotic_coordinate_clipped_cells={'nH': 0, 'NH': 0, 'dVdr': 0},
        cloudy_column_clipped_cells={'below': 0, 'above': 0},
    )
    return cells, emission


class LineLuminosityImageAccumulatorTests(unittest.TestCase):
    def test_two_slabs_and_irregular_batches_match_cellwise_reference(self):
        native_shape = (4, 4, 3)
        line_keys = ("halpha", "co21")
        accumulator = LineLuminosityImageAccumulator(line_keys, native_shape[:2])
        cell_indices = np.arange(np.prod(native_shape)).reshape(native_shape)
        valid_cube = (cell_indices % 7 != 0)
        emissivity_cube = np.stack((cell_indices + 1., 2. * cell_indices + 3.))
        attenuation_cube = (1. - .1 * (cell_indices % 3))
        volume_cube = 1. + .25 * (cell_indices % 5)

        for x_start in (0, 2):
            slab_shape = (2, 4, 3)
            valid = valid_cube[x_start:x_start + 2].ravel()
            intrinsic = emissivity_cube[:, x_start:x_start + 2].reshape(2, -1)
            transmitted = (emissivity_cube * attenuation_cube)[:, x_start:x_start + 2].reshape(2, -1)
            volume = volume_cube[x_start:x_start + 2].ravel()
            for start, stop in ((0, 5), (5, 18), (18, valid.size)):
                cells, emission = image_batch(
                    line_keys=line_keys,
                    x_start=x_start,
                    slab_shape=slab_shape,
                    batch_start=start,
                    valid_cells=valid[start:stop],
                    intrinsic_emissivity=intrinsic[:, start:stop] * volume[start:stop],
                    attenuated_emissivity=transmitted[:, start:stop] * volume[start:stop],
                    cell_volume_cm3=1.0,
                )
                accumulator.add_batch(cells=cells, emission=emission)

        expected_native = np.zeros((2, 2, 4, 4))
        for x in range(native_shape[0]):
            for y in range(native_shape[1]):
                for z in range(native_shape[2]):
                    if not valid_cube[x, y, z]:
                        continue
                    luminosity = emissivity_cube[:, x, y, z] * volume_cube[x, y, z]
                    expected_native[0, :, x, y] += luminosity
                    expected_native[1, :, x, y] += luminosity * attenuation_cube[x, y, z]
        # Each x-y sightline keeps its own summed luminosity when saved.
        np.testing.assert_allclose(accumulator.native_images, expected_native, rtol=1e-15)
        payload = accumulator.build_output()
        np.testing.assert_allclose(payload["line_luminosity_image_erg_s"], expected_native,
                                   rtol=1e-15)
        np.testing.assert_allclose(payload["total_luminosity_erg_s"],
                                   expected_native.sum(axis=(-2, -1)))
        np.testing.assert_array_equal(payload["image_shape"], [4, 4])
        np.testing.assert_array_equal(payload["dust_state_keys"], ["intrinsic", "attenuated"])
        self.assertEqual(str(payload["axis_order"]), "dust_state,line,x_pixel,y_pixel")
        # Producing output cannot alter the native image used for validation.
        np.testing.assert_allclose(accumulator.native_images, expected_native, rtol=1e-15)

    def test_image_keeps_odd_native_shape(self):
        accumulator = LineLuminosityImageAccumulator(("cii",), (5, 4))
        self.assertEqual(accumulator.native_xy_shape, (5, 4))
        self.assertEqual(accumulator.native_images.shape, (2, 1, 5, 4))


if __name__ == "__main__":
    unittest.main()
