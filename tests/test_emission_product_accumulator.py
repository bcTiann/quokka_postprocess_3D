"""Streaming LOS-z images conserve valid native-cell luminosities."""
import unittest

import numpy as np

from quokka2s.emission_product_accumulator import LineLuminosityImageAccumulator


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
                accumulator.add_slab_batch(
                    x_start=x_start, slab_shape=slab_shape, batch_start=start,
                    valid_mask=valid[start:stop],
                    intrinsic_emissivity=intrinsic[:, start:stop],
                    transmitted_emissivity=transmitted[:, start:stop],
                    cell_volume_cm3=volume[start:stop],
                )

        expected_native = np.zeros((2, 2, 4, 4))
        for x in range(native_shape[0]):
            for y in range(native_shape[1]):
                for z in range(native_shape[2]):
                    if not valid_cube[x, y, z]:
                        continue
                    luminosity = emissivity_cube[:, x, y, z] * volume_cube[x, y, z]
                    expected_native[0, :, x, y] += luminosity
                    expected_native[1, :, x, y] += luminosity * attenuation_cube[x, y, z]
        # All z cells have been summed into separate native x-y sightlines;
        # adjacent x-y pixels have not yet been merged.
        np.testing.assert_allclose(accumulator.native_images, expected_native, rtol=1e-15)
        payload = accumulator.finalize()
        expected = expected_native.reshape(2, 2, 2, 2, 2, 2).sum(axis=(3, 5))
        np.testing.assert_allclose(payload["line_luminosity_image_erg_s"], expected, rtol=1e-15)
        np.testing.assert_allclose(payload["total_luminosity_erg_s"], expected.sum(axis=(-2, -1)))
        np.testing.assert_array_equal(payload["variant_keys"], ["intrinsic", "attenuated"])
        self.assertEqual(str(payload["axis_order"]), "variant,line,x_pixel,y_pixel")
        # Producing output cannot alter the native image used for validation.
        np.testing.assert_allclose(accumulator.native_images, expected_native, rtol=1e-15)

    def test_rejects_odd_native_shape_and_invalid_batch_without_mutation(self):
        with self.assertRaisesRegex(ValueError, "even"):
            LineLuminosityImageAccumulator(("cii",), (5, 4))
        accumulator = LineLuminosityImageAccumulator(("cii",), (4, 4))
        with self.assertRaisesRegex(ValueError, "outside the slab"):
            accumulator.add_slab_batch(
                x_start=0, slab_shape=(2, 4, 3), batch_start=24,
                valid_mask=np.array([True]),
                intrinsic_emissivity=np.array([[1.]]),
                transmitted_emissivity=np.array([[.5]]), cell_volume_cm3=1.,
            )
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            accumulator.add_slab_batch(
                x_start=0, slab_shape=(2, 4, 3), batch_start=0,
                valid_mask=np.array([True]),
                intrinsic_emissivity=np.array([[1.]]),
                transmitted_emissivity=np.array([[np.nan]]), cell_volume_cm3=1.,
            )
        np.testing.assert_array_equal(accumulator.finalize()["line_luminosity_image_erg_s"], 0.)


if __name__ == "__main__":
    unittest.main()
