import unittest

import numpy as np

from quokka2s.pipeline.prep.physics_fields import (
    build_integrated_spectrum,
    build_spectral_cube,
)


class IntegratedSpectrumTests(unittest.TestCase):
    def test_direct_accumulation_matches_spatial_cube_sum(self):
        rng = np.random.default_rng(20260720)
        shape = (4, 3, 5)
        c_cms = 2.99792458e10
        nu_0 = 1.4204057518e9

        velocity = rng.uniform(-2.0e6, 2.0e6, size=shape)
        shifted = nu_0 * (1.0 - velocity / c_cms)
        luminosity = 10.0 ** rng.uniform(25.0, 30.0, size=shape)
        thermal_width = 10.0 ** rng.uniform(4.0, 6.5, size=shape)
        bandwidth = nu_0 * (5.0e6 / c_cms) * 2.0
        edges = np.linspace(
            nu_0 - bandwidth / 2.0,
            nu_0 + bandwidth / 2.0,
            308,
        )

        cube = build_spectral_cube(
            shifted, luminosity, thermal_width, edges, c_cms,
        )
        direct = build_integrated_spectrum(
            shifted,
            luminosity,
            thermal_width,
            edges,
            c_cms,
            cell_chunk=7,
        )

        np.testing.assert_allclose(
            direct,
            cube.sum(axis=(1, 2)),
            rtol=2.0e-14,
            atol=0.0,
        )

    def test_direct_accumulation_rejects_mismatched_shapes(self):
        edges = np.linspace(1.0, 2.0, 5)
        with self.assertRaises(ValueError):
            build_integrated_spectrum(
                np.ones((2, 2, 2)),
                np.ones((2, 2, 1)),
                np.ones((2, 2, 2)),
                edges,
                3.0e10,
            )


if __name__ == '__main__':
    unittest.main()
