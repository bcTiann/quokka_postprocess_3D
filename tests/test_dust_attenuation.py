"""Physical geometry and line-dependent attenuation for LOS-z spectra."""
import unittest

import numpy as np

from quokka2s.adopted_spectral_products import AdoptedSpectralAccumulator
from quokka2s.dust_attenuation import (
    LINE_WAVELENGTH_MICRON, attenuate_emissivities,
    extinction_cross_sections, load_draine_extinction,
    observer_side_hydrogen_column,
)


class DustAttenuationTests(unittest.TestCase):
    def test_foreground_column_uses_full_foreground_and_half_source_cell(self):
        nH = np.array([[[1., 2., 4.], [3., 6., 12.]]])
        result = observer_side_hydrogen_column(nH, 3.)
        np.testing.assert_allclose(result[0, 0], [1.5, 6., 15.])
        np.testing.assert_allclose(result[0, 1], [4.5, 18., 45.])

    def test_published_table_is_parsed_and_hi21_is_outside_it(self):
        wavelength, sigma = load_draine_extinction()
        self.assertEqual(wavelength.size, 1077)
        self.assertEqual(wavelength[0], 1.e-4)
        self.assertEqual(wavelength[-1], 1.e4)
        self.assertAlmostEqual(sigma[-1], 1.293e-28, delta=1.e-31)
        keys = ("halpha", "cii", "ciii_977", "co10", "hi21")
        values = extinction_cross_sections(keys, wavelength, sigma)
        self.assertGreater(values[2], values[0])
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[3])
        self.assertEqual(values[-1], 0.)
        self.assertGreater(LINE_WAVELENGTH_MICRON["hi21"], wavelength[-1])

    def test_cell_attenuation_precedes_spectral_binning(self):
        keys = ("halpha", "hi21")
        source = np.array([[4., 4.], [6., 6.]])
        # tau=1 in the far cell for Halpha; H I 21 cm remains unchanged.
        supplied = attenuate_emissivities(source, [0., 2.], [.5, 0.])
        np.testing.assert_allclose(supplied[0], [4., 4.*np.exp(-1.)])
        np.testing.assert_array_equal(supplied[1], source[1])
        accumulator = AdoptedSpectralAccumulator(keys, [-1., 0., 1.], 1.3806488e-16)
        accumulator.add([0., 0.], np.full((2, 2), 10.), supplied, 3.)
        payload, _ = accumulator.finalize()
        np.testing.assert_allclose(payload["input_luminosity_erg_s"][:, 1],
            [(4.+4.*np.exp(-1.))*3., 36.])


if __name__ == "__main__":
    unittest.main()
