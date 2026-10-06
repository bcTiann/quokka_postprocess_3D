"""Physical geometry and line-dependent attenuation for LOS-z spectra."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from quokka2s.physics.cell_emission import BatchEmission, LineEmission
from quokka2s.products.integrated_spectra import IntegratedSpectra
from quokka2s.snapshot_reader import CellBatch
from quokka2s.physics.dust_attenuation import DEFAULT_DRAINE_TABLE, LINE_WAVELENGTH_MICRON, attenuate_emissivities, extinction_cross_sections, load_draine_extinction, observer_side_hydrogen_column


class DustAttenuationTests(unittest.TestCase):
    def test_foreground_column_uses_full_foreground_and_half_source_cell(self):
        nH = np.array([[[1., 2., 4.], [3., 6., 12.]]])
        result = observer_side_hydrogen_column(nH, 3.)
        np.testing.assert_allclose(result[0, 0], [1.5, 6., 15.])
        np.testing.assert_allclose(result[0, 1], [4.5, 18., 45.])

    def test_published_table_is_parsed_and_hi21_is_outside_it(self):
        wavelength, sigma = load_draine_extinction()
        self.assertEqual(wavelength.size, 1077)
        self.assertEqual(sigma.shape, wavelength.shape)
        self.assertTrue(np.isfinite(wavelength).all())
        self.assertTrue(np.isfinite(sigma).all())
        self.assertTrue((wavelength > 0).all())
        self.assertTrue((sigma > 0).all())
        self.assertTrue((np.diff(wavelength) > 0).all())
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
        # The production path supplies one line and one scalar cross-section.
        halpha = attenuate_emissivities(source[0], [0., 2.], .5)
        hi21 = attenuate_emissivities(source[1], [0., 2.], 0.)
        np.testing.assert_array_equal(halpha, supplied[0])
        np.testing.assert_array_equal(hi21, supplied[1])
        cells = CellBatch(
            density_g_cm3=np.ones(2),
            temperature_QUOKKA_K=np.full(2, 100.),
            shielding_NH_cm2=np.zeros(2),
            velocity_gradient_s=np.zeros(2),
            foreground_NH_cm2=np.array([0., 2.]),
            velocity_z_kms=np.zeros(2),
            x_start=0,
            slab_shape=(1, 1, 2),
            batch_start=0,
            cell_volume_cm3=3.,
        )
        lines = {}
        for line_index, line_key in enumerate(keys):
            lines[line_key] = LineEmission(
                intrinsic_emissivity_erg_s_cm3=source[line_index],
                attenuated_emissivity_erg_s_cm3=supplied[line_index],
                temperature_K=np.full(2, 100.),
            )
        emission = BatchEmission(
            lines=lines,
            despotic_temperature_K=np.full(2, 100.),
            cold_cells=np.ones(2, dtype=bool),
            despotic_coordinate_clipped_cells={},
            cloudy_column_clipped_cells={},
        )
        accumulator = IntegratedSpectra(keys, [-1., 0., 1.])
        accumulator.add_batch(cells=cells, emission=emission)
        payload, _ = accumulator.build_output(projected_area_cm2=1.)
        np.testing.assert_allclose(payload["input_luminosity_erg_s"][1].sum(axis=-1),
            [(4.+4.*np.exp(-1.))*3., 36.])

    def test_harmless_header_edit_preserves_parsed_values(self):
        expected = load_draine_extinction()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opacity.txt"
            path.write_bytes(b"Additional explanatory header text.\n"
                             + DEFAULT_DRAINE_TABLE.read_bytes())
            actual = load_draine_extinction(path)
        for result, reference in zip(actual, expected):
            np.testing.assert_array_equal(result, reference)

    def test_parser_rejects_missing_extinction_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opacity.txt"
            path.write_text("Table header\n-----------\n1 0 0\n2 0 0\n")
            with self.assertRaisesRegex(ValueError, "column index"):
                load_draine_extinction(path)

    def test_parser_reads_only_required_columns_and_preserves_row_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opacity.txt"
            path.write_text(
                "Table header\n-----------\n"
                "3 unused unused 30 trailing comment\n"
                "1 unused unused 10\n"
                "2 unused unused 20 trailing comment\n"
            )
            wavelength, extinction = load_draine_extinction(path)
        np.testing.assert_array_equal(wavelength, [1., 2., 3.])
        np.testing.assert_array_equal(extinction, [10., 20., 30.])

    def test_parser_requires_a_header_separator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opacity.txt"
            path.write_text("Header without a data separator\n1 0 0 1\n")
            with self.assertRaisesRegex(ValueError, "Missing data separator"):
                load_draine_extinction(path)


if __name__ == "__main__":
    unittest.main()
