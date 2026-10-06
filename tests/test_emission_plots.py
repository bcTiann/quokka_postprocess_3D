"""Portable plotting stage: saved arrays are the only data inputs."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from quokka2s.figures.gas_phase_spectra import LINE_ORDER
from quokka2s.figures.emission_results import (
    combine_image_pixels_for_display,
    load_emission_products,
    load_phase_velocity_products,
    plot_emission_products,
)
from quokka2s.plot_emission_results import main


class EmissionPlotTests(unittest.TestCase):
    @staticmethod
    def save_products(
        directory: Path,
        *,
        full_snapshot: bool = True,
        processing_complete: bool | None = None,
        image_shape: tuple[int, int] = (256, 256),
        image_xy_origin: tuple[int, int] = (0, 0),
    ) -> None:
        """Write only the arrays actually consumed by the plotting stage."""
        keys = np.asarray(LINE_ORDER)
        images = np.zeros((2, len(keys), *image_shape))
        images[:, 0, 10, 20] = (10.0, 4.0)
        images[:, 1, 40, 30] = 2.0
        x_start, y_start = image_xy_origin
        nx, ny = image_shape
        complete = full_snapshot if processing_complete is None else processing_complete
        np.savez_compressed(
            directory / "images.npz",
            line_keys=keys,
            line_luminosity_image_erg_s=images,
            x_edges_kpc=np.linspace(0.0, 1.0, 257)[x_start:x_start + nx + 1],
            y_edges_kpc=np.linspace(-0.5, 0.5, 257)[y_start:y_start + ny + 1],
            full_snapshot=np.asarray(full_snapshot),
            processing_complete=np.asarray(complete),
        )
        edges = np.linspace(-200.0, 200.0, 401)
        spectra = np.zeros((2, len(keys), 2, 400))
        spectra[0, 0, 1, 200] = 10.0
        spectra[1, 0, 1, 200] = 4.0
        spectra[:, 1, 0, 200] = 2.0
        np.savez_compressed(
            directory / "spectra.npz",
            velocity_edges_kms=edges,
            velocity_kms=(edges[:-1] + edges[1:]) / 2,
            dL_dv_erg_s_per_kms=spectra,
            projected_area_cm2=np.asarray(1e44),
            line_centroid_window_kms=np.zeros((2, len(keys))),
            line_sigma_window_kms=np.full((2, len(keys)), 3.0),
            line_sigma_full_kms=np.full((2, len(keys)), 9.0),
            line_centroid_window_by_regime_kms=np.zeros((2, len(keys), 2)),
            line_sigma_window_by_regime_kms=np.full((2, len(keys), 2), 4.0),
        )
        histogram = np.zeros((6, 400))
        histogram[0, 200] = 100.0
        histogram[1, 200] = 50.0
        histogram[5] = histogram[:5].sum(axis=0)
        np.savez_compressed(
            directory / "phase_velocity.npz",
            histogram_mass_g=histogram,
            sigma_about_global_mean_kms=np.asarray((1., 2., np.nan, np.nan, np.nan, 3.)),
            sigma_internal_kms=np.asarray((1., 2., np.nan, np.nan, np.nan, 3.)),
        )

    def test_loader_and_plotter_use_only_needed_saved_products(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.save_products(root)
            images, spectra = load_emission_products(root)
            phase = load_phase_velocity_products(root)
            self.assertEqual(images["line_luminosity_image_erg_s"].shape,
                             (2, len(LINE_ORDER), 256, 256))
            self.assertEqual(spectra["dL_dv_erg_s_per_kms"].shape,
                             (2, len(LINE_ORDER), 2, 400))
            self.assertEqual(phase["histogram_mass_g"].shape, (6, 400))
            paths = plot_emission_products(root, root / "plots", formats=("png",))
            self.assertEqual(len(paths["images"]), 19)
            self.assertEqual(len(paths["spectra"]), 10)
            self.assertEqual(len(paths["gas_phases"]), 10)
            self.assertIn(root / "plots" / "png" / "line_luminosity_hi21.png", paths["images"])
            self.assertNotIn(root / "plots" / "png" / "line_luminosity_hi21_attenuated.png", paths["images"])
            self.assertIn(root / "plots" / "png" / "gas_phase_spectrum_ciii_977.png", paths["gas_phases"])
            for path in paths["images"] + paths["spectra"] + paths["gas_phases"]:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 1000)

    def test_plot_only_binning_sums_native_pixel_luminosities(self):
        native = np.arange(1., 17.).reshape(1, 1, 4, 4)
        saved = {
            "line_luminosity_image_erg_s": native,
            "x_edges_kpc": np.arange(5.),
            "y_edges_kpc": np.arange(5.),
        }
        values, x_edges, y_edges = combine_image_pixels_for_display(saved, 1)
        np.testing.assert_array_equal(values, native)
        np.testing.assert_array_equal(x_edges, saved["x_edges_kpc"])
        np.testing.assert_array_equal(y_edges, saved["y_edges_kpc"])
        binned, x_edges, y_edges = combine_image_pixels_for_display(saved, 2)
        np.testing.assert_array_equal(binned[0, 0], [[14., 22.], [46., 54.]])
        np.testing.assert_array_equal(x_edges, [0., 2., 4.])
        np.testing.assert_array_equal(y_edges, [0., 2., 4.])
        self.assertEqual(binned.sum(), native.sum())
        np.testing.assert_array_equal(saved["line_luminosity_image_erg_s"], native)
        for factor, message in ((0, "positive integer"), (1.5, "positive integer"), (3, "divide both")):
            with self.subTest(factor=factor), self.assertRaisesRegex(ValueError, message):
                combine_image_pixels_for_display(saved, factor)

    def test_titled_phase_overlay_uses_attenuated_spectrum_and_saved_statistics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.save_products(root)
            with patch("quokka2s.figures.emission_results.plot_phase_spectrum_overlay") as overlay:
                plot_emission_products(root, root / "plots_titled", formats=("png",),
                                       titled=True)
            line_spectra = overlay.call_args.kwargs["line_spectra"]
            halpha = line_spectra.line_keys.index("halpha")
            self.assertEqual(line_spectra.dL_dv_erg_s_per_kms[halpha, 1, 200], 4.0)
            self.assertEqual(line_spectra.dL_dv_erg_s_per_kms[halpha, :, 200].sum(), 4.0)
            self.assertEqual(line_spectra.line_sigma_full_kms[halpha], 9.0)
            self.assertNotIn('line_sigma_window_kms', vars(line_spectra))
            self.assertEqual(overlay.call_args.kwargs["figure_style"], "full")

    def test_unneeded_npz_fields_are_ignored(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.save_products(root)
            for filename in ("images.npz", "spectra.npz", "phase_velocity.npz"):
                with np.load(root / filename, allow_pickle=False) as saved:
                    payload = {key: saved[key].copy() for key in saved.files}
                payload["unused_metadata"] = np.asarray(filename)
                np.savez_compressed(root / filename, **payload)
            images, spectra = load_emission_products(root)
            phase = load_phase_velocity_products(root)
            for payload in (images, spectra, phase):
                self.assertNotIn("unused_metadata", payload)

    def test_completed_region_uses_global_edges_without_partial_diagnostic_flag(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.save_products(
                root,
                full_snapshot=False,
                processing_complete=True,
                image_shape=(64, 32),
                image_xy_origin=(64, 80),
            )
            images, _ = load_emission_products(root)
            self.assertFalse(bool(images['full_snapshot']))
            self.assertTrue(bool(images['processing_complete']))
            np.testing.assert_array_equal(images['x_edges_kpc'], np.linspace(0., 1., 257)[64:129])
            np.testing.assert_array_equal(images['y_edges_kpc'], np.linspace(-.5, .5, 257)[80:113])
            with patch('quokka2s.figures.emission_results.plot_line_images', return_value=[]) as image_plot, \
                    patch('quokka2s.figures.emission_results.plot_line_spectra', return_value=[]) as spectrum_plot, \
                    patch('quokka2s.figures.emission_results.plot_gas_phase_comparisons', return_value=[]) as phase_plot:
                plot_emission_products(
                    products_dir=root,
                    output_dir=root / 'plots',
                    formats=('png',),
                )
            self.assertEqual(image_plot.call_args.kwargs['image_values'].shape, (2, 10, 64, 32))
            for plot in (image_plot, spectrum_plot, phase_plot):
                self.assertEqual(plot.call_args.kwargs['diagnostic_suffix'], '')

    def test_legacy_products_use_full_snapshot_as_completion_flag(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for full_snapshot in (True, False):
                with self.subTest(full_snapshot=full_snapshot):
                    self.save_products(root, full_snapshot=full_snapshot)
                    with np.load(root / 'images.npz', allow_pickle=False) as saved:
                        fields = {key: saved[key] for key in saved.files if key != 'processing_complete'}
                    np.savez_compressed(root / 'images.npz', **fields)
                    images, _ = load_emission_products(root)
                    self.assertEqual(bool(images['processing_complete']), full_snapshot)

    def test_partial_products_require_explicit_diagnostic_flag(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.save_products(root, full_snapshot=False)
            with self.assertRaisesRegex(ValueError, "allow_partial=True"):
                plot_emission_products(root, root / "plots", formats=("png",))
            paths = plot_emission_products(root, root / "plots", formats=("png",),
                                           allow_partial=True)
            self.assertTrue(all(path.stem.endswith("_partial_diagnostic")
                               for path in paths["images"] + paths["spectra"]))
            self.assertTrue(all("_partial_diagnostic_" in path.stem
                                for path in paths["gas_phases"]))
            config = root / "plot.yaml"
            config.write_text("products: .\noutput_dir: cli\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "allow_partial=True"):
                main(["--config", str(config)])

    def test_invalid_output_format_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.save_products(root)
            with self.assertRaisesRegex(ValueError, "png and/or pdf"):
                plot_emission_products(root, root / "plots", formats=("jpg",))


if __name__ == "__main__":
    unittest.main()
