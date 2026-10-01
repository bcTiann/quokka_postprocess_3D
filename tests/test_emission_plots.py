"""Portable plotting stage: saved arrays are the only data inputs."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from quokka2s.emission_plots import load_emission_products, main, plot_emission_products


class EmissionPlotTests(unittest.TestCase):
    @staticmethod
    def save_products(directory: Path, *, mismatch: bool = False,
                      wrong_manifest: bool = False,
                      wrong_spectrum_total: bool = False,
                      wrong_image_total: bool = False,
                      wrong_velocity_range: bool = False,
                      full_snapshot: bool = True) -> None:
        keys = np.asarray(("halpha", "hi21"))
        variants = np.asarray(("intrinsic", "attenuated"))
        images = np.zeros((2, 2, 128, 128))
        images[:, 0, 10, 20] = (10.0, 4.0)
        images[:, 1, 40, 30] = 2.0
        image_total = images.sum(axis=(-2, -1))
        if wrong_image_total:
            image_total[0, 0] += 1.0
        np.savez_compressed(
            directory / "images.npz", line_keys=keys, variant_keys=variants,
            line_luminosity_image_erg_s=images,
            total_luminosity_erg_s=image_total,
            x_edges_kpc=np.linspace(0.0, 1.0, 129),
            y_edges_kpc=np.linspace(-0.5, 0.5, 129),
            source_manifest_sha256=np.asarray("a" * 64),
            full_snapshot=np.asarray(full_snapshot),
        )
        edges = np.linspace(-190.0 if wrong_velocity_range else -200.0, 200.0, 401)
        spectra = np.zeros((2, 2, 2, 400))
        spectra[0, 0, 1, 200] = 10.0
        spectra[1, 0, 1, 200] = 4.0
        spectra[:, 1, 0, 200] = 2.0
        input_luminosity = np.zeros((2, 2, 2))
        input_luminosity[:, 0, 1] = (10.0, 4.0)
        input_luminosity[:, 1, 0] = 2.0
        if wrong_spectrum_total:
            input_luminosity[0, 0, 1] += 1.0
        np.savez_compressed(
            directory / "spectra.npz",
            line_keys=keys[::-1] if mismatch else keys, variant_keys=variants,
            regime_keys=np.asarray(("T_QUOKKA_lt_3000K", "T_QUOKKA_ge_3000K")),
            velocity_edges_kms=edges,
            velocity_kms=(edges[:-1] + edges[1:]) / 2,
            dL_dv_erg_s_per_kms=spectra, projected_area_cm2=np.asarray(1e44),
            input_luminosity_erg_s=input_luminosity,
            source_manifest_sha256=np.asarray(("b" if wrong_manifest else "a") * 64),
            full_snapshot=np.asarray(full_snapshot),
        )

    def test_loader_and_plotter_use_only_saved_products(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.save_products(root)
            images, spectra = load_emission_products(root)
            self.assertEqual(images["line_luminosity_image_erg_s"].shape, (2, 2, 128, 128))
            self.assertEqual(spectra["dL_dv_erg_s_per_kms"].shape, (2, 2, 2, 400))
            paths = plot_emission_products(root, root / "plots", formats=("png",))
            self.assertEqual(len(paths["images"]), 3)
            self.assertEqual(len(paths["spectra"]), 2)
            self.assertIn(root / "plots" / "line_luminosity_hi21.png", paths["images"])
            self.assertNotIn(root / "plots" / "line_luminosity_hi21_attenuated.png", paths["images"])
            for path in paths["images"] + paths["spectra"]:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 1000)

    def test_mismatched_line_order_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.save_products(root, mismatch=True)
            with self.assertRaisesRegex(ValueError, "same order"):
                load_emission_products(root)

    def test_mismatched_provenance_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.save_products(root, wrong_manifest=True)
            with self.assertRaisesRegex(ValueError, "different provenance"):
                load_emission_products(root)

    def test_mismatched_luminosity_is_rejected(self):
        for option, message in (("wrong_spectrum_total", "input luminosities differ"),
                                ("wrong_image_total", "Image pixels do not sum")):
            with self.subTest(option=option), TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.save_products(root, **{option: True})
                with self.assertRaisesRegex(ValueError, message):
                    load_emission_products(root)

    def test_wrong_velocity_range_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.save_products(root, wrong_velocity_range=True)
            with self.assertRaisesRegex(ValueError, "exactly -200 to 200"):
                load_emission_products(root)

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
            config = root / "plot.yaml"
            config.write_text("products: .\noutput_dir: cli\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "allow_partial=True"):
                main(["--config", str(config)])


if __name__ == "__main__":
    unittest.main()
