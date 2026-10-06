"""Configuration parsing for the two-stage emission workflow."""
from pathlib import Path
import tempfile
import unittest

import yaml

from quokka2s.physics.dust_attenuation import DEFAULT_DRAINE_TABLE
from quokka2s.run_settings import load_plot_config, load_process_config


PROCESS_MINIMUM = {
    "dataset": "inputs/snapshots/plt0655228",
    "despotic_table": "inputs/tables/despotic/interpolated.npz",
    "cloudy_table": "tables/cloudy.npz",
    "output_dir": "results/processed",
}


class RunSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.config = self.directory / "workflow.yaml"

    def write(self, values):
        self.config.write_text(yaml.safe_dump(values), encoding="utf-8")
        return self.config

    def test_process_minimum_resolves_paths_and_uses_defaults(self):
        config = load_process_config(self.write(PROCESS_MINIMUM))
        self.assertEqual(config.dataset, self.directory / "inputs/snapshots/plt0655228")
        self.assertEqual(config.despotic_table, self.directory / "inputs/tables/despotic/interpolated.npz")
        self.assertEqual(config.cloudy_table, self.directory / "tables/cloudy.npz")
        self.assertEqual(config.output_dir, self.directory / "results/processed")
        self.assertEqual(config.dust_opacity_table, DEFAULT_DRAINE_TABLE.resolve())
        self.assertEqual((config.slab_nx, config.query_chunk, config.spectral_workers),
                         (8, 100000, 6))
        self.assertEqual(config.chunk_workers, 1)
        self.assertIsNone(config.max_slabs)

    def test_process_optional_settings_and_absolute_path(self):
        values = {
            **PROCESS_MINIMUM, "dataset": str(self.directory / "other/plt"),
            "despotic_table": "tables/interpolated.npz",
            "dust_opacity_table": "tables/dust.all",
            "slab_nx": 4, "query_chunk": 1000000, "spectral_workers": 2,
            "max_slabs": 1, "chunk_workers": 2,
        }
        config = load_process_config(self.write(values))
        self.assertEqual(config.dataset, self.directory / "other/plt")
        self.assertEqual(config.despotic_table, self.directory / "tables/interpolated.npz")
        self.assertEqual(config.dust_opacity_table, self.directory / "tables/dust.all")
        self.assertEqual((config.slab_nx, config.query_chunk, config.spectral_workers, config.max_slabs),
                         (4, 1000000, 2, 1))
        self.assertEqual(config.chunk_workers, 2)

    def test_plot_minimum_and_optional_flags(self):
        config = load_plot_config(self.write({"products": "results", "output_dir": "figures"}))
        self.assertEqual(config.products, self.directory / "results")
        self.assertEqual(config.output_dir, self.directory / "figures")
        self.assertIsNone(config.titled_output_dir)
        self.assertFalse(config.raw_luminosity)
        self.assertFalse(config.allow_partial)
        self.assertEqual(config.image_downsample_factor, 1)
        config = load_plot_config(self.write({
            "products": "results", "output_dir": "figures",
            "raw_luminosity": True, "allow_partial": True,
            "image_downsample_factor": 2,
            "titled_output_dir": "figures_titled",
        }))
        self.assertTrue(config.raw_luminosity)
        self.assertTrue(config.allow_partial)
        self.assertEqual(config.image_downsample_factor, 2)
        self.assertEqual(config.titled_output_dir, self.directory / "figures_titled")

    def test_missing_unknown_and_wrong_document_shape_fail(self):
        cases = [
            ({"dataset": "plt"}, "Missing setting"),
            ({**PROCESS_MINIMUM, "mystery": 1}, "Unknown setting"),
            (["not", "a", "mapping"], "YAML mapping"),
            ({**PROCESS_MINIMUM, 1: "bad"}, "string setting names"),
        ]
        for values, message in cases:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    load_process_config(self.write(values))

    def test_wrong_path_and_numeric_types_fail(self):
        cases = [
            ({**PROCESS_MINIMUM, "dataset": ""}, "dataset"),
            ({**PROCESS_MINIMUM, "dataset": True}, "dataset"),
            ({**PROCESS_MINIMUM, "slab_nx": True}, "slab_nx"),
            ({**PROCESS_MINIMUM, "query_chunk": 1000001}, "query_chunk"),
            ({**PROCESS_MINIMUM, "spectral_workers": 0}, "spectral_workers"),
            ({**PROCESS_MINIMUM, "chunk_workers": 0}, "chunk_workers"),
            ({**PROCESS_MINIMUM, "chunk_workers": True}, "chunk_workers"),
            ({**PROCESS_MINIMUM, "max_slabs": 1.5}, "max_slabs"),
            ({**PROCESS_MINIMUM, "dust_opacity_table": None}, "dust_opacity_table"),
        ]
        for values, message in cases:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    load_process_config(self.write(values))

    def test_plot_rejects_non_boolean_flags(self):
        for value in (0, 1, "yes", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "raw_luminosity"):
                    load_plot_config(self.write({
                        "products": "results", "output_dir": "figures",
                        "raw_luminosity": value,
                    }))

    def test_plot_rejects_invalid_downsample_factor(self):
        for value in (0, -1, 1.5, True, "2"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "image_downsample_factor"):
                    load_plot_config(self.write({
                        "products": "results", "output_dir": "figures",
                        "image_downsample_factor": value,
                    }))

    def test_no_environment_expansion(self):
        config = load_plot_config(self.write({
            "products": "$HOME/results", "output_dir": "figures",
        }))
        self.assertEqual(config.products, self.directory / "$HOME/results")

    def test_invalid_yaml_fails_as_value_error(self):
        self.config.write_text("products: [unclosed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid YAML"):
            load_plot_config(self.config)


if __name__ == "__main__":
    unittest.main()
