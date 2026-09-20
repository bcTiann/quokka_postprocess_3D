"""Jeans density-conversion provenance; no Cloudy calculation or table build."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import build_hm12_filtered_ism_sixline_bundles as bundles


VALUE_KEY = "jeans_length_hydrogen_mass_fraction"
SOURCE_KEY = VALUE_KEY + "_source"


class CloudyJeansBundleProvenanceTests(unittest.TestCase):
    def read_metadata(self, contents):
        with tempfile.TemporaryDirectory() as temporary:
            parameter = Path(temporary) / "test.par"
            parameter.write_text(contents)
            return bundles._jeans_mass_fraction_metadata(parameter)

    def test_absent_parameter_does_not_assign_a_new_default_to_old_runs(self):
        metadata = self.read_metadata(
            "coolingMapUseJeansLength = 1\n"
            "# coolingMapHydrogenMassFraction = 0.7157683773530885\n"
        )
        self.assertNotIn(VALUE_KEY, metadata)
        self.assertIn("runtime default not inferred", metadata[SOURCE_KEY])

    def test_explicit_fraction_is_recorded_without_substitution(self):
        for value in (0.7157683773530885, 0.76, 1.0):
            with self.subTest(value=value):
                metadata = self.read_metadata(
                    f"coolingMapHydrogenMassFraction = {value}\n"
                )
                self.assertEqual(metadata[VALUE_KEY], value)
                self.assertIn("explicit", metadata[SOURCE_KEY])

    def test_parser_supports_comments_and_last_assignment_wins(self):
        metadata = self.read_metadata(
            "coolingMapHydrogenMassFraction = 0.7157683773530885\n"
            "coolingmaphydrogenmassfraction +7.6e-1 # historical diagnostic\n"
        )
        self.assertEqual(metadata[VALUE_KEY], 0.76)
        self.assertIn("line 2", metadata[SOURCE_KEY])

    def test_invalid_explicit_values_are_rejected(self):
        for value in (
            "", "0", "-1", "1.001", "NaN", "inf", "1e999", "1e-999",
            "wrong", "0.76garbage",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite number with 0 < X_H <= 1"):
                    self.read_metadata(f"coolingMapHydrogenMassFraction = {value}\n")

    def test_metadata_is_forwarded_to_bundle_and_report_without_writing_a_table(self):
        # Intercept the NPZ writer; the synthetic data only tests the metadata path.
        for value in (None, 0.7157683773530885, 0.76):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                parameter = directory / "test.par"
                parameter.write_text(
                    "" if value is None else f"coolingMapHydrogenMassFraction = {value}\n"
                )
                attenuation = np.array([18.0, 21.0])
                density = np.linspace(-5.0, 6.0, 10)
                raw = np.full((8, 2, 10, 21), -24.0)
                argv = [
                    "bundle_test", "--stem", "test", "--parameter-file", str(parameter),
                    "--output-dir", str(directory), "--hm12-log-nh", "18", "21",
                ]
                with (
                    patch("sys.argv", argv),
                    patch.object(bundles, "_load_grid", return_value=(attenuation, density, raw)),
                    patch.object(bundles.np, "savez_compressed") as save,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    bundles.main()
                payload = save.call_args.kwargs
                report = json.loads((directory / "cloudy_test_failure_nodes.json").read_text())
                if value is None:
                    self.assertNotIn(VALUE_KEY, payload)
                    self.assertNotIn(VALUE_KEY, report)
                else:
                    self.assertEqual(float(payload[VALUE_KEY]), value)
                    self.assertEqual(report[VALUE_KEY], value)
                self.assertEqual(str(payload[SOURCE_KEY]), report[SOURCE_KEY])
                self.assertEqual(float(payload["jeans_length_cap_cm"]), bundles.JEANS_CAP_CM)
                np.testing.assert_array_equal(payload["log_nH"], density)
                np.testing.assert_array_equal(payload["log_NH_attenuation"], attenuation)
                np.testing.assert_array_equal(payload["log_emissivity_per_nH2"], raw)
                self.assertFalse(list(directory.glob("*.npz")))


if __name__ == "__main__":
    unittest.main()
