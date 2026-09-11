"""Coordinate and failure semantics for the fixed-depth Cloudy packer."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from quokka2s.tables.abundances import abundance_metadata
from scripts.build_cloudy_model_depth_bundle import AXIS_ORDER, LINES, pack_table


class CloudyModelDepthBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pc = 3.0856775809623245e18
        self.axes = {"log_NH_attenuation": [18.0, 20.0], "log_nH": [-1.0, 2.0],
                     "log_T": [2.123456789, 4.456789012], "log_L_model_pc": [-0.25, 2.0]}
        composition = abundance_metadata()
        self.parameter = self.root / "run.par"
        self.parameter.write_text("\n".join([
            "cloudyRunMode = 4", "coolingScaleFactor = 1", "coolingMapUseJeansLength = 0",
            *(f"lineMapLine = {item[1]}" for item in LINES),
            'command abundances "default.abn"',
            f'command element helium abundance {composition["gow_elemental_abundances"]["xHe"]:.17g} linear',
            f'command metals {composition["metal_reference_scale"]:.17g} linear',
            "loop [radius 1e30 * linear] " + " ".join(f"{self.pc * 10**value:.17g}" for value in self.axes["log_L_model_pc"]),
        ]) + "\n")
        self.manifest = {
            "schema_version": 1, "cloudy_version": "17.02", "axis_order": AXIS_ORDER,
            "status": "completed", "axes": self.axes, "pc_in_cm": self.pc,
            "parameter_file": "run.par", "abundance": composition, "maps": [],
            "radiation": {"source": "synthetic fixture"},
        }
        for attenuation in self.axes["log_NH_attenuation"]:
            for density in self.axes["log_nH"]:
                for depth in self.axes["log_L_model_pc"]:
                    name = f"map_{len(self.manifest['maps'])}.dat"
                    record = {"path": name, "log_NH_attenuation": attenuation,
                              "log_nH": density, "log_L_model_pc": depth}
                    self.manifest["maps"].append(record)
                    rows = [
                        "# Line Emissivity Map File", f"# hden {density:.17g}",
                        f'# init "sed/logNH{attenuation:g}.out"',
                        f"# radius 1e30 {self.pc * 10**depth:.17g} linear",
                        "#Te " + " ".join(item[2] for item in LINES),
                    ]
                    for temperature in self.axes["log_T"]:
                        # Each axis has an independently recognizable contribution.
                        value = -70 + attenuation + density + temperature + depth
                        rows.append(f"{temperature:.3f} " + " ".join(
                            f"{value + i * 0.01:.12g}" for i in range(len(LINES))))
                    (self.root / name).write_text("\n".join(rows) + "\n")
        self.manifest_path = self.root / "manifest.json"
        self.output = self.root / "bundle.npz"

    def pack(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        return pack_table(self.manifest_path, self.output)

    def rewrite_map(self, transform, map_index=0):
        path = self.root / self.manifest["maps"][map_index]["path"]
        path.write_text(transform(path.read_text()))

    def test_exact_axes_survive_rounding_and_map_order_is_irrelevant(self):
        self.manifest["maps"].reverse()
        summary = self.pack()
        self.assertEqual(summary["shape"], [8, 2, 2, 2, 2])
        self.assertEqual(summary["union_failure_nodes"], 0)
        with np.load(self.output, allow_pickle=False) as table:
            self.assertEqual(table["schema_version"].item(), 4)
            self.assertEqual(table["axis_order"].item(), AXIS_ORDER)
            for name, values in self.axes.items():
                np.testing.assert_array_equal(table[name], values)
            expected = -70 + 20 + 2 + self.axes["log_T"][1] - 0.25 + 0.03
            self.assertAlmostEqual(table["log_emissivity_per_nH2"][3, 1, 1, 1, 0], expected, places=9)
            provenance = json.loads(table["provenance_json"].item())
            self.assertEqual(len(provenance["maps"]), 8)
            self.assertEqual(len(provenance["parameter_file_sha256"]), 64)
            self.assertFalse(provenance["validation"]["cloudy_convergence_independently_verified"])
            self.assertIn("deepest-zone", table["normalization"].item())
        report = json.loads(Path(summary["report_path"]).read_text())
        self.assertEqual(report["product_sha256"], summary["product_sha256"])

    def test_crash_and_true_zero_remain_distinct(self):
        def change(text):
            rows = text.splitlines()
            rows[-2] = rows[-2].split()[0]
            parts = rows[-1].split()
            parts[1], parts[2], parts[3] = "-99", "-95", "nan"
            rows[-1] = " ".join(parts)
            return "\n".join(rows) + "\n"
        self.rewrite_map(change)
        summary = self.pack()
        self.assertEqual(summary["union_failure_nodes"], 2)
        self.assertEqual(summary["true_zero_line_values"], 1)
        with np.load(self.output) as table:
            self.assertTrue(table["failure_mask"][:, 0, 0, 0, 0].all())
            self.assertTrue(np.isnan(table["log_emissivity_per_nH2"][:, 0, 0, 0, 0]).all())
            self.assertFalse(table["zero_mask"][:, 0, 0, 0, 0].any())
            self.assertTrue(table["zero_mask"][0, 0, 0, 1, 0])
            self.assertFalse(table["failure_mask"][0, 0, 0, 1, 0])
            self.assertAlmostEqual(table["emissivity_per_nH2"][1, 0, 0, 1, 0] / 1e-95, 1)
            self.assertTrue(table["failure_mask"][2, 0, 0, 1, 0])
            self.assertTrue((table["emissivity_per_nH2"][table["failure_mask"]] == 0).all())
            self.assertFalse(table["interpolated_mask"].any())

    def test_wrong_hden_header_rejected(self):
        self.rewrite_map(lambda text: text.replace("# hden -1", "# hden 2"))
        with self.assertRaisesRegex(ValueError, "log_nH mismatch"):
            self.pack()

    def test_swapped_manifest_coordinates_rejected(self):
        self.manifest["maps"][0]["path"], self.manifest["maps"][1]["path"] = (
            self.manifest["maps"][1]["path"], self.manifest["maps"][0]["path"])
        with self.assertRaisesRegex(ValueError, "log_L_model_pc mismatch"):
            self.pack()

    def test_wrong_attenuation_header_rejected(self):
        self.rewrite_map(lambda text: text.replace("logNH18.out", "logNH19.out"))
        with self.assertRaisesRegex(ValueError, "log_NH_attenuation mismatch"):
            self.pack()

    def test_duplicate_map_coordinates_rejected(self):
        self.manifest["maps"][1] = {**self.manifest["maps"][0], "path": "duplicate.dat"}
        (self.root / "duplicate.dat").write_text((self.root / "map_0.dat").read_text())
        with self.assertRaisesRegex(ValueError, "duplicate map coordinates"):
            self.pack()

    def test_duplicate_map_path_rejected(self):
        self.manifest["maps"][1]["path"] = self.manifest["maps"][0]["path"]
        with self.assertRaisesRegex(ValueError, "duplicate map paths"):
            self.pack()

    def test_missing_map_rejected(self):
        self.manifest["maps"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete map list"):
            self.pack()

    def test_wrong_radius_header_rejected(self):
        self.rewrite_map(lambda text: text.replace("# radius 1e30", "# radius 1e29"))
        with self.assertRaisesRegex(ValueError, "unexpected radius header"):
            self.pack()

    def test_depth_units_checked_against_manifest(self):
        self.manifest["pc_in_cm"] *= 1.01
        with self.assertRaisesRegex(ValueError, "radius loop length values differ"):
            self.pack()

    def test_active_map_marker_rejected(self):
        (self.root / "run.mach").touch()
        with self.assertRaisesRegex(RuntimeError, "mach marker"):
            self.pack()

    def test_incomplete_temperature_rows_rejected(self):
        self.rewrite_map(lambda text: "\n".join(text.splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "incomplete temperatures"):
            self.pack()

    def test_off_grid_temperature_rejected(self):
        self.rewrite_map(lambda text: text.replace("2.123 ", "2.120 "))
        with self.assertRaisesRegex(ValueError, "off-grid or ambiguous temperature"):
            self.pack()

    def test_near_duplicate_temperature_rows_rejected(self):
        self.rewrite_map(lambda text: text.replace("4.457 ", "2.1234 "))
        with self.assertRaisesRegex(ValueError, "multiple rows map to the same temperature"):
            self.pack()

    def test_conflicting_jeans_setting_rejected(self):
        self.parameter.write_text(self.parameter.read_text().replace("coolingMapUseJeansLength = 0", "coolingMapUseJeansLength = 1"))
        with self.assertRaisesRegex(ValueError, "coolingMapUseJeansLength"):
            self.pack()

    def test_wrong_abundance_command_rejected(self):
        self.parameter.write_text(self.parameter.read_text().replace("command metals 1.5242485583008916 linear", "command metals 1 linear"))
        with self.assertRaisesRegex(ValueError, "metals command differs"):
            self.pack()

    def test_later_abundance_reset_rejected(self):
        self.parameter.write_text(self.parameter.read_text() + 'command abundances "default.abn"\n')
        with self.assertRaisesRegex(ValueError, "exactly default.abn"):
            self.pack()

    def test_manifest_abundance_mismatch_rejected(self):
        self.manifest["abundance"]["mass_fractions"]["Z"] = 0.01
        with self.assertRaisesRegex(ValueError, "manifest abundance differs"):
            self.pack()

    def test_running_manifest_rejected_even_if_all_maps_exist(self):
        self.manifest["status"] = "running"
        with self.assertRaisesRegex(ValueError, "status must be completed"):
            self.pack()

    def test_existing_output_is_preserved(self):
        self.output.write_bytes(b"existing product")
        with self.assertRaises(FileExistsError):
            self.pack()
        self.assertEqual(self.output.read_bytes(), b"existing product")


if __name__ == "__main__":
    unittest.main()
