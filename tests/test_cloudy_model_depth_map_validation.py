"""Independent retained-output gate: synthetic evidence, no Cloudy executable."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from quokka2s.tables.abundances import abundance_metadata
from scripts.build_cloudy_model_depth_bundle import AXIS_ORDER, LINES
from scripts.cloudy_model_depth_common import (
    ELEMENT_SYMBOLS, PC_IN_CM, common_commands, reference_log_abundances, sha256,
)
from scripts.validate_cloudy_model_depth_maps import validate_maps


class CloudyMapValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reference = self.root / "default.abn"
        self.reference.write_text("\n".join(f"{symbol} {value}" for symbol, value in
            zip(ELEMENT_SYMBOLS, [1., .1] + [1e-5] * 28)) + "\n")
        expected = reference_log_abundances(self.reference)
        composition = " ".join(f"{symbol}: {value:.4f}" for symbol, value in expected.items())
        self.parameter = self.root / "run.par"
        self.parameter.write_text("\n".join([
            "cloudyRunMode = 4", "coolingScaleFactor = 1", "coolingMapUseJeansLength = 0",
            *(f"lineMapLine = {item[1]}" for item in LINES),
            *(f"command {command}" for command in common_commands()),
            f"loop [radius 1e30 * linear] {10 * PC_IN_CM:.17g}",
        ]) + "\n")
        self.map = self.root / "map.dat"
        self.map.write_text("\n".join([
            "# hden 0", '# init "HM12_ATTENUATION_ISM_NH21/logNH20.out"',
            f"# radius 1e30 {10 * PC_IN_CM:.17g} linear",
            "#Te " + " ".join(item[2] for item in LINES),
            "4.000 " + " ".join(["-24"] * 7 + ["-99"]),
            "4.301 " + " ".join(["-24"] * 7 + ["-99"]),
        ]) + "\n")
        for suffix in (".cloudyIn", ".cloudyOut", ".lines", ".physical", ".radius", ".process"):
            self.map.with_suffix(suffix).write_text("")
        midpoint, width = .9 * 10 * PC_IN_CM, .2 * 10 * PC_IN_CM
        for temperature in (10000., 20000.):
            def section(suffix, label, text):
                with self.map.with_suffix(suffix).open("a") as handle:
                    handle.write(f"## {label} for T = {temperature:.3e}.\n" + text + "\n")
            section(".cloudyIn", "Input", "\n".join([
                *common_commands(), "hden 0", 'init "HM12_ATTENUATION_ISM_NH21/logNH20.out"',
                f"radius 1e30 {10 * PC_IN_CM:.17g} linear",
                f"constant temperature {temperature:.6f} K linear",
                'save last lines, emissivity "fake.lines.temp"',
                *(item[1] for item in LINES), "end of lines",
                'punch last physical conditions file = "fake.physical.temp"',
                'save last radius outer "fake.radius.temp"',
            ]))
            section(".cloudyOut", "Output", "Gas Phase Chemical Composition\n" + composition + "\n\n"
                    "Calculation stopped because outer radius reached. Iteration 3 of 3\n"
                    "Cloudy ends: 10 zones, 3 iterations. ExecTime(s) 1\nCloudy exited OK")
            section(".lines", "Line punch", "#depth\t" + "\t".join(item[1] for item in LINES) + "\n"
                    + "\t".join([str(midpoint)] + ["1e-24"] * 7 + ["0"]))
            section(".physical", "Physical conditions", f"{midpoint}\t{temperature}\t1\t.1\n")
            section(".radius", "Radius", f"10\t1e30\t{midpoint}\t{width}\n")
            with self.map.with_suffix(".process").open("a") as handle:
                handle.write(f"{temperature:.6e}\t0\n")
        self.manifest = dict(schema_version=1, cloudy_version="17.02", axis_order=AXIS_ORDER,
            status="completed", process_returncode=0, pc_in_cm=PC_IN_CM, abundance=abundance_metadata(),
            parameter_file="run.par", axes=dict(log_NH_attenuation=[20.], log_nH=[0.],
            log_T=[4., float(np.log10(20000))], log_L_model_pc=[1.]),
            maps=[dict(path="map.dat", log_NH_attenuation=20., log_nH=0., log_L_model_pc=1.)],
            provenance=dict(default_abn={"path": "default.abn", "sha256": sha256(self.reference)}))
        self.manifest_path = self.root / "manifest.json"
        self.report = self.root / "report.json"

    def validate(self, **kwargs):
        self.manifest_path.write_text(json.dumps(self.manifest))
        return validate_maps(self.manifest_path, self.report, **kwargs)

    def change(self, suffix, old, new):
        path = self.map.with_suffix(suffix)
        path.write_text(path.read_text().replace(old, new))

    def test_complete_evidence_passes_with_true_zero_and_hashes(self):
        digest = sha256(self.map)
        report = self.validate()
        self.assertTrue(report["passed"], report["states"])
        self.assertEqual(report["valid_state_count"], 2)
        self.assertEqual(report["maps"][0]["hashes"][".dat"], digest)
        self.assertFalse(np.asarray(report["diagnostic_failure_mask"]).any())
        self.assertFalse(np.asarray(report["raw_map_failure_mask"]).any())
        self.assertEqual(sha256(self.map), digest)
        self.assertEqual(report["states"][0]["physical_checks"]["emissivity_per_nH2"][-1], 0.)

    def test_early_crash_has_explicit_failures_without_fabricated_values(self):
        self.change(".dat", "4.301 " + " ".join(["-24"] * 7 + ["-99"]), "4.301")
        self.change(".process", "2.000000e+04\t0", "2.000000e+04\t1792")
        for suffix in (".lines", ".physical", ".radius"):
            path = self.map.with_suffix(suffix)
            text = path.read_text()
            path.write_text(text[:text.find("##", 2)])
        report = self.validate()
        self.assertEqual(report["valid_state_count"], 1)
        state = report["states"][1]
        self.assertTrue(state["raw_map_crash_row"])
        self.assertTrue(all(state["raw_line_failure_mask"]))
        self.assertEqual(state["process"]["exit_code"], 7)
        self.assertNotIn("emissivity_per_nH2", state["physical_checks"])

    def test_missing_process_cannot_pass_success_footer(self):
        self.map.with_suffix(".process").unlink()
        report = self.validate()
        self.assertEqual(report["valid_state_count"], 0)
        self.assertFalse(report["states"][0]["process"]["available"])

    def test_signal_is_decoded_and_rejected(self):
        self.change(".process", "2.000000e+04\t0", "2.000000e+04\t9")
        report = self.validate()
        self.assertEqual(report["states"][1]["process"]["signal"], 9)
        self.assertFalse(report["states"][1]["valid"])

    def test_duplicate_temperature_section_cannot_pass(self):
        self.change(".physical", "2.000e+04", "1.000e+04")
        report = self.validate()
        self.assertFalse(report["passed"])
        self.assertIn("duplicate temperature section", " ".join(report["states"][0]["issues"]))

    def test_offgrid_temperature_section_cannot_pass(self):
        self.change(".radius", "2.000e+04", "3.000e+04")
        self.assertFalse(self.validate()["passed"])

    def test_wrong_input_geometry_is_rejected_even_if_saved_geometry_matches(self):
        self.change(".cloudyIn", f"radius 1e30 {10 * PC_IN_CM:.17g}", f"radius 1e30 {PC_IN_CM:.17g}")
        report = self.validate()
        self.assertFalse(report["passed"])
        self.assertIn("input fixed model depth differs", " ".join(report["states"][0]["issues"]))

    def test_wrong_saved_geometry_is_rejected_even_if_input_matches(self):
        self.change(".radius", str(.9 * 10 * PC_IN_CM), str(.8 * 10 * PC_IN_CM))
        self.assertFalse(self.validate()["passed"])

    def test_wrong_sed_and_temperature_input_are_rejected(self):
        self.change(".cloudyIn", "logNH20.out", "logNH19.out")
        self.change(".cloudyIn", "20000.000000 K", "15000.000000 K")
        self.assertFalse(self.validate()["passed"])

    def test_nonzero_local_convergence_failures_are_rejected(self):
        self.change(".cloudyOut", "3 iterations.", "3 iterations. Failures: 0 thermal, 0 pressure, 1 ionization, 0 electron density.")
        self.assertFalse(self.validate()["passed"])

    def test_raw_map_saved_coefficient_disagreement_rejected(self):
        self.change(".dat", "-24", "-23")
        report = self.validate()
        self.assertFalse(report["passed"])
        self.assertAlmostEqual(report["states"][0]["map_saved_emission_max_error_dex"], 1.)

    def test_zero_sentinel_disagreement_rejected(self):
        self.change(".dat", "-99", "-95")
        self.assertFalse(self.validate()["passed"])

    def test_saved_line_order_disagreement_rejected(self):
        self.change(".lines", "H  1 6562.81A", "H  1 6000A")
        self.assertFalse(self.validate()["passed"])

    def test_abundance_reference_hash_mismatch_fails_all_states(self):
        self.manifest["provenance"]["default_abn"]["sha256"] = "0" * 64
        self.assertFalse(self.validate()["passed"])

    def test_explicit_reference_required_without_declared_path(self):
        self.manifest["provenance"] = {}
        with self.assertRaisesRegex(ValueError, "Provide --default-abn"):
            self.validate()
        self.assertTrue(self.validate(default_abn=self.reference)["passed"])

    def test_prepared_manifest_rejected(self):
        self.manifest["status"] = "prepared"
        with self.assertRaisesRegex(ValueError, "execution must be completed"):
            self.validate()

    def test_missing_map_temperature_is_retained_as_failed_state(self):
        text = self.map.read_text()
        self.map.write_text("\n".join(text.splitlines()[:-1]) + "\n")
        report = self.validate()
        self.assertEqual(report["state_count"], 2)
        self.assertEqual(report["valid_state_count"], 1)


if __name__ == "__main__":
    unittest.main()
