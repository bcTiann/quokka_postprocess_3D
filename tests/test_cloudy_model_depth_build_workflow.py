"""Preparation, launch guards, and real packer compatibility without Cloudy jobs."""

import json
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import build_cloudy_model_depth_tables as builder
from scripts.build_cloudy_model_depth_bundle import LINES as PACK_LINES, pack_table
from scripts.cloudy_model_depth_common import LOG_DEPTH_PC, LOG_T, PC_IN_CM, common_commands, sha256


class CloudyModelDepthBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "build"
        self.executable = self.root / "cloudy.exe"
        self.executable.write_text("#!/bin/sh\nexit 99\n")
        self.executable.chmod(0o755)
        self.sed = self.root / "source_seds"
        self.sed.mkdir()
        entries = []
        for column in builder.HM12_LOG_NH:
            name = f"logNH{column:.17g}"
            normalization = "f(nu) = -17.9 at 0.5 Ryd"
            (self.sed / f"{name}.out").write_text(
                f'table SED "{builder.SED_DIRECTORY_NAME}/{name}.sed"\n{normalization}\n')
            (self.sed / f"{name}.sed").write_text("# synthetic unit-test spectrum\n0.1 1e-20\n1 1e-20\n")
            entries.append(dict(hm12_log_NH_attenuation=column,
                                fnu_normalization_command=normalization,
                                roundtrip=dict(maximum_relevant_absolute_error_dex=1e-4)))
        report = dict(hm12_log_NH_attenuation=list(builder.HM12_LOG_NH),
                      ism_log_NH_attenuation=21.0,
                      extinguish_leak=0.0, external_grackle_hm12_used=False,
                      energy_mesh_identical_for_all_exports=True,
                      roundtrip_maximum_allowed_error_dex=0.001, entries=entries)
        (self.sed / "build_report.json").write_text(json.dumps(report))

    def prepare(self):
        return builder.prepare_build(output_dir=self.output, cloudy_exe=self.executable,
                                     sed_dir=self.sed, workers=3)

    def read_manifest(self):
        return json.loads((self.output / builder.MANIFEST_NAME).read_text())

    def write_maps(self, *, reverse=False, crash=False, depth_points=10):
        expected = builder.expected_maps(depth_points)
        coordinates = list(reversed(expected)) if reverse else expected
        for number, (file_record, coordinate) in enumerate(zip(expected, coordinates)):
            path = self.output / file_record["path"]
            density, column, depth = (coordinate[name] for name in
                                      ("log_nH", "log_NH_attenuation", "log_L_model_pc"))
            lines = [f"# hden {density:.17g}",
                     f'# init "{builder.SED_DIRECTORY_NAME}/logNH{column:.17g}.out"',
                     f"# radius 1e30 {PC_IN_CM * 10.0**depth:.17g} linear",
                     "#Te " + " ".join(line[2] for line in PACK_LINES)]
            for k, temperature in enumerate(LOG_T):
                if crash and number == 0 and k == 0:
                    lines.append(f"{temperature:.3f}")
                else:
                    lines.append(f"{temperature:.3f} " + " ".join(
                        f"{-20.0 + 0.1*density + 0.01*column + 0.2*depth + q:.4f}"
                        for q in range(len(PACK_LINES))))
            path.write_text("\n".join(lines) + "\n")

    def test_prepare_never_launches_and_preserves_exact_physical_inputs(self):
        with patch.object(builder.subprocess, "Popen") as launch:
            manifest = self.prepare()
        launch.assert_not_called()
        self.assertEqual(manifest["status"], "prepared")
        self.assertEqual(manifest["expected_map_count"], 700)
        self.assertEqual(manifest["expected_state_count"], 14700)
        self.assertEqual(manifest["axis_order"], "line,log_NH_attenuation,log_nH,log_T,log_L_model_pc")
        self.assertEqual(manifest["axes"]["log_L_model_pc"], LOG_DEPTH_PC.tolist())
        text = (self.output / "fixed_depth.par").read_text()
        for value in ("coolingMapUseJeansLength = 0", "saveCloudyOutputFiles = 1",
                      "exitOnCrash = 0", "coolingMapTpoints = 21"):
            self.assertIn(value, text)
        for command in common_commands():
            self.assertIn(f"command {command}\n", text)
        self.assertEqual(text.count("lineMapLine = "), 8)
        radius_line = next(line for line in text.splitlines() if line.startswith("loop [radius"))
        self.assertEqual(radius_line, "loop [radius 1e30 * linear] " + " ".join(
            f"{PC_IN_CM * 10.0**value:.17g}" for value in LOG_DEPTH_PC))
        density_line = next(line for line in text.splitlines() if line.startswith("loop [hden"))
        self.assertEqual(density_line, "loop [hden] " + " ".join(
            f"{value:.17g}" for value in builder.LOG_NH_DENSITY))
        self.assertEqual(manifest["command"][-3:], ["-np", "3", "fixed_depth.par"])
        for item in manifest["provenance"]["sed_files"]:
            self.assertEqual(sha256(Path(item["source"])), item["sha256"])
            self.assertEqual((self.output / item["path"]).read_bytes(), Path(item["source"]).read_bytes())
        self.assertFalse(manifest["validation"]["cloudy_convergence_independently_verified"])
        self.assertFalse(list((self.output / "raw_maps").iterdir()))

    def test_predicted_run_order_follows_density_then_init_then_radius(self):
        records = builder.expected_maps()
        self.assertEqual(records[0]["log_L_model_pc"], -0.25)
        self.assertEqual(records[9]["log_L_model_pc"], 2.0)
        self.assertEqual(records[10]["log_NH_attenuation"], 18.5)
        self.assertEqual(records[10]["log_L_model_pc"], -0.25)
        self.assertEqual(records[70]["log_nH"], builder.LOG_NH_DENSITY[1])
        self.assertEqual(records[-1]["log_nH"], 6.0)
        self.assertEqual(records[-1]["log_NH_attenuation"], 21.0)
        self.assertEqual(records[-1]["log_L_model_pc"], 2.0)

    def test_refined_grid_preparation_launch_guard_and_packing(self):
        with patch.object(builder.subprocess, "Popen") as launch:
            manifest = builder.prepare_build(output_dir=self.output, cloudy_exe=self.executable,
                                             sed_dir=self.sed, workers=3, depth_points=37)
        launch.assert_not_called()
        self.assertEqual(manifest["expected_state_count"], 54390)
        self.assertEqual(manifest["expected_map_count"], 2590)
        np.testing.assert_array_equal(manifest["axes"]["log_L_model_pc"],
                                      np.arange(-0.25, 2.001, 0.0625))
        text = (self.output / "fixed_depth.par").read_text().lower()
        self.assertNotIn("drmax", text)
        self.assertNotIn("drmin", text)
        builder._verify_prepared(self.output, manifest)
        with patch.object(builder.subprocess, "Popen") as launch:
            process = launch.return_value
            process.pid = 123
            def complete():
                self.write_maps(reverse=True, crash=True, depth_points=37)
                return 0
            process.wait.side_effect = complete
            process.poll.return_value = 0
            manifest = builder.run_build(self.output)
        self.assertEqual(manifest["status"], "completed")
        report = pack_table(self.output / builder.MANIFEST_NAME,
                            self.output / "refined_synthetic.npz")
        self.assertEqual(report["shape"], [8, 7, 10, 21, 37])
        self.assertEqual(report["union_failure_nodes"], 1)

    def test_refined_grid_cannot_silently_change_after_preparation(self):
        manifest = builder.prepare_build(output_dir=self.output, cloudy_exe=self.executable,
                                         sed_dir=self.sed, workers=3, depth_points=37)
        manifest["axes"]["log_L_model_pc"][1] += 0.001
        with self.assertRaisesRegex(ValueError, "adopted grid"):
            builder._verify_prepared(self.output, manifest)

    def test_unrecognized_depth_grid_fails_before_creating_directory(self):
        with self.assertRaisesRegex(ValueError, "depth_points"):
            builder.prepare_build(output_dir=self.output, cloudy_exe=self.executable,
                                  sed_dir=self.sed, workers=3, depth_points=38)
        self.assertFalse(self.output.exists())

    def test_cli_requires_mode_and_prepare_only_never_launches(self):
        argv = ["build_cloudy_model_depth_tables.py", "--output-dir", str(self.output),
                "--cloudy-exe", str(self.executable), "--sed-dir", str(self.sed),
                "--workers", "3"]
        with patch.object(builder.sys, "argv", argv), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                builder.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.output.exists())
        with patch.object(builder.sys, "argv", argv + ["--prepare-only"]), \
                patch.object(builder.subprocess, "Popen") as launch, \
                redirect_stdout(io.StringIO()):
            builder.main()
        launch.assert_not_called()
        self.assertEqual(self.read_manifest()["status"], "prepared")

    def test_existing_build_is_not_replaced(self):
        self.prepare()
        before = (self.output / builder.MANIFEST_NAME).read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual((self.output / builder.MANIFEST_NAME).read_bytes(), before)

    def test_bad_sed_validation_and_shell_unsafe_paths_fail_before_preparation(self):
        report_path = self.sed / "build_report.json"
        report = json.loads(report_path.read_text())
        report["entries"][0]["roundtrip"]["maximum_relevant_absolute_error_dex"] = 0.01
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "roundtrip did not pass"):
            self.prepare()
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, "shell metacharacters"):
            builder.prepare_build(output_dir=self.root / "unsafe;name", cloudy_exe=self.executable,
                                  sed_dir=self.sed, workers=3)

    def test_launch_rejects_changed_inputs_or_preexisting_output(self):
        self.prepare()
        parameter = self.output / "fixed_depth.par"
        original = parameter.read_bytes()
        parameter.write_text(parameter.read_text() + "command metals 10 linear\n")
        with patch.object(builder.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "Prepared input changed"):
                builder.run_build(self.output)
        launch.assert_not_called()
        parameter.write_bytes(original)
        (self.output / "raw_maps" / "partial.dat").write_text("partial evidence")
        with self.assertRaises(FileExistsError):
            builder.run_build(self.output)
        self.assertEqual(self.read_manifest()["status"], "prepared")

    def test_launch_rejects_modified_execution_command(self):
        manifest = self.prepare()
        manifest["command"] = ["echo", "different execution"]
        builder._write_json(self.output / builder.MANIFEST_NAME, manifest)
        with patch.object(builder.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "execution command differs"):
                builder.run_build(self.output)
        launch.assert_not_called()

    def test_actual_headers_override_filename_order_and_packer_accepts_manifest(self):
        self.prepare()
        with patch.object(builder.subprocess, "Popen") as launch:
            process = launch.return_value
            process.pid = 12345
            def complete():
                self.assertEqual(self.read_manifest()["status"], "running")
                self.write_maps(reverse=True, crash=True)
                return 0
            process.wait.side_effect = complete
            process.poll.return_value = 0
            manifest = builder.run_build(self.output)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["process_returncode"], 0)
        run1 = next(record for record in manifest["maps"] if record["path"].endswith("_run1.dat"))
        self.assertEqual(run1["log_nH"], 6.0)
        self.assertEqual(run1["log_NH_attenuation"], 21.0)
        self.assertEqual(run1["log_L_model_pc"], 2.0)
        self.assertEqual(run1["crash_row_count"], 1)
        self.assertTrue(manifest["validation"]["complete_coordinate_and_temperature_coverage"])
        self.assertFalse(manifest["validation"]["cloudy_convergence_independently_verified"])
        # Explicitly call the real packer in this unit test; the builder never
        # invokes it. One attempted crash remains a failure for all eight lines.
        product = self.output / "synthetic_table.npz"
        report = pack_table(self.output / builder.MANIFEST_NAME, product)
        self.assertEqual(report["shape"], [8, 7, 10, 21, 10])
        self.assertEqual(report["union_failure_nodes"], 1)
        with np.load(product) as source:
            self.assertTrue(source["failure_mask"][:, -1, -1, 0, -1].all())
        with self.assertRaisesRegex(ValueError, "never-launched"):
            builder.run_build(self.output)

    def test_process_exit_failure_and_zero_exit_with_missing_maps_are_distinct(self):
        self.prepare()
        with patch.object(builder.subprocess, "Popen") as launch:
            launch.return_value.pid = 1
            launch.return_value.wait.return_value = 7
            launch.return_value.poll.return_value = 7
            with self.assertRaisesRegex(RuntimeError, "status 7"):
                builder.run_build(self.output)
        manifest = self.read_manifest()
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["process_returncode"], 7)
        self.output = self.root / "build2"
        self.prepare()
        with patch.object(builder.subprocess, "Popen") as launch:
            launch.return_value.pid = 2
            launch.return_value.wait.return_value = 0
            launch.return_value.poll.return_value = 0
            with self.assertRaisesRegex(ValueError, "filenames are missing"):
                builder.run_build(self.output)
        manifest = self.read_manifest()
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["process_returncode"], 0)
        self.assertFalse(manifest["validation"]["complete_coordinate_and_temperature_coverage"])

    def test_interrupted_observation_does_not_mark_live_process_failed(self):
        self.prepare()
        with patch.object(builder.subprocess, "Popen") as launch:
            launch.return_value.pid = 12345
            launch.return_value.wait.side_effect = KeyboardInterrupt()
            launch.return_value.poll.return_value = None
            with self.assertRaises(KeyboardInterrupt):
                builder.run_build(self.output)
        manifest = self.read_manifest()
        self.assertEqual(manifest["status"], "running")
        self.assertIn("observation_interrupted_at", manifest)
        self.assertNotIn("failed_at", manifest)


if __name__ == "__main__":
    unittest.main()
