"""Check depth-pilot acceptance gates and interpolation against known states."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.cloudy_model_depth_common import (
    ELEMENT_SYMBOLS,
    LOG_T,
    PC_IN_CM,
    direct_input,
    inspect_direct_output,
)
from scripts.validate_cloudy_model_depth_pilot import compare_midpoints, run_case


class DirectOutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "state"
        self.expected = {symbol: -float(i) / 3 for i, symbol in enumerate(ELEMENT_SYMBOLS)}
        self.depth = PC_IN_CM * 10
        composition = "  ".join(f"{k}: {v:.4f}" for k, v in self.expected.items())
        self.output = (
            "Gas Phase Chemical Composition\n" + composition + "\n\n"
            "Iteration not converged because optical depths changed.\n"
            "Calculation stopped because outer radius reached. Iteration 3 of 3\n"
            "Cloudy ends: 10 zones, 3 iterations. ExecTime(s) 1.00\n"
            "[Stop in cdMain, Cloudy exited OK]\n"
        )
        self.root.with_suffix(".out").write_text(self.output)
        self._row(".radius", [10, 1e30, .9 * self.depth, .2 * self.depth])
        self._row(".physical", [.9 * self.depth, 1e4, 1, .1, 1e-20, 0, 1])
        self._row(".lines", [.9 * self.depth] + [1e-20] * 8)

    def _row(self, suffix, values):
        np.savetxt(self.root.with_suffix(suffix), np.asarray([values]))

    def check_output(self, returncode=0):
        result = inspect_direct_output(
            self.root, log_nH=0, log_T=4, log_L_pc=1,
            returncode=returncode, expected_abundances=self.expected,
        )
        # Invalid outputs must still be checkpointable without NaN/Infinity.
        json.dumps(result, allow_nan=False)
        return result

    def test_midpoint_plus_half_zone_width_reaches_requested_depth(self):
        result = self.check_output()
        self.assertTrue(result["valid"], result["issues"])
        self.assertAlmostEqual(result["actual_depth_cm"] / self.depth, 1)

    def test_exit_ok_does_not_override_local_convergence_failures(self):
        self.root.with_suffix(".out").write_text(self.output.replace(
            "3 iterations.", "3 iterations. Failures: 0 thermal, 0 pressure, 2 ionization, 0 electron density."
        ))
        result = self.check_output()
        self.assertFalse(result["valid"])
        self.assertEqual(result["local_convergence_failures"], [0, 0, 2, 0])

    def test_explicit_iteration_nonconvergence_is_rejected(self):
        self.root.with_suffix(".out").write_text(self.output.replace(
            "Cloudy ends:", "C-Iterate to convergence did not converge in 10 iterations.\nCloudy ends:"
        ))
        self.assertFalse(self.check_output()["valid"])

    def test_wrong_stop_reason_and_nonzero_exit_are_rejected(self):
        self.root.with_suffix(".out").write_text(self.output.replace(
            "outer radius reached.", "H column dens reached."
        ))
        self.assertFalse(self.check_output()["valid"])
        self.root.with_suffix(".out").write_text(self.output)
        self.assertFalse(self.check_output(returncode=1)["valid"])

    def test_wrong_composition_is_rejected(self):
        self.root.with_suffix(".out").write_text(self.output.replace("He: -0.3333", "He: -0.4333"))
        self.assertFalse(self.check_output()["valid"])

    def test_stale_matching_emission_and_physical_zone_is_rejected(self):
        self._row(".physical", [.5 * self.depth, 1e4, 1, .1, 1e-20, 0, 1])
        self._row(".lines", [.5 * self.depth] + [1e-20] * 8)
        self.assertFalse(self.check_output()["valid"])

    def test_nonfinite_temperature_or_geometry_is_rejected_and_serializable(self):
        self._row(".physical", [.9 * self.depth, np.nan, 1, .1, 1e-20, 0, 1])
        self.assertFalse(self.check_output()["valid"])
        self._row(".physical", [.9 * self.depth, 1e4, 1, .1, 1e-20, 0, 1])
        self._row(".radius", [10, 1e30, np.nan, .2 * self.depth])
        self.assertFalse(self.check_output()["valid"])

    def test_zero_emission_is_valid_but_missing_or_negative_line_is_not(self):
        self._row(".lines", [.9 * self.depth] + [0.] * 8)
        self.assertTrue(self.check_output()["valid"])
        self._row(".lines", [.9 * self.depth] + [1e-20] * 7)
        self.assertFalse(self.check_output()["valid"])
        self._row(".lines", [.9 * self.depth, -1e-20] + [1e-20] * 7)
        self.assertFalse(self.check_output()["valid"])

    def test_temperature_serialization_matches_two_stage_cialoop_format(self):
        text = direct_input("state", log_nH=0, log_T=float(LOG_T[8]), log_NH=20, log_L_pc=1)
        self.assertIn("constant temperature 8585.814000 K linear", text)


class MidpointTests(unittest.TestCase):
    @staticmethod
    def records(left, middle, right, *, valid=True):
        return [dict(track=0, depth_index=i, log_L_model_pc=i / 8,
                     checks=dict(valid=valid, emissivity_per_nH2=list(values)))
                for i, values in enumerate((left, middle, right))]

    def test_logarithmic_positive_and_linear_zero_support_rules(self):
        records = self.records(
            [1e-22, 0, 0, 1e-22],
            [1e-21, 1e-22, 0, 0],
            [1e-20, 2e-22, 0, 1e-22],
        )
        result = compare_midpoints(records)[0]
        self.assertTrue(result["valid"])
        # The nonzero corner's map entry is -21.6990 after four-place rounding.
        np.testing.assert_allclose(result["interpolated"], [1e-21, .5 * 10**-21.699, 0, 1e-22], rtol=1e-14)
        self.assertEqual(result["relative_error"][-1], None)
        self.assertEqual(result["zero_mismatch"], [False, False, False, True])

    def test_very_weak_positive_lines_do_not_underflow_geometric_mean(self):
        result = compare_midpoints(self.records([1e-250], [1e-240], [1e-230]))[0]
        self.assertAlmostEqual(result["interpolated"][0] / 1e-240, 1)
        self.assertFalse(result["zero_mismatch"][0])

    def test_invalid_endpoint_is_not_silently_filled(self):
        records = self.records([1.], [2.], [4.])
        records[0]["checks"]["valid"] = False
        self.assertFalse(compare_midpoints(records)[0]["valid"])

    def test_log_map_rounding_is_included_in_error_measurement(self):
        left_log, right_log = -20.000046, -18.000034
        result = compare_midpoints(self.records(
            [10**left_log], [1e-19], [10**right_log]
        ))[0]
        self.assertAlmostEqual(result["interpolated"][0] / 1e-19, 1)


class CheckpointTests(unittest.TestCase):
    def test_interrupted_prior_input_refuses_automatic_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / "track00_depth00.in").write_text("incomplete prior input")
            case = dict(id="track00_depth00", log_nH=0., log_T=4.,
                        log_NH=20., log_L_model_pc=1.)
            with patch("scripts.validate_cloudy_model_depth_pilot.subprocess.run") as run:
                with self.assertRaisesRegex(RuntimeError, "Incomplete prior case"):
                    run_case(case, base, Path("/nonexistent/cloudy"), {}, 10., {})
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
