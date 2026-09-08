from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
_INSTALLER = _ROOT / "scripts" / "apply_despotic_chemistry_patch.py"


class CheckedGOWIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("despotic") is None:
            raise unittest.SkipTest("DESPOTIC is an optional table-building dependency")

        native = importlib.import_module("despotic.chemistry.chemEvol")
        cls.GOW = importlib.import_module("despotic.chemistry.GOW").GOW
        cls.DespoticError = importlib.import_module("despotic.despoticError").despoticError

        spec = importlib.util.spec_from_file_location("chemistry_patch_under_test", _INSTALLER)
        installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer)
        source = Path(native.__file__).read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        if digest == installer.ORIGINAL_SHA256:
            # Exercise the exact installable patch without changing the package.
            source = source.replace(installer.BEFORE, installer.AFTER)
        if hashlib.sha256(source).hexdigest() != installer.PATCHED_SHA256:
            raise AssertionError("DESPOTIC chemEvol source differs from the reviewed patch")

        cls.module = ModuleType("despotic.chemistry._checked_integration_test")
        cls.module.__package__ = "despotic.chemistry"
        cls.module.__file__ = native.__file__
        exec(compile(source, native.__file__, "exec"), cls.module.__dict__)

    def setUp(self):
        class RecordingGOW(self.GOW):
            def __init__(self):
                self.specList = ["first", "second"]
                self.x = np.array([0.2, 0.8])
                self.rhs_calls = []
                self.applied = []

            def dxdt(self, state, time):
                self.rhs_calls.append((np.array(state, copy=True), time))
                return np.zeros_like(state)

            def applyAbundances(self, addEmitters=False):
                self.applied.append(self.x.copy())

        self.network = RecordingGOW()
        self.cloud = SimpleNamespace(chemnetwork=self.network, emitters={}, Tg=100.0, nH=10.0)
        self.initial = self.network.x.copy()
        self.trajectory = np.array([[0.2, 0.8], [0.25, 0.75], [0.3, 0.7]])

    def evolve(self):
        return self.module.chemEvol(self.cloud, 20.0, tInit=10.0, nOut=2, evolveTemp="fixed")

    def test_success_keeps_physical_times_and_never_retries(self):
        def integrate(rhs, initial, times, **kwargs):
            self.assertEqual(rhs, self.network.dxdt)
            np.testing.assert_array_equal(times, [10.0, 15.0, 20.0])
            np.testing.assert_array_equal(initial, self.initial)
            self.assertFalse(np.shares_memory(initial, self.network.x))
            self.assertEqual(kwargs, {
                "rtol": 1e-8, "atol": 1e-12, "mxstep": 10000, "full_output": True,
            })
            rhs(initial, 12.0)
            return self.trajectory.copy(), {"message": "Integration successful."}

        with patch.object(self.module, "odeint", side_effect=integrate) as ode:
            self.evolve()
        self.assertEqual(ode.call_count, 1)
        self.assertEqual(self.network.rhs_calls[0][1], 12.0)
        np.testing.assert_array_equal(self.network.x, self.trajectory[-1])
        self.assertEqual(len(self.network.applied), 1)
        self.assertEqual(self.cloud._fixed_chemistry_attempts, [
            {"offset_s": 0.0, "success": True, "message": "Integration successful."},
        ])

    def test_retry_discards_partial_output_and_restores_original_rhs_time(self):
        calls = []

        def integrate(rhs, initial, times, **kwargs):
            calls.append(times.copy())
            np.testing.assert_array_equal(initial, self.initial)
            np.testing.assert_array_equal(self.network.x, self.initial)
            self.assertEqual(self.network.applied, [])
            if len(calls) == 1:
                initial[:] = 99.0
                return np.full_like(self.trajectory, -999.0), {"message": "Excess work done"}
            np.testing.assert_array_equal(times, [0.0, 5.0, 10.0])
            rhs(initial, 4.0)
            return self.trajectory.copy(), {"message": "Integration successful."}

        with patch.object(self.module, "odeint", side_effect=integrate):
            self.evolve()
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0], [10.0, 15.0, 20.0])
        self.assertEqual(self.network.rhs_calls[-1][1], 14.0)
        np.testing.assert_array_equal(self.network.x, self.trajectory[-1])
        np.testing.assert_array_equal(self.network.applied, [self.trajectory[-1]])
        self.assertEqual([entry["offset_s"] for entry in self.cloud._fixed_chemistry_attempts], [0.0, 10.0])
        self.assertEqual([entry["success"] for entry in self.cloud._fixed_chemistry_attempts], [False, True])

    def test_two_failed_attempts_raise_without_applying_any_output(self):
        with patch.object(self.module, "odeint", return_value=(
            np.full_like(self.trajectory, 99.0), {"message": "Excess work done"},
        )) as ode:
            with self.assertRaisesRegex(self.DespoticError, "all same-state attempts rejected"):
                self.evolve()
        self.assertEqual(ode.call_count, 2)
        np.testing.assert_array_equal(self.network.x, self.initial)
        self.assertEqual(self.network.applied, [])
        self.assertTrue(all(not entry["success"] for entry in self.cloud._fixed_chemistry_attempts))

    def test_success_status_with_nonfinite_or_incomplete_output_is_rejected(self):
        for bad in (np.full_like(self.trajectory, np.nan),
                    np.full_like(self.trajectory, np.inf), self.trajectory[:1]):
            with self.subTest(shape=bad.shape, finite=np.isfinite(bad).all()):
                self.setUp()
                with patch.object(self.module, "odeint", side_effect=[
                    (bad.copy(), {"message": "Integration successful."}),
                    (self.trajectory.copy(), {"message": "Integration successful."}),
                ]) as ode:
                    self.evolve()
                self.assertEqual(ode.call_count, 2)
                np.testing.assert_array_equal(self.network.applied, [self.trajectory[-1]])
                self.assertEqual([entry["success"] for entry in self.cloud._fixed_chemistry_attempts], [False, True])

    def test_other_networks_keep_the_original_integrator_interface(self):
        applied = []
        network = SimpleNamespace(x=self.initial.copy(), specList=["first", "second"])
        network.dxdt = lambda state, time: np.zeros_like(state)
        network.applyAbundances = lambda **kwargs: applied.append(network.x.copy())
        self.cloud.chemnetwork = network
        with patch.object(self.module, "odeint", return_value=self.trajectory.copy()) as ode:
            self.evolve()
        self.assertEqual(ode.call_count, 1)
        self.assertEqual(ode.call_args.kwargs, {})
        self.assertEqual(ode.call_args.args[0], network.dxdt)
        np.testing.assert_array_equal(ode.call_args.args[2], [10.0, 15.0, 20.0])
        np.testing.assert_array_equal(applied, [self.trajectory[-1]])
        self.assertFalse(hasattr(self.cloud, "_fixed_chemistry_attempts"))


if __name__ == "__main__":
    unittest.main()
