import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

from quokka2s.tables.models import ThermalSolveError
from quokka2s.tables import solver

try:
    from despotic import cloud as NativeCloud
    from despotic.despoticError import despoticError
    from quokka2s.tables import thermal_solver
except ModuleNotFoundError as exc:
    if exc.name != "despotic":
        raise
    thermal_solver = None


@unittest.skipIf(thermal_solver is None, "DESPOTIC table-building dependency is optional")
class ThermalSolverTests(unittest.TestCase):
    def make_cloud(self, upper=None, target=412.0):
        cell = thermal_solver.CheckedCloud(initial_gas_upper=upper)
        cell.Tg, cell.Td, cell.comp.mu = 100.0, 10.0, 2.0
        cell.rad.TCMB = 2.73
        cell.dEdt = lambda **kwargs: {
            "maxAbsdEdtGas": 1.0,
            "dEdtGas": np.log(target / cell.Tg),
        }
        return cell

    def test_native_success_is_delegated_unchanged(self):
        cell = self.make_cloud()
        with patch.object(NativeCloud, "setGasTempEq", autospec=True, return_value=True) as solve:
            self.assertTrue(cell.setGasTempEq(escapeProbGeom="LVG"))
        solve.assert_called_once_with(cell, escapeProbGeom="LVG")
        self.assertEqual(cell.Tg, 100.0)
        self.assertIsNone(cell.last_gas_bracket)

    def test_false_native_subsolves_raise(self):
        for method in ("setGasTempEq", "setTempEq"):
            with self.subTest(method=method):
                cell = self.make_cloud()
                with patch.object(NativeCloud, method, return_value=False):
                    with self.assertRaises(ThermalSolveError):
                        getattr(cell, method)()

    def test_automatic_expansion_does_not_cap_temperature_or_change_globals(self):
        cell = self.make_cloud(upper=100.0, target=3000.0)
        native_bounds = (thermal_solver._native.Tlo, thermal_solver._native.Thi)
        self.assertTrue(cell.setGasTempEq())
        self.assertAlmostEqual(cell.Tg, 3000.0, places=7)
        self.assertEqual(cell.last_gas_bracket["upper_K"], 10000.0)
        self.assertEqual(cell.last_gas_bracket["expansions"], 2)
        self.assertEqual(native_bounds, (thermal_solver._native.Tlo, thermal_solver._native.Thi))

    def test_native_cmb_floor_is_used_by_residual(self):
        cell = self.make_cloud(upper=1000.0)
        temperatures = []
        rates = cell.dEdt
        def observed(**kwargs):
            temperatures.append(cell.Tg)
            return rates(**kwargs)
        cell.dEdt = observed
        self.assertTrue(cell.setGasTempEq())
        self.assertIn(2.73, temperatures)
        self.assertNotIn(1.0, temperatures)

    def test_nonfinite_residual_is_rejected(self):
        cell = self.make_cloud(upper=1000.0)
        cell.dEdt = lambda **kwargs: {"maxAbsdEdtGas": 1.0, "dEdtGas": np.nan}
        with self.assertRaisesRegex(ThermalSolveError, "not finite"):
            cell.setGasTempEq()

    def test_failed_level_evaluation_is_not_treated_as_a_sign(self):
        cell = self.make_cloud(upper=1000.0)
        rates = cell.dEdt
        def fail_at_upper(**kwargs):
            if cell.Tg > 900.0:
                raise despoticError("convergence failed for co")
            return rates(**kwargs)
        cell.dEdt = fail_at_upper
        with self.assertRaisesRegex(ThermalSolveError, "convergence failed for co"):
            cell.setGasTempEq()

    def test_same_sign_bracket_has_bounded_expansion(self):
        cell = self.make_cloud(upper=1000.0)
        cell.dEdt = lambda **kwargs: {"maxAbsdEdtGas": 1.0, "dEdtGas": 1.0}
        with self.assertRaisesRegex(ThermalSolveError, "expansion budget"):
            cell.setGasTempEq()
        self.assertEqual(len(cell.last_gas_bracket["evaluations"]),
                         thermal_solver.MAX_GAS_BRACKET_EXPANSIONS + 1)

    def make_solver_cloud(self, failure=None):
        cell = SimpleNamespace(
            comp=SimpleNamespace(
                mu=2.0, computeDerived=Mock(),
                computeCv=Mock(return_value=1.0), computeEint=Mock(return_value=1.0),
            ),
            dust=SimpleNamespace(), rad=SimpleNamespace(), chemabundances={},
            addEmitter=Mock(), lineLum=Mock(return_value=[]),
        )
        def chemistry(**kwargs):
            if failure is not None:
                # Simulate a thermal probe leaving its cell at an unusable
                # intermediate state. A retry must begin on a different cell.
                cell.Tg = 10000.0
                cell.comp.mu = 999.0
                raise failure
            self.assertEqual(cell.Tg, 100.0)
            self.assertEqual(cell.comp.mu, 2.0)
            cell.Tg = 42.0
            return True
        cell.setChemEq = Mock(side_effect=chemistry)
        return cell

    def run_mock_point(self, factory):
        attempts = []
        with (
            patch.object(solver, "_configure_despotic_home"),
            patch.object(solver, "_make_despotic_cloud", factory),
            patch.object(solver, "_validate_final_state", return_value={}),
        ):
            result = solver.solve_gow_lvg_point(
                100.0, 1.0e21, 1.0e-14, species=(), abundance_only=(),
                log_failures=False, attempt_log=attempts,
            )
        return result, attempts

    def test_point_native_success_does_not_fallback(self):
        cell = self.make_solver_cloud()
        factory = Mock(return_value=cell)
        result, attempts = self.run_mock_point(factory)
        factory.assert_called_once_with(None)
        self.assertFalse(result[-1])
        self.assertEqual(result[5], 42.0)
        self.assertEqual(len(attempts), 1)
        self.assertTrue(attempts[0].converged)

    def test_point_thermal_failure_retries_once_on_fresh_cloud(self):
        native = self.make_solver_cloud(ThermalSolveError("CO bracket evaluation failed"))
        fallback = self.make_solver_cloud()
        factory = Mock(side_effect=[native, fallback])
        result, attempts = self.run_mock_point(factory)
        self.assertEqual(factory.call_args_list, [call(None), call(1000.0)])
        self.assertIsNot(native, fallback)
        self.assertEqual(native.Tg, 10000.0)
        self.assertEqual(native.comp.mu, 999.0)
        self.assertFalse(result[-1])
        self.assertEqual(result[5], 42.0)
        self.assertEqual([attempt.converged for attempt in attempts], [False, True])

    def test_point_nonthermal_failure_does_not_fallback(self):
        cell = self.make_solver_cloud(RuntimeError("Chemical integration failed"))
        factory = Mock(return_value=cell)
        result, attempts = self.run_mock_point(factory)
        factory.assert_called_once_with(None)
        self.assertTrue(result[-1])
        self.assertTrue(np.isnan(result[5]))
        self.assertEqual(len(attempts), 1)
        self.assertFalse(attempts[0].converged)


if __name__ == "__main__":
    unittest.main()
