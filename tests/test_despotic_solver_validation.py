from __future__ import annotations

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from quokka2s.tables import ExplicitGrid, LineLumResult, build_gow_lvg_table
from quokka2s.tables.abundances import GOW_ELEMENTAL_ABUNDANCES
from quokka2s.tables.solver import _validate_final_state, solve_gow_lvg_point


class _Composition:
    def __init__(self):
        self.xHe = GOW_ELEMENTAL_ABUNDANCES["xHe"]
        self.mu, self.muH, self.qIon = 2.0, 1.4, 1e-11

    def computeDerived(self, nH):
        self.mu, self.muH, self.qIon = 2.0, 1.4, 1e-11

    def computeCv(self, temperature):
        return 1.5

    def computeEint(self, temperature):
        return 75.0


class _Cloud:
    def __init__(self, *, converged=True, gas_residual=0.0, dust_residual=0.0,
                 fail_levels=False, bad_line=False, change_temperature=False):
        self.nH, self.Tg, self.Td = 100.0, 50.0, 10.0
        self.comp, self.dust, self.rad = _Composition(), SimpleNamespace(), SimpleNamespace()
        self.chemabundances = {
            "H": 0.0, "H2": 0.5, "H+": 0.0, "e-": 1e-4,
            "H2+": 0.0, "H3+": 0.0,
            "He": GOW_ELEMENTAL_ABUNDANCES["xHe"], "He+": 0.0,
            "C": 0.0, "C+": GOW_ELEMENTAL_ABUNDANCES["xC"],
            "CO": 0.0, "HCO+": 0.0, "CHx": 0.0,
            "O": GOW_ELEMENTAL_ABUNDANCES["xO"], "O+": 0.0, "OHx": 0.0,
            "Si": 0.0, "Si+": GOW_ELEMENTAL_ABUNDANCES["xSi"],
        }
        self.emitters = {}
        self.addEmitter("CO", 0.0)
        self.converged, self.fail_levels, self.bad_line = converged, fail_levels, bad_line
        self.gas_residual, self.dust_residual = gas_residual, dust_residual
        self.change_temperature = change_temperature
        self.energy_calls, self.line_calls = [], []

    def addEmitter(self, name, abundance):
        self.emitters[name] = SimpleNamespace(
            energySkip=False, levPopInitialized=True, escapeProbInitialized=True,
            levPop=np.array([0.75, 0.25]), escapeProb=np.ones((2, 2)), tau=np.array([1.0]),
        )

    def setChemEq(self, **kwargs):
        self.Tg = 50.0
        return self.converged

    def dEdt(self, **kwargs):
        self.energy_calls.append(kwargs)
        if not kwargs.get("fixedLevPop", False):
            if self.fail_levels:
                # Cached initialized flags must not override a failed final solve.
                raise RuntimeError("convergence failed for CO")
            if self.change_temperature:
                self.Tg += 1.0
            return {"GammaCR": 1.0, "LambdaLine": {"CO": 1.0}}
        return {
            "dEdtGas": self.gas_residual, "maxAbsdEdtGas": 1.0,
            "dEdtDust": self.dust_residual, "maxAbsdEdtDust": 1.0,
        }

    def lineLum(self, name, **kwargs):
        self.line_calls.append(kwargs)
        if not kwargs.get("noRecompute", False):
            raise AssertionError("unchecked line population recomputation")
        return [
            {"upper": upper, "lower": upper - 1, "freq": upper * 1e11,
             "intIntensity": -2.0, "intTB": -1.0, "lumPerH": -float(upper),
             "tau": np.nan if self.bad_line else 1.0, "tauDust": 0.01}
            for upper in (1, 2)
        ]


class FinalStateValidationTests(unittest.TestCase):
    def test_energy_residual_uses_checked_populations_without_changing_physical_state(self):
        cell = _Cloud()
        before = (cell.Tg, cell.Td, dict(cell.chemabundances))
        energy = _validate_final_state(cell, ("CO",))
        self.assertEqual((cell.Tg, cell.Td, cell.chemabundances), before)
        self.assertEqual(cell.energy_calls, [
            {"escapeProbGeom": "LVG"},
            {"escapeProbGeom": "LVG", "sumOnly": True, "fixedLevPop": True},
        ])
        self.assertEqual(energy["validation.gas_relative_residual"], 0.0)

    def test_bad_thermal_residual_or_failed_level_solve_is_rejected(self):
        for kwargs, message in (
            ({"gas_residual": 1e-3}, "gas thermal residual"),
            ({"dust_residual": 1e-3}, "dust thermal residual"),
            ({"gas_residual": np.nan}, "energy rates"),
            ({"fail_levels": True}, "convergence failed for CO"),
            ({"change_temperature": True}, "changed the temperatures"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(RuntimeError, message):
                _validate_final_state(_Cloud(**kwargs), ("CO",))

    def test_stale_derived_state_is_detected_without_silently_repairing_it(self):
        cell = _Cloud()
        cell.comp.mu = 3.0
        with self.assertRaisesRegex(RuntimeError, "mu is inconsistent"):
            _validate_final_state(cell, ("CO",))
        self.assertEqual(cell.comp.mu, 3.0)
        self.assertEqual(cell.energy_calls, [])

    def test_wrong_element_total_and_invalid_population_are_rejected(self):
        cell = _Cloud()
        cell.chemabundances["C+"] *= 2.0
        with self.assertRaisesRegex(RuntimeError, "xC elemental total"):
            _validate_final_state(cell, ("CO",))
        cell = _Cloud()
        cell.emitters["CO"].levPop[0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "populations or escape probabilities"):
            _validate_final_state(cell, ("CO",))

    def test_network_hydrogen_conservation_and_negative_species(self):
        cell = _Cloud()
        cell.chemabundances["H3+"] = 1e-3
        with self.assertRaisesRegex(RuntimeError, "hydrogen total is not conserved"):
            _validate_final_state(cell, ("CO",))
        cell = _Cloud()
        cell.chemabundances["C"] = -1e-20
        with self.assertRaisesRegex(RuntimeError, "abundances are negative"):
            _validate_final_state(cell, ("CO",))

    def _solve(self, cell):
        attempts = []
        with (
            patch.dict(sys.modules, {
                "despotic": SimpleNamespace(cloud=lambda: cell),
                "despotic.chemistry": SimpleNamespace(GOW=object()),
            }),
            patch("quokka2s.tables.solver._configure_despotic_home"),
            patch("quokka2s.tables.solver._make_despotic_cloud", return_value=cell),
        ):
            result = solve_gow_lvg_point(
                100.0, 1e20, 1e-14, species=("CO",),
                log_failures=False, attempt_log=attempts,
            )
        return result, attempts

    def test_valid_solver_outputs_use_same_checked_populations_and_allow_absorption(self):
        cell = _Cloud()
        result, attempts = self._solve(cell)
        self.assertFalse(result[-1])
        self.assertEqual(result[0]["CO"].lumPerH, -1.0)
        self.assertEqual(result[0]["CO21"].lumPerH, -2.0)
        self.assertEqual(cell.line_calls, [{"escapeProbGeom": "LVG", "noRecompute": True}])
        self.assertEqual(result[5], 50.0)
        self.assertTrue(attempts[0].converged)

    def test_failures_return_only_nan_physical_outputs_and_keep_diagnostic_temperature(self):
        for kwargs in ({"converged": False}, {"gas_residual": 1e-3},
                       {"fail_levels": True}, {"bad_line": True}):
            with self.subTest(kwargs=kwargs):
                result, attempts = self._solve(_Cloud(**kwargs))
                self.assertTrue(result[-1])
                self.assertTrue(np.isnan(result[2:6]).all())
                self.assertTrue(all(np.isnan(value) for value in result[1].values()))
                self.assertTrue(all(np.isnan(value) for value in result[6].values()))
                for line in result[0].values():
                    self.assertTrue(np.isnan(tuple(vars(line).values())).all())
                self.assertFalse(attempts[0].converged)
                self.assertEqual(attempts[0].final_Tg, 50.0)

    def test_builder_discards_finite_outputs_from_a_failed_solver_attempt(self):
        finite_line = LineLumResult(1e11, 2.0, 3.0, 4.0, 5.0, 6.0)
        invalid_result = (
            {"CO": finite_line, "CO21": finite_line}, {"CO": 1e-4},
            2.0, 1.5, 75.0, 50.0, {"GammaCR": 1.0}, True,
        )
        with (
            patch("quokka2s.tables.builder.validated_solver_metadata", return_value={}),
            patch("quokka2s.tables.builder.solve_gow_lvg_point", return_value=invalid_result),
        ):
            table = build_gow_lvg_table(
                ExplicitGrid((100.0,)), ExplicitGrid((1e20,)), ExplicitGrid((1e-14,)),
                workers=1, show_progress=False,
            )
        self.assertTrue(table.failure_mask.all())
        for values in (table.tg_final, table.mu_values, table.cv_values, table.Eint_values):
            self.assertTrue(np.isnan(values).all())
        for record in table.species_data.values():
            self.assertTrue(np.isnan(record.abundance).all())
            if record.line is not None:
                self.assertTrue(np.isnan(record.line.lumPerH).all())
        self.assertIsNone(table.energy_terms)


if __name__ == "__main__":
    unittest.main()
