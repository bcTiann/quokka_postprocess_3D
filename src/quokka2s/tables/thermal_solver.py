"""Checked DESPOTIC thermal solves with an optional local gas bracket.

Import this module only when building tables: DESPOTIC is an optional dependency.
The fallback follows ``cloud.setGasTempEq`` at upstream commit ed18e5669adb,
retaining its residual, CMB treatment, Brent tolerances, and decade expansion.
It changes only the initial upper endpoint and rejects invalid evaluations.
"""
from __future__ import annotations

import importlib

import numpy as np
from despotic import cloud as _Cloud
from despotic.despoticError import despoticError

from .models import ThermalSolveError

_native = importlib.import_module("despotic.cloud")
MAX_GAS_BRACKET_EXPANSIONS = 12
_THERMAL_ERRORS = (despoticError, RuntimeError, ValueError, ArithmeticError)


class CheckedCloud(_Cloud):
    """Raise on failed thermal subsolves that DESPOTIC callers can ignore.

    ``initial_gas_upper=None`` leaves the native successful solve unchanged.
    A positive explicit upper endpoint enables a local alternative bracket;
    the caller must retry on a fresh cloud after a failed whole-point attempt.
    """

    def __init__(self, fileName=None, noWarn=False, verbose=False, *,
                 initial_gas_upper=None):
        if initial_gas_upper is not None:
            initial_gas_upper = float(initial_gas_upper)
            if not np.isfinite(initial_gas_upper) or initial_gas_upper <= 1.0:
                raise ValueError("initial_gas_upper must be finite and greater than 1 K")
        super().__init__(fileName=fileName, noWarn=noWarn, verbose=verbose)
        self.initial_gas_upper = initial_gas_upper
        self.last_gas_bracket = None

    def setGasTempEq(self, *args, **kwargs):
        self.last_gas_bracket = None
        try:
            if self.initial_gas_upper is None:
                success = super().setGasTempEq(*args, **kwargs)
            else:
                success = self._set_gas_temp_with_upper(*args, **kwargs)
        except ThermalSolveError:
            raise
        except _THERMAL_ERRORS as exc:
            raise ThermalSolveError(f"Gas thermal solve failed: {exc}") from exc
        if not success or not np.isfinite(self.Tg) or self.Tg <= 0.0:
            raise ThermalSolveError("Gas thermal solve did not converge to a valid temperature")
        return True

    def setTempEq(self, *args, **kwargs):
        try:
            success = super().setTempEq(*args, **kwargs)
        except ThermalSolveError:
            raise
        except _THERMAL_ERRORS as exc:
            raise ThermalSolveError(f"Gas/dust thermal solve failed: {exc}") from exc
        if (not success or not np.isfinite([self.Tg, self.Td]).all()
                or min(self.Tg, self.Td) <= 0.0):
            raise ThermalSolveError("Gas/dust thermal solve did not converge to valid temperatures")
        return True

    def _set_gas_temp_with_upper(
        self, c1Grav=0.0, c1Turb=0.0, thin=False, noClump=False,
        LTE=False, Tginit=None, fixedLevPop=False,
        escapeProbGeom="sphere", PsiUser=None, verbose=False,
    ):
        # Keep the pinned native initialization and dEdt arguments. As in that
        # implementation, fixedLevPop is not used by the gas residual.
        if self.comp.mu == 0.0:
            self.comp.computeDerived(self.nH)
        if Tginit is not None:
            self.Tg = Tginit
        elif self.Tg == 0.0:
            self.Tg = 10.0
        rates = self.dEdt(
            c1Grav=c1Grav, c1Turb=c1Turb, thin=thin, LTE=LTE,
            escapeProbGeom=escapeProbGeom, gasOnly=True, noClump=noClump,
            sumOnly=True, PsiUser=PsiUser,
        )
        scale = float(rates["maxAbsdEdtGas"])
        if not np.isfinite(scale) or scale <= 0.0:
            raise ThermalSolveError("Gas thermal residual has invalid luminosity scaling")

        def residual(log_temperature):
            value = float(_native._gasTempResid(
                log_temperature, self, c1Grav, c1Turb, thin, LTE,
                escapeProbGeom, PsiUser, noClump, scale, verbose,
            ))
            if not np.isfinite(value):
                raise ThermalSolveError("Gas thermal residual is not finite")
            return value

        lower = 1.0
        upper = self.initial_gas_upper
        # The residual changes self.Tg, so upper must not be read from self.Tg
        # after evaluating the lower endpoint.
        bracket = {"initial_upper_K": upper, "lower_K": lower, "evaluations": []}
        self.last_gas_bracket = bracket
        low_residual = residual(np.log(lower))
        bracket["lower_residual"] = low_residual
        for expansion in range(MAX_GAS_BRACKET_EXPANSIONS + 1):
            if not np.isfinite(upper):
                raise ThermalSolveError("Gas bracket expansion produced a non-finite temperature")
            high_residual = residual(np.log(upper))
            bracket["evaluations"].append({"upper_K": upper, "residual": high_residual})
            if (low_residual == 0.0 or high_residual == 0.0
                    or np.signbit(low_residual) != np.signbit(high_residual)):
                break
            if expansion == MAX_GAS_BRACKET_EXPANSIONS:
                raise ThermalSolveError("Gas root was not bracketed within the expansion budget")
            upper *= 10.0

        bracket["upper_K"] = upper
        bracket["expansions"] = expansion
        self.Tg = float(np.exp(_native.brentq(residual, np.log(lower), np.log(upper))))
        # Keep the native 1e-3 subsolve criterion. The whole-point caller checks
        # the final coupled gas/dust residual independently at its own tolerance.
        final_residual = residual(np.log(self.Tg))
        bracket["final_residual"] = final_residual
        bracket["temperature_K"] = self.Tg
        return abs(final_residual) <= 1.0e-3
