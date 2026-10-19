"""Internal abundance workspace for one serial fixed-temperature GOW RHS."""

import importlib

import numpy as np
from despotic.chemistry.abundanceDict import abundanceDict


_gow = importlib.import_module("despotic.chemistry.GOW")


class AbundanceWorkspace:
    """Recompute 18 abundances while reusing two native name/index mappings.

    network supplies GOW's elemental totals. expand(trial_abundances) reads
    shape (12,) in the trial's original dtype and returns a reused native
    abundanceDict with float64 values, shape (18,). It never changes the trial
    or network state.

    The returned object is internal scratch: the next call overwrites it.
    Use one workspace per serial integration attempt; do not retain its
    abundance output or share it between simultaneous/reentrant RHS calls.
    Complete derivative arrays are allocated separately and may be retained.
    """

    def __init__(self, *, network):
        self.network = network
        self.tracked_abundances = abundanceDict(
            _gow.specList,
            network.x,
        )
        self.extended_abundances = abundanceDict(
            _gow.specListExtended,
            np.zeros(len(_gow.specListExtended)),
        )

    def expand(self, trial_abundances):
        """Recompute H/He/O/C/Si/e with native formulas and original dtype."""
        # Read derived expressions from the original trial dtype. Reading them
        # from the float64 output copy would change native float32 arithmetic.
        abd = self.tracked_abundances
        abd.x = trial_abundances
        abd_out = self.extended_abundances
        abd_out.x[:len(_gow.specList)] = trial_abundances

        # Preserve the complete native expression and order for each species.
        abd_out['H'] = 1.0 - abd['H+'] - 2*abd['H2'] - 2*abd['H2+'] \
                       - 3*abd['H3+'] - abd['OHx'] - abd['CHx'] \
                       - abd['HCO+']
        abd_out['He'] = self.network.xHe - abd['He+']
        abd_out['O'] = self.network.xO - abd['O+'] - abd['CO'] - abd['HCO+'] \
                       - abd['OHx']
        abd_out['C'] = self.network.xC - abd['C+'] - abd['CO'] - abd['HCO+'] \
                       - abd['CHx']
        abd_out['Si'] = self.network.xSi - abd['Si+']
        abd_out['e-'] = abd['H+'] + abd['H2+'] + abd['H3+'] + \
                        abd['He+'] + abd['C+'] + abd['O+'] + \
                        abd['HCO+'] + abd['Si+']
        return abd_out
