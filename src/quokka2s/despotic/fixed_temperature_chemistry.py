"""GOW reaction derivatives prepared for one fixed-temperature ODE attempt.

Only table building imports DESPOTIC. Initial conditions, reaction formulas,
solver precision and acceptance remain native; the checked integrator creates
a new callable before each odeint attempt and retains no cache across them.
"""
from __future__ import annotations

import importlib

import numpy as np
from despotic.chemistry import GOW

from quokka2s.despotic.abundance_workspace import AbundanceWorkspace
from quokka2s.despotic.reaction_terms import PreparedReactionTerms
from quokka2s.despotic.two_body_rates import PreparedTwoBodyRates


_gow = importlib.import_module("despotic.chemistry.GOW")


class TableGOW(GOW):
    """Native GOW with an explicit preparation step for fixed-T integration.

    Initialization, stored abundances and variable-temperature derivatives
    remain native. The checked fixed-T integrator alone uses this callable.
    """

    def prepare_fixed_temperature_rhs(self):
        """Prepare fresh constants and workspaces for this one ODE attempt."""
        return FixedTemperatureChemistry(network=self)


class FixedTemperatureChemistry:
    """Return current trial derivatives, shape (12,), in abundance s^-1.

    network is the current TableGOW. Fixed density is cm^-3, temperature is K,
    column is cm^-2 and velocity is cm/s. Column is native GOW.NH (=colDen/2).
    __call__(trial_abundances, time_s) accepts the 12 evolving abundances per H;
    time_s is unused and retained for odeint compatibility.

    This is a serial, non-reentrant callable. Its abundance scratch is private;
    each returned derivative is a fresh array, independent of later calls.
    """

    def __init__(self, *, network):
        self.network = network
        self.temperature_K = network.temp
        self.sigma_nonthermal_cm_s = network.sigmaNT
        # Native cfac uses saved network.x, not the integrator's trial state.
        clumping_factor = network.cfac
        self.effective_density_cm3 = network.nH * clumping_factor
        self.column_cm2 = network.NH
        self.radiation_chi = network.chi
        self.extinction_AV = network.AV
        self.ionization_rate_s = network.ionRate
        self.dust_abundance = network.Zd
        self.attenuated_radiation = np.exp(-1.87*network.AV)*network.chi

        # Reuse mappings only; all trial values are recomputed inside the RHS.
        self.abundance_workspace = AbundanceWorkspace(network=network)
        initial_abundances = self.abundance_workspace.expand(network.x)
        density = self.effective_density_cm3
        self.two_body_rates = PreparedTwoBodyRates(
            temperature_K=self.temperature_K,
            effective_density_cm3=density,
            attenuated_radiation=self.attenuated_radiation,
            initial_hydrogen_density_cm3=initial_abundances['H']*density,
            initial_molecular_hydrogen_density_cm3=initial_abundances['H2']*density,
            initial_electron_density_cm3=initial_abundances['e-']*density,
        )
        self.reaction_terms = PreparedReactionTerms(
            effective_density_cm3=density,
            ionization_rate_s=self.ionization_rate_s,
            radiation_chi=self.radiation_chi,
            extinction_AV=self.extinction_AV,
            dust_abundance=self.dust_abundance,
        )

    def __call__(self, trial_abundances, time_s):
        """Return a fresh (12,) derivative [s^-1] in native reaction order."""
        # 1. These scalar values stay fixed until this RHS evaluation ends.
        abundances = self.abundance_workspace.expand(trial_abundances)
        hydrogen_abundance = abundances['H']
        molecular_hydrogen_abundance = abundances['H2']
        electron_abundance = abundances['e-']
        particle_mass = mean_particle_mass(abundances=abundances)
        velocity_dispersion_cm_s = np.sqrt(
            self.sigma_nonthermal_cm_s**2
            + _gow.kB*self.temperature_K / (particle_mass*_gow.mH)
        )
        density = self.effective_density_cm3
        molecular_hydrogen_column_cm2 = self.column_cm2*molecular_hydrogen_abundance

        # 2. Preserve CR-then-photo order; the same H2 column serves all shields.
        derivative = self.reaction_terms.cosmic_rays(
            abundance_array=abundances.x,
        )
        derivative += self.reaction_terms.photoreactions(
            abundance_array=abundances.x,
            shield_args=[
                [self.column_cm2*abundances['C'], molecular_hydrogen_column_cm2],
                [self.column_cm2*abundances['CO'], molecular_hydrogen_column_cm2],
                [molecular_hydrogen_column_cm2, velocity_dispersion_cm_s],
            ],
        )

        # 3. Reusing H/H2 values does not change the native arithmetic grouping.
        hydrogen_ionization = (
            2.3*molecular_hydrogen_abundance + 1.5*hydrogen_abundance
        )*self.ionization_rate_s
        derivative[0, 1] += hydrogen_ionization * hydrogen_abundance
        derivative[0, 0] -= 2*hydrogen_ionization * molecular_hydrogen_abundance
        derivative[0, 2] += 2*hydrogen_ionization * molecular_hydrogen_abundance

        # 4. All state-dependent coefficients still use this current trial.
        rate_coefficients = self.two_body_rates(
            nH=hydrogen_abundance*density,
            nH2=molecular_hydrogen_abundance*density,
            ne=electron_abundance*density,
        )
        derivative += self.reaction_terms.two_body(
            abundance_array=abundances.x,
            ratecoef=rate_coefficients,
        )
        return np.ravel(derivative)[:len(_gow.specList)]


def mean_particle_mass(*, abundances):
    """Return native mu in H-mass units using the already expanded trial state."""
    return (
        abundances['H+'] + abundances['H'] + 2*abundances['H2']
        + 3*abundances['H3+'] + 4*(abundances['He+']+abundances['He'])
    ) / (
        abundances['H+'] + abundances['H'] + abundances['H2']
        + abundances['H3+'] + abundances['He+'] + abundances['He']
    )
