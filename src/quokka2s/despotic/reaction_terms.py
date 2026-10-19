"""Fixed-density and radiation terms for the pinned GOW reaction network."""

import importlib

import numpy as np


_gow = importlib.import_module("despotic.chemistry.GOW")


class PreparedReactionTerms:
    """Return native matrix derivatives, shape (1,18), in abundance s^-1.

    Constructor scalars are effective hydrogen density [cm^-3], ionization
    rate [s^-1], radiation strength, extinction A_V and dust abundance. Each
    integration attempt gets a fresh object; all trial abundances stay live.

    Methods read extended abundances, shape (18,), without changing them.
    Photoreactions evaluate all three native shielding functions every call;
    two-body reactions retain the native gas-then-grain accumulation order.
    """

    def __init__(
        self,
        *,
        effective_density_cm3,
        ionization_rate_s,
        radiation_chi,
        extinction_AV,
        dust_abundance,
    ):
        self.cosmic_ray_reactions = _gow._cr
        self.photo_reactions = _gow._ph
        self.two_body_reactions = _gow._twobody
        self.effective_density_cm3 = effective_density_cm3
        self.dust_abundance = dust_abundance

        # These vectors use the exact native expressions, once per segment.
        self.cosmic_ray_coefficients = (
            self.cosmic_ray_reactions.rate_fac * ionization_rate_s
        )
        self.unshielded_photo_coefficients = (
            self.photo_reactions.rate_fac * radiation_chi
            * np.exp(-extinction_AV*self.photo_reactions.av_fac)
        )
        self.cosmic_ray_density_powers = (
            effective_density_cm3**(self.cosmic_ray_reactions.lhssum-1)
        )
        self.photo_density_powers = (
            effective_density_cm3**(self.photo_reactions.lhssum-1)
        )
        self.two_body_density_powers = (
            effective_density_cm3**(self.two_body_reactions.lhssum-1)
        )
        self.non_grain_reactions = np.logical_not(self.two_body_reactions.grain)
        # The pinned GOW CR/photo rows each contain one reactant, exponent 1.
        self.cosmic_ray_reactant_indices = self.cosmic_ray_reactions.lhsidx[:, 0]
        self.photo_reactant_indices = self.photo_reactions.lhsidx[:, 0]

    def cosmic_rays(self, abundance_array):
        """Return the native matrix (1,18), derivatives [s^-1]."""
        rates = (
            self.cosmic_ray_coefficients
            * abundance_array[self.cosmic_ray_reactant_indices]
            * self.cosmic_ray_density_powers
        )
        return self.cosmic_ray_reactions.mat.dot(rates)

    def photoreactions(self, abundance_array, shield_args):
        """Evaluate the three native shields and apply unary reaction rates."""
        reactions = self.photo_reactions
        fshield = np.zeros(3)
        fshield[0] = reactions.shield_func[0](*shield_args[0])
        fshield[1] = reactions.shield_func[1](*shield_args[1])
        fshield[2] = reactions.shield_func[2](*shield_args[2])

        ratecoef = self.unshielded_photo_coefficients.copy()
        ratecoef[reactions.shield_func_idx] *= fshield
        rates = (
            ratecoef
            * abundance_array[self.photo_reactant_indices]
            * self.photo_density_powers
        )
        return reactions.mat.dot(rates)

    def two_body(self, abundance_array, ratecoef):
        """Return gas plus grain derivatives for current coefficients (36,).

        The coefficients come from PreparedTwoBodyRates. Keep both full-size
        zero masks, matrix products and gas-then-grain addition as native GOW.
        """
        reactions = self.two_body_reactions
        x = abundance_array
        n = self.effective_density_cm3

        # Gas reactions: preserve (coefficient * abundance product) * n power.
        rcoef = np.copy(ratecoef)
        rcoef[reactions.grain] = 0.0
        rates = (
            rcoef
            * np.prod(x[reactions.lhsidx]**reactions.lhsstoich, axis=1)
            * self.two_body_density_powers
        )
        dxdt = reactions.mat.dot(rates)

        # Grain reactions: retain the native first-reactant rule and order.
        rgr = np.copy(ratecoef)
        rgr[reactions.grain] *= self.dust_abundance
        rgr[self.non_grain_reactions] = 0.0
        rates = rgr * x[reactions.lhsidx[:,0]] * n
        dxdt += reactions.mat.dot(rates)
        return dxdt
