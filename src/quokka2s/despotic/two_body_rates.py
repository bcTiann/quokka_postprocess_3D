"""GOW two-body coefficients prepared for one fixed-temperature ODE solve."""

import importlib

import numpy as np


_gow = importlib.import_module("despotic.chemistry.GOW")


class PreparedTwoBodyRates:
    """Prepare fixed terms; return current trial coefficients, shape (36,).

    Constructor scalars are temperature [K], clumping-adjusted hydrogen density
    [cm^-3], attenuated radiation [dimensionless], and initial H/H2/e densities
    [cm^-3]. A fresh object belongs to each fixed-temperature attempt.

    __call__(nH, nH2, ne) accepts current trial densities [cm^-3] and returns a
    new native-order coefficient array. Twenty-six values use the native
    template; ten composition-dependent entries are recomputed on every call.
    """

    def __init__(
        self,
        *,
        temperature_K,
        effective_density_cm3,
        attenuated_radiation,
        initial_hydrogen_density_cm3,
        initial_molecular_hydrogen_density_cm3,
        initial_electron_density_cm3,
    ):
        # Native sanitization preserves NumPy scalar types and behavior for NaN.
        T = np.maximum(temperature_K, _gow._small)
        self.effective_density_cm3 = effective_density_cm3

        # Keep all native static formulas in their original implementation.
        # Dynamic entries in this template are overwritten on every call below.
        self.coefficient_template = _gow._twobody_ratecoef(
            T=temperature_K,
            n=effective_density_cm3,
            nH=initial_hydrogen_density_cm3,
            nH2=initial_molecular_hydrogen_density_cm3,
            ne=initial_electron_density_cm3,
            chi=attenuated_radiation,
        )

        # Branching between H2O+ destruction by H2 and by electrons: rates 1-4.
        self.h2o_electron_rate = 5.3e-6 / np.sqrt(T)
        self.h3_oxygen_rate = 1.99e-9 * T**-0.190

        # H2 collisional dissociation: rates 21-22. Preserve both native
        # underflow branches; rates whose low/high limits vanish stay zero.
        logT = np.log10(T)
        logT4 = logT - 4.0
        self.h_low_density_rate = 6.67e-12*np.sqrt(T)*np.exp(-(1.0+63590./T))
        self.h_high_density_rate = 3.52e-9*np.exp(-43900./T)
        self.h_critical_density = 10.**(3.0 - 0.416*logT4 - 0.327*logT4**2)
        self.h2_critical_density = 10.**(4.845 - 1.3*logT4 + 1.62*logT4**2)
        self.h2_low_density_rate = (
            5.996e-30*T**4.1881 / (1.0+6.761e-6*T)**5.6881
            * np.exp(-54657.4/T)
        )
        self.h2_high_density_rate = 1.9e-9*np.exp(-53300./T)
        self.h_dissociation_active = (
            self.h_high_density_rate > 0.0 and self.h_low_density_rate > 0.0
        )
        self.h2_dissociation_active = (
            self.h2_low_density_rate > 0 and self.h2_high_density_rate > 0
        )
        if self.h_dissociation_active:
            self.log_h_high_density_rate = np.log(self.h_high_density_rate)
            self.log_h_low_density_rate = np.log(self.h_low_density_rate)
        if self.h2_dissociation_active:
            self.log_h2_high_density_rate = np.log(self.h2_high_density_rate)
            self.log_h2_low_density_rate = np.log(self.h2_low_density_rate)

        # Grain recombination: rates 32-35. psi still depends on trial ne.
        lnT = np.log(T)
        self.psi_numerator = 1.7*np.sqrt(T)
        self.h_grain_temperature_factor = 508.7*T**0.01586
        self.h_grain_psi_exponent = -0.4723-1.102e-5*lnT
        self.c_grain_temperature_factor = 433.1*T**0.4845
        self.c_grain_psi_exponent = -0.8120-1.333e-4*lnT
        self.he_grain_temperature_factor = 5115.*T**3.903e-7
        self.he_grain_psi_exponent = -0.4956-5.494e-7*lnT
        self.si_grain_temperature_factor = 43750.*T**1.635e-6
        self.si_grain_psi_exponent = -0.8964-7.538e-5*lnT

    def __call__(self, nH, nH2, ne):
        """Return 36 native-order coefficients for trial densities [cm^-3]."""
        nH = np.maximum(nH, _gow._small)
        nH2 = np.maximum(nH2, _gow._small)
        ne = np.maximum(ne, _gow._small)
        n = self.effective_density_cm3
        ratecoef = self.coefficient_template.copy()

        # Preserve native branching formulas and multiplication grouping.
        kH2OpH2 = 6.0e-10
        kH2Ope = self.h2o_electron_rate
        r = kH2OpH2 * nH2 / (kH2OpH2 * nH2 + kH2Ope * ne)
        ratecoef[1] = self.h3_oxygen_rate * r
        ratecoef[2] = self.h3_oxygen_rate * (1.0 - r)
        ratecoef[3] = 1.6e-9 * r
        ratecoef[4] = 1.6e-9 * (1.0 - r)

        # ncr is still calculated as native, even if both rates underflow.
        xH = nH / n
        xH2 = nH2 / n
        ncrH = self.h_critical_density
        ncrH2 = self.h2_critical_density
        ncr = 1.0/(xH/ncrH + xH2/ncrH2)
        if self.h_dissociation_active or self.h2_dissociation_active:
            # Reuse identical subexpressions without algebraic rearrangement.
            # In particular, low_weight is not 1-high_weight.
            relative_density = n/ncr
            density_denominator = 1.0+relative_density
            high_weight = relative_density / density_denominator
            low_weight = 1.0 / density_denominator
            if self.h_dissociation_active:
                ratecoef[21] = np.exp(
                    high_weight * self.log_h_high_density_rate
                    + low_weight * self.log_h_low_density_rate
                )
            if self.h2_dissociation_active:
                ratecoef[22] = np.exp(
                    high_weight * self.log_h2_high_density_rate
                    + low_weight * self.log_h2_low_density_rate
                )

        # Grain formulas retain their dependence on the current electron density.
        psi = self.psi_numerator/ne
        ratecoef[32] = 12.25e-14 / (
            1.0 + 8.074e-6*psi**1.378 * (
                1. + self.h_grain_temperature_factor
                * psi**self.h_grain_psi_exponent
            )
        )
        ratecoef[33] = 45.58e-14 / (
            1.0 + 6.089e-3*psi**1.128 * (
                1.0 + self.c_grain_temperature_factor
                * psi**self.c_grain_psi_exponent
            )
        )
        ratecoef[34] = 5.572e-14 / (
            1.0 + 3.185e-7*psi**1.512 * (
                1.0 + self.he_grain_temperature_factor
                * psi**self.he_grain_psi_exponent
            )
        )
        ratecoef[35] = 2.166e-14 / (
            1.0 + 5.678e-8*psi**1.874 * (
                1.0 + self.si_grain_temperature_factor
                * psi**self.si_grain_psi_exponent
            )
        )
        return ratecoef
