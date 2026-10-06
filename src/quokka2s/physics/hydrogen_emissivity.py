"""Analytic Halpha and optically thin H I 21-cm volume emissivities."""

import numpy as np
from unyt import cm, unyt_quantity

from ..constants import PLANCK_ERG_S, SPEED_OF_LIGHT_CM_S


HALPHA_WAVELENGTH_CM = 656.3e-7

# Retain the established unit conversions, evaluated once. Astropy supplies
# the constants; unyt uses the same CGS conversion as the original yt fields.
_planck_with_units = unyt_quantity(PLANCK_ERG_S, 'erg*s')
_light_speed_with_units = unyt_quantity(SPEED_OF_LIGHT_CM_S, 'cm/s')
HALPHA_PHOTON_ENERGY_ERG = float(
    ((_planck_with_units * _light_speed_with_units) / (HALPHA_WAVELENGTH_CM * cm))
    .in_cgs().value
)
_PLANCK_CGS_ERG_S = float(_planck_with_units.in_cgs().value)
HI21_FREQUENCY_HZ = 1420.405751768e6  # NIST / NRAO hyperfine frequency.
HI21_SPONTANEOUS_RATE_S = 2.85e-15   # Furlanetto et al. (2006), Phys. Rep.


def effective_halpha_recombination_coefficient(temperature_K):
    """Return the case-B Halpha coefficient [cm^3/s], Huang et al. (2025), Eq. 1.

    temperature_K is a scalar or NumPy array, from DESPOTIC in the cold branch.
    The returned scalar/array has the same shape; at 10000 K it is 1.17e-13.
    """
    temperature = np.asarray(temperature_K, dtype=float)
    T4 = np.maximum(temperature / 1.0e4, 1.0e-10)
    exponent = -0.942 - 0.031 * np.log(T4)
    return 1.17e-13 * np.power(T4, exponent)


def halpha_emissivity(temperature_K, electron_density_cm3, proton_density_cm3):
    """Return Halpha emissivity [erg/s/cm^3] from temperature and e-/H+ densities.

    Inputs come from the DESPOTIC lookup and are scalars or matching arrays.
    Example: three arrays of shape (Ncold,) return one (Ncold,) emissivity array.
    """
    coefficient = effective_halpha_recombination_coefficient(temperature_K)
    return (
        HALPHA_PHOTON_ENERGY_ERG * coefficient
        * electron_density_cm3 * proton_density_cm3
    )


def hi21_emissivity(neutral_hydrogen_density_cm3):
    """Return optically thin H I 21-cm emissivity [erg/s/cm^3].

    neutral_hydrogen_density_cm3 is DESPOTIC's neutral-H number density [cm^-3],
    a scalar or array. The factor 3/4 is the upper hyperfine-state population.
    """
    return (
        0.75 * neutral_hydrogen_density_cm3 * HI21_SPONTANEOUS_RATE_S
        * _PLANCK_CGS_ERG_S * HI21_FREQUENCY_HZ
    )
