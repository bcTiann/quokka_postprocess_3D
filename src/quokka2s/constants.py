"""Physical constants and unit conversions shared by the pipeline.

Astropy supplies fundamental constants. Convert them once to the units used
by NumPy arrays; the suffix of each name records those units. Model choices
such as elemental abundances, line wavelengths and temperature cuts stay
with the corresponding model.
"""

from astropy import constants as const
from astropy import units as u


BOLTZMANN_ERG_K = const.k_B.to_value("erg/K")
SPEED_OF_LIGHT_CM_S = const.c.to_value("cm/s")
SPEED_OF_LIGHT_KMS = const.c.to_value("km/s")
PLANCK_ERG_S = const.h.to_value("erg*s")
GRAVITATIONAL_CGS = const.G.to_value("cm**3/(g*s**2)")
ATOMIC_MASS_UNIT_G = const.u.to_value("g")
SOLAR_LUMINOSITY_ERG_S = const.L_sun.to_value("erg/s")
PARSEC_CM = u.pc.to(u.cm)

# Energy of one Rydberg [eV], used by radiation-field figure coordinates.
EV_PER_RYD = (const.Ryd * const.h * const.c).to_value("eV")
