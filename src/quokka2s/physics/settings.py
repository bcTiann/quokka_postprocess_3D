"""Physical choices shared by snapshot processing and table-domain checks.

File paths and execution options belong to the process configuration. These
settings describe the current physical model: native cells and +/-z shielding.
"""

# Preserve the adopted He/H number ratio and helium atomic weight in nH = X_H*rho/m_H.
X_H = 1.0 / (1.0 + 0.1 * 3.971)
# Floor for abs(div(v))/3 [s^-1], measured in the native plt0655228 snapshot.
SIMULATION_DVDR_MIN_S = 1.25685313685528378e-22
COLUMN_DENSITY_MEAN = 'harmonic'
COLUMN_DENSITY_DIRECTIONS = 'z'
