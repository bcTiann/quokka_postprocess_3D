"""Shared physical grid and ordered line list for the eight-line Cloudy build.

The runner writes these Cloudy labels to CIAOLoop's parameter file. The
packager uses the same order and checks the corresponding map-file headers.
The generator uses the same attenuation nodes and SED directory. The lookup
and packager share the serialized array order without changing its spelling.
"""

STEM = "hm2012_attgrid_ism_nh21_cmb_cr_defaultabund_eightline_jeans"
SED_DIRECTORY_NAME = "HM12_ATTENUATION_ISM_NH21"
HM12_LOG_NH = (18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0)
# Fixed incident-radiation recipe used by the builder and bundle metadata.
# ISM attenuation is log10(NH [cm^-2]); redshift is dimensionless; the
# nominal H0 cosmic-ray ionization rate is in s^-1. The parameter writer
# keeps the original six-decimal log10 rate command for this nominal value.
ISM_ATTENUATION_LOG_NH = 21.0
CMB_REDSHIFT = 0.0
COSMIC_RAY_H0_IONIZATION_RATE_S = 2.0e-17
# NPZ values have shape (line, attenuation column, density, temperature).
# Each coordinate axis stores log10 of its physical value, using cm^-2,
# cm^-3, and K respectively. Keep this exact serialized metadata string.
EXPECTED_AXIS_ORDER = "line,log_NH_attenuation,log_nH,log_T"
LOG_NH_DENSITY = (
    -4.71428571428571,
    -3.52380952380952,
    -2.33333333333333,
    -1.14285714285714,
    0.0476190476190476,
    1.23809523809524,
    2.42857142857143,
    3.61904761904762,
    4.80952380952381,
    6.0,
)
N_DENSITY = len(LOG_NH_DENSITY)
N_T = 21
# Preserve the original parameter-file spelling; parse the same values for
# bundle validation so the two scripts cannot drift numerically.
T_MIN_CLOUDY = "3.6"
T_MAX_CLOUDY = "1e9"
JEANS_CAP_CLOUDY = "3.086e20"
T_MIN_K = float(T_MIN_CLOUDY)
T_MAX_K = float(T_MAX_CLOUDY)
JEANS_CAP_CM = float(JEANS_CAP_CLOUDY)

# (output key, Cloudy parameter-file label, CIAOLoop map-file header)
LINES = (
    ("cii", "C  2 157.636m", "C_2_157.636m"),
    ("halpha", "H  1 6562.81A", "H_1_6562.81A"),
    ("hi21", "H  1 21.1207c", "H_1_21.1207c"),
    ("ciii_977", "C  3 977.020A", "C_3_977.020A"),
    ("ciii_1907", "C  3 1906.68A", "C_3_1906.68A"),
    ("ciii_1909", "C  3 1908.73A", "C_3_1908.73A"),
    ("civ_1548", "C  4 1548.19A", "C_4_1548.19A"),
    ("civ_1551", "C  4 1550.78A", "C_4_1550.78A"),
)
