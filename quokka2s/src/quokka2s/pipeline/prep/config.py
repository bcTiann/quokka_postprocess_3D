# config.py
from matplotlib.colors import LogNorm
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg 

# --- Input/Output ---
YT_DATASET_PATH = "~/quokka_postprocessing/examples/halpha_analysis/plt263168"
DESPOTIC_TABLE_PATH = "/Users/baochen/quokka_postprocessing/output_tables_testBig/despotic_table.npz"
OUTPUT_DIR = "plots/"

# --- Physics Parameters / Model Assumptions ---
X_H = 0.74  # Mass fraction of Hydrogen
A_LAMBDA_OVER_NH = 4e-22 * cm**2  # Dust extinction cross-section (mag * cm^2 / N_H)

# --- Simulation Control ---
PROJECTION_AXIS = 'x'     # Axis for projection ('x', 'y', or 'z')

# --- Plotting Parameters ---
FIGURE_UNITS = 'pc'
CMAP = 'viridis'


