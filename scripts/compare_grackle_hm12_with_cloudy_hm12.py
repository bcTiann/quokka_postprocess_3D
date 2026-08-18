#!/usr/bin/env python3
"""Read two existing Cloudy continuum files and plot them."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parents[1]
data_dir = (
    root / "work/cloudy_cooling_tools_history/examples/grackle/HM12_ISM_NH21"
)

# Cloudy save incident continuum columns:
#   column 0: photon energy [Ryd]
#   column 1: nu * 4 pi * J_nu [erg cm^-2 s^-1]
grackle = np.loadtxt(data_dir / "export_hm12_external.inc")
cloudy = np.loadtxt(data_dir / "export_hm12_native.inc")

energy = grackle[:, 0]
grackle_spectrum = grackle[:, 1]
cloudy_spectrum = cloudy[:, 1]

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.loglog(
    energy,
    np.where(grackle_spectrum > 0, grackle_spectrum, np.nan),
    label="Grackle HM2012 z=0",
    color="tab:blue",
    linewidth=2,
)
ax.loglog(
    energy,
    np.where(cloudy_spectrum > 0, cloudy_spectrum, np.nan),
    label="Cloudy table HM12 redshift 0",
    color="tab:orange",
    linestyle="--",
    linewidth=2,
)
ax.set_xlabel("Photon energy [Ryd]")
ax.set_ylabel(r"$\nu\,4\pi J_\nu$ [erg cm$^{-2}$ s$^{-1}$]")
ax.set_title("HM2012 spectrum: Grackle file vs Cloudy command")
ax.legend(frameon=False)
ax.grid(alpha=0.2, which="both")
fig.tight_layout()

output = root / "output/radiation_fields/hm12_grackle_vs_cloudy_simple.png"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=220)
print(output)
