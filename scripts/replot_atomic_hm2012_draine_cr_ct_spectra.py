#!/usr/bin/env python3
"""Replot existing atomic spectra with both CR and CR+charge-transfer states."""

from pathlib import Path

import numpy as np

from plot_atomic_hm2012_draine_cr_ct_spectra import (
    NEW_LABEL,
    SPECIES,
    SPECIES_TITLES,
    _plot,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/plt0655228_down1_Lext15kpc/atomic3line_rerun"


def main() -> None:
    with np.load(
        OUTPUT / "atomic_hm2012_draine_cr_ct_Tsplit_Rinf.npz",
        allow_pickle=False,
    ) as table:
        velocity = np.asarray(table["velocity_kms"], dtype=float)
        ct_spectra = np.asarray(table["dsigma_dv"], dtype=float)

    for species_index, species in enumerate(SPECIES):
        for geometry_index, (geometry, title_geometry) in enumerate((
            ("column", r"$(N_{\rm H}, n_{\rm H}, T)$"),
            ("JeansLength", r"$(n_{\rm H}, T;\ \mathrm{Jeans\ length})$"),
        )):
            if species == "cii":
                source_name = (
                    "CII_default_abund_radiation_comparison_column_NHonly_Tsplit_Rinf_nozoom.npz"
                    if geometry == "column"
                    else "CII_default_abund_radiation_comparison_JeansLength_Tsplit_Rinf_nozoom.npz"
                )
                labels = (
                    "DESPOTIC",
                    "Cloudy HM2012",
                    "Cloudy HM2012 + Draine",
                    "Cloudy HM2012 + Draine + CR",
                    NEW_LABEL,
                )
            else:
                source_name = (
                    f"{species}_pipeline_vs_default_abund_radiation_"
                    f"{geometry}_Tsplit_Rinf_nozoom.npz"
                )
                labels = (
                    "pipeline",
                    "Cloudy HM2012",
                    "Cloudy HM2012 + Draine",
                    "Cloudy HM2012 + Draine + CR",
                    NEW_LABEL,
                )

            with np.load(OUTPUT / source_name, allow_pickle=False) as table:
                old_velocity = np.asarray(table["velocity_kms"], dtype=float)
                old = np.asarray(table["dsigma_dv"], dtype=float)

            if not np.allclose(old_velocity, velocity, rtol=0.0, atol=1.0e-10):
                if np.allclose(old_velocity[::-1], velocity, rtol=0.0, atol=1.0e-10):
                    old = old[..., ::-1]
                else:
                    raise ValueError(f"velocity axes differ for {source_name}")

            curves = np.concatenate((
                old[:4],
                ct_spectra[species_index, geometry_index][None],
            ))
            output = OUTPUT / (
                f"{species}_HM2012_Draine_CR_and_CT_{geometry}_Tsplit_Rinf.png"
            )
            _plot(
                output,
                velocity,
                curves,
                labels,
                rf"{SPECIES_TITLES[species]} {title_geometry}, LOS y, $R=\infty$",
            )
            print(output)


if __name__ == "__main__":
    main()
