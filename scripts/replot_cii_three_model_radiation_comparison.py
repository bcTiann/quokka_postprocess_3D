"""Replot the selected three-model CII radiation comparison."""
from pathlib import Path

import numpy as np
from yt.units.yt_array import YTArray

from plot_cii_defaultabund_radiation_cr_comparisons import _plot
from quokka2s.pipeline.spectrum_units import DSIGMA_DV_UNIT


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/plt0655228_down1_Lext15kpc"

SOURCES = {
    "column": OUTPUT / (
        "CII_default_abund_radiation_comparison_column_NHonly_Tsplit_Rinf_nozoom_"
        "with_DraineCR_DraineOnly_CROnly_noHM.npz"
    ),
    "jeans": OUTPUT / (
        "CII_default_abund_radiation_comparison_JeansLength_Tsplit_Rinf_nozoom_"
        "with_DraineCR_DraineOnly_CROnly_noHM.npz"
    ),
}

SELECTED = {
    "column": (
        "despotic", "hm2012_NH", "hm2012_draine_NH",
        "hm2012_draine_cr_NH",
    ),
    "jeans": (
        "despotic", "hm2012", "hm2012_draine", "hm2012_draine_cr",
    ),
}
LABELS = (
    "DESPOTIC",
    "Cloudy HM2012",
    "Cloudy HM2012 + Draine",
    "Cloudy HM2012 + Draine + CR",
)
STYLES = (
    ("#0072B2", "--"),
    ("#9467BD", "-"),
    ("#D55E00", "-"),
    ("#009E73", "-"),
)


for geometry, source in SOURCES.items():
    with np.load(source, allow_pickle=False) as bundle:
        velocity = np.asarray(bundle["velocity_kms"], dtype=float)
        source_unit = str(np.asarray(bundle["dsigma_dv_units"]).item())
        curves = YTArray(
            np.asarray(bundle["dsigma_dv"], dtype=float), source_unit,
        ).to(DSIGMA_DV_UNIT).value
        keys = tuple(str(value) for value in bundle["model_keys"])
    indices = tuple(keys.index(key) for key in SELECTED[geometry])
    selected_curves = curves[np.asarray(indices)]
    if geometry == "column":
        title = r"[C II] 158 $\mu$m ($N_{\rm H}, n_{\rm H}, T$), LOS y, $R=\infty$"
        filename = "CII_DESPOTIC_HM2012_HM2012Draine_HM2012DraineCR_column_NH_Tsplit_Rinf_cgs.png"
    else:
        title = r"[C II] 158 $\mu$m ($n_{\rm H}, T$; Jeans length), LOS y, $R=\infty$"
        filename = "CII_DESPOTIC_HM2012_HM2012Draine_HM2012DraineCR_JeansLength_Tsplit_Rinf_cgs.png"
    _plot(
        OUTPUT / filename,
        velocity,
        selected_curves,
        SELECTED[geometry],
        LABELS,
        STYLES,
        title,
    )
    print(OUTPUT / filename)
