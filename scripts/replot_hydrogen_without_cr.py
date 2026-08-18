"""Replot the four established Halpha/HI comparisons without the CR curve."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quokka2s.pipeline.spectrum_units import DSIGMA_DV_UNIT, dsigma_dv_ylabel


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "output/plt0655228_down1_Lext15kpc/atomic3line_rerun"
KEEP = ("pipeline", "hm2012", "hm2012_draine")
LABELS = {
    "pipeline": "pipeline",
    "hm2012": "Cloudy HM2012",
    "hm2012_draine": "Cloudy HM2012 + Draine",
}
STYLES = {
    "pipeline": ("#0072B2", "--"),
    "hm2012": ("#9467BD", "-"),
    "hm2012_draine": ("#D55E00", "-"),
}


def plot_one(species: str, geometry: str) -> Path:
    source = INPUT_DIR / (
        f"{species}_pipeline_vs_default_abund_radiation_"
        f"{geometry}_Tsplit_Rinf_nozoom.npz"
    )
    with np.load(source, allow_pickle=False) as table:
        velocity = np.asarray(table["velocity_kms"], dtype=float)
        spectra = np.asarray(table["dsigma_dv"], dtype=float)
        models = tuple(str(value) for value in table["model_keys"].tolist())
        regimes = np.asarray(table["regime_keys"])
        units = str(np.asarray(table["dsigma_dv_units"]).item())
    if units != DSIGMA_DV_UNIT:
        raise ValueError(f"unexpected spectrum units {units!r}: {source}")
    curves = np.stack(tuple(spectra[models.index(model)] for model in KEEP))

    species_title = r"H$\alpha$" if species == "halpha" else "H I 21 cm"
    geometry_title = (
        r"$(N_{\rm H}, n_{\rm H}, T)$"
        if geometry == "column"
        else r"$(n_{\rm H}, T;\ \mathrm{Jeans\ length})$"
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    shared_max = float(np.nanmax(curves))
    for branch, axis in enumerate(axes):
        for model_index, model in enumerate(KEEP):
            color, linestyle = STYLES[model]
            axis.plot(
                velocity,
                curves[model_index, branch],
                color=color,
                linestyle=linestyle,
                linewidth=1.75,
                drawstyle="steps-mid",
                label=LABELS[model],
            )
        axis.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
        axis.set_xlabel(r"Velocity [km s$^{-1}$]")
        axis.set_ylabel(dsigma_dv_ylabel(DSIGMA_DV_UNIT))
        axis.set_title(
            r"$T_{\rm QUOKKA}<3000\,$K"
            if branch == 0
            else r"$T_{\rm QUOKKA}\geq3000\,$K"
        )
        axis.set_ylim(0.0, 1.05 * shared_max)
        axis.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        axis.legend(fontsize=8.5, frameon=False)
        axis.ticklabel_format(
            style="sci", axis="y", scilimits=(0, 0), useMathText=True,
        )
        axis.tick_params(axis="y", labelleft=True)
    fig.suptitle(rf"{species_title} {geometry_title}, LOS y, $R=\infty$")
    fig.tight_layout()

    output = source.with_name(source.stem + "_without_CR.png")
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(
        output.with_suffix(".npz"),
        velocity_kms=velocity,
        dsigma_dv=curves,
        model_keys=np.asarray(KEEP),
        regime_keys=regimes,
        dsigma_dv_units=np.asarray(units),
        source=np.asarray(str(source)),
    )
    print(f"Saved: {output}")
    return output


def main() -> None:
    for species in ("halpha", "hi21"):
        for geometry in ("column", "JeansLength"):
            plot_one(species, geometry)


if __name__ == "__main__":
    main()
