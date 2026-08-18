"""Replot the default-abundance Jeans CII spectra with concise labels."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quokka2s.pipeline.prep import config as cfg
from quokka2s.pipeline.spectrum_units import DSIGMA_DV_UNIT, dsigma_dv_ylabel


SOURCE = (
    Path(cfg.OUTPUT_DIR)
    / "CII_default_abund_HM2012_Draine_CR_JeansLength_Tsplit_Rinf_nozoom.npz"
)
OUTPUT_STEM = (
    "CII_default_abund_HM2012_Draine_CR_JeansLength_"
    "Tsplit_Rinf_nozoom_cleanlabels"
)
SELECTED = ("despotic", "hm2012", "hm2012_draine", "hm2012_draine_cr")
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


def main() -> None:
    with np.load(SOURCE, allow_pickle=False) as table:
        velocity = np.asarray(table["velocity_kms"], dtype=float)
        all_curves = np.asarray(table["dsigma_dv"], dtype=float)
        all_keys = tuple(str(value) for value in table["model_keys"].tolist())
        regimes = np.asarray(table["regime_keys"])
        units = str(np.asarray(table["dsigma_dv_units"]).item())
    if units != DSIGMA_DV_UNIT:
        raise ValueError(f"unexpected spectrum units: {units}")
    curves = np.stack([all_curves[all_keys.index(key)] for key in SELECTED])

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    shared_max = float(np.nanmax(curves))
    for branch, axis in enumerate(axes):
        for index, (label, style) in enumerate(zip(LABELS, STYLES)):
            axis.plot(
                velocity, curves[index, branch], color=style[0],
                linestyle=style[1], linewidth=1.8, drawstyle="steps-mid",
                label=label,
            )
        axis.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
        axis.set_xlabel(r"Velocity [km s$^{-1}$]")
        axis.set_ylabel(dsigma_dv_ylabel(DSIGMA_DV_UNIT))
        axis.set_title(
            r"$T_{\rm QUOKKA}<3000\,$K" if branch == 0
            else r"$T_{\rm QUOKKA}\geq3000\,$K"
        )
        axis.set_ylim(0.0, 1.05 * shared_max)
        axis.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        axis.legend(fontsize=8.5, frameon=False)
        axis.ticklabel_format(
            style="sci", axis="y", scilimits=(0, 0), useMathText=True,
        )
        axis.tick_params(axis="y", labelleft=True)
    fig.suptitle(
        r"[C II] 158 $\mu$m $(n_{\rm H}, T)$, LOS y, $R=\infty$"
    )
    fig.tight_layout()

    output_dir = Path(cfg.OUTPUT_DIR)
    png = output_dir / f"{OUTPUT_STEM}.png"
    npz = output_dir / f"{OUTPUT_STEM}.npz"
    if png.exists() or npz.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_STEM}")
    fig.savefig(png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(
        npz, velocity_kms=velocity, dsigma_dv=curves,
        model_keys=np.asarray(SELECTED), regime_keys=regimes,
        dsigma_dv_units=np.asarray(DSIGMA_DV_UNIT),
        completed_full_domain=np.asarray(True),
        source=np.asarray(str(SOURCE.resolve())),
    )
    print(f"Saved: {png}")
    print(f"Saved: {npz}")


if __name__ == "__main__":
    main()
