"""Compare edge-on and face-on hybrid [C II] surface spectra."""
from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import numpy as np

from ..base import BuildTask, PipelinePlotContext, PlotTask
from ..prep import config as _cfg
from ..spectrum_units import DSIGMA_DV_UNIT, dsigma_dv_ylabel


CPLUS_PROJECTION_FILENAME = "Cplus_LOS_y_vs_z_surface_spectrum.png"
CPLUS_PROJECTION_LOS = ("y", "z")
CPLUS_HYBRID_CONFIG = (
    {
        "name": "CPLUS_HYBRID_TOTAL",
        "freq_field": "C+_freq",
        "lum_field": "C+_luminosity",
        "width_field": "C+_thermal_width",
    },
)


class Build_CplusProjectionComparison(BuildTask):
    """Build the same hybrid [C II] model along y and z."""

    def __init__(self, config, R: float | None = None):
        super().__init__(config)
        self.R = R if R is not None else _cfg.SPECTRAL_RESOLUTION_R
        self.spectrum_schema = 1

    def compute(self, context: PipelinePlotContext) -> dict:
        from ..services import SpectrumStore

        provider = context.provider
        store = SpectrumStore(provider, species_config=CPLUS_HYBRID_CONFIG)
        spectra: dict[str, dict[str, np.ndarray]] = {}

        for los in CPLUS_PROJECTION_LOS:
            velocity, intrinsic = store.get_spectrum(
                "CPLUS_HYBRID_TOTAL", los, R=float("inf"),
            )
            _, observed = store.get_spectrum(
                "CPLUS_HYBRID_TOTAL", los, R=self.R,
            )
            spectra[los] = {
                "v_axis": velocity,
                "dsigma_dv": intrinsic,
                "dsigma_dv_obs": observed,
            }

        # Projected areas used by SpectrumStore for its surface normalization.
        widths_pc = np.asarray(context.ds.domain_width.to("pc"), dtype=float)
        projected_area_pc2 = {
            "y": float(widths_pc[0] * widths_pc[2]),  # x-z plane
            "z": float(widths_pc[0] * widths_pc[1]),  # x-y plane
        }

        del store
        provider._cached_grid = None
        gc.collect()
        return {
            "spectra": spectra,
            "projected_area_pc2": projected_area_pc2,
            "R": self.R,
            "dsigma_dv_units": DSIGMA_DV_UNIT,
        }


class Plot_CplusProjectionComparison(PlotTask):
    """Plot LOS-y and face-on LOS-z spectra on one absolute linear scale."""

    def _gather_inputs(self, context: PipelinePlotContext) -> dict:
        return self._load_one(context, "Build_CplusProjectionComparison")

    @staticmethod
    def _velocity_integral(velocity: np.ndarray, spectrum: np.ndarray) -> float:
        order = np.argsort(velocity)
        return float(np.trapezoid(spectrum[order], velocity[order]))

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        fig, axis = plt.subplots(1, 1, figsize=(7.6, 4.9))
        styles = {
            "y": ("#0072B2", "LOS y (x-z projected area)"),
            "z": ("#D55E00", "LOS z (face-on x-y area)"),
        }

        luminosities = {}
        for los in CPLUS_PROJECTION_LOS:
            block = results["spectra"][los]
            velocity = np.asarray(block["v_axis"])
            spectrum = np.asarray(block["dsigma_dv_obs"])
            sigma_l = self._velocity_integral(velocity, spectrum)
            luminosity = sigma_l * float(results["projected_area_pc2"][los])
            luminosities[los] = luminosity
            color, description = styles[los]
            axis.plot(
                velocity,
                spectrum,
                color=color,
                lw=1.7,
                drawstyle="steps-mid",
                label=(
                    f"{description}: "
                    rf"$\int\Sigma_L\,dv={sigma_l:.3g}\,L_\odot\,pc^{{-2}}$"
                ),
            )

        example = results["spectra"]["y"]["dsigma_dv_obs"]
        axis.axvline(0.0, color="0.55", ls=":", lw=0.8)
        axis.set_xlabel(r"Velocity [km s$^{-1}$]")
        axis.set_ylabel(dsigma_dv_ylabel(
            getattr(example, "units", results["dsigma_dv_units"])
        ))
        axis.ticklabel_format(
            style="sci", axis="y", scilimits=(0, 0), useMathText=True,
        )
        axis.grid(True, alpha=0.25, ls="--", lw=0.5)
        axis.legend(fontsize=8.2, frameon=False)
        mean_luminosity = float(np.mean(tuple(luminosities.values())))
        relative_spread = (
            abs(luminosities["y"] - luminosities["z"]) / mean_luminosity
        )
        axis.set_title(
            r"Hybrid [C II] 158 μm: edge-on versus face-on surface spectrum"
            "\n"
            rf"$L_{{\rm [CII]}}\simeq{mean_luminosity:.3g}\,L_\odot$ "
            rf"(projection difference {relative_spread:.2%}), $R={results['R']:.0e}$",
            fontsize=11,
        )
        fig.tight_layout()

        output = context.config.output_dir / CPLUS_PROJECTION_FILENAME
        fig.savefig(str(output), dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output}")
