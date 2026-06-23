# ============================================================================
# DEPRECATED 2026-06-23 — kept in-tree for reference (wrap-don't-delete).
# Standalone Hα comparison task (HalphaComparisonTask): never imported in
# tasks/__init__.py or registered.  Live Hα path = the H_alpha_luminosity field.
# ============================================================================
r'''
from __future__ import annotations

from typing import Dict

from ...plotting import create_plot
from ..base import AnalysisTask, PipelinePlotContext
from ..utils import make_axis_labels, shared_lognorm


class HalphaComparisonTask(AnalysisTask):
    """Plot Dust vs No-Dust maps with shared LogNorm and a ratio map."""

    def __init__(
        self,
        config,
        axis: str | None = None,
        figure_units: str | None = None,
        # epsilon_value: float = 1e-30,
    ):
        super().__init__(config)
        self.axis = axis or config.projection_axis
        self.figure_units = figure_units or config.figure_units
        self.xlabel, self.ylabel = make_axis_labels(self.axis, self.figure_units)
        # self.epsilon_value = epsilon_value
        self._extent = None

    def prepare(self, context: PipelinePlotContext) -> None:
        extent = context.results.get("halpha_extent")
        if extent is None:
            raise RuntimeError("HalphaComparisonTask requires previous H-alpha tasks to store 'halpha_extent'.")
        self._extent = extent[self.axis]

    def compute(self, context: PipelinePlotContext) -> Dict[str, object]:
        no_dust = context.results.get("halpha_no_dust")
        with_dust = context.results.get("halpha_with_dust")
        if no_dust is None or with_dust is None:
            raise RuntimeError("Missing H-alpha maps; run no-dust and dust tasks before comparison.")
        norm = shared_lognorm(no_dust, with_dust)
        # epsilon = no_dust.yt.quan(self.epsilon_value, no_dust.units)
        ratio = with_dust / (no_dust)
        return {"no_dust": no_dust, "with_dust": with_dust, "norm": norm, "ratio": ratio}

    def plot(self, context: PipelinePlotContext, results: Dict[str, object]) -> None:
        norm = results["norm"]
        for filename, data, title in [
            ("halpha_no_dust_shared.png", results["no_dust"], "H-alpha (No Dust)"),
            ("halpha_with_dust_shared.png", results["with_dust"], "H-alpha (Dust)"),
        ]:
            create_plot(
                data_2d=data.T.to_ndarray(),
                title=title,
                cbar_label=f"Surface Brightness ({data.units})",
                filename=str(self.config.output_dir / filename),
                extent=self._extent,
                xlabel=self.xlabel,
                ylabel=self.ylabel,
                norm=norm,
            )

        ratio = results["ratio"]
        create_plot(
            data_2d=ratio.T.to_ndarray(),
            title="Transmission (Dust / No Dust)",
            cbar_label="Fraction",
            filename=str(self.config.output_dir / "halpha_ratio.png"),
            extent=self._extent,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            norm=None,
            camp="viridis_r",
        )
'''
