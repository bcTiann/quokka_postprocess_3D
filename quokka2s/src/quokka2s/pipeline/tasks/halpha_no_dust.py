# ============================================================================
# DEPRECATED 2026-06-23 — kept in-tree for reference (wrap-don't-delete).
# Standalone Hα-no-dust task (HalphaNoDustTask): never imported or registered.
# ============================================================================
r'''
from __future__ import annotations

import numpy as np

from ...plotting import create_plot
from ...utils.axes import axis_index
from ..base import AnalysisTask, PipelinePlotContext
from ..utils import make_axis_labels


class HalphaNoDustTask(AnalysisTask):
    """Integrate H-alpha luminosity without dust attenuation."""

    def __init__(self, config, axis: str | None = None, figure_units: str | None = None):
        super().__init__(config)
        self.axis = axis
        self.axis_idx = axis_index(self.axis)
        self.figure_units = figure_units or config.figure_units
        self.xlabel, self.ylabel = make_axis_labels(self.axis, self.figure_units)
        self._lum_3d = None
        self._dx_3d = None
        self._extent = None
        self._table_lookup = None
        self._table_inputs = None

    def prepare(self, context: PipelinePlotContext) -> None:
        provider = context.provider
        self._lum_3d, self._extent = provider.get_slab_z(("gas", "Halpha_luminosity"))
        self._dx_3d, _ = provider.get_slab_z(("boxlib", "dx"))
        

    def compute(self, context: PipelinePlotContext):

        surface_brightness = np.sum(self._lum_3d * self._dx_3d, axis=self.axis_idx)
        if not np.isfinite(surface_brightness).any():
            raise RuntimeError("halpha_no_dust map has no finite values")
        context.results["halpha_no_dust"] = surface_brightness
        context.results["halpha_extent"] = self._extent
        return {"map": surface_brightness, "extent": self._extent[self.axis]}
    

    def plot(self, context: PipelinePlotContext, results):
        # pass
        output = self.config.output_dir / "halpha_no_dust_shared.png"
        create_plot(
            data_2d=results["map"].T.to_ndarray(),
            title="H-alpha (No Dust)",
            cbar_label=f"Surface Brightness ({results['map'].units})",
            filename=str(output),
            extent=results["extent"],
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            norm=self.config.extra_options.get("halpha_norm"),
        )
'''
