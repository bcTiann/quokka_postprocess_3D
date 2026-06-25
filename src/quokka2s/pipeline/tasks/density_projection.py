"""Density projection task."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LogNorm

from ...plotting import create_plot
from ...utils.axes import axis_index
from ..base import AnalysisTask, PipelinePlotContext
from ..utils import make_axis_labels


class DensityProjectionTask(AnalysisTask):
    """Simple density projection along the chosen axis."""

    def __init__(
        self,
        config,
        axis: str | None = None,
        figure_units: str | None = None,
        title: str = "Density Projection",
        filename: str = "density.png",
    ) -> None:
        super().__init__(config)
        self.axis = axis
        self.axis_idx = axis_index(self.axis)
        self.figure_units = figure_units or config.figure_units
        self.xlabel, self.ylabel = make_axis_labels(self.axis, self.figure_units)
        self.title = title
        self.filename = filename
        self._norm = LogNorm()

    def compute(self, context: PipelinePlotContext):
        rho_3d, extent = context.provider.get_slab_z(("gas", "density"))
        density_projection = np.sum(rho_3d, axis=self.axis_idx)
        return {"map": density_projection, "extent": extent[self.axis]}

    def plot(self, context: PipelinePlotContext, results):
        output = self.config.output_dir / self.filename
        create_plot(
            data_2d=results["map"].T.to_ndarray(),
            title=self.title,
            cbar_label=f"Density ({results['map'].units})",
            filename=str(output),
            extent=results["extent"],
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            norm=self._norm,
        )
