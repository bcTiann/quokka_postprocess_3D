from __future__ import annotations

import numpy as np
from yt.units import cm
from matplotlib.colors import LogNorm

from ...analysis import calculate_attenuation, calculate_cumulative_column_density
from ...plotting import create_plot, plot_multiview_grid
from ...utils.axes import axis_index
from ..base import AnalysisTask, PipelinePlotContext
from ..utils import make_axis_labels, shared_lognorm


class EmitterTask(AnalysisTask):
    """Integrate Emitter luminosity without dust attenuation."""

    def __init__(self, config, axis: str | None = None, figure_units: str | None = None):
        super().__init__(config)
        self.axis = axis
        self.axis_idx = axis_index(self.axis)
        self.figure_units = figure_units or config.figure_units
        self.xlabel, self.ylabel = make_axis_labels(self.axis, self.figure_units)
        self._lum_3d = None
        self._rho_3d = None
        self._dx_3d = None
        self._extent = None
        self._CO_lum_3d = None
        self._Cplus_lum_3d = None


    def prepare(self, context: PipelinePlotContext) -> None:
        provider = context.provider
        self._CO_lum_3d, self._extent = provider.get_slab_z(("gas", "CO_luminosity"))
        self._Cplus_lum_3d, self._extent = provider.get_slab_z(("gas", "C+_luminosity"))
        # self._rho_3d, _ = provider.get_slab_z(("gas", "density"))
        self._dx_3d, _ = provider.get_slab_z(("boxlib", "dx"))
        # self._dy_3d, self._bottom_extent = provider.get_slab_z(("boxlib", "dy"))
        # self._dz_3d, _ = provider.get_slab_z(("boxlib", "dz"))

    def compute(self, context: PipelinePlotContext):

        ############### No Dust ################
        CO_surface_brightness = np.sum(self._CO_lum_3d * self._dx_3d, axis=0)
        Cplus_surface_brightness = np.sum(self._Cplus_lum_3d * self._dx_3d, axis=0)
        context.results["CO"] = CO_surface_brightness
        context.results["Cplus"] = Cplus_surface_brightness

        

        return {
            "CO": CO_surface_brightness,
            "Cplus": Cplus_surface_brightness,
            "extent": self._extent[self.axis],
        }


    def plot(self, context: PipelinePlotContext, results):
        extent = [float(v.to(self.figure_units).value) for v in results["extent"]]

        shared_norm = shared_lognorm(
            results["CO"],
            results["Cplus"],
        )

        unit_label = str(results["CO"].in_cgs().units)

        plots_info = [
            {
                "title": "CO Emission",
                "label": f"Surface Brightness ({unit_label})",
                "norm": LogNorm(),
                "data_top": results["CO"].in_cgs().to_ndarray().T,
            },
            {
                "title": "C+ Emssion",
                "label": f"Surface Brightness ({unit_label})",
                "norm": LogNorm(),
                "data_top": results["Cplus"].in_cgs().to_ndarray().T,
            },
        ]

        plot_multiview_grid(
            plots_info=plots_info,
            extent_top=extent,
            filename=str(self.config.output_dir / "emitter.png"),
            top_ylabel=self.ylabel,
            top_xlabel=self.xlabel,
            include_bottom=False,
            units=self.figure_units,
        )

        