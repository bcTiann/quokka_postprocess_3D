from __future__ import annotations

import numpy as np
from yt.units import cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from ...analysis import calculate_attenuation, calculate_cumulative_column_density
from ...plotting import create_plot, plot_multiview_grid
from ...utils.axes import axis_index
from ..base import AnalysisTask, PipelinePlotContext
from ..utils import make_axis_labels, shared_lognorm


class EmitterTask(AnalysisTask):
    """Integrate Emitter luminosity and calculate projected temperature."""

    def __init__(self, config, axis: str | None = None, figure_units: str | None = None):
        super().__init__(config)
        self.axis = axis
        self.axis_idx = axis_index(self.axis)
        self.figure_units = figure_units or config.figure_units
        self.xlabel, self.ylabel = make_axis_labels(self.axis, self.figure_units)
        self._rho_3d = None
        self._T_3d = None
        self._dx_3d = None
        self._extent = None
        self._CO_lum_3d = None
        self._Cplus_lum_3d = None
        self._Halpha_lum_3d = None
        self._HI_lum_3d = None

    def prepare(self, context: PipelinePlotContext) -> None:
        provider = context.provider
        self._CO_lum_3d,     self._extent = provider.get_slab_z(("gas", "CO_luminosity"))
        self._Cplus_lum_3d,  _            = provider.get_slab_z(("gas", "C+_luminosity"))
        self._Halpha_lum_3d, _            = provider.get_slab_z(("gas", "H_alpha_luminosity"))
        self._HI_lum_3d,     _            = provider.get_slab_z(("gas", "HI_luminosity"))
        self._dx_3d, _ = provider.get_slab_z(("boxlib", "dx"))

        # density-weighted projected temperature
        self._rho_3d, _ = provider.get_slab_z(("gas", "density"))
        self._T_3d, _   = provider.get_slab_z(("gas", "temperature_despotic"))

    def compute(self, context: PipelinePlotContext):

        ############### Luminosity (LOS sum) ################
        CO_sb     = np.sum(self._CO_lum_3d     * self._dx_3d, axis=0)
        Cplus_sb  = np.sum(self._Cplus_lum_3d  * self._dx_3d, axis=0)
        Halpha_sb = np.sum(self._Halpha_lum_3d * self._dx_3d, axis=0)
        HI_sb     = np.sum(self._HI_lum_3d     * self._dx_3d, axis=0)

        ############### Density-weighted T projection ################
        mass_column = np.sum(self._rho_3d * self._dx_3d, axis=0)
        T_proj = np.sum(self._T_3d * self._rho_3d * self._dx_3d, axis=0) / mass_column

        context.results["CO"]     = CO_sb
        context.results["Cplus"]  = Cplus_sb
        context.results["Halpha"] = Halpha_sb
        context.results["HI"]     = HI_sb
        context.results["T_proj"] = T_proj

        return {
            "CO":     CO_sb,
            "Cplus":  Cplus_sb,
            "Halpha": Halpha_sb,
            "HI":     HI_sb,
            "T_proj": T_proj,
            "extent": self._extent[self.axis],
        }

    def plot(self, context: PipelinePlotContext, results):
        extent = [float(v.to(self.figure_units).value) for v in results["extent"]]

        unit_label = str(results["CO"].in_cgs().units)

        plots_info = [
            {
                "title": "CO Emission",
                "label": f"Surface Brightness ({unit_label})",
                "norm": LogNorm(),
                "data_top": results["CO"].in_cgs().to_ndarray().T,
            },
            {
                "title": "C+ Emission",
                "label": f"Surface Brightness ({unit_label})",
                "norm": LogNorm(),
                "data_top": results["Cplus"].in_cgs().to_ndarray().T,
            },
            {
                "title": "H-alpha Emission",
                "label": f"Surface Brightness ({unit_label})",
                "norm": LogNorm(),
                "data_top": results["Halpha"].in_cgs().to_ndarray().T,
            },
            {
                "title": "HI 21 cm Emission",
                "label": f"Surface Brightness ({unit_label})",
                "norm": LogNorm(),
                "data_top": results["HI"].in_cgs().to_ndarray().T,
            },
            {
                "title": "Density-Weighted Temperature",
                "label": "Temperature (K)",
                "norm": LogNorm(),
                "data_top": results["T_proj"].to_ndarray().T,
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



        # The bisection-error diagnostic is obsolete: with the (nH, NH, dVdr)
        # table T is read directly via interpolation, not solved via bisection.
        # Removed along with the temperature_error / temperature_eint fields.