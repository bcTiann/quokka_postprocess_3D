from __future__ import annotations

import numpy as np
from yt.units import cm
from matplotlib.colors import LogNorm
import yt 
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
        self._lum_3d = None
        self._rho_3d = None
        self._T_3d = None
        self._dx_3d = None
        self._extent = None
        self._CO_lum_3d = None
        self._Cplus_lum_3d = None
        self._HCOplus_lum_3d = None

    def prepare(self, context: PipelinePlotContext) -> None:
        provider = context.provider
        # 完全保留你的 get_slab_z
        self._CO_lum_3d, self._extent = provider.get_slab_z(("gas", "CO_luminosity"))
        self._Cplus_lum_3d, self._extent = provider.get_slab_z(("gas", "C+_luminosity"))
        self._HCOplus_lum_3d, self._extent = provider.get_slab_z(("gas", "HCO+_luminosity"))
        self._dx_3d, _ = provider.get_slab_z(("boxlib", "dx"))
        
        # 為了計算質量加權溫度，提取密度和溫度
        self._rho_3d, _ = provider.get_slab_z(("gas", "density"))
        self._T_3d, _ = provider.get_slab_z(("gas", "temperature"))

    def compute(self, context: PipelinePlotContext):


        ############### Luminosity (沿 X 視線累加) ################
        CO_surface_brightness = np.sum(self._CO_lum_3d * self._dx_3d, axis=0)
        Cplus_surface_brightness = np.sum(self._Cplus_lum_3d * self._dx_3d, axis=0)
        HCOplus_surface_brightness = np.sum(self._HCOplus_lum_3d * self._dx_3d, axis=0)
        
        ############### Temperature (質量加權平均) ################
        mass_column = np.sum(self._rho_3d * self._dx_3d, axis=0)
        T_proj = np.sum(self._T_3d * self._rho_3d * self._dx_3d, axis=0) / mass_column

        context.results["CO"] = CO_surface_brightness
        context.results["Cplus"] = Cplus_surface_brightness
        context.results["HCOplus"] = HCOplus_surface_brightness
        context.results["T_proj"] = T_proj

        return {
            "CO": CO_surface_brightness,
            "Cplus": Cplus_surface_brightness,
            "HCOplus": HCOplus_surface_brightness,
            "T_proj": T_proj,
            "extent": self._extent[self.axis],
        }

    def plot(self, context: PipelinePlotContext, results):
        extent = [float(v.to(self.figure_units).value) for v in results["extent"]]

        # 亮度共用比例尺
        shared_norm = shared_lognorm(
            results["CO"],
            results["Cplus"],
            results["HCOplus"],
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
                "title": "C+ Emission",  # 修正拼字
                "label": f"Surface Brightness ({unit_label})",
                "norm": LogNorm(),
                "data_top": results["Cplus"].in_cgs().to_ndarray().T,
            },
            {
                "title": "HCO+ Emission",  # 修正拼字
                "label": f"Surface Brightness ({unit_label})",
                "norm": LogNorm(),
                "data_top": results["HCOplus"].in_cgs().to_ndarray().T,
            },
            {
                "title": "Mass-Weighted Temperature",
                "label": "Temperature (K)",
                "norm": LogNorm(),  # 溫度必須用獨立的比例尺
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



        # --- 新增：PhasePlot 診斷邏輯 ---
        print("Generating Temperature Error Phase Plot...")
        
        # 獲取原始的 yt dataset 對象
        # 註：視你的 pipeline 實作而定，通常可以從 context.provider.ds 取得
        ds = context.provider.ds 
        
        # 取出數據。如果你只想看目前處理的這個 slab，可以使用 context.provider 提供的 region
        # 這裡示範使用全部數據 (all_data)
        ad = ds.all_data()
        
        # 建立相圖：nH vs ColumnDensity，顏色代表溫度誤差
        phase = yt.PhasePlot(
            ad,
            ('gas', 'number_density_H'),
            ('gas', 'column_density_H'),
            ('gas', 'temperature_error'),
            weight_field=None,  # 看單格平均誤差，不依質量加權
        )
        
        # 設定座標軸與配色
        phase.set_log(('gas', 'temperature_error'), True)
        phase.set_zlim(('gas', 'temperature_error'), 1e-3, 1.0)
        phase.set_cmap(('gas', 'temperature_error'), 'inferno')
        
        # 存檔路徑
        phase_path = self.config.output_dir / "temperature_error_phase.png"
        phase.save(str(phase_path))
        print(f"Diagnostic PhasePlot saved to: {phase_path}")