from __future__ import annotations

import numpy as np
from yt.units import cm, kb, mh, s, m, km
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

from quokka2s.tables import load_table
from quokka2s.tables.lookup import TableLookup
from quokka2s.pipeline.prep import config as cfg
import matplotlib.pyplot as plt
from tqdm import tqdm 

from ...analysis import calculate_attenuation, calculate_cumulative_column_density
from ...plotting import create_plot, plot_multiview_grid
from ...utils.axes import axis_index
from ..base import AnalysisTask, PipelinePlotContext
from ..utils import make_axis_labels, shared_lognorm
from quokka2s.pipeline.prep.physics_fields import get_one_sightline_spectrum


class CplusLine1DTask(AnalysisTask):
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
        self._T = None

    def prepare(self, context: PipelinePlotContext) -> None:
        provider = context.provider
        
        self._Bulk_Doppler_factor_x, _ = provider.get_slab_z(("gas", "Bulk_Doppler_factor_x"))
        self._CO_freq_field, _ = provider.get_slab_z(("gas", "CO_freq"))
        self._Cplus_freq_field, _ = provider.get_slab_z(("gas", "C+_freq"))
        self._HCOplus_freq_field, _ = provider.get_slab_z(("gas", "HCO+_freq"))

        self._CO_lum_3d, self._extent = provider.get_slab_z(("gas", "CO_luminosity"))
        self._Cplus_lum_3d, self._extent = provider.get_slab_z(("gas", "C+_luminosity"))
        self._HCOplus_lum_3d, self._extent = provider.get_slab_z(("gas", "HCO+_luminosity"))

        self._CO_thermal_width, _ = provider.get_slab_z(("gas", "CO_thermal_width"))
        self._Cplus_thermal_width, _ = provider.get_slab_z(("gas", "C+_thermal_width"))
        self._HCOplus_thermal_width, _ = provider.get_slab_z(("gas", "HCO+_thermal_width"))

        self._dx_3d, _ = provider.get_slab_z(("boxlib", "dx"))
        self._dy_3d, _ = provider.get_slab_z(("boxlib", "dy"))
        self._dz_3d, _ = provider.get_slab_z(("boxlib", "dz"))
        self._volume_3d = self._dx_3d * self._dy_3d * self._dz_3d
        self._area_x = self._dy_3d * self._dz_3d


    def compute(self, context: PipelinePlotContext):
        c = 3.0e8 * m / s
        # the freq center is the original unshifted freq of CO J 0 -> 1
        nu_0 = self._Cplus_freq_field[0, 0, 0].in_units("Hz")
        assert (self._Cplus_freq_field[0, 0, 0].in_units("Hz") == self._Cplus_freq_field[1, 90, 20].in_units("Hz")), "nu0 frequency should be same everywhere!"
        
        freq_3d = self._Cplus_freq_field.in_units("Hz")
        
        shifted_freq_3d = (self._Cplus_freq_field * self._Bulk_Doppler_factor_x).in_units("Hz")

        delta_freq_3d = shifted_freq_3d - freq_3d
        delta_freq_yz = np.sum(delta_freq_3d, axis=0)
        

        plt.figure(figsize=(12, 12))
        # 画图时用 sb_map
        plt.imshow(delta_freq_yz.T, origin='lower', cmap='viridis')
        plt.colorbar(label="delta_freq in yz surface(Hz)$]")
        out_path = self.config.output_dir / "C+_delta_freq_yz_surface.png"
        plt.savefig(str(out_path), dpi=600)
        print(f"Saved: {out_path}")

        lum_3d = (self._Cplus_lum_3d * self._volume_3d).in_units("erg/s")

        thermal_3d = self._Cplus_thermal_width.in_units("cm/s")

        nx, ny, nz = freq_3d.shape


        v_range = 50.0 * km / s # km/s
        
        bw_hz = nu_0 * (v_range / c) * 2.0
  
        n_channels = 1000

        
        freq_edges = np.linspace(nu_0 - bw_hz/2, nu_0 + bw_hz/2, n_channels + 1)
        freq_centers = 0.5 * (freq_edges[:-1] + freq_edges[1:])

        # Sepctrum cube (freq_channels, ny, nz)
        spec_cube = np.zeros((n_channels, ny, nz))

        print(f"Looping over {ny}x{nz} pixels with yt.units...")

        for j in tqdm(range(ny), desc="Rows"):
            for k in range(nz):
                f_line = shifted_freq_3d[:, j, k]
                l_line = lum_3d[:, j, k]
                t_line = thermal_3d[:, j, k]

                spec_1d = get_one_sightline_spectrum(
                    f_line, l_line, t_line,
                    freq_centers
                )

                spec_cube[:, j, k] = spec_1d.value


        # Plotting
        d_nu = (freq_edges[1] - freq_edges[0]).in_units("Hz").value
        # 1. 算出总光度 [erg/s]
        lum_map = np.sum(spec_cube, axis=0) * d_nu


        # 2. 获取像素面积 [cm^2]
        # 注意：这里假设 dy 和 dz 是常数或者对应的 2D 数组
        # 如果是均匀网格：
        pixel_area = self._area_x[0,0,0].in_units("cm**2").value
        
        # 3. 算出表面亮度 [erg/s/cm^2]
        sb_map = lum_map / pixel_area
        
        plt.figure(figsize=(12, 12))
        # 画图时用 sb_map
        plt.imshow(sb_map.T, origin='lower', cmap='viridis', norm=LogNorm())
        plt.colorbar(label="Surface Brightness [erg s$^{-1}$ cm$^{-2}$]")
        out_path = self.config.output_dir / "C+_Surface_Brightness.png"
        plt.savefig(str(out_path), dpi=300)
        print(f"Saved: {out_path}")


        print("Plotting Spectral Grid Map...")

        # 1. 准备 X 轴数据 (Velocity) - 使用 yt.units 处理
        # freq_centers: 之前生成的 YTArray [Hz]
        # nu_0: 之前定义的基准频率 YTQuantity [Hz]
        # 公式: v = c * (nu_rest - nu_obs) / nu_rest (射电天文学惯用定义)
        
        # 这一步 yt 会自动处理 Hz/Hz 抵消，剩下速度单位
        v_axis = c* (nu_0 - freq_centers) / nu_0
        
        # 转成 km/s 并剥离单位给 matplotlib
        v_axis_kms = v_axis.in_units("km/s").value
        
        # 2. 设定采样间隔 (均匀切分 5 份)
        n_grid = 10
        
        y_sampling = np.linspace(0, ny - 1, n_grid, dtype=int)
        z_sampling = np.linspace(0, nz - 1, n_grid, dtype=int)

        fig, axes = plt.subplots(n_grid, n_grid, figsize=(18, 24),
        sharex=True, sharey=False)

        axes_natural = np.flipud(axes)

        for i, z_idx in tqdm(enumerate(z_sampling), desc="Z-axis"):
            for j, y_idx in enumerate(y_sampling):

                ax = axes_natural[i, j]

                local_spec = spec_cube[:, y_idx, z_idx]

                ax.plot(v_axis_kms, local_spec, color='royalblue', lw=1.5, drawstyle='steps-mid')
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
                # ax.set_ylim(0, 1e11)

                ax.text(0.05, 0.85, f"Index({y_idx},{z_idx})", transform=ax.transAxes, 
                        fontsize=7, fontweight='bold')
                
                ax.grid(True, alpha=0.3, ls='--')
                ax.axvline(0, color='k', linestyle=':', alpha=0.5)

                
        fig.suptitle(f"C+ Spectral Grid Map", fontsize=18, y=0.92)
        fig.text(0.5, 0.05, "Velocity [km/s]", ha='center', fontsize=14)
        fig.text(0.08, 0.5, "Luminosity Density [erg/s/Hz]", va='center', rotation='vertical', fontsize=14)
        
        out_path = self.config.output_dir / "C+_Spectral_Grid.png"
        # wspace=0 可以让左右子图紧挨着，看起来更有“连续星系”的感觉
        plt.subplots_adjust(wspace=0.1, hspace=0.1) 
        plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
        print(f"Saved: {out_path}")

        return None


    def plot(self, context: PipelinePlotContext, results):
        return None
    

        