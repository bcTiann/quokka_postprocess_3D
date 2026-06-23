from __future__ import annotations

import numpy as np
from yt.units import m, s, km
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from concurrent.futures import ThreadPoolExecutor

from ..prep.physics_fields import build_spectral_cube
from ..prep import config as cfg

from ...utils.axes import axis_index
from ..base import AnalysisTask, PipelinePlotContext
from ..utils import make_axis_labels


SPECIES_CFG = [
    {
        'name': 'CO',
        'freq_field':  'CO_freq',
        'lum_field':   'CO_luminosity',
        'width_field': 'CO_thermal_width',
        'color': 'royalblue',
    },
    {
        'name': 'C+',
        'freq_field':  'C+_freq',
        'lum_field':   'C+_luminosity',
        'width_field': 'C+_thermal_width',
        'color': 'forestgreen',
    },
    {
        'name': 'HCO+',
        'freq_field':  'HCO+_freq',
        'lum_field':   'HCO+_luminosity',
        'width_field': 'HCO+_thermal_width',
        'color': 'crimson',
    },
]


class TripleLineTask(AnalysisTask):
    """Overlay CO, C+, HCO+ spectra on a shared 10x10 pixel grid.

    Uses sharex=True, sharey=True so amplitude differences between
    species are visible directly (no per-panel auto-scaling).
    """

    def __init__(self, config, axis: str | None = None, figure_units: str | None = None):
        super().__init__(config)
        self.axis = axis or 'x'
        self.axis_idx = axis_index(self.axis)
        self.figure_units = figure_units or config.figure_units
        self.xlabel, self.ylabel = make_axis_labels(self.axis, self.figure_units)
        self._doppler = None
        self._volume_3d = None
        self._area_x = None
        self._sp_data: dict[str, dict] = {}

    def prepare(self, context: PipelinePlotContext) -> None:
        provider = context.provider

        self._doppler, _ = provider.get_slab_z(('gas', 'Bulk_Doppler_factor_x'))

        dx, _ = provider.get_slab_z(('boxlib', 'dx'))
        dy, _ = provider.get_slab_z(('boxlib', 'dy'))
        dz, _ = provider.get_slab_z(('boxlib', 'dz'))
        self._volume_3d = dx * dy * dz
        self._area_x = dy * dz

        for sp in SPECIES_CFG:
            name = sp['name']
            freq,  _ = provider.get_slab_z(('gas', sp['freq_field']))
            lum,   _ = provider.get_slab_z(('gas', sp['lum_field']))
            width, _ = provider.get_slab_z(('gas', sp['width_field']))
            self._sp_data[name] = {'freq': freq, 'lum': lum, 'width': width}

    def _compute_one_species(self, sp_name: str, n_channels: int, c_cms: float) -> tuple:
        """Build the spectral cube for a single species.

        Runs independently of the other two species, so it can be dispatched
        to a thread.  numpy releases the GIL for exp/multiply/etc., so three
        threads genuinely run in parallel on separate CPU cores.

        Returns (sp_name, spec_cube, v_axis) so the caller can collect results
        in any order.
        """
        c = 3.0e8 * m / s
        v_range = 50.0 * km / s

        freq_3d = self._sp_data[sp_name]['freq'].in_units('Hz')
        nu_0 = freq_3d[0, 0, 0]

        lum_3d = (self._sp_data[sp_name]['lum'] * self._volume_3d).in_units('erg/s')
        thermal_3d = self._sp_data[sp_name]['width'].in_units('cm/s')
        shifted_freq_3d = (freq_3d * self._doppler).in_units('Hz')

        bw_hz = nu_0 * (v_range / c) * 2.0
        freq_edges = np.linspace(nu_0 - bw_hz / 2, nu_0 + bw_hz / 2, n_channels + 1)
        freq_centers = 0.5 * (freq_edges[:-1] + freq_edges[1:])

        nx, ny, nz = freq_3d.shape

        shifted_val = shifted_freq_3d.in_units('Hz').value
        lum_val     = lum_3d.in_units('erg/s').value
        thermal_val = thermal_3d.in_units('cm/s').value
        freq_edges_hz = freq_edges.in_units('Hz').value

        print(f'  [{sp_name}] building spectral cube (erf integration, {nx} slices) ...')
        spec_cube = build_spectral_cube(shifted_val, lum_val, thermal_val,
                                        freq_edges_hz, c_cms)
        print(f'  [{sp_name}] done.')

        v_axis = (c * (nu_0 - freq_centers) / nu_0).in_units('km/s').value
        from ..utils import apply_spectral_lsf
        dv_per_channel = abs(v_axis[1] - v_axis[0])
        spec_cube = apply_spectral_lsf(spec_cube, dv_per_channel,
                                       cfg.SPECTRAL_RESOLUTION_R, axis=0)
        return sp_name, spec_cube, v_axis

    def compute(self, context: PipelinePlotContext) -> dict:
        n_channels = 1000   # 1000 → 300: visually identical in 10×10 panels,
                           # but the inner array shrinks 3x so each LOS slice
                           # is 3x cheaper.  Combined with 3-way parallelism
                           # below this gives ~9x speedup on the compute step.
        c_cms = (3.0e8 * m / s).in_units('cm/s').value

        # Submit all three species simultaneously.  ThreadPoolExecutor shares
        # memory (no pickle overhead) and numpy's C extensions release the GIL,
        # so the three threads genuinely overlap on separate cores.
        print('Computing spectral cubes (3 species in parallel) ...')
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(self._compute_one_species, sp['name'], n_channels, c_cms)
                for sp in SPECIES_CFG
            ]
            results = {}
            for future in futures:
                sp_name, cube, v_axis = future.result()
                results[sp_name] = {'cube': cube, 'v_axis': v_axis}

        return results

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        n_grid = 10
        sample_cube = results[SPECIES_CFG[0]['name']]['cube']
        _, ny, nz = sample_cube.shape

        y_sampling = np.linspace(0, ny - 1, n_grid, dtype=int)
        z_sampling = np.linspace(0, nz - 1, n_grid, dtype=int)

        # Pre-compute global peak per species (for annotation)
        global_peaks = {
            sp['name']: results[sp['name']]['cube'].max()
            for sp in SPECIES_CFG
        }

        fig, axes = plt.subplots(
            n_grid, n_grid,
            figsize=(20, 24),
            sharex=True,
            sharey=True,   # normalized to [0,1] so sharey is meaningful
        )
        axes_natural = np.flipud(axes)

        for i, z_idx in enumerate(z_sampling):
            for j, y_idx in enumerate(y_sampling):
                ax = axes_natural[i, j]
                for sp in SPECIES_CFG:
                    name  = sp['name']
                    v     = results[name]['v_axis']
                    spec  = results[name]['cube'][:, y_idx, z_idx]
                    peak  = spec.max()
                    # normalize to local pixel peak so all lines are visible
                    norm  = spec / peak if peak > 0 else spec
                    ax.plot(v, norm, color=sp['color'], lw=1.2,
                            label=name, drawstyle='steps-mid')
                ax.set_ylim(-0.05, 1.15)
                ax.axvline(0, color='k', ls=':', alpha=0.4, lw=0.8)
                ax.grid(True, alpha=0.25, ls='--', lw=0.5)
                ax.text(0.04, 0.88, f'({y_idx},{z_idx})',
                        transform=ax.transAxes, fontsize=5.5)

        # y-axis label on leftmost column only
        for ax_row in axes_natural:
            ax_row[0].set_yticks([0, 0.5, 1.0])
            ax_row[0].yaxis.set_tick_params(labelsize=6)

        # x-axis ticks on bottom row only
        for ax in axes_natural[-1]:
            ax.xaxis.set_tick_params(labelsize=6)

        # legend + global peak info in top-left panel
        ax_legend = axes_natural[n_grid - 1, 0]
        ax_legend.legend(fontsize=7, loc='upper right',
                         framealpha=0.7, handlelength=1.2)

        # annotate global peaks in bottom-left corner of figure
        peak_txt = '\n'.join(
            f"{sp['name']} peak: {global_peaks[sp['name']]:.2e} erg/s/Hz"
            for sp in SPECIES_CFG
        )
        fig.text(0.01, 0.01, peak_txt, fontsize=7, family='monospace',
                 va='bottom', color='dimgray')

        fig.suptitle(
            'CO / C+ / HCO+  Spectral Grid  (each profile normalized to local peak)',
            fontsize=14, y=0.92,
        )
        fig.text(0.5, 0.055, 'Velocity [km/s]', ha='center', fontsize=13)
        fig.text(0.07, 0.5, 'Normalized Intensity',
                 va='center', rotation='vertical', fontsize=13)

        plt.subplots_adjust(wspace=0.04, hspace=0.04)
        out_path = self.config.output_dir / 'Triple_Spectral_Grid.png'
        plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out_path}')

        # --- 2D Peak Intensity Maps ---
        peak_maps = {
            sp['name']: results[sp['name']]['cube'].max(axis=0)
            for sp in SPECIES_CFG
        }

        # Shared log colorscale across all three species
        all_vals = np.concatenate([peak_maps[sp['name']].ravel() for sp in SPECIES_CFG])
        pos_vals = all_vals[all_vals > 0]
        vmin = np.percentile(pos_vals, 2)   # clip lowest 2% to suppress noise floor
        vmax = pos_vals.max()
        norm_log = mcolors.LogNorm(vmin=vmin, vmax=vmax)

        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5.5))
        for ax, sp in zip(axes2, SPECIES_CFG):
            name = sp['name']
            pm = peak_maps[name]
            im = ax.imshow(
                pm.T,
                origin='lower',
                norm=norm_log,
                cmap='inferno',
                aspect='auto',
            )
            ax.set_title(name, fontsize=14, color=sp['color'], fontweight='bold')
            ax.set_xlabel('y pixel', fontsize=11)
            ax.set_ylabel('z pixel', fontsize=11)
            fig2.colorbar(im, ax=ax, label='Peak Intensity [erg/s/Hz]',
                          fraction=0.046, pad=0.04)

        fig2.suptitle(
            'CO / C+ / HCO+  Peak Intensity Maps  (shared log colorscale)',
            fontsize=13, y=1.01,
        )
        plt.tight_layout()
        out_path2 = self.config.output_dir / 'Triple_Peak_Maps.png'
        plt.savefig(str(out_path2), dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f'Saved: {out_path2}')

        # --- Total (box-integrated) spectrum: absolute + normalized, paired ---
        self._plot_total_spectra(results, SPECIES_CFG, 'Triple_Total_Spectrum.png',
                                 'CO / C+ / HCO+  Total x-direction Spectrum')
        pairs = [
            ('CO',  'C+'),
            ('CO',  'HCO+'),
            ('C+',  'HCO+'),
        ]
        sp_by_name = {sp['name']: sp for sp in SPECIES_CFG}
        for a, b in pairs:
            tag = f"{a.replace('+','plus')}_{b.replace('+','plus')}"
            self._plot_total_spectra(
                results, [sp_by_name[a], sp_by_name[b]],
                f'{tag}_Total_Spectrum.png',
                f'{a} / {b}  Total x-direction Spectrum',
            )

        # --- All pairwise combinations, absolute + normalized ---
        pairs = [
            ('CO',  'C+',   'CO_Cplus'),
            ('CO',  'HCO+', 'CO_HCOplus'),
            ('C+',  'HCO+', 'Cplus_HCOplus'),
        ]
        sp_by_name = {sp['name']: sp for sp in SPECIES_CFG}
        for a, b, tag in pairs:
            pair_cfg = [sp_by_name[a], sp_by_name[b]]
            self._plot_two_species_grid(
                results, pair_cfg, n_grid, y_sampling, z_sampling,
                normalize=False,
                out_name=f'{tag}_Grid_Absolute.png',
                title=f'{a} / {b}  Spectral Grid  (absolute intensity, per-panel y-axis)',
                ylabel='Intensity [erg/s/Hz]',
                sharey=False,
            )
            self._plot_two_species_grid(
                results, pair_cfg, n_grid, y_sampling, z_sampling,
                normalize=True,
                out_name=f'{tag}_Grid_Normalized.png',
                title=f'{a} / {b}  Spectral Grid  (each profile normalized to local peak)',
                ylabel='Normalized Intensity',
            )

    def _plot_total_spectra(self, results: dict, species_cfg: list,
                            out_name: str, title: str) -> None:
        """Two-panel figure: total (box-integrated) spectra, absolute + normalized."""
        fig, (ax_abs, ax_norm) = plt.subplots(
            1, 2, figsize=(14, 5), sharey=False
        )

        for sp in species_cfg:
            name  = sp['name']
            v     = results[name]['v_axis']
            # sum over all (y, z) pixels → total spectrum
            total = results[name]['cube'].sum(axis=(1, 2))
            peak  = total.max()

            ax_abs.plot(v, total, color=sp['color'], lw=1.8,
                        label=name, drawstyle='steps-mid')
            norm_spec = total / peak if peak > 0 else total
            ax_norm.plot(v, norm_spec, color=sp['color'], lw=1.8,
                         label=name, drawstyle='steps-mid')

        for ax in (ax_abs, ax_norm):
            ax.axvline(0, color='k', ls=':', alpha=0.4, lw=0.9)
            ax.grid(True, alpha=0.25, ls='--', lw=0.5)
            ax.set_xlabel('Velocity [km/s]', fontsize=12)
            ax.legend(fontsize=11, loc='upper right', framealpha=0.7)

        ax_abs.set_title('Absolute Total Spectrum', fontsize=13)
        ax_abs.set_ylabel('Total Luminosity [erg/s/Hz]', fontsize=11)
        ax_abs.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

        ax_norm.set_title('Normalized Total Spectrum', fontsize=13)
        ax_norm.set_ylabel('Normalized Intensity', fontsize=11)
        ax_norm.set_ylim(-0.05, 1.15)

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

        out_path = self.config.output_dir / out_name
        plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out_path}')

    def _plot_two_species_grid(
        self,
        results: dict,
        species_cfg: list,
        n_grid: int,
        y_sampling,
        z_sampling,
        normalize: bool,
        out_name: str,
        title: str,
        ylabel: str,
        sharey: bool = True,
    ) -> None:
        global_peaks = {
            sp['name']: results[sp['name']]['cube'].max()
            for sp in species_cfg
        }

        fig, axes = plt.subplots(
            n_grid, n_grid,
            figsize=(20, 24),
            sharex=True,
            sharey=sharey,
        )
        axes_natural = np.flipud(axes)

        for i, z_idx in enumerate(z_sampling):
            for j, y_idx in enumerate(y_sampling):
                ax = axes_natural[i, j]
                for sp in species_cfg:
                    name = sp['name']
                    v    = results[name]['v_axis']
                    spec = results[name]['cube'][:, y_idx, z_idx]
                    if normalize:
                        peak = spec.max()
                        data = spec / peak if peak > 0 else spec
                    else:
                        data = spec
                    ax.plot(v, data, color=sp['color'], lw=1.2,
                            label=name, drawstyle='steps-mid')
                if normalize:
                    ax.set_ylim(-0.05, 1.15)
                ax.axvline(0, color='k', ls=':', alpha=0.4, lw=0.8)
                ax.grid(True, alpha=0.25, ls='--', lw=0.5)
                ax.text(0.04, 0.88, f'({y_idx},{z_idx})',
                        transform=ax.transAxes, fontsize=5.5)

        for ax_row in axes_natural:
            ax_row[0].yaxis.set_tick_params(labelsize=6)
            if normalize:
                ax_row[0].set_yticks([0, 0.5, 1.0])

        for ax in axes_natural[-1]:
            ax.xaxis.set_tick_params(labelsize=6)

        axes_natural[n_grid - 1, 0].legend(fontsize=7, loc='upper right',
                                            framealpha=0.7, handlelength=1.2)

        peak_txt = '\n'.join(
            f"{sp['name']} peak: {global_peaks[sp['name']]:.2e} erg/s/Hz"
            for sp in species_cfg
        )
        fig.text(0.01, 0.01, peak_txt, fontsize=7, family='monospace',
                 va='bottom', color='dimgray')

        fig.suptitle(title, fontsize=14, y=0.92)
        fig.text(0.5, 0.055, 'Velocity [km/s]', ha='center', fontsize=13)
        fig.text(0.07, 0.5, ylabel, va='center', rotation='vertical', fontsize=13)

        plt.subplots_adjust(wspace=0.04, hspace=0.04)
        out_path = self.config.output_dir / out_name
        plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out_path}')
