"""BinnedPixelGridTask: simulate finite spatial resolution by SUM-binning.

Each instrument pixel covers b×b simulation cells.  The full (ny//b × nz//b)
pixel array is shown, with one spectrum panel per binned pixel.  Panels are
capped at max_panels_per_side to keep the figure readable when b is small.
"""
from __future__ import annotations

import numpy as np
from yt.units import m, s, km
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..prep import config as cfg
from ..prep.physics_fields import build_spectral_cube
from ..utils import apply_spectral_lsf, apply_spatial_bin
from ..base import AnalysisTask, PipelinePlotContext


SPECIES_FIELDS = {
    'CO':      ('CO_freq',      'CO_luminosity',      'CO_thermal_width'),
    'C+':      ('C+_freq',      'C+_luminosity',      'C+_thermal_width'),
    'H_alpha': ('H_alpha_freq', 'H_alpha_luminosity', 'H_alpha_thermal_width'),
    'HI':      ('HI_freq',      'HI_luminosity',      'HI_thermal_width'),
}


class BinnedPixelGridTask(AnalysisTask):
    """Spectrum grid after SUM-binning b×b sim cells into 1 instrument pixel."""

    def __init__(self, config, species: str,
                 bin_size: int | None = None,
                 R: float | None = None,
                 max_panels_per_side: int = 12):
        super().__init__(config)
        if species not in SPECIES_FIELDS:
            raise ValueError(f"Unknown species '{species}'; choose from {list(SPECIES_FIELDS)}")
        self.species = species
        self.bin_size = bin_size if bin_size is not None else cfg.SPATIAL_BIN
        self.R = R if R is not None else cfg.SPECTRAL_RESOLUTION_R
        self.max_panels_per_side = max_panels_per_side
        self._freq = self._lum = self._width = self._doppler = None
        self._volume = None

    def prepare(self, context: PipelinePlotContext) -> None:
        provider = context.provider
        freq_f, lum_f, width_f = SPECIES_FIELDS[self.species]
        self._freq,    _ = provider.get_slab_z(('gas', freq_f))
        self._lum,     _ = provider.get_slab_z(('gas', lum_f))
        self._width,   _ = provider.get_slab_z(('gas', width_f))
        self._doppler, _ = provider.get_slab_z(('gas', 'Bulk_Doppler_factor_x'))
        dx, _ = provider.get_slab_z(('boxlib', 'dx'))
        dy, _ = provider.get_slab_z(('boxlib', 'dy'))
        dz, _ = provider.get_slab_z(('boxlib', 'dz'))
        self._volume = dx * dy * dz

    def compute(self, context: PipelinePlotContext) -> dict:
        c = 3.0e8 * m / s
        v_range = 50.0 * km / s
        n_channels = 350

        freq_3d = self._freq.in_units('Hz')
        nu_0    = freq_3d[0, 0, 0]
        shifted = (freq_3d * self._doppler).in_units('Hz')
        lum_3d  = (self._lum * self._volume).in_units('erg/s')
        therm   = self._width.in_units('cm/s')

        bw_hz      = nu_0 * (v_range / c) * 2.0
        freq_edges = np.linspace(nu_0 - bw_hz / 2, nu_0 + bw_hz / 2, n_channels + 1)
        freq_ctr   = 0.5 * (freq_edges[:-1] + freq_edges[1:])

        nx = freq_3d.shape[0]
        print(f'  [{self.species}] building spectral cube ({nx} slices) ...')
        cube = build_spectral_cube(
            shifted.in_units('Hz').value,
            lum_3d.in_units('erg/s').value,
            therm.in_units('cm/s').value,
            freq_edges.in_units('Hz').value,
            c.in_units('cm/s').value,
        )

        v_axis_kms = (c * (nu_0 - freq_ctr) / nu_0).in_units('km/s').value
        dv = abs(v_axis_kms[1] - v_axis_kms[0])

        cube = apply_spectral_lsf(cube, dv, self.R, axis=0)
        cube = apply_spatial_bin(cube, self.bin_size)
        print(f'  [{self.species}] binned cube shape: {cube.shape}')

        return {'cube': cube, 'v_axis_kms': v_axis_kms}

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        cube = results['cube']    # (n_ch, N1b, N2b)
        v    = results['v_axis_kms']
        _, n1b, n2b = cube.shape

        mp       = self.max_panels_per_side
        ny_plot  = min(n1b, mp)
        nz_plot  = min(n2b, mp)
        y_idx    = np.linspace(0, n1b - 1, ny_plot, dtype=int)
        z_idx    = np.linspace(0, n2b - 1, nz_plot, dtype=int)
        sampled  = (ny_plot < n1b) or (nz_plot < n2b)

        fig, axes = plt.subplots(nz_plot, ny_plot, figsize=(18, 22),
                                 sharex=True, sharey=False)
        axes = np.atleast_2d(axes)
        axes_nat = np.flipud(axes)

        for i, Z in enumerate(tqdm(z_idx, desc=f'{self.species} Z panels')):
            for j, Y in enumerate(y_idx):
                ax = axes_nat[i, j]
                ax.plot(v, cube[:, Y, Z], color='royalblue', lw=1.2,
                        drawstyle='steps-mid')
                ax.ticklabel_format(style='sci', axis='y',
                                    scilimits=(0, 0), useMathText=True)
                ax.axvline(0, color='k', ls=':', alpha=0.4, lw=0.8)
                ax.grid(True, alpha=0.25, ls='--', lw=0.5)
                ax.text(0.04, 0.86, f'({Y},{Z})', transform=ax.transAxes,
                        fontsize=6)

        sample_note = f' [sampled {ny_plot}×{nz_plot}]' if sampled else ''
        fig.suptitle(
            f'{self.species}  Binned Pixel Grid'
            f'  (bin={self.bin_size}, R={self.R:.0e})'
            f'  {n1b}×{n2b} pixels{sample_note}',
            fontsize=13, y=0.93,
        )
        fig.text(0.5, 0.05, 'Velocity [km/s]', ha='center', fontsize=13)
        fig.text(0.07, 0.5, 'Pixel Spectrum [erg/s/Hz]',
                 va='center', rotation='vertical', fontsize=13)
        plt.subplots_adjust(wspace=0.08, hspace=0.08)

        safe_name = self.species.replace('+', 'plus')
        out_path  = self.config.output_dir / \
                    f'{safe_name}_BinnedPixelGrid_b{self.bin_size}.png'
        plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out_path}')
