"""SigmaSFROverlayTask: place simulation σ_CO, σ_C+ on the Lenkić+24 Fig 2 plane.

For CO and C+, compute the galaxy-averaged LOS velocity dispersion via the
same machinery as SpaxelSigmaTask (per-spaxel σ from emissivity-weighted
moments, aggregated to galaxy σ via luminosity-weighted quadratic mean).
For each LOS direction (x, y), plot the resulting log10(σ) as a horizontal
reference line on top of the Lenkić et al. 2024 Eq. 10 fit:

    log σ_mol = 0.19 × log Σ_SFR + 1.33   (solid line)
    log σ_ion = log σ_mol + log(2.5)      (dashed line, Girard+21 offset)

Σ_SFR is not yet known; the four reference lines (CO/C+ × x/y) extend
across the full Σ_SFR axis. When Σ_SFR is later provided, these can be
collapsed into point markers — see __init__ kwarg `sigma_sfr`.

Output: sigma_sfr_overlay.png
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..base import AnalysisTask, PipelinePlotContext
from ..utils import spaxel_moments_along_axis
from .integrated_spectrum import SPECIES_CFG
from .spaxel_sigma import _aggregate_lum, LOS_CFG


FIG2_SPECIES_NAMES = ('CO', 'C+')

# Lenkić et al. 2024 Eq. 10 (ODR fit over DYNAMO molecular gas):
LENKIC_SLOPE     = 0.19
LENKIC_INTERCEPT = 1.33
# Girard+21 ionized-gas offset shown as dashed reference in the same figure:
IONIZED_OFFSET   = 2.5

LOGSFR_RANGE   = (-4.0, 2.0)
LOGSIGMA_RANGE = ( 0.6, 2.6)

LOS_LINESTYLE = {'x': '-', 'y': '--'}


class SigmaSFROverlayTask(AnalysisTask):
    """Overlay simulation σ_CO, σ_C+ on the Lenkić+24 Fig 2 (left panel) plane."""

    def __init__(self, config, sigma_sfr: float | None = None):
        super().__init__(config)
        # log10(Σ_SFR) for the simulation; None → draw horizontal reference lines.
        self.sigma_sfr = sigma_sfr
        self._species_cfg = [s for s in SPECIES_CFG if s['name'] in FIG2_SPECIES_NAMES]
        self._vel: dict[str, np.ndarray] = {}
        self._lum: dict[str, np.ndarray] = {}

    def prepare(self, context: PipelinePlotContext) -> None:
        p = context.provider
        vx, _ = p.get_slab_z(('gas', 'velocity_x'))
        vy, _ = p.get_slab_z(('gas', 'velocity_y'))
        self._vel['x'] = vx.in_units('km/s').value
        self._vel['y'] = vy.in_units('km/s').value
        for sp in self._species_cfg:
            lum, _ = p.get_slab_z(('gas', sp['lum_field']))
            self._lum[sp['name']] = lum.value

    def compute(self, context: PipelinePlotContext) -> dict:
        sigmas: dict[tuple[str, str], float] = {}
        for sp in self._species_cfg:
            name = sp['name']
            w = self._lum[name]
            for los in LOS_CFG:
                los_name = los['name']
                axis     = los['axis']
                _, sigma_map, W_map = spaxel_moments_along_axis(
                    w, self._vel[los_name], axis
                )
                sigmas[(name, los_name)] = _aggregate_lum(sigma_map, W_map)

        print('\n=== Galaxy σ for Lenkić Fig 2 overlay ===')
        print(f'{"species":<6} {"LOS":>4} {"σ_galaxy [km/s]":>18} {"log10 σ":>10}')
        for (name, los_name), val in sigmas.items():
            log_val = float('nan') if not np.isfinite(val) or val <= 0 else np.log10(val)
            print(f'{name:<6} {los_name:>4} {val:>18.3f} {log_val:>10.3f}')
        print('=========================================\n')

        return {'sigmas': sigmas}

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        sigmas = results['sigmas']

        fig, ax = plt.subplots(figsize=(8, 6))

        sfr_axis = np.linspace(*LOGSFR_RANGE, 200)
        sigma_mol_line = LENKIC_SLOPE * sfr_axis + LENKIC_INTERCEPT
        sigma_ion_line = sigma_mol_line + np.log10(IONIZED_OFFSET)
        ax.plot(sfr_axis, sigma_mol_line, color='black', lw=1.6,
                label=r'Lenkić+24 Eq.10  $\log\sigma_{\rm mol}$')
        ax.plot(sfr_axis, sigma_ion_line, color='black', lw=1.2, ls='--',
                label=r'$2.5\,\sigma_{\rm mol}$ (Girard+21 offset)')

        color_by_species = {sp['name']: sp['color'] for sp in self._species_cfg}
        for (name, los_name), sigma in sigmas.items():
            if not np.isfinite(sigma) or sigma <= 0:
                continue
            log_sigma = np.log10(sigma)
            color = color_by_species[name]
            ls    = LOS_LINESTYLE[los_name]
            label = f'{name}  LOS={los_name}   σ={sigma:.2f} km/s'
            if self.sigma_sfr is None:
                ax.axhline(log_sigma, color=color, ls=ls, lw=1.4,
                           alpha=0.85, label=label)
            else:
                ax.plot(self.sigma_sfr, log_sigma,
                        marker='o' if los_name == 'x' else 's',
                        color=color, ms=10, mec='black', mew=1.0,
                        label=label, zorder=5)

        ax.set_xlim(*LOGSFR_RANGE)
        ax.set_ylim(*LOGSIGMA_RANGE)
        ax.set_xlabel(r'$\log\,\Sigma_{\rm SFR}\;[\mathrm{M_{\odot}\,yr^{-1}\,kpc^{-2}}]$',
                      fontsize=12)
        ax.set_ylabel(r'$\log\,\sigma\;[\mathrm{km\,s^{-1}}]$', fontsize=12)
        title_tag = (r'simulation $\sigma$ overlay'
                     if self.sigma_sfr is None
                     else fr'simulation point at $\log\Sigma_{{\rm SFR}}={self.sigma_sfr:.2f}$')
        ax.set_title(f'Lenkić+24 Fig 2 (left panel) — {title_tag}', fontsize=12)
        ax.grid(True, alpha=0.3, ls='--', lw=0.5)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.85)

        plt.tight_layout()
        out = self.config.output_dir / 'sigma_sfr_overlay.png'
        plt.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
