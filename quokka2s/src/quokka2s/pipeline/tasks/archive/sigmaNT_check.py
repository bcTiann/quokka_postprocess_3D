from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ..base import AnalysisTask, PipelinePlotContext
from ..utils import make_axis_labels, weighted_percentile

# Per-species plot styling
SPECIES_STYLE = {
    'CO':   {'color': 'royalblue',   'label': 'CO'},
    'C+':   {'color': 'forestgreen', 'label': 'C+'},
    'HCO+': {'color': 'crimson',     'label': 'HCO+'},
}


class SigmaNTCheckTask(AnalysisTask):
    """Sanity-check the fixed sigmaNT = 2 km/s assumption in DESPOTIC.

    For each cell we estimate the local effective velocity dispersion from
    the *resolved* velocity gradient along the LOS (x) direction:

        sigma_eff(i,j,k) = |v_x(i+1,j,k) - v_x(i,j,k)|

    This is the velocity difference across one cell — the scale over which
    photons can be reabsorbed before being Doppler-shifted out of the line
    core (Sobolev / LVG picture).  Comparing this distribution to the
    hard-coded 2 km/s tells us whether that assumption is representative.

    Outputs
    -------
    sigmaNT_histogram.png            — CO vol-weighted vs CO-lum-weighted (original)
    sigmaNT_histogram_per_species.png — 3-panel: one panel per species
    sigmaNT_map.png                  — 2D projection of median sigma_eff along LOS
    """

    def __init__(self, config, axis: str | None = None, figure_units: str | None = None):
        super().__init__(config)
        self.axis = axis or 'x'
        self.figure_units = figure_units or config.figure_units
        self.xlabel, self.ylabel = make_axis_labels(self.axis, self.figure_units)

    def prepare(self, context: PipelinePlotContext) -> None:
        provider = context.provider
        self._vx,       _ = provider.get_slab_z(('gas', 'velocity_x'))
        self._co_lum,   _ = provider.get_slab_z(('gas', 'CO_luminosity'))
        self._cplus_lum,_ = provider.get_slab_z(('gas', 'C+_luminosity'))
        self._hco_lum,  _ = provider.get_slab_z(('gas', 'HCO+_luminosity'))
        self._dx,       _ = provider.get_slab_z(('boxlib', 'dx'))
        self._dy,       _ = provider.get_slab_z(('boxlib', 'dy'))
        self._dz,       _ = provider.get_slab_z(('boxlib', 'dz'))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _species_weighted_stats(self, delta_vx, lum_cell):
        """Compute luminosity-weighted stats for one species.

        Parameters
        ----------
        delta_vx : ndarray, shape (nx-1, ny, nz)
        lum_cell : ndarray, shape (nx, ny, nz)  — luminosity per cell [erg/s]

        Returns
        -------
        dv_w, w_w, stats  — filtered arrays + stats dict
        """
        weight = 0.5 * (lum_cell[:-1, :, :] + lum_cell[1:, :, :])
        dv_flat = delta_vx.ravel()
        w_flat  = weight.ravel()
        mask    = (dv_flat > 0) & (w_flat > 0)
        dv_w    = dv_flat[mask]
        w_w     = w_flat[mask]
        w_w     = w_w / w_w.sum()

        stats = {
            'median':      weighted_percentile(dv_w, w_w, 50),
            'p75':         weighted_percentile(dv_w, w_w, 75),
            'p90':         weighted_percentile(dv_w, w_w, 90),
            'frac_above':  float(np.sum(w_w[dv_w > 2.0])) * 100,
        }
        return dv_w, w_w, stats

    # ------------------------------------------------------------------
    # pipeline interface
    # ------------------------------------------------------------------
    def compute(self, context: PipelinePlotContext) -> dict:
        vx_kms = self._vx.in_units('km/s').value          # (nx, ny, nz)
        vol    = (self._dx * self._dy * self._dz).in_units('cm**3').value

        co_lum_cell   = self._co_lum.in_units('erg/s/cm**3').value   * vol
        cplus_lum_cell= self._cplus_lum.in_units('erg/s/cm**3').value * vol
        hco_lum_cell  = self._hco_lum.in_units('erg/s/cm**3').value  * vol

        # velocity difference between adjacent cells along x  (nx-1, ny, nz)
        delta_vx = np.abs(np.diff(vx_kms, axis=0))

        # volume-weighted baseline
        flat_v = delta_vx.ravel()
        flat_v = flat_v[flat_v > 0]
        stats_vol = {
            'median': float(np.median(flat_v)),
            'p75':    float(np.percentile(flat_v, 75)),
            'p90':    float(np.percentile(flat_v, 90)),
            'p99':    float(np.percentile(flat_v, 99)),
        }

        # per-species luminosity-weighted stats
        dv_co,   w_co,   stats_co   = self._species_weighted_stats(delta_vx, co_lum_cell)
        dv_cplus,w_cplus,stats_cplus= self._species_weighted_stats(delta_vx, cplus_lum_cell)
        dv_hco,  w_hco,  stats_hco  = self._species_weighted_stats(delta_vx, hco_lum_cell)

        # 2D map: median along LOS
        sigma_map = np.median(delta_vx, axis=0)

        print("\n=== sigma_NT sanity check ===")
        print(f"  [Volume-weighted]  median = {stats_vol['median']:.2f} km/s  "
              f"p90 = {stats_vol['p90']:.2f} km/s  "
              f">2 km/s: {np.mean(flat_v > 2.0)*100:.1f}%")
        for name, st in [('CO', stats_co), ('C+', stats_cplus), ('HCO+', stats_hco)]:
            print(f"  [{name}-lum-weighted]  median = {st['median']:.2f} km/s  "
                  f"p90 = {st['p90']:.2f} km/s  "
                  f">2 km/s: {st['frac_above']:.1f}%")
        print(f"  DESPOTIC assumed   = 2.00 km/s")
        print("=============================\n")

        return {
            'flat_v':    flat_v,
            'sigma_map': sigma_map,
            'stats_vol': stats_vol,
            # per-species
            'species': {
                'CO':   {'dv': dv_co,   'w': w_co,   'stats': stats_co},
                'C+':   {'dv': dv_cplus,'w': w_cplus,'stats': stats_cplus},
                'HCO+': {'dv': dv_hco,  'w': w_hco,  'stats': stats_hco},
            },
        }

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        self._plot_histogram(results)           # original CO-only comparison
        self._plot_per_species(results)         # new 3-panel figure
        self._plot_map(results)

    # ------------------------------------------------------------------
    # plot helpers
    # ------------------------------------------------------------------
    def _plot_histogram(self, results: dict) -> None:
        """Original single-panel: volume-weighted vs CO-luminosity-weighted."""
        flat_v   = results['flat_v']
        co       = results['species']['CO']
        dv_w     = co['dv']
        w_w      = co['w']
        stats_v  = results['stats_vol']
        stats_co = co['stats']

        lo   = max(min(flat_v.min(), dv_w.min()), 0.01)
        hi   = max(stats_v['p99'] * 3, 10.0)
        bins = np.logspace(np.log10(lo), np.log10(hi), 80)

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.hist(flat_v, bins=bins, color='steelblue', alpha=0.55,
                density=True, label='Volume-weighted (all cells)')
        ax.hist(dv_w, bins=bins, weights=w_w / np.diff(np.log10(bins)).mean(),
                color='darkorange', alpha=0.65, label='CO-luminosity-weighted')

        ax.axvline(2.0, color='crimson', lw=2.0, ls='--',
                   label='DESPOTIC assumed σ_NT = 2 km/s')
        ax.axvline(stats_v['median'],  color='steelblue',  lw=1.5, ls='-',
                   label=f"Vol median = {stats_v['median']:.2f} km/s")
        ax.axvline(stats_co['median'], color='darkorange', lw=1.5, ls='-',
                   label=f"CO median  = {stats_co['median']:.2f} km/s")

        ax.set_xscale('log')
        ax.set_xlabel('|Δv_x| per cell  [km/s]', fontsize=12)
        ax.set_ylabel('Probability density', fontsize=12)
        ax.set_title('σ_NT sanity check: volume-weighted vs CO-luminosity-weighted', fontsize=11)
        ax.legend(fontsize=9, loc='upper left')

        txt = (f"Volume-weighted\n"
               f"  median = {stats_v['median']:.2f} km/s\n"
               f"  p90    = {stats_v['p90']:.2f} km/s\n"
               f"  >2 km/s: {np.mean(flat_v > 2.0)*100:.1f}%\n\n"
               f"CO-lum-weighted\n"
               f"  median = {stats_co['median']:.2f} km/s\n"
               f"  p90    = {stats_co['p90']:.2f} km/s\n"
               f"  >2 km/s: {stats_co['frac_above']:.1f}%")
        ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                va='top', ha='right', fontsize=8.5,
                bbox=dict(boxstyle='round', fc='white', alpha=0.85))

        plt.tight_layout()
        out = self.config.output_dir / 'sigmaNT_histogram.png'
        plt.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')

    def _plot_per_species(self, results: dict) -> None:
        """3-panel figure: one panel per species, each showing volume-weighted
        vs species-luminosity-weighted distribution."""
        flat_v  = results['flat_v']
        stats_v = results['stats_vol']
        species = results['species']

        lo   = max(flat_v.min(), 0.01)
        hi   = max(stats_v['p99'] * 3, 10.0)
        bins = np.logspace(np.log10(lo), np.log10(hi), 80)
        bin_width_log = np.diff(np.log10(bins)).mean()

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

        for ax, sp_name in zip(axes, ['CO', 'C+', 'HCO+']):
            style  = SPECIES_STYLE[sp_name]
            sp     = species[sp_name]
            dv_w   = sp['dv']
            w_w    = sp['w']
            stats  = sp['stats']

            # volume-weighted background
            ax.hist(flat_v, bins=bins, color='steelblue', alpha=0.45,
                    density=True, label='Volume-weighted')

            # species-luminosity-weighted
            ax.hist(dv_w, bins=bins,
                    weights=w_w / bin_width_log,
                    color=style['color'], alpha=0.70,
                    label=f'{sp_name}-lum-weighted')

            # reference lines
            ax.axvline(2.0, color='black', lw=1.8, ls='--',
                       label='DESPOTIC assumed σ_NT = 2 km/s')
            ax.axvline(stats_v['median'], color='steelblue', lw=1.3, ls=':',
                       label=f"Vol median = {stats_v['median']:.2f} km/s")
            ax.axvline(stats['median'], color=style['color'], lw=1.8, ls='-',
                       label=f"{sp_name} median = {stats['median']:.2f} km/s")

            ax.set_xscale('log')
            ax.set_xlabel('|Δv_x| per cell  [km/s]', fontsize=11)
            ax.set_ylabel('Probability density', fontsize=11)
            ax.set_title(f'{sp_name}-luminosity-weighted', fontsize=12)
            ax.legend(fontsize=8, loc='upper left')

            txt = (f"── Volume-weighted (all gas) ──\n"
                   f"  median = {stats_v['median']:.2f} km/s\n\n"
                   f"── {sp_name}-lum-weighted ──\n"
                   f"  median = {stats['median']:.2f} km/s\n"
                   f"  p90    = {stats['p90']:.2f} km/s\n"
                   f"  {stats['frac_above']:.1f}% of {sp_name} emission\n"
                   f"  from |Δv| > 2 km/s regions\n\n"
                   f"  DESPOTIC assumed σ_NT = 2.0 km/s")
            ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                    va='top', ha='right', fontsize=8,
                    bbox=dict(boxstyle='round', fc='white', alpha=0.85))

        fig.suptitle('σ_NT sanity check: volume-weighted vs per-species luminosity-weighted',
                     fontsize=13)
        plt.tight_layout()
        out = self.config.output_dir / 'sigmaNT_histogram_per_species.png'
        plt.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')

    def _plot_map(self, results: dict) -> None:
        sigma_map = results['sigma_map']
        stats     = results['stats_vol']

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(
            sigma_map.T,
            origin='lower',
            norm=mcolors.LogNorm(vmin=0.1, vmax=max(stats['p99'], 10.0)),
            cmap='plasma',
            aspect='auto',
        )
        cbar = fig.colorbar(im, ax=ax, label='Median |Δv_x| along LOS [km/s]')
        cbar.ax.axhline(2.0, color='white', lw=1.5, ls='--')
        ax.set_xlabel(self.xlabel, fontsize=11)
        ax.set_ylabel(self.ylabel, fontsize=11)
        ax.set_title('Spatial map of local velocity gradient\n'
                     '(white dashed = 2 km/s DESPOTIC assumption)', fontsize=11)

        plt.tight_layout()
        out = self.config.output_dir / 'sigmaNT_map.png'
        plt.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
