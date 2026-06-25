"""Species 1D emission-line spectra, split into Build + Plot tasks.

``Build_SpeciesSpectrum`` (compute + store): the single source of truth for the
emission-line spectra.  For each species it builds one 'total' (all-cell) 1D
spectrum (intrinsic R=∞) via `build_spectral_cube`, then an LSF-convolved
variant.  It also reads σ_gas from ``Build_VelocityPhase``'s stored result (for
the IntegratedSpectrum vertical-line markers) and stores everything.

``Plot_SpeciesSpectrum`` (plot only): renders
  IntegratedSpectrum_{CO,Cplus,H_alpha,HI}.png   (intrinsic + LSF overlay)
  IntegratedSpectrum_overlay.png                 (4 species, peak-normalised)
from the stored result.

The PhaseSpectrumOverlay plots live in their own Plot task because they also
need the velocity PDFs from ``Build_VelocityPhase``.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from ..prep import config as _cfg
from ..base import BuildTask, PlotTask, PipelinePlotContext
from ..utils import PHASE_ORDER, PHASE_LABEL_LINE
from .integrated_spectrum import SPECIES_CFG, V_RANGE_KMS
from .velocity_phase import PHASE_COLOR


TOTAL_COLOR = 'black'


# ── shared helpers (moment + Gaussian fit) ──────────────────────────────
def _gaussian(v, A, v0, sigma):
    return A * np.exp(-0.5 * ((v - v0) / sigma) ** 2)


def _fit_gaussian(v: np.ndarray, spec: np.ndarray) -> tuple[float, float, np.ndarray]:
    peak = spec.max()
    if peak <= 0:
        return np.nan, np.nan, np.zeros_like(spec)
    spec_norm = spec / peak
    v0_guess  = v[np.argmax(spec_norm)]
    try:
        popt, _ = curve_fit(_gaussian, v, spec_norm,
                            p0=[1.0, v0_guess, 5.0],
                            bounds=([0, -50, 0.1], [2.0, 50, 100]),
                            maxfev=10000)
        return float(popt[1]), abs(float(popt[2])), _gaussian(v, *popt) * peak
    except Exception:
        return np.nan, np.nan, np.zeros_like(spec)


def _moment_sigma(v: np.ndarray, spec: np.ndarray) -> tuple[float, float]:
    total = spec.sum()
    if total <= 0:
        return np.nan, np.nan
    w      = spec / total
    v_mean = float(np.sum(v * w))
    sigma  = float(np.sqrt(np.sum((v - v_mean) ** 2 * w)))
    return v_mean, sigma


# ─── Build ──────────────────────────────────────────────────────────────────
class Build_SpeciesSpectrum(BuildTask):
    """Build the per-species 'total' 1D spectra (+ LSF variant); store them."""

    def __init__(self, config, R: float | None = None):
        super().__init__(config)
        self.R = R if R is not None else _cfg.SPECTRAL_RESOLUTION_R

    def compute(self, context: PipelinePlotContext) -> dict:
        provider = context.provider

        # 1) Evict any in-RAM fields carried over from prior tasks before we
        # start allocating species cubes — otherwise OOM kicks in on 16 GB Mac
        # at down=1.  The disk-backed field intermediates remain.
        import gc
        provider._cached_grid = None
        gc.collect()

        # 2) σ_gas needed for IntegratedSpectrum vertical-line markers — read
        # from Build_VelocityPhase's stored result (must run before this task).
        from ..intermediate_io import load_one_build
        vp = load_one_build(self.config.output_dir, 'Build_VelocityPhase', self.config)
        sigma_x_gas = float(vp['total_x']['sigma'])
        sigma_y_gas = float(vp['total_y']['sigma'])
        sigma_z_gas = float(vp['total_z']['sigma'])
        print(f'  σ_gas (loaded from Build_VelocityPhase result): '
              f'x={sigma_x_gas:.2f} km/s, y={sigma_y_gas:.2f} km/s, '
              f'z={sigma_z_gas:.2f} km/s')

        # Per-species build: 3 LOS (x/y/z) × 1 'total' = 3 unique 1D spectra each.
        # (2026-06-19 refactor — phase decomposition removed; downstream
        # consumers only need 'total'.  SpectrumStore.get_spectrum(phase=None)
        # hits the 'total' code path — one cube covering all cells.)
        # (2026-06-25 — LOS x/y/z all built so PhaseSpectrumOverlay can plot each
        # projection; the per-LOS emission curve must match its LOS velocity axis.)
        from ..services import SpectrumStore
        # 2 workers: ~13 GB peak per species — under the 16 GB Mac limit.  Building
        # 3 LOS (not 1) triples the per-species wall time; peak RAM is unchanged
        # (still ≤2 cubes in flight; the store reuses lum/width/volume across LOS).
        N_WORKERS = 2
        spectra: dict[str, dict[str, dict]] = {
            sp['name']: {'x': {}, 'y': {}, 'z': {}} for sp in SPECIES_CFG
        }
        for sp in SPECIES_CFG:
            name  = sp['name']
            store = SpectrumStore(provider)
            jobs  = [(name, los, None) for los in ('x', 'y', 'z')]  # phase=None → 'total'
            print(f'SpeciesSpectrum [{name}]: '
                  f'{len(jobs)} (los × total) build, {N_WORKERS} workers ...')

            def _build_one(args):
                sp_name, los, ph = args
                v_axis, dsigma = store.get_spectrum(sp_name, los, phase=ph,
                                                    R=float('inf'))
                return sp_name, los, v_axis, dsigma

            with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
                for sp_name, los, v_axis, d in pool.map(_build_one, jobs):
                    spectra[sp_name][los]['total'] = {
                        'v_axis':    v_axis,
                        'dsigma_dv': d,
                    }
            del store
            provider._cached_grid = None
            gc.collect()

        # LSF-convolved variant (R = self.R) applied post-hoc to the 'total'
        # spectrum only.  Stored under key 'dsigma_dv_obs' for IntegratedSpectrum.
        if np.isfinite(self.R) and self.R > 0:
            from ..services.spectrum_service import apply_spectral_lsf
            for sp in SPECIES_CFG:
                name = sp['name']
                for los in ('x', 'y', 'z'):
                    block = spectra[name][los]['total']
                    v_axis = block['v_axis']
                    dv = abs(v_axis[1] - v_axis[0])
                    block['dsigma_dv_obs'] = apply_spectral_lsf(
                        block['dsigma_dv'], dv, self.R, axis=0)

        return {
            'spectra':     spectra,
            'sigma_v_gas': {'x': sigma_x_gas, 'y': sigma_y_gas, 'z': sigma_z_gas},
            'R':           self.R,
        }


# ─── Plot ───────────────────────────────────────────────────────────────────
class Plot_SpeciesSpectrum(PlotTask):
    """Render the IntegratedSpectrum figures from Build_SpeciesSpectrum."""

    def __init__(self, config):
        super().__init__(config)

    def _gather_inputs(self, context: PipelinePlotContext) -> dict:
        return self._load_one(context, 'Build_SpeciesSpectrum')

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        self._plot_integrated_individual(results)
        self._plot_integrated_overlay(results)

    # IntegratedSpectrum_{species}.png × 4 — intrinsic faint + LSF prominent
    def _plot_integrated_individual(self, results: dict) -> None:
        spectra = results['spectra']
        R       = results['R']
        for sp in SPECIES_CFG:
            name = sp['name']
            fig, axes = plt.subplots(1, 1, figsize=(8, 5), sharey=False)
            axes = [axes]
            for ax, los, title in [
                (axes[0], 'y', 'Line-of-sight: y  (x-z plane)'),
            ]:
                v        = spectra[name][los]['total']['v_axis']
                spec     = spectra[name][los]['total']['dsigma_dv']
                spec_obs = spectra[name][los]['total'].get('dsigma_dv_obs', spec)
                sigma_gas = results['sigma_v_gas'][los]

                ax.plot(v, spec, color=sp['color'], lw=0.8, alpha=0.35,
                        drawstyle='steps-mid', label='intrinsic')
                ax.plot(v, spec_obs, color=sp['color'], lw=1.5,
                        drawstyle='steps-mid', label=f'observed (R={R:.0e})')

                _, sigma_gauss, fit_curve = _fit_gaussian(v, spec_obs)
                if np.isfinite(sigma_gauss):
                    ax.plot(v, fit_curve, color='black', lw=1.2, ls='--',
                            label=f'Gaussian  σ={sigma_gauss:.1f} km/s')
                _, sigma_mom = _moment_sigma(v, spec_obs)
                if np.isfinite(sigma_mom):
                    ax.axvspan(-sigma_mom, sigma_mom, alpha=0.08, color=sp['color'],
                               label=f'Moment   σ={sigma_mom:.1f} km/s')
                if np.isfinite(sigma_gas):
                    ax.axvline( sigma_gas, color='darkorange', lw=1.4, ls='-.',
                               label=f'σ_gas={sigma_gas:.1f} km/s')
                    ax.axvline(-sigma_gas, color='darkorange', lw=1.4, ls='-.')

                ax.axvline(0, color='gray', ls=':', lw=0.8, alpha=0.6)
                ax.set_xlabel('Velocity [km/s]', fontsize=12)
                ax.set_ylabel(r'$\mathrm{d}\Sigma/\mathrm{d}v$ [erg s$^{-1}$ Hz$^{-1}$ cm$^{-2}$]',
                              fontsize=11)
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
                ax.set_title(title, fontsize=12)
                ax.legend(fontsize=9, framealpha=0.85)
                ax.grid(True, alpha=0.25, ls='--', lw=0.5)

            fig.suptitle(f'{name}  Spatially Integrated Spectrum  (R={R:.0e})',
                         fontsize=13)
            tag = name.replace('+', 'plus')
            out = self.config.output_dir / f'IntegratedSpectrum_{tag}.png'
            plt.tight_layout()
            plt.savefig(str(out), dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved: {out}')

    # IntegratedSpectrum_overlay.png — 4 species, peak-normalised
    def _plot_integrated_overlay(self, results: dict) -> None:
        spectra = results['spectra']
        R       = results['R']
        fig, axes = plt.subplots(1, 1, figsize=(8, 5))
        axes = [axes]
        for ax, los, title in [
            (axes[0], 'y', 'Line-of-sight: y  (x-z plane)'),
        ]:
            for sp in SPECIES_CFG:
                name = sp['name']
                v        = spectra[name][los]['total']['v_axis']
                spec_obs = spectra[name][los]['total'].get(
                    'dsigma_dv_obs', spectra[name][los]['total']['dsigma_dv'])
                peak = spec_obs.max()
                norm = spec_obs / peak if peak > 0 else spec_obs
                ax.plot(v, norm, color=sp['color'], lw=1.5,
                        drawstyle='steps-mid', label=name)
            ax.axvline(0, color='gray', ls=':', lw=0.8, alpha=0.6)
            ax.set_xlabel('Velocity [km/s]', fontsize=12)
            ax.set_ylabel('Normalized Intensity', fontsize=12)
            ax.set_ylim(-0.05, 1.15)
            ax.legend(fontsize=10, framealpha=0.8)
            ax.set_title(title, fontsize=12)
            ax.grid(True, alpha=0.25, ls='--', lw=0.5)
        labels = ' / '.join(s['name'] for s in SPECIES_CFG)
        fig.suptitle(f'{labels}  Spatially Integrated Spectra (normalized, observed R={R:.0e})',
                     fontsize=13)
        out = self.config.output_dir / 'IntegratedSpectrum_overlay.png'
        plt.tight_layout()
        plt.savefig(str(out), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')


# ============================================================================
# DEPRECATED 2026-06-19 (wrapped 2026-06-23) — kept in-tree for reference.
# _plot_phase_resolved: compute() no longer builds per-phase species spectra
# (only 'total'), so this per-phase plot would KeyError.  Uses TOTAL_COLOR,
# PHASE_COLOR, PHASE_ORDER, PHASE_LABEL_LINE, V_RANGE_KMS (all still imported).
# ============================================================================
r'''
    def _plot_phase_resolved(self, results: dict) -> None:
        spectra = results['spectra']
        n_species = len(SPECIES_CFG)
        fig, axes = plt.subplots(
            n_species, 1, figsize=(7, 3.0 * n_species), sharex='all', squeeze=False,
        )
        for r, sp in enumerate(SPECIES_CFG):
            name = sp['name']
            y_max = 0.0
            for los_name in ('y',):
                tot = spectra[name][los_name]['total']['dsigma_dv']
                if tot.size and np.isfinite(tot).any():
                    y_max = max(y_max, float(np.nanmax(tot)))
            for col, los_name in enumerate(('y',)):
                ax = axes[r, col]
                v_axis     = spectra[name][los_name]['total']['v_axis']
                total_spec = spectra[name][los_name]['total']['dsigma_dv']
                ax.plot(v_axis, total_spec, color=TOTAL_COLOR, lw=2.0, label='total', zorder=10)
                for ph in PHASE_ORDER:
                    spec_ph = spectra[name][los_name][ph]['dsigma_dv']
                    ax.plot(v_axis, spec_ph, color=PHASE_COLOR[ph], lw=1.3, label=ph)
                ax.axvline(0, color='gray', ls=':', lw=0.6, alpha=0.6)
                ax.set_xlim(-V_RANGE_KMS, V_RANGE_KMS)
                if y_max > 0:
                    ax.set_ylim(-0.03 * y_max, 1.10 * y_max)
                ax.set_title(f'{name}  —  LOS={los_name}', fontsize=11)
                ax.grid(True, alpha=0.3, ls='--', lw=0.5)
                if r == 0 and col == 1:
                    ax.legend(fontsize=8, loc='upper right', framealpha=0.85, ncol=2)
                if r == n_species - 1:
                    ax.set_xlabel('Velocity [km/s]', fontsize=11)
                if col == 0:
                    ax.set_ylabel(r'd$\Sigma$/dv  [erg s$^{-1}$ Hz$^{-1}$ cm$^{-2}$]', fontsize=9)
        fig.suptitle('Phase-resolved emission spectra\n' + PHASE_LABEL_LINE, fontsize=12)
        plt.tight_layout()
        out = self.config.output_dir / 'PhaseResolvedSpectrum.png'
        plt.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
'''
