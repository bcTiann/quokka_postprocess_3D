"""PhaseResolvedSpectrumTask: emission spectra decomposed by ISM phase.

For each species in SPECIES_CFG (CO, C+, Hα, HI), build a task-local
``SpectrumStore`` and obtain a 1D spectrum from cells in one
ISM phase at a time:

    ε_phase(ν) = Σ_{cells in phase} cell_spectrum(ν)

The "total" (all-cell) spectrum is **not** computed independently — phase
masks are disjoint + cover all cells and ``build_spectral_cube`` is linear
in luminosity, so ``total = sum(phase_spectra)`` algebraically.  This
saves 8 ``build_spectral_cube`` calls (one per species×LOS).

Plots: one figure per species, 2 columns (LOS=x, LOS=y), each panel
overlays the 5 phase spectra + the derived total (no normalisation;
absolute units in erg/s/Hz/cm²).

Output: PhaseResolvedSpectrum_{species}.png
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

from ..base import AnalysisTask, PipelinePlotContext
from ..utils import PHASE_ORDER, PHASE_LABEL_LINE, classify_temperature_phase
from .integrated_spectrum import SPECIES_CFG, V_RANGE_KMS
from .phase_sigmaV import PHASE_COLOR


TOTAL_COLOR = 'black'


class PhaseResolvedSpectrumTask(AnalysisTask):
    """Per-phase emission spectra for each species, both LOS, absolute units."""

    def __init__(self, config):
        super().__init__(config)

    def compute(self, context: PipelinePlotContext) -> dict:
        # Sanity: print phase populations.  T cube is local; released on return.
        T_unyt, _ = context.provider.get_slab_z(('gas', 'temperature_despotic'))
        T_K = T_unyt.in_units('K').value
        del T_unyt
        masks = classify_temperature_phase(T_K)
        print('\n=== Phase cell counts (for PhaseResolvedSpectrum) ===')
        total = T_K.size
        for ph in PHASE_ORDER:
            n = int(masks[ph].sum())
            print(f'  {ph:<6}  {n:>10}  ({100*n/total:>5.2f} %)')
        print('=====================================================\n')
        del T_K, masks   # SpectrumStore loads its own T below; free now.

        # Fresh SpectrumStore per species — keeps only one species' lum+width
        # in memory at a time.  Each species still does its own 2 LOS × 5 phase
        # = 10 builds in parallel, but never holds 4 species' primitives at once.
        from ..services import SpectrumStore
        spectra: dict[str, dict[str, dict]] = {
            sp['name']: {'x': {}, 'y': {}} for sp in SPECIES_CFG
        }
        # With chunked build_spectral_cube each build's transient is ~0.7 GB
        # at down=1, so 6 workers ≈ 4 GB transient on top of ~3 GB persistent.
        n_workers = 6
        import gc
        for sp in SPECIES_CFG:
            name = sp['name']
            store = SpectrumStore(context.provider)
            jobs = [(name, los, ph) for los in ('x', 'y') for ph in PHASE_ORDER]
            print(f'PhaseResolvedSpectrum [{name}]: dispatching {len(jobs)} builds, {n_workers} workers ...')

            def _build_one(args):
                sp_name, los, ph = args
                v_axis, dsigma = store.get_spectrum(sp_name, los, phase=ph, R=float('inf'))
                return sp_name, los, ph, v_axis, dsigma

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for sp_name, los, ph, v_axis, d in pool.map(_build_one, jobs):
                    spectra[sp_name][los][ph] = (v_axis, d)
            del store
            context.provider._cached_grid = None
            gc.collect()

        # 'total' is the algebraic sum of disjoint phase contributions — no
        # extra build_spectral_cube call needed.
        for sp in SPECIES_CFG:
            name = sp['name']
            for los in ('x', 'y'):
                phase_specs = [spectra[name][los][ph][1] for ph in PHASE_ORDER]
                v_axis      =  spectra[name][los][PHASE_ORDER[0]][0]
                total_spec  = np.sum(phase_specs, axis=0)
                spectra[name][los]['total'] = (v_axis, total_spec)

        return spectra

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        for sp in SPECIES_CFG:
            self._plot_one_species(sp, results[sp['name']])

    def _plot_one_species(self, sp: dict, los_data: dict) -> None:
        name = sp['name']
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        # Common y-limit set by the total spectrum's max across both LOS.
        y_max = 0.0
        for los_name in ('x', 'y'):
            _, tot = los_data[los_name]['total']
            if tot.size and np.isfinite(tot).any():
                y_max = max(y_max, float(np.nanmax(tot)))

        for col, los_name in enumerate(('x', 'y')):
            ax = axes[col]
            v_axis, total_spec = los_data[los_name]['total']
            ax.plot(v_axis, total_spec, color=TOTAL_COLOR, lw=2.0,
                    label='total', zorder=10)
            for ph in PHASE_ORDER:
                _, spec_ph = los_data[los_name][ph]
                ax.plot(v_axis, spec_ph, color=PHASE_COLOR[ph],
                        lw=1.3, label=ph)
            ax.axvline(0, color='gray', ls=':', lw=0.6, alpha=0.6)
            ax.set_xlabel('Velocity [km/s]', fontsize=11)
            ax.set_xlim(-V_RANGE_KMS, V_RANGE_KMS)
            if y_max > 0:
                ax.set_ylim(-0.03 * y_max, 1.10 * y_max)
            ax.set_title(f'{name} — LOS={los_name}', fontsize=12)
            ax.grid(True, alpha=0.3, ls='--', lw=0.5)
            ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
        axes[0].set_ylabel(r'd$\Sigma$/dv  [erg s$^{-1}$ Hz$^{-1}$ cm$^{-2}$]',
                           fontsize=11)
        fig.suptitle(f'Phase-resolved emission spectrum: {name}\n'
                     + PHASE_LABEL_LINE, fontsize=11)
        plt.tight_layout()

        safe_name = name.replace('+', 'plus').replace('_', '')
        out = self.config.output_dir / f'PhaseResolvedSpectrum_{safe_name}.png'
        plt.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
