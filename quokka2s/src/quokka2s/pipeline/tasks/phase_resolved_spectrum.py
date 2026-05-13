"""PhaseResolvedSpectrumTask: emission spectra decomposed by ISM phase.

For each species in SPECIES_CFG (CO, C+, Hα, HI), build the spatially
integrated spectrum from cells belonging to one ISM phase at a time:

    ε_phase(ν) = Σ_{cells in phase} cell_spectrum(ν)

This is mathematically equivalent to running IntegratedSpectrumTask with
the luminosity zeroed-out everywhere except the cells of one phase. Since
the phase masks are disjoint and complete, summing the 5 phase spectra
recovers the full integrated spectrum exactly.

Plots: one figure per species, 2 columns (LOS=x, LOS=y), each panel
overlays the 5 phase spectra + the total (no normalisation; absolute
units in erg/s/Hz/cm²).

Output: PhaseResolvedSpectrum_{species}.png
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt

from ..base import AnalysisTask, PipelinePlotContext
from ..prep.physics_fields import build_spectral_cube
from ..utils import PHASE_ORDER, PHASE_LABEL_LINE, classify_temperature_phase
from .integrated_spectrum import SPECIES_CFG, N_CHANNELS, V_RANGE_KMS
from .phase_sigmaV import PHASE_COLOR


TOTAL_COLOR = 'black'

# Speed of light in cgs (cm/s). Hardcoded scalar — avoids 48× yt unit
# conversions on a 16M-cell cube.
_C_CGS = 2.99792458e10


class PhaseResolvedSpectrumTask(AnalysisTask):
    """Per-phase emission spectra for each species, both LOS, absolute units."""

    def __init__(self, config):
        super().__init__(config)
        self._T = None
        self._volume_3d = None
        self._doppler_x = None
        self._doppler_y = None
        self._cell_area: dict[str, float] = {}
        self._sp_data: dict[str, dict] = {}

    def prepare(self, context: PipelinePlotContext) -> None:
        p = context.provider
        self._T,         _ = p.get_slab_z(('gas', 'temperature_despotic'))
        self._doppler_x, _ = p.get_slab_z(('gas', 'Bulk_Doppler_factor_x'))
        self._doppler_y, _ = p.get_slab_z(('gas', 'Bulk_Doppler_factor_y'))

        dx, _ = p.get_slab_z(('boxlib', 'dx'))
        dy, _ = p.get_slab_z(('boxlib', 'dy'))
        dz, _ = p.get_slab_z(('boxlib', 'dz'))
        self._volume_3d = dx * dy * dz
        self._cell_area['yz'] = (dy * dz)[0, 0, 0].in_units('cm**2').value
        self._cell_area['xz'] = (dx * dz)[0, 0, 0].in_units('cm**2').value

        for sp in SPECIES_CFG:
            name = sp['name']
            freq,  _ = p.get_slab_z(('gas', sp['freq_field']))
            lum,   _ = p.get_slab_z(('gas', sp['lum_field']))
            width, _ = p.get_slab_z(('gas', sp['width_field']))
            self._sp_data[name] = {'freq': freq, 'lum': lum, 'width': width}

    def _prepare_los_data(self, sp_name: str, doppler) -> dict:
        """Per-(species, LOS) heavy lifting. Done once, reused across all 6 phases.

        Strips yt units to plain ndarrays so the inner per-phase loop never
        triggers another unit conversion on the 16M-cell cube.
        """
        freq_3d = self._sp_data[sp_name]['freq'].in_units('Hz').value
        nu_0    = float(freq_3d.flat[0])
        lum_3d  = (self._sp_data[sp_name]['lum'] * self._volume_3d
                   ).in_units('erg/s').value
        therm   = self._sp_data[sp_name]['width'].in_units('cm/s').value
        # doppler is dimensionless; freq × doppler still in Hz.
        shifted = (self._sp_data[sp_name]['freq'] * doppler
                   ).in_units('Hz').value

        v_range_cgs = V_RANGE_KMS * 1.0e5
        bw_hz       = nu_0 * (v_range_cgs / _C_CGS) * 2.0
        freq_edges  = np.linspace(nu_0 - bw_hz / 2, nu_0 + bw_hz / 2,
                                  N_CHANNELS + 1)
        freq_ctr    = 0.5 * (freq_edges[:-1] + freq_edges[1:])
        v_axis_kms  = (_C_CGS * (nu_0 - freq_ctr) / nu_0) * 1.0e-5  # cm/s → km/s

        return {
            'nu_0':       nu_0,
            'lum_3d':     lum_3d,
            'therm':      therm,
            'shifted':    shifted,
            'freq_edges': freq_edges,
            'v_axis_kms': v_axis_kms,
        }

    def _spectrum_from_prepared(self, prep: dict, plane_key: str, mask_3d):
        """Build spectrum from pre-computed (species, LOS) data + a phase mask."""
        if mask_3d is None:
            lum_eff = prep['lum_3d']
        else:
            # bool × float64 → float64 directly, no astype needed.
            lum_eff = prep['lum_3d'] * mask_3d

        cube = build_spectral_cube(
            prep['shifted'],
            lum_eff,
            prep['therm'],
            prep['freq_edges'],
            _C_CGS,
        )

        _, n1, n2      = cube.shape
        total_lum      = cube.sum(axis=(1, 2))
        total_area_cm2 = n1 * n2 * self._cell_area[plane_key]
        dsigma_dv      = total_lum / total_area_cm2
        return dsigma_dv, prep['v_axis_kms']

    def compute(self, context: PipelinePlotContext) -> dict:
        T_K = self._T.in_units('K').value
        masks_3d = classify_temperature_phase(T_K)

        # Print phase populations as a quick sanity check.
        print('\n=== Phase cell counts (for PhaseResolvedSpectrum) ===')
        total = T_K.size
        for ph in PHASE_ORDER:
            n = int(masks_3d[ph].sum())
            print(f'  {ph:<6}  {n:>10}  ({100*n/total:>5.2f} %)')
        print('=====================================================\n')

        spectra: dict[str, dict[str, dict]] = {
            sp['name']: {'x': {}, 'y': {}} for sp in SPECIES_CFG
        }
        los_specs = [
            ('x', self._doppler_x, 'yz'),
            ('y', self._doppler_y, 'xz'),
        ]
        phase_keys: list[tuple[str, np.ndarray | None]] = [
            (ph, masks_3d[ph]) for ph in PHASE_ORDER
        ] + [('total', None)]

        # Stage 1: 8 heavy preps (4 species × 2 LOS). Each is one pass over
        # the 16M-cell cube. Caches lum×vol and freq×doppler as plain ndarrays.
        prepared: dict[tuple[str, str], dict] = {}
        for sp in SPECIES_CFG:
            name = sp['name']
            for los_name, doppler, _plane in los_specs:
                print(f'  prep  [{name}/{los_name}] ...')
                prepared[(name, los_name)] = self._prepare_los_data(name, doppler)

        # Stage 2: 48 spectrum builds. Each reads from prepared cache,
        # applies a phase mask, calls build_spectral_cube. Parallelized across
        # 6 workers like IntegratedSpectrumTask does.
        plane_by_los = {los_name: plane for los_name, _, plane in los_specs}

        def worker(args):
            name, los_name, ph_key, mask = args
            d, v = self._spectrum_from_prepared(
                prepared[(name, los_name)], plane_by_los[los_name], mask
            )
            return name, los_name, ph_key, (v, d)

        tasks = [
            (sp['name'], los_name, ph_key, mask)
            for sp in SPECIES_CFG
            for los_name, _, _ in los_specs
            for ph_key, mask in phase_keys
        ]

        print(f'  building {len(tasks)} spectra with 6 workers ...')
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(worker, t) for t in tasks]
            for fut in as_completed(futures):
                name, los_name, ph_key, result = fut.result()
                spectra[name][los_name][ph_key] = result
                print(f'    done  [{name}/{los_name}/{ph_key}]')

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
