"""Plot_PhaseSpectrumOverlay (pure Plot task).

Overlays each species' integrated line profile on the 5 ISM-phase velocity
PDFs.  Computes nothing; reads two Build results fresh at plot time (see
``_gather_inputs``):
  - Build_VelocityPhase   → fixed-range velocity PDFs per phase
  - Build_SpeciesSpectrum → species total spectra
The Build tasks must run first (``--mode compute``).

Produces (2026-06-20 redesign; LOS=y only per the los-y-default convention):

  PhaseSpectrumOverlay_<species>_losy_bin{bin_size}_R{R_tag}.png   (1 per species)

each a single axis: the 5 phase PDFs (tab10, cold→hot) + the species emission
curve (black) on a shared (v_LOS, normalised intensity) panel.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..base import PlotTask, PipelinePlotContext
from ..utils import PHASE_ORDER, PHASE_LABEL_LINE
from .integrated_spectrum import SPECIES_CFG, V_RANGE_KMS


# 2026-06-20 redesign: drop 'total' (visual sum of the 5 phases — redundant)
# and put all 5 phase PDFs + 1 species emission curve on a SINGLE axis per
# species.  Palette is matplotlib's tab10 ordered cold→hot, plus black for
# the species emission — all six well-separated.
PHASE_PLOT_ORDER = PHASE_ORDER                # ('CNM','UNM','WNM','WIM','HIM')
PHASE_COLOR = {
    'CNM': '#1f77b4',     # tab10 blue
    'UNM': '#17becf',     # tab10 cyan
    'WNM': '#2ca02c',     # tab10 green
    'WIM': '#ff7f0e',     # tab10 orange
    'HIM': '#d62728',     # tab10 red
}
SPECTRUM_COLOR = 'black'


def _moment_sigma(v: np.ndarray, spec: np.ndarray) -> tuple[float, float]:
    total = spec.sum()
    if total <= 0:
        return float('nan'), float('nan')
    w      = spec / total
    v_mean = float(np.sum(v * w))
    sigma  = float(np.sqrt(np.sum((v - v_mean) ** 2 * w)))
    return v_mean, sigma


class Plot_PhaseSpectrumOverlay(PlotTask):
    """Species spectrum × phase velocity PDF overlay (1 PNG per species)."""

    def __init__(self, config, bin_size: int = 1, R: float = np.inf):
        super().__init__(config, name='Plot_PhaseSpectrumOverlay')
        self.bin_size = int(bin_size)
        if self.bin_size != 1:
            raise NotImplementedError('bin_size > 1 not supported.')
        self.R = float(R)

    def _gather_inputs(self, context: PipelinePlotContext) -> dict:
        """Load the velocity PDFs (Build_VelocityPhase) + species spectra
        (Build_SpeciesSpectrum) fresh from disk (cache-key validated)."""
        vp = self._load_one(context, 'Build_VelocityPhase')
        ss = self._load_one(context, 'Build_SpeciesSpectrum')
        total_3d = vp.get('total_3d')
        if total_3d is not None and 'sigma_3d' in total_3d:
            sigma_3d_total = float(total_3d['sigma_3d'])
        else:
            # Compatibility with older intermediates that predate total_3d.
            sigma_3d_total = float(np.sqrt(sum(
                float(vp[f'total_{axis}']['sigma'])**2 for axis in ('x', 'y', 'z')
            )))
        return {
            'pdf':            vp['pdf_fixed'],    # fixed-range PDFs from VelocityPhase
            'spectra':        ss['spectra'],      # species × LOS × 'total'
            'sigma_3d_total': sigma_3d_total,     # all-gas 3D context for legend
        }

    def plot(self, context: PipelinePlotContext, inputs: dict) -> None:
        # 2026-06-25: one PNG per (species, LOS) for all three LOS x/y/z.
        for los in ('x', 'y', 'z'):
            for sp in SPECIES_CFG:
                self._plot_one_species_one_los(inputs, los, sp)

    def _plot_one_species_one_los(self, results: dict, los: str,
                                  sp_cfg: dict) -> None:
        """One PNG, 1 axis: 5 phase mass PDFs + 'total' + 1 species emission
        curve overlaid on a single (v_LOS, normalised intensity) panel.

        2026-06-20 redesign: all five phases + the species spectrum on one
        axis for legible cross-phase comparison.  2026-06-25: 'total'
        (all-gas) PDF re-added (gray dashed) at the user's request, and the
        plot runs for all three LOS (x/y/z) — each figure's σ is the
        dispersion of the velocity component along THAT LOS (σ_x/σ_y/σ_z),
        and the species emission curve is the LOS-matched projection.
        """
        pdf     = results['pdf']
        spectra = results['spectra']

        # Coerce h5py-loaded string keys back to native python strings.
        def _key(d, key):
            if key in d:
                return d[key]
            for k in d.keys():
                if (isinstance(k, bytes) and k.decode() == key) or str(k) == key:
                    return d[k]
            raise KeyError(key)

        name = sp_cfg['name']
        bin_centers = np.asarray(_key(_key(pdf, los), 'bin_centers'))

        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        x_window = (-V_RANGE_KMS, V_RANGE_KMS)

        # Species spectrum (drawn last so it sits on top).
        spec_dict = _key(_key(spectra, name), los)
        spec_total_block = _key(spec_dict, 'total')
        spec_v = np.asarray(_key(spec_total_block, 'v_axis'))
        try:
            spec_y = np.asarray(_key(spec_total_block, 'dsigma_dv_obs'))
        except KeyError:
            spec_y = np.asarray(_key(spec_total_block, 'dsigma_dv'))
        spec_peak = spec_y.max()
        spec_norm = spec_y / spec_peak if spec_peak > 0 else spec_y
        _, sigma_obs = _moment_sigma(spec_v, spec_y)
        sigma_obs_str = f'{sigma_obs:.1f}' if np.isfinite(sigma_obs) else 'nan'
        sigma_3d_total = float(results['sigma_3d_total'])

        # 5 phase PDFs (in PHASE_ORDER = cold→hot).  σ shown is the dispersion
        # of the velocity component ALONG this figure's LOS (σ_x / σ_y / σ_z).
        for phase in PHASE_PLOT_ORDER:
            phase_pdf = _key(_key(pdf, los), phase)
            counts    = np.asarray(_key(phase_pdf, 'counts'))
            sigma_p   = float(np.asarray(_key(phase_pdf, 'sigma')))
            peak      = counts.max() if counts.size else 0.0
            if peak <= 0:
                continue

            sigma_str = f'{sigma_p:.1f}' if np.isfinite(sigma_p) else 'nan'
            label = rf'{phase}   $\sigma_{los}$ = {sigma_str} km/s'
            ax.step(bin_centers, counts / peak, where='mid',
                    color=PHASE_COLOR[phase], lw=1.6, label=label)

        # 'total' (all-gas) mass PDF — gray dashed, distinct from the 5 phases
        # (black is reserved for the species emission curve).
        total_pdf    = _key(_key(pdf, los), 'total')
        total_counts = np.asarray(_key(total_pdf, 'counts'))
        total_sigma  = float(np.asarray(_key(total_pdf, 'sigma')))
        total_peak   = total_counts.max() if total_counts.size else 0.0
        if total_peak > 0:
            tsig_str = f'{total_sigma:.1f}' if np.isfinite(total_sigma) else 'nan'
            ax.step(bin_centers, total_counts / total_peak, where='mid',
                    color='0.4', ls='--', lw=1.6,
                    label=rf'total   $\sigma_{los}$ = {tsig_str} km/s')

        # Species emission on top.
        ax.step(spec_v, spec_norm, where='mid',
                color=SPECTRUM_COLOR, lw=2.0, alpha=0.9,
                label=rf'{name} emission   $\sigma_v$ = {sigma_obs_str} km/s')

        # Context-only legend entry.  σ_3D,total is not a LOS line width, so it
        # deliberately has no curve or marker on this signed LOS-velocity axis.
        if np.isfinite(sigma_3d_total):
            ax.plot([], [], ls='none', marker=None, color='none',
                    label=rf'all gas   $\sigma_{{\rm total}}$ = '
                          rf'{sigma_3d_total:.1f} km/s')

        ax.axvline(0, color='gray', ls=':', lw=0.8, alpha=0.6)
        ax.set_xlim(*x_window)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.25, ls='--', lw=0.5)
        ax.set_xlabel(rf'$v_{los}$  [km/s]', fontsize=11)
        ax.set_ylabel('Normalised intensity / mass PDF', fontsize=11)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.85,
                  handlelength=1.6, borderpad=0.4,
                  labelspacing=0.35, handletextpad=0.5)

        R_tag = 'inf' if not np.isfinite(self.R) else f'{int(self.R)}'
        if np.isfinite(self.R):
            title = f'{name} line profile vs phase velocity PDFs   (LOS={los}, R={int(self.R)})'
        else:
            title = f'{name} line profile vs phase velocity PDFs   (LOS={los})'
        ax.set_title(title, fontsize=12)
        # Phase definition legend as a small footer.
        fig.text(0.5, -0.01, PHASE_LABEL_LINE, fontsize=8,
                 ha='center', va='top', color='gray')

        plt.tight_layout()
        species_safe = name.replace('+', 'plus')
        out = self.config.output_dir / (
            f'PhaseSpectrumOverlay_{species_safe}_los{los}'
            f'_bin{self.bin_size}_R{R_tag}.png'
        )
        plt.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
