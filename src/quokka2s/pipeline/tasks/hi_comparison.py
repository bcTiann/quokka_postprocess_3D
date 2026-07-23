"""Dedicated comparison plots for the two all-temperature H I models.

This plot task deliberately stays separate from the main 2x4 phase figure,
whose H I panel uses the DESPOTIC/QUOKKA two-regime emissivity.  It produces:

* ``HI_phase_comparison.png`` — phase diagrams with zmax/1e6 color floor
* ``HI_phase_comparison_zMax_over_1e4.png`` — same panels with zmax/1e4 floor
* ``HI_spectrum_DESPOTIC_Rinf.png`` — DESPOTIC-temperature spectrum only
* ``HI_spectrum_QUOKKA_Rinf.png`` — QUOKKA-temperature spectrum only
* ``HI_spectrum_overlay_Rinf.png`` — both intrinsic spectra overlaid
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ..base import PlotTask, PipelinePlotContext
from .phase_hist import PHASE_HISTOGRAM_SCHEMA


_HI_PHASE_CONFIG = (
    (
        'HI_DSP_T_DSP',
        r'T_{\rm DESPOTIC}',
        r'$\mathrm{H\,I}$: DESPOTIC $n_{\rm HI}$ at all temperatures',
    ),
    (
        'HI_QK_T_QK',
        r'T_{\rm QUOKKA}',
        r'$\mathrm{H\,I}$: QUOKKA $\mu$ method at all temperatures',
    ),
)

_HI_DESPOTIC_COLOR = '#0072B2'  # Okabe-Ito blue
_HI_QUOKKA_COLOR = '#D55E00'    # Okabe-Ito vermillion

_HI_SPECTRUM_CONFIG = (
    ('HI_DESPOTIC', _HI_DESPOTIC_COLOR, 'DESPOTIC'),
    ('HI_QUOKKA', _HI_QUOKKA_COLOR, 'QUOKKA'),
)

_HI_SPECTRUM_LOS = ('x', 'y')


def _intrinsic_surface_spectrum(block: dict) -> np.ndarray:
    """Return the unconvolved (R=infinity) luminosity surface-density spectrum."""
    return np.asarray(block['dsigma_dv'])


def _as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


class Plot_HIComparison(PlotTask):
    """Render the all-temperature H I phase and absolute-spectrum comparison."""

    def __init__(
        self,
        config,
        phase_filename: str = 'HI_phase_comparison.png',
        phase_4dex_filename: str = 'HI_phase_comparison_zMax_over_1e4.png',
        despotic_spectrum_filename: str = 'HI_spectrum_DESPOTIC_Rinf.png',
        quokka_spectrum_filename: str = 'HI_spectrum_QUOKKA_Rinf.png',
        overlay_spectrum_filename: str = 'HI_spectrum_overlay_Rinf.png',
    ):
        super().__init__(config, name='Plot_HIComparison')
        self.phase_filename = phase_filename
        self.phase_4dex_filename = phase_4dex_filename
        self.despotic_spectrum_filename = despotic_spectrum_filename
        self.quokka_spectrum_filename = quokka_spectrum_filename
        self.overlay_spectrum_filename = overlay_spectrum_filename

    def _gather_inputs(self, context: PipelinePlotContext) -> dict:
        phase_panels: dict[str, dict] = {}
        required_tags = {tag for tag, _, _ in _HI_PHASE_CONFIG}
        for data in self._load_all(context, 'Build_PhaseHist'):
            if int(data.get('histogram_schema', 1)) != PHASE_HISTOGRAM_SCHEMA:
                continue
            tag = _as_text(data.get('tag', ''))
            if tag in required_tags:
                phase_panels[tag] = data

        missing = sorted(required_tags - phase_panels.keys())
        if missing:
            raise RuntimeError(
                f'Plot_HIComparison: missing phase histograms {missing}; '
                'run --mode compute first.'
            )

        return {
            'phase_panels': phase_panels,
            'species_spectrum': self._load_one(context, 'Build_SpeciesSpectrum'),
        }

    def plot(self, context: PipelinePlotContext, inputs: dict) -> None:
        self._plot_phase(
            context,
            inputs['phase_panels'],
            dynamic_range=1.0e6,
            filename=self.phase_filename,
        )
        self._plot_phase(
            context,
            inputs['phase_panels'],
            dynamic_range=1.0e4,
            filename=self.phase_4dex_filename,
        )
        self._plot_spectra(context, inputs['species_spectrum'])

    def _plot_phase(
        self,
        context: PipelinePlotContext,
        panels: dict,
        *,
        dynamic_range: float,
        filename: str,
    ) -> None:
        positive_logs = []
        for tag, _, _ in _HI_PHASE_CONFIG:
            H = np.asarray(panels[tag]['H'])
            positive = H[H > 0]
            if positive.size:
                positive_logs.append(float(np.log10(positive.max())))
        zmax = max(positive_logs) if positive_logs else 0.0
        norm = Normalize(vmin=zmax - np.log10(dynamic_range), vmax=zmax)

        x_lo = min(float(np.asarray(panels[tag]['x_edges'])[0])
                   for tag, _, _ in _HI_PHASE_CONFIG)
        x_hi = max(float(np.asarray(panels[tag]['x_edges'])[-1])
                   for tag, _, _ in _HI_PHASE_CONFIG)
        y_lo = min(float(np.asarray(panels[tag]['y_edges'])[0])
                   for tag, _, _ in _HI_PHASE_CONFIG)
        y_hi = max(float(np.asarray(panels[tag]['y_edges'])[-1])
                   for tag, _, _ in _HI_PHASE_CONFIG)

        # Match the physical panel size and near-square aspect used by each
        # panel in phase_combined.png.  A horizontal colorbar above each axis
        # also keeps the main panels from being squeezed by one tall shared
        # colorbar on the right.
        panel_inch = 3.4
        fig = plt.figure(figsize=(panel_inch * 2 + 0.4, panel_inch + 0.5))
        grid = fig.add_gridspec(
            2, 2,
            height_ratios=[0.06, 1.0],
            hspace=0.05,
            wspace=0.30,
            left=0.08, right=0.99, top=0.92, bottom=0.15,
        )
        caxes = [fig.add_subplot(grid[0, col]) for col in range(2)]
        axes = [fig.add_subplot(grid[1, col]) for col in range(2)]
        axes[1].sharex(axes[0])
        axes[1].sharey(axes[0])

        for ax, cax, (tag, temperature_label, _title) in zip(
            axes, caxes, _HI_PHASE_CONFIG,
        ):
            data = panels[tag]
            H = np.asarray(data['H'])
            x_edges = np.asarray(data['x_edges'])
            y_edges = np.asarray(data['y_edges'])
            with np.errstate(divide='ignore'):
                view = np.where(H > 0, np.log10(H), np.nan)
            image = ax.pcolormesh(
                x_edges,
                y_edges,
                view.T,
                shading='flat',
                cmap='viridis_r',
                norm=norm,
            )
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel(r'$\log_{10}\,\rho$ [g cm$^{-3}$]')
            ax.set_ylabel(rf'$\log_{{10}}\,{temperature_label}$ [K]')
            ax.tick_params(labelsize=8)

            cbar = fig.colorbar(image, cax=cax, orientation='horizontal')
            cbar.ax.tick_params(
                labelsize=8, top=True, bottom=False,
                labeltop=True, labelbottom=False,
            )
            cax.set_title(
                r'$\log_{10}\,L_{\rm HI,bin}$ [erg s$^{-1}$]',
                fontsize=9,
                pad=4,
            )
        out = context.config.output_dir / filename
        fig.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')

    @staticmethod
    def _format_spectrum_axis(ax, los: str) -> None:
        ax.set_xlabel('Velocity [km/s]')
        ax.set_ylabel(
            r'$\mathrm{d}\Sigma_L/\mathrm{d}\nu$ '
            r'[erg s$^{-1}$ Hz$^{-1}$ cm$^{-2}$]'
        )
        ax.set_title(f'LOS {los}')
        ax.ticklabel_format(
            style='sci', axis='y', scilimits=(0, 0), useMathText=True,
        )

    def _plot_single_spectrum(
        self,
        context: PipelinePlotContext,
        spectra: dict,
        *,
        species: str,
        color: str,
        temperature_name: str,
        filename: str,
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=False)
        for ax, los in zip(axes, _HI_SPECTRUM_LOS):
            block = spectra[species][los]['total']
            ax.plot(
                np.asarray(block['v_axis']),
                _intrinsic_surface_spectrum(block),
                color=color,
                lw=1.6,
                drawstyle='steps-mid',
            )
            self._format_spectrum_axis(ax, los)

        fig.suptitle(
            rf'$\mathrm{{H\,I}}$: $T_{{\rm {temperature_name}}}$, $R=\infty$',
            fontsize=12,
        )
        fig.tight_layout()
        out = context.config.output_dir / filename
        fig.savefig(str(out), dpi=250, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')

    def _plot_overlay_spectrum(
        self,
        context: PipelinePlotContext,
        spectra: dict,
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=False)
        for ax, los in zip(axes, _HI_SPECTRUM_LOS):
            for species, color, label in _HI_SPECTRUM_CONFIG:
                block = spectra[species][los]['total']
                ax.plot(
                    np.asarray(block['v_axis']),
                    _intrinsic_surface_spectrum(block),
                    color=color,
                    lw=1.6,
                    drawstyle='steps-mid',
                    label=label,
                )
            self._format_spectrum_axis(ax, los)
            ax.legend(frameon=False)

        fig.suptitle(
            r'$\mathrm{H\,I}$: DESPOTIC and QUOKKA, $R=\infty$',
            fontsize=12,
        )
        fig.tight_layout()
        out = context.config.output_dir / self.overlay_spectrum_filename
        fig.savefig(str(out), dpi=250, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')

    def _plot_spectra(self, context: PipelinePlotContext, results: dict) -> None:
        spectra = results['spectra']
        self._plot_single_spectrum(
            context,
            spectra,
            species='HI_DESPOTIC',
            color=_HI_DESPOTIC_COLOR,
            temperature_name='DESPOTIC',
            filename=self.despotic_spectrum_filename,
        )
        self._plot_single_spectrum(
            context,
            spectra,
            species='HI_QUOKKA',
            color=_HI_QUOKKA_COLOR,
            temperature_name='QUOKKA',
            filename=self.quokka_spectrum_filename,
        )
        self._plot_overlay_spectrum(context, spectra)
