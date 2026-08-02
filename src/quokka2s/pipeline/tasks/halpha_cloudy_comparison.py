"""Compare pipeline and Cloudy H-alpha spectra below/above 3000 K."""
from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..base import BuildTask, PipelinePlotContext, PlotTask
from ..prep import config as _cfg
from ..spectrum_units import DSIGMA_DV_UNIT, dsigma_dv_ylabel


HALPHA_CUTOFF_K = 3000.0
HALPHA_COMPARISON_LOS = ('y',)
HALPHA_COMPARISON_FILENAME = (
    'Halpha_Cloudy_vs_pipeline_Tsplit_Rinf.png'
)

HALPHA_COMPARISON_CONFIG = (
    {
        'name': 'HALPHA_PIPELINE_LOW',
        'freq_field': 'H_alpha_freq',
        'lum_field': 'H_alpha_luminosity',
        'width_field': 'H_alpha_thermal_width',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'lt',
        'selection_cutoff_K': HALPHA_CUTOFF_K,
        'branch': 'low',
        'color': '#0072B2',
        'label': 'pipeline',
    },
    {
        'name': 'HALPHA_CLOUDY_LOW',
        'freq_field': 'H_alpha_freq',
        'lum_field': 'H_alpha_luminosity_cloudy_low',
        'width_field': 'H_alpha_thermal_width_quokka',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'lt',
        'selection_cutoff_K': HALPHA_CUTOFF_K,
        'branch': 'low',
        'color': '#D55E00',
        'label': 'Cloudy HM2012',
    },
    {
        'name': 'HALPHA_PIPELINE_HIGH',
        'freq_field': 'H_alpha_freq',
        'lum_field': 'H_alpha_luminosity',
        'width_field': 'H_alpha_thermal_width',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'ge',
        'selection_cutoff_K': HALPHA_CUTOFF_K,
        'branch': 'high',
        'color': '#0072B2',
        'label': 'pipeline',
    },
    {
        'name': 'HALPHA_CLOUDY_HIGH',
        'freq_field': 'H_alpha_freq',
        'lum_field': 'H_alpha_luminosity_cloudy_high',
        'width_field': 'H_alpha_thermal_width_quokka',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'ge',
        'selection_cutoff_K': HALPHA_CUTOFF_K,
        'branch': 'high',
        'color': '#D55E00',
        'label': 'Cloudy HM2012',
    },
)


class Build_HalphaCloudyComparison(BuildTask):
    """Build four intrinsic spectra using identical channels and LOS."""

    def __init__(self, config):
        super().__init__(config)
        self.spectrum_schema = 1
        # Fold both Cloudy table identities into this task's L2 cache filename
        # without invalidating unrelated pipeline field caches.
        self.low_table_mtime = Path(_cfg.CLOUDY_HALPHA_LOWT_TABLE_PATH).stat().st_mtime
        self.high_table_mtime = Path(_cfg.CLOUDY_HALPHA_HIGH_TABLE_PATH).stat().st_mtime

    def compute(self, context: PipelinePlotContext) -> dict:
        from ..services import SpectrumStore

        provider = context.provider
        spectra: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        provider._cached_grid = None
        gc.collect()

        for model in HALPHA_COMPARISON_CONFIG:
            name = model['name']
            spectra[name] = {}
            store = SpectrumStore(provider, species_config=(model,))
            for los in HALPHA_COMPARISON_LOS:
                velocity, spectrum = store.get_spectrum(
                    name, los, R=float('inf'),
                )
                spectra[name][los] = {
                    'v_axis': velocity,
                    'dsigma_dv': spectrum,
                }
            del store
            provider._cached_grid = None
            gc.collect()

        return {
            'spectra': spectra,
            'temperature_cutoff_K': HALPHA_CUTOFF_K,
            'R': float('inf'),
            'dsigma_dv_units': DSIGMA_DV_UNIT,
        }


class Plot_HalphaCloudyComparison(PlotTask):
    """Draw cold and hot absolute H-alpha model comparisons side by side."""

    def _gather_inputs(self, context: PipelinePlotContext) -> dict:
        return self._load_one(context, 'Build_HalphaCloudyComparison')

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        los = HALPHA_COMPARISON_LOS[0]
        spectra = results['spectra']
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=False)

        finite_maxima = []
        for model in HALPHA_COMPARISON_CONFIG:
            values = np.asarray(
                spectra[model['name']][los]['dsigma_dv'], dtype=float,
            )
            finite = values[np.isfinite(values)]
            if finite.size:
                finite_maxima.append(float(finite.max()))
        shared_max = max(finite_maxima, default=0.0)
        if shared_max <= 0.0:
            raise ValueError('H-alpha comparison spectra contain no positive values')

        for axis, branch, title in (
            (axes[0], 'low', r'$T_{\rm QUOKKA}<3000\,$K'),
            (axes[1], 'high', r'$T_{\rm QUOKKA}\geq3000\,$K'),
        ):
            branch_models = [
                model for model in HALPHA_COMPARISON_CONFIG
                if model['branch'] == branch
            ]
            for model in branch_models:
                block = spectra[model['name']][los]
                axis.plot(
                    np.asarray(block['v_axis']),
                    np.asarray(block['dsigma_dv']),
                    color=model['color'],
                    lw=1.6,
                    drawstyle='steps-mid',
                    label=model['label'],
                )
            example = spectra[branch_models[0]['name']][los]['dsigma_dv']
            axis.axvline(0.0, color='0.55', ls=':', lw=0.8)
            axis.set_xlabel(r'Velocity [km s$^{-1}$]')
            axis.set_ylabel(dsigma_dv_ylabel(
                getattr(example, 'units', results['dsigma_dv_units'])
            ))
            axis.set_title(title)
            axis.grid(True, alpha=0.25, ls='--', lw=0.5)
            axis.legend(fontsize=9, frameon=False)
            axis.set_ylim(0.0, 1.05 * shared_max)
            axis.ticklabel_format(
                style='sci', axis='y', scilimits=(0, 0), useMathText=True,
            )

        # Preserve the shared absolute main-axis scale while making the much
        # fainter low-temperature profiles legible in a local inset.
        low_axis = axes[0]
        inset = low_axis.inset_axes([0.08, 0.52, 0.48, 0.40])
        low_peak = 0.0
        for model in (
            item for item in HALPHA_COMPARISON_CONFIG
            if item['branch'] == 'low'
        ):
            block = spectra[model['name']][los]
            velocity = np.asarray(block['v_axis'])
            values = np.asarray(block['dsigma_dv'])
            low_peak = max(low_peak, float(np.nanmax(values)))
            inset.plot(
                velocity,
                values,
                color=model['color'],
                lw=1.25,
                drawstyle='steps-mid',
            )
        inset.set_xlim(-35.0, 35.0)
        inset.set_ylim(0.0, 1.08 * low_peak)
        inset.ticklabel_format(
            style='sci', axis='y', scilimits=(0, 0), useMathText=True,
        )
        inset.tick_params(labelsize=7)
        inset.grid(True, alpha=0.2, ls='--', lw=0.4)
        inset.set_title('zoom', fontsize=8)

        fig.suptitle(
            r'Comparison of the H$\alpha$ spectrum from pipeline and Cloudy'
            r', LOS y, $R=\infty$'
        )
        fig.tight_layout()
        output = context.config.output_dir / HALPHA_COMPARISON_FILENAME
        fig.savefig(str(output), dpi=250, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {output}')
