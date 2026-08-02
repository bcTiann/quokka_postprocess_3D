"""Cold [C II] spectrum comparison between DESPOTIC and Cloudy."""
from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import numpy as np

from ..base import BuildTask, PipelinePlotContext, PlotTask
from ..prep import config as _cfg
from ..spectrum_units import DSIGMA_DV_UNIT, dsigma_dv_ylabel


CUTOFF_K = 3000.0
LOS = 'y'
FILENAME = 'Cplus_TQK_lt3000_DESPOTIC_Cloudy_Rinf.png'

MODELS = (
    {
        'name': 'CPLUS_DESPOTIC_TQK_LT3000_DIAGNOSTIC',
        'freq_field': 'C+_freq',
        'lum_field': 'C+_luminosity_despotic',
        'width_field': 'C+_thermal_width',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'lt',
        'selection_cutoff_K': CUTOFF_K,
        'color': '#0072B2',
        'label': 'DESPOTIC',
    },
    {
        'name': 'CPLUS_CLOUDY_TQK_LT3000_DIAGNOSTIC',
        'freq_field': 'C+_freq',
        'lum_field': 'C+_luminosity_cloudy_lowT_diagnostic',
        'width_field': 'C+_thermal_width',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'lt',
        'selection_cutoff_K': CUTOFF_K,
        'color': '#D55E00',
        'label': 'Cloudy HM2012',
    },
)


class Build_CplusLowCloudyComparison(BuildTask):
    """Build intrinsic cold spectra for both emissivity models."""

    def __init__(self, config):
        super().__init__(config)
        self.R = float('inf')
        self.spectrum_schema = 1

    def compute(self, context: PipelinePlotContext) -> dict:
        from ..services import SpectrumStore

        spectra = {}
        provider = context.provider
        for model in MODELS:
            store = SpectrumStore(provider, species_config=(model,))
            velocity, values = store.get_spectrum(
                model['name'], LOS, R=float('inf'),
            )
            spectra[model['name']] = {
                'v_axis': velocity,
                'dsigma_dv': values,
            }
            del store
            provider._cached_grid = None
            gc.collect()

        first, second = (spectra[model['name']] for model in MODELS)
        np.testing.assert_allclose(first['v_axis'], second['v_axis'])
        return {
            'spectra': spectra,
            'temperature_cutoff_K': CUTOFF_K,
            'R': float('inf'),
            'los': LOS,
            'dsigma_dv_units': DSIGMA_DV_UNIT,
            'cloudy_table': _cfg.CLOUDY_CII_LOWT_DIAGNOSTIC_TABLE_PATH,
        }


class Plot_CplusLowCloudyComparison(PlotTask):
    """Plot the two cold intrinsic spectra on a linear y axis."""

    def _gather_inputs(self, context: PipelinePlotContext) -> dict:
        return self._load_one(context, 'Build_CplusLowCloudyComparison')

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        fig, axis = plt.subplots(1, 1, figsize=(7.5, 4.9))
        for model in MODELS:
            block = results['spectra'][model['name']]
            axis.plot(
                np.asarray(block['v_axis']),
                np.asarray(block['dsigma_dv']),
                color=model['color'], lw=1.6, drawstyle='steps-mid',
                label=model['label'],
            )

        example = results['spectra'][MODELS[0]['name']]['dsigma_dv']
        axis.axvline(0.0, color='0.55', ls=':', lw=0.8)
        axis.set_xlabel(r'Velocity [km s$^{-1}$]')
        axis.set_ylabel(dsigma_dv_ylabel(
            getattr(example, 'units', results['dsigma_dv_units'])
        ))
        axis.grid(True, alpha=0.25, ls='--', lw=0.5)
        axis.legend(fontsize=8.5, frameon=False)
        axis.ticklabel_format(
            style='sci', axis='y', scilimits=(0, 0), useMathText=True,
        )
        axis.set_title(
            r'[C II] 158 μm: $T_{\rm QUOKKA}<3000\,$K, '
            r'LOS y, R=$\infty$',
            fontsize=12,
        )
        fig.tight_layout()
        output = context.config.output_dir / FILENAME
        fig.savefig(str(output), dpi=250, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {output}')
