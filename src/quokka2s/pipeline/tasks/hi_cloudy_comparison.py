"""Compare QUOKKA-mu, DESPOTIC, and Cloudy H I 21-cm spectra."""
from __future__ import annotations

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..base import BuildTask, PipelinePlotContext, PlotTask
from ..prep import config as _cfg
from ..spectrum_units import DSIGMA_DV_UNIT, dsigma_dv_ylabel


HI_CLOUDY_CUTOFF_K = 3000.0
HI_CLOUDY_COMPARISON_LOS = ('y',)
HI_CLOUDY_COMPARISON_FILENAME = (
    'HI21_QUOKKA_Cloudy_DESPOTIC_Tsplit_no_lowT_QUOKKA_Rinf.png'
)

HI_CLOUDY_COMPARISON_CONFIG = (
    {
        'name': 'HI_QUOKKA_LOW',
        'freq_field': 'HI_freq',
        'lum_field': 'HI_luminosity_quokka',
        'width_field': 'HI_thermal_width_quokka',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'lt',
        'selection_cutoff_K': HI_CLOUDY_CUTOFF_K,
        'branch': 'low',
        'color': '#009E73',
        'label': r'QUOKKA $\mu$',
    },
    {
        'name': 'HI_DESPOTIC_LOW',
        'freq_field': 'HI_freq',
        'lum_field': 'HI_luminosity_despotic',
        'width_field': 'HI_thermal_width_quokka',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'lt',
        'selection_cutoff_K': HI_CLOUDY_CUTOFF_K,
        'branch': 'low',
        'color': '#0072B2',
        'label': 'DESPOTIC',
    },
    {
        'name': 'HI_CLOUDY_LOW',
        'freq_field': 'HI_freq',
        'lum_field': 'HI_luminosity_cloudy',
        'width_field': 'HI_thermal_width_quokka',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'lt',
        'selection_cutoff_K': HI_CLOUDY_CUTOFF_K,
        'branch': 'low',
        'color': '#D55E00',
        'label': 'Cloudy HM2012',
    },
    {
        'name': 'HI_QUOKKA_HIGH',
        'freq_field': 'HI_freq',
        'lum_field': 'HI_luminosity_quokka',
        'width_field': 'HI_thermal_width_quokka',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'ge',
        'selection_cutoff_K': HI_CLOUDY_CUTOFF_K,
        'branch': 'high',
        'color': '#009E73',
        'label': r'QUOKKA $\mu$',
    },
    {
        'name': 'HI_DESPOTIC_HIGH',
        'freq_field': 'HI_freq',
        'lum_field': 'HI_luminosity_despotic',
        'width_field': 'HI_thermal_width_quokka',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'ge',
        'selection_cutoff_K': HI_CLOUDY_CUTOFF_K,
        'branch': 'high',
        'color': '#0072B2',
        'label': 'DESPOTIC',
    },
    {
        'name': 'HI_CLOUDY_HIGH',
        'freq_field': 'HI_freq',
        'lum_field': 'HI_luminosity_cloudy',
        'width_field': 'HI_thermal_width_quokka',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'ge',
        'selection_cutoff_K': HI_CLOUDY_CUTOFF_K,
        'branch': 'high',
        'color': '#D55E00',
        'label': 'Cloudy HM2012',
    },
)

# The low-temperature QUOKKA-mu neutral fraction is not physically valid for
# this diagnostic.  Keep it in the cached build result for provenance, but do
# not include it in the revised comparison figure.
HI_CLOUDY_PLOT_CONFIG = tuple(
    model for model in HI_CLOUDY_COMPARISON_CONFIG
    if model['name'] != 'HI_QUOKKA_LOW'
)


class Build_HICloudyComparison(BuildTask):
    """Build six intrinsic spectra with a common LOS and channel grid."""

    def __init__(self, config):
        super().__init__(config)
        self.spectrum_schema = 2
        self.cloudy_table_mtime = Path(
            _cfg.CLOUDY_HI21_TABLE_PATH
        ).stat().st_mtime

    def compute(self, context: PipelinePlotContext) -> dict:
        from ..services import SpectrumStore

        provider = context.provider
        spectra: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        provider._cached_grid = None
        gc.collect()

        for model in HI_CLOUDY_COMPARISON_CONFIG:
            name = model['name']
            spectra[name] = {}
            store = SpectrumStore(provider, species_config=(model,))
            for los in HI_CLOUDY_COMPARISON_LOS:
                velocity, spectrum = store.get_spectrum(
                    name, los, R=float('inf'),
                )
                spectra[name][los] = {
                    'v_axis': velocity,
                    'dsigma_dv': spectrum,
                }
                print(
                    f'  [{name}] peak={float(spectrum.max()):.6e} '
                    f'{DSIGMA_DV_UNIT}'
                )
            del store
            provider._cached_grid = None
            gc.collect()

        return {
            'spectra': spectra,
            'temperature_cutoff_K': HI_CLOUDY_CUTOFF_K,
            'R': float('inf'),
            'dsigma_dv_units': DSIGMA_DV_UNIT,
        }


class Plot_HICloudyComparison(PlotTask):
    """Draw the cold and hot H I model comparisons on one absolute scale."""

    def _gather_inputs(self, context: PipelinePlotContext) -> dict:
        return self._load_one(context, 'Build_HICloudyComparison')

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        los = HI_CLOUDY_COMPARISON_LOS[0]
        spectra = results['spectra']
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=False)

        maxima = []
        for model in HI_CLOUDY_PLOT_CONFIG:
            values = np.asarray(
                spectra[model['name']][los]['dsigma_dv'], dtype=float,
            )
            finite = values[np.isfinite(values)]
            if finite.size:
                maxima.append(float(finite.max()))
        shared_max = max(maxima, default=0.0)
        if shared_max <= 0.0:
            raise ValueError('H I comparison spectra contain no positive values')

        branch_peak: dict[str, float] = {}
        for axis, branch, title in (
            (axes[0], 'low', r'$T_{\rm QUOKKA}<3000\,$K'),
            (axes[1], 'high', r'$T_{\rm QUOKKA}\geq3000\,$K'),
        ):
            models = [
                model for model in HI_CLOUDY_PLOT_CONFIG
                if model['branch'] == branch
            ]
            branch_peak[branch] = 0.0
            for model in models:
                block = spectra[model['name']][los]
                values = np.asarray(block['dsigma_dv'])
                branch_peak[branch] = max(
                    branch_peak[branch], float(np.nanmax(values)),
                )
                axis.plot(
                    np.asarray(block['v_axis']), values,
                    color=model['color'], lw=1.6,
                    drawstyle='steps-mid', label=model['label'],
                )
            example = spectra[models[0]['name']][los]['dsigma_dv']
            axis.axvline(0.0, color='0.55', ls=':', lw=0.8)
            axis.set_xlabel(r'Velocity [km s$^{-1}$]')
            axis.set_ylabel(dsigma_dv_ylabel(
                getattr(example, 'units', results['dsigma_dv_units'])
            ))
            axis.set_title(title)
            axis.set_ylim(0.0, 1.05 * shared_max)
            axis.grid(True, alpha=0.25, ls='--', lw=0.5)
            axis.legend(fontsize=9, frameon=False)
            axis.ticklabel_format(
                style='sci', axis='y', scilimits=(0, 0), useMathText=True,
            )

        # Keep the requested common absolute scale, while retaining a readable
        # view of a much fainter branch when its peak is below 10% of the total.
        for axis, branch in zip(axes, ('low', 'high')):
            if branch_peak[branch] >= 0.10 * shared_max:
                continue
            inset = axis.inset_axes([0.08, 0.52, 0.48, 0.40])
            for model in (
                item for item in HI_CLOUDY_PLOT_CONFIG
                if item['branch'] == branch
            ):
                block = spectra[model['name']][los]
                inset.plot(
                    np.asarray(block['v_axis']),
                    np.asarray(block['dsigma_dv']),
                    color=model['color'], lw=1.25,
                    drawstyle='steps-mid',
                )
            inset.set_xlim(-35.0, 35.0)
            inset.set_ylim(0.0, 1.08 * branch_peak[branch])
            inset.ticklabel_format(
                style='sci', axis='y', scilimits=(0, 0), useMathText=True,
            )
            inset.tick_params(labelsize=7)
            inset.grid(True, alpha=0.2, ls='--', lw=0.4)
            inset.set_title('zoom', fontsize=8)

        fig.suptitle(
            r'Comparison of the H I 21-cm spectrum from QUOKKA $\mu$, '
            r'DESPOTIC, and Cloudy'
            r', LOS y, $R=\infty$'
        )
        fig.tight_layout()
        output = context.config.output_dir / HI_CLOUDY_COMPARISON_FILENAME
        fig.savefig(str(output), dpi=250, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {output}')
