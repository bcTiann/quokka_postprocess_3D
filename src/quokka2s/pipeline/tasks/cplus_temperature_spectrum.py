"""[C II] spectra split by the T_QUOKKA = 3000 K model boundary."""
from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import numpy as np

from ..base import BuildTask, PipelinePlotContext, PlotTask
from ..prep import config as _cfg
from ..spectrum_units import DSIGMA_DV_UNIT, dsigma_dv_ylabel


CPLUS_TEMPERATURE_CUTOFF_K = 3000.0
CPLUS_TEMPERATURE_SPECTRUM_LOS = ('y',)
CPLUS_TEMPERATURE_SPECTRUM_FILENAME = 'Cplus_temperature_split_spectrum.png'

CPLUS_TEMPERATURE_COMPONENTS = (
    {
        'name': 'CPLUS_DESPOTIC_TQK_LT3000',
        'freq_field': 'C+_freq',
        'lum_field': 'C+_luminosity',
        'width_field': 'C+_thermal_width',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'lt',
        'selection_cutoff_K': CPLUS_TEMPERATURE_CUTOFF_K,
        'color': '#0072B2',
        'label': r'DESPOTIC: $T_{\rm QUOKKA}<3000\,\mathrm{K}$',
    },
    {
        'name': 'CPLUS_CLOUDY_TQK_GE3000',
        'freq_field': 'C+_freq',
        'lum_field': 'C+_luminosity',
        'width_field': 'C+_thermal_width',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'ge',
        'selection_cutoff_K': CPLUS_TEMPERATURE_CUTOFF_K,
        'color': '#D55E00',
        'label': r'Cloudy: $T_{\rm QUOKKA}\geq3000\,\mathrm{K}$',
    },
)


def combine_cplus_component_spectra(
    cold: np.ndarray,
    hot: np.ndarray,
) -> np.ndarray:
    """Return the exact hybrid total from the two disjoint cell sets."""
    cold_array, hot_array = np.broadcast_arrays(cold, hot)
    return cold_array + hot_array


class Build_CplusTemperatureSpectrum(BuildTask):
    """Build DESPOTIC-cold, Cloudy-hot, and summed [C II] spectra."""

    def __init__(self, config, R: float | None = None):
        super().__init__(config)
        self.R = R if R is not None else _cfg.SPECTRAL_RESOLUTION_R
        self.spectrum_schema = 1

    def compute(self, context: PipelinePlotContext) -> dict:
        from ..services import SpectrumStore

        provider = context.provider
        spectra: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        provider._cached_grid = None
        gc.collect()

        for component in CPLUS_TEMPERATURE_COMPONENTS:
            name = component['name']
            spectra[name] = {}
            store = SpectrumStore(provider, species_config=(component,))
            for los in CPLUS_TEMPERATURE_SPECTRUM_LOS:
                v_axis, intrinsic = store.get_spectrum(
                    name, los, R=float('inf'),
                )
                _, observed = store.get_spectrum(name, los, R=self.R)
                spectra[name][los] = {
                    'v_axis': v_axis,
                    'dsigma_dv': intrinsic,
                    'dsigma_dv_obs': observed,
                }
            del store
            provider._cached_grid = None
            gc.collect()

        spectra['CPLUS_TOTAL'] = {}
        cold_name, hot_name = (
            component['name'] for component in CPLUS_TEMPERATURE_COMPONENTS
        )
        for los in CPLUS_TEMPERATURE_SPECTRUM_LOS:
            cold = spectra[cold_name][los]
            hot = spectra[hot_name][los]
            np.testing.assert_allclose(cold['v_axis'], hot['v_axis'])
            spectra['CPLUS_TOTAL'][los] = {
                'v_axis': cold['v_axis'],
                'dsigma_dv': combine_cplus_component_spectra(
                    cold['dsigma_dv'], hot['dsigma_dv'],
                ),
                'dsigma_dv_obs': combine_cplus_component_spectra(
                    cold['dsigma_dv_obs'], hot['dsigma_dv_obs'],
                ),
            }

        return {
            'spectra': spectra,
            'temperature_cutoff_K': CPLUS_TEMPERATURE_CUTOFF_K,
            'R': self.R,
            'dsigma_dv_units': DSIGMA_DV_UNIT,
        }


class Plot_CplusTemperatureSpectrum(PlotTask):
    """Plot cold, hot, and total absolute [C II] spectra together."""

    def _gather_inputs(self, context: PipelinePlotContext) -> dict:
        return self._load_one(context, 'Build_CplusTemperatureSpectrum')

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        los = CPLUS_TEMPERATURE_SPECTRUM_LOS[0]
        spectra = results['spectra']
        fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.9))

        for component in CPLUS_TEMPERATURE_COMPONENTS:
            block = spectra[component['name']][los]
            ax.plot(
                np.asarray(block['v_axis']),
                np.asarray(block['dsigma_dv_obs']),
                color=component['color'],
                lw=1.5,
                drawstyle='steps-mid',
                label=component['label'],
            )

        total = spectra['CPLUS_TOTAL'][los]
        ax.plot(
            np.asarray(total['v_axis']),
            np.asarray(total['dsigma_dv_obs']),
            color='black',
            lw=2.0,
            drawstyle='steps-mid',
            label='Total = DESPOTIC cold + Cloudy hot',
        )
        ax.axvline(0.0, color='0.55', ls=':', lw=0.8)
        ax.set_xlabel(r'Velocity [km s$^{-1}$]')
        ax.set_ylabel(dsigma_dv_ylabel(
            getattr(total['dsigma_dv_obs'], 'units', results['dsigma_dv_units'])
        ))
        ax.set_title(f'[C II] 158 μm, LOS {los}, R={results["R"]:.0e}')
        ax.ticklabel_format(
            style='sci', axis='y', scilimits=(0, 0), useMathText=True,
        )
        ax.grid(True, alpha=0.25, ls='--', lw=0.5)
        ax.legend(fontsize=8.5, frameon=False)
        fig.tight_layout()

        output = context.config.output_dir / CPLUS_TEMPERATURE_SPECTRUM_FILENAME
        fig.savefig(str(output), dpi=250, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {output}')
