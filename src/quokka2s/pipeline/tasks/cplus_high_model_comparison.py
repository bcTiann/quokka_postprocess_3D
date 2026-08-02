"""High-temperature [C II] spectrum comparison: Saha, DESPOTIC, Cloudy."""
from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import numpy as np

from ..base import BuildTask, PipelinePlotContext, PlotTask
from ..prep import config as _cfg
from ..spectrum_units import DSIGMA_DV_UNIT, dsigma_dv_ylabel


CPLUS_HIGH_MODEL_CUTOFF_K = 3000.0
CPLUS_HIGH_MODEL_LOS = ('y',)
CPLUS_HIGH_MODEL_FILENAME = 'Cplus_TQK_ge3000_DESPOTIC_Cloudy_Saha_Rinf.png'
CPLUS_HIGH_NO_SAHA_FILENAME = 'Cplus_TQK_ge3000_DESPOTIC_Cloudy_noSaha_Rinf.png'
CPLUS_LOW_MODEL_FILENAME = 'Cplus_TQK_lt3000_DESPOTIC_Saha_Rinf.png'

CPLUS_HIGH_MODEL_CONFIG = (
    {
        'name': 'CPLUS_SAHA_TQK_GE3000',
        'freq_field': 'C+_freq',
        'lum_field': 'C+_luminosity_saha',
        'width_field': 'C+_thermal_width',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'ge',
        'selection_cutoff_K': CPLUS_HIGH_MODEL_CUTOFF_K,
        'color': '#CC79A7',
        'label': 'legacy Saha + LTE',
    },
    {
        'name': 'CPLUS_DESPOTIC_TQK_GE3000',
        'freq_field': 'C+_freq',
        'lum_field': 'C+_luminosity_despotic',
        'width_field': 'C+_thermal_width',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'ge',
        'selection_cutoff_K': CPLUS_HIGH_MODEL_CUTOFF_K,
        'color': '#0072B2',
        'label': 'DESPOTIC',
    },
    {
        'name': 'CPLUS_CLOUDY_TQK_GE3000',
        'freq_field': 'C+_freq',
        'lum_field': 'C+_luminosity',
        'width_field': 'C+_thermal_width',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'ge',
        'selection_cutoff_K': CPLUS_HIGH_MODEL_CUTOFF_K,
        'color': '#D55E00',
        'label': 'Cloudy HM2012',
    },
)

CPLUS_COLD_SAHA_CONFIG = (
    {
        'name': 'CPLUS_SAHA_TQK_LT3000',
        'freq_field': 'C+_freq',
        'lum_field': 'C+_luminosity_saha_cold_diagnostic',
        'width_field': 'C+_thermal_width',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'lt',
        'selection_cutoff_K': CPLUS_HIGH_MODEL_CUTOFF_K,
        'color': '#CC79A7',
        'label': 'legacy Saha + LTE',
    },
)
CPLUS_HIGH_MODEL_PLOT_CONFIG = CPLUS_HIGH_MODEL_CONFIG


class Build_CplusHighModelComparison(BuildTask):
    """Build the three T_QUOKKA >= 3000 K model spectra."""

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

        for model in CPLUS_HIGH_MODEL_CONFIG:
            name = model['name']
            spectra[name] = {}
            store = SpectrumStore(provider, species_config=(model,))
            for los in CPLUS_HIGH_MODEL_LOS:
                velocity, intrinsic = store.get_spectrum(
                    name, los, R=float('inf'),
                )
                _, observed = store.get_spectrum(name, los, R=self.R)
                spectra[name][los] = {
                    'v_axis': velocity,
                    'dsigma_dv': intrinsic,
                    'dsigma_dv_obs': observed,
                }
            del store
            provider._cached_grid = None
            gc.collect()

        return {
            'spectra': spectra,
            'temperature_cutoff_K': CPLUS_HIGH_MODEL_CUTOFF_K,
            'R': self.R,
            'dsigma_dv_units': DSIGMA_DV_UNIT,
        }


class Build_CplusColdSahaComparison(BuildTask):
    """Build only the new T_QUOKKA < 3000 K Saha diagnostic spectrum."""

    def __init__(self, config, R: float | None = None):
        super().__init__(config)
        self.R = R if R is not None else _cfg.SPECTRAL_RESOLUTION_R
        self.spectrum_schema = 1

    def compute(self, context: PipelinePlotContext) -> dict:
        from ..services import SpectrumStore

        provider = context.provider
        model = CPLUS_COLD_SAHA_CONFIG[0]
        store = SpectrumStore(provider, species_config=CPLUS_COLD_SAHA_CONFIG)
        spectra: dict[str, dict[str, dict[str, np.ndarray]]] = {
            model['name']: {},
        }
        for los in CPLUS_HIGH_MODEL_LOS:
            velocity, intrinsic = store.get_spectrum(
                model['name'], los, R=float('inf'),
            )
            _, observed = store.get_spectrum(
                model['name'], los, R=self.R,
            )
            spectra[model['name']][los] = {
                'v_axis': velocity,
                'dsigma_dv': intrinsic,
                'dsigma_dv_obs': observed,
            }
        return {
            'spectra': spectra,
            'temperature_cutoff_K': CPLUS_HIGH_MODEL_CUTOFF_K,
            'R': self.R,
            'dsigma_dv_units': DSIGMA_DV_UNIT,
        }


class Plot_CplusHighModelComparison(PlotTask):
    """Plot requested hot three-model and cold two-model diagnostics."""

    def _gather_inputs(self, context: PipelinePlotContext) -> dict:
        return {
            'high_models': self._load_one(
                context, 'Build_CplusHighModelComparison',
            ),
            'cold_saha': self._load_one(
                context, 'Build_CplusColdSahaComparison',
            ),
            'temperature_split': self._load_one(
                context, 'Build_CplusTemperatureSpectrum',
            ),
        }

    def plot(self, context: PipelinePlotContext, inputs: dict) -> None:
        results = inputs['high_models']
        los = CPLUS_HIGH_MODEL_LOS[0]
        spectra = results['spectra']
        requested_R = _cfg.SPECTRAL_RESOLUTION_R
        spectrum_key = 'dsigma_dv' if np.isinf(requested_R) else 'dsigma_dv_obs'
        resolution_label = r'\infty' if np.isinf(requested_R) else f'{requested_R:.0e}'
        fig, axis = plt.subplots(1, 1, figsize=(7.5, 4.9))
        for model in CPLUS_HIGH_MODEL_PLOT_CONFIG:
            block = spectra[model['name']][los]
            velocity = np.asarray(block['v_axis'])
            values = np.asarray(block[spectrum_key])
            axis.plot(
                velocity, values, color=model['color'], lw=1.6,
                drawstyle='steps-mid',
                label=model['label'],
            )

        example = spectra[CPLUS_HIGH_MODEL_PLOT_CONFIG[0]['name']][los][spectrum_key]
        ylabel = dsigma_dv_ylabel(
            getattr(example, 'units', results['dsigma_dv_units'])
        )
        axis.axvline(0.0, color='0.55', ls=':', lw=0.8)
        axis.set_xlabel(r'Velocity [km s$^{-1}$]')
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25, ls='--', lw=0.5)
        axis.legend(fontsize=8.5, frameon=False)
        axis.ticklabel_format(
            style='sci', axis='y', scilimits=(0, 0), useMathText=True,
        )
        axis.set_title(
            r'[C II] 158 μm: $T_{\rm QUOKKA}\geq3000\,$K, '
            f'LOS {los}, R=${resolution_label}$',
            fontsize=12,
        )
        fig.tight_layout()

        output = context.config.output_dir / CPLUS_HIGH_MODEL_FILENAME
        fig.savefig(str(output), dpi=250, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {output}')

        fig, axis = plt.subplots(1, 1, figsize=(7.5, 4.9))
        for model in CPLUS_HIGH_MODEL_PLOT_CONFIG:
            if model['name'] == 'CPLUS_SAHA_TQK_GE3000':
                continue
            block = spectra[model['name']][los]
            axis.plot(
                np.asarray(block['v_axis']),
                np.asarray(block[spectrum_key]),
                color=model['color'],
                lw=1.6,
                drawstyle='steps-mid',
                label=model['label'],
            )
        axis.axvline(0.0, color='0.55', ls=':', lw=0.8)
        axis.set_xlabel(r'Velocity [km s$^{-1}$]')
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25, ls='--', lw=0.5)
        axis.legend(fontsize=8.5, frameon=False)
        axis.ticklabel_format(
            style='sci', axis='y', scilimits=(0, 0), useMathText=True,
        )
        axis.set_title(
            r'[C II] 158 μm: $T_{\rm QUOKKA}\geq3000\,$K, '
            f'LOS {los}, R=${resolution_label}$',
            fontsize=12,
        )
        fig.tight_layout()

        output = context.config.output_dir / CPLUS_HIGH_NO_SAHA_FILENAME
        fig.savefig(str(output), dpi=250, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {output}')

        fig, axis = plt.subplots(1, 1, figsize=(7.5, 4.9))
        cold_despotic = inputs['temperature_split']['spectra'][
            'CPLUS_DESPOTIC_TQK_LT3000'
        ][los]
        cold_saha = inputs['cold_saha']['spectra'][
            'CPLUS_SAHA_TQK_LT3000'
        ][los]
        for block, color, label in (
            (cold_despotic, '#0072B2', 'DESPOTIC'),
            (cold_saha, '#CC79A7', 'legacy Saha + LTE'),
        ):
            axis.plot(
                np.asarray(block['v_axis']),
                np.asarray(block[spectrum_key]),
                color=color,
                lw=1.6,
                drawstyle='steps-mid',
                label=label,
            )
        axis.axvline(0.0, color='0.55', ls=':', lw=0.8)
        axis.set_xlabel(r'Velocity [km s$^{-1}$]')
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25, ls='--', lw=0.5)
        axis.legend(fontsize=8.5, frameon=False)
        axis.ticklabel_format(
            style='sci', axis='y', scilimits=(0, 0), useMathText=True,
        )
        axis.set_title(
            r'[C II] 158 μm: $T_{\rm QUOKKA}<3000\,$K, '
            f'LOS {los}, R=${resolution_label}$',
            fontsize=12,
        )
        fig.tight_layout()

        output = context.config.output_dir / CPLUS_LOW_MODEL_FILENAME
        fig.savefig(str(output), dpi=250, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {output}')
