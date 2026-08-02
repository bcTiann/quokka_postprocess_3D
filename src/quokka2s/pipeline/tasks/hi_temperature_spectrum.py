"""Absolute H I 21-cm spectra for selected sets and their complements.

The three curves isolate the neutral-hydrogen prescription and temperature
choice:

1. QUOKKA mu-derived n_HI, T_QUOKKA >= 3000 K;
2. DESPOTIC-table n_HI, T_DESPOTIC < 3000 K;
3. DESPOTIC-table n_HI, T_QUOKKA < 3000 K.
All spectra are intrinsic (R=infinity) and plotted in absolute, unnormalised
surface spectral-luminosity units.
"""
from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import numpy as np

from ..base import BuildTask, PipelinePlotContext, PlotTask
from ..spectrum_units import DSIGMA_DV_UNIT, dsigma_dv_ylabel


HI_TEMPERATURE_CUTOFF_K = 3000.0
HI_TEMPERATURE_SPECTRUM_LOS = ('y',)
HI_TEMPERATURE_SPECTRUM_FILENAME = 'HI_temperature_selected_spectrum_Rinf.png'
HI_TEMPERATURE_COMPLEMENT_FILENAME = 'HI_temperature_complement_spectrum_Rinf.png'


HI_TEMPERATURE_SPECTRUM_CONFIG = (
    {
        'name': 'HI_QUOKKA_TQK_GE3000',
        'freq_field': 'HI_freq',
        'lum_field': 'HI_luminosity_quokka',
        'width_field': 'HI_thermal_width_quokka',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'ge',
        'selection_cutoff_K': HI_TEMPERATURE_CUTOFF_K,
        'color': '#D55E00',
        'label': (
            r'QUOKKA $n_{\rm HI}$, '
            r'$T_{\rm QUOKKA}\geq3000\,\mathrm{K}$'
        ),
    },
    {
        'name': 'HI_DESPOTIC_TDSP_LT3000',
        'freq_field': 'HI_freq',
        'lum_field': 'HI_luminosity_despotic',
        'width_field': 'HI_thermal_width_despotic',
        'selection_temperature_field': 'temperature_despotic',
        'selection_operator': 'lt',
        'selection_cutoff_K': HI_TEMPERATURE_CUTOFF_K,
        'color': '#0072B2',
        'label': (
            r'DESPOTIC $n_{\rm HI}$, '
            r'$T_{\rm DESPOTIC}<3000\,\mathrm{K}$'
        ),
    },
    {
        'name': 'HI_DESPOTIC_TQK_LT3000',
        'freq_field': 'HI_freq',
        'lum_field': 'HI_luminosity_despotic',
        'width_field': 'HI_thermal_width_quokka',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'lt',
        'selection_cutoff_K': HI_TEMPERATURE_CUTOFF_K,
        'color': '#009E73',
        'label': (
            r'DESPOTIC $n_{\rm HI}$, '
            r'$T_{\rm QUOKKA}<3000\,\mathrm{K}$'
        ),
    },
)


# Exact set complements of HI_TEMPERATURE_SPECTRUM_CONFIG.  Each pair is
# disjoint and its union contains all cells.
HI_TEMPERATURE_COMPLEMENT_CONFIG = (
    {
        'name': 'HI_QUOKKA_TQK_LT3000',
        'freq_field': 'HI_freq',
        'lum_field': 'HI_luminosity_quokka',
        'width_field': 'HI_thermal_width_quokka',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'lt',
        'selection_cutoff_K': HI_TEMPERATURE_CUTOFF_K,
        'color': '#D55E00',
        'label': (
            r'QUOKKA $n_{\rm HI}$, '
            r'$T_{\rm QUOKKA}<3000\,\mathrm{K}$'
        ),
    },
    {
        'name': 'HI_DESPOTIC_TDSP_GE3000',
        'freq_field': 'HI_freq',
        'lum_field': 'HI_luminosity_despotic',
        'width_field': 'HI_thermal_width_despotic',
        'selection_temperature_field': 'temperature_despotic',
        'selection_operator': 'ge',
        'selection_cutoff_K': HI_TEMPERATURE_CUTOFF_K,
        'color': '#0072B2',
        'label': (
            r'DESPOTIC $n_{\rm HI}$, '
            r'$T_{\rm DESPOTIC}\geq3000\,\mathrm{K}$'
        ),
    },
    {
        'name': 'HI_DESPOTIC_TQK_GE3000',
        'freq_field': 'HI_freq',
        'lum_field': 'HI_luminosity_despotic',
        'width_field': 'HI_thermal_width_quokka',
        'selection_temperature_field': 'temperature_quokka',
        'selection_operator': 'ge',
        'selection_cutoff_K': HI_TEMPERATURE_CUTOFF_K,
        'color': '#009E73',
        'label': (
            r'DESPOTIC $n_{\rm HI}$, '
            r'$T_{\rm QUOKKA}\geq3000\,\mathrm{K}$'
        ),
    },
)


def shared_hi_spectrum_ylim(
    species_results: dict,
    selected_results: dict,
    *,
    headroom: float = 1.05,
) -> tuple[float, float]:
    """Return one absolute y range shared by the H I spectrum figures.

    The maximum is taken over LOS y, the two all-temperature H I curves,
    and all three temperature-selected curves. Consequently no figure
    can hide an order-of-magnitude difference through automatic axis scaling.
    """
    if headroom <= 1.0:
        raise ValueError('headroom must be greater than 1')

    full_spectra = species_results['spectra']
    selected_spectra = selected_results['spectra']
    maxima: list[float] = []

    for species in ('HI_DESPOTIC', 'HI_QUOKKA'):
        for los in HI_TEMPERATURE_SPECTRUM_LOS:
            values = np.asarray(
                full_spectra[species][los]['total']['dsigma_dv'],
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            if finite.size:
                maxima.append(float(finite.max()))

    for case in HI_TEMPERATURE_SPECTRUM_CONFIG:
        for los in HI_TEMPERATURE_SPECTRUM_LOS:
            values = np.asarray(
                selected_spectra[case['name']][los]['dsigma_dv'],
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            if finite.size:
                maxima.append(float(finite.max()))

    global_max = max(maxima, default=0.0)
    if global_max <= 0.0:
        raise ValueError('H I spectra contain no positive finite values')
    return 0.0, headroom * global_max


def _build_temperature_spectra(
    context: PipelinePlotContext,
    cases: tuple[dict, ...],
) -> dict:
    """Build one configured set of temperature-selected H I spectra."""
    # Local import avoids the package-initialisation cycle created by
    # spectrum_service importing the canonical species configuration from
    # tasks.integrated_spectrum.
    from ..services import SpectrumStore

    provider = context.provider
    provider._cached_grid = None
    gc.collect()

    spectra: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for case in cases:
        name = case['name']
        spectra[name] = {}

        # One case per store keeps peak memory bounded. The common expensive
        # luminosity fields are served by the existing field cache; only the
        # inexpensive temperature mask is task-local.
        store = SpectrumStore(provider, species_config=(case,))
        for los in HI_TEMPERATURE_SPECTRUM_LOS:
            v_axis, dsigma_dv = store.get_spectrum(
                name,
                los,
                R=float('inf'),
            )
            spectra[name][los] = {
                'v_axis': v_axis,
                'dsigma_dv': dsigma_dv,
            }
            print(
                f'  [{name}] los={los} intrinsic peak='
                f'{float(dsigma_dv.max()):.3e} {DSIGMA_DV_UNIT}'
            )

        del store
        provider._cached_grid = None
        gc.collect()

    return {
        'spectra': spectra,
        'temperature_cutoff_K': HI_TEMPERATURE_CUTOFF_K,
        'spectral_resolution_R': float('inf'),
        'dsigma_dv_units': DSIGMA_DV_UNIT,
    }


class Build_HITemperatureSpectrum(BuildTask):
    """Compute the three temperature-selected intrinsic H I spectra."""

    def __init__(self, config):
        super().__init__(config)
        # v2 assigns T_QUOKKA == 3000 K to the hot analytic branch, matching
        # the canonical two-regime emissivity without leaving a boundary gap.
        self.spectrum_schema = 3

    def compute(self, context: PipelinePlotContext) -> dict:
        return _build_temperature_spectra(
            context, HI_TEMPERATURE_SPECTRUM_CONFIG,
        )


class Build_HITemperatureComplementSpectrum(BuildTask):
    """Compute exact temperature-set complements of the selected spectra."""

    def __init__(self, config):
        super().__init__(config)
        self.spectrum_schema = 3

    def compute(self, context: PipelinePlotContext) -> dict:
        return _build_temperature_spectra(
            context, HI_TEMPERATURE_COMPLEMENT_CONFIG,
        )


class Plot_HITemperatureSpectrum(PlotTask):
    """Overlay the three absolute H I spectra for LOS y."""

    build_task_name = 'Build_HITemperatureSpectrum'
    cases = HI_TEMPERATURE_SPECTRUM_CONFIG
    default_filename = HI_TEMPERATURE_SPECTRUM_FILENAME
    figure_title = (
        r'$\mathrm{H\,I}$ 21-cm temperature-selected spectra '
        r'$(R=\infty)$ — absolute, not normalised'
    )

    def __init__(
        self,
        config,
        filename: str | None = None,
    ):
        super().__init__(config)
        self.filename = filename or self.default_filename

    def _gather_inputs(self, context: PipelinePlotContext) -> dict:
        result = self._load_one(context, self.build_task_name)
        if self.build_task_name == 'Build_HITemperatureSpectrum':
            selected = result
        else:
            selected = self._load_one(
                context, 'Build_HITemperatureSpectrum',
            )
        return {
            'result': result,
            'selected': selected,
            'full': self._load_one(context, 'Build_SpeciesSpectrum'),
        }

    def plot(self, context: PipelinePlotContext, inputs: dict) -> None:
        spectra = inputs['result']['spectra']
        ylim = shared_hi_spectrum_ylim(inputs['full'], inputs['selected'])
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))

        for los in HI_TEMPERATURE_SPECTRUM_LOS:
            for case in self.cases:
                block = spectra[case['name']][los]
                ax.plot(
                    np.asarray(block['v_axis']),
                    np.asarray(block['dsigma_dv']),
                    color=case['color'],
                    lw=1.6,
                    drawstyle='steps-mid',
                    label=case['label'],
                )

            ax.axvline(0.0, color='0.55', ls=':', lw=0.8)
            ax.set_xlabel(r'Velocity [km s$^{-1}$]')
            plotted_spectrum = spectra[self.cases[0]['name']][los]['dsigma_dv']
            ax.set_ylabel(dsigma_dv_ylabel(
                getattr(
                    plotted_spectrum,
                    'units',
                    inputs['result']['dsigma_dv_units'],
                )
            ))
            ax.set_title(f'LOS {los}')
            ax.set_ylim(*ylim)
            ax.ticklabel_format(
                style='sci', axis='y', scilimits=(0, 0), useMathText=True,
            )
            ax.grid(True, alpha=0.25, ls='--', lw=0.5)
            ax.legend(fontsize=8.5, frameon=False)

        fig.suptitle(self.figure_title, fontsize=12)
        fig.tight_layout()
        out = context.config.output_dir / self.filename
        fig.savefig(str(out), dpi=250, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')


class Plot_HITemperatureComplementSpectrum(Plot_HITemperatureSpectrum):
    """Plot the exact complements using the same y range as selected H I."""

    build_task_name = 'Build_HITemperatureComplementSpectrum'
    cases = HI_TEMPERATURE_COMPLEMENT_CONFIG
    default_filename = HI_TEMPERATURE_COMPLEMENT_FILENAME
    figure_title = (
        r'$\mathrm{H\,I}$ 21-cm complementary temperature spectra '
        r'$(R=\infty)$ — absolute, not normalised'
    )
