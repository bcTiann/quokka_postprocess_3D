"""Two-panel [C II] DESPOTIC/Cloudy comparison with a shared y scale."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..base import PipelinePlotContext, PlotTask
from ..intermediate_io import _glob_one_taskcache, _load_results
from ..spectrum_units import dsigma_dv_ylabel
from .cplus_high_model_comparison import CPLUS_HIGH_MODEL_LOS
from .cplus_low_cloudy_comparison import MODELS as LOW_MODELS


FILENAME = 'Cplus_DESPOTIC_vs_Cloudy_Tsplit_shared_ylim_Rinf.png'


class Plot_CplusCloudyCombinedComparison(PlotTask):
    """Plot low/high DESPOTIC and Cloudy spectra in the same format."""

    def _gather_inputs(self, context: PipelinePlotContext) -> dict:
        # The completed low-T result predates a change to the unrelated
        # production high-T CII table, whose mtime is part of the global cache
        # identity.  Its own low-T table, dataset, and task definition did not
        # change, so reuse that spectrum without forcing a 3D recomputation.
        low_path = _glob_one_taskcache(
            context.config.output_dir, 'Build_CplusLowCloudyComparison',
        )
        if low_path is None:
            raise RuntimeError('Build_CplusLowCloudyComparison result not found')
        return {
            'low': _load_results(low_path),
            'high': self._load_one(context, 'Build_CplusHighModelComparison'),
        }

    def plot(self, context: PipelinePlotContext, inputs: dict) -> None:
        low = inputs['low']
        high = inputs['high']
        los = CPLUS_HIGH_MODEL_LOS[0]
        high_models = (
            {
                'name': 'CPLUS_DESPOTIC_TQK_GE3000',
                'color': '#0072B2',
                'label': 'DESPOTIC',
            },
            {
                'name': 'CPLUS_CLOUDY_TQK_GE3000',
                'color': '#D55E00',
                'label': 'Cloudy HM2012',
            },
        )

        all_values = []
        for model in LOW_MODELS:
            all_values.append(np.asarray(
                low['spectra'][model['name']]['dsigma_dv'], dtype=float,
            ))
        for model in high_models:
            all_values.append(np.asarray(
                high['spectra'][model['name']][los]['dsigma_dv'], dtype=float,
            ))
        finite_maxima = [
            float(values[np.isfinite(values)].max())
            for values in all_values if np.isfinite(values).any()
        ]
        shared_max = max(finite_maxima, default=0.0)
        if shared_max <= 0.0:
            raise ValueError('[C II] comparison spectra contain no positive values')

        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=False)
        panels = (
            (
                axes[0],
                [
                    (
                        low['spectra'][model['name']],
                        model['color'],
                        model['label'],
                    )
                    for model in LOW_MODELS
                ],
                r'$T_{\rm QUOKKA}<3000\,$K',
            ),
            (
                axes[1],
                [
                    (
                        high['spectra'][model['name']][los],
                        model['color'],
                        model['label'],
                    )
                    for model in high_models
                ],
                r'$T_{\rm QUOKKA}\geq3000\,$K',
            ),
        )

        for axis, curves, title in panels:
            for block, color, label in curves:
                axis.plot(
                    np.asarray(block['v_axis']),
                    np.asarray(block['dsigma_dv']),
                    color=color,
                    lw=1.6,
                    drawstyle='steps-mid',
                    label=label,
                )
            example = curves[0][0]['dsigma_dv']
            axis.axvline(0.0, color='0.55', ls=':', lw=0.8)
            axis.set_xlabel(r'Velocity [km s$^{-1}$]')
            axis.set_ylabel(dsigma_dv_ylabel(
                getattr(example, 'units', low['dsigma_dv_units'])
            ))
            axis.set_title(title)
            axis.set_ylim(0.0, 1.05 * shared_max)
            axis.ticklabel_format(
                style='sci', axis='y', scilimits=(0, 0), useMathText=True,
            )
            axis.grid(True, alpha=0.25, ls='--', lw=0.5)
            axis.legend(fontsize=9, frameon=False)

        fig.suptitle(
            r'Comparison of the [C II] spectrum from DESPOTIC and Cloudy'
            r', LOS y, $R=\infty$'
        )
        fig.tight_layout()
        output = context.config.output_dir / FILENAME
        fig.savefig(str(output), dpi=250, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {output}')
