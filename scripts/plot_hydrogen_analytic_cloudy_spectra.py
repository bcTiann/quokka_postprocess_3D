#!/usr/bin/env python3
"""Compare cached Cloudy and analytic Halpha/HI spectra in both regimes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_overleaf_adopted_losz_spectra import DEFAULT_BUNDLE, LINE_TITLES
from quokka2s.pipeline.spectrum_units import dsigma_dv_ylabel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--output-root', type=Path,
                        default=Path('output/2026-09-07_hydrogen_comparison'))
    args = parser.parse_args()
    metadata = json.loads(args.bundle.with_suffix('.json').read_text())
    if metadata['los'] != 'z' or not metadata['completed_full_domain']:
        raise ValueError('Need a completed LOS-z run')
    with np.load(args.bundle, allow_pickle=False) as source:
        if str(source['los'].item()) != 'z' or not source['completed_full_domain'].item():
            raise ValueError('Invalid spectrum bundle')
        velocity = source['velocity_kms']
        keys = list(source['line_keys'])
        # This producer explicitly stores reference order CII, Halpha, HI.
        if keys[:3] != ['cii', 'halpha', 'hi21']:
            raise ValueError('Unexpected reference line order')
        cloudy, analytic = source['dsigma_dv'], source['reference_dsigma_dv']
        unit = str(source['dsigma_dv_units'].item())
        area = float(source['projected_area_cm2'].item())
        dv = float(np.diff(velocity)[0])
        np.testing.assert_allclose(np.diff(velocity), dv)
        for values, total in ((cloudy, source['captured_luminosity_erg_s']),
                              (analytic, source['reference_captured_luminosity_erg_s'])):
            np.testing.assert_allclose(values.sum(axis=-1) * dv * area, total, rtol=1e-12)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('Invalid spectrum values')
    for folder in ('png', 'pdf', 'metadata'):
        (args.output_root / folder).mkdir(parents=True, exist_ok=True)
    outputs = {}
    for key in ('halpha', 'hi21'):
        index = keys.index(key)
        stem = f'{key}_analytic_cloudy_both_regimes_Rinf_LOSz'
        paths = {fmt: args.output_root / fmt / f'{stem}.{fmt}' for fmt in ('png', 'pdf')}
        if any(p.exists() for p in paths.values()):
            raise FileExistsError(f'Refusing to overwrite {stem}')
        ymax = float(max(cloudy[index].max(), analytic[index].max())) * 1.05
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
        for branch, ax in enumerate(axes):
            ax.plot(velocity, cloudy[index, branch], color='#D55E00',
                    linewidth=1.9, drawstyle='steps-mid', label='Cloudy', zorder=2)
            label = ('Analytic (DESPOTIC densities)' if branch == 0
                     else r'Analytic (QUOKKA $\mu$)')
            ax.plot(velocity, analytic[index, branch], color='#0072B2', linestyle='--',
                    linewidth=1.9, drawstyle='steps-mid', label=label, zorder=3)
            ax.axvline(0, color='.55', linestyle=':', linewidth=.8, zorder=1)
            ax.set_xlabel(r'Velocity [km s$^{-1}$]')
            ax.set_ylabel(dsigma_dv_ylabel(unit))
            ax.set_title(r'$T_{\rm QUOKKA}<3000\,$K' if branch == 0
                         else r'$T_{\rm QUOKKA}\geq3000\,$K')
            ax.set_ylim(0, ymax if ymax > 0 else 1)
            ax.set_xlim(velocity[0] - dv / 2, velocity[-1] + dv / 2)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
            ax.tick_params(axis='y', labelleft=True)
            ax.grid(True, alpha=.25, linestyle='--', linewidth=.5)
            ax.legend(frameon=False, fontsize=9)
        fig.suptitle(LINE_TITLES[key] + r', LOS z, $R=\infty$')
        fig.tight_layout()
        fig.savefig(paths['png'], dpi=250, bbox_inches='tight')
        fig.savefig(paths['pdf'], bbox_inches='tight')
        plt.close(fig)
        outputs[key] = {fmt: str(path) for fmt, path in paths.items()}
    manifest = dict(source_bundle=str(args.bundle.resolve()),
                    cloudy_table=metadata['cloudy_table'],
                    radiation=metadata['radiation'],
                    column_definition=metadata['column_density_definition'],
                    low='T_DESPOTIC for both models; analytic uses DESPOTIC densities',
                    high='T_QUOKKA for both models; analytic uses QUOKKA mu-derived densities',
                    analytic_halpha='Case-B recombination only, Draine equation 14.8',
                    analytic_hi='Optically thin, upper-level fraction 3/4',
                    unit=unit, recomputed=False, outputs=outputs)
    path = args.output_root / 'metadata/hydrogen_analytic_cloudy_both_regimes.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(outputs, indent=2))


if __name__ == '__main__':
    main()
