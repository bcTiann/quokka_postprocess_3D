"""Render the ten manuscript gas-distribution and emission phase panels."""
from __future__ import annotations

import numpy as np

from ..products.emission_phase_histograms import PANELS
from quokka2s.figures.line_labels import LINE_TITLES


# Keep the complete numerical bundle; select one transition per species for
# the manuscript figure. The first four panels describe the gas distribution.
DISPLAY_PANEL_KEYS = (
    'mass_T_QK', 'mass_T_DSP', 'mass_T_2R', 'NH_rho',
    'halpha', 'hi21', 'cii', 'ciii_977', 'civ_1548', 'co21',
)


# Show six decades below the largest populated bin in each colorbar group.
COLORBAR_DYNAMIC_RANGE = 1.0e6


def _unit_latex(unit_str: str) -> str:
    """A yt unit NAME (e.g. 'erg/s', 'g') → LaTeX for the colorbar label.

    The unit is rendered to LaTeX here, at plot time (display); the Build tasks
    store only the weight's natural unit STRING (data).  Falls back to the plain
    name if rendering fails or is empty (e.g. dimensionless)."""
    if not unit_str:
        return ''
    try:
        from unyt import Unit
        return Unit(unit_str).latex_repr or unit_str
    except Exception:
        return unit_str


def plot_panels(panels, png, pdf):
    """Ten selected panels, original absolute-bin coloring and mass scale."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    definitions = {panel[0]: panel for panel in PANELS}
    displayed = tuple(definitions[key] for key in DISPLAY_PANEL_KEYS)
    maxima = {}
    for key, _, group in displayed:
        maxima[group] = max(maxima.get(group, 0.0), float(panels[key]['H'].max()))
    norms = {group: Normalize(np.log10(value) - np.log10(COLORBAR_DYNAMIC_RANGE),
                              np.log10(value))
             for group, value in maxima.items() if value > 0}
    temp_panels = [panels[key] for key, _, _ in displayed if key != 'NH_rho']
    rho_lim = (min(p['x_edges'][0] for p in temp_panels),
               max(p['x_edges'][-1] for p in temp_panels))
    temp_lim = (min(p['y_edges'][0] for p in temp_panels),
                max(p['y_edges'][-1] for p in temp_panels))
    fig = plt.figure(figsize=(9.3, 15.5))
    grid = fig.add_gridspec(5, 2, left=.085, right=.985, bottom=.04,
                            top=.955, wspace=.32, hspace=.42)
    for index, (key, temperature, group) in enumerate(displayed):
        inner = grid[index // 2, index % 2].subgridspec(2, 1,
                      height_ratios=[.055, 1], hspace=.07)
        cax, ax = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        panel = panels[key]
        values = np.ma.masked_less_equal(panel['H'], 0)
        im = ax.pcolormesh(panel['x_edges'], panel['y_edges'],
                          np.ma.log10(values).T, cmap='viridis_r',
                          norm=norms.get(group, Normalize(0, 1)), rasterized=True)
        is_mass = key.startswith('mass') or key == 'NH_rho'
        unit = _unit_latex('g' if is_mass else 'erg/s')
        quantity = r'M_{\rm bin}' if is_mass else r'L_{\rm bin}'
        # This figure uses a short C II title; UV wavelengths use shared labels.
        line_title = 'C II' if key == 'cii' else LINE_TITLES.get(key, '')
        label = f'{line_title}  ' + rf'$\log_{{10}} {quantity}$ [${unit}$]'
        bar = fig.colorbar(im, cax=cax, orientation='horizontal')
        bar.ax.tick_params(labelsize=8, top=True, labeltop=True,
                           bottom=False, labelbottom=False)
        cax.set_title(label.strip(), fontsize=10, pad=5)
        if key == 'NH_rho':
            ax.set_xlabel(r'$\log_{10} N_{\rm H}$ [cm$^{-2}$]')
            ax.set_ylabel(r'$\log_{10} \rho$ [g cm$^{-3}$]')
        else:
            ax.set_xlim(*rho_lim)
            ax.set_ylim(*temp_lim)
            ax.set_xlabel(r'$\log_{10} \rho$ [g cm$^{-3}$]')
            ax.set_ylabel(rf'$\log_{{10}} T_{{\rm {temperature}}}$ [K]')
        ax.tick_params(labelsize=9)
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
