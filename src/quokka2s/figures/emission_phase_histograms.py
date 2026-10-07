"""Render the ten manuscript gas-distribution and emission phase panels."""
from __future__ import annotations

from quokka2s.figures.line_labels import LINE_TITLES


def _unit_latex(unit_str: str) -> str:
    """Format a saved physical-unit name for the colorbar's text label."""
    from unyt import Unit

    return Unit(unit_str).latex_repr or unit_str


def plot_panels(panels, display_panel_keys, png, pdf):
    """Draw saved log10 bins, color limits, axis bounds and panel descriptors.

    panels and display_panel_keys are read from phase_histograms.npz. The saved
    keys supply the order; no histogram, log transform or data selection occurs here.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(9.3, 15.5))
    grid = fig.add_gridspec(
        5,
        2,
        left=.085,
        right=.985,
        bottom=.04,
        top=.955,
        wspace=.32,
        hspace=.42,
    )
    for index, key in enumerate(display_panel_keys):
        inner = grid[index // 2, index % 2].subgridspec(
            2,
            1,
            height_ratios=[.055, 1],
            hspace=.07,
        )
        cax = fig.add_subplot(inner[0])
        ax = fig.add_subplot(inner[1])
        panel = panels[key]
        im = ax.pcolormesh(
            panel['x_edges'],
            panel['y_edges'],
            panel['log10_H'].T,
            cmap='viridis_r',
            norm=Normalize(*panel['color_limits_log10']),
            rasterized=True,
        )
        unit = _unit_latex(str(panel['weight_unit']))
        quantity = r'M_{\rm bin}' if str(panel['quantity']) == 'mass' else r'L_{\rm bin}'
        # This figure uses a short C II title; UV wavelengths use shared labels.
        line_title = 'C II' if key == 'cii' else LINE_TITLES.get(key, '')
        label = f'{line_title}  ' + rf'$\log_{{10}} {quantity}$ [${unit}$]'
        bar = fig.colorbar(
            im,
            cax=cax,
            orientation='horizontal',
        )
        bar.ax.tick_params(
            labelsize=8,
            top=True,
            labeltop=True,
            bottom=False,
            labelbottom=False,
        )
        cax.set_title(label.strip(), fontsize=10, pad=5)
        ax.set_xlim(*panel['x_limits_log10'])
        ax.set_ylim(*panel['y_limits_log10'])
        if key == 'NH_rho':
            ax.set_xlabel(r'$\log_{10} N_{\rm H}$ [cm$^{-2}$]')
            ax.set_ylabel(r'$\log_{10} \rho$ [g cm$^{-3}$]')
        else:
            temperature = str(panel['temperature_descriptor'])
            ax.set_xlabel(r'$\log_{10} \rho$ [g cm$^{-3}$]')
            ax.set_ylabel(rf'$\log_{{10}} T_{{\rm {temperature}}}$ [K]')
        ax.tick_params(labelsize=9)
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
