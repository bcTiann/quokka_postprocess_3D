"""Render the five-field x slice used for the manuscript's Figure 1.

The caller supplies saved log-valued panels and their numerical display ranges.
This module draws them; it does not load snapshots, query tables or mask fields.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Each entry is (saved panel key, colorbar label, colormap).
TABLE_INPUT_PANELS = (
    (
        'nH_slice',
        r'$\log_{10}\,n_{\rm H}\ [\rm cm^{-3}]$',
        'inferno',
    ),
    (
        'NH_slice',
        r'$\log_{10}\,N_{\rm H}\ [\rm cm^{-2}]$',
        'cividis',
    ),
    (
        'dVdr_slice',
        r'$\log_{10}\,\frac{dV}{dr}$ [s$^{-1}$]',
        'plasma',
    ),
    (
        'T_qk_slice',
        r'$\log_{10}\,T_{\rm QUOKKA}\ [\rm K]$',
        'turbo',
    ),
    (
        'T_dsp_slice',
        r'$\log_{10}\,T_{\rm DESPOTIC}\ [\rm K]$',
        'turbo',
    ),
)


def draw_slice_panel(figure, axis, panel, payload, extent_kpc):
    """Draw one log-valued slice and its colorbar above the image."""
    key, label, cmap = panel
    if payload[key + '_is_empty']:
        axis.set_title(f'{label}\n(empty)', fontsize=9)
        return
    log_minimum, log_maximum = payload[key + '_log_limits']

    image = axis.imshow(
        payload[key + '_log10'].T,
        origin='lower',
        extent=extent_kpc,
        aspect='equal',
        cmap=cmap,
        norm=Normalize(vmin=log_minimum, vmax=log_maximum),
    )
    axis.tick_params(axis='both', labelsize=8)
    divider = make_axes_locatable(axis)
    colorbar_axis = divider.append_axes('top', size='2.5%', pad=0.2)
    colorbar = figure.colorbar(
        image,
        cax=colorbar_axis,
        orientation='horizontal',
    )
    colorbar.ax.tick_params(
        labelsize=7,
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
    )
    colorbar_ticks = payload[key + '_colorbar_ticks']
    if colorbar_ticks.size:
        colorbar.set_ticks(colorbar_ticks)
    colorbar_axis.set_title(label, fontsize=8, pad=2)


def plot_table_input_slice(
    payload,
    extent_kpc,
    output_path,
    *,
    slice_index,
    dataset_name='',
    show_title=False,
    save_pdf=True,
):
    """Save Figure 1 from the numerical panels stored in slice_data.npz.

    payload : dict[str, ndarray]
        Saved output from products.table_input_slices.prepare_slice_plot_data().
        {key}_log10 has shape (Ny, Nz), with masks and logarithms already
        applied. Limits, temperature ticks and empty-panel flags are saved too.
    extent_kpc : sequence of four floats
        (y_min, y_max, z_min, z_max), for example (0, 1, -4, 4).
    output_path : str or Path
        PNG destination. With save_pdf=True, also write the matching PDF.
    slice_index : int
        Original x-cell index, for example 216. Used only in the title.
    dataset_name : str
        Snapshot name used in the optional title, for example 'plt0655228'.

    Returns None. The plot retains the original five-panel layout and scales.
    """
    figure, axes = plt.subplots(
        1,
        len(TABLE_INPUT_PANELS),
        figsize=(2.8 * len(TABLE_INPUT_PANELS), 18),
        sharey=True,
        gridspec_kw={'wspace': 0.05, 'top': 0.93, 'bottom': 0.04},
    )
    for axis, panel in zip(axes, TABLE_INPUT_PANELS):
        draw_slice_panel(
            figure=figure,
            axis=axis,
            panel=panel,
            payload=payload,
            extent_kpc=extent_kpc,
        )
    axes[0].set_ylabel('z [kpc]', fontsize=10)
    for axis in axes:
        axis.set_xlabel('y [kpc]', fontsize=9)
    if show_title:
        figure.suptitle(
            f'{Path(dataset_name).name}   (down=1,  $L_{{\\rm ext}}$ = 0 kpc)\n'
            f'y–z slice at x = index {slice_index}',
            fontsize=13,
            y=0.99,
        )

    output_path = Path(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches='tight')
    if save_pdf:
        figure.savefig(output_path.with_suffix('.pdf'), dpi=200, bbox_inches='tight')
    plt.close(figure)
    print(f'  Saved: {output_path}')
