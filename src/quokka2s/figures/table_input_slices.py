"""Render the five-field x slice used for the manuscript's Figure 1.

The caller supplies already calculated, masked 2D arrays. This module only
sets display ranges and draws them; it does not load snapshots or tables.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


# Each entry is (array key, colorbar label, colormap, log minimum, log maximum,
# temperature tick group). Arrays are in CGS before the logarithm is taken.
TABLE_INPUT_PANELS = (
    (
        'nH_slice',
        r'$\log_{10}\,n_{\rm H}\ [\rm cm^{-3}]$',
        'inferno', None, None, None,
    ),
    (
        'NH_slice',
        r'$\log_{10}\,N_{\rm H}\ [\rm cm^{-2}]$',
        'cividis', None, None, None,
    ),
    (
        'dVdr_slice',
        r'$\log_{10}\,\frac{dV}{dr}$ [s$^{-1}$]',
        'plasma', None, None, None,
    ),
    (
        'T_qk_slice',
        r'$\log_{10}\,T_{\rm QUOKKA}\ [\rm K]$',
        'turbo', 2.0, 8.0, 'T',
    ),
    (
        'T_dsp_slice',
        r'$\log_{10}\,T_{\rm DESPOTIC}\ [\rm K]$',
        'turbo', 2.0, 8.0, 'T',
    ),
)
TABLE_INPUT_PANEL_KEYS = tuple(panel[0] for panel in TABLE_INPUT_PANELS)


def prepare_slice_panels(slices):
    """Return log-valued images and their positive-data ranges.

    slices : dict[str, ndarray], each shape (ny, nz)
        The five arrays named in TABLE_INPUT_PANEL_KEYS, in CGS or kelvin.
    Returns : dict[str, dict | None]
        Images have shape (nz, ny), placing z vertically. Nonpositive values
        are displayed as NaN; a panel without positive values is None.
    """
    panel_state = {}
    for key in TABLE_INPUT_PANEL_KEYS:
        data = np.asarray(slices[key]).T * 1.0
        positive = data > 0
        if not positive.any():
            panel_state[key] = None
            continue
        log_data = np.where(
            positive,
            np.log10(np.where(positive, data, 1.0)),
            np.nan,
        )
        panel_state[key] = {
            'log_data': log_data,
            'p_lo': float(np.nanmin(log_data)),
            'p_hi': float(np.nanmax(log_data)),
        }
    return panel_state


def draw_slice_panel(figure, axis, panel, state, extent_kpc):
    """Draw one log-valued slice and its colorbar above the image."""
    key, label, cmap, log_minimum, log_maximum, group = panel
    if state is None:
        axis.set_title(f'{label}\n(empty)', fontsize=9)
        return
    if log_minimum is None:
        log_minimum = state['p_lo']
    if log_maximum is None:
        log_maximum = state['p_hi']

    image = axis.imshow(
        state['log_data'],
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
    if group == 'T' and log_minimum is not None and log_maximum is not None:
        low_tick = int(math.ceil(log_minimum))
        high_tick = int(math.floor(log_maximum))
        tick_step = 2 if high_tick - low_tick > 4 else 1
        colorbar.set_ticks(list(range(low_tick, high_tick + 1, tick_step)))
    colorbar_axis.set_title(label, fontsize=8, pad=2)


def plot_table_input_slice(
    slices,
    extent_kpc,
    output_path,
    *,
    slice_index,
    dataset_name='',
    show_title=False,
    save_pdf=True,
):
    """Save Figure 1 from five already calculated x-slice arrays.

    slices : dict[str, ndarray], each shape (ny, nz)
        nH_slice [cm^-3], NH_slice [cm^-2], dVdr_slice [s^-1],
        T_qk_slice and T_dsp_slice [K]. Mask excluded cells with NaN first.
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
    panel_state = prepare_slice_panels(slices)
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
            state=panel_state[panel[0]],
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
