"""Draw edge-on and face-on gas maps from saved numerical arrays."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_gas_projection_maps(payload, report, stem):
    """Save the established two-view four-field PNG/PDF figure.

    payload : dict[str, ndarray]
        MultiviewAccumulator.payload() plus extents and edge-view particles.
        Native arrays are (y,z) for edge view and (x,y) for face view.
    report : dict
        X_H and hydrogen_mass_g convert projected gas mass to N_H [cm^-2].
    stem : str or Path
        Output path without extension; writes .png and .pdf.

    Returns : dict
        Display selections and each column's shared normalization range.
    """
    stem = Path(stem)
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    from matplotlib.ticker import LogLocator, LogFormatterMathtext, MaxNLocator

    factor = report['X_H']/report['hydrogen_mass_g']
    columns = [
        ('rho_g_cm3', r'Density slice', r'$\rho\;[\mathrm{g\,cm^{-3}}]$', 1., 'viridis'),
        ('sigma_g_cm2', r'Hydrogen column', r'$N_{\rm H}^{\rm LOS}\;[\mathrm{cm^{-2}}]$', factor, 'viridis'),
        ('vz_kms', r'Vertical velocity', r'$\langle v_z\rangle_\rho\;[\mathrm{km\,s^{-1}}]$', 1., 'RdBu_r'),
        ('T_mixed_K', r'Temperature', r'$\langle T\rangle_\rho\;[\mathrm{K}]$', 1., 'viridis'),
    ]
    # Equal physical aspect in both views, including the entire 8-kpc height.
    # Keep one shared scale per column and native pixels (no display smoothing).
    fig = plt.figure(figsize=(8.25, 16.4))
    left, right, top = .075, .975, .97
    gap = .062
    panel_width = (right-left-3*gap)/4
    panel_inches = panel_width*fig.get_figwidth()
    edge_height = panel_inches*8/fig.get_figheight()
    face_height = panel_inches/fig.get_figheight()
    edge_bottom = top-edge_height
    face_top = edge_bottom-.037
    face_bottom = face_top-face_height
    norms = {}
    for col, (key, title, label, conversion, cmap) in enumerate(columns):
        values = [payload[f'valid_{view}_{key}']*conversion for view in ('edge', 'face')]
        finite = np.concatenate([arr[np.isfinite(arr)] for arr in values])
        if key == 'vz_kms':
            bound = float(np.max(np.abs(finite)))
            norm = TwoSlopeNorm(vmin=-bound, vcenter=0., vmax=bound)
        else:
            positive = finite[finite > 0]
            norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
        norms[key] = dict(minimum=norm.vmin, maximum=norm.vmax, clipped=False)
        x = left+col*(panel_width+gap)
        for row, (view, arr, y, h) in enumerate(zip(
                ('edge', 'face'), values, (edge_bottom, face_bottom), (edge_height, face_height))):
            ax = fig.add_axes([x, y, panel_width, h])
            extent = payload[f'extent_{view}_kpc']
            im = ax.imshow(arr.T, origin='lower', extent=extent, aspect='equal',
                           interpolation='nearest', cmap=cmap, norm=norm, rasterized=True)
            ax.tick_params(labelsize=8, direction='out', length=3, pad=2)
            ax.set_xlabel(r'$y\;[\mathrm{kpc}]$' if row == 0 else r'$x\;[\mathrm{kpc}]$',
                          fontsize=9, labelpad=3)
            ax.set_xticks([0., .5])
            if row == 0:
                ax.set_yticks(np.arange(-3., 4.))
                ax.set_title(title, fontsize=10, pad=8)
                ax.scatter(payload['particles_edge_y_kpc'], payload['particles_edge_z_kpc'],
                           s=.48, c='#ec2424', marker='o', linewidths=0, alpha=.75,
                           rasterized=True)
            else:
                ax.set_yticks([0., .5])
            if col == 0:
                ax.set_ylabel(r'$z\;[\mathrm{kpc}]$' if row == 0 else r'$y\;[\mathrm{kpc}]$',
                              fontsize=9, labelpad=3)
            else:
                ax.tick_params(labelleft=False)
        cax = fig.add_axes([x-.006, face_bottom-.064, panel_width+.012, .008])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.ax.tick_params(labelsize=7.5, length=3, pad=2)
        if isinstance(norm, LogNorm):
            cb.locator = LogLocator(base=10, numticks=4)
            cb.formatter = LogFormatterMathtext()
            cb.update_ticks()
            cb.ax.minorticks_off()
        else:
            cb.locator = MaxNLocator(nbins=3, symmetric=True)
            cb.update_ticks()
        cb.set_label(label, fontsize=9, labelpad=4)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.pdf', '.png'):
        fig.savefig(stem.with_suffix(ext), dpi=300, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)
    return {
        'selection': 'valid',
        'temperature': 'mixed',
        'column': 'LOS integral of nH',
        'normalization': norms,
        'figure_size_inches': [8.25, 16.4],
        'spatial_aspect': 'equal',
    }
