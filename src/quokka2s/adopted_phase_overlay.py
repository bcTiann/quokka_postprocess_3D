"""Compare accepted line profiles with gas mass velocity distributions."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .adopted_spectral_products import LINE_TITLES, REGIME_KEYS


LINE_ORDER = (
    'halpha', 'hi21', 'cii', 'ciii_977', 'ciii_1907', 'ciii_1909',
    'civ_1548', 'civ_1551', 'co10', 'co21',
)
PHASE_ORDER = ('CNM', 'UNM', 'WNM', 'WIM', 'HIM')
PHASE_COLORS = ('#0072B2', '#56B4E9', '#009E73', '#E69F00', '#D55E00')
# Display-only zoom: retain the full saved profiles for normalization and moments.
DISPLAY_VELOCITY_LIMITS_KMS = {
    'cii': (-100., 100.), 'co10': (-100., 100.), 'co21': (-100., 100.),
}


def accepted_display_profiles(payload):
    """Select exactly the curves displayed in the current adopted spectra.

    Halpha/HI/CII use total; CIII/CIV hot; CO cold. Saved source arrays are
    copied, never modified. Moments describe the saved velocity window only.
    """
    keys = tuple(str(key) for key in payload['line_keys'])
    if len(keys) != len(set(keys)) or set(keys) != set(LINE_ORDER):
        raise ValueError('Expected the ten accepted individual transitions')
    if tuple(str(key) for key in payload['regime_keys']) != REGIME_KEYS:
        raise ValueError('Unknown spectrum temperature regimes')
    edges = np.asarray(payload['velocity_edges_kms'], dtype=float)
    velocity = np.asarray(payload['velocity_kms'], dtype=float)
    widths = np.diff(edges)
    if (edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all()
            or np.any(widths <= 0) or velocity.shape != widths.shape
            or not np.array_equal(velocity, .5*(edges[1:]+edges[:-1]))):
        raise ValueError('Invalid saved spectrum channel coordinates')
    values = np.asarray(payload['dL_dv_erg_s_per_kms'], dtype=float)
    if (values.shape != (len(keys), 2, velocity.size)
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError('Invalid saved spectrum values')
    if not np.array_equal(values.sum(axis=1), payload['total_dL_dv_erg_s_per_kms']):
        raise ValueError('Saved total differs from cold plus hot spectra')
    profiles, report = {}, {}
    for key in LINE_ORDER:
        index = keys.index(key)
        if key.startswith(('ciii_', 'civ_')):
            if np.any(values[index, 0] != 0):
                raise ValueError('Accepted CIII/CIV cold contribution must be zero')
            branch, profile = 'hot', values[index, 1]
        elif key.startswith('co'):
            branch, profile = 'cold', values[index, 0]
        else:
            branch, profile = 'total', values[index].sum(axis=0)
        profiles[key] = profile.copy()
        weight = profile*widths
        total = float(weight.sum())
        mean = float(np.sum(weight*velocity)/total) if total else None
        sigma = float(np.sqrt(np.sum(weight*(velocity-mean)**2)/total)) if total else None
        report[key] = dict(display_branch=branch, captured_luminosity_erg_s=total,
                           mean_velocity_kms=mean, sigma_line_window_kms=sigma)
    return profiles, report


def plot_phase_spectrum_overlay(payload, phase_payload, report, output_stem, *,
                                line_keys=LINE_ORDER, figure_style='latex'):
    """Render publication panels or complete standalone figures.

    The LaTeX style leaves titles and explanatory text to manuscript captions.
    Velocity limits affect the view only, not curve normalization or moments.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    profiles, line_report = accepted_display_profiles(payload)
    if figure_style not in ('latex', 'full'):
        raise ValueError('figure_style must be latex or full')
    line_keys = tuple(line_keys)
    if not line_keys or len(set(line_keys)) != len(line_keys) or not set(line_keys) <= set(LINE_ORDER):
        raise ValueError('Choose unique transitions from LINE_ORDER')
    edges = np.asarray(payload['velocity_edges_kms'])
    if not np.array_equal(edges, phase_payload['velocity_edges_kms']):
        raise ValueError('Phase and line velocity channels must be identical')
    velocity = np.asarray(payload['velocity_kms'])
    histogram = np.asarray(phase_payload['histogram_mass_g'])
    if (tuple(phase_payload['phase_keys']) != (*PHASE_ORDER, 'total')
            or histogram.shape != (6, velocity.size)
            or not np.isfinite(histogram).all() or np.any(histogram < 0)):
        raise ValueError('Invalid phase histograms')

    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    groups = report['velocity_statistics']['groups']
    for key in line_keys:
        full = figure_style == 'full'
        fig = plt.figure(figsize=(9.1, 5.25 if full else 4.15))
        grid = fig.add_gridspec(1, 2, left=.095, right=.985, bottom=.20 if full else .17,
                               top=.85 if full else .98, width_ratios=(1, .40), wspace=.05)
        ax = fig.add_subplot(grid[0])
        legend_ax = fig.add_subplot(grid[1])
        legend_ax.axis('off')
        handles, labels = [], []
        for phase_index, (phase, color) in enumerate(zip(PHASE_ORDER, PHASE_COLORS)):
            values = histogram[phase_index]
            peak = values.max()
            if peak <= 0:
                continue
            ax.stairs(values/peak, edges, color=color, lw=1.3)
            sigma = groups[phase]['sigma_about_global_mean_kms']
            handles.append(Line2D([], [], color=color, lw=1.6))
            labels.append(phase+'\n'+rf'$\sigma_{{z,p}}={sigma:.1f}$ km/s')
        total = histogram[-1]
        if total.max() > 0:
            ax.stairs(total/total.max(), edges, color='.45', ls='--', lw=1.5)
            handles.append(Line2D([], [], color='.45', ls='--', lw=1.6))
            sigma = groups['total']['sigma_internal_kms']
            labels.append('All gas\n'+rf'$\sigma_z={sigma:.1f}$ km/s')
        profile = profiles[key]
        if profile.max() > 0:
            ax.plot(velocity, profile/profile.max(), color='#111111', ls='--', lw=2.2, zorder=5)
            handles.append(Line2D([], [], color='#111111', ls='--', lw=2.2))
            sigma = line_report[key]['sigma_line_window_kms']
            labels.append('Line\n'+rf'$\sigma_{{\rm line}}={sigma:.1f}$ km/s')
        legend_ax.legend(handles, labels, loc='center left', frameon=False,
                         fontsize=9.3, handlelength=2.2, labelspacing=.65,
                         borderaxespad=0., handletextpad=.55)
        ax.set_xlim(*DISPLAY_VELOCITY_LIMITS_KMS.get(key, (edges[0], edges[-1])))
        ax.set_ylim(-.035, 1.07)
        ax.axvline(0, color='.65', ls=':', lw=.8)
        ax.grid(alpha=.18)
        ax.set_xlabel(r'$v_z$ [km s$^{-1}$]', fontsize=11)
        ax.set_ylabel('Peak-normalised profile', fontsize=11)
        ax.tick_params(labelsize=10)
        if full:
            fig.suptitle(LINE_TITLES[key], fontsize=15, y=.965)
            fig.text(.5, .902, 'LOS z', ha='center', fontsize=10)
            footer = (
                r'Phase cuts [K]: CNM $<200$; UNM $200$-$3000$; WNM $3000$-$10^4$; '
                r'WIM $10^4$-$10^{5.5}$; HIM $\geq10^{5.5}$.'
            )
            fig.text(.5, .052, footer, ha='center', va='center', fontsize=8.4)
        line_stem = stem.with_name(f'{stem.name}_{key}')
        for extension in ('png', 'pdf'):
            fig.savefig(line_stem.with_suffix('.'+extension), dpi=200)
        plt.close(fig)
