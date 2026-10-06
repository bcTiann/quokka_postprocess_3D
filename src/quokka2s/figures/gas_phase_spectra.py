"""Compare saved line spectra with gas mass velocity distributions."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from quokka2s.emission_results import GasPhaseResults, SpectralResults
from quokka2s.figures.line_labels import LINE_TITLES
from quokka2s.figures.figure_files import save_figure_formats
from quokka2s.products import HOT_REGIME_KEY


LINE_ORDER = (
    'halpha', 'hi21', 'cii', 'ciii_977', 'ciii_1907', 'ciii_1909',
    'civ_1548', 'civ_1551', 'co10', 'co21',
)
PHASE_COLORS = {
    'CNM': '#0072B2',
    'UNM': '#56B4E9',
    'WNM': '#009E73',
    'WIM': '#E69F00',
    'HIM': '#D55E00',
}
# Peaks use the saved channels; sigma labels use the complete Gaussian moments.
DISPLAY_VELOCITY_LIMITS_KMS = (-50., 50.)


def select_display_profiles(line_spectra: SpectralResults) -> dict[str, np.ndarray]:
    """Select attenuated total Halpha/HI/CII/CO and hot CIII/CIV profiles.

    line_spectra holds saved line/dust/regime names and packed spectra.
    Returns independent (V,) display copies [erg/s/(km/s)], keyed by line.
    Example: profiles["co10"] contains the saved cold+hot CO(1-0) spectrum.
    """
    profiles = {}
    for key in LINE_ORDER:
        regime = HOT_REGIME_KEY if display_branch_for_line(key) == "hot" else "total"
        selected = line_spectra.for_line(
            line=key,
            dust_state="attenuated",
            regime=regime,
        )
        profiles[key] = selected.dL_dv_erg_s_per_kms.copy()
    return profiles


def display_branch_for_line(key: str) -> str:
    """Name the temperature component displayed for one transition."""
    if key.startswith(('ciii_', 'civ_')):
        return 'hot'
    return 'total'


def read_display_profile_statistics(line_spectra: SpectralResults) -> dict[str, dict]:
    """Read total-line full sigma and window-centroid metadata [km/s].

    Saved total-line full moments include emission beyond the velocity window.
    CIII/CIV have zero cold light, so total full sigma also describes their hot
    display profile. Drawing keeps saved velocity coordinates without recentering.
    Example: statistics["co10"]["sigma_line_full_kms"] is its saved full sigma.
    """
    statistics = {}
    for key in LINE_ORDER:
        total = line_spectra.for_line(
            line=key,
            dust_state="attenuated",
            regime="total",
        )
        statistics[key] = {
            "display_branch": display_branch_for_line(key),
            "mean_velocity_kms": total.centroid_window_kms,
            "sigma_line_full_kms": total.sigma_full_kms,
        }
    return statistics


def check_phase_comparison_options(line_keys: tuple[str, ...], figure_style: str) -> None:
    """Check the user's transition selection and requested figure style."""
    if figure_style not in ('latex', 'full'):
        raise ValueError('figure_style must be latex or full')
    if (not line_keys or len(set(line_keys)) != len(line_keys)
            or not set(line_keys) <= set(LINE_ORDER)):
        raise ValueError('Choose unique transitions from LINE_ORDER')


def create_phase_comparison_figure(*, full):
    """Create the profile axes and a separate legend area for the chosen style."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.1, 5.25 if full else 4.15))
    grid = fig.add_gridspec(
        1,
        2,
        left=.095,
        right=.985,
        bottom=.20 if full else .17,
        top=.85 if full else .98,
        width_ratios=(1, .40),
        wspace=.05,
    )
    ax = fig.add_subplot(grid[0])
    legend_ax = fig.add_subplot(grid[1])
    legend_ax.axis('off')
    return fig, ax, legend_ax


def draw_gas_mass_profile_curves(
    ax,
    line_key: str,
    gas_phases: GasPhaseResults,
):
    """Draw named phase/all-gas profiles on their own saved coordinates.

    gas_phases provides mass [g per channel] and dispersions [km/s]. Each
    display curve is divided by its full-window peak; WIM/HIM are faint for CO.
    Returns legend handles/labels. Underlying saved arrays are unchanged.
    """
    from matplotlib.lines import Line2D

    handles = []
    labels = []
    for phase in gas_phases.phase_keys:
        if phase == 'total':
            continue
        color = PHASE_COLORS[phase]
        saved_phase = gas_phases.for_phase(phase=phase)
        values = saved_phase.histogram_mass_g
        peak = values.max()
        if peak <= 0:
            continue
        alpha = 1.
        if line_key in ('co10', 'co21') and phase in ('WIM', 'HIM'):
            alpha = .25
        ax.plot(saved_phase.velocity_kms, values / peak, color=color, lw=1.3, alpha=alpha)
        sigma = saved_phase.sigma_about_global_mean_kms
        handles.append(Line2D([], [], color=color, lw=1.6, alpha=alpha))
        labels.append(phase + '\n' + rf'$\sigma_{{z,p}}={sigma:.1f}$ km/s')

    all_gas = gas_phases.for_phase(phase="total")
    all_gas_profile = all_gas.histogram_mass_g
    if all_gas_profile.max() > 0:
        ax.plot(
            all_gas.velocity_kms,
            all_gas_profile / all_gas_profile.max(),
            color='.45',
            ls='--',
            lw=1.5,
        )
        handles.append(Line2D([], [], color='.45', ls='--', lw=1.6))
        sigma = all_gas.sigma_internal_kms
        labels.append('All gas\n' + rf'$\sigma_z={sigma:.1f}$ km/s')
    return handles, labels


def draw_line_profile_curve(
    ax,
    velocity_kms: np.ndarray,
    profile: np.ndarray,
    sigma_line_full_kms: float,
):
    """Draw a nonzero emission profile and return its legend handle and label.

    profile has shape (V,) [erg/s/(km/s)], selected by select_display_profiles().
    Its legend uses the saved complete-Gaussian dispersion [km/s], including
    emission outside the channel window. Drawing retains the saved channels.
    An empty profile returns no legend entry.
    """
    from matplotlib.lines import Line2D

    if profile.max() <= 0:
        return None, None
    ax.plot(
        velocity_kms,
        profile / profile.max(),
        color='#111111',
        ls='--',
        lw=2.2,
        zorder=5,
    )
    handle = Line2D([], [], color='#111111', ls='--', lw=2.2)
    label = 'Line\n' + rf'$\sigma_{{\rm line}}={sigma_line_full_kms:.1f}$ km/s'
    return handle, label


def draw_phase_comparison_curves(
    ax,
    legend_ax,
    key,
    velocity,
    gas_phases: GasPhaseResults,
    profile,
    line_statistics,
):
    """Draw gas and line curves, then place their combined legend beside the axes."""
    handles, labels = draw_gas_mass_profile_curves(
        ax=ax,
        line_key=key,
        gas_phases=gas_phases,
    )
    line_handle, line_label = draw_line_profile_curve(
        ax=ax,
        velocity_kms=velocity,
        profile=profile,
        sigma_line_full_kms=line_statistics['sigma_line_full_kms'],
    )
    if line_handle is not None:
        handles.append(line_handle)
        labels.append(line_label)
    legend_ax.legend(
        handles,
        labels,
        loc='center left',
        frameon=False,
        fontsize=9.3,
        handlelength=2.2,
        labelspacing=.65,
        borderaxespad=0.,
        handletextpad=.55,
    )



def format_phase_temperature_boundary(temperature_K: float) -> str:
    """Format a saved phase cut using the figure's established LaTeX style.

    Lower cuts use plain kelvin values; cuts at 10^4 K and above use powers
    of ten. Examples: 200 -> "200", 1e4 -> "10^4", 10**5.5 -> "10^{5.5}".
    Only the display text changes; phase selection belongs to the producer.
    """
    if temperature_K < 1.0e4:
        return f'{temperature_K:g}'
    exponent = f'{np.log10(temperature_K):g}'
    if len(exponent) == 1:
        return f'10^{exponent}'
    return f'10^{{{exponent}}}'


def format_phase_cut_footer(gas_phases: GasPhaseResults) -> str:
    """Describe the saved producer's phase names and lower-inclusive cuts.

    phase_keys contains the temperature phases in ascending order and total.
    phase_boundaries_K separates those phases. The saved values, rather than
    a second set of physical definitions in this renderer, supply the text.
    """
    phase_keys = tuple(key for key in gas_phases.phase_keys if key != 'total')
    cuts = tuple(
        format_phase_temperature_boundary(temperature_K=float(boundary))
        for boundary in gas_phases.phase_boundaries_K
    )
    descriptions = [rf'{phase_keys[0]} $<{cuts[0]}$']
    for phase, lower, upper in zip(phase_keys[1:-1], cuts[:-1], cuts[1:]):
        descriptions.append(rf'{phase} ${lower}$-${upper}$')
    descriptions.append(rf'{phase_keys[-1]} $\geq{cuts[-1]}$')
    return 'Phase cuts [K]: ' + '; '.join(descriptions) + '.'


def format_phase_comparison_figure(fig, ax, key, *, gas_phases: GasPhaseResults, full):
    """Apply the display zoom, axes, and optional title and phase-cut footer."""
    ax.set_xlim(*DISPLAY_VELOCITY_LIMITS_KMS)
    ax.set_ylim(-.035, 1.07)
    ax.axvline(0, color='.65', ls=':', lw=.8)
    ax.grid(alpha=.18)
    ax.set_xlabel(r'$v_z$ [km s$^{-1}$]', fontsize=11)
    ax.set_ylabel('Peak-normalised profile', fontsize=11)
    ax.tick_params(labelsize=10)
    if full:
        fig.suptitle(LINE_TITLES[key], fontsize=15, y=.965)
        fig.text(.5, .902, 'LOS z', ha='center', fontsize=10)
        footer = format_phase_cut_footer(gas_phases=gas_phases)
        fig.text(.5, .052, footer, ha='center', va='center', fontsize=8.4)


def plot_phase_spectrum_overlay(
    line_spectra: SpectralResults,
    gas_phases: GasPhaseResults,
    output_stem: str | Path,
    *,
    line_keys=LINE_ORDER,
    figure_style='latex',
    formats=('png', 'pdf'),
):
    """Write a separate peak-normalized gas/line figure for each transition.

    Parameters
    ----------
    line_spectra : SpectralResults
        Saved spectral arrays with named dust/regime axes. This renderer
        selects attenuated light; line coordinates remain unchanged.
    gas_phases : GasPhaseResults
        Saved histograms and dispersions from phase_velocity.npz.
    output_stem : str or pathlib.Path
        Destination without an extension; filenames append each line key.
    line_keys : sequence of str
        Transitions to draw, defaulting to all ten.
    figure_style : {"latex", "full"}
        Caption-only figures or standalone figures with titles and phase cuts.
    formats : tuple of str
        Figure extensions, normally ("png", "pdf").

    Returns
    -------
    list of pathlib.Path
        Saved files in png/ and pdf/ subdirectories. Each profile is divided
        by its own full-window peak.

    Examples
    --------
    ``plot_phase_spectrum_overlay(line_spectra, gas_phases, figure_stem)``
    writes files such as ``png/gas_phase_spectrum_co10.png``.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    line_keys = tuple(line_keys)
    profiles = select_display_profiles(line_spectra)
    line_statistics = read_display_profile_statistics(
        line_spectra=line_spectra,
    )
    check_phase_comparison_options(
        line_keys=line_keys,
        figure_style=figure_style,
    )
    if not formats or any(extension not in ("png", "pdf") for extension in formats):
        raise ValueError("formats must contain png and/or pdf")

    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    full = figure_style == 'full'
    paths = []
    for key in line_keys:
        fig, ax, legend_ax = create_phase_comparison_figure(full=full)
        draw_phase_comparison_curves(
            ax=ax,
            legend_ax=legend_ax,
            key=key,
            velocity=line_spectra.velocity_kms,
            gas_phases=gas_phases,
            profile=profiles[key],
            line_statistics=line_statistics[key],
        )
        format_phase_comparison_figure(
            fig=fig,
            ax=ax,
            key=key,
            gas_phases=gas_phases,
            full=full,
        )
        line_stem = stem.with_name(f'{stem.name}_{key}')
        paths.extend(save_figure_formats(
            figure=fig,
            stem=line_stem,
            formats=formats,
            bbox_inches=None,
        ))
        plt.close(fig)
    return paths
