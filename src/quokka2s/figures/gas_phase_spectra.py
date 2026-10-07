"""Compare saved line spectra with gas mass velocity distributions."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from quokka2s.emission_results import GasPhaseResults, LineSpectrum
from quokka2s.figures.line_labels import LINE_TITLES
from quokka2s.figures.figure_files import save_figure_formats
from quokka2s.figures.display_settings import VELOCITY_LIMITS_KMS


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


def create_phase_comparison_figure(*, titled: bool):
    """Create the profile axes and a separate legend area for the chosen style."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.1, 5.25 if titled else 4.15))
    grid = fig.add_gridspec(
        1,
        2,
        left=.095,
        right=.985,
        bottom=.20 if titled else .17,
        top=.85 if titled else .98,
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

    gas_phases holds saved normalized curves, coordinates and comparison sigma.
    WIM/HIM are faint for CO. Returns handles and labels in the saved phase
    order, followed by All gas. Underlying saved arrays are unchanged.
    """
    from matplotlib.lines import Line2D

    handles = []
    labels = []
    for phase_key in gas_phases.phase_keys:
        if phase_key == 'total':
            continue
        phase = gas_phases.for_phase(phase_key)
        if not phase.profile_has_mass:
            continue
        color = PHASE_COLORS[phase_key]
        alpha = 1.
        if line_key in ('co10', 'co21') and phase_key in ('WIM', 'HIM'):
            alpha = .25
        ax.plot(
            phase.velocity_kms,
            phase.peak_normalized_mass_profile,
            color=color,
            lw=1.3,
            alpha=alpha,
        )
        sigma = phase.comparison_sigma_kms
        handles.append(Line2D([], [], color=color, lw=1.6, alpha=alpha))
        labels.append(phase_key + '\n' + rf'$\sigma_{{z,p}}={sigma:.1f}$ km/s')

    total_gas = gas_phases.for_phase('total')
    if total_gas.profile_has_mass:
        ax.plot(
            total_gas.velocity_kms,
            total_gas.peak_normalized_mass_profile,
            color='.45',
            ls='--',
            lw=1.5,
        )
        handles.append(Line2D([], [], color='.45', ls='--', lw=1.6))
        sigma = total_gas.comparison_sigma_kms
        labels.append('All gas\n' + rf'$\sigma_z={sigma:.1f}$ km/s')
    return handles, labels


def draw_line_profile_curve(
    ax,
    line_spectrum: LineSpectrum,
):
    """Draw a nonzero emission profile and return its legend handle and label.

    line_spectrum contains the saved normalized attenuated total profile, (V,).
    Its saved full sigma [km/s] includes complete cell Gaussians outside the
    channel window. Drawing retains its own saved velocity coordinates.
    An empty profile returns no legend entry.
    """
    from matplotlib.lines import Line2D

    if not line_spectrum.profile_has_light:
        return None, None
    ax.plot(
        line_spectrum.velocity_kms,
        line_spectrum.peak_normalized_profile,
        color='#111111',
        ls='--',
        lw=2.2,
        zorder=5,
    )
    handle = Line2D([], [], color='#111111', ls='--', lw=2.2)
    sigma = line_spectrum.sigma_full_kms
    label = 'Line\n' + rf'$\sigma_{{\rm line}}={sigma:.1f}$ km/s'
    return handle, label


def draw_phase_comparison_curves(
    ax,
    legend_ax,
    key,
    gas_phases: GasPhaseResults,
    line_spectrum: LineSpectrum,
):
    """Draw gas and line curves, then place their combined legend beside the axes."""
    handles, labels = draw_gas_mass_profile_curves(
        ax=ax,
        line_key=key,
        gas_phases=gas_phases,
    )
    line_handle, line_label = draw_line_profile_curve(
        ax=ax,
        line_spectrum=line_spectrum,
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


def format_phase_comparison_figure(
    fig,
    ax,
    key: str,
    *,
    phase_cut_footer: str,
    titled: bool,
):
    """Apply the display zoom, axes, and optional title and phase-cut footer."""
    ax.set_xlim(*VELOCITY_LIMITS_KMS)
    ax.set_ylim(-.035, 1.07)
    ax.axvline(0, color='.65', ls=':', lw=.8)
    ax.grid(alpha=.18)
    ax.set_xlabel(r'$v_z$ [km s$^{-1}$]', fontsize=11)
    ax.set_ylabel('Peak-normalised profile', fontsize=11)
    ax.tick_params(labelsize=10)
    if titled:
        fig.suptitle(LINE_TITLES[key], fontsize=15, y=.965)
        fig.text(.5, .902, 'LOS z', ha='center', fontsize=10)
        fig.text(.5, .052, phase_cut_footer, ha='center', va='center', fontsize=8.4)


def plot_phase_spectrum_overlay(
    line_spectra: dict[str, LineSpectrum],
    gas_phases: GasPhaseResults,
    output_stem: str | Path,
    *,
    titled: bool,
    formats: tuple[str, ...],
) -> list[Path]:
    """Write a separate peak-normalized gas/line figure for each transition.

    Parameters
    ----------
    line_spectra : dict of str to LineSpectrum
        Named attenuated total profiles with their own saved coordinates
        and full-line dispersions [km/s]. Prepared once for both styles.
    gas_phases : GasPhaseResults
        Saved gas coordinates, normalized mass profiles and comparison sigma.
    output_stem : str or pathlib.Path
        Destination without an extension; filenames append each line key.
    titled : bool
        Add standalone titles and phase cuts using the taller figure layout.
    formats : tuple of str
        Figure extensions, normally ("png", "pdf").

    Returns
    -------
    list of pathlib.Path
        Saved files in png/ and pdf/ subdirectories. Profiles were already
        normalized by process using their own full-window peaks.

    Examples
    --------
    ``plot_phase_spectrum_overlay(..., titled=False, formats=("png",))``
    writes files such as ``png/gas_phase_spectrum_co10.png``.
    """
    import matplotlib.pyplot as plt

    stem = Path(output_stem)
    phase_cut_footer = format_phase_cut_footer(gas_phases=gas_phases)
    paths = []
    for key, line_spectrum in line_spectra.items():
        fig, ax, legend_ax = create_phase_comparison_figure(titled=titled)
        draw_phase_comparison_curves(
            ax=ax,
            legend_ax=legend_ax,
            key=key,
            gas_phases=gas_phases,
            line_spectrum=line_spectrum,
        )
        format_phase_comparison_figure(
            fig=fig,
            ax=ax,
            key=key,
            phase_cut_footer=phase_cut_footer,
            titled=titled,
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
