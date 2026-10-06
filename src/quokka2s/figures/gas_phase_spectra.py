"""Compare saved line spectra with gas mass velocity distributions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quokka2s.figures.line_labels import LINE_TITLES


LINE_ORDER = (
    'halpha', 'hi21', 'cii', 'ciii_977', 'ciii_1907', 'ciii_1909',
    'civ_1548', 'civ_1551', 'co10', 'co21',
)
PHASE_ORDER = ('CNM', 'UNM', 'WNM', 'WIM', 'HIM')
PHASE_COLORS = ('#0072B2', '#56B4E9', '#009E73', '#E69F00', '#D55E00')
# Peaks use the saved channels; sigma labels use the complete Gaussian moments.
DISPLAY_VELOCITY_LIMITS_KMS = (-50., 50.)


@dataclass(frozen=True)
class LineSpectraForComparison:
    """One dust state's saved spectra, before choosing displayed components.

    Attributes
    ----------
    line_keys : tuple of str, length L
        Saved line order, normally the ten transitions in LINE_ORDER.
    velocity_kms : numpy.ndarray, shape (V,)
        Saved channel centres [km/s], normally V = 400.
    dL_dv_erg_s_per_kms : numpy.ndarray, shape (L, 2, V)
        One dust state's line spectra [erg/s/(km/s)]. The process/plot
        command selects dust-attenuated spectra before constructing this object.
        Axis 1 is cold, then hot; no cell emission is recalculated here.
    line_centroid_window_kms : numpy.ndarray, shape (L,)
        Process-calculated total-line centroid [km/s] within the saved channels.
        Retained as profile metadata; drawing uses the saved velocity axis.
    line_sigma_full_kms : numpy.ndarray, shape (L,)
        Process-calculated whole-box dispersion [km/s], including each cell's
        complete Gaussian and emission outside the saved channels. Used in
        the line legend. Cold and hot luminosities are combined for every line.

    Examples
    --------
    ``spectra.dL_dv_erg_s_per_kms[0, 1]`` is the first line's hot profile.
    """

    line_keys: tuple[str, ...]
    velocity_kms: np.ndarray
    dL_dv_erg_s_per_kms: np.ndarray
    line_centroid_window_kms: np.ndarray
    line_sigma_full_kms: np.ndarray


@dataclass(frozen=True)
class GasPhaseVelocityProfiles:
    """Saved gas histograms and dispersions used beside each line profile.

    Attributes
    ----------
    histogram_mass_g : numpy.ndarray, shape (6, V)
        Gas mass [g per channel]; rows are CNM, UNM, WNM, WIM, HIM, all gas.
    sigma_about_global_mean_kms : numpy.ndarray, shape (6,)
        Each group's dispersion about the common gas mean [km/s]. The first
        five entries provide the phase labels.
    sigma_internal_kms : numpy.ndarray, shape (6,)
        Each group's dispersion about its own mean [km/s]. The final entry
        provides the all-gas label.

    Examples
    --------
    ``gas_phases.histogram_mass_g[0]`` is the CNM mass profile; the final
    row is the all-gas mass profile.
    """

    histogram_mass_g: np.ndarray
    sigma_about_global_mean_kms: np.ndarray
    sigma_internal_kms: np.ndarray


def select_display_profiles(
    line_spectra: LineSpectraForComparison,
) -> dict[str, np.ndarray]:
    """Choose total Halpha/HI/CII/CO and hot CIII/CIV profiles.

    Parameters
    ----------
    line_spectra : LineSpectraForComparison
        One dust state's saved spectra; the process/plot command uses the
        attenuated state. See the class for array shapes and units.

    Returns
    -------
    dict of str to numpy.ndarray, each shape (V,)
        Independent profile copies [erg/s/(km/s)], keyed by line name.

    Examples
    --------
    ``profiles = select_display_profiles(line_spectra)`` then
    ``profiles['co10']`` contains the cold + hot CO(1-0) spectrum.
    """
    profiles = {}
    for key in LINE_ORDER:
        line_index = line_spectra.line_keys.index(key)
        cold_and_hot = line_spectra.dL_dv_erg_s_per_kms[line_index]
        if key.startswith(('ciii_', 'civ_')):
            profile = cold_and_hot[1]
        else:
            profile = cold_and_hot.sum(axis=0)
        profiles[key] = profile.copy()
    return profiles


def display_branch_for_line(key: str) -> str:
    """Name the temperature component displayed for one transition."""
    if key.startswith(('ciii_', 'civ_')):
        return 'hot'
    return 'total'


def read_display_profile_statistics(
    line_spectra: LineSpectraForComparison,
) -> dict[str, dict]:
    """Read the saved full-profile sigma and window-centroid metadata.

    Parameters
    ----------
    line_spectra : LineSpectraForComparison
        Saved whole-box line statistics [km/s], calculated by process.

    Returns
    -------
    dict of str to dict
        Each line's displayed branch, mean_velocity_kms, and
        sigma_line_full_kms. No moments are recalculated by plot. CIII/CIV
        have zero cold emission, so their whole-box moments equal hot moments.

    Examples
    --------
    ``statistics['co10']['sigma_line_full_kms']`` is CO(1-0)'s full
    whole-box dispersion, combining its cold and hot emission.
    """
    statistics = {}
    for key in LINE_ORDER:
        line_index = line_spectra.line_keys.index(key)
        branch = display_branch_for_line(key)
        centroid = line_spectra.line_centroid_window_kms[line_index]
        sigma = line_spectra.line_sigma_full_kms[line_index]
        statistics[key] = {
            "display_branch": branch,
            "mean_velocity_kms": float(centroid),
            "sigma_line_full_kms": float(sigma),
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
    velocity_kms: np.ndarray,
    gas_phases: GasPhaseVelocityProfiles,
):
    """Draw phase/all-gas mass profiles and return their legend handles and labels.

    gas_phases supplies (6, V) histograms [g per channel] and (6,)
    dispersions [km/s]. Each curve is divided by its own full-window peak;
    WIM/HIM are faint only for CO figures.
    """
    from matplotlib.lines import Line2D

    handles = []
    labels = []
    for phase_index, (phase, color) in enumerate(zip(PHASE_ORDER, PHASE_COLORS)):
        values = gas_phases.histogram_mass_g[phase_index]
        peak = values.max()
        if peak <= 0:
            continue
        alpha = 1.
        if line_key in ('co10', 'co21') and phase in ('WIM', 'HIM'):
            alpha = .25
        ax.plot(velocity_kms, values / peak, color=color, lw=1.3, alpha=alpha)
        sigma = gas_phases.sigma_about_global_mean_kms[phase_index]
        handles.append(Line2D([], [], color=color, lw=1.6, alpha=alpha))
        labels.append(phase + '\n' + rf'$\sigma_{{z,p}}={sigma:.1f}$ km/s')

    all_gas_profile = gas_phases.histogram_mass_g[-1]
    if all_gas_profile.max() > 0:
        ax.plot(
            velocity_kms,
            all_gas_profile / all_gas_profile.max(),
            color='.45',
            ls='--',
            lw=1.5,
        )
        handles.append(Line2D([], [], color='.45', ls='--', lw=1.6))
        sigma = gas_phases.sigma_internal_kms[-1]
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
    gas_phases: GasPhaseVelocityProfiles,
    profile,
    line_statistics,
):
    """Draw gas and line curves, then place their combined legend beside the axes."""
    handles, labels = draw_gas_mass_profile_curves(
        ax=ax,
        line_key=key,
        velocity_kms=velocity,
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



def format_phase_comparison_figure(fig, ax, key, *, full):
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
        footer = (
            r'Phase cuts [K]: CNM $<200$; UNM $200$-$3000$; WNM $3000$-$10^4$; '
            r'WIM $10^4$-$10^{5.5}$; HIM $\geq10^{5.5}$.'
        )
        fig.text(.5, .052, footer, ha='center', va='center', fontsize=8.4)


def plot_phase_spectrum_overlay(
    line_spectra: LineSpectraForComparison,
    gas_phases: GasPhaseVelocityProfiles,
    output_stem: str | Path,
    *,
    line_keys=LINE_ORDER,
    figure_style='latex',
    formats=('png', 'pdf'),
):
    """Write a separate peak-normalized gas/line figure for each transition.

    Parameters
    ----------
    line_spectra : LineSpectraForComparison
        One dust state's saved spectral arrays. process/plot uses attenuated
        spectra. No summed-profile or channel-centre copies are needed.
    gas_phases : GasPhaseVelocityProfiles
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
    None
        Writes figures; each profile is divided by its own full-window peak.

    Examples
    --------
    ``plot_phase_spectrum_overlay(line_spectra, gas_phases, figure_stem)``
    writes files such as ``gas_phase_spectrum_co10.png``.
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
        format_phase_comparison_figure(fig, ax, key, full=full)
        line_stem = stem.with_name(f'{stem.name}_{key}')
        for extension in formats:
            fig.savefig(line_stem.with_suffix('.' + extension), dpi=200)
        plt.close(fig)
