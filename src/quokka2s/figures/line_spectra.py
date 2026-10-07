"""Draw saved intrinsic and attenuated total line spectra."""
from __future__ import annotations

from pathlib import Path

from quokka2s.emission_results import SpectralResults
from quokka2s.figures.figure_files import save_figure_formats
from quokka2s.figures.line_labels import LINE_TITLES
from quokka2s.figures.display_settings import VELOCITY_LIMITS_KMS


def plot_line_spectra(
    spectra: SpectralResults,
    output: Path,
    *,
    per_projected_area: bool,
    diagnostic_suffix: str,
    titled: bool,
    formats: tuple[str, ...],
) -> list[Path]:
    """Draw saved total profiles on their original velocity channels [km/s].

    spectra owns line/dust names and both saved profile units. per_projected_area
    selects the stored light per area [erg/s/cm^2/(km/s)] rather than raw light
    [erg/s/(km/s)]; the renderer performs no unit conversion. Returns
    one figure per line/format; the visible window remains +/-50 km/s.
    Example: spectra.for_line(line="halpha", dust_state="attenuated") selects Halpha.
    """
    import matplotlib.pyplot as plt

    if per_projected_area:
        ylabel = r"$d\Sigma_L/dv$ [erg s$^{-1}$ cm$^{-2}$ (km s$^{-1}$)$^{-1}$]"
    else:
        ylabel = r"$dL/dv$ [erg s$^{-1}$ (km s$^{-1}$)$^{-1}$]"
    spectrum_paths = []
    for key in spectra.line_keys:
        intrinsic = spectra.for_line(
            line=key,
            dust_state="intrinsic",
            regime="total",
        )
        attenuated = spectra.for_line(
            line=key,
            dust_state="attenuated",
            regime="total",
        )
        if per_projected_area:
            intrinsic_profile = intrinsic.dSigmaL_dv_erg_s_cm2_per_kms
            attenuated_profile = attenuated.dSigmaL_dv_erg_s_cm2_per_kms
        else:
            intrinsic_profile = intrinsic.dL_dv_erg_s_per_kms
            attenuated_profile = attenuated.dL_dv_erg_s_per_kms
        figure, axis = plt.subplots(figsize=(6.2, 3.5))
        if key == "hi21":
            axis.plot(
                intrinsic.velocity_kms,
                intrinsic_profile,
                color="#242424",
                lw=1.8,
            )
        else:
            axis.plot(
                intrinsic.velocity_kms,
                intrinsic_profile,
                color="#242424",
                lw=1.5,
                ls="--",
                label="Intrinsic",
            )
            axis.plot(
                attenuated.velocity_kms,
                attenuated_profile,
                color="#C53D46",
                lw=1.8,
                label="Dust attenuated",
            )
            axis.legend(frameon=False, fontsize=9)
        axis.set_xlim(*VELOCITY_LIMITS_KMS)
        axis.set_ylim(bottom=0)
        axis.set_xlabel(r"$v_z$ [km s$^{-1}$]")
        axis.set_ylabel(ylabel)
        if titled:
            axis.set_title(LINE_TITLES[key])
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        axis.grid(alpha=0.18)
        figure.tight_layout()
        spectrum_paths.extend(save_figure_formats(
            figure=figure,
            stem=output / f"spectrum_{key}{diagnostic_suffix}",
            formats=formats,
        ))
        plt.close(figure)
    return spectrum_paths
