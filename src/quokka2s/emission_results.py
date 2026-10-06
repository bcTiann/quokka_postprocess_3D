"""Read named image, spectrum and gas-phase results without simulation inputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ImageResults:
    """Saved luminosity images and their own pixel coordinates.

    Arrays use (dust state, line, x, y); luminosities are erg/s per pixel and
    x/y edges are kpc. Names retain the order stored in images.npz.
    Example: images.for_line(line="halpha", dust_state="attenuated") is (Nx, Ny).
    """

    line_keys: tuple[str, ...]
    dust_state_keys: tuple[str, ...]
    axis_order: str
    line_luminosity_image_erg_s: np.ndarray
    x_edges_kpc: np.ndarray
    y_edges_kpc: np.ndarray

    def for_line(self, line: str, dust_state: str) -> np.ndarray:
        """Return the selected (Nx, Ny) luminosity view [erg/s per pixel]."""
        line_index = self.line_keys.index(line)
        dust_index = self.dust_state_keys.index(dust_state)
        return self.line_luminosity_image_erg_s[dust_index, line_index]


@dataclass(frozen=True)
class LineSpectrum:
    """One saved line profile and its explicitly named moments.

    velocity_kms and dL_dv_erg_s_per_kms are (V,) views of saved coordinates
    and light [erg/s/(km/s)]. Window moments refer to saved channels. Full
    moments include complete cell Gaussians and are saved only for total
    cold+hot profiles; they are None when selecting a single regime.
    """

    velocity_kms: np.ndarray
    dL_dv_erg_s_per_kms: np.ndarray
    centroid_window_kms: float
    sigma_window_kms: float
    centroid_full_kms: float | None
    sigma_full_kms: float | None


@dataclass(frozen=True)
class SpectralResults:
    """Saved packed spectra with their own line, dust and regime names.

    Profiles use (dust state, line, regime, channel) [erg/s/(km/s)]; total
    profiles use (dust state, line, channel). Total moments use (dust, line),
    regime window moments use (dust, line, regime), and coordinates are km/s.
    Example: spectra.for_line("co10", "attenuated", "total") uses saved totals.
    """

    line_keys: tuple[str, ...]
    dust_state_keys: tuple[str, ...]
    regime_keys: tuple[str, ...]
    axis_order: str
    velocity_edges_kms: np.ndarray
    velocity_kms: np.ndarray
    dL_dv_erg_s_per_kms: np.ndarray
    total_dL_dv_erg_s_per_kms: np.ndarray
    projected_area_cm2: float
    line_centroid_window_kms: np.ndarray
    line_sigma_window_kms: np.ndarray
    line_centroid_full_kms: np.ndarray
    line_sigma_full_kms: np.ndarray
    line_centroid_window_by_regime_kms: np.ndarray
    line_sigma_window_by_regime_kms: np.ndarray

    def for_line(self, line: str, dust_state: str, regime: str = "total") -> LineSpectrum:
        """Select saved light and moments by name without recomputing a profile.

        regime is "total" or an exact saved name, e.g. "T_QUOKKA_ge_3000K".
        The returned profile is a (V,) view. Full moments remain None for a
        regime because spectra.npz stores full moments only for total lines.
        """
        line_index = self.line_keys.index(line)
        dust_index = self.dust_state_keys.index(dust_state)
        if regime == "total":
            profile = self.total_dL_dv_erg_s_per_kms[dust_index, line_index]
            centroid_window = float(self.line_centroid_window_kms[dust_index, line_index])
            sigma_window = float(self.line_sigma_window_kms[dust_index, line_index])
            centroid_full = float(self.line_centroid_full_kms[dust_index, line_index])
            sigma_full = float(self.line_sigma_full_kms[dust_index, line_index])
        else:
            regime_index = self.regime_keys.index(regime)
            profile = self.dL_dv_erg_s_per_kms[dust_index, line_index, regime_index]
            centroid_window = float(
                self.line_centroid_window_by_regime_kms[dust_index, line_index, regime_index]
            )
            sigma_window = float(
                self.line_sigma_window_by_regime_kms[dust_index, line_index, regime_index]
            )
            centroid_full = None
            sigma_full = None
        return LineSpectrum(
            velocity_kms=self.velocity_kms,
            dL_dv_erg_s_per_kms=profile,
            centroid_window_kms=centroid_window,
            sigma_window_kms=sigma_window,
            centroid_full_kms=centroid_full,
            sigma_full_kms=sigma_full,
        )


@dataclass(frozen=True)
class GasPhaseProfile:
    """One named phase's (V,) saved mass histogram [g] and dispersions [km/s]."""

    velocity_kms: np.ndarray
    histogram_mass_g: np.ndarray
    sigma_about_global_mean_kms: float
    sigma_internal_kms: float


@dataclass(frozen=True)
class GasPhaseResults:
    """Saved gas histograms, phase names and their own velocity edges.

    Histograms use (phase, channel) [g per channel]; dispersions use (phase,)
    [km/s]. Saved phase_boundaries_K separate the named temperature phases,
    excluding total. Names normally include CNM, UNM, WNM, WIM, HIM and total.
    Example: gas_phases.for_phase("CNM") retains the phase file's coordinates.
    """

    phase_keys: tuple[str, ...]
    phase_boundaries_K: np.ndarray
    velocity_edges_kms: np.ndarray
    histogram_mass_g: np.ndarray
    sigma_about_global_mean_kms: np.ndarray
    sigma_internal_kms: np.ndarray

    def for_phase(self, phase: str) -> GasPhaseProfile:
        """Return one phase's saved histogram view and named statistics.

        Coordinates are channel centres [km/s], derived from this product's
        saved edges. Gas profiles do not borrow a line spectrum's velocity grid.
        """
        phase_index = self.phase_keys.index(phase)
        velocity = 0.5 * (self.velocity_edges_kms[:-1] + self.velocity_edges_kms[1:])
        return GasPhaseProfile(
            velocity_kms=velocity,
            histogram_mass_g=self.histogram_mass_g[phase_index],
            sigma_about_global_mean_kms=float(self.sigma_about_global_mean_kms[phase_index]),
            sigma_internal_kms=float(self.sigma_internal_kms[phase_index]),
        )


@dataclass(frozen=True)
class EmissionResults:
    """Three processed products shared by all figure versions.

    processing_complete means the requested region was completed; full_snapshot
    additionally means that region is the complete original box. No dataset,
    lookup table, calculator or per-cell arrays are retained.
    """

    images: ImageResults
    spectra: SpectralResults
    gas_phases: GasPhaseResults
    processing_complete: bool
    full_snapshot: bool


def _read_product_arrays(path: Path, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Read current NPZ fields as independent arrays, then close the file."""
    with np.load(path, allow_pickle=False) as saved:
        return {field: saved[field].copy() for field in fields}


def read_emission_results(directory: str | Path) -> EmissionResults:
    """Read the current image, spectrum and gas-phase NPZ products once.

    Parameters
    ----------
    directory : str or pathlib.Path
        Process output containing images.npz, spectra.npz and phase_velocity.npz.

    Returns
    -------
    EmissionResults
        Named products with their own saved coordinates and statistics. Reading
        performs no snapshot access or emission calculation. Partial results
        retain their completion flags for the plotting entry point to handle.

    Example: results = read_emission_results("output/region/processed").
    """
    directory = Path(directory)
    image_arrays = _read_product_arrays(
        path=directory / "images.npz",
        fields=(
            "line_keys", "dust_state_keys", "axis_order", "line_luminosity_image_erg_s",
            "x_edges_kpc", "y_edges_kpc", "processing_complete", "full_snapshot",
        ),
    )
    spectral_arrays = _read_product_arrays(
        path=directory / "spectra.npz",
        fields=(
            "line_keys", "dust_state_keys", "regime_keys", "axis_order",
            "velocity_edges_kms", "velocity_kms", "dL_dv_erg_s_per_kms",
            "total_dL_dv_erg_s_per_kms", "projected_area_cm2",
            "line_centroid_window_kms", "line_sigma_window_kms",
            "line_centroid_full_kms", "line_sigma_full_kms",
            "line_centroid_window_by_regime_kms", "line_sigma_window_by_regime_kms",
        ),
    )
    phase_arrays = _read_product_arrays(
        path=directory / "phase_velocity.npz",
        fields=(
            "phase_keys", "phase_boundaries_K", "velocity_edges_kms", "histogram_mass_g",
            "sigma_about_global_mean_kms", "sigma_internal_kms",
        ),
    )
    images = ImageResults(
        line_keys=tuple(str(key) for key in image_arrays["line_keys"]),
        dust_state_keys=tuple(str(key) for key in image_arrays["dust_state_keys"]),
        axis_order=str(image_arrays["axis_order"]),
        line_luminosity_image_erg_s=image_arrays["line_luminosity_image_erg_s"],
        x_edges_kpc=image_arrays["x_edges_kpc"],
        y_edges_kpc=image_arrays["y_edges_kpc"],
    )
    spectra = SpectralResults(
        line_keys=tuple(str(key) for key in spectral_arrays["line_keys"]),
        dust_state_keys=tuple(str(key) for key in spectral_arrays["dust_state_keys"]),
        regime_keys=tuple(str(key) for key in spectral_arrays["regime_keys"]),
        axis_order=str(spectral_arrays["axis_order"]),
        velocity_edges_kms=spectral_arrays["velocity_edges_kms"],
        velocity_kms=spectral_arrays["velocity_kms"],
        dL_dv_erg_s_per_kms=spectral_arrays["dL_dv_erg_s_per_kms"],
        total_dL_dv_erg_s_per_kms=spectral_arrays["total_dL_dv_erg_s_per_kms"],
        projected_area_cm2=float(spectral_arrays["projected_area_cm2"]),
        line_centroid_window_kms=spectral_arrays["line_centroid_window_kms"],
        line_sigma_window_kms=spectral_arrays["line_sigma_window_kms"],
        line_centroid_full_kms=spectral_arrays["line_centroid_full_kms"],
        line_sigma_full_kms=spectral_arrays["line_sigma_full_kms"],
        line_centroid_window_by_regime_kms=spectral_arrays["line_centroid_window_by_regime_kms"],
        line_sigma_window_by_regime_kms=spectral_arrays["line_sigma_window_by_regime_kms"],
    )
    gas_phases = GasPhaseResults(
        phase_keys=tuple(str(key) for key in phase_arrays["phase_keys"]),
        phase_boundaries_K=phase_arrays["phase_boundaries_K"],
        velocity_edges_kms=phase_arrays["velocity_edges_kms"],
        histogram_mass_g=phase_arrays["histogram_mass_g"],
        sigma_about_global_mean_kms=phase_arrays["sigma_about_global_mean_kms"],
        sigma_internal_kms=phase_arrays["sigma_internal_kms"],
    )
    return EmissionResults(
        images=images,
        spectra=spectra,
        gas_phases=gas_phases,
        processing_complete=bool(image_arrays["processing_complete"]),
        full_snapshot=bool(image_arrays["full_snapshot"]),
    )
