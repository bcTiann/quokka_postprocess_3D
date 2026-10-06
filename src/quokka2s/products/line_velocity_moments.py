"""Luminosity-weighted moments of complete cell Gaussian line profiles."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Preserve the adopted 1 cm/s floor in both full moments and channel integrals.
MINIMUM_GAUSSIAN_WIDTH_KMS = 1.0e-5


@dataclass(frozen=True)
class LineVelocityMoments:
    """Small running totals for one line, dust state and gas-temperature branch.

    Attributes
    ----------
    luminosity_erg_s : float
        Total cell luminosity, including emission outside the saved channels.
    centroid_kms : float
        Luminosity-weighted mean of cell LOS velocities [km/s].
    centered_second_moment : float
        Sum of L * ((v - centroid)^2 + thermal_width^2) [erg/s * (km/s)^2].
        This includes each cell's complete Gaussian, not only its channel values.

    Example: two equal-light cells at -10 and +10 km/s with 3 km/s thermal
    widths have centroid 0 and sigma sqrt(10^2 + 3^2) km/s.
    """

    luminosity_erg_s: float = 0.0
    centroid_kms: float = 0.0
    centered_second_moment: float = 0.0

    @classmethod
    def from_cells(cls, *, velocity_kms, thermal_width_kms, luminosity_erg_s):
        """Summarize one line's already-selected cells without a velocity window.

        Inputs are matching (R,) NumPy arrays: LOS velocities and Gaussian
        widths [km/s], and cell luminosities [erg/s]. They come from the same
        inputs used by channel integration. Zero-light cells add no moments.
        Returns LineVelocityMoments; no cell arrays are retained.
        """
        emitting_cells = luminosity_erg_s > 0.0
        if not np.any(emitting_cells):
            return cls()
        luminosity = luminosity_erg_s[emitting_cells]
        velocity = velocity_kms[emitting_cells]
        # Use the same 1 cm/s Gaussian-width floor as the channel kernel.
        thermal_width = np.maximum(
            thermal_width_kms[emitting_cells],
            MINIMUM_GAUSSIAN_WIDTH_KMS,
        )
        total_luminosity = float(np.sum(luminosity, dtype=np.float64))

        # Shift velocities before summing to preserve a narrow width even when
        # the entire gas population has a large common velocity offset.
        velocity_offset = velocity - velocity[0]
        mean_offset = float(np.sum(luminosity * velocity_offset) / total_luminosity)
        centroid = float(velocity[0] + mean_offset)
        bulk_variance = (velocity_offset - mean_offset) ** 2
        thermal_variance = thermal_width ** 2
        centered_second = float(np.sum(luminosity * (bulk_variance + thermal_variance)))
        return cls(
            luminosity_erg_s=total_luminosity,
            centroid_kms=centroid,
            centered_second_moment=centered_second,
        )

    def merged(self, other):
        """Return moments for both batches, leaving both input objects unchanged.

        other is LineVelocityMoments for the same line and dust state.
        The merge includes the separation of the two batch centroids, so it
        also works for parallel batches or combining cold and hot branches.
        """
        if self.luminosity_erg_s == 0.0:
            return other
        if other.luminosity_erg_s == 0.0:
            return self
        luminosity = self.luminosity_erg_s + other.luminosity_erg_s
        centroid_difference = other.centroid_kms - self.centroid_kms
        other_fraction = other.luminosity_erg_s / luminosity
        centroid = self.centroid_kms + centroid_difference * other_fraction
        separation_moment = centroid_difference ** 2 * self.luminosity_erg_s * other_fraction
        centered_second = (
            self.centered_second_moment
            + other.centered_second_moment
            + separation_moment
        )
        return LineVelocityMoments(
            luminosity_erg_s=luminosity,
            centroid_kms=centroid,
            centered_second_moment=centered_second,
        )

    @property
    def sigma_kms(self):
        """Full-profile dispersion [km/s]; NaN when this line has zero light."""
        if self.luminosity_erg_s == 0.0:
            return np.nan
        return float(np.sqrt(self.centered_second_moment / self.luminosity_erg_s))

    @property
    def raw_second_moment_kms2(self):
        """Full-profile <v^2> [(km/s)^2]; NaN when this line has zero light.

        sigma^2 = <v^2> - centroid^2. Accumulation keeps the centered form
        above to avoid subtracting two large, nearly equal numbers at the end.
        """
        if self.luminosity_erg_s == 0.0:
            return np.nan
        return float(self.centered_second_moment / self.luminosity_erg_s + self.centroid_kms ** 2)
