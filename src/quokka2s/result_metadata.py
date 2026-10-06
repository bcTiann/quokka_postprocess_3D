"""Small geometry and dust records attached before results are checked or saved."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from quokka2s.line_definitions import LINE_DEFINITIONS

if TYPE_CHECKING:
    from quokka2s.physics.cell_emission import CellEmissionCalculator
    from quokka2s.snapshot_reader import Snapshot


@dataclass(frozen=True)
class ResultMetadata:
    """Interpretation of the selected region and the ordered line arrays.

    processing_region : dict
        Original x/y index bounds, full z range, selected shape and area [cm^2].
    x_edges_kpc, y_edges_kpc : ndarray
        Selected native pixel edges, shapes (Nx+1,) and (Ny+1,), in kpc.
    rest_wavelength_micron, dust_cross_section_cm2_H : ndarray, shape (L,)
        Physical line values, in the calculator's output line order.

    Example
    -------
    Metadata for x=[64,128) has 65 x edges at the original global coordinates.
    It contains no yt dataset, cell arrays, readers or execution state.
    """

    processing_region: dict
    x_edges_kpc: np.ndarray
    y_edges_kpc: np.ndarray
    rest_wavelength_micron: np.ndarray
    dust_cross_section_cm2_H: np.ndarray
    dust_observer_side: str = "outer -z boundary face"

    @classmethod
    def from_processing_inputs(
        cls,
        *,
        snapshot: Snapshot,
        emission_calculator: CellEmissionCalculator,
    ) -> ResultMetadata:
        """Extract geometry and adopted dust values once from the loaded inputs.

        snapshot provides original geometry and selected native index bounds.
        emission_calculator provides line order and fixed cross-sections.
        Returns a small ResultMetadata record; input objects are not retained.
        """
        x_start, x_stop = snapshot.xy_region["x"]
        y_start, y_stop = snapshot.xy_region["y"]
        left_kpc = snapshot.dataset.domain_left_edge.to("kpc").value
        right_kpc = snapshot.dataset.domain_right_edge.to("kpc").value
        x_edges_kpc = np.linspace(left_kpc[0], right_kpc[0], snapshot.shape[0] + 1)
        y_edges_kpc = np.linspace(left_kpc[1], right_kpc[1], snapshot.shape[1] + 1)

        processing_region = {
            "xy_region": {axis: list(bounds) for axis, bounds in snapshot.xy_region.items()},
            "z_index_range": [0, snapshot.shape[2]],
            "shape": list(snapshot.processing_shape),
            "cell_count": snapshot.processing_cell_count,
            "projected_area_cm2": snapshot.processing_area_cm2,
        }
        wavelengths = []
        cross_sections = []
        for key in emission_calculator.line_keys:
            wavelengths.append(LINE_DEFINITIONS[key].rest_wavelength_micron)
            cross_sections.append(emission_calculator.dust_cross_section_cm2_H[key])

        return cls(
            processing_region=processing_region,
            x_edges_kpc=x_edges_kpc[x_start:x_stop + 1],
            y_edges_kpc=y_edges_kpc[y_start:y_stop + 1],
            rest_wavelength_micron=np.asarray(wavelengths),
            dust_cross_section_cm2_H=np.asarray(cross_sections),
        )
