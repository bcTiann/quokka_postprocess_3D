"""Locate existing template corners and prepare original weighted scatter terms.

Means and widths use km/s, luminosity[cell,profile] uses erg/s. The shared
engine validates cells before these steps. No parameter-grid approximation,
integer-shift rule or per-corner support is changed here.
"""
from __future__ import annotations

import numpy as np

from .corner_scatter import CornerTerms, CornerIncrement
from ._gaussian_kernels import locate_and_prepare_corner_terms


def prepare_cell_corner_increment(spec, centers, sigmas, luminosity, profiles):
    """Locate and prepare covered cells without retaining N-length corner arrays."""
    if luminosity.shape[0] != centers.size:
        raise ValueError("Raw corner coordinates must match the luminosity rows")
    table = spec.table
    prepared = locate_and_prepare_corner_terms(
        centers=centers,
        sigmas=sigmas,
        delta_v=table.delta_v,
        sigma_nodes=table.sigma_nodes,
        phase_counts=table.phase_counts,
        phase_starts=table.phase_starts,
        phase_values=table.phase_axis_values,
        narrow_layers=table.narrow_phase_layers.view(np.uint8),
        transition_intervals=table.phase_transition_intervals,
        transition_radii=table.phase_transition_radii,
        right_starts=table.phase_right_starts,
        inverse_steps=table.phase_inverse_steps,
        minimum_shifts=spec.row_minimum_shifts,
        shift_counts=spec.row_shift_counts,
        row_offsets=spec.row_offsets,
        output_length=spec.slot_count,
    )
    return _freeze_corner_increment(spec, luminosity, profiles, prepared)


def _freeze_corner_increment(spec, luminosity, profiles, prepared):
    """Validate/freeze owned terms and retain only the wave's borrowed luminosity."""
    destination_slots, source_cells, weights, touched, positive_corners = prepared
    terms = CornerTerms(
        destination_slots=destination_slots,
        source_cells=source_cells,
        weights=weights,
        input_length=int(luminosity.shape[0]),
        output_length=spec.slot_count,
    )
    profile_ids = profiles.copy()
    profile_ids.flags.writeable = False
    touched.flags.writeable = False
    borrowed_luminosity = luminosity.view()
    borrowed_luminosity.flags.writeable = False
    increment = CornerIncrement(
        terms=terms,
        luminosity=borrowed_luminosity,
        profiles=profile_ids,
        touched_rows=touched,
        spec=spec,
    )
    metrics = {
        "positive_corner_entries": positive_corners,
        "window_corner_entries": int(terms.destination_slots.size),
        "raw_corner_int32_groups": int(terms.destination_slots.itemsize == 4),
        "raw_corner_int64_groups": int(terms.destination_slots.itemsize == 8),
        "raw_corner_owned_array_bytes": int(
            terms.destination_slots.nbytes + terms.source_cells.nbytes + terms.weights.nbytes
            + touched.nbytes + profile_ids.nbytes
        ),
        "borrowed_luminosity_logical_bytes": int(luminosity.nbytes),
    }
    return increment, metrics
