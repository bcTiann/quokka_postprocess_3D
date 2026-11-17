"""Locate existing template corners and prepare original weighted scatter terms.

Means and widths use km/s, luminosity[cell,profile] uses erg/s. The shared
engine validates cells before these steps. No parameter-grid approximation,
integer-shift rule or per-corner support is changed here.
"""
from __future__ import annotations

import numpy as np

from .corner_scatter import CornerTerms, CornerIncrement


def locate_corners(table, centers, sigmas):
    """Locate covered rows already validated by worker group preparation."""
    coordinate = centers if table.delta_v == 1.0 else centers / table.delta_v
    integer = np.floor(coordinate)
    phases = coordinate - integer
    shifts = integer.astype(np.int64)
    # Sigma layers are log-spaced except for the final endpoint. Estimate the
    # upper layer in one NumPy pass, then compare the actual stored knots so
    # values at/next to a knot keep searchsorted(side='left') semantics.
    upper = np.ceil(
        (np.log(sigmas) - table.log_sigma_nodes[0]) / table.parameters.log_sigma_step
    ).astype(np.int64)
    np.clip(upper, 0, table.sigma_nodes.size - 1, out=upper)
    upper -= (upper > 0) & (sigmas <= table.sigma_nodes[np.maximum(upper - 1, 0)])
    upper += sigmas > table.sigma_nodes[upper]
    # Extremely fine user grids can magnify logarithm rounding by more than
    # one layer. Correct only such rows with the reference binary search.
    inconsistent = (sigmas > table.sigma_nodes[upper]) | (
        (upper > 0) & (sigmas <= table.sigma_nodes[np.maximum(upper - 1, 0)])
    )
    if np.any(inconsistent):
        upper[inconsistent] = np.searchsorted(table.sigma_nodes, sigmas[inconsistent], side='left')
    lower = np.maximum(upper - 1, 0)
    lower_sigma = table.sigma_nodes[lower]
    upper_sigma = table.sigma_nodes[upper]
    fraction = sigmas - lower_sigma
    np.divide(
        fraction,
        upper_sigma - lower_sigma,
        out=fraction,
        where=upper != lower,
    )
    np.clip(fraction, 0.0, 1.0, out=fraction)
    low_left, low_right, low_fraction = table._phase_brackets(lower, phases)
    matching_phase_grids = (
        (table.phase_counts[lower] == table.phase_counts[upper])
        & ~table.narrow_phase_layers[lower]
        & ~table.narrow_phase_layers[upper]
    )
    if np.all(matching_phase_grids):
        # Wide adjacent layers often use the identical [0,1] knot vector.
        # Only their flat row offsets differ; repeat no phase search or divide.
        row_offset = table.phase_starts[upper] - table.phase_starts[lower]
        high_left = low_left + row_offset
        high_right = low_right + row_offset
        high_fraction = low_fraction
    elif np.count_nonzero(matching_phase_grids) > matching_phase_grids.size // 2:
        # A few narrow rows must not repeat the wide phase search for a whole
        # group. Search only the exceptions when most phase grids are shared.
        row_offset = table.phase_starts[upper] - table.phase_starts[lower]
        high_left = low_left + row_offset
        high_right = low_right + row_offset
        high_fraction = low_fraction.copy()
        selected = ~matching_phase_grids
        high_left[selected], high_right[selected], high_fraction[selected] = table._phase_brackets(
            upper[selected], phases[selected],
        )
    else:
        high_left, high_right, high_fraction = table._phase_brackets(upper, phases)
    lower_fraction = 1.0 - fraction
    return shifts, (low_left, low_right, high_left, high_right), (
        lower_fraction * (1.0 - low_fraction),
        lower_fraction * low_fraction,
        fraction * (1.0 - high_fraction),
        fraction * high_fraction,
    )


def prepare_corner_increment(
    spec, shifts, rows, fractions, luminosity, profiles,
):
    """Construct unsorted terms from four template rows per selected cell.

    rows and fractions contain four vectors[N]; shifts[N] are integer channels.
    Each term maps a local luminosity row to the flat destination slot for its
    template row plus shift. Only the existing corner/window filter is applied.
    """
    table = spec.table
    largest_index = max(spec.slot_count, luminosity.shape[0])
    index_dtype = np.int32 if largest_index <= np.iinfo(np.int32).max else np.int64
    source = np.arange(shifts.size, dtype=index_dtype)
    selected_slots = []
    selected_cells = []
    selected_weights = []
    touched = np.zeros(table.row_min_bins.size, dtype=bool)
    positive_corners = 0
    for ids, fraction in zip(rows, fractions):
        positive = fraction != 0.0
        positive_corners += int(np.count_nonzero(positive))
        # The prepared row already stores exactly the valid integer shifts.
        # Testing this relative index is the same window-overlap filter; it
        # avoids recomputing the bin endpoints and checking the bounds twice.
        relative = shifts - spec.row_minimum_shifts[ids]
        retained = positive & (relative >= 0) & (relative < spec.row_shift_counts[ids])
        retained_rows = ids[retained]
        slots = spec.row_offsets[retained_rows] + relative[retained]
        selected_slots.append(slots.astype(index_dtype, copy=False))
        selected_cells.append(source[retained])
        selected_weights.append(fraction[retained])
        touched[retained_rows] = True
    terms = CornerTerms(
        destination_slots=np.concatenate(selected_slots),
        source_cells=np.concatenate(selected_cells),
        weights=np.concatenate(selected_weights),
        input_length=int(luminosity.shape[0]),
        output_length=spec.slot_count,
    )
    if not np.isfinite(terms.weights).all():
        raise FloatingPointError("Raw corner weights must remain finite")
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
