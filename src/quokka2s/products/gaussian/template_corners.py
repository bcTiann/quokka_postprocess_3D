"""Private nearest stored-node approximation, with one COO term per covered cell.

Sigma distance is measured in linear units between the stored log-spaced nodes.
Phase distance uses actual knots of that selected layer. Exact ties choose the
lower sigma/phase. Original-cell accounts and outside-width integrals live in
the unchanged engine; this file only changes covered template-node selection.
"""
from __future__ import annotations

import numpy as np

from .corner_scatter import CornerTerms, CornerIncrement


def _nearest_phase_row(table, layer_ids, phases):
    """Find actual phase neighbours once, then retain only the nearer row."""
    counts = table.phase_counts[layer_ids]
    local = np.floor(phases * (counts - 1)).astype(np.int64)
    narrow = table.narrow_phase_layers[layer_ids]
    if np.any(narrow):
        selected_layers = layer_ids[narrow]
        selected_phases = phases[narrow]
        middle = table.phase_transition_intervals[selected_layers]
        radius = table.phase_transition_radii[selected_layers]
        right_start = table.phase_right_starts[selected_layers]
        inverse_step = table.phase_inverse_steps[selected_layers]
        selected_local = np.floor(selected_phases * inverse_step).astype(np.int64)
        gap = (selected_phases > radius) & (selected_phases < right_start)
        selected_local[gap] = middle[gap]
        right = selected_phases >= right_start
        selected_local[right] = (
            middle[right] + 1
            + np.floor(
                (selected_phases[right] - right_start[right]) * inverse_step[right]
            ).astype(np.int64)
        )
        local[narrow] = selected_local
    np.maximum(local, 0, out=local)
    np.minimum(local, counts - 1, out=local)
    terminal = phases >= 1.0
    local[terminal] = counts[terminal] - 1
    lower = table.phase_starts[layer_ids] + local
    lower -= phases < table.phase_axis_values[lower]
    upper = np.where(terminal, lower, lower + 1)
    too_low = (~terminal) & (phases >= table.phase_axis_values[upper])
    lower += too_low
    upper += too_low
    choose_upper = (
        phases - table.phase_axis_values[lower]
        > table.phase_axis_values[upper] - phases
    )
    return np.where(choose_upper, upper, lower)


def locate_nearest_node(table, centers, sigmas):
    """Choose one linear-sigma layer and one of its actual phase knots.

    Inputs are covered, validated original Gaussian means/widths [km/s]. A
    selected phase1 is represented as phase0 with shift+1, including negative
    means: mu=-0.02 -> floor=-1, phase=.98 -> canonical mu=0 when phase1 wins.
    No four-node interpolation fractions or four-corner vectors are generated.
    """
    coordinate = centers if table.delta_v == 1.0 else centers / table.delta_v
    integer = np.floor(coordinate)
    phases = coordinate - integer
    shifts = integer.astype(np.int64)
    upper = np.ceil(
        (np.log(sigmas) - table.log_sigma_nodes[0]) / table.parameters.log_sigma_step
    ).astype(np.int64)
    np.clip(upper, 0, table.sigma_nodes.size - 1, out=upper)
    upper -= (upper > 0) & (sigmas <= table.sigma_nodes[np.maximum(upper - 1, 0)])
    upper += sigmas > table.sigma_nodes[upper]
    inconsistent = (sigmas > table.sigma_nodes[upper]) | (
        (upper > 0) & (sigmas <= table.sigma_nodes[np.maximum(upper - 1, 0)])
    )
    if np.any(inconsistent):
        upper[inconsistent] = np.searchsorted(table.sigma_nodes, sigmas[inconsistent], side='left')
    lower = np.maximum(upper - 1, 0)
    choose_upper = sigmas - table.sigma_nodes[lower] > table.sigma_nodes[upper] - sigmas
    layers = np.where(choose_upper, upper, lower)
    rows = _nearest_phase_row(table, layers, phases)
    carry = table.phase_axis_values[rows] == 1.0
    shifts += carry
    rows[carry] = table.phase_starts[layers[carry]]
    return shifts, rows


def prepare_nearest_increment(spec, shifts, rows, luminosity, profiles):
    """Build one unit-weight term per cell whose chosen node overlaps the window."""
    table = spec.table
    largest_index = max(spec.slot_count, luminosity.shape[0])
    index_dtype = np.int32 if largest_index <= np.iinfo(np.int32).max else np.int64
    relative = shifts - spec.row_minimum_shifts[rows]
    retained = (relative >= 0) & (relative < spec.row_shift_counts[rows])
    retained_rows = rows[retained]
    slots = spec.row_offsets[retained_rows] + relative[retained]
    source = np.arange(shifts.size, dtype=index_dtype)
    terms = CornerTerms(
        destination_slots=slots.astype(index_dtype, copy=False),
        source_cells=source[retained],
        weights=np.ones(slots.size, dtype=np.float64),
        input_length=int(luminosity.shape[0]),
        output_length=spec.slot_count,
    )
    touched = np.zeros(table.row_min_bins.size, dtype=bool)
    touched[retained_rows] = True
    touched.flags.writeable = False
    profile_ids = profiles.copy()
    profile_ids.flags.writeable = False
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
        "positive_corner_entries": int(shifts.size),
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
