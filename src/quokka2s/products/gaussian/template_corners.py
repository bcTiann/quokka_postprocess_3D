"""Locate existing template corners and prepare original weighted scatter terms.

Means and widths use km/s, luminosity[cell,profile] uses erg/s. The shared
engine validates cells before these steps. No parameter-grid approximation,
integer-shift rule or per-corner support is changed here.
"""
from __future__ import annotations

import numpy as np

from .corner_scatter import CornerTerms, CornerIncrement


def locate_corners(table, centers, sigmas):
    """Locate covered rows already checked by TemplateEngine.integrate."""
    coordinate = centers / table.delta_v
    integer = np.floor(coordinate)
    phases = coordinate - integer
    shifts = integer.astype(np.int64)
    upper = np.searchsorted(table.sigma_nodes, sigmas, side='left')
    lower = np.maximum(upper - 1, 0)
    lower_log = table.log_sigma_nodes[lower]
    upper_log = table.log_sigma_nodes[upper]
    fraction = np.divide(
        np.log(sigmas) - lower_log,
        upper_log - lower_log,
        out=np.zeros_like(sigmas),
        where=upper != lower,
    )
    np.clip(fraction, 0.0, 1.0, out=fraction)
    low_left, low_right, low_fraction = table._phase_brackets(lower, phases)
    high_left, high_right, high_fraction = table._phase_brackets(upper, phases)
    return shifts, (low_left, low_right, high_left, high_right), (
        (1.0 - fraction) * (1.0 - low_fraction),
        (1.0 - fraction) * low_fraction,
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
    output_first = int(table.output_bin_indices[0])
    output_stop = output_first + table.K
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
        minimum = shifts + table.row_min_bins[ids]
        stop = minimum + table.row_bin_counts[ids]
        retained = positive & (minimum < output_stop) & (stop > output_first)
        retained_rows = ids[retained]
        relative = shifts[retained] - spec.row_minimum_shifts[retained_rows]
        if (
            np.any(relative < 0)
            or np.any(relative >= spec.row_shift_counts[retained_rows])
        ):
            raise ValueError("Raw corner shift is outside its exact support/window overlap")
        slots = spec.row_offsets[retained_rows] + relative
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
