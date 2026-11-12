"""Locate original four template corners and contract their stored supports.

Each corner is identified by a prepared phase/log-sigma row and an integer
channel shift. Standard SciPy sparse products combine identical corner light
and expand finite-support rows without materializing cell-by-channel arrays.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def locate_corners(table, centers, sigmas):
    """Return the original four template-row/shift/weight arrays for covered cells."""
    _, sigmas, shifts, phases = table._validate_inputs(centers, sigmas)
    upper_layers = np.searchsorted(table.sigma_nodes, sigmas, side="left")
    lower_layers = np.maximum(upper_layers - 1, 0)
    lower_log = table.log_sigma_nodes[lower_layers]
    upper_log = table.log_sigma_nodes[upper_layers]
    fraction = np.divide(
        np.log(sigmas) - lower_log,
        upper_log - lower_log,
        out=np.zeros_like(sigmas),
        where=upper_layers != lower_layers,
    )
    np.clip(fraction, 0.0, 1.0, out=fraction)
    low_left, low_right, low_fraction = table._phase_brackets(lower_layers, phases)
    high_left, high_right, high_fraction = table._phase_brackets(upper_layers, phases)
    return shifts, (low_left, low_right, high_left, high_right), (
        (1.0 - fraction) * (1.0 - low_fraction),
        (1.0 - fraction) * low_fraction,
        fraction * (1.0 - high_fraction),
        fraction * high_fraction,
    )


def combine_corner_light(table, shifts, rows, fractions, luminosity):
    """Combine only identical (template row, integer shift) corners in this batch.

    A sparse interpolation matrix[unique corner, original cell] multiplies
    luminosity[cell,profile]. It avoids allocating a four-corner luminosity cube.
    Corners with no stored support in the observation window are exact zeros.
    """
    source_rows = np.arange(shifts.size, dtype=np.int64)
    selected_rows = []
    selected_shifts = []
    selected_cells = []
    selected_fractions = []
    positive_corners = 0
    output_first = int(table.output_bin_indices[0])
    output_stop = output_first + table.K
    for node_rows, weights in zip(rows, fractions):
        positive = weights != 0.0
        positive_corners += int(np.count_nonzero(positive))
        minimum = shifts + table.row_min_bins[node_rows]
        stop = minimum + table.row_bin_counts[node_rows]
        retained = positive & (minimum < output_stop) & (stop > output_first)
        selected_rows.append(node_rows[retained])
        selected_shifts.append(shifts[retained])
        selected_cells.append(source_rows[retained])
        selected_fractions.append(weights[retained])
    row_ids = np.concatenate(selected_rows)
    integer_shifts = np.concatenate(selected_shifts)
    cells = np.concatenate(selected_cells)
    weights = np.concatenate(selected_fractions)
    metrics = {
        "positive_corner_entries": positive_corners,
        "window_corner_entries": int(row_ids.size),
        "unique_corner_keys": 0,
        "corner_collisions": 0,
        "sparse_interpolation_entries": int(row_ids.size),
    }
    if row_ids.size == 0:
        return row_ids, integer_shifts, np.empty((0, luminosity.shape[1])), metrics

    # Normal finite-support batches fit this reversible one-int key. The
    # structured fallback preserves arbitrary safe original integer shifts.
    first_shift = int(integer_shifts.min())
    span = int(integer_shifts.max()) - first_shift + 1
    maximum_row = int(table.phase_axis_values.size - 1)
    if span <= np.iinfo(np.int64).max // max(1, maximum_row + 1):
        encoded = row_ids * span + (integer_shifts - first_shift)
        unique, inverse = np.unique(encoded, return_inverse=True)
        unique_rows = unique // span
        unique_shifts = unique % span + first_shift
    else:
        keys = np.column_stack((row_ids, integer_shifts))
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        unique_rows = unique[:, 0]
        unique_shifts = unique[:, 1]
    interpolation = csr_matrix(
        (weights, (inverse, cells)),
        shape=(unique_rows.size, luminosity.shape[0]),
    )
    combined = interpolation @ luminosity
    metrics.update(
        {
            "unique_corner_keys": int(unique_rows.size),
            "corner_collisions": int(row_ids.size - unique_rows.size),
            "sparse_interpolation_entries": int(interpolation.nnz),
            "grouped_luminosity_bytes": int(combined.nbytes),
        }
    )
    return unique_rows, unique_shifts, combined, metrics


def expand_unique_corners(
    table,
    row_ids,
    shifts,
    luminosity,
    *,
    maximum_support_entries=262144,
):
    """Contract stored corner supports with grouped luminosities, in bounded blocks.

    CSR rows are unique corners; columns are physical observation channels.
    Its transpose multiplies all profile accounts without an [entry,profile]
    product buffer. No CDF or new support mask is evaluated here.
    """
    output = np.zeros((table.K, luminosity.shape[1]))
    metrics = {
        "support_entries_gathered": 0,
        "nonzero_support_entries": 0,
        "support_blocks": 0,
        "logical_product_updates": 0,
    }
    if row_ids.size == 0:
        return output, metrics
    output_first = int(table.output_bin_indices[0])
    first = np.maximum(0, shifts + table.row_min_bins[row_ids] - output_first)
    stop = np.minimum(
        table.K,
        shifts + table.row_min_bins[row_ids]
        + table.row_bin_counts[row_ids] - output_first,
    )
    lengths = np.maximum(0, stop - first)
    cumulative = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)))
    begin = 0
    while begin < row_ids.size:
        target = int(cumulative[begin]) + maximum_support_entries
        end = max(begin + 1, int(np.searchsorted(cumulative, target, side="right") - 1))
        end = min(end, row_ids.size)
        counts = lengths[begin:end]
        offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
        owners = np.repeat(np.arange(end - begin), counts)
        relative = np.arange(int(offsets[-1])) - np.repeat(offsets[:-1], counts)
        channels = first[begin:end][owners] + relative
        rows = row_ids[begin:end][owners]
        stored_bins = (
            output_first + channels
            - shifts[begin:end][owners] - table.row_min_bins[rows]
        )
        values = table.flat_values[table.row_value_offsets[rows] + stored_bins]
        retained = values != 0.0
        retained_counts = np.bincount(owners[retained], minlength=end - begin)
        indptr = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(retained_counts)))
        support = csr_matrix(
            (values[retained], channels[retained], indptr),
            shape=(end - begin, table.K),
        )
        output += support.T @ luminosity[begin:end]
        updates = int(support.nnz * luminosity.shape[1])
        metrics["support_entries_gathered"] += int(values.size)
        metrics["nonzero_support_entries"] += int(support.nnz)
        metrics["support_blocks"] += 1
        metrics["logical_product_updates"] += updates
        begin = end
    return output, metrics
