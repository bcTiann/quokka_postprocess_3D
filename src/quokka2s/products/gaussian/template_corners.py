"""Locate existing template corners and combine only identical corner light.

Means and widths use km/s. Covered inputs have already passed the common engine
validation. Keys retain original stored-template row IDs and integer-bin shifts;
no new interpolation grid, cutoff or brightness threshold is introduced.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


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


def combine_corner_light(table, shifts, rows, fractions, luminosity):
    """Sort keys once and directly build interpolation CSR from its row pointers.

    CSR columns need not be sorted for sparse/dense multiplication. Every cell
    contributes at most once to a retained template-row/shift key: nonzero corners
    lie on distinct parameter nodes. No extra column sort/canonicalization is
    needed for the algebraic sum. The original corner-major tie order is stable.
    """
    source = np.arange(shifts.size, dtype=np.int64)
    selected_nodes = []
    selected_shifts = []
    selected_cells = []
    selected_weights = []
    positive_corners = 0
    output_first = int(table.output_bin_indices[0])
    output_stop = output_first + table.K
    for ids, fraction in zip(rows, fractions):
        positive = fraction != 0.0
        positive_corners += int(np.count_nonzero(positive))
        minimum = shifts + table.row_min_bins[ids]
        stop = minimum + table.row_bin_counts[ids]
        retained = positive & (minimum < output_stop) & (stop > output_first)
        selected_nodes.append(ids[retained])
        selected_shifts.append(shifts[retained])
        selected_cells.append(source[retained])
        selected_weights.append(fraction[retained])
    row_ids = np.concatenate(selected_nodes)
    integer_shifts = np.concatenate(selected_shifts)
    cells = np.concatenate(selected_cells)
    weights = np.concatenate(selected_weights)
    metrics = {
        "positive_corner_entries": positive_corners,
        "window_corner_entries": int(row_ids.size),
    }
    if not row_ids.size:
        metrics['unique_corner_keys'] = 0
        return row_ids, integer_shifts, np.empty((0, luminosity.shape[1])), metrics
    first_shift = int(integer_shifts.min())
    span = int(integer_shifts.max()) - first_shift + 1
    if span > np.iinfo(np.int64).max // max(1, table.phase_axis_values.size):
        raise ValueError('Covered compact corner keys exceed int64 encoding capacity')
    encoded = row_ids * span + (integer_shifts - first_shift)
    order = np.argsort(encoded, kind='stable')
    sorted_keys = encoded[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    indptr = np.r_[boundaries, sorted_keys.size]
    unique = sorted_keys[boundaries]
    interpolation = csr_matrix(
        (weights[order], cells[order], indptr),
        shape=(unique.size, luminosity.shape[0]),
    )
    combined = interpolation @ luminosity
    if not np.isfinite(combined).all():
        raise FloatingPointError('Compact corner luminosity accumulation became nonfinite')
    metrics.update({
        "unique_corner_keys": int(unique.size),
        "interpolation_entries": int(interpolation.nnz),
        "interpolation_csr_bytes": int(
            interpolation.data.nbytes
            + interpolation.indices.nbytes
            + interpolation.indptr.nbytes
        ),
        "grouped_luminosity_bytes": int(combined.nbytes),
    })
    return unique // span, unique % span + first_shift, combined, metrics
