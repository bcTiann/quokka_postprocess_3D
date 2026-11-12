"""Locate original template corners and sum only identical corner luminosity.

Covered means/widths use km/s and luminosity[cell,profile] uses erg/s. The common
engine validates those arrays before calling these helpers. Coordinate labels
may be compressed exactly for implementation work/memory limits; the original
phase/log-sigma nodes, four weights, finite support and brightness are unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

# These limits choose a summation implementation, not Gaussian coverage.
MAXIMUM_HISTOGRAM_BINS = 2**22
MAXIMUM_BINS_PER_TERM = 4.0
MAXIMUM_AXIS_BINS = 262144
MAXIMUM_HISTOGRAM_ACCOUNTS = 4


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
    """Combine exact existing corner terms without changing nodes or weights.

    Up to four accounts use bounded dense histograms when the compressed axes
    are compact enough. More accounts and wide/sparse ranges use stable CSR
    reduction; bounded integer sort keys may use uint32. All returned node
    coordinates remain int64, and weights/luminosities remain float64.
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
    # Coordinate compression is exact: observed axes retain their original order.
    account_count_allows_histogram = luminosity.shape[1] <= MAXIMUM_HISTOGRAM_ACCOUNTS
    maximum_row = int(row_ids.max()) if account_count_allows_histogram else 0
    can_compress = (
        account_count_allows_histogram
        and maximum_row + 1 <= MAXIMUM_AXIS_BINS
        and span <= MAXIMUM_AXIS_BINS
    )
    observed_rows = None
    observed_shifts = None
    code_span = span
    if can_compress:
        row_counts = np.bincount(row_ids, minlength=maximum_row + 1)
        observed_rows = np.flatnonzero(row_counts)
        del row_counts
        row_map = np.empty(maximum_row + 1, dtype=np.int64)
        row_map[observed_rows] = np.arange(observed_rows.size)
        shift_offsets = integer_shifts - first_shift
        shift_counts = np.bincount(shift_offsets, minlength=span)
        observed_offsets = np.flatnonzero(shift_counts)
        del shift_counts
        shift_map = np.empty(span, dtype=np.int64)
        shift_map[observed_offsets] = np.arange(observed_offsets.size)
        observed_shifts = observed_offsets + first_shift
        code_span = observed_shifts.size
        encoded = row_map[row_ids] * code_span + shift_map[shift_offsets]
        del row_map, shift_map, shift_offsets, observed_offsets
    else:
        encoded = row_ids * span + (integer_shifts - first_shift)
    base = int(encoded.min())
    maximum_key = int(encoded.max())
    histogram_bins = maximum_key - base + 1
    use_histogram = (
        account_count_allows_histogram
        and histogram_bins <= MAXIMUM_HISTOGRAM_BINS
        and histogram_bins <= MAXIMUM_BINS_PER_TERM * encoded.size
    )
    sort_key_itemsize = 0
    interpolation_bytes = 0
    if use_histogram:
        relative = encoded - base
        counts = np.bincount(relative, minlength=histogram_bins)
        occupied = np.flatnonzero(counts)
        del counts
        unique = occupied + base
        combined = np.empty((unique.size, luminosity.shape[1]))
        for profile in range(luminosity.shape[1]):
            weighted = np.asarray(luminosity[cells, profile], dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                weighted *= weights
            histogram = np.bincount(relative, weights=weighted, minlength=histogram_bins)
            combined[:, profile] = histogram[occupied]
            del histogram, weighted
    else:
        # More accounts use contiguous CSR row updates instead of P histograms.
        if base >= 0 and maximum_key <= np.iinfo(np.uint32).max:
            encoded = encoded.astype(np.uint32)
        sort_key_itemsize = int(encoded.dtype.itemsize)
        order = np.argsort(encoded, kind="stable")
        sorted_keys = encoded[order]
        boundaries = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        indptr = np.r_[boundaries, sorted_keys.size]
        unique = sorted_keys[boundaries].astype(np.int64, copy=False)
        interpolation = csr_matrix(
            (weights[order], cells[order], indptr),
            shape=(unique.size, luminosity.shape[0]),
        )
        combined = interpolation @ luminosity
        interpolation_bytes = int(
            interpolation.data.nbytes
            + interpolation.indices.nbytes
            + interpolation.indptr.nbytes
        )
    if not np.isfinite(combined).all():
        raise FloatingPointError("Compact corner luminosity accumulation became nonfinite")
    row_codes = unique // code_span
    shift_codes = unique % code_span
    unique_rows = row_codes if observed_rows is None else observed_rows[row_codes]
    unique_shifts = (
        shift_codes + first_shift
        if observed_shifts is None
        else observed_shifts[shift_codes]
    )
    metrics.update(
        {
            "unique_corner_keys": int(unique.size),
            "grouped_luminosity_bytes": int(combined.nbytes),
            "bincount_groups": int(use_histogram),
            "csr_groups": int(not use_histogram),
            "uint32_sort_groups": int(sort_key_itemsize == 4),
            "bincount_terms_reduced": int(encoded.size) if use_histogram else 0,
            "bincount_bins_allocated": (
                int((luminosity.shape[1] + 1) * histogram_bins)
                if use_histogram else 0
            ),
            "csr_interpolation_bytes_allocated": interpolation_bytes,
        }
    )
    return unique_rows, unique_shifts, combined, metrics
