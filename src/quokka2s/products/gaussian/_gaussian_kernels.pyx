# cython: language_level=3
"""Typed Gaussian loops over Python-owned, runtime-configured template arrays.

Inputs are read-only; each call returns fresh NumPy-owned outputs. Python
validates immutable table metadata before these bounded worker calls. The
loops release the GIL without changing the template grid or reduction order.
"""
import numpy as np
cimport cython
from libc.math cimport exp, expm1, fabs, floor, isfinite
from libc.stdint cimport int32_t, int64_t, uint8_t, uint64_t
from scipy.special.cython_special cimport erf, log_ndtr


ctypedef fused corner_index_t:
    int32_t
    int64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int corner_destination(
    int64_t shift,
    int64_t row,
    double weight,
    const int64_t[::1] minimum_shifts,
    const int64_t[::1] shift_counts,
    const int64_t[::1] row_offsets,
    Py_ssize_t output_length,
    int64_t* destination,
) noexcept nogil:
    """Check one original corner and its row-specific observed-window overlap."""
    cdef int64_t minimum
    cdef int64_t span
    cdef int64_t offset
    cdef uint64_t relative
    if row < 0 or row >= minimum_shifts.shape[0]:
        return -1
    if weight == 0.0:
        return 0
    minimum = minimum_shifts[row]
    span = shift_counts[row]
    if span < 0:
        return -2
    if shift < minimum:
        return 0
    # Unsigned subtraction avoids signed overflow for rejected far-away shifts.
    relative = <uint64_t>shift - <uint64_t>minimum
    if relative >= <uint64_t>span:
        return 0
    offset = row_offsets[row]
    if offset < 0 or offset >= output_length or relative >= <uint64_t>(output_length - offset):
        return -3
    if not isfinite(weight):
        return -4
    destination[0] = offset + <int64_t>relative
    return 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int phase_bracket(
    Py_ssize_t layer,
    double phase,
    const int64_t[::1] phase_counts,
    const int64_t[::1] phase_starts,
    const double[::1] phase_values,
    const uint8_t[::1] narrow_layers,
    const int64_t[::1] transition_intervals,
    const double[::1] transition_radii,
    const double[::1] right_starts,
    const double[::1] inverse_steps,
    int64_t* left,
    int64_t* right,
    double* fraction,
) noexcept nogil:
    """Use the existing piecewise-uniform search and actual stored knots."""
    cdef int64_t count = phase_counts[layer]
    cdef int64_t local
    cdef int64_t first = phase_starts[layer]
    cdef int64_t stop = phase_starts[layer + 1]
    cdef int64_t middle
    cdef double denominator
    # Validate this segment before arithmetic or any phase-knot dereference.
    if first < 0 or stop > phase_values.shape[0] or first >= stop:
        return 1
    if count < 2 or count != stop - first:
        return 1
    local = <int64_t>floor(phase * (count - 1))
    if narrow_layers[layer]:
        middle = transition_intervals[layer]
        if phase > transition_radii[layer] and phase < right_starts[layer]:
            local = middle
        elif phase >= right_starts[layer]:
            local = middle + 1 + <int64_t>floor(
                (phase - right_starts[layer]) * inverse_steps[layer]
            )
        else:
            local = <int64_t>floor(phase * inverse_steps[layer])
    if local < 0:
        local = 0
    elif local >= count:
        local = count - 1
    if phase >= 1.0:
        local = count - 1
    left[0] = first + local
    if left[0] < first or left[0] >= stop:
        return 1
    if phase < phase_values[left[0]]:
        left[0] -= 1
    if phase >= 1.0:
        right[0] = left[0]
    else:
        right[0] = left[0] + 1
        if right[0] >= stop:
            return 1
        if phase >= phase_values[right[0]]:
            left[0] += 1
            right[0] += 1
    if left[0] < first or right[0] >= stop:
        return 1
    denominator = phase_values[right[0]] - phase_values[left[0]]
    fraction[0] = (phase - phase_values[left[0]]) / denominator if denominator != 0.0 else 0.0
    if fraction[0] < -1e-12 or fraction[0] > 1.0 + 1e-12:
        return 1
    if fraction[0] < 0.0:
        fraction[0] = 0.0
    elif fraction[0] > 1.0:
        fraction[0] = 1.0
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int locate_cell(
    double center,
    double sigma,
    double delta_v,
    const double[::1] sigma_nodes,
    const int64_t[::1] phase_counts,
    const int64_t[::1] phase_starts,
    const double[::1] phase_values,
    const uint8_t[::1] narrow_layers,
    const int64_t[::1] transition_intervals,
    const double[::1] transition_radii,
    const double[::1] right_starts,
    const double[::1] inverse_steps,
    int64_t* shift,
    int64_t* rows,
    double* weights,
) noexcept nogil:
    """Locate one Gaussian for fused corner preparation."""
    cdef Py_ssize_t layers = sigma_nodes.shape[0]
    cdef Py_ssize_t first = 0
    cdef Py_ssize_t stop = layers
    cdef Py_ssize_t middle
    cdef Py_ssize_t lower
    cdef Py_ssize_t upper
    cdef double coordinate = center / delta_v
    cdef double integer
    cdef double phase
    cdef double sigma_fraction
    cdef double lower_sigma_fraction
    cdef double low_fraction
    cdef double high_fraction
    if not isfinite(coordinate) or coordinate <= -4611686018427387904.0 or coordinate >= 4611686018427387904.0:
        return 1
    if not isfinite(sigma) or sigma < sigma_nodes[0] or sigma > sigma_nodes[layers - 1]:
        return 2
    integer = floor(coordinate)
    phase = coordinate - integer
    shift[0] = <int64_t>integer
    while first < stop:
        middle = first + (stop - first) // 2
        if sigma_nodes[middle] < sigma:
            first = middle + 1
        else:
            stop = middle
    upper = first
    lower = upper - 1 if upper > 0 else 0
    sigma_fraction = (
        (sigma - sigma_nodes[lower]) / (sigma_nodes[upper] - sigma_nodes[lower])
        if upper != lower else 0.0
    )
    if sigma_fraction < 0.0:
        sigma_fraction = 0.0
    elif sigma_fraction > 1.0:
        sigma_fraction = 1.0
    if phase_bracket(lower, phase, phase_counts, phase_starts, phase_values,
                     narrow_layers, transition_intervals, transition_radii,
                     right_starts, inverse_steps, &rows[0], &rows[1], &low_fraction):
        return 3
    if not narrow_layers[lower] and not narrow_layers[upper] and phase_counts[lower] == phase_counts[upper]:
        rows[2] = rows[0] + phase_starts[upper] - phase_starts[lower]
        rows[3] = rows[1] + phase_starts[upper] - phase_starts[lower]
        high_fraction = low_fraction
    elif phase_bracket(upper, phase, phase_counts, phase_starts, phase_values,
                       narrow_layers, transition_intervals, transition_radii,
                       right_starts, inverse_steps, &rows[2], &rows[3], &high_fraction):
        return 3
    lower_sigma_fraction = 1.0 - sigma_fraction
    weights[0] = lower_sigma_fraction * (1.0 - low_fraction)
    weights[1] = lower_sigma_fraction * low_fraction
    weights[2] = sigma_fraction * (1.0 - high_fraction)
    weights[3] = sigma_fraction * high_fraction
    return 0


cdef void validate_location_grid(
    Py_ssize_t count,
    Py_ssize_t sigma_count,
    double delta_v,
    const double[::1] sigma_nodes,
    const int64_t[::1] phase_counts,
    const int64_t[::1] phase_starts,
    const double[::1] phase_values,
    const uint8_t[::1] narrow_layers,
    const int64_t[::1] transition_intervals,
    const double[::1] transition_radii,
    const double[::1] right_starts,
    const double[::1] inverse_steps,
) except *:
    """Validate buffer sizes before fused preparation uses unchecked grid pointers."""
    cdef Py_ssize_t layers = sigma_nodes.shape[0]
    if sigma_count != count or layers < 1:
        raise ValueError("Gaussian coordinates and sigma nodes have invalid shapes")
    if (
        phase_counts.shape[0] != layers or phase_starts.shape[0] != layers + 1
        or narrow_layers.shape[0] != layers or transition_intervals.shape[0] != layers
        or transition_radii.shape[0] != layers or right_starts.shape[0] != layers
        or inverse_steps.shape[0] != layers
    ):
        raise ValueError("Gaussian phase metadata must match the sigma layers")
    if phase_starts[0] != 0 or phase_starts[layers] != phase_values.shape[0]:
        raise ValueError("Gaussian phase starts must cover the stored phase knots")
    if not isfinite(delta_v) or delta_v <= 0.0:
        raise ValueError("Gaussian channel width must be finite and positive")


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int locate_and_fill_terms(
    const double[::1] centers,
    const double[::1] sigmas,
    double delta_v,
    const double[::1] sigma_nodes,
    const int64_t[::1] phase_counts,
    const int64_t[::1] phase_starts,
    const double[::1] phase_values,
    const uint8_t[::1] narrow_layers,
    const int64_t[::1] transition_intervals,
    const double[::1] transition_radii,
    const double[::1] right_starts,
    const double[::1] inverse_steps,
    const int64_t[::1] minimum_shifts,
    const int64_t[::1] shift_counts,
    const int64_t[::1] row_offsets,
    Py_ssize_t output_length,
    corner_index_t[::1] destinations,
    corner_index_t[::1] sources,
    double[::1] weights,
    uint8_t[::1] touched,
    Py_ssize_t* retained,
    Py_ssize_t* nonzero,
) noexcept nogil:
    """Locate once per cell and write corner-major capacity, then compact stably."""
    cdef Py_ssize_t count = centers.shape[0]
    cdef Py_ssize_t cell
    cdef Py_ssize_t corner
    cdef Py_ssize_t position
    cdef Py_ssize_t read_position
    cdef Py_ssize_t write_position = 0
    cdef int64_t shift
    cdef int64_t cell_rows[4]
    cdef double cell_weights[4]
    cdef int64_t destination
    cdef int status
    retained[0] = 0
    nonzero[0] = 0
    for cell in range(count):
        status = locate_cell(
            centers[cell], sigmas[cell], delta_v, sigma_nodes, phase_counts,
            phase_starts, phase_values, narrow_layers, transition_intervals,
            transition_radii, right_starts, inverse_steps,
            &shift, cell_rows, cell_weights,
        )
        if status:
            return status
        for corner in range(4):
            position = corner * count + cell
            destinations[position] = -1
            if cell_weights[corner] != 0.0:
                nonzero[0] += 1
            status = corner_destination(
                shift, cell_rows[corner], cell_weights[corner], minimum_shifts,
                shift_counts, row_offsets, output_length, &destination,
            )
            if status < 0:
                return status
            if status:
                destinations[position] = <corner_index_t>destination
                sources[position] = <corner_index_t>cell
                weights[position] = cell_weights[corner]
                touched[cell_rows[corner]] = 1
                retained[0] += 1
    if 0 < retained[0] < destinations.shape[0]:
        # Write positions never exceed read positions, preserving every
        # original corner/cell entry without an extra buffer or a sort.
        for read_position in range(destinations.shape[0]):
            if destinations[read_position] >= 0:
                if write_position != read_position:
                    destinations[write_position] = destinations[read_position]
                    sources[write_position] = sources[read_position]
                    weights[write_position] = weights[read_position]
                write_position += 1
        if write_position != retained[0]:
            return -5
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def locate_and_prepare_corner_terms(
    const double[::1] centers,
    const double[::1] sigmas,
    double delta_v,
    const double[::1] sigma_nodes,
    const int64_t[::1] phase_counts,
    const int64_t[::1] phase_starts,
    const double[::1] phase_values,
    const uint8_t[::1] narrow_layers,
    const int64_t[::1] transition_intervals,
    const double[::1] transition_radii,
    const double[::1] right_starts,
    const double[::1] inverse_steps,
    const int64_t[::1] minimum_shifts,
    const int64_t[::1] shift_counts,
    const int64_t[::1] row_offsets,
    Py_ssize_t output_length,
):
    """Return exact owned COO arrays without nine full-length location vectors.

    Three 4*N-capacity arrays receive corner-major entries. Invalid slots use
    -1, and a stable in-place compaction retains the original COO order. All
    C memoryviews are released before resizing the independent owners to E.
    """
    cdef Py_ssize_t count = centers.shape[0]
    cdef Py_ssize_t row_count = phase_values.shape[0]
    validate_location_grid(
        count, sigmas.shape[0], delta_v, sigma_nodes, phase_counts, phase_starts,
        phase_values, narrow_layers, transition_intervals, transition_radii,
        right_starts, inverse_steps,
    )
    if minimum_shifts.shape[0] != row_count or shift_counts.shape[0] != row_count or row_offsets.shape[0] != row_count + 1:
        raise ValueError("Raw corner shift metadata has incompatible shapes")
    if output_length < 0 or row_offsets[0] != 0 or row_offsets[row_count] != output_length:
        raise ValueError("Raw corner offsets must span the prepared output")
    if count > np.iinfo(np.intp).max // 4:
        raise OverflowError("Four corner vectors exceed platform allocation limits")
    cdef Py_ssize_t capacity = 4 * count
    cdef Py_ssize_t retained = 0
    cdef Py_ssize_t nonzero = 0
    cdef int error
    cdef bint use_int32 = max(output_length, count) <= np.iinfo(np.int32).max
    index_dtype = np.int32 if use_int32 else np.int64
    destination_slots = np.empty(capacity, dtype=index_dtype)
    source_cells = np.empty(capacity, dtype=index_dtype)
    term_weights = np.empty(capacity, dtype=np.float64)
    touched_rows = np.zeros(row_count, dtype=np.bool_)
    cdef double[::1] output_weights = term_weights
    cdef uint8_t[::1] touched = touched_rows.view(np.uint8)
    cdef int32_t[::1] destinations32
    cdef int32_t[::1] sources32
    cdef int64_t[::1] destinations64
    cdef int64_t[::1] sources64
    if use_int32:
        destinations32 = destination_slots
        sources32 = source_cells
        with nogil:
            error = locate_and_fill_terms[int32_t](
                centers, sigmas, delta_v, sigma_nodes, phase_counts, phase_starts,
                phase_values, narrow_layers, transition_intervals, transition_radii,
                right_starts, inverse_steps, minimum_shifts, shift_counts,
                row_offsets, output_length, destinations32, sources32,
                output_weights, touched, &retained, &nonzero,
            )
    else:
        destinations64 = destination_slots
        sources64 = source_cells
        with nogil:
            error = locate_and_fill_terms[int64_t](
                centers, sigmas, delta_v, sigma_nodes, phase_counts, phase_starts,
                phase_values, narrow_layers, transition_intervals, transition_radii,
                right_starts, inverse_steps, minimum_shifts, shift_counts,
                row_offsets, output_length, destinations64, sources64,
                output_weights, touched, &retained, &nonzero,
            )
    if error == 1:
        raise ValueError("Gaussian integer-bin shifts must fit safe signed int64 coordinates")
    if error == 2:
        raise ValueError("Corner location requires valid template-covered sigma values")
    if error == 3:
        raise AssertionError("Arithmetic phase search failed to bracket an input")
    if error in (-1, -2, -3):
        raise ValueError("A raw corner row, shift span or destination is outside its prepared bounds")
    if error == -4:
        raise FloatingPointError("Raw corner weights must remain finite")
    if error:
        raise RuntimeError("Raw corner compaction did not retain the counted entries")
    destinations32 = None
    sources32 = None
    destinations64 = None
    sources64 = None
    output_weights = None
    if retained != capacity:
        destination_slots.resize(retained, refcheck=False)
        source_cells.resize(retained, refcheck=False)
        term_weights.resize(retained, refcheck=False)
    return destination_slots, source_cells, term_weights, touched_rows, nonzero


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double stable_interval(
    double lower,
    double upper,
    const double[::1] quadrature_nodes,
    const double[::1] quadrature_weights,
    double sqrt_two,
    double sqrt_two_pi,
    double narrow_threshold,
) noexcept nogil:
    """Keep the NumPy reference's tiny-interval, tail and crossing formulas."""
    cdef double width
    cdef double midpoint
    cdef double half_width
    cdef double point
    cdef double total
    cdef double log_large
    cdef double log_small
    cdef Py_ssize_t index
    if lower >= upper:
        return 0.0
    width = upper - lower
    midpoint = 0.5 * lower + 0.5 * upper
    if isfinite(midpoint) and width * (1.0 + fabs(midpoint)) < narrow_threshold:
        half_width = 0.5 * width
        total = 0.0
        for index in range(quadrature_nodes.shape[0]):
            point = midpoint + half_width * quadrature_nodes[index]
            total += (exp(-0.5 * point * point) / sqrt_two_pi) * quadrature_weights[index]
        return half_width * total
    if upper <= 0.0:
        log_large = log_ndtr(upper)
        log_small = log_ndtr(lower)
        return exp(log_large) * -expm1(log_small - log_large)
    if lower >= 0.0:
        log_large = log_ndtr(-lower)
        log_small = log_ndtr(-upper)
        return exp(log_large) * -expm1(log_small - log_large)
    return 0.5 * (erf(upper / sqrt_two) - erf(lower / sqrt_two))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def original_window_probabilities(
    const double[::1] centers,
    const double[::1] sigmas,
    double lower_edge,
    double upper_edge,
    double cutoff_sigma,
    double full_saturation,
    double clipped_interior_mass,
    const double[::1] quadrature_nodes,
    const double[::1] quadrature_weights,
    double sqrt_two,
    double sqrt_two_pi,
    double narrow_threshold,
):
    """Return independent original-cell full/clipped window probabilities [N].

    Means, widths and edges share km/s. This does not use template corners or
    luminosity accounts; NumPy retains the subsequent account reduction.
    """
    cdef Py_ssize_t count = centers.shape[0]
    if sigmas.shape[0] != count:
        raise ValueError("Window means and sigma must have matching lengths")
    if quadrature_nodes.shape[0] != quadrature_weights.shape[0] or quadrature_nodes.shape[0] == 0:
        raise ValueError("Stable interval quadrature requires matching nonempty vectors")
    if not isfinite(lower_edge) or not isfinite(upper_edge) or lower_edge >= upper_edge:
        raise ValueError("Observed window edges must be finite and increasing")
    if not isfinite(cutoff_sigma) or cutoff_sigma <= 0.0:
        raise ValueError("Gaussian cutoff must be finite and positive")
    if not isfinite(full_saturation) or full_saturation <= 0.0:
        raise ValueError("Full-window saturation must be finite and positive")
    if not isfinite(clipped_interior_mass) or not 0.0 <= clipped_interior_mass <= 1.0:
        raise ValueError("Clipped interior probability must be finite and within [0,1]")
    if not isfinite(sqrt_two) or sqrt_two <= 0.0 or not isfinite(sqrt_two_pi) or sqrt_two_pi <= 0.0:
        raise ValueError("Gaussian normalization factors must be finite and positive")
    if not isfinite(narrow_threshold) or narrow_threshold <= 0.0:
        raise ValueError("Narrow-interval threshold must be finite and positive")
    full = np.empty(count, dtype=np.float64)
    clipped = np.empty(count, dtype=np.float64)
    cdef double[::1] full_values = full
    cdef double[::1] clipped_values = clipped
    cdef double lower
    cdef double upper
    cdef double clipped_lower
    cdef double clipped_upper
    cdef Py_ssize_t index
    cdef bint invalid = False
    with nogil:
        for index in range(count):
            if not isfinite(centers[index]) or not isfinite(sigmas[index]) or sigmas[index] <= 0.0:
                invalid = True
                break
            lower = (lower_edge - centers[index]) / sigmas[index]
            upper = (upper_edge - centers[index]) / sigmas[index]
            if lower <= -full_saturation and upper >= full_saturation:
                full_values[index] = 1.0
            else:
                full_values[index] = stable_interval(
                    lower, upper, quadrature_nodes, quadrature_weights,
                    sqrt_two, sqrt_two_pi, narrow_threshold,
                )
            if lower <= -cutoff_sigma and upper >= cutoff_sigma:
                clipped_values[index] = clipped_interior_mass
            elif lower < -cutoff_sigma or upper > cutoff_sigma:
                clipped_lower = max(-cutoff_sigma, min(cutoff_sigma, lower))
                clipped_upper = max(-cutoff_sigma, min(cutoff_sigma, upper))
                clipped_values[index] = stable_interval(
                    clipped_lower, clipped_upper, quadrature_nodes, quadrature_weights,
                    sqrt_two, sqrt_two_pi, narrow_threshold,
                )
            else:
                clipped_values[index] = full_values[index]
    if invalid:
        raise ValueError("Original-cell window inputs must be finite with positive sigma")
    return full, clipped
