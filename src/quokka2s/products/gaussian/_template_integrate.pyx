# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Fuse four-template wide interiors into one coefficient pass.

All probabilities come from the prepared fractional-phase/log-sigma table. Each
cell uses at most 400 stack doubles, never an input-batch by channel matrix.
The main batch worker remains responsible for parallelism and ordered merging.
"""
from libc.math cimport floor, isfinite
from libc.stdint cimport int64_t, uint8_t
from libc.string cimport memset

import numpy as np


cdef extern from *:
    """
    #include <stddef.h>
    #include <math.h>
    #if defined(__aarch64__)
    #include <arm_neon.h>
    #endif

    static int template_has_neon(void) {
    #if defined(__aarch64__)
        return 1;
    #else
        return 0;
    #endif
    }

    /* Independent channels may be vectorized without changing cell order. */
    static inline void template_axpy_compiler(
        double * restrict output, const double * restrict values,
        double scale, size_t count
    ) {
    #if defined(__clang__)
    #pragma clang loop vectorize(enable) interleave(enable)
    #endif
        for (size_t index = 0; index < count; ++index) {
            double product = scale * values[index];
            output[index] = output[index] + product;
        }
    }

    static inline void template_axpy_neon(
        double * restrict output, const double * restrict values,
        double scale, size_t count
    ) {
        size_t index = 0;
    #if defined(__aarch64__)
        const float64x2_t scales = vdupq_n_f64(scale);
        for (; index + 1 < count; index += 2) {
            const float64x2_t data = vld1q_f64(values + index);
            const float64x2_t current = vld1q_f64(output + index);
            const float64x2_t product = vmulq_f64(data, scales);
            vst1q_f64(output + index, vaddq_f64(current, product));
        }
    #endif
        for (; index < count; ++index) {
            double product = scale * values[index];
            output[index] = output[index] + product;
        }
    }

    /* Every source is valid over this common support interval. */
    static inline void template_four_packets(
        double * restrict output,
        const double * restrict first, const double * restrict second,
        const double * restrict third, const double * restrict fourth,
        double w0, double w1, double w2, double w3, size_t count
    ) {
        size_t index = 0;
    #if defined(__aarch64__)
        const float64x2_t weight0 = vdupq_n_f64(w0);
        const float64x2_t weight1 = vdupq_n_f64(w1);
        const float64x2_t weight2 = vdupq_n_f64(w2);
        const float64x2_t weight3 = vdupq_n_f64(w3);
        const float64x2_t zero = vdupq_n_f64(0.0);
        for (; index + 1 < count; index += 2) {
            float64x2_t value = vaddq_f64(zero, vmulq_f64(weight0, vld1q_f64(first + index)));
            value = vaddq_f64(value, vmulq_f64(weight1, vld1q_f64(second + index)));
            value = vaddq_f64(value, vmulq_f64(weight2, vld1q_f64(third + index)));
            value = vaddq_f64(value, vmulq_f64(weight3, vld1q_f64(fourth + index)));
            vst1q_f64(output + index, value);
        }
    #endif
        for (; index < count; ++index) {
            double product = w0 * first[index];
            double value = 0.0 + product;
            product = w1 * second[index];
            value = value + product;
            product = w2 * third[index];
            value = value + product;
            product = w3 * fourth[index];
            output[index] = value + product;
        }
    }
    """
    int template_has_neon() noexcept nogil
    void template_four_packets(double *, const double *, const double *, const double *, const double *, double, double, double, double, size_t) noexcept nogil
    void template_axpy_compiler(double *, const double *, double, size_t) noexcept nogil
    void template_axpy_neon(double *, const double *, double, size_t) noexcept nogil


cdef inline Py_ssize_t sigma_upper(
    double value, const double *nodes, Py_ssize_t count,
) noexcept nogil:
    cdef Py_ssize_t first = 0
    cdef Py_ssize_t stop = count
    cdef Py_ssize_t middle
    while first < stop:
        middle = first + (stop - first) // 2
        if nodes[middle] < value:
            first = middle + 1
        else:
            stop = middle
    return first


cdef inline void phase_bracket(
    Py_ssize_t layer, double phase, const int64_t *starts,
    const double *values, Py_ssize_t *lower, Py_ssize_t *upper,
    double *fraction,
) noexcept nogil:
    """Search actual per-layer phases, preserving exact grid-point meanings."""
    cdef Py_ssize_t first = starts[layer]
    cdef Py_ssize_t stop = starts[layer + 1]
    cdef Py_ssize_t middle
    while first < stop:
        middle = first + (stop - first) // 2
        if values[middle] <= phase:
            first = middle + 1
        else:
            stop = middle
    lower[0] = first - 1
    upper[0] = first
    if phase >= 1.0:
        upper[0] = lower[0]
        fraction[0] = 0.0
    else:
        fraction[0] = (phase - values[lower[0]]) / (values[upper[0]] - values[lower[0]])
        if fraction[0] < 0.0:
            fraction[0] = 0.0
        elif fraction[0] > 1.0:
            fraction[0] = 1.0


cdef inline Py_ssize_t sigma_upper_fast(
    double value, double log_value, const double *nodes,
    const double *log_nodes, Py_ssize_t count,
) noexcept nogil:
    """Arithmetic log-layer guess corrected against actual stored sigma values."""
    cdef double step
    cdef double position
    cdef Py_ssize_t selected
    if count < 2:
        return sigma_upper(value, nodes, count)
    step = log_nodes[1] - log_nodes[0]
    if not isfinite(step) or step <= 0.0:
        return sigma_upper(value, nodes, count)
    position = (log_value - log_nodes[0]) / step
    selected = <Py_ssize_t>floor(position) + 1
    if selected < 0:
        selected = 0
    elif selected >= count:
        selected = count - 1
    # The final sigma endpoint can shorten its last log interval. Comparisons
    # with actual nodes retain searchsorted(side='left') at every boundary.
    while selected > 0 and value <= nodes[selected - 1]:
        selected -= 1
    while selected < count - 1 and value > nodes[selected]:
        selected += 1
    return selected


cdef inline void phase_bracket_fast(
    Py_ssize_t layer, double phase, double sigma, double delta_v,
    const int64_t *starts, const double *values,
    Py_ssize_t *lower, Py_ssize_t *upper, double *fraction,
) noexcept nogil:
    """Guess a uniform/edge-segment phase index, then correct actual-axis ties."""
    cdef Py_ssize_t first = starts[layer]
    cdef Py_ssize_t stop = starts[layer + 1]
    cdef Py_ssize_t count = stop - first
    cdef Py_ssize_t segments
    cdef Py_ssize_t selected
    cdef Py_ssize_t right_first
    cdef double radius = 5.0 * sigma / delta_v
    cdef double left_end
    cdef double right_begin
    cdef double position
    if phase >= 1.0:
        lower[0] = stop - 1
        upper[0] = stop - 1
        fraction[0] = 0.0
        return
    if radius < 0.5:
        segments = count // 2 - 1
        if segments < 1:
            phase_bracket(layer, phase, starts, values, lower, upper, fraction)
            return
        left_end = values[first + segments]
        right_first = first + segments + 1
        right_begin = values[right_first]
        if phase < left_end:
            position = phase / left_end * segments
            selected = first + <Py_ssize_t>floor(position)
        elif phase < right_begin:
            selected = first + segments
        else:
            position = (phase - right_begin) / (1.0 - right_begin) * segments
            selected = right_first + <Py_ssize_t>floor(position)
    else:
        position = phase * (count - 1)
        selected = first + <Py_ssize_t>floor(position)
    if selected < first:
        selected = first
    elif selected > stop - 2:
        selected = stop - 2
    # np.linspace and mirrored edge knots have rounded coordinates. The index
    # guess never defines weights: actual stored neighbors do, as in NumPy.
    while selected > first and phase < values[selected]:
        selected -= 1
    while selected < stop - 2 and phase >= values[selected + 1]:
        selected += 1
    lower[0] = selected
    upper[0] = selected + 1
    fraction[0] = (phase - values[selected]) / (values[selected + 1] - values[selected])
    if fraction[0] < 0.0:
        fraction[0] = 0.0
    elif fraction[0] > 1.0:
        fraction[0] = 1.0


cdef inline bint locate_row(
    double center, double sigma, double log_sigma, double delta_v,
    const double *sigma_nodes, const double *log_sigma_nodes,
    Py_ssize_t layers, const int64_t *phase_starts, const double *phases,
    const int64_t *row_min, const int64_t *row_count, const int64_t *row_offset,
    int64_t output_first, Py_ssize_t channels,
    Py_ssize_t *corner_offsets, Py_ssize_t *corner_first,
    Py_ssize_t *corner_stop, double *corner_weights,
    Py_ssize_t *first_bin, Py_ssize_t *stop_bin,
) noexcept nogil:
    """Return the union of the four actual corner supports, not cell support."""
    cdef double coordinate = center / delta_v
    cdef int64_t shift = <int64_t>floor(coordinate)
    cdef double phase = coordinate - <double>shift
    cdef Py_ssize_t high_layer = sigma_upper_fast(sigma, log_sigma, sigma_nodes, log_sigma_nodes, layers)
    cdef Py_ssize_t low_layer = high_layer - 1 if high_layer > 0 else 0
    cdef double sigma_fraction = 0.0
    cdef double low_fraction
    cdef double high_fraction
    cdef Py_ssize_t rows[4]
    cdef Py_ssize_t corner
    cdef Py_ssize_t selected
    cdef int64_t begin
    cdef int64_t end
    if high_layer != low_layer:
        sigma_fraction = (log_sigma - log_sigma_nodes[low_layer]) / (log_sigma_nodes[high_layer] - log_sigma_nodes[low_layer])
        if sigma_fraction < 0.0:
            sigma_fraction = 0.0
        elif sigma_fraction > 1.0:
            sigma_fraction = 1.0
    phase_bracket_fast(low_layer, phase, sigma_nodes[low_layer], delta_v, phase_starts, phases, &rows[0], &rows[1], &low_fraction)
    phase_bracket_fast(high_layer, phase, sigma_nodes[high_layer], delta_v, phase_starts, phases, &rows[2], &rows[3], &high_fraction)
    corner_weights[0] = (1.0 - sigma_fraction) * (1.0 - low_fraction)
    corner_weights[1] = (1.0 - sigma_fraction) * low_fraction
    corner_weights[2] = sigma_fraction * (1.0 - high_fraction)
    corner_weights[3] = sigma_fraction * high_fraction
    first_bin[0] = channels
    stop_bin[0] = 0
    for corner in range(4):
        selected = rows[corner]
        corner_first[corner] = 0
        corner_stop[corner] = 0
        corner_offsets[corner] = 0
        if corner_weights[corner] == 0.0:
            continue
        # All endpoint bins are known zero. Omitting them changes no mass.
        begin = row_min[selected] + shift - output_first + 1
        end = row_min[selected] + row_count[selected] + shift - output_first - 1
        if begin < 0:
            begin = 0
        if end > channels:
            end = channels
        if begin >= end:
            continue
        corner_first[corner] = <Py_ssize_t>begin
        corner_stop[corner] = <Py_ssize_t>end
        corner_offsets[corner] = <Py_ssize_t>(row_offset[selected] + output_first - shift - row_min[selected])
        if begin < first_bin[0]:
            first_bin[0] = <Py_ssize_t>begin
        if end > stop_bin[0]:
            stop_bin[0] = <Py_ssize_t>end
    return first_bin[0] < stop_bin[0]


cdef inline int64_t fill_coefficients(
    double *coefficients, const double *table,
    const Py_ssize_t *offsets, const Py_ssize_t *first,
    const Py_ssize_t *stop, const double *weights,
    Py_ssize_t first_bin, Py_ssize_t stop_bin,
) noexcept nogil:
    """Write four-corner wide-support interiors once, retaining corner order.

    All edges retain the scalar candidate's contiguous corner SAXPY calls.
    Only an intersection containing at least 32 bins uses four safe pointers.
    """
    cdef Py_ssize_t corner
    cdef Py_ssize_t begin
    cdef Py_ssize_t end
    cdef Py_ssize_t count
    cdef Py_ssize_t common_first = first_bin
    cdef Py_ssize_t common_stop = stop_bin
    cdef int64_t values_read = 0
    cdef bint four_active = True
    memset(coefficients + first_bin, 0, (stop_bin - first_bin) * sizeof(double))
    for corner in range(4):
        if weights[corner] == 0.0 or first[corner] == stop[corner]:
            four_active = False
            break
        if first[corner] > common_first:
            common_first = first[corner]
        if stop[corner] < common_stop:
            common_stop = stop[corner]
    if not four_active or common_stop - common_first < 32:
        for corner in range(4):
            begin = first[corner]
            count = stop[corner] - begin
            if count == 0:
                continue
            template_axpy_neon(coefficients + begin, table + (offsets[corner] + begin), weights[corner], count)
            values_read += count
        return values_read
    # Edges need different zero supports. Each channel still sees corners 0..3.
    for corner in range(4):
        begin = first[corner]
        end = min(stop[corner], common_first)
        count = end - begin
        if count > 0:
            template_axpy_neon(coefficients + begin, table + (offsets[corner] + begin), weights[corner], count)
            values_read += count
    count = common_stop - common_first
    template_four_packets(
        coefficients + common_first,
        table + (offsets[0] + common_first),
        table + (offsets[1] + common_first),
        table + (offsets[2] + common_first),
        table + (offsets[3] + common_first),
        weights[0], weights[1], weights[2], weights[3], count,
    )
    values_read += 4 * count
    for corner in range(4):
        begin = max(first[corner], common_stop)
        count = stop[corner] - begin
        if count > 0:
            template_axpy_neon(coefficients + begin, table + (offsets[corner] + begin), weights[corner], count)
            values_read += count
    return values_read


def _validate_table_arrays(
    const double[::1] sigma_nodes,
    const double[::1] log_sigma_nodes,
    const int64_t[::1] phase_starts,
    const double[::1] phase_values,
    const int64_t[::1] row_min,
    const int64_t[::1] row_count,
    const int64_t[::1] row_offset,
    const double[::1] table_values,
    const int64_t[::1] output_bins,
):
    """Validate static table structure once, before any cell-owned work begins."""
    layers = sigma_nodes.shape[0]
    rows = phase_values.shape[0]
    if layers < 1 or log_sigma_nodes.shape[0] != layers:
        raise ValueError("Sigma/log-sigma arrays must have the same nonzero length")
    if phase_starts.shape[0] != layers + 1 or phase_starts[0] != 0 or phase_starts[layers] != rows:
        raise ValueError("Phase starts must delimit every sigma layer")
    if row_min.shape[0] != rows or row_count.shape[0] != rows or row_offset.shape[0] != rows:
        raise ValueError("Each phase row needs bin geometry and a value offset")
    if output_bins.shape[0] < 1 or output_bins.shape[0] > 400:
        raise ValueError("The native stack supports one to 400 output channels")
    if not np.array_equal(np.asarray(output_bins), output_bins[0] + np.arange(output_bins.shape[0], dtype=np.int64)):
        raise ValueError("Output bins must be consecutive signed integer bins")
    if not np.isfinite(np.asarray(table_values)).all() or np.any(np.asarray(table_values) < 0.0):
        raise ValueError("Stored template masses must be finite and nonnegative")
    for layer in range(layers):
        start = phase_starts[layer]
        stop = phase_starts[layer + 1]
        if stop - start < 2:
            raise ValueError("Each sigma layer requires both phase endpoints")
        if phase_values[start] != 0.0 or phase_values[stop - 1] != 1.0:
            raise ValueError("Layer phases must span the closed interval [0,1]")
        if np.any(np.diff(np.asarray(phase_values)[start:stop]) <= 0.0):
            raise ValueError("Layer phases must be strictly increasing")
    if not np.isfinite(np.asarray(sigma_nodes)).all() or np.any(np.asarray(sigma_nodes) <= 0.0) or np.any(np.diff(np.asarray(sigma_nodes)) <= 0.0):
        raise ValueError("Sigma nodes must be finite positive and increasing")
    if not np.isfinite(np.asarray(log_sigma_nodes)).all():
        raise ValueError("Log-sigma nodes must be finite")
    if np.any(np.asarray(row_count) < 3) or np.any(np.asarray(row_offset) < 0):
        raise ValueError("Every template row requires zero guards and positive offsets")
    if np.any(np.asarray(row_offset) + np.asarray(row_count) > table_values.shape[0]):
        raise ValueError("Template row metadata exceeds the prepared values")


def validate_table(*args, **kwargs):
    """Validate immutable table metadata; excluded from per-batch query timing."""
    _validate_table_arrays(*args, **kwargs)


cdef inline void validate_cells(
    const double[::1] centers, const double[::1] sigmas,
    const double[::1] log_sigmas, const double[:, ::1] luminosity,
    double delta_v, double sigma_min, double sigma_max,
) except *:
    cdef Py_ssize_t row
    cdef Py_ssize_t profile
    cdef int invalid = 0
    if sigmas.shape[0] != centers.shape[0] or log_sigmas.shape[0] != centers.shape[0] or luminosity.shape[0] != centers.shape[0]:
        raise ValueError("Cell coordinate and luminosity lengths must match")
    if luminosity.shape[1] < 1 or luminosity.shape[1] > 12:
        raise ValueError("The width-group kernel supports one to twelve accounts")
    if not isfinite(delta_v) or delta_v <= 0.0:
        raise ValueError("Channel width must be finite and positive")
    with nogil:
        for row in range(centers.shape[0]):
            if not isfinite(centers[row]) or not isfinite(sigmas[row]) or not isfinite(log_sigmas[row]):
                invalid = 1
            elif sigmas[row] < sigma_min or sigmas[row] > sigma_max:
                invalid = 1
            elif not isfinite(centers[row] / delta_v) or abs(centers[row] / delta_v) >= 4611686018427387904.0:
                invalid = 1
            for profile in range(luminosity.shape[1]):
                if not isfinite(luminosity[row, profile]) or luminosity[row, profile] < 0.0:
                    invalid = 1
    if invalid:
        raise ValueError("Require finite in-range Gaussians, safe signed shifts and nonnegative finite luminosities")


def accumulate(
    const double[::1] centers,
    const double[::1] sigmas,
    const double[::1] log_sigmas,
    const double[:, ::1] luminosity,
    const double[::1] sigma_nodes,
    const double[::1] log_sigma_nodes,
    const int64_t[::1] phase_starts,
    const double[::1] phase_values,
    const int64_t[::1] row_min,
    const int64_t[::1] row_count,
    const int64_t[::1] row_offset,
    const double[::1] table_values,
    const int64_t[::1] output_bins,
    double delta_v=1.0,
):
    """Return float64 channel mass[P,K] and logical work counts.

    Each input Gaussian interpolates four immutable template rows, then adds
    its luminosity-weighted channels immediately. No cell-by-channel matrix or
    luminosity-node table is retained. Static metadata must pass validate_table.
    """
    cdef Py_ssize_t count = centers.shape[0]
    cdef Py_ssize_t profiles = luminosity.shape[1]
    cdef Py_ssize_t channels = output_bins.shape[0]
    cdef Py_ssize_t row
    cdef Py_ssize_t profile
    cdef Py_ssize_t item
    cdef Py_ssize_t active_count
    cdef Py_ssize_t first_bin
    cdef Py_ssize_t stop_bin
    cdef Py_ssize_t offsets[4]
    cdef Py_ssize_t first[4]
    cdef Py_ssize_t stop[4]
    cdef uint8_t active_profiles[12]
    cdef double weights[4]
    cdef double coefficients[400]
    cdef double *output_pointer
    cdef const double *light_pointer
    cdef int64_t active_rows = 0
    cdef int64_t bins_visited = 0
    cdef int64_t corner_values_read = 0
    cdef int64_t product_updates = 0
    cdef double[:, ::1] output
    if channels < 1 or channels > 400 or sigma_nodes.shape[0] < 1:
        raise ValueError("Require a prepared nonempty table and at most 400 channels")
    validate_cells(centers, sigmas, log_sigmas, luminosity, delta_v, sigma_nodes[0], sigma_nodes[sigma_nodes.shape[0] - 1])
    owner = np.zeros((profiles, channels), dtype=np.float64)
    output = owner
    output_pointer = &output[0, 0]
    with nogil:
        for row in range(count):
            active_count = 0
            for profile in range(profiles):
                if luminosity[row, profile] > 0.0:
                    active_profiles[active_count] = <uint8_t>profile
                    active_count += 1
            if active_count == 0:
                continue
            active_rows += 1
            if not locate_row(
                centers[row], sigmas[row], log_sigmas[row], delta_v,
                &sigma_nodes[0], &log_sigma_nodes[0], sigma_nodes.shape[0],
                &phase_starts[0], &phase_values[0], &row_min[0],
                &row_count[0], &row_offset[0], output_bins[0], channels,
                &offsets[0], &first[0], &stop[0], &weights[0], &first_bin, &stop_bin,
            ):
                continue
            bins_visited += stop_bin - first_bin
            corner_values_read += fill_coefficients(
                &coefficients[0], &table_values[0], &offsets[0], &first[0],
                &stop[0], &weights[0], first_bin, stop_bin,
            )
            light_pointer = &luminosity[row, 0]
            for item in range(active_count):
                profile = active_profiles[item]
                template_axpy_compiler(
                    output_pointer + profile * channels + first_bin,
                    &coefficients[first_bin],
                    light_pointer[profile],
                    stop_bin - first_bin,
                )
            product_updates += active_count * (stop_bin - first_bin)
    return owner, {
        "kernel_rows": int(count),
        "active_rows": int(active_rows),
        "support_bins_visited": int(bins_visited),
        "template_corner_values_read": int(corner_values_read),
        "product_updates": int(product_updates),
        "neon_available": bool(template_has_neon()),
        "native_threads_per_batch": 1,
        "cell_by_channel_matrix_allocated": False,
        "per_cell_coefficient_scratch_bytes": 400 * 8,
    }


def probabilities(
    const double[::1] centers,
    const double[::1] sigmas,
    const double[::1] log_sigmas,
    const double[::1] sigma_nodes,
    const double[::1] log_sigma_nodes,
    const int64_t[::1] phase_starts,
    const double[::1] phase_values,
    const int64_t[::1] row_min,
    const int64_t[::1] row_count,
    const int64_t[::1] row_offset,
    const double[::1] table_values,
    const int64_t[::1] output_bins,
    double delta_v=1.0,
):
    """Return diagnostic masses[N,K], using the same fused row lookup."""
    cdef Py_ssize_t count = centers.shape[0]
    cdef Py_ssize_t channels = output_bins.shape[0]
    cdef Py_ssize_t row
    cdef Py_ssize_t first_bin
    cdef Py_ssize_t stop_bin
    cdef Py_ssize_t offsets[4]
    cdef Py_ssize_t first[4]
    cdef Py_ssize_t stop[4]
    cdef double weights[4]
    cdef double[:, ::1] output
    if channels < 1 or channels > 400 or sigma_nodes.shape[0] < 1:
        raise ValueError("Require a prepared table and one to 400 channels")
    dummy_light = np.zeros((count, 1), dtype=np.float64)
    validate_cells(centers, sigmas, log_sigmas, dummy_light, delta_v, sigma_nodes[0], sigma_nodes[sigma_nodes.shape[0] - 1])
    owner = np.zeros((count, channels), dtype=np.float64)
    output = owner
    with nogil:
        for row in range(count):
            if locate_row(
                centers[row], sigmas[row], log_sigmas[row], delta_v,
                &sigma_nodes[0], &log_sigma_nodes[0], sigma_nodes.shape[0],
                &phase_starts[0], &phase_values[0], &row_min[0],
                &row_count[0], &row_offset[0], output_bins[0], channels,
                &offsets[0], &first[0], &stop[0], &weights[0], &first_bin, &stop_bin,
            ):
                fill_coefficients(
                    &output[row, 0], &table_values[0], &offsets[0], &first[0],
                    &stop[0], &weights[0], first_bin, stop_bin,
                )
    return owner


def kernel_info():
    return {
        "neon_available": bool(template_has_neon()),
        "accumulation": "compiler-vectorized contiguous channels",
        "probability_arithmetic": "float64 four corners in order, separate multiply/add",
        "threads_per_call": 1,
        "coefficient_kernel": "One-pass four-template NEON over common interiors of at least 32 bins",
        "coordinate_search": "Arithmetic log/phase guesses corrected against actual stored axes",
        "cell_by_channel_matrix_in_accumulate": False,
    }
