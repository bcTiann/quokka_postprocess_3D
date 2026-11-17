"""Apply existing trilinear table weights with standard SciPy sparse products."""

from itertools import product
from math import prod

import numpy as np
from scipy.sparse import csr_matrix


def trilinear_brackets(*, axis, coordinate):
    """Locate Q coordinates on actual increasing nodes, retaining right ties.

    Logarithmic physical-table axes are close to uniformly spaced. Predicting
    the interval from their mean spacing avoids binary searches for most rows.
    Each prediction is checked against the actual neighbouring nodes; all
    mismatches, knots at the final node and nonfinite rows use searchsorted.
    Fractions always use the original nodes, including on nonuniform axes.
    """
    with np.errstate(all="ignore"):
        mean_spacing = (axis[-1] - axis[0]) / (axis.size - 1)
    if not np.isfinite(mean_spacing) or mean_spacing <= 0.0:
        upper = np.searchsorted(axis, coordinate, side="right")
        upper = np.clip(upper, 1, axis.size - 1)
        lower = upper - 1
    else:
        bounded = np.clip(coordinate, axis[0], axis[-1])
        with np.errstate(invalid="ignore"):
            upper = ((bounded - axis[0]) / mean_spacing).astype(np.intp) + 1
        np.clip(upper, 1, axis.size - 1, out=upper)
        lower = upper - 1
        retry = (
            ~np.isfinite(coordinate)
            | (coordinate < axis[lower])
            | (coordinate >= axis[upper])
        )
        if np.any(retry):
            upper[retry] = np.clip(
                np.searchsorted(axis, coordinate[retry], side="right"),
                1,
                axis.size - 1,
            )
            lower[retry] = upper[retry] - 1
    fraction = (coordinate - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, fraction


def trilinear_corners(*, brackets, grid_shape):
    """Yield eight flat table-node indices and weights for Q prepared queries.

    brackets holds each axis's lower/upper indices and upper-node fractions.
    Flat C-order indices are shared by mask inspection and field interpolation;
    the original 000,...,111 corner and weight-multiplication order is retained.
    """
    axis_indices = [(lower, upper) for lower, upper, _ in brackets]
    axis_weights = [(1.0 - fraction, fraction) for _, _, fraction in brackets]
    # Corners 000/001 share their first two indices and weight factors.
    # Reuse that work for each pair without changing the 000,...,111 order.
    for first_side, second_side in product((0, 1), repeat=2):
        first_two_weight = np.multiply(
            axis_weights[0][first_side],
            axis_weights[1][second_side],
            dtype=np.float64,
        )
        first_two_index = (
            axis_indices[0][first_side] * grid_shape[1]
            + axis_indices[1][second_side]
        ) * grid_shape[2]
        for third_side in (0, 1):
            node_indices = first_two_index + axis_indices[2][third_side]
            weight = np.multiply(
                first_two_weight,
                axis_weights[2][third_side],
                dtype=np.float64,
            )
            yield node_indices, weight


def trilinear_weight_matrix(*, grid_shape, corners, selected):
    """Return CSR[selected query, grid node] with all eight corner entries.

    grid_shape=(N0,N1,N2) describes the source table. Each corner from
    trilinear_corners() supplies a (Q,) flat-node index and a (Q,) weight array;
    selected is a (Q,) mask. The supplied 000,...,111 corner order is retained.
    Explicit zero weights remain: removing them would change 0 * NaN propagation
    from unavailable table values. No query or table is retained.
    """
    query_count = int(np.count_nonzero(selected))
    node_count = prod(grid_shape)
    entry_count = 8 * query_count
    largest_index = max(node_count, entry_count)
    index_dtype = (
        np.int64 if largest_index > np.iinfo(np.int32).max else np.int32
    )
    indices = np.empty((query_count, 8), dtype=index_dtype)
    weights = np.empty((query_count, 8))
    all_selected = bool(np.all(selected))
    for corner, (node_indices, weight) in enumerate(corners):
        if all_selected:
            indices[:, corner] = node_indices
            weights[:, corner] = weight
        else:
            indices[:, corner] = node_indices[selected]
            weights[:, corner] = weight[selected]
    pointers = np.arange(0, entry_count + 8, 8, dtype=index_dtype)
    return csr_matrix(
        (weights.ravel(), indices.ravel(), pointers),
        shape=(query_count, node_count),
        copy=False,
    )


def interpolate_regular_grid(*, interpolator, points):
    """Read current linear RGI values at flattened points[Q,3] by CSR.

    DESPOTIC's process-time scalar and multi-field RGIs have ascending 3D
    grids, NaN outside values, and bounds_error=False. Other RGI settings keep
    their existing library behavior, including single-node axes, nearest or
    spline methods, custom fills, extrapolation and bounds errors. Coefficients
    are read on each call so mutations and replacements remain visible.
    """
    axes = interpolator.grid
    nan_fill = (
        isinstance(interpolator.fill_value, (float, np.floating))
        and bool(np.isnan(interpolator.fill_value))
    )
    if (
        interpolator.method != "linear"
        or interpolator.bounds_error
        or not nan_fill
        or len(axes) != 3
        or any(axis.size < 2 for axis in axes)
    ):
        return interpolator(points)

    brackets = []
    outside = np.any(~np.isfinite(points), axis=1)
    for axis, coordinate in zip(axes, points.T):
        brackets.append(trilinear_brackets(axis=axis, coordinate=coordinate))
        outside |= (coordinate < axis[0]) | (coordinate > axis[-1])
    grid_shape = tuple(axis.size for axis in axes)
    with np.errstate(invalid="ignore"):
        matrix = trilinear_weight_matrix(
            grid_shape=grid_shape,
            corners=trilinear_corners(brackets=brackets, grid_shape=grid_shape),
            selected=np.ones(points.shape[0], dtype=bool),
        )
        output_shape = interpolator.values.shape[3:]
        values = matrix @ interpolator.values.reshape(prod(grid_shape), -1)
        values[outside] = np.nan
    return values.reshape((points.shape[0], *output_shape))
