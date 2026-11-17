"""Apply runtime trilinear tables with fused arithmetic or SciPy products."""

from itertools import product
from math import prod

import numpy as np
from scipy.sparse import csr_matrix

from ._table_kernels import interpolate_regular_fields


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
    """Read current linear RGI values at flattened points[Q,3].

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

    # The physical tables use Float64 log coordinates. Keep SciPy's existing
    # arithmetic for other dtypes and explicitly customized NumPy error policies.
    default_errors = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}
    if (
        points.dtype == np.float64
        and interpolator.values.dtype == np.float64
        and all(axis.dtype == np.float64 for axis in axes)
        and np.geterr() == default_errors
    ):
        grid_shape = tuple(axis.size for axis in axes)
        output_shape = interpolator.values.shape[3:]
        values = interpolate_regular_fields(
            axis0=axes[0],
            axis1=axes[1],
            axis2=axes[2],
            points=points,
            fields=interpolator.values.reshape(prod(grid_shape), -1),
        )
        return values.reshape((points.shape[0], *output_shape))

    brackets = []
    outside = np.any(~np.isfinite(points), axis=1)
    for axis, coordinate in zip(axes, points.T):
        upper = np.searchsorted(axis, coordinate, side="right")
        upper = np.clip(upper, 1, axis.size - 1)
        lower = upper - 1
        fraction = (coordinate - axis[lower]) / (axis[upper] - axis[lower])
        brackets.append((lower, upper, fraction))
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
