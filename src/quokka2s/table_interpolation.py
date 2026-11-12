"""Apply existing trilinear table weights with standard SciPy sparse products."""

from itertools import product
from math import prod

import numpy as np
from scipy.sparse import csr_matrix


def trilinear_weight_matrix(*, grid_shape, corners, selected):
    """Return CSR[selected query, grid node] with all eight corner entries.

    grid_shape=(N0,N1,N2) describes the source table, while each corner has
    three (Q,) index arrays and one (Q,) weight array. selected is a (Q,) mask.
    Flattening preserves grid-axis order and the supplied 000,...,111 corner
    order. Explicit zero weights remain: removing them would change 0 * NaN
    propagation from unavailable table values. No query or table is retained.
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
    for corner, (coordinates, weight) in enumerate(corners):
        if all_selected:
            first, second, third = coordinates
            weights[:, corner] = weight
        else:
            first, second, third = (
                coordinate[selected] for coordinate in coordinates
            )
            weights[:, corner] = weight[selected]
        indices[:, corner] = (
            (first * grid_shape[1] + second) * grid_shape[2] + third
        )
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
        upper = np.searchsorted(axis, coordinate, side="right")
        upper = np.clip(upper, 1, axis.size - 1)
        lower = upper - 1
        fraction = (coordinate - axis[lower]) / (axis[upper] - axis[lower])
        brackets.append((lower, upper, fraction))
        outside |= (coordinate < axis[0]) | (coordinate > axis[-1])
    corners = []
    with np.errstate(invalid="ignore"):
        for sides in product((0, 1), repeat=3):
            coordinates = []
            weight = np.ones(points.shape[0])
            for side, (lower, upper, fraction) in zip(sides, brackets):
                coordinates.append(upper if side else lower)
                weight *= fraction if side else 1.0 - fraction
            corners.append((tuple(coordinates), weight))

        grid_shape = tuple(axis.size for axis in axes)
        matrix = trilinear_weight_matrix(
            grid_shape=grid_shape,
            corners=corners,
            selected=np.ones(points.shape[0], dtype=bool),
        )
        output_shape = interpolator.values.shape[3:]
        values = matrix @ interpolator.values.reshape(prod(grid_shape), -1)
        values[outside] = np.nan
    return values.reshape((points.shape[0], *output_shape))
