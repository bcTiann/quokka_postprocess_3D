# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Sequential float64 luminosity sums for the columns touched by one batch."""

from libc.stdint cimport int64_t, uint8_t


def accumulate_line_columns(
    const double[::1] intrinsic,
    const double[::1] attenuated,
    const uint8_t[::1] missing,
    Py_ssize_t first_z,
    Py_ssize_t column_depth,
    double cell_volume_cm3,
    double[:, :, ::1] column_sums,
    Py_ssize_t line_index,
):
    """Match bincount's ascending-cell multiply/add order for both dust states.

    Input arrays describe one contiguous C-order batch, shape (B,). Output is
    (2, line_count, touched_columns), in erg/s. The first column can begin at
    first_z, and the last column can be incomplete. Missing skips only this line.
    """
    cdef Py_ssize_t count = intrinsic.shape[0]
    cdef Py_ssize_t column
    cdef Py_ssize_t cell
    cdef Py_ssize_t begin = 0
    cdef Py_ssize_t stop
    cdef Py_ssize_t available
    cdef double luminosity
    if attenuated.shape[0] != count or missing.shape[0] != count:
        raise ValueError("Image emissivity and missing arrays must have equal lengths")
    if column_depth < 1 or first_z < 0 or first_z >= column_depth:
        raise ValueError("Invalid column depth or first z position")
    if column_sums.shape[0] != 2 or line_index < 0 or line_index >= column_sums.shape[1]:
        raise ValueError("Invalid compact image line/dust shape")
    if count and column_sums.shape[2] != (first_z + count + column_depth - 1) // column_depth:
        raise ValueError("Compact output does not match the touched column count")
    with nogil:
        for column in range(column_sums.shape[2]):
            available = column_depth - first_z if column == 0 else column_depth
            stop = begin + available
            if stop > count:
                stop = count
            for cell in range(begin, stop):
                if not missing[cell]:
                    luminosity = intrinsic[cell] * cell_volume_cm3
                    column_sums[0, line_index, column] = column_sums[0, line_index, column] + luminosity
                    luminosity = attenuated[cell] * cell_volume_cm3
                    column_sums[1, line_index, column] = column_sums[1, line_index, column] + luminosity
            begin = stop


def merge_column_sums(
    double[:, :, ::1] dense_images,
    const int64_t[::1] pixel_ids,
    const double[:, :, ::1] column_sums,
):
    """Merge compact values into (dust,line,flattened_xy) in original batch order."""
    cdef Py_ssize_t dust
    cdef Py_ssize_t line
    cdef Py_ssize_t column
    if dense_images.shape[0] != column_sums.shape[0] or dense_images.shape[1] != column_sums.shape[1]:
        raise ValueError("Dense and compact image dust/line shapes differ")
    if column_sums.shape[2] != pixel_ids.shape[0]:
        raise ValueError("Pixel IDs and compact columns differ in length")
    for column in range(pixel_ids.shape[0]):
        if pixel_ids[column] < 0 or pixel_ids[column] >= dense_images.shape[2]:
            raise ValueError("A compact pixel ID lies outside the dense image")
        if column and pixel_ids[column] <= pixel_ids[column - 1]:
            raise ValueError("Compact pixel IDs must be increasing and unique")
    with nogil:
        for dust in range(column_sums.shape[0]):
            for line in range(column_sums.shape[1]):
                for column in range(pixel_ids.shape[0]):
                    dense_images[dust, line, pixel_ids[column]] = dense_images[dust, line, pixel_ids[column]] + column_sums[dust, line, column]
