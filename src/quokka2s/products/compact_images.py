"""Use compact image increments only for histogram worker products.

The main image accumulator and its saved products remain unchanged. Each
worker owns only the columns touched by its one contiguous input batch.
"""

from __future__ import annotations

import time

import numpy as np

from ._image_columns import accumulate_line_columns, merge_column_sums


def empty_statistics():
    return {
        "add_batch_calls": 0,
        "input_cells": 0,
        "summed_columns": 0,
        "compact_bytes": 0,
        "maximum_columns_per_partial": 0,
        "add_batch_seconds": 0.0,
        "dense_merge_seconds": 0.0,
    }


class CompactImageAccumulator:
    """Store (2, line_count, touched_columns) sequential sums for one batch.

    Geometry matches the original dense accumulator. A batch can start/end
    inside a z column, cross y/x boundaries, or cover a narrower y tile.
    It must remain one contiguous C-order slice of its stated slab.
    """

    def __init__(self, line_keys, native_xy_shape, image_xy_origin=(0, 0)):
        self.line_keys = tuple(line_keys)
        self.native_xy_shape = tuple(int(value) for value in native_xy_shape)
        self.image_xy_origin = tuple(int(value) for value in image_xy_origin)
        self.pixel_ids = np.empty(0, dtype=np.int64)
        self.column_sums = np.empty((2, len(self.line_keys), 0), dtype=np.float64)
        self.compact_statistics = empty_statistics()
        self._added = False

    def add_batch(self, cells, emission):
        """Sum retained float64 emissivities times volume without dense images."""
        if self._added:
            raise RuntimeError("A compact worker partial accepts exactly one input batch")
        began = time.perf_counter()
        count = cells.density_g_cm3.size
        shape = tuple(int(value) for value in cells.slab_shape)
        if len(shape) != 3 or any(value < 1 for value in shape):
            raise ValueError("Image batches require a positive three-dimensional slab")
        batch_start = int(cells.batch_start)
        if batch_start < 0 or batch_start + count > np.prod(shape):
            raise ValueError("The contiguous image batch lies outside its slab")
        depth = shape[2]
        first_column = batch_start // depth
        column_count = (batch_start % depth + count + depth - 1) // depth if count else 0
        local_columns = first_column + np.arange(column_count, dtype=np.int64)
        local_x = int(cells.x_start) - self.image_xy_origin[0] + local_columns // shape[1]
        local_y = int(cells.y_start) - self.image_xy_origin[1] + local_columns % shape[1]
        if np.any(local_x < 0) or np.any(local_x >= self.native_xy_shape[0]):
            raise ValueError("Image batch x columns lie outside the processing region")
        if np.any(local_y < 0) or np.any(local_y >= self.native_xy_shape[1]):
            raise ValueError("Image batch y columns lie outside the processing region")
        self.pixel_ids = np.ascontiguousarray(local_x * self.native_xy_shape[1] + local_y)
        self.column_sums = np.zeros((2, len(self.line_keys), column_count), dtype=np.float64)
        for line_index, line_key in enumerate(self.line_keys):
            line = emission.lines[line_key]
            intrinsic = line.intrinsic_emissivity_erg_s_cm3
            attenuated = line.attenuated_emissivity_erg_s_cm3
            missing = line.emissivity_is_missing
            for values in (intrinsic, attenuated):
                if values.shape != (count,) or values.dtype != np.dtype(np.float64) or not values.flags.c_contiguous:
                    raise ValueError("Compact images require contiguous float64 line arrays")
            if missing.shape != (count,) or missing.dtype != np.dtype(np.bool_) or not missing.flags.c_contiguous:
                raise ValueError("Compact images require a contiguous per-line Boolean missing mask")
            accumulate_line_columns(
                intrinsic=intrinsic,
                attenuated=attenuated,
                missing=missing.view(np.uint8),
                first_z=batch_start % depth,
                column_depth=depth,
                cell_volume_cm3=cells.cell_volume_cm3,
                column_sums=self.column_sums,
                line_index=line_index,
            )
        self._added = True
        self.compact_statistics.update({
            "add_batch_calls": 1,
            "input_cells": int(count),
            "summed_columns": int(column_count),
            "compact_bytes": int(self.pixel_ids.nbytes + self.column_sums.nbytes),
            "maximum_columns_per_partial": int(column_count),
            "add_batch_seconds": time.perf_counter() - began,
        })
        return self

    def merge_into_dense(self, target):
        """Apply this worker once to the unchanged dense image accumulator."""
        if self.native_xy_shape != target.native_xy_shape or self.image_xy_origin != target.image_xy_origin:
            raise ValueError("Cannot merge images from different processing geometry")
        if self.line_keys != target.line_keys:
            raise ValueError("Cannot merge images with different line order")
        began = time.perf_counter()
        merge_column_sums(
            dense_images=target.native_images.reshape(2, len(self.line_keys), -1),
            pixel_ids=self.pixel_ids,
            column_sums=self.column_sums,
        )
        if not hasattr(target, "compact_statistics"):
            target.compact_statistics = empty_statistics()
        for name, value in self.compact_statistics.items():
            if name == "maximum_columns_per_partial":
                target.compact_statistics[name] = max(target.compact_statistics[name], value)
            else:
                target.compact_statistics[name] += value
        target.compact_statistics["dense_merge_seconds"] += time.perf_counter() - began
        return target

    def build_output(self):
        raise RuntimeError("Merge compact worker images into the main dense accumulator before output")
