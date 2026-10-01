"""Stream native-cell line luminosities into LOS-z images.

The caller supplies the same validity mask for intrinsic and transmitted
emissivities. A batch is a contiguous slice of a C-order flattened x slab;
only its x and y indices are needed because all z cells are summed into the
native x-y image. Neighbouring 2x2 pixels are combined only at finalization.
No emissivity or luminosity cube is retained.
"""
from __future__ import annotations

import numpy as np


VARIANT_KEYS = ("intrinsic", "attenuated")


class LineLuminosityImageAccumulator:
    """Accumulate one luminosity image per line and attenuation variant.

    ``native_xy_shape`` is the full snapshot's (x, y) cell count. The final
    image has half as many pixels along each axis, and each pixel contains
    the *sum*, rather than the average, of its 2x2 original sightlines.
    Batches first accumulate into the native-resolution x-y image; the
    2x2 summation is performed by ``finalize`` after all slabs.
    """

    def __init__(self, line_keys, native_xy_shape):
        self.line_keys = tuple(line_keys)
        if not self.line_keys or len(set(self.line_keys)) != len(self.line_keys):
            raise ValueError("line_keys must be nonempty and unique")
        dimensions = tuple(native_xy_shape)
        if (len(dimensions) != 2 or any(not isinstance(n, (int, np.integer)) for n in dimensions)
                or any(n <= 0 or n % 2 for n in dimensions)):
            raise ValueError("native_xy_shape must contain two positive even integers")
        self.native_xy_shape = tuple(int(n) for n in dimensions)
        self.image_shape = tuple(n // 2 for n in dimensions)
        self.native_images = np.zeros((len(VARIANT_KEYS), len(self.line_keys), *self.native_xy_shape))

    def add_slab_batch(self, *, x_start, slab_shape, batch_start, valid_mask,
                       intrinsic_emissivity, transmitted_emissivity,
                       cell_volume_cm3):
        """Add a flat batch from one x slab; emissivities have (line,cell) shape.

        ``batch_start`` is the first cell's flat index *within the slab*, not
        within the full snapshot. ``slab_shape`` is (slab_x, full_y, full_z).
        The flat batch uses NumPy C order, so z varies fastest.
        """
        if not isinstance(x_start, (int, np.integer)) or not isinstance(batch_start, (int, np.integer)):
            raise ValueError("x_start and batch_start must be integers")
        x_start, batch_start = int(x_start), int(batch_start)
        shape = tuple(slab_shape)
        if (len(shape) != 3 or any(not isinstance(n, (int, np.integer)) for n in shape)
                or any(n <= 0 for n in shape)):
            raise ValueError("slab_shape must contain three positive integers")
        nx, ny, nz = (int(n) for n in shape)
        if (x_start < 0 or x_start + nx > self.native_xy_shape[0]
                or ny != self.native_xy_shape[1]):
            raise ValueError("slab x/y extent exceeds the native image")
        valid = np.asarray(valid_mask)
        if valid.ndim != 1 or valid.dtype != np.bool_:
            raise ValueError("valid_mask must be a one-dimensional boolean array")
        cell_count = valid.size
        if batch_start < 0 or batch_start + cell_count > nx * ny * nz:
            raise ValueError("batch lies outside the slab")
        expected = (len(self.line_keys), cell_count)
        intrinsic = np.asarray(intrinsic_emissivity, dtype=float)
        transmitted = np.asarray(transmitted_emissivity, dtype=float)
        if intrinsic.shape != expected or transmitted.shape != expected:
            raise ValueError("emissivities must have shape (line,cell)")
        volume = np.asarray(cell_volume_cm3, dtype=float)
        if (volume.shape not in ((), (cell_count,)) or not np.isfinite(volume).all()
                or np.any(volume <= 0.)):
            raise ValueError("cell_volume_cm3 must be positive, scalar or shape (cell,)")

        selected = np.flatnonzero(valid)
        if not selected.size:
            return self
        for name, epsilon in (("intrinsic", intrinsic), ("transmitted", transmitted)):
            values = epsilon[:, selected]
            if not np.isfinite(values).all() or np.any(values < 0.):
                raise ValueError(f"{name} emissivities of valid cells must be finite and nonnegative")
        selected_volume = volume if volume.ndim == 0 else volume[selected]
        luminosities = np.stack((intrinsic[:, selected], transmitted[:, selected])) * selected_volume
        if not np.isfinite(luminosities).all():
            raise ValueError("emissivity times cell volume must remain finite")

        slab_indices = batch_start + selected
        x_indices = x_start + slab_indices // (ny * nz)
        y_indices = (slab_indices // nz) % ny
        pixel_flat = x_indices * self.native_xy_shape[1] + y_indices
        pixel_count = self.native_xy_shape[0] * self.native_xy_shape[1]
        for variant in range(len(VARIANT_KEYS)):
            for line in range(len(self.line_keys)):
                self.native_images[variant, line] += np.bincount(
                    pixel_flat, weights=luminosities[variant, line],
                    minlength=pixel_count,
                ).reshape(self.native_xy_shape)
        return self

    def finalize(self):
        """Sum neighbouring native pixels and return images in erg/s."""
        downsampled = self.native_images.reshape(
            len(VARIANT_KEYS), len(self.line_keys),
            self.image_shape[0], 2, self.image_shape[1], 2,
        ).sum(axis=(3, 5))
        return {
            "line_keys": np.asarray(self.line_keys),
            "variant_keys": np.asarray(VARIANT_KEYS),
            "axis_order": np.asarray("variant,line,x_pixel,y_pixel"),
            "native_xy_shape": np.asarray(self.native_xy_shape),
            "image_shape": np.asarray(self.image_shape),
            "line_luminosity_image_erg_s": downsampled,
            "total_luminosity_erg_s": downsampled.sum(axis=(-2, -1)),
        }
