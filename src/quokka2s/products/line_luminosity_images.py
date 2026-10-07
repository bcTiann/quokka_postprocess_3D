"""Stream native-cell line luminosities into LOS-z images.

Each line skips only its own missing emissivities, for both dust states.
A batch is a contiguous slice of a C-order flattened x slab;
only its x and y indices are needed because all z cells are summed into the
native x-y image. Optional coarser images are prepared from saved native pixels.
No emissivity or luminosity cube is retained.
"""
from __future__ import annotations

import numpy as np


from quokka2s.products import DUST_STATES
# Images and independent cell totals sum the same positive luminosities in
# different orders. The million-cell comparison differed by up to 9.4e-13.
IMAGE_LUMINOSITY_RTOL = 1e-10


class LineLuminosityImageAccumulator:
    """Accumulate one luminosity image per line and dust state.

    ``native_xy_shape`` is the selected region's (x, y) cell count. Each saved
    pixel contains the summed luminosity of its original z sightline.
    """

    def __init__(self, line_keys, native_xy_shape, image_xy_origin=(0, 0)):
        """Create zero-filled native-resolution luminosity images.

        Parameters
        ----------
        line_keys : sequence of str
            Nonempty, unique line names from CellEmissionCalculator.line_keys, in that order.
        native_xy_shape : tuple of int, shape (2,)
            Selected cell counts (Nx, Ny), from snapshot.processing_shape[:2].
            Full-box processing keeps (256, 256).
        image_xy_origin : tuple of int, shape (2,)
            Selected image's lower x/y indices in the original grid. Defaults
            to (0, 0); e.g. a 64:128, 80:144 region starts at (64, 80).

        Notes
        -----
        native_images has shape (2, Nline, Nx, Ny), in erg/s. Its first axis
        is intrinsic, then attenuated; each pixel sums the original z column.

        Examples
        --------
        images = LineLuminosityImageAccumulator(
            line_keys=emission_calculator.line_keys,
            native_xy_shape=snapshot.processing_shape[:2],
            image_xy_origin=snapshot.processing_xy_origin,
        )
        """
        self.line_keys = tuple(line_keys) # ("cii", "halpha", "hi21", "ciii_977", "ciii_1907", "ciii_1909", "civ_1548", "civ_1551", "co10", "co21")
        self.native_xy_shape = tuple(int(n) for n in native_xy_shape) # (256, 256)
        self.image_xy_origin = tuple(int(n) for n in image_xy_origin)
        # Full-box example: (2, 10, 256, 256). The four axes are:
        #   axis 0, size   2: 0 = intrinsic, 1 = dust-attenuated.
        #   axis 1, size  10: lines in the same order as CellEmissionCalculator.line_keys.
        #   axis 2, size 256: x pixel index.
        #   axis 3, size 256: y pixel index.
        # Each entry stores luminosity [erg/s] and starts at zero.
        # Example: native_images[0, 1, 10, 20] is the intrinsic Halpha
        # luminosity at local pixel (x=10, y=20), for the current line order.
        self.native_images = np.zeros((len(DUST_STATES), len(self.line_keys), *self.native_xy_shape)) # (2, 10, 256, 256)

    def add_batch(self, cells, emission):
        """Add retained cell luminosities to their original x-y pixels.

        Parameters
        ----------
        cells : CellBatch
            From SlabArrays.batch(); supplies the slab location, flat batch offset
            and cell_volume_cm3 [cm^3]. Cell order is unchanged from the snapshot.
        emission : BatchEmission
            From CellEmissionCalculator.calculate(); each line has its own
            emissivity_is_missing Boolean array, shape (B,).
            lines["halpha"] contains before/after dust epsilon, each (B,)
            [erg/s/cm^3]. Dictionary insertion order does not select image rows.

        Returns
        -------
        LineLuminosityImageAccumulator
            This accumulator after adding emissivity times cell volume.

        Examples
        --------
        images.add_batch(cells, emission)
        """
        # Geometry is shared: map all batch positions once, then select each
        # line's available cells. A missing CO value does not remove Halpha.
        all_pixel_indices = self.map_cells_to_image_pixels(
            x_start=cells.x_start,
            y_start=cells.y_start,
            slab_shape=cells.slab_shape,
            batch_start=cells.batch_start,
            selected_indices=np.arange(cells.density_g_cm3.size),
        )
        for line_index, line_key in enumerate(self.line_keys):
            line = emission.lines[line_key]
            selected_indices = np.flatnonzero(~line.emissivity_is_missing)
            if selected_indices.size == 0:
                continue
            pixel_indices = all_pixel_indices[selected_indices]
            for dust_index, dust_state in enumerate(DUST_STATES):
                if dust_state == "intrinsic":
                    emissivity = line.intrinsic_emissivity_erg_s_cm3
                else:
                    emissivity = line.attenuated_emissivity_erg_s_cm3
                selected_emissivity = emissivity[selected_indices]
                luminosity = selected_emissivity * cells.cell_volume_cm3
                self.accumulate_line_pixel_luminosity(
                    pixel_indices=pixel_indices,
                    luminosity=luminosity,
                    dust_index=dust_index,
                    line_index=line_index,
                )
        return self

    def merge(self, other):
        """Add another batch's images to this accumulator.

        Parameters
        ----------
        other : LineLuminosityImageAccumulator
            Independently accumulated images with the same line order and grid.

        Returns
        -------
        LineLuminosityImageAccumulator
            This accumulator; the other accumulator is unchanged.

        Examples
        --------
        images.merge(batch_images)
        """
        if (self.native_xy_shape != other.native_xy_shape
                or self.image_xy_origin != other.image_xy_origin):
            raise ValueError('Cannot merge images from different native x-y regions')
        self.native_images += other.native_images
        return self

    def map_cells_to_image_pixels(
        self, x_start, y_start, slab_shape, batch_start, selected_indices,
    ):
        """Map compact slab positions (R,) to region-local flat x-y pixels (R,).

        C-order cell storage changes z first, then y, then x. Thus every z cell
        with the same x and y maps to the same image pixel. A slab starting at
        (64, 80), with image_xy_origin=(64, 80), maps its first z column to
        local pixel (0, 0). slab_shape[1] is selected Ny, not the original Ny.
        """
        ny = slab_shape[1]
        nz = slab_shape[2]
        slab_indices = batch_start + selected_indices
        x_indices = x_start + slab_indices // (ny * nz)
        y_indices = y_start + (slab_indices // nz) % ny
        image_x = x_indices - self.image_xy_origin[0]
        image_y = y_indices - self.image_xy_origin[1]
        return image_x * self.native_xy_shape[1] + image_y

    def accumulate_line_pixel_luminosity(
        self,
        pixel_indices,
        luminosity,
        dust_index,
        line_index,
    ):
        """Sum one line's (R,) cell luminosities into its native image [erg/s]."""
        pixel_count = self.native_xy_shape[0] * self.native_xy_shape[1]
        pixel_luminosity = np.bincount(
            pixel_indices,
            weights=luminosity,
            minlength=pixel_count,
        )
        self.native_images[dust_index, line_index] += pixel_luminosity.reshape(
            self.native_xy_shape,
        )

    def build_output(self):
        """Copy the accumulated images into an NPZ-ready dictionary.

        Returns
        -------
        dict
            line_luminosity_image_erg_s: (2, Nline, Nx, Ny) [erg/s].
            total_luminosity_erg_s: (2, Nline) [erg/s], summed over image pixels.
            dust_state_keys is intrinsic then attenuated; line_keys keeps input order.
            Other entries record the axis order and native image shape.

        Examples
        --------
        payload = images.build_output()
        intrinsic_halpha = payload['line_luminosity_image_erg_s'][0, halpha_index]
        """
        images = self.native_images.copy()
        payload = {
            "line_keys": np.asarray(self.line_keys),
            "dust_state_keys": np.asarray(DUST_STATES),
            "axis_order": np.asarray("dust_state,line,x_pixel,y_pixel"),
            "native_xy_shape": np.asarray(self.native_xy_shape),
            "image_shape": np.asarray(self.native_xy_shape),
            "image_xy_origin": np.asarray(self.image_xy_origin),
            "line_luminosity_image_erg_s": images,
            "total_luminosity_erg_s": images.sum(axis=(-2, -1)),
        }
        add_image_display_fields(payload=payload)
        return payload


def add_image_display_fields(payload: dict[str, np.ndarray]) -> None:
    """Save each line's shared intrinsic/attenuated colour limits.

    payload contains luminosity images (2, L, Nx, Ny) [erg/s per pixel].
    Adds color_limits_erg_s (L, 2), ordered minimum/maximum,
    line_has_emission (L,), and image_is_hidden (2, L, Nx, Ny).
    The mask hides nonpositive pixels on logarithmic figures without changing
    raw luminosities. Existing image values are unchanged.
    Example: both Halpha maps use [Halpha maximum / 1e5, Halpha maximum].
    """
    images = payload["line_luminosity_image_erg_s"]
    line_maximum = images.max(axis=(0, 2, 3))
    payload["color_limits_erg_s"] = np.column_stack((
        line_maximum / 1e5,
        line_maximum,
    ))
    payload["line_has_emission"] = line_maximum > 0.0
    payload["image_is_hidden"] = images <= 0.0


def prepare_coarser_image(payload: dict[str, np.ndarray], factor: int) -> dict:
    """Sum neighbouring native luminosity pixels into a saved image product.

    Parameters
    ----------
    payload : dict of ndarray
        Native images.npz fields, including images (2, L, Nx, Ny) [erg/s]
        and physical x/y edges [kpc]. The original payload is not modified.
    factor : int
        Native pixels per axis of the new pixel; must divide Nx and Ny.

    Returns
    -------
    dict
        Coarser luminosities, their own edges and recalculated colour limits.
        Raw/native geometry metadata is retained. factor=2 sums four original
        pixels; it does not average them or re-query any emission table.
    """
    images = payload["line_luminosity_image_erg_s"]
    nx, ny = images.shape[-2:]
    if type(factor) is not int or factor <= 0:
        raise ValueError("image_downsample_factor must be a positive integer")
    if nx % factor or ny % factor:
        raise ValueError("image_downsample_factor must divide both image dimensions")
    prepared = dict(payload)
    grouped_pixels = images.reshape(
        *images.shape[:2],
        nx // factor,
        factor,
        ny // factor,
        factor,
    )
    prepared["line_luminosity_image_erg_s"] = grouped_pixels.sum(axis=(3, 5))
    prepared["x_edges_kpc"] = payload["x_edges_kpc"][::factor]
    prepared["y_edges_kpc"] = payload["y_edges_kpc"][::factor]
    prepared["image_shape"] = np.asarray([nx // factor, ny // factor])
    prepared["image_downsample_factor"] = np.asarray(factor)
    prepared["total_luminosity_erg_s"] = prepared["line_luminosity_image_erg_s"].sum(
        axis=(-2, -1),
    )
    add_image_display_fields(payload=prepared)
    return prepared


def check_image_luminosity(image_totals, cell_totals, line_keys):
    """Compare image-integrated luminosities with independent cell sums.

    Parameters
    ----------
    image_totals : array-like, shape (2, Nline)
        total_luminosity_erg_s from the image accumulator's build_output() [erg/s].
    cell_totals : array-like, shape (2, Nline)
        Independent cell luminosities, summed over cold/hot branches [erg/s].
    line_keys : sequence of str
        Names in the arrays' line order; dust states are intrinsic, attenuated.

    Returns
    -------
    dict
        Relative tolerance, largest relative difference, and its dust state/line.
        Different floating-point summation orders are allowed at rtol=1e-10;
        a larger discrepancy raises ValueError.

    Examples
    --------
    report = check_image_luminosity(image_totals, cell_totals, line_keys)
    """
    image_totals = np.asarray(image_totals, dtype=float)
    cell_totals = np.asarray(cell_totals, dtype=float)
    difference = np.abs(image_totals - cell_totals)
    relative = np.zeros_like(difference)
    positive = cell_totals > 0
    relative[positive] = difference[positive] / cell_totals[positive]
    relative[~positive] = np.where(difference[~positive] == 0, 0, np.inf)
    worst = np.unravel_index(np.argmax(relative), relative.shape)
    dust_state = DUST_STATES[worst[0]]
    line_key = line_keys[worst[1]]
    if not np.allclose(image_totals, cell_totals,
                       rtol=IMAGE_LUMINOSITY_RTOL, atol=0):
        raise ValueError(
            'Image pixels do not sum to cell luminosities: '
            f'{dust_state} {line_key} has relative difference {relative[worst]:.3e} '
            f'(image={image_totals[worst]:.6e}, cells={cell_totals[worst]:.6e})'
        )
    return {
        'relative_tolerance': IMAGE_LUMINOSITY_RTOL,
        'maximum_relative_difference': float(relative[worst]),
        'worst_dust_state': dust_state,
        'worst_line': line_key,
    }
