"""Read density and temperature for cold-gas nH–NH phase distributions.

The exact two-pass full-z column sums come from NativeBlockRangeReader.
This reader does not need momentum or velocity-gradient neighbour fields:
each hydrogen column uses only density at the same x–y position.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .table_input_reader import NativeBlockRangeReader, NativeColumnState


@dataclass(frozen=True)
class ColdGasLayer:
    """Three owned, read-only float64 XYZ arrays for one native z block.

    density_g_cm3, temperature_QUOKKA_K and shielding_NH_cm2 have shape
    (selected_Nx, selected_Ny, native_block_Nz), for example (256, 256, 128).
    Their units are g/cm^3, K and cm^-2. Global x_start/y_start/z_start locate
    the arrays in the original box; a selected region changes x/y only.

    The layer includes both cold and hot cells. The accumulator selects
    temperature_QUOKKA_K < 3000 K; the reader never queries emission tables.
    Example: shielding_NH_cm2[0, 0, 0] is the complete +/-z harmonic-mean
    column for the cell at (x_start, y_start, z_start).
    """

    density_g_cm3: np.ndarray
    temperature_QUOKKA_K: np.ndarray
    shielding_NH_cm2: np.ndarray
    x_start: int
    y_start: int
    z_start: int

    def __post_init__(self):
        """Keep the numerical layer immutable while its accumulator reads it."""
        for values in (
            self.density_g_cm3,
            self.temperature_QUOKKA_K,
            self.shielding_NH_cm2,
        ):
            values.flags.writeable = False


class ColdGasColumnReader(NativeBlockRangeReader):
    """Read rho/T and derive exact full-z NH without velocity calculations.

    snapshot : Snapshot
        Opened dataset and native geometry, optionally with an x/y region.
        Construction reads metadata only. The inherited iter_layers() keeps
        the ascending density pass and descending block order unchanged.

    Each selected block reads density once in each pass and temperature once
    in the descending pass. Arrays remain bounded to the current z layer.
    No x/y halo is required for these same-column calculations; the original
    full-z bounds and per-cell addition order still determine both NH sums.

    Example::

        reader = ColdGasColumnReader(snapshot=snapshot)
        for layer in reader.iter_layers():
            accumulator.add_layer(
                layer=layer,
                cell_volume_cm3=snapshot.cell_volume_cm3,
            )
    """

    def calculate_layer(
        self,
        *,
        state: NativeColumnState,
        z_index: int,
    ) -> ColdGasLayer:
        """Read one rho/T block and advance that XY column's upper NH carry.

        state comes from prepare_column_states(); z_index selects a native
        block in its complete z column. Returns three C-order XYZ arrays in
        g/cm^3, K and cm^-2, with their original global starting indices.
        """
        column = state.column
        block = column.blocks[z_index]
        selected_xy = self._core_xy_slices(
            column=column,
            block=block,
        )

        # 1. Attach the same density units used by the emission/range reader.
        native_density = self.z_layer_fields.read_field(
            block=block,
            field_name="gasDensity",
        )
        density = self.native.attach_density_units(
            values=native_density[selected_xy],
        )

        # 2. Continue both complete +/-z columns, preserving every addition.
        shielding_column, next_upper_column = self._shielding_column_for_layer(
            density=density,
            lower_column=state.lower_boundaries[z_index],
            upper_column=state.upper_column,
        )

        # 3. QUOKKA stores raw temperature values interpreted as K. Its yt
        # metadata may have an empty unit label; preserve the existing raw
        # numerical values instead of introducing a temperature conversion.
        native_temperature = self.z_layer_fields.read_field(
            block=block,
            field_name="temperature",
        )
        layer = ColdGasLayer(
            density_g_cm3=np.array(
                density.to("g/cm**3"),
                dtype=float,
                order="C",
            ),
            temperature_QUOKKA_K=np.array(
                native_temperature[selected_xy],
                dtype=float,
                order="C",
            ),
            shielding_NH_cm2=shielding_column,
            x_start=column.x_start,
            y_start=column.y_start,
            z_start=block.start[2],
        )
        state.upper_column = next_upper_column
        return layer
