"""Derive six physical fields directly from one owned full-z native tile.

The gradient pass uses original raw density and momenta plus global XY neighbour
faces. After it finishes, x/y momenta are released. Density and z momentum are
then converted in place, temperature is reused unchanged, and only the two
column-density fields are newly allocated. No converted field copy is written.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from typing import TYPE_CHECKING
import weakref

import numpy as np

from .bounded_z_reader import scalar_unit_stages
from .native_snapshot_reader import NATIVE_FIELD_NAMES
from .physics import settings
from .physics._slab_derive import fill_tile_columns_inplace, fill_tile_velocity_gradient
from .physics.gas_fields import HYDROGEN_MASS_G
from .snapshot_reader import SlabArrays

if TYPE_CHECKING:
    from .native_column_reader import NativeColumnRawTile
    from .native_snapshot_reader import NativeSnapshotReader
    from .snapshot_reader import Snapshot


def _check_raw_array(array, *, name, shape, writable_owner):
    """Reject a layout that would require an implicit full-tile data copy."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Native tile field {name} must be a NumPy array")
    if array.dtype != np.dtype(np.float64) or array.shape != shape:
        raise ValueError(f"Native tile field {name} must have float64 shape {shape}")
    if not array.flags.c_contiguous:
        raise ValueError(f"Native tile field {name} must be C-contiguous")
    if writable_owner and (not array.flags.writeable or not array.flags.owndata):
        raise ValueError(f"Native tile field {name} must own writable memory")
    if writable_owner and isinstance(array, np.memmap):
        raise ValueError(f"Native tile field {name} must not be a writable file mapping")


def _check_raw_tile(native_reader, raw_tile):
    """Validate geometry/ownership without scanning or copying large arrays."""
    native_nx, native_ny, native_nz = native_reader._native_shape
    if not (
        0 <= raw_tile.x_start < raw_tile.x_stop <= native_nx
        and 0 <= raw_tile.y_start < raw_tile.y_stop <= native_ny
        and native_nz >= 2
    ):
        raise ValueError("Native column tiles require valid XY bounds and full z columns")
    shape = (
        raw_tile.x_stop - raw_tile.x_start,
        raw_tile.y_stop - raw_tile.y_start,
        native_nz,
    )
    if set(raw_tile.fields) != set(NATIVE_FIELD_NAMES):
        raise ValueError("A native tile must contain exactly the five required raw fields")
    owner_ids = set()
    for name in NATIVE_FIELD_NAMES:
        array = raw_tile.fields[name]
        _check_raw_array(array, name=name, shape=shape, writable_owner=True)
        owner_ids.add(id(array))
    if len(owner_ids) != len(NATIVE_FIELD_NAMES):
        raise ValueError("Native tile fields must use separate owned allocations")
    x_shape = (2, shape[1], native_nz)
    y_shape = (shape[0], 2, native_nz)
    for name, expected_shape in (
        ("x_density_faces", x_shape),
        ("x_momentum_faces", x_shape),
        ("y_density_faces", y_shape),
        ("y_momentum_faces", y_shape),
    ):
        array = getattr(raw_tile, name)
        _check_raw_array(array, name=name, shape=expected_shape, writable_owner=False)
        if any(np.shares_memory(array, core) for core in raw_tile.fields.values()):
            raise ValueError("Native tile neighbour faces must not alias mutable core fields")
    return shape


def derive_native_column_tile(
    snapshot: Snapshot,
    native_reader: NativeSnapshotReader,
    raw_tile: NativeColumnRawTile,
    threads: int,
    on_stage=None,
):
    """Consume owned raw arrays and return (SlabArrays, timing/memory receipt).

    Raw values use native snapshot units. Returned read-only fields use the same
    cgs, K and km/s units as the standard slab reader. Cell order is XYZ with z
    contiguous. The density/Mz allocations become physical density/velocity;
    x/y momenta are removed from raw_tile.fields before column arrays allocate.
    A failed tile must be reread rather than reused after partial conversion.
    """
    began = time.perf_counter()
    if type(threads) is not int or threads < 1:
        raise ValueError("Native column derivation threads must be a positive integer")
    shape = _check_raw_tile(native_reader, raw_tile)
    nx, ny, nz = shape
    column_count = nx * ny
    task_count = min(threads, column_count)
    raw_input_bytes = sum(array.nbytes for array in raw_tile.fields.values())
    face_bytes = sum(
        getattr(raw_tile, name).nbytes
        for name in ("x_density_faces", "x_momentum_faces", "y_density_faces", "y_momentum_faces")
    )
    density = raw_tile.fields["gasDensity"]
    temperature = raw_tile.fields["temperature"]
    momentum_x = raw_tile.fields["x-GasMomentum"]
    momentum_y = raw_tile.fields["y-GasMomentum"]
    momentum_z = raw_tile.fields["z-GasMomentum"]
    xy_momentum_references = (weakref.ref(momentum_x), weakref.ref(momentum_y))
    gradient = np.empty(shape, dtype=np.float64, order="C")
    scratch = np.empty((task_count, 2, nz), dtype=np.float64)
    density_to_gas, gas_density_to_cgs, velocity_factors, shielding_to_cgs = scalar_unit_stages(native_reader)
    widths_cgs = tuple(float(width.to("cm").value) for width in native_reader._cell_widths)
    dz_foreground = float(snapshot.cell_widths[2].to("cm").value)
    dz_shielding = float(native_reader._cell_widths[2].in_cgs().value)

    def calculate_gradient(task):
        fill_tile_velocity_gradient(
            density_raw=density,
            momentum_x=momentum_x,
            momentum_y=momentum_y,
            momentum_z=momentum_z,
            x_density_faces=raw_tile.x_density_faces,
            x_momentum_faces=raw_tile.x_momentum_faces,
            y_density_faces=raw_tile.y_density_faces,
            y_momentum_faces=raw_tile.y_momentum_faces,
            gradient_output=gradient,
            scratch=scratch[task],
            velocity_factors=velocity_factors,
            dx_cm=widths_cgs[0],
            dy_cm=widths_cgs[1],
            dz_gradient_cm=widths_cgs[2],
            gradient_floor_s=float(settings.SIMULATION_DVDR_MIN_S),
            first_column=task * column_count // task_count,
            stop_column=(task + 1) * column_count // task_count,
        )

    with ThreadPoolExecutor(max_workers=task_count, thread_name_prefix="native-column-fields") as pool:
        if on_stage is not None:
            on_stage("gradient_from_raw_fields")
        gradient_began = time.perf_counter()
        list(pool.map(calculate_gradient, range(task_count)))
        gradient_seconds = time.perf_counter() - gradient_began

        # Every gradient task has completed, including its neighbour reads.
        # Clear both closure cells and dictionary entries before two NH arrays
        # allocate; keeping either reference would retain two full raw fields.
        del momentum_x
        del momentum_y
        raw_tile.fields.pop("x-GasMomentum")
        raw_tile.fields.pop("y-GasMomentum")
        xy_momenta_released = all(reference() is None for reference in xy_momentum_references)
        foreground = np.empty(shape, dtype=np.float64, order="C")
        shielding = np.empty(shape, dtype=np.float64, order="C")

        def calculate_columns(task):
            return fill_tile_columns_inplace(
                density=density,
                momentum_z=momentum_z,
                foreground_output=foreground,
                shielding_output=shielding,
                scratch=scratch[task],
                velocity_factors=velocity_factors,
                density_to_gas=density_to_gas,
                gas_density_to_cgs=gas_density_to_cgs,
                hydrogen_fraction=float(settings.X_H),
                hydrogen_mass_g=float(HYDROGEN_MASS_G),
                dz_foreground_cm=dz_foreground,
                dz_shielding_cm=dz_shielding,
                shielding_to_cgs=shielding_to_cgs,
                first_column=task * column_count // task_count,
                stop_column=(task + 1) * column_count // task_count,
            )

        if on_stage is not None:
            on_stage("inplace_units_and_columns")
        columns_began = time.perf_counter()
        invalid_counts = list(pool.map(calculate_columns, range(task_count)))
        units_and_columns_seconds = time.perf_counter() - columns_began
    if sum(invalid_counts):
        raise ValueError("Expected finite nonnegative hydrogen density in full columns")

    fields = {
        "density_g_cm3": density,
        "foreground_NH_cm2": foreground,
        "temperature_QUOKKA_K": temperature,
        "shielding_NH_cm2": shielding,
        "velocity_gradient_s": gradient,
        "velocity_z_kms": momentum_z,
    }
    final_six_field_bytes = sum(array.nbytes for array in fields.values())
    for array in fields.values():
        array.flags.writeable = False
    flattened_fields = {name: array.ravel() for name, array in fields.items()}
    slab = SlabArrays(
        **flattened_fields,
        x_start=raw_tile.x_start,
        y_start=raw_tile.y_start,
        shape=shape,
        native_y_size=native_reader._native_shape[1],
        cell_volume_cm3=snapshot.cell_volume_cm3,
    )
    timings = {
        "gradient_seconds": gradient_seconds,
        "units_and_columns_seconds": units_and_columns_seconds,
        "fused_derivation_seconds": time.perf_counter() - began,
        "raw_input_bytes": raw_input_bytes,
        "face_bytes": face_bytes,
        "scratch_bytes": scratch.nbytes,
        "final_six_field_bytes": final_six_field_bytes,
        "peak_tile_and_working_array_bytes": max(
            raw_input_bytes + gradient.nbytes,
            final_six_field_bytes,
        ) + face_bytes + scratch.nbytes,
        "raw_xy_momenta_removed_from_tile": True,
        "raw_xy_momenta_released": xy_momenta_released,
        "density_temperature_z_momentum_storage_reused": True,
        "gradient_uses_global_neighbour_faces": True,
        "columns": column_count,
        "threads": task_count,
    }
    return slab, timings
