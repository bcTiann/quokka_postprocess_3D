"""Extend DESPOTIC axes while retaining directly solved raw-table nodes."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from quokka2s.despotic.table_data import DespoticTable, ExplicitGrid
from quokka2s.despotic.table_files import load_table

if TYPE_CHECKING:
    from quokka2s.despotic.table_builder import SpeciesSpec


SOLVER_METADATA_FIELDS = (
    "composition",
    "despotic",
    "numerical_settings",
    "final_validation",
)
INTERPOLATION_FIELDS = {
    "source_failure_mask",
    "interpolation_target_mask",
    "filled_T_mu_node_mask",
    "filled_all_physical_fields_node_mask",
    "remaining_unavailable_T_mu_node_mask",
    "remaining_unavailable_any_physical_field_node_mask",
}


def extend_logarithmic_axis(
    old_values: np.ndarray,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    """Add outward nodes at the existing logarithmic spacing; never move old nodes.

    old_values is a positive increasing (N,) axis, in its original physical
    units. minimum/maximum use those same units. Returns a new (M,) float
    array that covers both the requested range and every old node. Bounds
    inside the old range leave it unchanged. Extension needs at least two
    uniformly log-spaced old nodes; an unchanged axis need not be uniform.
    Example: [1, 10, 100] extended to [0.2, 200] becomes
    [0.1, 1, 10, 100, 1000], with the three old values copied exactly.
    """
    old_values = np.asarray(old_values, dtype=float)
    ExplicitGrid(tuple(old_values))
    if not np.isfinite([minimum, maximum]).all() or not 0 < minimum <= maximum:
        raise ValueError("Extension bounds must be finite, positive and increasing")

    extend_lower = minimum < old_values[0]
    extend_upper = maximum > old_values[-1]
    if not extend_lower and not extend_upper:
        return old_values.copy()
    if old_values.size < 2:
        raise ValueError("Axis extension needs at least two old nodes for log spacing")

    logarithmic_values = np.log10(old_values)
    logarithmic_step = (
        logarithmic_values[-1] - logarithmic_values[0]
    ) / (old_values.size - 1)
    if logarithmic_step <= 0:
        raise ValueError("Old coordinates are too close to determine logarithmic spacing")
    if not np.allclose(
        np.diff(logarithmic_values),
        logarithmic_step,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Axis extension needs uniformly logarithmic old nodes")

    lower_nodes = np.empty(0)
    if extend_lower:
        lower_nodes = new_lower_axis_nodes(
            first_old_value=old_values[0],
            logarithmic_step=logarithmic_step,
            minimum=minimum,
        )
    upper_nodes = np.empty(0)
    if extend_upper:
        upper_nodes = new_upper_axis_nodes(
            last_old_value=old_values[-1],
            logarithmic_step=logarithmic_step,
            maximum=maximum,
        )

    # Concatenating the original middle preserves every existing float64 value.
    return np.concatenate((lower_nodes, old_values, upper_nodes))


def new_lower_axis_nodes(
    *,
    first_old_value: float,
    logarithmic_step: float,
    minimum: float,
) -> np.ndarray:
    """Return only new lower nodes, increasing toward the first old value."""
    first_logarithm = np.log10(first_old_value)
    distance_in_steps = (first_logarithm - np.log10(minimum)) / logarithmic_step
    node_count = max(1, int(np.ceil(distance_in_steps)))
    while True:
        steps = np.arange(node_count, 0, -1, dtype=float)
        with np.errstate(under="ignore"):
            nodes = np.power(10.0, first_logarithm - logarithmic_step * steps)
        if np.any(nodes <= 0):
            raise ValueError("Lower axis extension exceeds positive float64 values")
        if nodes[0] <= minimum:
            return nodes
        # A rounded log10 endpoint may otherwise miss the requested bound.
        node_count += 1


def new_upper_axis_nodes(
    *,
    last_old_value: float,
    logarithmic_step: float,
    maximum: float,
) -> np.ndarray:
    """Return only new upper nodes, increasing away from the last old value."""
    last_logarithm = np.log10(last_old_value)
    distance_in_steps = (np.log10(maximum) - last_logarithm) / logarithmic_step
    node_count = max(1, int(np.ceil(distance_in_steps)))
    while True:
        steps = np.arange(1, node_count + 1, dtype=float)
        with np.errstate(over="ignore"):
            nodes = np.power(10.0, last_logarithm + logarithmic_step * steps)
        if not np.isfinite(nodes).all():
            raise ValueError("Upper axis extension exceeds finite float64 values")
        if nodes[-1] >= maximum:
            return nodes
        node_count += 1


def extend_table_axes(
    source_table: DespoticTable,
    bounds: Mapping[str, Mapping[str, float]],
) -> tuple[ExplicitGrid, ExplicitGrid, ExplicitGrid]:
    """Return extended (nH, NH, dVdr) grids, retaining exact old coordinates.

    source_table comes from read_raw_table_for_extension(). bounds uses the
    snapshot-domain axes format, e.g. bounds['nH']={'minimum': 0.1,
    'maximum': 1000}. Omitted axes/endpoints retain their old bounds.
    Units are cm^-3, cm^-2 and s^-1 respectively.
    Each returned ExplicitGrid is the complete extended axis, not just new nodes.
    """
    source_axes = (
        ("nH", source_table.nH_values),
        ("NH", source_table.col_density_values),
        ("dVdr", source_table.dVdr_values),
    )
    grids = []
    for name, old_values in source_axes:
        requested_bounds = bounds.get(name, {})
        new_values = extend_logarithmic_axis(
            old_values=old_values,
            minimum=float(requested_bounds.get("minimum", old_values[0])),
            maximum=float(requested_bounds.get("maximum", old_values[-1])),
        )
        grids.append(ExplicitGrid(tuple(new_values)))
    return grids[0], grids[1], grids[2]


def map_old_axis_positions(
    old_values: np.ndarray,
    new_values: np.ndarray,
) -> np.ndarray:
    """Return each exact old coordinate's index in the new increasing axis.

    Both inputs are one-dimensional physical axes in matching units. The
    returned integer array has the old axis shape (N,). No near-match or
    interpolation is allowed. Example: old=[1, 10], new=[0.1, 1, 10, 100]
    returns [1, 2], even though both old indices have moved.
    """
    old_values = np.asarray(old_values, dtype=float)
    new_values = np.asarray(new_values, dtype=float)
    positions = np.searchsorted(new_values, old_values)
    if np.any(positions >= new_values.size):
        raise ValueError("Extended axis does not contain every exact old coordinate")
    if not np.array_equal(new_values[positions], old_values):
        raise ValueError("Extended axis does not contain every exact old coordinate")
    return positions


def read_raw_table_for_extension(path: str | Path) -> DespoticTable:
    """Load a raw DESPOTIC NPZ read-only; reject numerically filled derivatives.

    path names an existing raw.npz. Returned axes/fields are the normal
    DespoticTable arrays; failure_mask retains failed old points so they can
    remain missing without another solver attempt. Nothing is written here.
    """
    with np.load(Path(path), allow_pickle=True) as source:
        if INTERPOLATION_FIELDS.intersection(source.files):
            raise ValueError("Extend the raw DESPOTIC table, not an interpolated table")
        if "failure_mask" not in source.files:
            raise ValueError("Raw extension requires the original solver failure_mask")
        metadata = {}
        if "build_metadata_json" in source.files:
            metadata = json.loads(str(source["build_metadata_json"].item()))
        artifact_kind = str(metadata.get("artifact_kind", "")).casefold()
        interpolation_metadata = (
            "interpolation_processing" in metadata
            or "source_table_build_metadata" in metadata
            or "interpolated_nodes" in metadata.get("final_validation", {})
            or "interpolat" in artifact_kind
        )
        if interpolation_metadata:
            raise ValueError("Extend the raw DESPOTIC table, not an interpolated table")
    return load_table(path)


def validate_reuse_settings(
    source_table: DespoticTable,
    current_metadata: Mapping[str, object],
    species_specs: Sequence[SpeciesSpec] | None = None,
) -> None:
    """Require the old point solutions to use the current solver and species.

    current_metadata is returned by validated_solver_metadata(); species_specs
    is the builder's SpeciesSpec sequence. Snapshot bounds, sampling and reuse
    records may differ because they do not change one point's physical solve.
    Raises ValueError on a missing/different solver setting; returns None.
    """
    expected_modes = (
        (source_table.chemistry_network, "GOW", "chemistry network"),
        (source_table.escape_geometry, "LVG", "escape geometry"),
        (source_table.temperature_mode, "iterateDust", "temperature mode"),
    )
    for actual, expected, description in expected_modes:
        if actual != expected:
            raise ValueError(f"Cannot reuse a table with a different {description}")

    source_metadata = source_table.build_metadata
    if source_metadata is None:
        raise ValueError("Raw table has no solver settings to verify before reuse")
    for name in SOLVER_METADATA_FIELDS:
        if name not in source_metadata or name not in current_metadata:
            raise ValueError(f"Missing DESPOTIC reuse setting: {name}")
        if source_metadata[name] != current_metadata[name]:
            raise ValueError(f"DESPOTIC reuse settings differ: {name}")

    if species_specs is not None:
        expected_species = [(spec.name, spec.is_emitter) for spec in species_specs]
        if ("CO", True) in expected_species:
            expected_species.append(("CO21", True))
        actual_species = [
            (name, source_table.species_data[name].is_emitter)
            for name in source_table.species
        ]
        if actual_species != expected_species:
            raise ValueError("DESPOTIC reuse species or emitter roles differ")
