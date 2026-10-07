"""Fill failed DESPOTIC nodes inside the valid-data convex hull.

The original failure_mask records solver provenance. Numerical availability is
recorded separately because most failed nodes can be interpolated. Successful
solver results, including high-temperature results, are preserved bit for bit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.interpolate import griddata

from quokka2s.despotic.table_data import LINE_RESULT_FIELDS
from quokka2s.file_provenance import file_sha256
from quokka2s.paths import resolve_path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "inputs" / "tables" / "despotic" / "raw.npz"
DEFAULT_OUTPUT = ROOT / "inputs" / "tables" / "despotic" / "interpolated.npz"
CORE_FIELDS = ("tg_final", "mu_values", "cv_values", "Eint_values")
# Frequencies are restored from existing nodes, never numerically interpolated.
LINE_VALUE_FIELDS = tuple(name for name in LINE_RESULT_FIELDS if name != "freq")
METADATA_FIELDS = {
    "version", "chemistry_network", "escape_geometry", "temperature_mode",
    "nH_values", "col_density_values", "dVdr_values", "species_names",
    "species_is_emitter", "attempts", "failure_mask", "build_metadata_json",
    "energy_term_names",
}


def _fill_in_hull(values: np.ndarray, log_axes: tuple[np.ndarray, ...]) -> np.ndarray:
    """Use the accepted log-coordinate, 3D linear interpolation policy."""
    finite = np.isfinite(values)
    if not finite.any() or finite.all():
        return values.copy()
    use_log = bool((values[finite] > 0).all())
    if use_log:
        with np.errstate(divide="ignore", invalid="ignore"):
            work = np.where(values > 0, np.log10(values), np.nan)
    else:
        work = values.copy()
    points = np.array(np.meshgrid(*log_axes, indexing="ij")).reshape(3, -1).T
    flat = work.ravel().copy()
    known = np.isfinite(flat)
    flat[~known] = griddata(points[known], flat[known], points[~known], method="linear")
    filled = flat.reshape(values.shape)
    return 10 ** filled if use_log else filled


def _table_fields(
    data: dict[str, np.ndarray], shape: tuple[int, ...]
) -> tuple[list[str], list[str]]:
    """List each solver field explicitly and reject incomplete/new table schemas."""
    species = [str(name) for name in data["species_names"]]
    emitters = np.asarray(data["species_is_emitter"], dtype=bool)
    if len(species) != len(emitters) or len(set(species)) != len(species):
        raise ValueError("Invalid species_names/species_is_emitter metadata")

    fields = list(CORE_FIELDS)
    frequencies: list[str] = []
    for name, is_emitter in zip(species, emitters):
        fields.append(f"{name}_abundance")
        if is_emitter:
            frequencies.append(f"{name}_freq")
            fields.extend(f"{name}_{field}" for field in LINE_VALUE_FIELDS)
    if "energy_term_names" in data:
        fields.extend(f"energy::{name}" for name in data["energy_term_names"])

    expected = set(fields + frequencies) | METADATA_FIELDS
    unexpected = set(data) - expected
    if unexpected:
        raise ValueError(f"Unknown raw-table fields: {sorted(unexpected)}")
    if len(set(fields + frequencies)) != len(fields + frequencies):
        raise ValueError("Duplicate physical field in raw-table metadata")
    for key in fields + frequencies:
        if key not in data:
            raise ValueError(f"Missing raw-table field: {key}")
        if data[key].shape != shape:
            raise ValueError(f"Raw-table field {key} has shape {data[key].shape}, expected {shape}")
    return fields, frequencies


def interpolate_table(source: Path, output: Path, *, force: bool = False) -> dict[str, int]:
    """Write a numerical derivative of a raw table, keeping solved nodes exact."""
    source = resolve_path(source)
    output = resolve_path(output)
    if source == output:
        raise ValueError("Source and output must be different files")
    if output.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --force to replace it")
    source_hash = file_sha256(source)
    with np.load(source, allow_pickle=True) as raw:
        data = {key: raw[key].copy() for key in raw.files}
    if "interpolation_target_mask" in data:
        raise ValueError("Source is already an interpolated table")
    shape = data["tg_final"].shape
    failure = np.asarray(data["failure_mask"], dtype=bool)
    if failure.shape != shape:
        raise ValueError("failure_mask shape differs from tg_final")
    target = failure | ~np.isfinite(data["tg_final"]) | ~np.isfinite(data["mu_values"])
    log_axes = tuple(np.log10(data[key]) for key in
                     ("nH_values", "col_density_values", "dVdr_values"))
    fields, frequencies = _table_fields(data, shape)
    for index, key in enumerate(fields, start=1):
        original = np.asarray(data[key], dtype=float)
        if not np.isfinite(original[~target]).all():
            raise ValueError(f"Independent nonfinite solved values in {key}")
        work = original.copy()
        work[target] = np.nan
        filled = _fill_in_hull(work, log_axes)
        filled[~target] = original[~target]
        data[key] = filled
        if index == 1 or index % 10 == 0 or index == len(fields):
            print(f"[interpolate_failed] fields {index}/{len(fields)}", flush=True)

    for key in frequencies:
        frequency = np.asarray(data[key], dtype=float).copy()
        known = np.isfinite(frequency)
        if not known.any():
            raise ValueError(f"No finite transition frequency in {key}")
        reference = float(frequency[known][0])
        if not np.allclose(frequency[known], reference, rtol=1e-12, atol=0):
            raise ValueError(f"Transition frequency varies in {key}")
        frequency[~known] = reference
        data[key] = frequency

    for key in CORE_FIELDS:
        finite = np.isfinite(data[key])
        if not np.all(data[key][finite] > 0):
            raise ValueError(f"Nonpositive finite values in {key}")
    for key in (key for key in fields if key.endswith("_abundance")):
        finite = np.isfinite(data[key])
        if not np.all(data[key][finite] >= 0):
            raise ValueError(f"Negative finite abundance in {key}")

    available_t_mu = np.isfinite(data["tg_final"]) & np.isfinite(data["mu_values"])
    all_physical = np.logical_and.reduce(
        [np.isfinite(data[key]) for key in fields + frequencies]
    )
    masks = {
        "source_failure_mask": failure.copy(),
        "interpolation_target_mask": target,
        "filled_T_mu_node_mask": target & available_t_mu,
        "filled_all_physical_fields_node_mask": target & all_physical,
        "remaining_unavailable_T_mu_node_mask": ~available_t_mu,
        "remaining_unavailable_any_physical_field_node_mask": ~all_physical,
    }
    data.update(masks)
    counts = {key: int(value.sum()) for key, value in masks.items()}
    original_metadata = (json.loads(str(data["build_metadata_json"].item()))
                         if "build_metadata_json" in data else {})
    metadata = dict(original_metadata)
    metadata["artifact_kind"] = "Convex-hull interpolated DESPOTIC table"
    metadata["source_table_build_metadata"] = original_metadata
    metadata["interpolation_processing"] = {
        "source_table_sha256": source_hash,
        "coordinate_space": "log10(nH), log10(NH), log10(dVdr)",
        "interpolation": "linear griddata inside the finite-support convex hull; no extrapolation",
        "values": "log10 for strictly positive support; linear otherwise",
        "target_policy": "source failure_mask or nonfinite source T/mu",
        "solved_values": "preserved bitwise, including Tg > 1e6 K",
        "failure_mask": "unchanged source solver-failure provenance",
        "equilibrium_validation": "interpolated nodes were not solved or equilibrium-validated",
        "node_counts": counts,
    }
    metadata["final_validation"] = {
        "scope": "Only directly solved source nodes were solver-validated.",
        "source_solver_validation_policy": original_metadata.get("final_validation"),
        "interpolated_nodes": "No equilibrium solve; interpolated validation fields are numerical values only.",
    }
    data["build_metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))

    if file_sha256(source) != source_hash:
        raise ValueError("Source table changed during interpolation")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
    ) as temporary:
        staged = Path(temporary.name)
        try:
            np.savez_compressed(temporary, **data)
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
    try:
        with np.load(staged, allow_pickle=True) as reloaded:
            if set(reloaded.files) != set(data):
                raise ValueError("Staged table has an incomplete schema")
        if output.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite {output}; pass --force to replace it")
        staged.replace(output)
    finally:
        staged.unlink(missing_ok=True)
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="replace an existing output")
    args = parser.parse_args(argv)
    counts = interpolate_table(args.source, args.output, force=args.force)
    print(f"[interpolate_failed] saved {args.output}")
    for name, count in counts.items():
        print(f"[interpolate_failed] {name}: {count}")


if __name__ == "__main__":
    main()
