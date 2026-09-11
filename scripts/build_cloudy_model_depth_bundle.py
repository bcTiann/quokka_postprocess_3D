#!/usr/bin/env python3
"""Pack completed fixed-depth CIAOLoop maps without filling failed nodes.

The manifest supplies the exact grid axes. Map headers independently identify
each density, attenuating column, and model thickness; file order is irrelevant.
A completed map means every requested temperature was attempted, not that every
Cloudy calculation converged. One-column crash rows remain masked as failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import tempfile
from pathlib import Path

import numpy as np

from quokka2s.tables.abundances import abundance_metadata


AXIS_NAMES = ("log_NH_attenuation", "log_nH", "log_T", "log_L_model_pc")
AXIS_ORDER = "line," + ",".join(AXIS_NAMES)
LINES = (
    ("cii", "C  2 157.636m", "C_2_157.636m"),
    ("halpha", "H  1 6562.81A", "H_1_6562.81A"),
    ("hi21", "H  1 21.1207c", "H_1_21.1207c"),
    ("ciii_977", "C  3 977.020A", "C_3_977.020A"),
    ("ciii_1907", "C  3 1906.68A", "C_3_1906.68A"),
    ("ciii_1909", "C  3 1908.73A", "C_3_1908.73A"),
    ("civ_1548", "C  4 1548.19A", "C_4_1548.19A"),
    ("civ_1551", "C  4 1550.78A", "C_4_1550.78A"),
)
T_TOLERANCE_DEX = 5.1e-4  # CIAOLoop prints temperatures to three decimal places.
COORDINATE_TOLERANCE_DEX = 1.0e-12
ZERO_SENTINEL = -99.0
INNER_RADIUS_CM = 1.0e30


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _axis_index(value: float, axis: np.ndarray, name: str, tolerance: float) -> int:
    if not np.isfinite(value):
        raise ValueError(f"non-finite {name} coordinate")
    distances = np.abs(axis - value)
    matches = np.flatnonzero(distances <= tolerance)
    if matches.size != 1:
        raise ValueError(f"off-grid or ambiguous {name} coordinate {value}")
    return int(matches[0])


def _same_coordinate(actual: float, expected: float, name: str, path: Path) -> None:
    if not np.isfinite(actual) or abs(actual - expected) > COORDINATE_TOLERANCE_DEX:
        raise ValueError(f"{name} mismatch in {path}: {actual} versus {expected}")


def _validate_parameter_file(path: Path, axes: dict, pc_in_cm: float) -> dict:
    """Check the settings that establish the bundle's physical interpretation."""
    assignments: dict[str, list[str]] = {}
    commands: list[list[str]] = []
    radius_loops: list[tuple[list[str], list[str]]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("command "):
            commands.append(shlex.split(line[8:], comments=True))
            continue
        match = re.fullmatch(r"loop\s+\[(.*?)\]\s+(.*?)\s*(?:#.*)?", line)
        if match:
            template, values = shlex.split(match[1]), shlex.split(match[2], comments=True)
            if template and template[0].lower() == "radius":
                radius_loops.append((template, values))
            elif template and template[0].lower() in ("abundances", "element", "metals", "grains"):
                raise ValueError("composition-changing loops are incompatible with the shared composition")
            continue
        if "=" in line:
            name, value = line.split("=", 1)
            assignments.setdefault(name.strip(), []).append(value.split("#", 1)[0].strip())

    for name, value in (("coolingMapUseJeansLength", 0), ("cloudyRunMode", 4), ("coolingScaleFactor", 1)):
        candidates = assignments.get(name, [])
        if len(candidates) != 1 or float(candidates[0]) != value:
            raise ValueError(f"parameter file requires exactly one {name} = {value}")
    line_settings = assignments.get("lineMapLine", [])
    if [" ".join(value.split()) for value in line_settings] != [
        " ".join(item[1].split()) for item in LINES
    ]:
        raise ValueError("parameter file lineMapLine definitions do not match the eight lines")

    abundance = abundance_metadata()
    relevant = [(i, tokens) for i, tokens in enumerate(commands)
                if tokens and tokens[0].lower() in ("abundances", "element", "metals")]
    if len(relevant) != 3:
        raise ValueError("parameter file requires exactly default.abn, helium, and metals commands")
    default = [(i, tokens) for i, tokens in relevant if tokens[0].lower() == "abundances"]
    helium = [(i, tokens) for i, tokens in relevant if tokens[0].lower() == "element"]
    metals = [(i, tokens) for i, tokens in relevant if tokens[0].lower() == "metals"]
    if len(default) != 1 or default[0][1] != ["abundances", "default.abn"]:
        raise ValueError('parameter file must explicitly use abundances "default.abn"')
    expected_helium = abundance["gow_elemental_abundances"]["xHe"]
    for records, expected_prefix, expected_value, label in (
        (helium, ["element", "helium", "abundance"], expected_helium, "helium"),
        (metals, ["metals"], abundance["metal_reference_scale"], "metals"),
    ):
        if len(records) != 1:
            raise ValueError(f"parameter file requires one {label} command")
        index, tokens = records[0]
        if (len(tokens) != len(expected_prefix) + 2
                or [token.lower() for token in tokens[:-2]] != expected_prefix
                or tokens[-1].lower() != "linear"
                or not np.isclose(float(tokens[-2]), expected_value, rtol=1e-13, atol=0)
                or index < default[0][0]):
            raise ValueError(f"parameter file {label} command differs from shared composition")

    if len(radius_loops) != 1:
        raise ValueError("parameter file requires exactly one fixed-depth radius loop")
    template, values = radius_loops[0]
    if (len(template) != 4 or float(template[1]) != INNER_RADIUS_CM
            or template[2:] != ["*", "linear"]):
        raise ValueError("radius loop must be [radius 1e30 * linear]")
    lengths = np.asarray(values, dtype=float)
    if lengths.size != axes["log_L_model_pc"].size or np.any(lengths <= 0) or not np.isfinite(lengths).all():
        raise ValueError("radius loop length values are incomplete or invalid")
    if not np.allclose(np.sort(np.log10(lengths / pc_in_cm)), axes["log_L_model_pc"],
                       rtol=0, atol=COORDINATE_TOLERANCE_DEX):
        raise ValueError("radius loop length values differ from manifest axis")
    for tokens in commands:
        lowered = [token.lower() for token in tokens]
        if lowered and (lowered[0] in ("radius", "grains") or
                        lowered[:2] in (["stop", "column"], ["stop", "thickness"])):
            raise ValueError(f"conflicting geometry/composition command: {' '.join(tokens)}")
    return {"parameter_settings_checked": True, "internal_jeans_length_enabled": False,
            "shared_abundance_commands_checked": True, "fixed_depth_loop_checked": True}


def _parse_map(path: Path, pc_in_cm: float) -> tuple[dict, dict[float, np.ndarray | None]]:
    headers: dict[str, list] = {name: [] for name in ("log_nH", "log_NH_attenuation", "log_L_model_pc", "lines")}
    rows: dict[float, np.ndarray | None] = {}
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if line.startswith("#"):
            body = line[1:].strip()
            # Prose comments need not contain balanced shell-style quotation.
            if not body or body.split(maxsplit=1)[0] not in ("hden", "init", "radius", "Te"):
                continue
            tokens = shlex.split(body)
            if not tokens:
                continue
            if tokens[0] == "hden":
                if len(tokens) != 2:
                    raise ValueError(f"unexpected hden header: {path}:{number}")
                headers["log_nH"].append(float(tokens[1]))
            elif tokens[0] == "init":
                if len(tokens) != 2:
                    raise ValueError(f"unexpected init header: {path}:{number}")
                match = re.search(r"(?:^|/)logNH([-+0-9.eE]+)\.out$", tokens[1])
                if match:
                    headers["log_NH_attenuation"].append(float(match[1]))
            elif tokens[0] == "radius":
                if len(tokens) != 4 or tokens[-1] != "linear" or float(tokens[1]) != INNER_RADIUS_CM:
                    raise ValueError(f"unexpected radius header: {path}:{number}")
                thickness = float(tokens[2])
                if not np.isfinite(thickness) or not 0 < thickness < INNER_RADIUS_CM:
                    raise ValueError(f"invalid radius thickness: {path}:{number}")
                headers["log_L_model_pc"].append(float(np.log10(thickness / pc_in_cm)))
            elif tokens[0] == "Te":
                headers["lines"].append(tuple(tokens[1:]))
            continue
        if not line:
            continue
        columns = line.split()
        if len(columns) not in (1, len(LINES) + 1):
            raise ValueError(f"incomplete or malformed map row: {path}:{number}")
        log_t = float(columns[0])
        if not np.isfinite(log_t) or log_t in rows:
            raise ValueError(f"duplicate or non-finite temperature: {path}:{number}")
        rows[log_t] = np.asarray(columns[1:], dtype=float) if len(columns) > 1 else None
    if any(len(value) != 1 for value in headers.values()):
        raise ValueError(f"missing or repeated coordinate/line headers: {path}")
    if headers["lines"][0] != tuple(item[2] for item in LINES):
        raise ValueError(f"unexpected line header: {path}")
    return {name: values[0] for name, values in headers.items() if name != "lines"}, rows


def pack_table(manifest_path: Path, output_path: Path) -> dict:
    """Validate and package a completed manifest; refuse to replace any product."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    report_path = output_path.with_suffix(".failure_report.json")
    if output_path.suffix != ".npz":
        raise ValueError("output must use the .npz suffix")
    if output_path.exists() or report_path.exists():
        raise FileExistsError(f"output or failure report already exists: {output_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("axis_order") != AXIS_ORDER:
        raise ValueError("unexpected manifest schema version or axis order")
    if manifest.get("cloudy_version") != "17.02":
        raise ValueError("manifest must identify Cloudy 17.02")
    if manifest.get("status") != "completed":
        raise ValueError("manifest status must be completed before packaging")
    if manifest.get("abundance") != abundance_metadata():
        raise ValueError("manifest abundance differs from the shared composition")
    pc_in_cm = float(manifest["pc_in_cm"])
    if not np.isfinite(pc_in_cm) or pc_in_cm <= 0:
        raise ValueError("manifest pc_in_cm must be finite and positive")
    axes = {name: np.asarray(manifest["axes"][name], dtype=float) for name in AXIS_NAMES}
    for name, axis in axes.items():
        if axis.ndim != 1 or not axis.size or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError(f"{name} must be a finite, strictly increasing nonempty axis")
    parameter_path = _path(manifest["parameter_file"], manifest_path.parent)
    physical_checks = _validate_parameter_file(parameter_path, axes, pc_in_cm)
    maps = manifest["maps"]
    shape = tuple(axes[name].size for name in AXIS_NAMES)
    expected_maps = shape[0] * shape[1] * shape[3]
    if len(maps) != expected_maps:
        raise ValueError(f"incomplete map list: expected {expected_maps}, found {len(maps)}")
    paths = [_path(record["path"], manifest_path.parent) for record in maps]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate map paths in manifest")
    for directory in set(path.parent for path in paths):
        if any(directory.glob("*.mach")):
            raise RuntimeError(f"CIAOLoop .mach marker remains; maps may be active: {directory}")

    raw = np.full((len(LINES), *shape), np.nan)
    seen = np.zeros((shape[0], shape[1], shape[3]), dtype=bool)
    sources = []
    for record, path in zip(maps, paths, strict=True):
        coords, rows = _parse_map(path, pc_in_cm)
        indices = []
        for name in ("log_NH_attenuation", "log_nH", "log_L_model_pc"):
            expected = float(record[name])
            index = _axis_index(expected, axes[name], name, COORDINATE_TOLERANCE_DEX)
            _same_coordinate(coords[name], expected, name, path)
            indices.append(index)
        i, j, l = indices
        if seen[i, j, l]:
            raise ValueError(f"duplicate map coordinates: {coords}")
        seen[i, j, l] = True
        if len(rows) != shape[2]:
            raise ValueError(f"incomplete temperatures in {path}: expected {shape[2]}, found {len(rows)}")
        seen_t = np.zeros(shape[2], dtype=bool)
        crash_temperatures = []
        for log_t, values in rows.items():
            k = _axis_index(log_t, axes["log_T"], "temperature", T_TOLERANCE_DEX)
            if seen_t[k]:
                raise ValueError(f"multiple rows map to the same temperature in {path}")
            seen_t[k] = True
            if values is not None:
                raw[:, i, j, k, l] = np.where(np.isfinite(values), values, np.nan)
            else:
                crash_temperatures.append(k)
        if not seen_t.all():
            raise ValueError(f"incomplete temperature coverage: {path}")
        sources.append({"path": str(path), "sha256": _sha256(path), "coordinates": coords,
                        "one_column_crash_temperature_indices": crash_temperatures})
    if not seen.all():
        raise ValueError("map grid is not a complete Cartesian product")

    failure = ~np.isfinite(raw)
    zero = (~failure) & (raw == ZERO_SENTINEL)
    coefficient = np.zeros_like(raw)
    positive = ~failure & ~zero
    with np.errstate(over="ignore", under="ignore"):
        coefficient[positive] = 10.0 ** raw[positive]
    if not np.isfinite(coefficient).all() or np.any(coefficient[positive] <= 0):
        raise ValueError("finite nonzero log coefficient overflows or underflows float64")
    union_failure = np.any(failure, axis=0)
    provenance = {
        "manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
        "parameter_file": str(parameter_path), "parameter_file_sha256": _sha256(parameter_path),
        "maps": sources, "manifest_provenance": manifest.get("provenance", {}),
        "validation": {**physical_checks, "complete_coordinate_and_temperature_coverage": True,
                       "cloudy_convergence_independently_verified": False,
                       "interpretation": "Map rows show attempted states; finite values retain CIAOLoop output without additional convergence certification."},
    }
    report = {
        "product": str(output_path), "shape": list(raw.shape), "axis_order": AXIS_ORDER,
        "map_count": len(maps), "union_failure_nodes": int(union_failure.sum()),
        "line_failure_counts": {item[0]: int(failure[index].sum()) for index, item in enumerate(LINES)},
        "true_zero_line_values": int(zero.sum()), "failed_node_policy": "unavailable; no numerical fill",
        "failure_nodes": [{"indices": [int(value) for value in index],
                           **{name: float(axes[name][index[a]]) for a, name in enumerate(AXIS_NAMES)},
                           "failed_lines": [LINES[q][0] for q in range(len(LINES)) if failure[(q, *index)]]}
                          for index in np.argwhere(union_failure)],
        "provenance": provenance,
    }
    payload = {
        "schema_version": np.asarray(4, dtype=np.int32), "axis_order": np.asarray(AXIS_ORDER),
        "cloudy_version": np.asarray("17.02"), **axes,
        "line_keys": np.asarray([item[0] for item in LINES]),
        "line_labels": np.asarray([item[1] for item in LINES]),
        "log_emissivity_per_nH2": raw, "emissivity_per_nH2": coefficient,
        "failure_mask": failure, "original_failure_mask": failure.copy(), "zero_mask": zero,
        "interpolated_mask": np.zeros_like(failure), "zero_sentinel": np.asarray(ZERO_SENTINEL),
        "geometry": np.asarray("fixed model thickness; radius 1e30 L_cm linear"),
        "pc_in_cm": np.asarray(pc_in_cm), "inner_radius_cm": np.asarray(INNER_RADIUS_CM),
        "internal_jeans_length_enabled": np.asarray(False),
        "normalization": np.asarray("local deepest-zone emissivity / n_H^2"),
        "composition_label": np.asarray(abundance_metadata()["setup"]),
        "abundance_json": np.asarray(_json(manifest["abundance"])),
        "radiation_json": np.asarray(_json(manifest.get("radiation", {}))),
        "provenance_json": np.asarray(_json(provenance)), "manifest_json": np.asarray(_json(manifest)),
        "parameter_file": np.asarray(str(parameter_path)),
        "parameter_file_sha256": np.asarray(provenance["parameter_file_sha256"]),
        "manifest_sha256": np.asarray(provenance["manifest_sha256"]),
        "simulation_NH_policy": np.asarray("preserve original cell NH; clip only attenuation lookup coordinate to table bounds; no extrapolation"),
        "density_temperature_depth_out_of_bounds_policy": np.asarray("raise"),
        "failed_node_policy": np.asarray("unavailable; no numerical fill"),
        "interpolation_policy": np.asarray("multilinear in log coefficient for positive corners; linear coefficient if a true-zero corner contributes; raise on positive-weight failure"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **payload)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    report["product_sha256"] = _sha256(output_path)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return {"output_path": str(output_path), "report_path": str(report_path),
            "shape": list(raw.shape), "map_count": len(maps),
            "union_failure_nodes": report["union_failure_nodes"],
            "true_zero_line_values": report["true_zero_line_values"],
            "product_sha256": report["product_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(pack_table(args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()
