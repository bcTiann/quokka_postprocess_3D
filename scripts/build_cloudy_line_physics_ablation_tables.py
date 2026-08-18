"""Build the four-state HM2012 line-physics ablation lookup tables.

The input is four completed 10 x 10 CIAOLoop_lines grids.  Every Cloudy run
contains 20 temperatures and three output columns: [C II] 158 micron,
H-alpha, and H I 21 cm.  A successful non-positive line emissivity is written
by CIAOLoop_lines as -99 and is a real zero.  A crashed calculation instead
leaves an absent or short data row; those nodes remain unavailable failures.

Two products are written:

* one canonical five-dimensional bundle with axis order
  (state, line, log_nH, log_NH, log_T); and
* twelve schema-2 three-dimensional views (one per state and line) that are
  directly readable by quokka2s.cloudy_cii_lookup.CloudyCIILookup.

No failed node is interpolated or otherwise filled by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


STATE_LABELS = ("baseline", "mol", "ct", "mol_ct")
LINE_KEYS = ("cii", "halpha", "hi21")
LINE_LABELS = (
    "C  2 157.636m",
    "H  1 6562.81A",
    "H  1 21.1207c",
)
DAT_LINE_LABELS = (
    "C_2_157.636m",
    "H_1_6562.81A",
    "H_1_21.1207c",
)

EXPECTED_NH_POINTS = 10
EXPECTED_COLUMN_POINTS = 10
EXPECTED_T_POINTS = 20
T_MIN_K = 1.0e1
T_MAX_K = 1.0e9
ZERO_SENTINEL_MAX = -90.0
TEMPERATURE_TOLERANCE_DEX = 5.1e-4

LOOP_RE = re.compile(r"^#\s*(hden|stop column density)\s+(.+?)\s*$")
HEADER_RE = re.compile(r"^#Te\s+(.+?)\s*$")
RUN_DATA_RE = re.compile(r"_run([1-9][0-9]*)\.dat$")

# These are command-presence facts, rather than a claim that Cloudy's
# detailed H2 model was explicitly enabled when the disabling command is
# absent.
EXPECTED_NO_H2_COMMAND = {
    "baseline": True,
    "mol": False,
    "ct": True,
    "mol_ct": False,
}
EXPECTED_NO_CT_COMMAND = {
    "baseline": True,
    "mol": True,
    "ct": False,
    "mol_ct": False,
}


@dataclass(frozen=True)
class RunData:
    log_nH: float
    log_NH: float
    values_by_temperature: dict[float, np.ndarray]
    short_temperatures: tuple[float, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalised_parameter_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            lines.append(" ".join(line.lower().split()))
    return lines


def _validate_parameter_file(path: Path, state: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing parameter file for {state}: {path}")
    lines = _normalised_parameter_lines(path)
    no_h2 = any(line == "command no h2 molecule" for line in lines)
    no_ct = any(line == "command no charge transfer" for line in lines)
    if no_h2 != EXPECTED_NO_H2_COMMAND[state]:
        raise ValueError(
            f"{state} parameter file has unexpected 'no H2 molecule' state: "
            f"{path}"
        )
    if no_ct != EXPECTED_NO_CT_COMMAND[state]:
        raise ValueError(
            f"{state} parameter file has unexpected 'no charge transfer' "
            f"state: {path}"
        )

    line_commands: list[str] = []
    for line in lines:
        if not line.startswith("linemapline"):
            continue
        value = line[len("linemapline"):].strip()
        if value.startswith("="):
            value = value[1:].strip()
        line_commands.append(value)
    expected = [" ".join(label.lower().split()) for label in LINE_LABELS]
    if line_commands != expected:
        raise ValueError(
            f"{state} parameter file lineMapLine commands differ from the "
            f"expected three-line order: {path}"
        )


def _parse_run(path: Path) -> RunData:
    """Parse one run without converting absent/short rows into zeros."""
    log_nH: float | None = None
    log_NH: float | None = None
    header_labels: tuple[str, ...] | None = None
    values_by_temperature: dict[float, np.ndarray] = {}
    short_temperatures: list[float] = []
    seen_temperatures: set[float] = set()

    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        match = LOOP_RE.match(raw_line)
        if match:
            if match.group(1) == "hden":
                log_nH = float(match.group(2))
            else:
                log_NH = float(match.group(2))
            continue

        header_match = HEADER_RE.match(raw_line)
        if header_match:
            header_labels = tuple(header_match.group(1).split())
            continue
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        columns = raw_line.split()
        try:
            log_temperature = float(columns[0])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"invalid data row at {path}:{line_number}: {raw_line!r}"
            ) from exc
        if log_temperature in seen_temperatures:
            raise ValueError(
                f"duplicate log(T)={log_temperature} in {path}"
            )
        seen_temperatures.add(log_temperature)

        # CIAOLoop_lines prints exactly the temperature and no emissivities
        # after a Cloudy crash.  Other column counts are file corruption, not
        # a recognised failure representation, and must not be accepted.
        if len(columns) == 1:
            short_temperatures.append(log_temperature)
            continue
        if len(columns) != 1 + len(LINE_LABELS):
            raise ValueError(
                f"unexpected data-column count at {path}:{line_number}: "
                f"expected 1 (explicit failure) or 4 (success), found "
                f"{len(columns)}"
            )
        try:
            values = np.asarray([float(value) for value in columns[1:]], dtype=float)
        except ValueError as exc:
            raise ValueError(
                f"non-numeric emissivity at {path}:{line_number}: {raw_line!r}"
            ) from exc
        if not np.isfinite(values).all():
            short_temperatures.append(log_temperature)
            continue
        values_by_temperature[log_temperature] = values

    if log_nH is None or log_NH is None:
        raise ValueError(f"missing hden/stop-column metadata in {path}")
    if header_labels != DAT_LINE_LABELS:
        raise ValueError(
            f"unexpected or missing data-column header in {path}: "
            f"{header_labels!r}"
        )
    if len(seen_temperatures) != EXPECTED_T_POINTS:
        raise RuntimeError(
            f"incomplete temperature sweep in {path}: expected "
            f"{EXPECTED_T_POINTS} explicit rows, found "
            f"{len(seen_temperatures)}"
        )
    return RunData(
        log_nH=log_nH,
        log_NH=log_NH,
        values_by_temperature=values_by_temperature,
        short_temperatures=tuple(short_temperatures),
    )


def _temperature_index(log_temperature: float, log_T: np.ndarray, path: Path) -> int:
    index = int(np.abs(log_T - log_temperature).argmin())
    residual = abs(float(log_T[index]) - log_temperature)
    if residual > TEMPERATURE_TOLERANCE_DEX:
        raise ValueError(
            f"reported log(T)={log_temperature} is off the requested grid "
            f"by {residual:.6g} dex: {path}"
        )
    return index


def _load_state_directory(
    path: Path,
    log_T: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    if not path.is_dir():
        raise FileNotFoundError(f"missing state output directory: {path}")
    active_markers = sorted(path.glob("*.mach"))
    if active_markers:
        raise RuntimeError(
            f"state output is still active ({len(active_markers)} .mach "
            f"markers remain): {path}"
        )
    files = sorted(path.glob("*_run*.dat"))
    if len(files) != EXPECTED_NH_POINTS * EXPECTED_COLUMN_POINTS:
        raise ValueError(
            f"expected {EXPECTED_NH_POINTS * EXPECTED_COLUMN_POINTS} run files "
            f"in {path}, found {len(files)}"
        )

    run_files: dict[int, Path] = {}
    for file_path in files:
        match = RUN_DATA_RE.search(file_path.name)
        if match is None:
            raise ValueError(f"unexpected run filename: {file_path}")
        run_id = int(match.group(1))
        if run_id in run_files:
            raise ValueError(f"duplicate run id {run_id} in {path}")
        run_files[run_id] = file_path
    expected_run_ids = set(
        range(1, EXPECTED_NH_POINTS * EXPECTED_COLUMN_POINTS + 1)
    )
    if set(run_files) != expected_run_ids:
        raise ValueError(
            f"run ids in {path} are not exactly 1..100; "
            f"missing={sorted(expected_run_ids - set(run_files))[:12]}, "
            f"extra={sorted(set(run_files) - expected_run_ids)[:12]}"
        )

    run_manifests = sorted(path.glob("*.run"))
    if len(run_manifests) != 1:
        raise ValueError(
            f"expected exactly one CIAOLoop .run manifest in {path}, found "
            f"{len(run_manifests)}"
        )
    manifest_rows = [
        line for line in run_manifests[0].read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(manifest_rows) != len(expected_run_ids):
        raise RuntimeError(
            f"CIAOLoop manifest is incomplete in {path}: expected 100 run "
            f"rows, found {len(manifest_rows)}"
        )

    records = [
        (run_id, run_files[run_id], _parse_run(run_files[run_id]))
        for run_id in sorted(run_files)
    ]
    log_nH = np.unique([record.log_nH for _, _, record in records])
    log_NH = np.unique([record.log_NH for _, _, record in records])
    if log_nH.size != EXPECTED_NH_POINTS or log_NH.size != EXPECTED_COLUMN_POINTS:
        raise ValueError(
            f"expected 10x10 axes in {path}, found "
            f"{log_nH.size}x{log_NH.size}"
        )

    expected_keys = {(float(n), float(c)) for n in log_nH for c in log_NH}
    actual_keys = {(record.log_nH, record.log_NH) for _, _, record in records}
    if actual_keys != expected_keys or len(actual_keys) != len(records):
        raise ValueError(
            f"run metadata in {path} is not one complete 10x10 Cartesian grid"
        )

    raw_log = np.full(
        (len(LINE_LABELS), log_nH.size, log_NH.size, log_T.size),
        np.nan,
        dtype=float,
    )
    nH_index = {float(value): index for index, value in enumerate(log_nH)}
    NH_index = {float(value): index for index, value in enumerate(log_NH)}
    short_rows: list[dict[str, object]] = []

    for run_id, file_path, record in records:
        i = nH_index[record.log_nH]
        j = NH_index[record.log_NH]
        expected_run_id = i * EXPECTED_COLUMN_POINTS + j + 1
        if run_id != expected_run_id:
            raise ValueError(
                f"run {run_id} metadata maps to grid ({i},{j}), which should "
                f"be run {expected_run_id}: {file_path}"
            )
        assigned_indices: set[int] = set()
        for run_log_T, values in record.values_by_temperature.items():
            k = _temperature_index(run_log_T, log_T, file_path)
            if k in assigned_indices:
                raise ValueError(f"duplicate temperature grid node in {file_path}")
            assigned_indices.add(k)
            raw_log[:, i, j, k] = values
        for run_log_T in record.short_temperatures:
            k = _temperature_index(run_log_T, log_T, file_path)
            if k in assigned_indices:
                raise ValueError(
                    f"temperature appears as both valid and short row in {file_path}"
                )
            assigned_indices.add(k)
            short_rows.append({
                "path": str(file_path.resolve()),
                "log_nH": float(record.log_nH),
                "log_NH": float(record.log_NH),
                "log_T": float(log_T[k]),
                "T_index": int(k),
            })

    return log_nH, log_NH, raw_log, short_rows


def _view_filename(state: str, line_key: str) -> str:
    return f"cloudy_lines_hm2012_z0_ablation_{state}_{line_key}_10x10x20.npz"


def _write_view(
    path: Path,
    *,
    state: str,
    line_index: int,
    log_nH: np.ndarray,
    log_NH: np.ndarray,
    log_T: np.ndarray,
    raw_log: np.ndarray,
    coefficient: np.ndarray,
    failure_mask: np.ndarray,
    zero_mask: np.ndarray,
    parameter_file: Path,
    parameter_sha256: str,
) -> None:
    np.savez_compressed(
        path,
        schema_version=np.asarray(2, dtype=np.int32),
        log_nH=log_nH,
        log_NH=log_NH,
        log_T=log_T,
        log_emissivity_per_nH2=raw_log,
        emissivity_per_nH2=coefficient,
        failure_mask=failure_mask,
        zero_mask=zero_mask,
        original_failure_mask=failure_mask.copy(),
        interpolated_mask=np.zeros_like(failure_mask, dtype=bool),
        out_of_bounds_policy=np.asarray("raise"),
        line_label=np.asarray(LINE_LABELS[line_index]),
        line_key=np.asarray(LINE_KEYS[line_index]),
        state_label=np.asarray(state),
        uv_background=np.asarray("HM2012 z=0 shielded"),
        cloudy_version=np.asarray("17.02"),
        carbon_abundance_log10=np.asarray(-3.795880),
        no_h2_molecule_command=np.asarray(EXPECTED_NO_H2_COMMAND[state]),
        no_charge_transfer_command=np.asarray(EXPECTED_NO_CT_COMMAND[state]),
        molecular_treatment=np.asarray(
            "disabled by 'no H2 molecule'"
            if EXPECTED_NO_H2_COMMAND[state]
            else "Cloudy default; detailed H2 not explicitly requested"
        ),
        charge_transfer_treatment=np.asarray(
            "disabled by 'no charge transfer'"
            if EXPECTED_NO_CT_COMMAND[state]
            else "Cloudy default enabled treatment"
        ),
        column_model=np.asarray("explicit stop column density; no Jeans length"),
        normalization=np.asarray("local deepest-zone emissivity / n_H^2"),
        interpolation_policy=np.asarray(
            "log coefficient if all 8 corners positive; "
            "linear coefficient if any contributing corner is a true zero"
        ),
        failed_node_policy=np.asarray(
            "unavailable; runtime raises if interpolation weight > 1e-12"
        ),
        zero_sentinel_max=np.asarray(ZERO_SENTINEL_MAX),
        parameter_file=np.asarray(str(parameter_file.resolve())),
        parameter_sha256=np.asarray(parameter_sha256),
    )


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    example_dir = project_root / "work/cloudy_cooling_tools_history/examples/grackle"
    parser = argparse.ArgumentParser(description=__doc__)
    for state in STATE_LABELS:
        parser.add_argument(
            f"--{state.replace('_', '-')}-input",
            type=Path,
            default=example_dir / f"hm_2012_lines_{state}_10x10x20_output",
        )
        parser.add_argument(
            f"--{state.replace('_', '-')}-par",
            type=Path,
            default=example_dir / f"hm_2012_lines_{state}_10x10x20.par",
        )
    parser.add_argument(
        "--bundle-output",
        type=Path,
        default=(
            project_root
            / "data/cloudy_lines_hm2012_z0_physics_ablation_4state_3line_10x10x20.npz"
        ),
    )
    parser.add_argument(
        "--views-dir",
        type=Path,
        default=project_root / "data/cloudy_line_physics_ablation_10x10x20_views",
    )
    parser.add_argument(
        "--failure-manifest",
        type=Path,
        default=(
            project_root
            / "data/cloudy_lines_hm2012_z0_physics_ablation_10x10x20_failures.json"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing bundle, views, and failure manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    state_inputs = {
        state: Path(getattr(args, f"{state}_input")) for state in STATE_LABELS
    }
    parameter_files = {
        state: Path(getattr(args, f"{state}_par")) for state in STATE_LABELS
    }
    for state, path in parameter_files.items():
        _validate_parameter_file(path, state)

    bundle_output = args.bundle_output.resolve()
    views_dir = args.views_dir.resolve()
    failure_manifest = args.failure_manifest.resolve()
    view_paths = {
        (state, line_key): views_dir / _view_filename(state, line_key)
        for state in STATE_LABELS
        for line_key in LINE_KEYS
    }
    products = [bundle_output, failure_manifest, *view_paths.values()]
    existing = [path for path in products if path.exists()]
    if existing and not args.force:
        preview = "\n".join(f"  {path}" for path in existing[:12])
        raise SystemExit(
            "Refusing to overwrite existing ablation products; pass --force "
            f"to replace them:\n{preview}"
        )

    log_T = np.linspace(np.log10(T_MIN_K), np.log10(T_MAX_K), EXPECTED_T_POINTS)
    state_raw: list[np.ndarray] = []
    state_short_rows: dict[str, list[dict[str, object]]] = {}
    reference_log_nH: np.ndarray | None = None
    reference_log_NH: np.ndarray | None = None

    for state in STATE_LABELS:
        log_nH, log_NH, raw_log, short_rows = _load_state_directory(
            state_inputs[state], log_T,
        )
        if reference_log_nH is None:
            reference_log_nH = log_nH
            reference_log_NH = log_NH
        elif not (
            np.array_equal(reference_log_nH, log_nH)
            and np.array_equal(reference_log_NH, log_NH)
        ):
            raise ValueError(f"{state} axes differ from the baseline axes")
        state_raw.append(raw_log)
        state_short_rows[state] = short_rows

    assert reference_log_nH is not None and reference_log_NH is not None
    raw_log = np.stack(state_raw, axis=0)
    expected_shape = (
        len(STATE_LABELS), len(LINE_LABELS), EXPECTED_NH_POINTS,
        EXPECTED_COLUMN_POINTS, EXPECTED_T_POINTS,
    )
    if raw_log.shape != expected_shape:
        raise AssertionError(f"internal grid shape {raw_log.shape} != {expected_shape}")

    original_failure_mask = ~np.isfinite(raw_log)
    failure_mask = original_failure_mask.copy()
    zero_mask = (~failure_mask) & (raw_log <= ZERO_SENTINEL_MAX)
    coefficient = np.zeros(raw_log.shape, dtype=float)
    positive = (~failure_mask) & (~zero_mask)
    coefficient[positive] = np.power(10.0, raw_log[positive])
    interpolated_mask = np.zeros_like(failure_mask, dtype=bool)

    bundle_output.parent.mkdir(parents=True, exist_ok=True)
    views_dir.mkdir(parents=True, exist_ok=True)
    failure_manifest.parent.mkdir(parents=True, exist_ok=True)
    parameter_hashes = {
        state: _sha256(parameter_files[state]) for state in STATE_LABELS
    }

    np.savez_compressed(
        bundle_output,
        bundle_schema_version=np.asarray(1, dtype=np.int32),
        table_kind=np.asarray("Cloudy line-physics ablation bundle"),
        axis_order=np.asarray("state,line,log_nH,log_NH,log_T"),
        state_labels=np.asarray(STATE_LABELS),
        line_keys=np.asarray(LINE_KEYS),
        line_labels=np.asarray(LINE_LABELS),
        log_nH=reference_log_nH,
        log_NH=reference_log_NH,
        log_T=log_T,
        log_emissivity_per_nH2=raw_log,
        emissivity_per_nH2=coefficient,
        failure_mask=failure_mask,
        original_failure_mask=original_failure_mask,
        zero_mask=zero_mask,
        interpolated_mask=interpolated_mask,
        no_h2_molecule_command=np.asarray(
            [EXPECTED_NO_H2_COMMAND[state] for state in STATE_LABELS],
            dtype=bool,
        ),
        no_charge_transfer_command=np.asarray(
            [EXPECTED_NO_CT_COMMAND[state] for state in STATE_LABELS],
            dtype=bool,
        ),
        input_directories=np.asarray(
            [str(state_inputs[state].resolve()) for state in STATE_LABELS]
        ),
        parameter_files=np.asarray(
            [str(parameter_files[state].resolve()) for state in STATE_LABELS]
        ),
        parameter_sha256=np.asarray(
            [parameter_hashes[state] for state in STATE_LABELS]
        ),
        uv_background=np.asarray("HM2012 z=0 shielded"),
        cloudy_version=np.asarray("17.02"),
        carbon_abundance_log10=np.asarray(-3.795880),
        column_model=np.asarray("explicit stop column density; no Jeans length"),
        normalization=np.asarray("local deepest-zone emissivity / n_H^2"),
        zero_sentinel_max=np.asarray(ZERO_SENTINEL_MAX),
        failed_node_policy=np.asarray("unavailable; no numerical fill"),
        out_of_bounds_policy=np.asarray("raise"),
    )

    view_files: list[str] = []
    summaries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for state_index, state in enumerate(STATE_LABELS):
        for line_index, line_key in enumerate(LINE_KEYS):
            view_path = view_paths[(state, line_key)]
            local_raw = raw_log[state_index, line_index]
            local_coefficient = coefficient[state_index, line_index]
            local_failure = failure_mask[state_index, line_index]
            local_zero = zero_mask[state_index, line_index]
            _write_view(
                view_path,
                state=state,
                line_index=line_index,
                log_nH=reference_log_nH,
                log_NH=reference_log_NH,
                log_T=log_T,
                raw_log=local_raw,
                coefficient=local_coefficient,
                failure_mask=local_failure,
                zero_mask=local_zero,
                parameter_file=parameter_files[state],
                parameter_sha256=parameter_hashes[state],
            )
            view_files.append(str(view_path))
            summaries.append({
                "state": state,
                "line_key": line_key,
                "line_label": LINE_LABELS[line_index],
                "failures": int(np.count_nonzero(local_failure)),
                "true_zeros": int(np.count_nonzero(local_zero)),
                "positive_nodes": int(np.count_nonzero(local_coefficient > 0.0)),
                "view": str(view_path),
            })
            for i, j, k in np.argwhere(local_failure):
                failures.append({
                    "state": state,
                    "line_key": line_key,
                    "nH_index": int(i),
                    "NH_index": int(j),
                    "T_index": int(k),
                    "log_nH": float(reference_log_nH[i]),
                    "log_NH": float(reference_log_NH[j]),
                    "log_T": float(log_T[k]),
                    "temperature_K": float(10.0 ** log_T[k]),
                })

    failure_manifest.write_text(json.dumps({
        "bundle": str(bundle_output),
        "bundle_shape": list(raw_log.shape),
        "axis_order": ["state", "line", "log_nH", "log_NH", "log_T"],
        "state_labels": list(STATE_LABELS),
        "line_keys": list(LINE_KEYS),
        "line_labels": list(LINE_LABELS),
        "views": view_files,
        "summaries": summaries,
        "short_rows": state_short_rows,
        "total_failed_line_nodes": int(np.count_nonzero(failure_mask)),
        "total_true_zero_line_nodes": int(np.count_nonzero(zero_mask)),
        "failures": failures,
        "interpolation_applied": False,
    }, indent=2) + "\n")

    print(f"Wrote bundle: {bundle_output}")
    print(f"Bundle shape: {raw_log.shape} (state,line,nH,NH,T)")
    print(f"Wrote {len(view_files)} schema-2 views: {views_dir}")
    print(f"Failure manifest: {failure_manifest}")
    for summary in summaries:
        print(
            "  {state:8s} {line_key:6s}: failures={failures:4d}, "
            "true_zeros={true_zeros:4d}, positive={positive_nodes:4d}".format(
                **summary
            )
        )


if __name__ == "__main__":
    main()
