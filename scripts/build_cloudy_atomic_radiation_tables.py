"""Build three-state CII, Halpha, and HI 21-cm Cloudy lookup tables.

Each CIAOLoop_lines grid contains all three lines from one shared Cloudy
solution.  Products retain unavailable crash nodes as failures and Cloudy's
explicit -99 values as physical zero-emissivity nodes.  No interpolation is
performed here.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np


STATES = ("hm2012", "hm2012_draine", "hm2012_draine_cr")
STATE_STEMS = {
    "hm2012": "hm2012_defaultabund_multiline",
    "hm2012_draine": "hm2012_plus_draine_defaultabund_multiline",
    "hm2012_draine_cr": "hm2012_plus_draine_cr_defaultabund_multiline",
}
RADIATION = {
    "hm2012": "HM2012 z=0",
    "hm2012_draine": "HM2012 z=0 + table Draine",
    "hm2012_draine_cr": (
        "HM2012 z=0 + table Draine + cosmic rays rate 2e-17 s^-1"
    ),
}
CR_RATE = {"hm2012": 0.0, "hm2012_draine": 0.0,
           "hm2012_draine_cr": 2.0e-17}
NO_CHARGE_TRANSFER = {state: True for state in STATES}
LINES = (
    ("cii", "C  2 157.636m", "C_2_157.636m"),
    ("halpha", "H  1 6562.81A", "H_1_6562.81A"),
    ("hi21", "H  1 21.1207c", "H_1_21.1207c"),
)

N_NH = 10
N_COLUMN = 10
N_T = 21
T_MIN_K = 3.6
T_MAX_K = 1.0e9
ZERO_LIMIT = -90.0
T_TOLERANCE_DEX = 5.1e-4
RUN_RE = re.compile(r"_run([1-9][0-9]*)\.dat$")
HDEN_RE = re.compile(r"^#\s*hden\s+(.+?)\s*$")
COLUMN_RE = re.compile(r"^#\s*stop column density\s+(.+?)\s*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_run(path: Path, *, needs_column: bool) -> tuple[
    float, float | None, dict[float, np.ndarray | None]
]:
    log_nh: float | None = None
    log_column: float | None = None
    header: tuple[str, ...] | None = None
    values: dict[float, np.ndarray | None] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        match = HDEN_RE.match(raw)
        if match:
            log_nh = float(match.group(1))
            continue
        match = COLUMN_RE.match(raw)
        if match:
            log_column = float(match.group(1))
            continue
        if raw.startswith("#Te"):
            header = tuple(raw.split()[1:])
            continue
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        columns = raw.split()
        if len(columns) not in (1, 1 + len(LINES)):
            raise ValueError(
                f"bad data row at {path}:{line_number}: expected 1 or "
                f"{1 + len(LINES)} columns, found {len(columns)}"
            )
        log_t = float(columns[0])
        if log_t in values:
            raise ValueError(f"duplicate log(T)={log_t} in {path}")
        values[log_t] = (
            np.asarray([float(item) for item in columns[1:]], dtype=float)
            if len(columns) > 1 else None
        )
    if log_nh is None:
        raise ValueError(f"missing hden metadata in {path}")
    if needs_column and log_column is None:
        raise ValueError(f"missing stop-column metadata in {path}")
    expected_header = tuple(item[2] for item in LINES)
    if header != expected_header:
        raise ValueError(f"unexpected line header {header!r} in {path}")
    if len(values) != N_T:
        raise ValueError(
            f"incomplete temperature sweep in {path}: "
            f"expected {N_T}, found {len(values)}"
        )
    return log_nh, log_column, values


def _temperature_index(log_t: float, axis: np.ndarray, path: Path) -> int:
    index = int(np.abs(axis - log_t).argmin())
    residual = abs(float(axis[index]) - log_t)
    if residual > T_TOLERANCE_DEX:
        raise ValueError(
            f"off-grid log(T)={log_t} by {residual:g} dex in {path}"
        )
    return index


def _run_files(directory: Path, expected: int) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    active = sorted(directory.glob("*.mach"))
    if active:
        raise RuntimeError(f"stale/active .mach files remain in {directory}")
    files: dict[int, Path] = {}
    for path in directory.glob("*_run*.dat"):
        match = RUN_RE.search(path.name)
        if match is None:
            raise ValueError(f"unexpected run filename: {path}")
        files[int(match.group(1))] = path
    expected_ids = set(range(1, expected + 1))
    if set(files) != expected_ids:
        raise ValueError(
            f"run ids in {directory} are not 1..{expected}: "
            f"missing={sorted(expected_ids - set(files))}, "
            f"extra={sorted(set(files) - expected_ids)}"
        )
    manifests = list(directory.glob("*.run"))
    if len(manifests) != 1:
        raise ValueError(f"expected one .run manifest in {directory}")
    manifest_rows = [
        line for line in manifests[0].read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(manifest_rows) != expected:
        raise ValueError(
            f"manifest in {directory} has {len(manifest_rows)} rows, "
            f"expected {expected}"
        )
    return files


def _load_column(directory: Path, log_t: np.ndarray) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    files = _run_files(directory, N_NH * N_COLUMN)
    records = [(run_id, *_parse_run(path, needs_column=True))
               for run_id, path in sorted(files.items())]
    log_nh = np.unique([record[1] for record in records])
    log_column = np.unique([record[2] for record in records])
    if log_nh.size != N_NH or log_column.size != N_COLUMN:
        raise ValueError(f"axes in {directory} are not 10x10")
    raw = np.full((len(LINES), N_NH, N_COLUMN, N_T), np.nan)
    nh_index = {float(value): index for index, value in enumerate(log_nh)}
    column_index = {
        float(value): index for index, value in enumerate(log_column)
    }
    for run_id, run_nh, run_column, values in records:
        i = nh_index[float(run_nh)]
        j = column_index[float(run_column)]
        if run_id != i * N_COLUMN + j + 1:
            raise ValueError(f"run ordering mismatch for run {run_id}")
        for reported_t, line_values in values.items():
            k = _temperature_index(reported_t, log_t, files[run_id])
            if line_values is not None and np.isfinite(line_values).all():
                raw[:, i, j, k] = line_values
    return log_nh, log_column, raw


def _load_jeans(directory: Path, log_t: np.ndarray) -> tuple[
    np.ndarray, np.ndarray
]:
    files = _run_files(directory, N_NH)
    records = [(run_id, *_parse_run(path, needs_column=False))
               for run_id, path in sorted(files.items())]
    log_nh = np.asarray([record[1] for record in records], dtype=float)
    if np.any(np.diff(log_nh) <= 0.0):
        raise ValueError(f"density order is not increasing in {directory}")
    raw = np.full((len(LINES), N_NH, N_T), np.nan)
    for density_index, (run_id, _, _, values) in enumerate(records):
        for reported_t, line_values in values.items():
            k = _temperature_index(reported_t, log_t, files[run_id])
            if line_values is not None and np.isfinite(line_values).all():
                raw[:, density_index, k] = line_values
    return log_nh, raw


def _linear_products(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    failure = ~np.isfinite(raw)
    zero = (~failure) & (raw <= ZERO_LIMIT)
    coefficient = np.zeros(raw.shape, dtype=float)
    positive = (~failure) & (~zero)
    coefficient[positive] = np.power(10.0, raw[positive])
    return coefficient, failure, zero


def _metadata(state: str, geometry: str, parameter_file: Path) -> dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray(2, dtype=np.int32),
        "state_label": np.asarray(state),
        "radiation_field": np.asarray(RADIATION[state]),
        "uv_background": np.asarray(RADIATION[state]),
        "cosmic_ray_h0_ionization_rate_s": np.asarray(CR_RATE[state]),
        "cloudy_version": np.asarray("17.02"),
        "composition_label": np.asarray(
            "Cloudy 17.02 default abundances; no element abundance overrides"
        ),
        "carbon_abundance_log10": np.asarray(np.nan),
        "no_h2_molecule_command": np.asarray(True),
        "no_charge_transfer_command": np.asarray(
            NO_CHARGE_TRANSFER.get(state, True)
        ),
        "geometry": np.asarray(geometry),
        "normalization": np.asarray("local deepest-zone emissivity / n_H^2"),
        "out_of_bounds_policy": np.asarray("raise"),
        "failed_node_policy": np.asarray(
            "unavailable; runtime rejects unless explicitly filled after "
            "simulation sampling"
        ),
        "parameter_file": np.asarray(str(parameter_file.resolve())),
        "parameter_sha256": np.asarray(_sha256(parameter_file)),
    }


def _write_views(
    output_dir: Path,
    *,
    state: str,
    geometry: str,
    log_nh: np.ndarray,
    log_t: np.ndarray,
    raw: np.ndarray,
    parameter_file: Path,
    log_column: np.ndarray | None = None,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for line_index, (line_key, line_label, _) in enumerate(LINES):
        line_raw = raw[line_index]
        coefficient, failure, zero = _linear_products(line_raw)
        filename = (
            f"cloudy_{line_key}_{state}_{geometry}_"
            f"{'10x10x21' if geometry == 'column' else '10x21'}.npz"
        )
        path = output_dir / filename
        payload = {
            "log_nH": log_nh,
            "log_T": log_t,
            "log_emissivity_per_nH2": line_raw,
            "emissivity_per_nH2": coefficient,
            "failure_mask": failure,
            "original_failure_mask": failure.copy(),
            "zero_mask": zero,
            "interpolated_mask": np.zeros_like(failure, dtype=bool),
            "line_key": np.asarray(line_key),
            "line_label": np.asarray(line_label),
            **_metadata(state, geometry, parameter_file),
        }
        if log_column is not None:
            payload["log_NH"] = log_column
            payload["axis_order"] = np.asarray("log_nH,log_NH,log_T")
            payload["column_model"] = np.asarray("explicit stop column density")
        else:
            payload["axis_order"] = np.asarray("log_nH,log_T")
            payload["column_model"] = np.asarray(
                "Jeans length with 100 pc maximum"
            )
            payload["jeans_length_cap_cm"] = np.asarray(3.086e20)
        np.savez_compressed(path, **payload)
        records.append({
            "path": str(path.resolve()),
            "state": state,
            "geometry": geometry,
            "line": line_key,
            "shape": list(line_raw.shape),
            "failure_nodes": int(np.count_nonzero(failure)),
            "zero_nodes": int(np.count_nonzero(zero)),
            "positive_nodes": int(np.count_nonzero((~failure) & (~zero))),
        })
    return records


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    examples = root / "work/cloudy_cooling_tools_history/examples/grackle"
    output_dir = root / "data/cloudy_atomic_defaultabund_radiation_3state_views"
    report_path = root / "data/cloudy_atomic_defaultabund_radiation_3state_failures.json"
    log_t = np.linspace(np.log10(T_MIN_K), np.log10(T_MAX_K), N_T)
    report_records: list[dict[str, object]] = []

    for state in STATES:
        stem = STATE_STEMS[state]
        column_dir = examples / f"{stem}_column_10x10x21_output"
        column_par = examples / f"{stem}_column_10x10x21.par"
        jeans_dir = examples / f"{stem}_jeans_10x21_output"
        jeans_par = examples / f"{stem}_jeans_10x21.par"
        log_nh, log_column, column_raw = _load_column(column_dir, log_t)
        report_records.extend(_write_views(
            output_dir, state=state, geometry="column", log_nh=log_nh,
            log_column=log_column, log_t=log_t, raw=column_raw,
            parameter_file=column_par,
        ))
        jeans_log_nh, jeans_raw = _load_jeans(jeans_dir, log_t)
        if not np.array_equal(log_nh, jeans_log_nh):
            raise ValueError(f"column and Jeans density axes differ for {state}")
        report_records.extend(_write_views(
            output_dir, state=state, geometry="jeans", log_nh=jeans_log_nh,
            log_t=log_t, raw=jeans_raw, parameter_file=jeans_par,
        ))

    report = {
        "states": list(STATES),
        "lines": [item[0] for item in LINES],
        "temperature_bounds_K": [T_MIN_K, T_MAX_K],
        "products": report_records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
