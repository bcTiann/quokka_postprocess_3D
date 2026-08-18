#!/usr/bin/env python3
"""Build raw six-line lookup bundles from completed CIAOLoop grids.

Unavailable Cloudy crash rows remain NaN and are recorded in failure masks.
The value -99 from CIAOLoop is retained in the logarithmic array and converted
to an exact physical zero in the linear emissivity array.  No failure is
interpolated by this builder.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import re
from pathlib import Path

import numpy as np


LINES = (
    ("cii", "C  2 157.636m", "C_2_157.636m"),
    ("halpha", "H  1 6562.81A", "H_1_6562.81A"),
    ("hi21", "H  1 21.1207c", "H_1_21.1207c"),
    ("ciii_977", "C  3 977.020A", "C_3_977.020A"),
    ("ciii_1907", "C  3 1906.68A", "C_3_1906.68A"),
    ("ciii_1909", "C  3 1908.73A", "C_3_1908.73A"),
)
N_NH = 10
N_COLUMN = 10
N_T = 21
T_MIN_K = 3.6
T_MAX_K = 1.0e9
ZERO_LIMIT = -90.0
T_TOLERANCE_DEX = 5.1e-4
JEANS_CAP_CM = 3.086e20
RUN_RE = re.compile(r"_run([1-9][0-9]*)\.dat$")
HDEN_RE = re.compile(r"^#\s*hden\s+(.+?)\s*$")
COLUMN_RE = re.compile(r"^#\s*stop column density\s+(.+?)\s*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse(path: Path, needs_column: bool) -> tuple[
    float, float | None, dict[float, np.ndarray | None]
]:
    log_nh = None
    log_column = None
    header = None
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
                f"bad row at {path}:{line_number}: found {len(columns)} columns"
            )
        log_t = float(columns[0])
        if log_t in values:
            raise ValueError(f"duplicate temperature {log_t}: {path}")
        values[log_t] = (
            np.asarray(columns[1:], dtype=float) if len(columns) > 1 else None
        )
    if log_nh is None or (needs_column and log_column is None):
        raise ValueError(f"missing loop metadata: {path}")
    if header != tuple(item[2] for item in LINES):
        raise ValueError(f"unexpected line header {header!r}: {path}")
    if len(values) != N_T:
        raise ValueError(f"expected {N_T} temperatures, found {len(values)}: {path}")
    return log_nh, log_column, values


def _files(directory: Path, expected: int) -> dict[int, Path]:
    if list(directory.glob("*.mach")):
        raise RuntimeError(f"CIAOLoop jobs are still active: {directory}")
    result = {}
    for path in directory.glob("*_run*.dat"):
        match = RUN_RE.search(path.name)
        if match:
            result[int(match.group(1))] = path
    if set(result) != set(range(1, expected + 1)):
        raise ValueError(f"run files are not exactly 1..{expected}: {directory}")
    return result


def _temperature_index(value: float, axis: np.ndarray, path: Path) -> int:
    index = int(np.abs(axis - value).argmin())
    if abs(float(axis[index]) - value) > T_TOLERANCE_DEX:
        raise ValueError(f"off-grid temperature {value}: {path}")
    return index


def _load_column(directory: Path, log_t: np.ndarray):
    files = _files(directory, N_NH * N_COLUMN)
    records = [(run, *_parse(path, True)) for run, path in sorted(files.items())]
    log_nh = np.unique([record[1] for record in records])
    log_column = np.unique([record[2] for record in records])
    if (log_nh.size, log_column.size) != (N_NH, N_COLUMN):
        raise ValueError("column axes are not 10x10")
    raw = np.full((len(LINES), N_NH, N_COLUMN, N_T), np.nan)
    nh_index = {float(value): i for i, value in enumerate(log_nh)}
    column_index = {float(value): i for i, value in enumerate(log_column)}
    for run, run_nh, run_column, values in records:
        i = nh_index[float(run_nh)]
        j = column_index[float(run_column)]
        if run != i * N_COLUMN + j + 1:
            raise ValueError(f"run ordering mismatch: {run}")
        for reported_t, line_values in values.items():
            k = _temperature_index(reported_t, log_t, files[run])
            if line_values is not None and np.isfinite(line_values).all():
                raw[:, i, j, k] = line_values
    return log_nh, log_column, raw


def _load_jeans(directory: Path, log_t: np.ndarray):
    files = _files(directory, N_NH)
    records = [(run, *_parse(path, False)) for run, path in sorted(files.items())]
    log_nh = np.asarray([record[1] for record in records])
    if np.any(np.diff(log_nh) <= 0.0):
        raise ValueError("Jeans density axis is not increasing")
    raw = np.full((len(LINES), N_NH, N_T), np.nan)
    for i, (run, _, _, values) in enumerate(records):
        for reported_t, line_values in values.items():
            k = _temperature_index(reported_t, log_t, files[run])
            if line_values is not None and np.isfinite(line_values).all():
                raw[:, i, k] = line_values
    return log_nh, raw


def _payload(raw: np.ndarray) -> dict[str, np.ndarray]:
    failure = ~np.isfinite(raw)
    zero = (~failure) & (raw <= ZERO_LIMIT)
    coefficient = np.zeros_like(raw)
    positive = (~failure) & (~zero)
    coefficient[positive] = np.power(10.0, raw[positive])
    return {
        "log_emissivity_per_nH2": raw,
        "emissivity_per_nH2": coefficient,
        "failure_mask": failure,
        "original_failure_mask": failure.copy(),
        "zero_mask": zero,
        "interpolated_mask": np.zeros_like(failure),
    }


def _common_metadata(
    parameter_template: Path,
    *,
    charge_transfer_enabled: bool,
    cosmic_ray_rate_s: float,
    cmb_redshift: float | None,
    molecular_network_enabled: bool,
) -> dict[str, np.ndarray]:
    radiation_field = (
        "Cloudy table HM12 redshift 0 + table ISM filtered by "
        "extinguish column=21 leak=0"
    )
    if cmb_redshift is not None:
        radiation_field += f" + CMB redshift {cmb_redshift:g}"
    return {
        "schema_version": np.asarray(1, dtype=np.int32),
        "line_keys": np.asarray([item[0] for item in LINES]),
        "line_labels": np.asarray([item[1] for item in LINES]),
        "cloudy_version": np.asarray("17.02"),
        "radiation_field": np.asarray(radiation_field),
        "cmb_included": np.asarray(cmb_redshift is not None),
        "cmb_redshift": np.asarray(
            np.nan if cmb_redshift is None else cmb_redshift, dtype=float
        ),
        "external_grackle_hm12_used": np.asarray(False),
        "composition_label": np.asarray("Cloudy 17.02 default abundances"),
        "no_h2_molecule_command": np.asarray(not molecular_network_enabled),
        "molecular_network_enabled": np.asarray(molecular_network_enabled),
        "molecular_treatment": np.asarray(
            "Cloudy default; detailed H2 not explicitly requested"
            if molecular_network_enabled
            else "disabled by 'no H2 molecule'"
        ),
        "no_charge_transfer_command": np.asarray(not charge_transfer_enabled),
        "charge_transfer_enabled": np.asarray(charge_transfer_enabled),
        "cosmic_ray_h0_ionization_rate_s": np.asarray(cosmic_ray_rate_s),
        "grains_added": np.asarray(False),
        "normalization": np.asarray("local deepest-zone emissivity / n_H^2"),
        "failed_node_policy": np.asarray("unavailable; no numerical fill"),
        "parameter_template": np.asarray(parameter_template.name),
        "parameter_template_sha256": np.asarray(_sha256(parameter_template)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stem",
        default=(
            "hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_"
            "defaultabund_sixline"
        ),
    )
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--runtime-grackle-dir",
        type=Path,
        default=root / "runtime/cloudy_sixline/examples/grackle",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "data")
    parser.add_argument("--charge-transfer-enabled", action="store_true")
    parser.add_argument("--cosmic-ray-rate-s", type=float, default=0.0)
    parser.add_argument("--cmb-redshift", type=float, default=None)
    parser.add_argument("--molecular-network-enabled", action="store_true")
    args = parser.parse_args()
    examples = args.runtime_grackle_dir.resolve()
    templates = root / "vendor/cloudy_cooling_tools/examples/grackle"
    data_dir = args.output_dir.resolve()
    stem = args.stem
    log_t = np.linspace(np.log10(T_MIN_K), np.log10(T_MAX_K), N_T)

    column_input = examples / f"{stem}_column_10x10x21_output"
    jeans_input = examples / f"{stem}_jeans_10x21_output"
    column_template = templates / f"{stem}_column_10x10x21.par.in"
    jeans_template = templates / f"{stem}_jeans_10x21.par.in"
    for template in (column_template, jeans_template):
        if not template.is_file():
            raise FileNotFoundError(template)
    log_nh, log_column, column_raw = _load_column(column_input, log_t)
    jeans_log_nh, jeans_raw = _load_jeans(jeans_input, log_t)
    # CIAOLoop's interval syntax and explicit-value syntax can print the same
    # grid coordinate with a few last-bit decimal differences.  Require
    # agreement far below the precision relevant to table interpolation, then
    # store one canonical density axis in both products.
    if not np.allclose(log_nh, jeans_log_nh, rtol=0.0, atol=1.0e-12):
        raise ValueError("column and Jeans density axes differ")
    jeans_log_nh = log_nh.copy()

    data_dir.mkdir(exist_ok=True)
    column_output = data_dir / f"cloudy_{stem}_column_10x10x21.npz"
    jeans_output = data_dir / f"cloudy_{stem}_jeans_10x21.npz"
    np.savez_compressed(
        column_output,
        axis_order=np.asarray("line,log_nH,log_NH,log_T"),
        geometry=np.asarray("explicit stop column density"),
        log_nH=log_nh,
        log_NH=log_column,
        log_T=log_t,
        **_payload(column_raw),
        **_common_metadata(
            column_template,
            charge_transfer_enabled=args.charge_transfer_enabled,
            cosmic_ray_rate_s=args.cosmic_ray_rate_s,
            cmb_redshift=args.cmb_redshift,
            molecular_network_enabled=args.molecular_network_enabled,
        ),
    )
    np.savez_compressed(
        jeans_output,
        axis_order=np.asarray("line,log_nH,log_T"),
        geometry=np.asarray("Jeans length with 100 pc maximum"),
        jeans_length_cap_cm=np.asarray(JEANS_CAP_CM),
        log_nH=jeans_log_nh,
        log_T=log_t,
        **_payload(jeans_raw),
        **_common_metadata(
            jeans_template,
            charge_transfer_enabled=args.charge_transfer_enabled,
            cosmic_ray_rate_s=args.cosmic_ray_rate_s,
            cmb_redshift=args.cmb_redshift,
            molecular_network_enabled=args.molecular_network_enabled,
        ),
    )

    report = {"products": []}
    for geometry, output, raw in (
        ("column", column_output, column_raw),
        ("jeans", jeans_output, jeans_raw),
    ):
        masks = ~np.isfinite(raw)
        union = np.any(masks, axis=0)
        report["products"].append({
            "geometry": geometry,
            "path": str(output.resolve()),
            "shape": list(raw.shape),
            "union_failure_nodes": int(np.count_nonzero(union)),
            "line_failure_masks_identical": all(
                np.array_equal(masks[0], masks[index])
                for index in range(1, len(LINES))
            ),
            "failure_nodes": [
                {
                    "indices": [int(value) for value in index],
                    "log_nH": float(log_nh[index[0]]),
                    **(
                        {"log_NH": float(log_column[index[1]]),
                         "log_T": float(log_t[index[2]]),
                         "temperature_K": float(10.0 ** log_t[index[2]])}
                        if geometry == "column"
                        else {"log_T": float(log_t[index[1]]),
                              "temperature_K": float(10.0 ** log_t[index[1]])}
                    ),
                }
                for index in np.argwhere(union)
            ],
        })
    report_path = data_dir / f"cloudy_{stem}_failure_nodes.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
