#!/usr/bin/env python3
"""Package the completed 7 x 10 x 21 CIAOLoop six-line Jeans grid.

Cloudy crash rows remain unavailable NaNs and are recorded in failure masks.
CIAOLoop's -99 true-zero sentinel becomes an exact zero only in the linear
coefficient array. This builder never fills or smooths a failed node.
"""

from __future__ import annotations

import argparse
import hashlib
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
N_DENSITY = 10
N_T = 21
T_MIN_K = 3.6
T_MAX_K = 1.0e9
ZERO_LIMIT = -90.0
T_TOLERANCE_DEX = 5.1e-4
JEANS_CAP_CM = 3.086e20
RUN_RE = re.compile(r"_run([1-9][0-9]*)\.dat$")
HDEN_RE = re.compile(r"^#\s*hden\s+(.+?)\s*$")
INIT_RE = re.compile(
    r'^#\s*init\s+"[^"]*logNH([0-9]+(?:\.[0-9]+)?)\.out"\s*$'
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse(path: Path) -> tuple[float, float, dict[float, np.ndarray | None]]:
    log_nh = None
    log_nh_attenuation = None
    header = None
    values: dict[float, np.ndarray | None] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        match = HDEN_RE.match(raw)
        if match:
            log_nh = float(match.group(1))
            continue
        match = INIT_RE.match(raw)
        if match:
            log_nh_attenuation = float(match.group(1))
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
    if log_nh is None or log_nh_attenuation is None:
        raise ValueError(f"missing hden or attenuation-init metadata: {path}")
    if header != tuple(item[2] for item in LINES):
        raise ValueError(f"unexpected line header {header!r}: {path}")
    if len(values) != N_T:
        raise ValueError(f"expected {N_T} temperatures, found {len(values)}: {path}")
    return log_nh_attenuation, log_nh, values


def _files(directory: Path, expected: int) -> dict[int, Path]:
    if list(directory.glob("*.mach")):
        raise RuntimeError(f"CIAOLoop jobs are still active: {directory}")
    result: dict[int, Path] = {}
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


def _load_grid(
    directory: Path,
    log_t: np.ndarray,
    requested_attenuation_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = requested_attenuation_axis.size * N_DENSITY
    files = _files(directory, expected)
    records = [
        (run, path, *_parse(path)) for run, path in sorted(files.items())
    ]
    attenuation_axis = np.unique([record[2] for record in records])
    density_axis = np.unique([record[3] for record in records])
    if not np.array_equal(attenuation_axis, requested_attenuation_axis):
        raise ValueError(
            f"attenuation axis {attenuation_axis} differs from requested "
            f"{requested_attenuation_axis}"
        )
    if density_axis.size != N_DENSITY:
        raise ValueError(f"expected {N_DENSITY} hden values, found {density_axis}")

    raw = np.full(
        (len(LINES), attenuation_axis.size, density_axis.size, N_T), np.nan
    )
    seen = np.zeros((attenuation_axis.size, density_axis.size), dtype=bool)
    attenuation_index = {float(value): index for index, value in enumerate(attenuation_axis)}
    density_index = {float(value): index for index, value in enumerate(density_axis)}
    for _, path, attenuation, density, values in records:
        i = attenuation_index[float(attenuation)]
        j = density_index[float(density)]
        if seen[i, j]:
            raise ValueError(
                f"duplicate attenuation/density pair ({attenuation}, {density})"
            )
        seen[i, j] = True
        for reported_t, line_values in values.items():
            k = _temperature_index(reported_t, log_t, path)
            if line_values is not None and np.isfinite(line_values).all():
                raw[:, i, j, k] = line_values
    if not seen.all():
        raise ValueError("attenuation/density grid is incomplete")
    return attenuation_axis, density_axis, raw


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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stem",
        default="hm2012_attgrid_ism_nh21_cmb_cr_defaultabund_sixline_jeans",
    )
    parser.add_argument(
        "--runtime-grackle-dir",
        type=Path,
        default=root / "runtime/cloudy_sixline/examples/grackle",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "data")
    parser.add_argument("--parameter-file", type=Path, required=True)
    parser.add_argument(
        "--hm12-log-nh",
        type=float,
        nargs="+",
        default=(18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0),
    )
    args = parser.parse_args()

    examples = args.runtime_grackle_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    parameter_file = args.parameter_file.expanduser().resolve()
    if not parameter_file.is_file():
        raise FileNotFoundError(parameter_file)
    requested_attenuation = np.asarray(args.hm12_log_nh, dtype=float)
    if requested_attenuation.ndim != 1 or np.any(np.diff(requested_attenuation) <= 0):
        raise ValueError("HM2012 attenuation axis must be strictly increasing")
    log_t = np.linspace(np.log10(T_MIN_K), np.log10(T_MAX_K), N_T)
    input_directory = examples / f"{args.stem}_7x10x21_output"
    attenuation, density, raw = _load_grid(
        input_directory, log_t, requested_attenuation
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"cloudy_{args.stem}_7x10x21.npz"
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(3, dtype=np.int32),
        axis_order=np.asarray("line,log_NH_attenuation,log_nH,log_T"),
        line_keys=np.asarray([item[0] for item in LINES]),
        line_labels=np.asarray([item[1] for item in LINES]),
        cloudy_version=np.asarray("17.02"),
        log_NH_attenuation=attenuation,
        log_nH=density,
        log_T=log_t,
        geometry=np.asarray("Jeans length with 100 pc maximum"),
        jeans_length_cap_cm=np.asarray(JEANS_CAP_CM),
        radiation_field=np.asarray(
            "HM2012 separately quick-extinguished over log NH=18..21; "
            "table ISM separately quick-extinguished at log NH=21; CMB z=0"
        ),
        hm12_attenuation_method=np.asarray("extinguish column=<grid> leak=0"),
        ism_log_NH_attenuation=np.asarray(21.0),
        cmb_included=np.asarray(True),
        cmb_redshift=np.asarray(0.0),
        external_grackle_hm12_used=np.asarray(False),
        cosmic_ray_h0_ionization_rate_s=np.asarray(2.0e-17),
        composition_label=np.asarray("Cloudy 17.02 default abundances"),
        molecular_treatment=np.asarray(
            "Cloudy default simple molecular network; detailed H2 not requested"
        ),
        charge_transfer_enabled=np.asarray(True),
        grains_added=np.asarray(False),
        turbulence_added=np.asarray(False),
        normalization=np.asarray("local deepest-zone emissivity / n_H^2"),
        simulation_NH_policy=np.asarray(
            "clip log10 NH to [18,21], then interpolate; no extrapolation"
        ),
        density_temperature_out_of_bounds_policy=np.asarray("raise"),
        failed_node_policy=np.asarray("unavailable; no numerical fill"),
        parameter_file=np.asarray(parameter_file.name),
        parameter_file_sha256=np.asarray(_sha256(parameter_file)),
        **_payload(raw),
    )

    masks = ~np.isfinite(raw)
    union = np.any(masks, axis=0)
    report = {
        "product": str(output_path),
        "shape": list(raw.shape),
        "axis_order": "line,log_NH_attenuation,log_nH,log_T",
        "union_failure_nodes": int(np.count_nonzero(union)),
        "line_failure_masks_identical": all(
            np.array_equal(masks[0], masks[index])
            for index in range(1, len(LINES))
        ),
        "failure_nodes": [
            {
                "indices": [int(value) for value in index],
                "log_NH_attenuation": float(attenuation[index[0]]),
                "log_nH": float(density[index[1]]),
                "log_T": float(log_t[index[2]]),
                "temperature_K": float(10.0 ** log_t[index[2]]),
            }
            for index in np.argwhere(union)
        ],
    }
    report_path = output_dir / f"cloudy_{args.stem}_failure_nodes.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
