"""Build one explicit-column Cloudy CII table from CIAOLoop_lines output."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


LOOP_RE = re.compile(r"^#\s*(hden|stop column density)\s+(.+?)\s*$")
RUN_RE = re.compile(r"_run([1-9][0-9]*)\.dat$")
ZERO_LIMIT = -90.0
TOLERANCE_DEX = 5.1e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_run(path: Path) -> tuple[float, float, dict[float, float | None]]:
    log_nh = log_ncol = None
    header = None
    values: dict[float, float | None] = {}
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        match = LOOP_RE.match(raw)
        if match:
            if match.group(1) == "hden":
                log_nh = float(match.group(2))
            else:
                log_ncol = float(match.group(2))
            continue
        if raw.startswith("#Te"):
            header = tuple(raw.split()[1:])
            continue
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        columns = raw.split()
        if len(columns) not in (1, 2):
            raise ValueError(f"bad row at {path}:{number}: {raw!r}")
        log_t = float(columns[0])
        if log_t in values:
            raise ValueError(f"duplicate log(T)={log_t}: {path}")
        values[log_t] = float(columns[1]) if len(columns) == 2 else None
    if log_nh is None or log_ncol is None:
        raise ValueError(f"missing loop metadata: {path}")
    if header != ("C_2_157.636m",):
        raise ValueError(f"unexpected line header {header!r}: {path}")
    return log_nh, log_ncol, values


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--parameter-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-manifest", type=Path, required=True)
    parser.add_argument("--state-label", default="draine_gow_cr")
    parser.add_argument(
        "--composition-label", default="GOW H/He/C/O/Si gas-phase abundances",
    )
    parser.add_argument(
        "--radiation-field", default="HM2012 z=0 plus Draine",
    )
    parser.add_argument(
        "--cosmic-ray-rate", type=float, default=2.0e-17,
        help="explicit Cloudy H0 cosmic-ray ionization rate; use 0 when absent",
    )
    parser.add_argument("--t-min", type=float, default=3.6)
    parser.add_argument("--t-max", type=float, default=1.0e9)
    parser.add_argument("--t-points", type=int, default=21)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in ("input_dir", "parameter_file", "output", "failure_manifest"):
        setattr(args, name, getattr(args, name).resolve())
    if (args.output.exists() or args.failure_manifest.exists()) and not args.force:
        raise FileExistsError("output exists; pass --force to replace it")
    if list(args.input_dir.glob("*.mach")):
        raise RuntimeError(f"Cloudy jobs are still active: {args.input_dir}")

    files = sorted(args.input_dir.glob("*_run*.dat"))
    if len(files) != 100:
        raise ValueError(f"expected 100 run files, found {len(files)}")
    parsed = []
    ids = set()
    for path in files:
        match = RUN_RE.search(path.name)
        if match is None:
            raise ValueError(f"unexpected run filename: {path}")
        run_id = int(match.group(1))
        if run_id in ids:
            raise ValueError(f"duplicate run id {run_id}")
        ids.add(run_id)
        parsed.append((run_id, path, *_read_run(path)))
    if ids != set(range(1, 101)):
        raise ValueError("run ids are not exactly 1..100")

    log_nh = np.unique([record[2] for record in parsed])
    log_ncol = np.unique([record[3] for record in parsed])
    log_t = np.linspace(np.log10(args.t_min), np.log10(args.t_max), args.t_points)
    if log_nh.size != 10 or log_ncol.size != 10:
        raise ValueError(f"expected 10x10 axes, found {log_nh.size}x{log_ncol.size}")
    raw = np.full((1, 1, 10, 10, args.t_points), np.nan)
    short_rows: list[dict[str, float | int]] = []
    for run_id, path, nh, ncol, values in parsed:
        i = int(np.flatnonzero(log_nh == nh)[0])
        j = int(np.flatnonzero(log_ncol == ncol)[0])
        if run_id != i * 10 + j + 1:
            raise ValueError(f"run ordering mismatch: {path}")
        if len(values) != args.t_points:
            raise ValueError(
                f"expected {args.t_points} temperature rows in {path}, "
                f"found {len(values)}"
            )
        for reported_t, value in values.items():
            k = int(np.abs(log_t - reported_t).argmin())
            if abs(float(log_t[k]) - reported_t) > TOLERANCE_DEX:
                raise ValueError(f"off-grid log(T)={reported_t}: {path}")
            if value is None:
                short_rows.append({
                    "run": run_id, "nH_index": i, "NH_index": j,
                    "T_index": k, "log_nH": float(nh),
                    "log_NH": float(ncol), "log_T": float(log_t[k]),
                })
            else:
                raw[0, 0, i, j, k] = value

    failure = ~np.isfinite(raw)
    zero = (~failure) & (raw <= ZERO_LIMIT)
    coefficient = np.zeros_like(raw)
    positive = (~failure) & (~zero)
    coefficient[positive] = np.power(10.0, raw[positive])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        bundle_schema_version=np.asarray(1, dtype=np.int32),
        table_kind=np.asarray("Cloudy explicit-column CII single-state bundle"),
        axis_order=np.asarray("state,line,log_nH,log_NH,log_T"),
        state_labels=np.asarray((args.state_label,)),
        line_keys=np.asarray(("cii",)),
        line_labels=np.asarray(("C  2 157.636m",)),
        log_nH=log_nh, log_NH=log_ncol, log_T=log_t,
        log_emissivity_per_nH2=raw,
        emissivity_per_nH2=coefficient,
        failure_mask=failure,
        original_failure_mask=failure.copy(),
        zero_mask=zero,
        interpolated_mask=np.zeros_like(failure),
        out_of_bounds_policy=np.asarray("raise"),
        normalization=np.asarray("local deepest-zone emissivity / n_H^2"),
        column_model=np.asarray("explicit stop column density; no Jeans length"),
        uv_background=np.asarray(args.radiation_field),
        radiation_field=np.asarray(args.radiation_field),
        cloudy_version=np.asarray("17.02"),
        composition_label=np.asarray(args.composition_label),
        cosmic_ray_h0_ionization_rate_s=np.asarray(args.cosmic_ray_rate),
        cosmic_ray_command_present=np.asarray(args.cosmic_ray_rate > 0.0),
        no_h2_molecule_command=np.asarray(True),
        no_charge_transfer_command=np.asarray(True),
        failed_node_policy=np.asarray("unavailable; no numerical fill"),
        parameter_file=np.asarray(str(args.parameter_file)),
        parameter_sha256=np.asarray(_sha256(args.parameter_file)),
        input_directory=np.asarray(str(args.input_dir)),
    )
    manifest = {
        "output": str(args.output),
        "shape": list(raw.shape),
        "state_label": args.state_label,
        "failure_nodes": int(np.count_nonzero(failure)),
        "true_zero_nodes": int(np.count_nonzero(zero)),
        "positive_nodes": int(np.count_nonzero(positive)),
        "short_rows": short_rows,
        "parameter_file": str(args.parameter_file),
        "parameter_sha256": _sha256(args.parameter_file),
    }
    args.failure_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.failure_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
