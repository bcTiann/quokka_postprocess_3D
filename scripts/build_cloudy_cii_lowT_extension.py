"""Merge the 3.6--10 K CII runs with the existing 10--1e9 K table.

Only the baseline and molecular+charge-transfer states are included.  The
10 K endpoint is independently recomputed and compared with the old table;
it is used for QA, while the old table remains authoritative at 10 K.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


STATES = ("baseline", "mol_ct")
OLD_STATE_INDICES = (0, 3)
EXPECTED_T = 4
EXPECTED_AXIS = 10
ZERO_LIMIT = -90.0
RUN_RE = re.compile(r"_run([1-9][0-9]*)\.dat$")
LOOP_RE = re.compile(r"^#\s*(hden|stop column density)\s+(.+?)\s*$")
TOLERANCE_DEX = 5.1e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_run(path: Path) -> tuple[float, float, dict[float, float | None]]:
    log_nh = None
    log_ncol = None
    values: dict[float, float | None] = {}
    header = None
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
        temperature = float(columns[0])
        if temperature in values:
            raise ValueError(f"duplicate log(T)={temperature} in {path}")
        values[temperature] = float(columns[1]) if len(columns) == 2 else None
    if log_nh is None or log_ncol is None:
        raise ValueError(f"missing loop metadata: {path}")
    if header != ("C_2_157.636m",):
        raise ValueError(f"unexpected line header {header!r}: {path}")
    if len(values) != EXPECTED_T:
        raise ValueError(f"expected {EXPECTED_T} temperatures in {path}, found {len(values)}")
    return log_nh, log_ncol, values


def _load_directory(path: Path, log_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if list(path.glob("*.mach")):
        raise RuntimeError(f"Cloudy jobs are still active: {path}")
    files = sorted(path.glob("*_run*.dat"))
    if len(files) != EXPECTED_AXIS ** 2:
        raise ValueError(f"expected 100 runs in {path}, found {len(files)}")
    parsed = []
    seen_ids = set()
    for file in files:
        match = RUN_RE.search(file.name)
        if match is None:
            raise ValueError(f"unexpected filename: {file}")
        run_id = int(match.group(1))
        if run_id in seen_ids:
            raise ValueError(f"duplicate run {run_id}: {path}")
        seen_ids.add(run_id)
        parsed.append((run_id, file, *_parse_run(file)))
    if seen_ids != set(range(1, 101)):
        raise ValueError(f"run ids are not exactly 1..100: {path}")
    log_nh = np.unique([item[2] for item in parsed])
    log_ncol = np.unique([item[3] for item in parsed])
    if log_nh.size != 10 or log_ncol.size != 10:
        raise ValueError(f"grid is not 10x10: {path}")
    raw = np.full((10, 10, EXPECTED_T), np.nan)
    for run_id, file, nh, ncol, values in parsed:
        i = int(np.flatnonzero(log_nh == nh)[0])
        j = int(np.flatnonzero(log_ncol == ncol)[0])
        if run_id != i * 10 + j + 1:
            raise ValueError(f"run ordering mismatch: {file}")
        for reported_t, value in values.items():
            k = int(np.abs(log_t - reported_t).argmin())
            if abs(float(log_t[k]) - reported_t) > TOLERANCE_DEX:
                raise ValueError(f"off-grid log(T)={reported_t}: {file}")
            if value is not None:
                raw[i, j, k] = value
    return log_nh, log_ncol, raw


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    examples = root / "work/cloudy_cooling_tools_history/examples/grackle"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=root / "data/cloudy_lines_hm2012_z0_physics_ablation_4state_3line_10x10x20.npz")
    parser.add_argument("--baseline", type=Path, default=examples / "hm_2012_cii_baseline_10x10x4_T3p6_10_output")
    parser.add_argument("--mol-ct", type=Path, default=examples / "hm_2012_cii_mol_ct_10x10x4_T3p6_10_output")
    parser.add_argument("--baseline-par", type=Path, default=examples / "hm_2012_cii_baseline_10x10x4_T3p6_10.par")
    parser.add_argument("--mol-ct-par", type=Path, default=examples / "hm_2012_cii_mol_ct_10x10x4_T3p6_10.par")
    parser.add_argument("--output", type=Path, default=root / "data/cloudy_cii_hm2012_z0_baseline_molct_10x10x23_T3p6_to1e9.npz")
    parser.add_argument("--report", type=Path, default=root / "data/cloudy_cii_hm2012_z0_baseline_molct_10x10x23_T3p6_to1e9.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in vars(args):
        value = getattr(args, name)
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if (args.output.exists() or args.report.exists()) and not args.force:
        raise FileExistsError("output exists; pass --force to replace it")

    low_log_t = np.linspace(np.log10(3.6), 1.0, EXPECTED_T)
    low_state = []
    axes = None
    for path in (args.baseline, args.mol_ct):
        nh, ncol, raw = _load_directory(path, low_log_t)
        if axes is None:
            axes = (nh, ncol)
        elif not (np.array_equal(axes[0], nh) and np.array_equal(axes[1], ncol)):
            raise ValueError("low-temperature state axes differ")
        low_state.append(raw)
    low_raw = np.stack(low_state)

    with np.load(args.old, allow_pickle=False) as old:
        old_nh = np.asarray(old["log_nH"], dtype=float)
        old_ncol = np.asarray(old["log_NH"], dtype=float)
        old_log_t = np.asarray(old["log_T"], dtype=float)
        old_raw = np.asarray(old["log_emissivity_per_nH2"], dtype=float)[list(OLD_STATE_INDICES), 0]
    assert axes is not None
    if not (np.array_equal(axes[0], old_nh) and np.array_equal(axes[1], old_ncol)):
        raise ValueError("extension axes differ from old table")
    if abs(float(old_log_t[0]) - 1.0) > 1e-12:
        raise ValueError("old table does not begin at 10 K")

    new_overlap = low_raw[..., -1]
    old_overlap = old_raw[..., 0]
    comparable = np.isfinite(new_overlap) & np.isfinite(old_overlap) & (new_overlap > ZERO_LIMIT) & (old_overlap > ZERO_LIMIT)
    overlap_dex = np.abs(new_overlap[comparable] - old_overlap[comparable])
    both_present = np.isfinite(new_overlap) & np.isfinite(old_overlap)
    class_mismatch = np.count_nonzero(
        both_present
        & ((new_overlap <= ZERO_LIMIT) != (old_overlap <= ZERO_LIMIT))
    )
    missing_new_overlap = int(np.count_nonzero(~np.isfinite(new_overlap)))
    missing_old_overlap = int(np.count_nonzero(~np.isfinite(old_overlap)))
    both_missing_overlap = int(np.count_nonzero(
        ~np.isfinite(new_overlap) & ~np.isfinite(old_overlap)
    ))
    max_overlap = float(overlap_dex.max()) if overlap_dex.size else 0.0
    if class_mismatch or max_overlap > 2.0e-3 or overlap_dex.size < 190:
        raise RuntimeError(
            "10 K overlap QA failed: "
            f"new missing={missing_new_overlap}, old missing={missing_old_overlap}, "
            f"class mismatches={class_mismatch}, comparable={overlap_dex.size}, "
            f"max |delta log10|={max_overlap:g}"
        )

    # Keep the old table authoritative at 10 K, but allow a newly successful
    # overlap value to repair an old failure.  A node missing in both remains
    # a real unavailable failure for the runtime sampling check.
    repaired_old_raw = old_raw.copy()
    replace_old = ~np.isfinite(repaired_old_raw[..., 0]) & np.isfinite(new_overlap)
    repaired_old_raw[..., 0][replace_old] = new_overlap[replace_old]
    merged_log_t = np.concatenate((low_log_t[:-1], old_log_t))
    raw = np.concatenate((low_raw[..., :-1], repaired_old_raw), axis=-1)[:, None]
    failure = ~np.isfinite(raw)
    zero = (~failure) & (raw <= ZERO_LIMIT)
    coefficient = np.zeros(raw.shape)
    positive = (~failure) & (~zero)
    coefficient[positive] = np.power(10.0, raw[positive])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        bundle_schema_version=np.asarray(1, dtype=np.int32),
        table_kind=np.asarray("Cloudy extended low-temperature CII table"),
        axis_order=np.asarray("state,line,log_nH,log_NH,log_T"),
        state_labels=np.asarray(STATES), line_keys=np.asarray(("cii",)),
        line_labels=np.asarray(("C  2 157.636m",)),
        log_nH=old_nh, log_NH=old_ncol, log_T=merged_log_t,
        log_emissivity_per_nH2=raw, emissivity_per_nH2=coefficient,
        failure_mask=failure, original_failure_mask=failure.copy(), zero_mask=zero,
        interpolated_mask=np.zeros_like(failure),
        out_of_bounds_policy=np.asarray("raise"),
        normalization=np.asarray("local deepest-zone emissivity / n_H^2"),
        column_model=np.asarray("explicit stop column density; no Jeans length"),
        uv_background=np.asarray("HM2012 z=0 shielded"), cloudy_version=np.asarray("17.02"),
        carbon_abundance_log10=np.asarray(-3.795880), zero_sentinel_max=np.asarray(ZERO_LIMIT),
        old_table=np.asarray(str(args.old)), old_table_sha256=np.asarray(_sha256(args.old)),
        extension_directories=np.asarray((str(args.baseline), str(args.mol_ct))),
        extension_parameter_files=np.asarray((str(args.baseline_par), str(args.mol_ct_par))),
        extension_parameter_sha256=np.asarray((_sha256(args.baseline_par), _sha256(args.mol_ct_par))),
    )
    report = {
        "output": str(args.output), "shape": list(raw.shape),
        "states": list(STATES), "temperature_K": (10.0 ** merged_log_t).tolist(),
        "new_temperature_K": (10.0 ** low_log_t[:-1]).tolist(),
        "overlap_temperature_K": 10.0,
        "overlap_comparable_nodes": int(overlap_dex.size),
        "overlap_missing_new_nodes_replaced_by_old_table": missing_new_overlap,
        "overlap_missing_old_nodes": missing_old_overlap,
        "overlap_missing_in_both_tables": both_missing_overlap,
        "old_overlap_failures_repaired_by_new_table": int(np.count_nonzero(replace_old)),
        "overlap_class_mismatches": int(class_mismatch),
        "overlap_max_abs_delta_dex": max_overlap,
        "failure_nodes": int(np.count_nonzero(failure)),
        "true_zero_nodes": int(np.count_nonzero(zero)),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
