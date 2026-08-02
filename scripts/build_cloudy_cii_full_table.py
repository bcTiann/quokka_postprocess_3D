"""Build a strict full-temperature HM2012 [C II] runtime table.

Unlike the earlier coarse-table builder, this program never fills missing
Cloudy calculations.  A numerical ``-99`` emitted by CIAOLoop_lines is a
successful Cloudy zero; an absent temperature row is a genuine failure and
prevents creation of the production NPZ until a supplemental rerun supplies
that row.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


LOOP_RE = re.compile(r'^#\s*(hden|stop column density)\s+(.+?)\s*$')
ZERO_SENTINEL_MAX = -90.0


def read_run(path: Path) -> tuple[float, float, np.ndarray]:
    """Read one run, allowing an empty data section after a total failure."""
    log_nH = log_NH = None
    rows: list[tuple[float, float]] = []
    for line in path.read_text().splitlines():
        match = LOOP_RE.match(line)
        if match:
            if match.group(1) == 'hden':
                log_nH = float(match.group(2))
            else:
                log_NH = float(match.group(2))
            continue
        if not line or line.startswith('#'):
            continue
        columns = line.split()
        if len(columns) >= 2:
            rows.append((float(columns[0]), float(columns[1])))
    if log_nH is None or log_NH is None:
        raise ValueError(f'missing loop metadata in {path}')
    return log_nH, log_NH, np.asarray(rows, dtype=float).reshape(-1, 2)


def load_directory(path: Path) -> dict[tuple[float, float], list[tuple[Path, np.ndarray]]]:
    files = sorted(path.glob('*_run*.dat'))
    if not files:
        raise FileNotFoundError(f'no CIAOLoop .dat files in {path}')
    records: dict[tuple[float, float], list[tuple[Path, np.ndarray]]] = {}
    for file_path in files:
        log_nH, log_NH, rows = read_run(file_path)
        records.setdefault((log_nH, log_NH), []).append((file_path, rows))
    return records


def _temperature_indices(rows: np.ndarray, log_T: np.ndarray, path: Path) -> np.ndarray:
    if rows.size == 0:
        return np.empty(0, dtype=int)
    indices = np.abs(rows[:, 0, None] - log_T[None, :]).argmin(axis=1)
    residual = np.abs(rows[:, 0] - log_T[indices])
    if np.any(residual > 5.1e-4):
        bad = rows[np.argmax(residual), 0]
        raise ValueError(f'reported log(T)={bad} is not on the requested grid: {path}')
    if np.unique(indices).size != indices.size:
        raise ValueError(f'duplicate temperature row in {path}')
    return indices


def build_table(
    primary_dir: Path,
    supplement_dirs: list[Path],
    output: Path,
    failure_manifest: Path,
    *,
    t_min: float,
    t_max: float,
    n_T: int,
    allow_unused_failures: bool = False,
) -> None:
    if not (t_min > 0.0 and t_max > t_min and n_T >= 2):
        raise ValueError('invalid temperature grid')
    log_T = np.linspace(np.log10(t_min), np.log10(t_max), n_T)
    primary = load_directory(primary_dir)
    log_nH = np.unique([key[0] for key in primary])
    log_NH = np.unique([key[1] for key in primary])
    expected_keys = {(n, c) for n in log_nH for c in log_NH}
    if set(primary) != expected_keys:
        missing_runs = sorted(expected_keys - set(primary))
        extra_runs = sorted(set(primary) - expected_keys)
        raise ValueError(
            f'primary grid is not Cartesian: missing={missing_runs[:8]}, '
            f'extra={extra_runs[:8]}'
        )
    if len(expected_keys) != 240 or log_nH.size != 16 or log_NH.size != 15:
        raise ValueError(
            f'expected a 16x15 primary grid (240 runs), found '
            f'{log_nH.size}x{log_NH.size} ({len(expected_keys)} runs)'
        )

    raw_log = np.full((log_nH.size, log_NH.size, n_T), np.nan)
    source = np.full(raw_log.shape, '', dtype='U256')
    nH_index = {value: i for i, value in enumerate(log_nH)}
    NH_index = {value: i for i, value in enumerate(log_NH)}
    overlap_differences: list[float] = []

    def merge(records, *, supplemental: bool) -> None:
        for (run_log_nH, run_log_NH), files_and_rows in records.items():
            if (run_log_nH, run_log_NH) not in expected_keys:
                raise ValueError(
                    f'supplement contains a point outside the primary axes: '
                    f'{(run_log_nH, run_log_NH)}'
                )
            i = nH_index[run_log_nH]
            j = NH_index[run_log_NH]
            for path, rows in files_and_rows:
                indices = _temperature_indices(rows, log_T, path)
                for row, k in zip(rows, indices):
                    new_value = float(row[1])
                    old_value = raw_log[i, j, k]
                    if np.isfinite(old_value):
                        # Supplements exist to fill missing rows, never to
                        # silently replace a successful primary calculation.
                        if supplemental:
                            overlap_differences.append(abs(new_value - old_value))
                        continue
                    raw_log[i, j, k] = new_value
                    source[i, j, k] = str(path)

    merge(primary, supplemental=False)
    for directory in supplement_dirs:
        merge(load_directory(directory), supplemental=True)

    failure_mask = ~np.isfinite(raw_log)
    failures = []
    for i, j, k in np.argwhere(failure_mask):
        failures.append({
            'nH_index': int(i),
            'NH_index': int(j),
            'T_index': int(k),
            'log_nH': float(log_nH[i]),
            'log_NH': float(log_NH[j]),
            'log_T': float(log_T[k]),
            'temperature_K': float(10.0 ** log_T[k]),
        })
    failure_manifest.parent.mkdir(parents=True, exist_ok=True)
    failure_manifest.write_text(json.dumps({
        'primary_dir': str(primary_dir.resolve()),
        'supplement_dirs': [str(p.resolve()) for p in supplement_dirs],
        'grid_shape': list(raw_log.shape),
        'missing_calculations': len(failures),
        'failures': failures,
    }, indent=2) + '\n')
    if failures and not allow_unused_failures:
        print(f'Production table NOT written: {len(failures)} Cloudy calculations missing.')
        print(f'Failure manifest: {failure_manifest}')
        for item in failures[:12]:
            print(
                '  log_nH={log_nH:.6g} log_NH={log_NH:.6g} '
                'log_T={log_T:.6g} T={temperature_K:.6g} K'.format(**item)
            )
        raise SystemExit(2)

    if failures:
        print(
            f'Writing failure-aware table with {len(failures)} unavailable nodes. '
            'Runtime lookup will raise if any unavailable node has positive '
            'interpolation weight.'
        )

    zero_mask = raw_log <= ZERO_SENTINEL_MAX
    coefficient = np.zeros(raw_log.shape, dtype=float)
    positive = ~zero_mask & ~failure_mask
    coefficient[positive] = np.power(10.0, raw_log[positive])

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(2, dtype=np.int32),
        log_nH=log_nH,
        log_NH=log_NH,
        log_T=log_T,
        emissivity_per_nH2=coefficient,
        log_emissivity_per_nH2=raw_log,
        zero_mask=zero_mask,
        failure_mask=failure_mask,
        line_label=np.asarray('C  2 157.636m'),
        uv_background=np.asarray('HM2012 z=0 shielded'),
        cloudy_version=np.asarray('17.02'),
        carbon_abundance_log10=np.asarray(-3.795880),
        h2_molecule_enabled=np.asarray(False),
        charge_transfer_enabled=np.asarray(False),
        column_model=np.asarray('explicit stop column density; no Jeans length'),
        interpolation_policy=np.asarray(
            'log coefficient if all 8 corners positive; '
            'linear coefficient if any corner zero'
        ),
        out_of_bounds_policy=np.asarray(
            'temperature_above_max_zero; other_axes_raise'
        ),
        temperature_above_max_basis=np.asarray(
            '5x5 sparse HM2012 scan: latest first exact zero at '
            'log10(T/K)=6.436; no positive reappearance through 1e9 K'
        ),
        failed_node_policy=np.asarray(
            'unavailable; runtime raises if interpolation weight > 1e-12'
        ),
    )
    max_overlap = max(overlap_differences, default=0.0)
    print(
        f'Wrote {output}: shape={raw_log.shape}, '
        f'Cloudy_zero_nodes={int(zero_mask.sum())}, '
        f'failures={int(failure_mask.sum())}, '
        f'max_primary_vs_supplement_overlap_dex={max_overlap:.6g}'
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_example = (
        project_root / 'work/cloudy_cooling_tools_history/examples/grackle'
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--primary-dir', type=Path,
        default=default_example / 'hm_2012_cii_cloudy_full_output',
    )
    parser.add_argument(
        '--supplement-dir', action='append', type=Path, default=[],
        help='Directory from a targeted solver rerun; may be repeated.',
    )
    parser.add_argument(
        '--output', type=Path,
        default=project_root / 'data/cloudy_cii_hm2012_z0_full.npz',
    )
    parser.add_argument(
        '--failure-manifest', type=Path,
        default=(default_example / 'hm_2012_cii_cloudy_full_failures.json'),
    )
    parser.add_argument('--t-min', type=float, default=3.0e3)
    parser.add_argument('--t-max', type=float, default=2.7289777828080403e6)
    parser.add_argument('--t-points', type=int, default=31)
    parser.add_argument(
        '--allow-unused-failures', action='store_true',
        help=(
            'Write a table retaining failed nodes as unavailable. The runtime '
            'lookup raises if a query gives any failed node positive weight.'
        ),
    )
    args = parser.parse_args()
    build_table(
        args.primary_dir,
        args.supplement_dir,
        args.output,
        args.failure_manifest,
        t_min=args.t_min,
        t_max=args.t_max,
        n_T=args.t_points,
        allow_unused_failures=args.allow_unused_failures,
    )


if __name__ == '__main__':
    main()
