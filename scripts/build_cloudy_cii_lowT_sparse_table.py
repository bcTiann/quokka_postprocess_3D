"""Build the sparse 10--3000 K Cloudy [C II] diagnostic table.

This is deliberately separate from the production DESPOTIC/Cloudy hybrid.
Missing Cloudy rows abort the build and are recorded; no numerical filling is
performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_cloudy_cii_full_table import (
    ZERO_SENTINEL_MAX,
    _temperature_indices,
    load_directory,
)


def build_table(
    input_dir: Path,
    output: Path,
    failure_manifest: Path,
) -> None:
    log_t = np.linspace(np.log10(10.0), np.log10(3000.0), 21)
    records = load_directory(input_dir)
    log_nh = np.unique([key[0] for key in records])
    log_nh_column = np.unique([key[1] for key in records])
    expected = {(n, c) for n in log_nh for c in log_nh_column}
    if set(records) != expected or log_nh.size != 5 or log_nh_column.size != 5:
        raise ValueError(
            f'expected a Cartesian 5x5 sparse grid, found '
            f'{log_nh.size}x{log_nh_column.size}'
        )

    raw_log = np.full((5, 5, log_t.size), np.nan)
    nh_index = {value: index for index, value in enumerate(log_nh)}
    column_index = {
        value: index for index, value in enumerate(log_nh_column)
    }
    for (run_log_nh, run_log_column), files_and_rows in records.items():
        i = nh_index[run_log_nh]
        j = column_index[run_log_column]
        for path, rows in files_and_rows:
            indices = _temperature_indices(rows, log_t, path)
            for row, k in zip(rows, indices):
                if np.isfinite(raw_log[i, j, k]):
                    raise ValueError(f'duplicate result for {(run_log_nh, run_log_column, k)}')
                raw_log[i, j, k] = float(row[1])

    failures = []
    for i, j, k in np.argwhere(~np.isfinite(raw_log)):
        failures.append({
            'nH_index': int(i),
            'NH_index': int(j),
            'T_index': int(k),
            'log_nH': float(log_nh[i]),
            'log_NH': float(log_nh_column[j]),
            'log_T': float(log_t[k]),
            'temperature_K': float(10.0 ** log_t[k]),
        })
    failure_manifest.parent.mkdir(parents=True, exist_ok=True)
    failure_manifest.write_text(json.dumps({
        'diagnostic_only': True,
        'input_dir': str(input_dir.resolve()),
        'grid_shape': list(raw_log.shape),
        'missing_calculations': len(failures),
        'failures': failures,
    }, indent=2) + '\n')
    if failures:
        print(f'Diagnostic table NOT written: {len(failures)} calculations missing.')
        print(f'Failure manifest: {failure_manifest}')
        raise SystemExit(2)

    zero_mask = raw_log <= ZERO_SENTINEL_MAX
    coefficient = np.zeros_like(raw_log)
    coefficient[~zero_mask] = np.power(10.0, raw_log[~zero_mask])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(2, dtype=np.int32),
        diagnostic_only=np.asarray(True),
        log_nH=log_nh,
        log_NH=log_nh_column,
        log_T=log_t,
        emissivity_per_nH2=coefficient,
        log_emissivity_per_nH2=raw_log,
        zero_mask=zero_mask,
        failure_mask=np.zeros_like(zero_mask),
        line_label=np.asarray('C  2 157.636m'),
        uv_background=np.asarray('HM2012 z=0 shielded'),
        cloudy_version=np.asarray('17.02'),
        carbon_abundance_log10=np.asarray(-3.795880),
        h2_molecule_enabled=np.asarray(False),
        charge_transfer_enabled=np.asarray(False),
        column_model=np.asarray('explicit stop column density; no Jeans length'),
        intended_use=np.asarray('low-temperature comparison with DESPOTIC only'),
        interpolation_policy=np.asarray(
            'log coefficient if all 8 corners positive; '
            'linear coefficient if any corner zero'
        ),
        out_of_bounds_policy=np.asarray('raise'),
    )
    print(
        f'Wrote {output}: shape={coefficient.shape}, '
        f'Cloudy_zero_nodes={int(zero_mask.sum())}, failures=0'
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    example = root / 'work/cloudy_cooling_tools_history/examples/grackle'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-dir', type=Path,
        default=example / 'hm_2012_cii_lowT_sparse_output',
    )
    parser.add_argument(
        '--output', type=Path,
        default=root / 'data/cloudy_cii_hm2012_z0_lowT_sparse_diagnostic.npz',
    )
    parser.add_argument(
        '--failure-manifest', type=Path,
        default=example / 'hm_2012_cii_lowT_sparse_failures.json',
    )
    args = parser.parse_args()
    build_table(args.input_dir, args.output, args.failure_manifest)


if __name__ == '__main__':
    main()
