"""Build the runtime HM2012 [C II] lookup table from CIAOLoop_lines output."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from quokka2s.cloudy_cii_lookup import fill_failures_along_log_temperature


LOOP_RE = re.compile(r'^#\s*(hden|stop column density)\s+(.+?)\s*$')


def _read_run(path: Path) -> tuple[float, float, np.ndarray]:
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
    if log_nH is None or log_NH is None or not rows:
        raise ValueError(f'incomplete Cloudy output: {path}')
    return log_nH, log_NH, np.asarray(rows, dtype=float)


def build_table(input_dir: Path, output: Path) -> None:
    files = sorted(input_dir.glob('*_run*.dat'))
    if not files:
        raise FileNotFoundError(f'no CIAOLoop .dat files in {input_dir}')

    records = [_read_run(path) for path in files]
    log_nH = np.unique([record[0] for record in records])
    log_NH = np.unique([record[1] for record in records])
    # The production parameter file requests 20 temperatures.  Failed Cloudy
    # temperatures are omitted from a run's .dat file, so individual runs can
    # contain fewer rows and must be aligned by their reported log(T) values.
    n_T = 20
    expected_runs = log_nH.size * log_NH.size
    if len(records) != expected_runs:
        raise ValueError(f'found {len(records)} runs, expected {expected_runs}')

    # CIAOLoop writes log(T) with only three decimal places.  Reconstruct the
    # exact logarithmic axis from the parameter-file endpoints used for this
    # production grid, then match every surviving row to its nearest sample.
    log_T = np.linspace(np.log10(3000.0), np.log10(4.349742488077094e4), n_T)

    raw = np.full((log_nH.size, log_NH.size, n_T), np.nan)
    nH_index = {value: i for i, value in enumerate(log_nH)}
    NH_index = {value: i for i, value in enumerate(log_NH)}
    for run_log_nH, run_log_NH, rows in records:
        temperature_index = np.abs(
            rows[:, 0, None] - log_T[None, :]
        ).argmin(axis=1)
        residual = np.abs(rows[:, 0] - log_T[temperature_index])
        if np.any(residual > 5.1e-4):
            raise ValueError('reported Cloudy temperature is not on the requested grid')
        if np.unique(temperature_index).size != temperature_index.size:
            raise ValueError('duplicate temperature row in one Cloudy output')
        raw[
            nH_index[run_log_nH], NH_index[run_log_NH], temperature_index
        ] = rows[:, 1]

    filled, failure_mask = fill_failures_along_log_temperature(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        log_nH=log_nH,
        log_NH=log_NH,
        log_T=log_T,
        log_emissivity_per_nH2=filled,
        failure_mask=failure_mask,
        line_label=np.asarray('C  2 157.636m'),
        uv_background=np.asarray('HM2012 z=0 shielded'),
        cloudy_version=np.asarray('17.02'),
        carbon_abundance_log10=np.asarray(-3.795880),
    )
    print(
        f'Wrote {output}: shape={filled.shape}, '
        f'interpolated_failures={int(failure_mask.sum())}'
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-dir', type=Path,
        default=(project_root / 'work/cloudy_cooling_tools_history/examples/'
                 'grackle/hm_2012_cii_cloudy_coarse_output'),
    )
    parser.add_argument(
        '--output', type=Path,
        default=project_root / 'data/cloudy_cii_hm2012_z0_coarse.npz',
    )
    args = parser.parse_args()
    build_table(args.input_dir, args.output)


if __name__ == '__main__':
    main()
