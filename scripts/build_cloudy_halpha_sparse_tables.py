"""Convert the completed sparse Cloudy H-alpha scans to strict NPZ tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analyze_cloudy_halpha_temperature_tail import load_grid


def write_table(
    input_dir: Path,
    output: Path,
    *,
    t_min: float,
    t_max: float,
    t_points: int,
) -> None:
    log_nh, log_nh_column, log_t, raw_log, failure_mask = load_grid(
        input_dir,
        t_min=t_min,
        t_max=t_max,
        t_points=t_points,
    )
    coefficient = np.zeros_like(raw_log)
    valid = ~failure_mask
    coefficient[valid] = np.power(10.0, raw_log[valid])
    zero_mask = valid & (coefficient == 0.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(2, dtype=np.int64),
        log_nH=log_nh,
        log_NH=log_nh_column,
        log_T=log_t,
        log_emissivity_per_nH2=np.where(valid, raw_log, 0.0),
        emissivity_per_nH2=coefficient,
        failure_mask=failure_mask,
        zero_mask=zero_mask,
        out_of_bounds_policy=np.asarray("raise"),
        line_label=np.asarray("H  1 6562.81A"),
        radiation_field=np.asarray("HM2012 z=0"),
    )
    print(
        f"Wrote {output}: shape={raw_log.shape}, "
        f"failures={int(failure_mask.sum())}, zeros={int(zero_mask.sum())}"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    example = root / "work/cloudy_cooling_tools_history/examples/grackle"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--low-input",
        type=Path,
        default=example / "hm_2012_halpha_lowT_sparse_output",
    )
    parser.add_argument(
        "--high-input",
        type=Path,
        default=example / "hm_2012_halpha_sparse_output",
    )
    parser.add_argument(
        "--low-output",
        type=Path,
        default=root / "data/cloudy_halpha_hm2012_z0_lowT_sparse.npz",
    )
    parser.add_argument(
        "--high-output",
        type=Path,
        default=root / "data/cloudy_halpha_hm2012_z0_highT_sparse.npz",
    )
    args = parser.parse_args()

    write_table(
        args.low_input,
        args.low_output,
        t_min=10.0,
        t_max=3000.0,
        t_points=21,
    )
    write_table(
        args.high_input,
        args.high_output,
        t_min=3000.0,
        t_max=1.0e9,
        t_points=21,
    )


if __name__ == "__main__":
    main()
