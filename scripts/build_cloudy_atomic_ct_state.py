"""Build CII, Halpha, and HI views for HM2012+Draine+CR+charge transfer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import build_cloudy_atomic_radiation_tables as base


STATE = "hm2012_draine_cr_ct"
STEM = "hm2012_plus_draine_cr_ct_defaultabund_multiline"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    examples = root / "work/cloudy_cooling_tools_history/examples/grackle"
    output_dir = root / "data/cloudy_atomic_defaultabund_radiation_3state_views"
    report_path = root / "data/cloudy_atomic_hm2012_draine_cr_ct_failures.json"
    log_t = np.linspace(np.log10(base.T_MIN_K), np.log10(base.T_MAX_K), base.N_T)

    base.STATE_STEMS[STATE] = STEM
    base.RADIATION[STATE] = (
        "HM2012 z=0 + table Draine + cosmic rays rate 2e-17 s^-1 "
        "+ charge transfer"
    )
    base.CR_RATE[STATE] = 2.0e-17
    base.NO_CHARGE_TRANSFER[STATE] = False

    column_dir = examples / f"{STEM}_column_10x10x21_output"
    column_par = examples / f"{STEM}_column_10x10x21.par"
    jeans_dir = examples / f"{STEM}_jeans_10x21_output"
    jeans_par = examples / f"{STEM}_jeans_10x21.par"

    log_nh, log_column, column_raw = base._load_column(column_dir, log_t)
    records = base._write_views(
        output_dir,
        state=STATE,
        geometry="column",
        log_nh=log_nh,
        log_column=log_column,
        log_t=log_t,
        raw=column_raw,
        parameter_file=column_par,
    )
    jeans_log_nh, jeans_raw = base._load_jeans(jeans_dir, log_t)
    if not np.array_equal(log_nh, jeans_log_nh):
        raise ValueError("column and Jeans density axes differ")
    records.extend(base._write_views(
        output_dir,
        state=STATE,
        geometry="jeans",
        log_nh=jeans_log_nh,
        log_t=log_t,
        raw=jeans_raw,
        parameter_file=jeans_par,
    ))

    report = {
        "state": STATE,
        "lines": [item[0] for item in base.LINES],
        "temperature_bounds_K": [base.T_MIN_K, base.T_MAX_K],
        "charge_transfer_enabled": True,
        "molecular_network_enabled": False,
        "products": records,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
