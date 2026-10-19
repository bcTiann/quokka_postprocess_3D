"""Prepare cold nH–NH distributions for every snapshot in selected jobs."""
from __future__ import annotations

import argparse
from pathlib import Path

from quokka2s.cold_gas_survey import read_cold_phase_grid, run_cold_gas_survey
from quokka2s.table_input_jobs import read_job_survey_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cold_gas_phase.yaml"))
    parser.add_argument("--extension-config", type=Path, default=Path("configs/despotic_extension.yaml"))
    args = parser.parse_args(argv)
    settings = read_job_survey_config(path=args.config)
    grid = read_cold_phase_grid(
        original_table=settings.despotic_table,
        extension_config=args.extension_config.expanduser().resolve(),
    )
    run_cold_gas_survey(settings=settings, grid=grid)


if __name__ == "__main__":
    main()
