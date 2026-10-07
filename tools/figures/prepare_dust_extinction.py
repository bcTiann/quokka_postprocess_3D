#!/usr/bin/env python3
"""Save the full Draine grid and line samples; this command does not draw."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quokka2s.paths import resolve_path
from quokka2s.physics.dust_attenuation import DEFAULT_DRAINE_TABLE
from quokka2s.products.dust_extinction import (
    DUST_EXTINCTION_STEM,
    prepare_dust_extinction_data,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_DRAINE_TABLE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    table_path = resolve_path(args.table)
    output_dir = resolve_path(args.output_dir)
    payload, report = prepare_dust_extinction_data(table_path=table_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / DUST_EXTINCTION_STEM
    np.savez_compressed(stem.with_suffix(".npz"), **payload)
    stem.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(stem.with_suffix(".npz"))


if __name__ == "__main__":
    main()
