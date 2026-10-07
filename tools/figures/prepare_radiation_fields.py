#!/usr/bin/env python3
"""Save one historical Cloudy radiation recipe; this command does not draw.

Prepare components: --recipe components [--components-only]
Prepare unattenuated fields: --recipe unattenuated [--include-cmb]
The optional flags select the established figure bundle names. Components-only
saves the same numerical components/sums as the two-panel figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quokka2s.paths import resolve_path
from quokka2s.products.radiation_fields import (
    COMBINED_RADIATION_STEM,
    COMPONENT_RADIATION_STEM,
    UNATTENUATED_CMB_RADIATION_STEM,
    UNATTENUATED_RADIATION_STEM,
    calculate_component_radiation_data,
    calculate_unattenuated_radiation_data,
)


def save_radiation_data(payload, report, output_dir, stem_name, recipe):
    """Write already prepared arrays and the historical recipe's JSON metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / stem_name
    npz_path = stem.with_suffix(".npz")
    np.savez_compressed(npz_path, **payload)
    if recipe == "components":
        report["outputs"] = {
            "plot_png": str(stem.with_suffix(".png")),
            "plot_pdf": str(stem.with_suffix(".pdf")),
            "npz": str(npz_path),
        }
    else:
        report["outputs"] = {
            "png": str(stem.with_suffix(".png")),
            "pdf": str(stem.with_suffix(".pdf")),
            "npz": str(npz_path),
        }
    stem.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(npz_path)


def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("components", "unattenuated"), required=True)
    parser.add_argument("--data-dir", type=Path, default=project_root / "runtime/cloudy_eightline/sed")
    parser.add_argument("--output-dir", type=Path, default=project_root / "output/radiation_fields")
    parser.add_argument("--components-only", action="store_true")
    parser.add_argument("--include-cmb", action="store_true")
    args = parser.parse_args()
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    if args.recipe == "components":
        if args.include_cmb:
            parser.error("--include-cmb belongs to the unattenuated recipe")
        payload, report = calculate_component_radiation_data(data_dir=data_dir)
        stem_name = COMPONENT_RADIATION_STEM if args.components_only else COMBINED_RADIATION_STEM
    else:
        if args.components_only:
            parser.error("--components-only belongs to the components recipe")
        payload, report = calculate_unattenuated_radiation_data(
            data_dir=data_dir,
            include_cmb=args.include_cmb,
        )
        stem_name = UNATTENUATED_CMB_RADIATION_STEM if args.include_cmb else UNATTENUATED_RADIATION_STEM
    save_radiation_data(
        payload=payload,
        report=report,
        output_dir=output_dir,
        stem_name=stem_name,
        recipe=args.recipe,
    )


if __name__ == "__main__":
    main()
