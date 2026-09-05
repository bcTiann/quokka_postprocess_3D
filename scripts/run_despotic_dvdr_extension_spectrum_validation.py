#!/usr/bin/env python3
"""Run matched legacy/extended dV/dr spectra and compare their outputs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "output/plt0655228_down1_Lext15kpc/dvdr_extension_validation"
)
TABLES = {
    "legacy": ROOT / "output_tables_3D_GOW_LVG/despotic_table_co10_co21_clean.npz",
    "extended": ROOT / "output_tables_3D_GOW_LVG/despotic_table_co10_co21_dvdr_fullrange_clean.npz",
}


def _is_complete(report_path: Path) -> bool:
    if not report_path.is_file():
        return False
    try:
        return bool(json.loads(report_path.read_text())["completed_full_domain"])
    except (KeyError, OSError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    spectrum_script = ROOT / "scripts/plot_hm12_filtered_ism_sixline_spectra.py"

    for case, table in TABLES.items():
        tag = "legacy_dvdr" if case == "legacy" else "extended_dvdr"
        for los in ("y", "z"):
            output_dir = output_root / case / f"LOS{los}"
            report_path = output_dir / f"{tag}_sixline_Tsplit_Rinf_LOS{los}.json"
            if _is_complete(report_path):
                print(f"[reuse] {report_path}", flush=True)
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(spectrum_script),
                "--despotic-table", str(table),
                "--recompute-tdsp",
                "--recompute-dvdr",
                "--los", los,
                "--workers", str(args.workers),
                "--output-dir", str(output_dir),
                "--filename-tag", tag,
            ]
            print(f"[run] {case} LOS {los}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/compare_despotic_dvdr_extension_spectra.py"),
            "--output-root", str(output_root),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
