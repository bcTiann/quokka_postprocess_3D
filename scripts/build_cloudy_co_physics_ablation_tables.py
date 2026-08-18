"""Build four-state HM2012 CO(1-0) and CO(2-1) lookup tables.

This configures and reuses the validated generic four-state line-table builder
so CO products follow exactly the same schema and failure policy as the CII,
H-alpha, and H I tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import build_cloudy_line_physics_ablation_tables as builder


LINE_KEYS = ("co10", "co21")
LINE_LABELS = ("CO 2600.05m", "CO 1300.05m")
DAT_LINE_LABELS = ("CO_2600.05m", "CO_1300.05m")


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    example_dir = project_root / "work/cloudy_cooling_tools_history/examples/grackle"
    parser = argparse.ArgumentParser(description=__doc__)
    for state in builder.STATE_LABELS:
        option = state.replace("_", "-")
        parser.add_argument(
            f"--{option}-input",
            type=Path,
            default=example_dir / f"hm_2012_co_{state}_10x10x20_output",
        )
        parser.add_argument(
            f"--{option}-par",
            type=Path,
            default=example_dir / f"hm_2012_co_{state}_10x10x20.par",
        )
    parser.add_argument(
        "--bundle-output",
        type=Path,
        default=(
            project_root
            / "data/cloudy_co_hm2012_z0_physics_ablation_4state_2line_10x10x20.npz"
        ),
    )
    parser.add_argument(
        "--views-dir",
        type=Path,
        default=project_root / "data/cloudy_co_physics_ablation_10x10x20_views",
    )
    parser.add_argument(
        "--failure-manifest",
        type=Path,
        default=(
            project_root
            / "data/cloudy_co_hm2012_z0_physics_ablation_10x10x20_failures.json"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing CO bundle, views, and failure manifest",
    )
    return parser.parse_args()


def _view_filename(state: str, line_key: str) -> str:
    return f"cloudy_co_hm2012_z0_ablation_{state}_{line_key}_10x10x20.npz"


def main() -> None:
    builder.LINE_KEYS = LINE_KEYS
    builder.LINE_LABELS = LINE_LABELS
    builder.DAT_LINE_LABELS = DAT_LINE_LABELS
    builder._parse_args = _parse_args
    builder._view_filename = _view_filename
    builder.main()


if __name__ == "__main__":
    main()
