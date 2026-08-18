#!/usr/bin/env python3
"""Run a sparse three-line A/B test for HM12 radiation interfaces.

The matched models differ only in the incident continuum:

1. the Grackle external HM12 z=0 init file;
2. Cloudy 17.02 native ``table HM12 redshift 0``;
3. external HM12 plus the selectively extinguished Black (1987) ISRF built by
   ``build_hm12_filtered_ism_sed.py``.

All models use nH=1 cm^-3, stop NH=1e20 cm^-2, nine temperatures from 10 K to
1e9 K, default Cloudy abundances, default charge transfer, no H2 molecule,
and no cosmic rays or grains.  CIAOLoop_lines reports final-zone local line
emissivity divided by nH^2.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = {
    "hm12_external": 'loop [init "HM12_UVB/z_*.out"] 0.0000e+00',
    "hm12_native": "command table HM12 redshift 0",
    "hm12_plus_filtered_ism": (
        'loop [init "HM12_ISM_NH21/z_*.out"] 0.0000e+00'
    ),
}

LINE_NAMES = ("CII_157p636um", "Halpha_6562p81A", "HI_21p1207cm")


def _parameter_text(
    cloudy_exe: Path,
    output_prefix: str,
    output_dir: Path,
    radiation_command: str,
) -> str:
    return f"""############################################################
## Sparse HM12/ISM radiation-interface validation: {output_prefix}
############################################################
cloudyExe = {cloudy_exe}
saveCloudyOutputFiles = 0
exitOnCrash = 0
outputFilePrefix = {output_prefix}
outputDir = {output_dir}
runStartIndex = 1
test = 0
cloudyRunMode = 4
lineMapLine = C  2 157.636m
lineMapLine = H  1 6562.81A
lineMapLine = H  1 21.1207c
coolingMapTmin = 1e1
coolingMapTmax = 1e9
coolingMapTpoints = 9
coolingScaleFactor = 1
coolingMapUseJeansLength = 0
command iterate to convergence
command stop temperature off
command set WeakHeatCool -20
command no H2 molecule
loop [hden] 0
loop [stop column density] 20
{radiation_command}
"""


def _load_map(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape != (9, 4):
        raise ValueError(f"expected a 9x4 line map, got {data.shape}: {path}")
    return data


def _line_comparison(first: np.ndarray, second: np.ndarray) -> dict:
    # CIAOLoop_lines uses -99 as an explicit zero-emissivity sentinel.
    first_positive = first > -90.0
    second_positive = second > -90.0
    matched = first_positive & second_positive
    result = {
        "positive_in_both": int(matched.sum()),
        "zero_in_both": int((~first_positive & ~second_positive).sum()),
        "one_sided_zero": int(np.logical_xor(first_positive, second_positive).sum()),
    }
    if np.any(matched):
        difference = second[matched] - first[matched]
        result.update(
            maximum_absolute_difference_dex=float(np.max(np.abs(difference))),
            median_difference_dex=float(np.median(difference)),
        )
    return result


def _plot(path: Path, maps: dict[str, np.ndarray]) -> None:
    colors = {
        "hm12_external": "tab:blue",
        "hm12_native": "tab:orange",
        "hm12_plus_filtered_ism": "black",
    }
    labels = {
        "hm12_external": "external HM12 (Grackle)",
        "hm12_native": "Cloudy native HM12",
        "hm12_plus_filtered_ism": "external HM12 + filtered ISM",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), sharex=True)
    for line_index, (axis, line_name) in enumerate(zip(axes, LINE_NAMES)):
        for model_name, data in maps.items():
            values = data[:, line_index + 1]
            axis.plot(
                data[:, 0],
                np.where(values > -90.0, values, np.nan),
                marker="o",
                color=colors[model_name],
                label=labels[model_name],
            )
        axis.set_title(line_name)
        axis.set_xlabel(r"$\log_{10} T$ [K]")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel(
        r"$\log_{10}(\epsilon/n_{\rm H}^2)$ [erg s$^{-1}$ cm$^3$]"
    )
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        r"Sparse radiation-interface test: $n_H=1$ cm$^{-3}$, "
        r"$N_H=10^{20}$ cm$^{-2}$"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    grackle = project_root / "work/cloudy_cooling_tools_history/examples/grackle"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cloudy-exe",
        type=Path,
        default=Path("/Users/tianbaochen/cloudy/c17.02/source/cloudy.exe"),
    )
    parser.add_argument(
        "--work-dir", type=Path, default=grackle / "hm12_ism_validation"
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--force",
        action="store_true",
        help="delete only this validation directory before recreating it",
    )
    args = parser.parse_args()
    args.cloudy_exe = args.cloudy_exe.resolve()
    args.work_dir = args.work_dir.resolve()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.force and args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    cialoop = grackle.parents[1] / "CIAOLoop_lines"
    required = (
        args.cloudy_exe,
        cialoop,
        grackle / "HM12_UVB/z_0.0000e+00.out",
        grackle / "HM12_ISM_NH21/z_0.0000e+00.out",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    parameter_dir = args.work_dir / "parameters"
    parameter_dir.mkdir(exist_ok=True)
    map_paths = {}
    for model_name, radiation_command in MODELS.items():
        output_prefix = f"validation_{model_name}"
        output_dir = args.work_dir / f"{model_name}_output"
        parameter_path = parameter_dir / f"{model_name}.par"
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"refusing to mix old and new results in {output_dir}; use --force"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        parameter_path.write_text(
            _parameter_text(
                args.cloudy_exe, output_prefix, output_dir, radiation_command
            )
        )
        log_path = args.work_dir / f"{model_name}.log"
        with log_path.open("w") as log:
            subprocess.run(
                [str(cialoop), "-np", str(args.workers), str(parameter_path)],
                cwd=grackle,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        map_path = output_dir / f"{output_prefix}_run1.dat"
        if not map_path.exists():
            raise FileNotFoundError(map_path)
        map_paths[model_name] = map_path

    maps = {name: _load_map(path) for name, path in map_paths.items()}
    reference = maps["hm12_external"]
    for name, data in maps.items():
        if not np.array_equal(reference[:, 0], data[:, 0]):
            raise ValueError(f"temperature grids differ for {name}")

    comparisons = {}
    for model_name in ("hm12_native", "hm12_plus_filtered_ism"):
        comparisons[model_name + "_minus_hm12_external"] = {
            line_name: _line_comparison(
                reference[:, line_index + 1], maps[model_name][:, line_index + 1]
            )
            for line_index, line_name in enumerate(LINE_NAMES)
        }

    report = {
        "cloudy_executable": str(args.cloudy_exe),
        "cialoop_lines": str(cialoop),
        "grid": {
            "log10_nH_cm-3": [0.0],
            "log10_NH_cm-2": [20.0],
            "log10_temperature_K": reference[:, 0].tolist(),
        },
        "physics_held_fixed": {
            "abundances": "Cloudy defaults",
            "charge_transfer": "Cloudy default (enabled)",
            "H2_molecule": "disabled",
            "cosmic_rays": "not added",
            "grains": "not added",
            "reported_quantity": (
                "final-zone local line emissivity / nH^2, log10 "
                "[erg s^-1 cm^3]"
            ),
        },
        "map_files": {name: str(path) for name, path in map_paths.items()},
        "comparisons": comparisons,
    }
    report_path = args.work_dir / "validation_report.json"
    plot_path = args.work_dir / "three_line_sparse_validation.png"
    _plot(plot_path, maps)
    report["plot"] = str(plot_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
