#!/usr/bin/env python3
"""Build the seven incident SEDs used by the Cloudy Jeans tables.

For each HM2012 foreground attenuation column in
``log10(N_H / cm^-2) = 18, 18.5, ..., 21``, this script asks Cloudy 17.02 to
export ``extinguish(N_H)[table HM12 redshift 0]`` and adds it in linear
intensity units to a separately exported
``extinguish(column=21, leak=0)[table ISM]``.

The CMB is deliberately not baked into these custom SEDs. The line-table
parameter file adds ``CMB redshift 0`` as a separate Cloudy continuum. Every
custom SED is read back by Cloudy and compared with its target before the
manifest is accepted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


RYDBERG_HZ = 3.2898419602508e15
HM12_ATTENUATION_LOG_NH = (18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0)
ISM_ATTENUATION_LOG_NH = 21.0
NORMALIZATION_ENERGY_RYD = 0.5
# Cloudy's table-SED interpolation and text continuum export introduce a
# sub-per-mille round-trip difference near sharp edges.  This gate still
# rejects any normalization or unit mistake by many orders of magnitude.
ROUNDTRIP_MAX_ERROR_DEX = 1.0e-3
SED_ZERO_FLOOR_BELOW_MIN_DEX = 6.0
ROUNDTRIP_RELEVANT_DYNAMIC_RANGE_DEX = 30.0


def run_cloudy(cloudy_exe: Path, output_dir: Path, root: str, text: str) -> Path:
    """Run one continuum-only Cloudy input and return its incident export."""
    input_path = output_dir / f"{root}.in"
    input_path.write_text(text.rstrip() + "\n")
    subprocess.run([str(cloudy_exe), "-r", root], cwd=output_dir, check=True)
    incident = output_dir / f"{root}.inc"
    if not incident.is_file():
        raise FileNotFoundError(incident)
    return incident


def export_input(title: str, continuum_commands: str, save_name: str) -> str:
    """Return a minimal Cloudy model used only to export its incident SED."""
    return f"""title {title}
{continuum_commands.rstrip()}
hden -10
constant temperature 1e4 K
stop zone 1
set dr 0
save incident continuum "{save_name}"
"""


def load_incident(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"unexpected save incident continuum format: {path}")
    if np.any(np.diff(data[:, 0]) <= 0.0):
        raise ValueError(f"non-increasing energy mesh: {path}")
    return data


def _number_token(value: float) -> str:
    return f"{value:g}"


def _root_token(value: float) -> str:
    return _number_token(value).replace(".", "p")


def _log_interpolate_positive(
    energy: np.ndarray,
    values: np.ndarray,
    target_energy: float,
) -> float:
    positive = (energy > 0.0) & (values > 0.0)
    if np.count_nonzero(positive) < 2:
        raise ValueError("SED has fewer than two positive samples")
    positive_energy = energy[positive]
    if not positive_energy[0] <= target_energy <= positive_energy[-1]:
        raise ValueError(f"normalization energy is outside SED: {target_energy}")
    return float(
        np.exp(
            np.interp(
                np.log(target_energy),
                np.log(positive_energy),
                np.log(values[positive]),
            )
        )
    )


def normalization_command(
    energy: np.ndarray,
    nu_f_nu: np.ndarray,
    *,
    anchor_ryd: float = NORMALIZATION_ENERGY_RYD,
) -> str:
    """Return the Cloudy ``f(nu)`` command matching the target SED."""
    anchor_nu_f_nu = _log_interpolate_positive(energy, nu_f_nu, anchor_ryd)
    f_nu = anchor_nu_f_nu / (anchor_ryd * RYDBERG_HZ)
    if not np.isfinite(f_nu) or f_nu <= 0.0:
        raise ValueError("computed f(nu) normalization is not positive and finite")
    return f"f(nu) = {np.log10(f_nu):.10f} at {anchor_ryd:g} Ryd"


def write_sed(
    path: Path,
    energy: np.ndarray,
    nu_f_nu: np.ndarray,
    *,
    hm12_log_nh: float,
) -> None:
    with path.open("w") as handle:
        handle.write(
            "# Cloudy table HM12 redshift 0 after extinguish column="
            f"{hm12_log_nh:g} leak=0 plus table ISM after extinguish "
            f"column={ISM_ATTENUATION_LOG_NH:g} leak=0\n"
        )
        for index, (photon_energy, intensity) in enumerate(zip(energy, nu_f_nu)):
            suffix = " nuFnu" if index == 0 else ""
            handle.write(f"{photon_energy:.8e} {intensity:.8e}{suffix}\n")


def _roundtrip_errors(target: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    positive = target > 0.0
    if not np.any(positive):
        raise ValueError("target SED contains no positive values")
    if np.any(actual[positive] <= 0.0):
        raise ValueError("Cloudy round-trip erased a positive target SED value")
    ratio = actual[positive] / target[positive]
    dex = np.abs(np.log10(ratio))
    relative = np.abs(ratio - 1.0)
    relevant_target = target[positive]
    relevant = relevant_target >= (
        relevant_target.max() * 10.0 ** (-ROUNDTRIP_RELEVANT_DYNAMIC_RANGE_DEX)
    )
    return {
        "maximum_relative_error": float(relative.max()),
        "p99_relative_error": float(np.quantile(relative, 0.99)),
        "maximum_absolute_error_dex": float(dex.max()),
        "p99_absolute_error_dex": float(np.quantile(dex, 0.99)),
        "relevant_dynamic_range_dex": ROUNDTRIP_RELEVANT_DYNAMIC_RANGE_DEX,
        "relevant_points": int(np.count_nonzero(relevant)),
        "maximum_relevant_absolute_error_dex": float(dex[relevant].max()),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    runtime_grackle = project_root / "runtime/cloudy_sixline/examples/grackle"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cloudy-exe",
        type=Path,
        default=Path(os.environ["CLOUDY_EXE"]) if "CLOUDY_EXE" in os.environ else None,
        help="Cloudy 17.02 executable (or set CLOUDY_EXE)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=runtime_grackle / "HM12_ATTENUATION_ISM_NH21",
    )
    parser.add_argument(
        "--hm12-log-nh",
        type=float,
        nargs="+",
        default=HM12_ATTENUATION_LOG_NH,
        help="HM2012 extinguish-column grid in log10(cm^-2)",
    )
    parser.add_argument("--ism-log-nh", type=float, default=ISM_ATTENUATION_LOG_NH)
    parser.add_argument("--leak", type=float, default=0.0)
    parser.add_argument(
        "--normalization-energy-ryd",
        type=float,
        default=NORMALIZATION_ENERGY_RYD,
    )
    args = parser.parse_args()
    if args.cloudy_exe is None:
        parser.error("--cloudy-exe is required unless CLOUDY_EXE is set")
    if args.normalization_energy_ryd <= 0.0:
        parser.error("--normalization-energy-ryd must be positive")
    hm12_log_nh = tuple(float(value) for value in args.hm12_log_nh)
    if len(hm12_log_nh) < 2 or any(
        right <= left for left, right in zip(hm12_log_nh, hm12_log_nh[1:])
    ):
        parser.error("--hm12-log-nh must contain at least two increasing values")

    cloudy_exe = args.cloudy_exe.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not cloudy_exe.is_file():
        raise FileNotFoundError(cloudy_exe)

    ism_path = run_cloudy(
        cloudy_exe,
        output_dir,
        "export_ism_extinguished_nh21",
        export_input(
            "export quick-extinguished Cloudy table ISM",
            "table ISM\n"
            f"extinguish column = {args.ism_log_nh:g} leak = {args.leak:g}",
            "export_ism_extinguished_nh21.inc",
        ),
    )
    ism_data = load_incident(ism_path)
    energy = ism_data[:, 0]
    ism_extinguished = ism_data[:, 1]

    entries: list[dict[str, object]] = []
    for log_nh in hm12_log_nh:
        label = _number_token(log_nh)
        root_label = _root_token(log_nh)
        hm12_path = run_cloudy(
            cloudy_exe,
            output_dir,
            f"export_hm12_extinguished_nh{root_label}",
            export_input(
                f"export Cloudy HM12 extinguished by log NH={label}",
                "table HM12 redshift 0\n"
                f"extinguish column = {label} leak = {args.leak:g}",
                f"export_hm12_extinguished_nh{root_label}.inc",
            ),
        )
        hm12_data = load_incident(hm12_path)
        if not np.array_equal(energy, hm12_data[:, 0]):
            raise ValueError(f"Cloudy energy mesh differs for HM12 log NH={label}")

        combined = hm12_data[:, 1] + ism_extinguished
        if np.any(combined < 0.0):
            raise ValueError(f"combined SED is negative for HM12 log NH={label}")
        positive_combined = combined[combined > 0.0]
        if positive_combined.size == 0:
            raise ValueError(f"combined SED is identically zero for log NH={label}")
        zero_floor = float(
            positive_combined.min() * 10.0 ** (-SED_ZERO_FLOOR_BELOW_MIN_DEX)
        )
        serialized_sed = np.where(combined > 0.0, combined, zero_floor)
        fnu_command = normalization_command(
            energy,
            serialized_sed,
            anchor_ryd=args.normalization_energy_ryd,
        )
        sed_name = f"logNH{label}.sed"
        init_name = f"logNH{label}.out"
        sed_path = output_dir / sed_name
        init_path = output_dir / init_name
        write_sed(sed_path, energy, serialized_sed, hm12_log_nh=log_nh)
        init_path.write_text(
            f'table SED "{output_dir.name}/{sed_name}"\n{fnu_command}\n'
        )

        roundtrip_path = run_cloudy(
            cloudy_exe,
            output_dir,
            f"verify_combined_nh{root_label}",
            export_input(
                f"verify combined SED for HM12 log NH={label}",
                f'table SED "{sed_name}"\n{fnu_command}',
                f"verify_combined_nh{root_label}.inc",
            ),
        )
        roundtrip = load_incident(roundtrip_path)
        if not np.array_equal(energy, roundtrip[:, 0]):
            raise ValueError(f"round-trip energy mesh differs for log NH={label}")
        errors = _roundtrip_errors(serialized_sed, roundtrip[:, 1])
        if errors["maximum_relevant_absolute_error_dex"] > ROUNDTRIP_MAX_ERROR_DEX:
            raise RuntimeError(
                f"round-trip error exceeds {ROUNDTRIP_MAX_ERROR_DEX:g} dex "
                f"for HM12 log NH={label}: {errors}"
            )
        entries.append(
            {
                "hm12_log_NH_attenuation": log_nh,
                "combined_sed": str(sed_path),
                "cialoop_init": str(init_path),
                "hm12_export": str(hm12_path),
                "fnu_normalization_command": fnu_command,
                "roundtrip_incident": str(roundtrip_path),
                "roundtrip": errors,
                "physical_zero_points": int(np.count_nonzero(combined == 0.0)),
                "serialized_zero_floor_nuFnu": zero_floor,
            }
        )

    report = {
        "schema_version": 1,
        "definition": (
            "quick-extinguished Cloudy-native HM2012 plus separately "
            "quick-extinguished Cloudy table ISM; CMB is added later by the "
            "line-table parameter file"
        ),
        "cloudy_executable": str(cloudy_exe),
        "external_grackle_hm12_used": False,
        "hm12_log_NH_attenuation": list(hm12_log_nh),
        "ism_log_NH_attenuation": float(args.ism_log_nh),
        "extinguish_leak": float(args.leak),
        "normalization_energy_Ryd": float(args.normalization_energy_ryd),
        "normalization_note": (
            "Each f(nu) value is calculated from that combined SED at the "
            "stated energy; it is not an arbitrary non-zero scale."
        ),
        "table_SED_zero_policy": (
            "Cloudy table SED requires positive flux. Exact zeros are serialized "
            f"at {SED_ZERO_FLOOR_BELOW_MIN_DEX:g} dex below that SED's smallest "
            "positive value; physical component arrays remain zero."
        ),
        "roundtrip_maximum_allowed_error_dex": ROUNDTRIP_MAX_ERROR_DEX,
        "energy_points": int(energy.size),
        "energy_mesh_identical_for_all_exports": True,
        "ism_extinguished_export": str(ism_path),
        "entries": entries,
    }
    report_path = output_dir / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
