#!/usr/bin/env python3
"""Build HM2012 plus the net transmitted spectrum of a physical ISM slab.

The foreground slab is illuminated only by Cloudy's ``table ISM`` field.  Its
shielded-face net transmitted continuum (direct transmitted plus outward
diffuse emission) is then added linearly to Cloudy's native HM2012 field.  CMB
is deliberately not included here; the line-grid parameter file adds it as a
separate Cloudy command so that the quick-extinguish and physical-slab grids
use the same CMB treatment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


RYDBERG_HZ = 3.2898419602508e15
Z0_TOKEN = "0.0000e+00"


def run_cloudy(cloudy_exe: Path, output_dir: Path, root: str, text: str) -> None:
    (output_dir / f"{root}.in").write_text(text.rstrip() + "\n")
    subprocess.run([str(cloudy_exe), "-r", root], cwd=output_dir, check=True)


def load_columns(path: Path, usecols: tuple[int, ...]) -> np.ndarray:
    data = np.loadtxt(path, comments="#", usecols=usecols)
    if data.ndim != 2 or np.any(np.diff(data[:, 0]) <= 0.0):
        raise ValueError(f"unexpected Cloudy continuum file: {path}")
    return data


def load_transmitted(path: Path) -> np.ndarray:
    rows = []
    for raw in path.read_text().splitlines():
        columns = raw.split()
        if len(columns) < 2:
            continue
        try:
            rows.append((float(columns[0]), float(columns[1])))
        except ValueError:
            continue
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"unexpected save transmitted continuum format: {path}")
    return data


def write_sed(path: Path, energy: np.ndarray, nu_f_nu: np.ndarray) -> None:
    with path.open("w") as handle:
        handle.write(
            "# Cloudy-native HM2012 plus net transmitted table ISM through "
            "nH=8 cm^-3, NH=1e21 cm^-2 slab with ISM grains\n"
        )
        for index, (photon_energy, intensity) in enumerate(zip(energy, nu_f_nu)):
            suffix = " nuFnu" if index == 0 else ""
            handle.write(f"{photon_energy:.8e} {intensity:.8e}{suffix}\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cloudy-exe",
        type=Path,
        default=Path(os.environ["CLOUDY_EXE"])
        if "CLOUDY_EXE" in os.environ
        else None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            root
            / "runtime/cloudy_sixline/examples/grackle"
            / "HM12_NATIVE_PHYSICAL_ISM_NH8_NH21_GRAINS"
        ),
    )
    args = parser.parse_args()
    if args.cloudy_exe is None:
        parser.error("--cloudy-exe is required unless CLOUDY_EXE is set")
    cloudy_exe = args.cloudy_exe.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not cloudy_exe.is_file():
        raise FileNotFoundError(cloudy_exe)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_cloudy(
        cloudy_exe,
        output_dir,
        "export_hm12_native",
        """title export Cloudy-native HM2012
table HM12 redshift 0
hden -10
constant temperature 1e4 K
stop zone 1
set dr 0
save incident continuum "export_hm12_native.inc"
""",
    )
    run_cloudy(
        cloudy_exe,
        output_dir,
        "physical_ism_slab_nh8_nh21_grains",
        """title physical table ISM foreground slab nH=8 cm-3 NH=1e21 cm-2
table ISM
abundances ISM no grains
grains ISM
hden 0.903089987
stop column density 21
stop temperature off
iterate to convergence
save continuum "physical_ism_slab_nh8_nh21_grains.con" last
save transmitted continuum "physical_ism_slab_nh8_nh21_grains.trn"
save overview "physical_ism_slab_nh8_nh21_grains.ovr" last
""",
    )

    hm12_path = output_dir / "export_hm12_native.inc"
    slab_path = output_dir / "physical_ism_slab_nh8_nh21_grains.con"
    transmitted_path = (
        output_dir / "physical_ism_slab_nh8_nh21_grains.trn"
    )
    hm12_data = load_columns(hm12_path, (0, 1))
    slab_data = load_columns(slab_path, (0, 1, 2, 3, 4))
    transmitted_data = load_transmitted(transmitted_path)
    energy = hm12_data[:, 0]
    if not np.array_equal(energy, slab_data[:, 0]):
        raise ValueError("HM2012 and physical-slab energy meshes differ")

    hm12 = hm12_data[:, 1]
    incident_ism = slab_data[:, 1]
    direct_transmitted_ism = slab_data[:, 2]
    outward_emitted_ism = slab_data[:, 3]
    net_transmitted_ism = slab_data[:, 4]
    # ``save continuum`` prints only four significant digits.  Its separately
    # rounded direct and diffuse columns therefore reproduce the printed net
    # column to about 1e-3, not machine precision.
    if not np.allclose(
        net_transmitted_ism,
        direct_transmitted_ism + outward_emitted_ism,
        rtol=1.1e-3,
        atol=0.0,
    ):
        raise ValueError("Cloudy net-transmitted column is not direct + diffuse")
    if not np.allclose(energy, transmitted_data[:, 0], rtol=1.0e-5, atol=0.0):
        raise ValueError("save continuum and save transmitted meshes differ")
    if not np.array_equal(net_transmitted_ism, transmitted_data[:, 1]):
        raise ValueError("save continuum net column and .trn intensity differ")
    combined = hm12 + net_transmitted_ism

    positive_combined = combined > 0.0
    nu_f_nu_at_one_ryd = np.exp(
        np.interp(
            0.0,
            np.log(energy[positive_combined]),
            np.log(combined[positive_combined]),
        )
    )
    fnu_command = (
        f"f(nu) = {np.log10(nu_f_nu_at_one_ryd / RYDBERG_HZ):.10f} "
        "at 1 Ryd"
    )
    sed_path = output_dir / f"z_{Z0_TOKEN}.sed"
    write_sed(sed_path, energy, combined)
    (output_dir / f"z_{Z0_TOKEN}.out").write_text(
        f'table SED "{output_dir.name}/{sed_path.name}"\n{fnu_command}\n'
    )

    run_cloudy(
        cloudy_exe,
        output_dir,
        "verify_combined_roundtrip",
        f"""title verify HM2012 plus physical-slab transmitted ISM
table SED "{sed_path.name}"
{fnu_command}
hden -10
constant temperature 1e4 K
stop zone 1
set dr 0
save incident continuum "verify_combined_roundtrip.inc"
""",
    )
    roundtrip = load_columns(
        output_dir / "verify_combined_roundtrip.inc", (0, 1)
    )
    if not np.array_equal(energy, roundtrip[:, 0]):
        raise ValueError("round-trip energy mesh differs")
    positive = combined > 0.0
    ratio = roundtrip[positive, 1] / combined[positive]
    report = {
        "definition": (
            "Cloudy-native HM2012 plus shielded-face net transmitted table ISM "
            "from a converged nH=8 cm^-3, NH=1e21 cm^-2 slab with ISM grains"
        ),
        "physical_ism_output": (
            "save continuum column 5 = direct transmitted + outward diffuse emission"
        ),
        "cmb_included": False,
        "cloudy_executable": str(cloudy_exe),
        "energy_points": int(energy.size),
        "fnu_normalization_command": fnu_command,
        "roundtrip": {
            "maximum_relative_error": float(np.max(np.abs(ratio - 1.0))),
            "maximum_absolute_error_dex": float(np.max(np.abs(np.log10(ratio)))),
        },
        "outputs": {
            "combined_sed": str(sed_path),
            "cialoop_init": str(output_dir / f"z_{Z0_TOKEN}.out"),
            "hm12_incident": str(hm12_path),
            "physical_slab_continuum": str(slab_path),
            "physical_slab_transmitted": str(
                transmitted_path
            ),
            "roundtrip_incident": str(
                output_dir / "verify_combined_roundtrip.inc"
            ),
        },
    }
    np.savez_compressed(
        output_dir / "components.npz",
        energy_Ryd=energy,
        hm2012=hm12,
        ism_incident=incident_ism,
        ism_direct_transmitted=direct_transmitted_ism,
        ism_outward_emitted=outward_emitted_ism,
        ism_net_transmitted=net_transmitted_ism,
        combined=combined,
    )
    (output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
