#!/usr/bin/env python3
"""Build Cloudy-native HM12 plus selectively extinguished Black/ISM and CMB.

The production incident field is

    J_total = J_Cloudy_native_HM12
            + Extinguish(NH=1e21, leak=0)[J_ISM]
            + J_CMB(z=0).

Both components are exported with Cloudy 17.02 ``save incident continuum``.
The filtered ISM component and native HM12 are added on Cloudy's common energy
mesh in linear nu*4*pi*J_nu units.  No Grackle HM12 ``.out`` file is read.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


Z0_TOKEN = "0.0000e+00"
RYDBERG_HZ = 3.2898419602508e15


def run_cloudy(cloudy_exe: Path, output_dir: Path, root: str, text: str) -> Path:
    input_path = output_dir / f"{root}.in"
    input_path.write_text(text.rstrip() + "\n")
    subprocess.run([str(cloudy_exe), "-r", root], cwd=output_dir, check=True)
    incident = output_dir / f"{root}.inc"
    if not incident.exists():
        raise FileNotFoundError(incident)
    return incident


def export_input(title: str, continuum_commands: str, save_name: str) -> str:
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


def write_sed(
    path: Path,
    energy: np.ndarray,
    nu_f_nu: np.ndarray,
    *,
    include_cmb: bool,
) -> None:
    with path.open("w") as handle:
        handle.write(
            "# Cloudy table HM12 redshift 0 plus table ISM filtered with "
            "extinguish column=21 leak=0"
            + (" plus CMB redshift 0" if include_cmb else "")
            + "\n"
        )
        for index, (photon_energy, intensity) in enumerate(zip(energy, nu_f_nu)):
            suffix = " nuFnu" if index == 0 else ""
            handle.write(f"{photon_energy:.8e} {intensity:.8e}{suffix}\n")


def nearest_values(energy: np.ndarray, arrays: dict[str, np.ndarray]) -> dict:
    result = {}
    for target in (0.1, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0):
        index = int(np.argmin(np.abs(energy - target)))
        result[str(target)] = {
            "actual_energy_Ryd": float(energy[index]),
            **{name: float(values[index]) for name, values in arrays.items()},
        }
    return result


def plot_components(
    path: Path,
    energy: np.ndarray,
    hm12_native: np.ndarray,
    ism_plain: np.ndarray,
    ism_filtered: np.ndarray,
    cmb: np.ndarray | None,
    combined: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ax.loglog(energy, np.where(hm12_native > 0, hm12_native, np.nan),
              label="Cloudy table HM12 redshift 0", linewidth=1.8)
    ax.loglog(energy, np.where(ism_plain > 0, ism_plain, np.nan),
              label="Cloudy table ISM", linewidth=1.6)
    ax.loglog(energy, np.where(ism_filtered > 0, ism_filtered, np.nan),
              label=r"ISM + extinguish $N_H=10^{21}$, leak=0",
              linestyle="--", linewidth=1.8)
    if cmb is not None:
        ax.loglog(energy, np.where(cmb > 0, cmb, np.nan),
                  label=r"CMB ($z=0$)", linestyle=":", linewidth=1.8)
    ax.loglog(energy, np.where(combined > 0, combined, np.nan),
              label=("combined incident spectrum" if cmb is not None
                     else "native HM12 + filtered ISM"),
              color="black", linewidth=2.0)
    ax.axvline(1.0, color="0.4", linestyle=":", linewidth=1.0)
    ax.axvline(4.0, color="0.55", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Photon energy [Ryd]")
    ax.set_ylabel(r"$\nu\,4\pi J_\nu$ [erg cm$^{-2}$ s$^{-1}$]")
    ax.set_title(
        "Cloudy incident radiation field"
        + (" including the CMB" if cmb is not None else "")
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_paper_spectrum(
    png_path: Path,
    pdf_path: Path,
    energy: np.ndarray,
    hm12_native: np.ndarray,
    ism_filtered: np.ndarray,
    cmb: np.ndarray,
    combined: np.ndarray,
) -> None:
    """Plot only the three adopted components and their final sum."""
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.loglog(
        energy, np.where(hm12_native > 0, hm12_native, np.nan),
        label=r"HM2012 ($z=0$)", linewidth=1.8,
    )
    ax.loglog(
        energy, np.where(ism_filtered > 0, ism_filtered, np.nan),
        label=r"attenuated ISM ($N_{\rm H,fg}=10^{21}\ {\rm cm}^{-2}$)",
        linestyle="--", linewidth=1.8,
    )
    ax.loglog(
        energy, np.where(combined > 0, combined, np.nan),
        label="combined incident spectrum", color="black", linewidth=2.2,
    )
    # Draw the CMB last so that it remains visible where it dominates the
    # combined spectrum at low photon energy.
    ax.loglog(
        energy, np.where(cmb > 0, cmb, np.nan),
        label=r"CMB ($z=0$)", linestyle=":", linewidth=2.0, zorder=5,
    )
    ax.set_xlabel("Photon energy [Ryd]")
    ax.set_ylabel(r"$\nu\,4\pi J_\nu$ [erg cm$^{-2}$ s$^{-1}$]")
    ax.set_ylim(1.0e-28, 1.0e-1)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.16)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
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
        "--output-dir",
        type=Path,
        default=grackle / "HM12_NATIVE_ISM_NH21",
    )
    parser.add_argument("--foreground-log-nh", type=float, default=21.0)
    parser.add_argument("--leak", type=float, default=0.0)
    parser.add_argument("--include-cmb", action="store_true")
    args = parser.parse_args()
    cloudy_exe = args.cloudy_exe.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cloudy_exe.exists():
        raise FileNotFoundError(cloudy_exe)

    exports = {
        "hm12_native": run_cloudy(
            cloudy_exe,
            output_dir,
            "export_hm12_native",
            export_input(
                "export Cloudy native table HM12 redshift 0",
                "table HM12 redshift 0",
                "export_hm12_native.inc",
            ),
        ),
        "ism_plain": run_cloudy(
            cloudy_exe,
            output_dir,
            "export_ism_plain",
            export_input("export Cloudy table ISM", "table ISM", "export_ism_plain.inc"),
        ),
        "ism_filtered": run_cloudy(
            cloudy_exe,
            output_dir,
            "export_ism_filtered",
            export_input(
                "export selectively extinguished Cloudy table ISM",
                "table ISM\n"
                f"extinguish column = {args.foreground_log_nh:g} "
                f"leak = {args.leak:g}",
                "export_ism_filtered.inc",
            ),
        ),
    }
    if args.include_cmb:
        exports["cmb"] = run_cloudy(
            cloudy_exe,
            output_dir,
            "export_cmb_z0",
            export_input(
                "export Cloudy CMB at redshift 0",
                "CMB redshift 0",
                "export_cmb_z0.inc",
            ),
        )

    loaded = {name: load_incident(path) for name, path in exports.items()}
    energy = loaded["hm12_native"][:, 0]
    for name, data in loaded.items():
        if not np.array_equal(energy, data[:, 0]):
            raise ValueError(f"Cloudy energy mesh differs for {name}")

    hm12_native = loaded["hm12_native"][:, 1]
    ism_plain = loaded["ism_plain"][:, 1]
    ism_filtered = loaded["ism_filtered"][:, 1]
    cmb = loaded["cmb"][:, 1] if args.include_cmb else None
    combined = hm12_native + ism_filtered
    if cmb is not None:
        combined = combined + cmb

    positive_hm12 = hm12_native > 0.0
    nu_f_nu_at_one_ryd = np.exp(
        np.interp(
            0.0,
            np.log(energy[positive_hm12]),
            np.log(hm12_native[positive_hm12]),
        )
    )
    fnu_at_one_ryd = nu_f_nu_at_one_ryd / RYDBERG_HZ
    fnu_command = f"f(nu) = {np.log10(fnu_at_one_ryd):.10f} at 1 Ryd"

    sed_path = output_dir / f"z_{Z0_TOKEN}.sed"
    write_sed(sed_path, energy, combined, include_cmb=args.include_cmb)
    relative_dir = output_dir.relative_to(grackle)
    init_path = output_dir / f"z_{Z0_TOKEN}.out"
    init_path.write_text(
        f'table SED "{relative_dir}/{sed_path.name}"\n{fnu_command}\n'
    )

    roundtrip_path = run_cloudy(
        cloudy_exe,
        output_dir,
        "verify_combined_roundtrip",
        export_input(
            "verify native HM12 plus filtered ISM"
            + (" plus CMB" if args.include_cmb else "")
            + " custom SED",
            f'table SED "{sed_path.name}"\n{fnu_command}',
            "verify_combined_roundtrip.inc",
        ),
    )
    roundtrip = load_incident(roundtrip_path)
    positive = combined > 0.0
    relative_error = np.abs(roundtrip[positive, 1] / combined[positive] - 1.0)
    dex_error = np.abs(np.log10(roundtrip[positive, 1] / combined[positive]))

    plot_path = output_dir / (
        "native_hm12_filtered_ism_cmb_components.png"
        if args.include_cmb else "native_hm12_filtered_ism_components.png"
    )
    plot_components(
        plot_path, energy, hm12_native, ism_plain, ism_filtered, cmb, combined
    )
    paper_plot_png = None
    paper_plot_pdf = None
    if cmb is not None:
        paper_plot_png = output_dir / "cloudy_combined_incident_spectrum.png"
        paper_plot_pdf = output_dir / "cloudy_combined_incident_spectrum.pdf"
        plot_paper_spectrum(
            paper_plot_png,
            paper_plot_pdf,
            energy,
            hm12_native,
            ism_filtered,
            cmb,
            combined,
        )

    report = {
        "definition": (
            "Cloudy table HM12 redshift 0 + table ISM after selective "
            f"extinguish column={args.foreground_log_nh:g}, leak={args.leak:g}"
            + (" + CMB redshift 0" if args.include_cmb else "")
        ),
        "cloudy_executable": str(cloudy_exe),
        "external_grackle_hm12_used": False,
        "energy_points": int(energy.size),
        "energy_mesh_identical_for_all_exports": True,
        "fnu_normalization_command": fnu_command,
        "selected_energy_values_nuFnu": nearest_values(
            energy,
            {
                "hm12_native": hm12_native,
                "ism_plain": ism_plain,
                "ism_filtered": ism_filtered,
                **({"cmb": cmb} if cmb is not None else {}),
                "combined": combined,
                "roundtrip": roundtrip[:, 1],
            },
        ),
        "roundtrip": {
            "maximum_relative_error": float(relative_error.max()),
            "p99_relative_error": float(np.quantile(relative_error, 0.99)),
            "maximum_absolute_error_dex": float(dex_error.max()),
            "p99_absolute_error_dex": float(np.quantile(dex_error, 0.99)),
        },
        "outputs": {
            "combined_sed": str(sed_path),
            "cialoop_init": str(init_path),
            "exports": {name: str(path) for name, path in exports.items()},
            "roundtrip_incident": str(roundtrip_path),
            "plot": str(plot_path),
            **(
                {
                    "paper_plot_png": str(paper_plot_png),
                    "paper_plot_pdf": str(paper_plot_pdf),
                }
                if paper_plot_png is not None and paper_plot_pdf is not None
                else {}
            ),
        },
    }
    report_path = output_dir / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
