#!/usr/bin/env python3
"""Compare Cloudy's quick ISM ``extinguish`` screen with physical gas slabs.

The physical foreground calculations use the built-in Black (1987) ``table
ISM`` field, a plane-parallel constant-density slab, thermal equilibrium, and
stop at a fixed total hydrogen column.  The diagnostic deliberately does not
add grains or cosmic rays, so that the first comparison isolates Cloudy's gas
radiative-transfer calculation from those additional assumptions.
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


DEFAULT_DENSITIES = (0.1, 1.0, 10.0)
EV_PER_RYD = 13.605693122994
RADIATION_X_MIN_RYD = 8.0 / EV_PER_RYD
RADIATION_X_MAX_RYD = 1.0e3
RADIATION_Y_DYNAMIC_RANGE_DEX = 8.0


def density_token(density: float) -> str:
    return f"{density:g}".replace(".", "p").replace("-", "m")


def run_cloudy(
    cloudy_exe: Path,
    output_dir: Path,
    root: str,
    text: str,
    expected_outputs: tuple[str, ...],
) -> None:
    output_path = output_dir / f"{root}.out"
    if output_path.exists() and all(
        (output_dir / name).exists() and (output_dir / name).stat().st_size > 0
        for name in expected_outputs
    ):
        if "Cloudy ends:" in output_path.read_text(errors="replace"):
            print(f"Reusing completed Cloudy model: {root}")
            return
    (output_dir / f"{root}.in").write_text(text.rstrip() + "\n")
    subprocess.run([str(cloudy_exe), "-r", root], cwd=output_dir, check=True)


def quick_extinguish_input(log_nh: float) -> str:
    return f"""title quick extinguish comparison for table ISM
table ISM
extinguish column = {log_nh:g} leak = 0
hden -10
constant temperature 1e4 K
stop zone 1
set dr 0
save incident continuum "ism_extinguish_nh{log_nh:g}.inc"
"""


def slab_input(
    density: float, log_nh: float, root: str, *, include_ism_grains: bool
) -> str:
    composition = (
        "abundances ISM no grains\n"
        "grains ISM\n"
        if include_ism_grains
        else ""
    )
    return f"""title physical table ISM foreground slab nH={density:g} cm-3
table ISM
{composition}hden {np.log10(density):.12g}
stop column density {log_nh:g}
stop temperature off
iterate to convergence
set WeakHeatCool -20
save continuum "{root}.con" last
save transmitted continuum "{root}.trn"
save overview "{root}.ovr" last
"""


def load_numeric(path: Path, usecols: tuple[int, ...]) -> np.ndarray:
    data = np.loadtxt(path, comments="#", usecols=usecols)
    if data.ndim != 2:
        raise ValueError(f"unexpected numeric data in {path}")
    return data


def positive(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.0, values, np.nan)


def interp_log_energy(
    source_energy: np.ndarray, source_values: np.ndarray, target_energy: np.ndarray
) -> np.ndarray:
    """Log-energy interpolation in linear intensity; zeros remain zero."""
    return np.interp(
        np.log(target_energy),
        np.log(source_energy),
        source_values,
        left=0.0,
        right=0.0,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cloudy-exe",
        type=Path,
        default=Path("/Users/tianbaochen/cloudy/c17.02/source/cloudy.exe"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            project_root
            / "work/cloudy_cooling_tools_history/examples/grackle"
            / "ISM_PHYSICAL_SLAB_NH21"
        ),
    )
    parser.add_argument("--log-nh", type=float, default=21.0)
    parser.add_argument(
        "--densities",
        type=float,
        nargs="+",
        default=DEFAULT_DENSITIES,
        metavar="NH_CM3",
    )
    args = parser.parse_args()

    cloudy_exe = args.cloudy_exe.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not cloudy_exe.exists():
        raise FileNotFoundError(cloudy_exe)

    quick_root = "ism_extinguish_nh21"
    run_cloudy(
        cloudy_exe,
        output_dir,
        quick_root,
        quick_extinguish_input(args.log_nh),
        (f"ism_extinguish_nh{args.log_nh:g}.inc",),
    )
    quick_path = output_dir / f"ism_extinguish_nh{args.log_nh:g}.inc"
    quick = load_numeric(quick_path, (0, 1))
    energy = quick[:, 0]
    quick_values = quick[:, 1]

    variants: dict[str, dict[str, dict[str, np.ndarray | float | str]]] = {}
    for variant_name, include_ism_grains in (
        ("gas_only", False),
        ("ism_grains", True),
    ):
        slabs: dict[str, dict[str, np.ndarray | float | str]] = {}
        suffix = "_grains" if include_ism_grains else ""
        for density in args.densities:
            token = density_token(density)
            root = f"ism_slab_nh{token}_NH{args.log_nh:g}{suffix}"
            expected = (f"{root}.con", f"{root}.trn", f"{root}.ovr")
            run_cloudy(
                cloudy_exe,
                output_dir,
                root,
                slab_input(
                    density,
                    args.log_nh,
                    root,
                    include_ism_grains=include_ism_grains,
                ),
                expected,
            )
            continuum_path = output_dir / f"{root}.con"
            transmitted_path = output_dir / f"{root}.trn"
            overview_path = output_dir / f"{root}.ovr"
            continuum = load_numeric(continuum_path, (0, 1, 2, 3, 4))
            # save transmitted continuum is intentionally Cloudy-readable and
            # has header metadata; save continuum column 5 is the same
            # observable net transmitted field in an easy-to-plot form.
            slab_energy = continuum[:, 0]
            slabs[token] = {
                "density_cm-3": density,
                "root": root,
                "direct": interp_log_energy(slab_energy, continuum[:, 2], energy),
                "outward_diffuse": interp_log_energy(
                    slab_energy, continuum[:, 3], energy
                ),
                "net": interp_log_energy(slab_energy, continuum[:, 4], energy),
                "continuum_file": str(continuum_path),
                "transmitted_file": str(transmitted_path),
                "overview_file": str(overview_path),
            }
        variants[variant_name] = slabs

    colors = {"0p1": "#0072B2", "1": "#D55E00", "10": "#009E73"}
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 8.0), sharex=True)
    for axis in axes:
        axis.loglog(
            energy,
            positive(quick_values),
            color="black",
            linestyle="--",
            linewidth=2.2,
            label=r"quick: extinguish column=21, leak=0",
            zorder=10,
        )
        axis.axvline(1.0, color="0.55", linestyle=":", linewidth=1.0)
        axis.grid(True, which="both", alpha=0.17)
        axis.set_ylabel(r"$\nu\,4\pi J_\nu$ [erg cm$^{-2}$ s$^{-1}$]")

    for variant_name, slabs in variants.items():
        has_grains = variant_name == "ism_grains"
        linestyle = ":" if has_grains else "-"
        variant_label = "ISM grains" if has_grains else "no grains"
        for token, data in slabs.items():
            density = float(data["density_cm-3"])
            color = colors.get(token)
            common_label = (
                rf"$n_{{\rm H}}={density:g}$ cm$^{{-3}}$, {variant_label}"
            )
            axes[0].loglog(
                energy,
                positive(np.asarray(data["direct"])),
                color=color,
                linestyle=linestyle,
                linewidth=2.0 if has_grains else 1.5,
                label=common_label,
            )
            axes[1].loglog(
                energy,
                positive(np.asarray(data["net"])),
                color=color,
                linestyle=linestyle,
                linewidth=2.0 if has_grains else 1.5,
                label=common_label,
            )

    axes[0].set_title(
        r"Direct transmitted ISM: fixed $N_{\rm H}=10^{21}$ cm$^{-2}$"
    )
    axes[1].set_title("Net transmitted ISM: direct + foreground emission")
    axes[1].set_xlabel("Photon energy [Ryd]")
    for axis in axes:
        axis.set_xlim(RADIATION_X_MIN_RYD, RADIATION_X_MAX_RYD)
        axis.legend(frameon=False, fontsize=8.5, loc="best")

    all_positive = [quick_values[quick_values > 0.0]]
    for slabs in variants.values():
        for data in slabs.values():
            for name in ("direct", "net"):
                values = np.asarray(data[name])
                all_positive.append(values[values > 0.0])
    maximum = max(float(values.max()) for values in all_positive if values.size)
    y_top = 10.0 ** np.ceil(np.log10(maximum))
    for axis in axes:
        axis.set_ylim(y_top / 10.0**RADIATION_Y_DYNAMIC_RANGE_DEX, y_top)

    fig.tight_layout()
    png_path = output_dir / "ism_extinguish_vs_physical_slabs_with_grains_NH21.png"
    pdf_path = output_dir / "ism_extinguish_vs_physical_slabs_with_grains_NH21.pdf"
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    npz_path = output_dir / "ism_extinguish_vs_physical_slabs_with_grains_NH21.npz"
    arrays: dict[str, np.ndarray] = {
        "energy_Ryd": energy,
        "extinguish_quick": quick_values,
    }
    for variant_name, slabs in variants.items():
        for token, data in slabs.items():
            prefix = f"{variant_name}_slab_nh_{token}"
            arrays[f"{prefix}_direct"] = np.asarray(data["direct"])
            arrays[f"{prefix}_outward_diffuse"] = np.asarray(
                data["outward_diffuse"]
            )
            arrays[f"{prefix}_net"] = np.asarray(data["net"])
    np.savez_compressed(npz_path, **arrays)

    report = {
        "purpose": "compare quick extinguish with explicit foreground slabs",
        "cloudy_executable": str(cloudy_exe),
        "foreground_column_density_cm-2": 10.0 ** args.log_nh,
        "foreground_densities_cm-3": list(args.densities),
        "physical_slab_assumptions": {
            "geometry": "open plane-parallel slab (Cloudy default)",
            "density": "constant hydrogen density",
            "temperature": "Cloudy thermal equilibrium; default temperature stop disabled",
            "abundances": "Cloudy defaults",
            "variants": {
                "gas_only": "Cloudy default abundances; no grains",
                "ism_grains": (
                    "abundances ISM no grains + grains ISM; depleted ISM "
                    "gas-phase abundances and one ISM grain population"
                ),
            },
            "cosmic_rays": False,
            "molecular_network": "Cloudy default",
            "charge_transfer": "Cloudy default",
            "iterations": "iterate to convergence",
        },
        "curve_definitions": {
            "quick": "table ISM + extinguish column=21 leak=0; saved incident field",
            "direct": "save continuum column 3: attenuated incident only",
            "outward_diffuse": "save continuum column 4: outward newly emitted continuum and lines",
            "net": "save continuum column 5 = direct + outward diffuse; same physical quantity saved by save transmitted continuum",
        },
        "inputs_and_raw_outputs": {
            "quick_input": str(output_dir / f"{quick_root}.in"),
            "quick_incident": str(quick_path),
            "slab_variants": variants,
        },
        "outputs": {
            "png": str(png_path),
            "pdf": str(pdf_path),
            "npz": str(npz_path),
        },
    }
    # Convert arrays out of the nested raw-output metadata before JSON output.
    for slabs in report["inputs_and_raw_outputs"]["slab_variants"].values():
        for slab in slabs.values():
            for key in ("direct", "outward_diffuse", "net"):
                slab.pop(key)
    report_path = output_dir / (
        "ism_extinguish_vs_physical_slabs_with_grains_NH21.json"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
