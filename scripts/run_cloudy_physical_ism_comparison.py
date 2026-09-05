#!/usr/bin/env python3
"""Run the physical-ISM table, LOS-z spectrum, and quick-ISM comparison."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


TABLE_STEM = (
    "cloudy_hm2012_native_plus_physical_ism_nh8_nh21_grains_cmb_cr_mol_ct_"
    "defaultabund_sixline"
)


def run(command: list[str], *, root: Path) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloudy-exe", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--resume-after-column",
        action="store_true",
        help="reuse the completed column grid and restart at the Jeans grid",
    )
    parser.add_argument(
        "--spectra-only",
        action="store_true",
        help="reuse both completed Cloudy tables and run sampling/plots only",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    output_root = (
        root / "output/plt0655228_down1_Lext0kpc_z2ray_harmonic"
    )
    physical_spectrum_dir = (
        output_root / "native_hm12_physical_ism_cmb_cr_mol_ct_sixline_LOSz"
    )
    physical_tag = (
        "nativeHM2012_physicalISM_nH8_NH21_grains_CMB_CR_molecular_"
        "charge_transfer_z2ray"
    )
    physical_spectrum = (
        physical_spectrum_dir
        / f"{physical_tag}_sixline_Tsplit_Rinf_LOSz.npz"
    )
    quick_spectrum = (
        output_root
        / "native_hm12_filtered_black_ism_cmb_cr_mol_ct_sixline_LOSz"
        / "nativeHM2012_filteredISM_CMB_CR_molecular_charge_transfer_z2ray_"
        "sixline_Tsplit_Rinf_LOSz.npz"
    )
    if not quick_spectrum.is_file():
        raise FileNotFoundError(f"old quick-extinguish spectrum missing: {quick_spectrum}")

    runtime_grackle = root / "runtime/cloudy_sixline/examples/grackle"
    stem_without_prefix = TABLE_STEM.removeprefix("cloudy_")
    if args.spectra_only:
        for suffix in ("column_10x10x21", "jeans_10x21"):
            table = root / f"data/{TABLE_STEM}_{suffix}.npz"
            if not table.is_file():
                raise FileNotFoundError(table)
    elif args.resume_after_column:
        column_files = list(
            (runtime_grackle / f"{stem_without_prefix}_column_10x10x21_output")
            .glob("*_run*.dat")
        )
        if len(column_files) != 100:
            raise RuntimeError(
                f"resume requires 100 completed column run files, found {len(column_files)}"
            )
        jeans_output = runtime_grackle / f"{stem_without_prefix}_jeans_10x21_output"
        if jeans_output.exists() and any(jeans_output.iterdir()):
            raise RuntimeError(
                "Jeans output directory is not empty; preserve/remove the interrupted "
                f"directory before resume: {jeans_output}"
            )
        jeans_output.mkdir(parents=True, exist_ok=True)
        run(
            [
                "perl",
                str(root / "vendor/cloudy_cooling_tools/CIAOLoop_lines"),
                "-np",
                str(args.workers),
                f"{stem_without_prefix}_jeans_10x21.par",
            ],
            root=runtime_grackle,
        )
        run(
            [
                sys.executable,
                str(root / "scripts/build_hm12_filtered_ism_sixline_bundles.py"),
                "--stem",
                stem_without_prefix,
                "--runtime-grackle-dir",
                str(runtime_grackle),
                "--output-dir",
                str(root / "data"),
                "--charge-transfer-enabled",
                "--cosmic-ray-rate-s",
                "2e-17",
                "--cmb-redshift",
                "0",
                "--molecular-network-enabled",
                "--radiation-field-description",
                (
                    "Cloudy-native HM2012 + shielded-face net transmitted table ISM "
                    "through nH=8 cm^-3, NH=1e21 cm^-2 slab with ISM grains"
                ),
                "--foreground-grains",
            ],
            root=root,
        )
    else:
        build_command = [
            sys.executable,
            str(root / "scripts/build_cloudy_sixline_tables.py"),
            "--cloudy-exe",
            str(args.cloudy_exe.expanduser().resolve()),
            "--radiation-model",
            "physical",
            "--workers",
            str(args.workers),
        ]
        if args.force:
            build_command.append("--force")
        run(build_command, root=root)

    # The adopted comparison uses the +/-z harmonic column cache with no
    # lateral extension.  Set these explicitly so ambient shell defaults
    # cannot silently select the legacy cache tree.
    os.environ["LEXT_KPC"] = "0"
    os.environ["COLDEN_DIRECTIONS"] = "z"
    os.environ["QUOKKA_CACHE_ROOT"] = str(root / "intermediates_z2ray_harmonic")
    spectrum_command = [
        sys.executable,
        str(root / "scripts/plot_hm12_filtered_ism_sixline_spectra.py"),
        "--column-table",
        str(root / f"data/{TABLE_STEM}_column_10x10x21.npz"),
        "--jeans-table",
        str(root / f"data/{TABLE_STEM}_jeans_10x21.npz"),
        "--los",
        "z",
        "--workers",
        str(args.workers),
        "--output-dir",
        str(physical_spectrum_dir),
        "--state-key",
        "cloudy_native_hm2012_physical_ism_nh8_nh21_grains_cmb_cr_mol_ct_z2ray",
        "--cloudy-label",
        "Cloudy HM2012 + physical-slab transmitted ISM",
        "--filename-tag",
        physical_tag,
        "--skip-figures",
    ]
    if args.force:
        spectrum_command.append("--force")
    run(spectrum_command, root=root)

    run(
        [
            sys.executable,
            str(root / "scripts/plot_cloudy_physical_vs_quick_ism_spectra.py"),
            "--quick-spectrum",
            str(quick_spectrum),
            "--physical-spectrum",
            str(physical_spectrum),
            "--output-dir",
            str(output_root / "cloudy_physical_vs_quick_ism_sixline_LOSz"),
        ],
        root=root,
    )
    print("\nPhysical-slab Cloudy comparison completed.", flush=True)


if __name__ == "__main__":
    main()
