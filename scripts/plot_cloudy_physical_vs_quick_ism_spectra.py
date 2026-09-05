#!/usr/bin/env python3
"""Plot physical-slab versus quick-extinguished Cloudy line spectra."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quokka2s.pipeline.spectrum_units import DSIGMA_DV_UNIT, dsigma_dv_ylabel


LINE_KEYS = ("cii", "halpha", "hi21", "ciii_977", "ciii_1907", "ciii_1909")
LINE_TITLES = {
    "cii": r"C II 158 $\mu$m",
    "halpha": r"H$\alpha$",
    "hi21": "H I 21 cm",
    "ciii_977": r"C III 977.020 $\AA$",
    "ciii_1907": r"C III] 1906.68 $\AA$",
    "ciii_1909": r"C III] 1908.73 $\AA$",
}
GEOMETRY_TITLES = {
    "column": r"$(N_{\rm H}^{z\pm,\,harm},n_{\rm H},T)$",
    "jeans": r"$(n_{\rm H},T)$ Jeans length",
}
REFERENCE_LABELS = {
    "cii": "DESPOTIC",
    "halpha": "Pipeline",
    "hi21": "Pipeline",
}


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        result = {name: np.asarray(source[name]) for name in source.files}
    expected = {
        "line_keys": LINE_KEYS,
        "geometry_keys": ("column", "jeans"),
        "regime_keys": ("T_QUOKKA_lt_3000K", "T_QUOKKA_ge_3000K"),
    }
    for name, values in expected.items():
        actual = tuple(str(value) for value in result[name].tolist())
        if actual != values:
            raise ValueError(f"unexpected {name} {actual}: {path}")
    if result["dsigma_dv"].shape[:3] != (6, 2, 2):
        raise ValueError(f"unexpected spectrum shape: {path}")
    if result["reference_dsigma_dv"].shape[:2] != (3, 2):
        raise ValueError(f"unexpected reference spectrum shape: {path}")
    if not bool(result["completed_full_domain"].item()):
        raise ValueError(f"spectrum does not cover the full domain: {path}")
    if str(result["los"].item()) != "z":
        raise ValueError(f"comparison expects LOS z: {path}")
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    old_default = (
        root
        / "output/plt0655228_down1_Lext0kpc_z2ray_harmonic"
        / "native_hm12_filtered_black_ism_cmb_cr_mol_ct_sixline_LOSz"
        / "nativeHM2012_filteredISM_CMB_CR_molecular_charge_transfer_z2ray_"
        "sixline_Tsplit_Rinf_LOSz.npz"
    )
    new_default = (
        root
        / "output/plt0655228_down1_Lext0kpc_z2ray_harmonic"
        / "native_hm12_physical_ism_cmb_cr_mol_ct_sixline_LOSz"
        / "nativeHM2012_physicalISM_nH8_NH21_grains_CMB_CR_molecular_"
        "charge_transfer_z2ray_sixline_Tsplit_Rinf_LOSz.npz"
    )
    output_default = (
        root
        / "output/plt0655228_down1_Lext0kpc_z2ray_harmonic"
        / "cloudy_physical_vs_quick_ism_sixline_LOSz"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick-spectrum", type=Path, default=old_default)
    parser.add_argument("--physical-spectrum", type=Path, default=new_default)
    parser.add_argument("--output-dir", type=Path, default=output_default)
    args = parser.parse_args()
    quick_path = args.quick_spectrum.expanduser().resolve()
    physical_path = args.physical_spectrum.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    quick = load(quick_path)
    physical = load(physical_path)
    velocity = np.asarray(quick["velocity_kms"], dtype=float)
    if not np.array_equal(velocity, physical["velocity_kms"]):
        raise ValueError("velocity axes differ")
    quick_values = np.asarray(quick["dsigma_dv"], dtype=float)
    physical_values = np.asarray(physical["dsigma_dv"], dtype=float)
    quick_reference = np.asarray(quick["reference_dsigma_dv"], dtype=float)
    physical_reference = np.asarray(physical["reference_dsigma_dv"], dtype=float)
    if not np.array_equal(quick_reference, physical_reference):
        raise ValueError("independent DESPOTIC/pipeline reference spectra differ")
    quick_units = str(quick["dsigma_dv_units"].item())
    physical_units = str(physical["dsigma_dv_units"].item())
    if quick_units != physical_units or quick_units != DSIGMA_DV_UNIT:
        raise ValueError("spectrum units differ")

    figures = {}
    ratios = {}
    dv = float(np.mean(np.diff(velocity)))
    for line_index, line in enumerate(LINE_KEYS):
        for geometry_index, geometry in enumerate(("column", "jeans")):
            curves = np.stack(
                (
                    quick_values[line_index, geometry_index],
                    physical_values[line_index, geometry_index],
                )
            )
            reference = (
                quick_reference[line_index]
                if line in REFERENCE_LABELS
                else None
            )
            plotted_curves = (
                np.concatenate((curves, reference[None]), axis=0)
                if reference is not None
                else curves
            )
            shared_max = float(np.nanmax(plotted_curves))
            fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
            for branch, axis in enumerate(axes):
                # Draw the solid physical-slab result first.  The dashed
                # quick-extinguish curve is intentionally drawn last and at a
                # higher z-order so coincident sections remain visible.
                axis.plot(
                    velocity,
                    curves[1, branch],
                    color="#D55E00",
                    linewidth=1.8,
                    drawstyle="steps-mid",
                    label="Cloudy HM2012 + physical-slab transmitted ISM",
                    zorder=2,
                )
                if reference is not None:
                    axis.plot(
                        velocity,
                        reference[branch],
                        color="#000000",
                        linestyle="-.",
                        linewidth=1.7,
                        drawstyle="steps-mid",
                        label=REFERENCE_LABELS[line],
                        zorder=3,
                    )
                axis.plot(
                    velocity,
                    curves[0, branch],
                    color="#0072B2",
                    linestyle="--",
                    linewidth=1.8,
                    drawstyle="steps-mid",
                    label="Cloudy HM2012 + quick-extinguished ISM",
                    zorder=4,
                )
                axis.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
                axis.set_xlabel(r"Velocity [km s$^{-1}$]")
                axis.set_ylabel(dsigma_dv_ylabel(DSIGMA_DV_UNIT))
                axis.set_title(
                    r"$T_{\rm QUOKKA}<3000\,$K"
                    if branch == 0
                    else r"$T_{\rm QUOKKA}\geq3000\,$K"
                )
                axis.set_ylim(0.0, 1.05 * shared_max if shared_max > 0.0 else 1.0)
                axis.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
                axis.legend(fontsize=7.7, frameon=False)
                axis.ticklabel_format(
                    style="sci", axis="y", scilimits=(0, 0), useMathText=True
                )
                axis.tick_params(axis="y", labelleft=True)
            fig.suptitle(
                f"{LINE_TITLES[line]} {GEOMETRY_TITLES[geometry]}, "
                r"LOS z, $R=\infty$"
            )
            fig.tight_layout()
            suffix = "_with_reference" if reference is not None else ""
            stem = (
                f"{line}_cloudy_physical_vs_quick_ism_{geometry}_"
                f"Tsplit_LOSz{suffix}"
            )
            png = output_dir / f"{stem}.png"
            pdf = output_dir / f"{stem}.pdf"
            fig.savefig(png, dpi=250, bbox_inches="tight")
            fig.savefig(pdf, bbox_inches="tight")
            plt.close(fig)
            figures[f"{line}_{geometry}"] = {"png": str(png), "pdf": str(pdf)}

            quick_integral = np.sum(curves[0], axis=-1) * dv
            physical_integral = np.sum(curves[1], axis=-1) * dv
            ratio = np.divide(
                physical_integral,
                quick_integral,
                out=np.full(2, np.nan),
                where=quick_integral != 0.0,
            )
            ratios[f"{line}_{geometry}"] = {
                "T_QUOKKA_lt_3000K": float(ratio[0]),
                "T_QUOKKA_ge_3000K": float(ratio[1]),
            }

    report = {
        "quick_spectrum": str(quick_path),
        "physical_spectrum": str(physical_path),
        "comparison": (
            "physical-slab versus quick-extinguished Cloudy, with the existing "
            "independent DESPOTIC/pipeline reference added where available"
        ),
        "independent_reference_policy": {
            "cii": "DESPOTIC emissivity on the same cells",
            "halpha": (
                "pipeline analytic recombination emission; DESPOTIC e-/H+ below "
                "3000 K and QUOKKA-mu-derived e-/H+ otherwise"
            ),
            "hi21": (
                "pipeline analytic emission; DESPOTIC n_HI below 3000 K and "
                "QUOKKA-mu-derived n_HI otherwise"
            ),
            "ciii": "no DESPOTIC or pipeline reference available",
        },
        "lines": list(LINE_KEYS),
        "geometries": ["column", "jeans"],
        "physical_to_quick_integrated_spectrum_ratio": ratios,
        "figures": figures,
    }
    report_path = output_dir / "cloudy_physical_vs_quick_ism_sixline_LOSz.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
