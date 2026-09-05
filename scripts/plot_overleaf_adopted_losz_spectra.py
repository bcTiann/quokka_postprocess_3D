#!/usr/bin/env python3
"""Plot the LOS-z spectra using the emission methods adopted in the paper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quokka2s.pipeline.spectrum_units import dsigma_dv_ylabel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = (
    ROOT
    / "output/plt0655228_down1_Lext15kpc/dvdr_extension_validation"
    / "extended/LOSz/extended_dvdr_sixline_Tsplit_Rinf_LOSz.npz"
)
DEFAULT_REPORT_ROOT = Path("/Users/tianbaochen/Documents/Report/2026-09-05")

LINE_TITLES = {
    "cii": r"C II 158 $\mu$m",
    "halpha": r"H$\alpha$",
    "hi21": "H I 21 cm",
    "ciii_977": r"C III 977 $\AA$",
    "ciii_1907": r"C III] 1907 $\AA$",
    "ciii_1909": r"C III] 1909 $\AA$",
    "co10": "CO(1-0)",
    "co21": "CO(2-1)",
}


def _plot_two_regime_spectrum(
    *,
    velocity: np.ndarray,
    spectra: np.ndarray,
    labels: tuple[str, str],
    colors: tuple[str, str],
    title: str,
    unit: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    if spectra.shape != (2, velocity.size):
        raise ValueError(f"unexpected spectrum shape: {spectra.shape}")
    shared_max = float(np.nanmax(spectra))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    for branch, axis in enumerate(axes):
        axis.plot(
            velocity,
            spectra[branch],
            color=colors[branch],
            linewidth=1.9,
            drawstyle="steps-mid",
            label=labels[branch],
        )
        axis.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
        axis.set_xlabel(r"Velocity [km s$^{-1}$]")
        axis.set_ylabel(dsigma_dv_ylabel(unit))
        axis.set_title(
            r"$T_{\rm QUOKKA}<3000\,$K"
            if branch == 0
            else r"$T_{\rm QUOKKA}\geq3000\,$K"
        )
        axis.set_ylim(0.0, 1.05 * shared_max if shared_max > 0.0 else 1.0)
        axis.ticklabel_format(
            style="sci", axis="y", scilimits=(0, 0), useMathText=True
        )
        axis.tick_params(axis="y", labelleft=True)
        axis.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        axis.legend(frameon=False, fontsize=9)
    fig.suptitle(title + r", LOS z, $R=\infty$")
    fig.tight_layout()
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args()

    bundle = args.bundle.expanduser().resolve()
    report_root = args.report_root.expanduser().resolve()
    png_dir = report_root / "png"
    pdf_dir = report_root / "pdf"
    metadata_dir = report_root / "metadata"
    for directory in (png_dir, pdf_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with np.load(bundle, allow_pickle=False) as source:
        if str(np.asarray(source["los"]).item()) != "z":
            raise ValueError("the adopted-paper plot requires an LOS-z bundle")
        if not bool(np.asarray(source["completed_full_domain"]).item()):
            raise ValueError("the input bundle is not a completed full-domain run")

        velocity = np.asarray(source["velocity_kms"], dtype=float)
        unit = str(np.asarray(source["dsigma_dv_units"]).item())
        line_keys = tuple(str(item) for item in source["line_keys"])
        co_keys = tuple(str(item) for item in source["co_line_keys"])
        cloudy = np.asarray(source["dsigma_dv"], dtype=float)
        reference = np.asarray(source["reference_dsigma_dv"], dtype=float)
        co = np.asarray(source["co_dsigma_dv"], dtype=float)

    line_index = {key: index for index, key in enumerate(line_keys)}
    co_index = {key: index for index, key in enumerate(co_keys)}
    required = {"cii", "halpha", "hi21", "ciii_977", "ciii_1907", "ciii_1909"}
    if not required.issubset(line_index):
        raise ValueError(f"missing Cloudy lines: {sorted(required - set(line_index))}")
    if not {"co10", "co21"}.issubset(co_index):
        raise ValueError("the bundle must contain CO(1-0) and CO(2-1)")

    adopted: dict[str, tuple[np.ndarray, tuple[str, str], tuple[str, str]]] = {}

    # Halpha and H I use the analytic DESPOTIC-density treatment below
    # 3000 K and the Cloudy emissivity above the temperature split.
    for key, reference_index in (("halpha", 1), ("hi21", 2)):
        values = np.stack(
            (reference[reference_index, 0], cloudy[line_index[key], 1]), axis=0
        )
        low_label = (
            "DESPOTIC densities + analytic Hα"
            if key == "halpha"
            else "DESPOTIC n(H I) + analytic H I"
        )
        adopted[key] = (values, (low_label, "Cloudy"), ("#0072B2", "#D55E00"))

    # All carbon lines use the Cloudy table: T_DESPOTIC is used as the lookup
    # temperature below 3000 K and T_QUOKKA above it.
    for key in ("cii", "ciii_977", "ciii_1907", "ciii_1909"):
        adopted[key] = (
            cloudy[line_index[key]],
            (r"Cloudy ($T_{\rm DESPOTIC}$)", r"Cloudy ($T_{\rm QUOKKA}$)"),
            ("#D55E00", "#D55E00"),
        )

    # CO uses the DESPOTIC emissivity and equilibrium temperature in both
    # displayed T_QUOKKA regimes.
    for key in ("co10", "co21"):
        adopted[key] = (
            co[co_index[key]],
            ("DESPOTIC", "DESPOTIC"),
            ("#0072B2", "#0072B2"),
        )

    outputs: dict[str, dict[str, str]] = {}
    for key in (
        "halpha",
        "hi21",
        "cii",
        "ciii_977",
        "ciii_1907",
        "ciii_1909",
        "co10",
        "co21",
    ):
        values, labels, colors = adopted[key]
        stem = f"{key}_adopted_method_Tsplit_Rinf_LOSz"
        png_path = png_dir / f"{stem}.png"
        pdf_path = pdf_dir / f"{stem}.pdf"
        if png_path.exists() or pdf_path.exists():
            raise FileExistsError(f"refusing to overwrite existing figure: {stem}")
        _plot_two_regime_spectrum(
            velocity=velocity,
            spectra=values,
            labels=labels,
            colors=colors,
            title=LINE_TITLES[key],
            unit=unit,
            png_path=png_path,
            pdf_path=pdf_path,
        )
        outputs[key] = {"png": str(png_path), "pdf": str(pdf_path)}

    manifest = {
        "source_bundle": str(bundle),
        "los": "z",
        "temperature_split_K": 3000.0,
        "unit": unit,
        "method": {
            "halpha": "DESPOTIC e-/H+ and analytic emissivity below 3000 K; Cloudy above",
            "hi21": "DESPOTIC H I density and analytic emissivity below 3000 K; Cloudy above",
            "carbon": "Cloudy with T_DESPOTIC below 3000 K and T_QUOKKA above",
            "co": "DESPOTIC emissivity and T_DESPOTIC in both T_QUOKKA regimes",
        },
        "outputs": outputs,
    }
    manifest_path = metadata_dir / "adopted_method_LOSz_spectra.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved 8 PNG files to {png_dir}")
    print(f"Saved 8 PDF files to {pdf_dir}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
