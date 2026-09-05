#!/usr/bin/env python3
"""Compare spectra made with the legacy and extended DESPOTIC dV/dr axes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quokka2s.pipeline.spectrum_units import DSIGMA_DV_UNIT, dsigma_dv_ylabel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "output/plt0655228_down1_Lext15kpc/dvdr_extension_validation"
)
ATOMIC_TITLES = {
    "cii": r"C II 158 $\mu$m",
    "halpha": r"H$\alpha$",
    "hi21": "H I 21 cm",
    "ciii_977": r"C III 977.020 $\AA$",
    "ciii_1907": r"C III] 1906.68 $\AA$",
    "ciii_1909": r"C III] 1908.73 $\AA$",
}
CO_TITLES = {"co10": "CO(1-0)", "co21": "CO(2-1)"}


def _bundle_path(root: Path, case: str, los: str) -> Path:
    tag = "legacy_dvdr" if case == "legacy" else "extended_dvdr"
    return root / case / f"LOS{los}" / f"{tag}_sixline_Tsplit_Rinf_LOS{los}.npz"


def _report_path(root: Path, case: str, los: str) -> Path:
    tag = "legacy_dvdr" if case == "legacy" else "extended_dvdr"
    return root / case / f"LOS{los}" / f"{tag}_sixline_Tsplit_Rinf_LOS{los}.json"


def _fractional_change(old: float, new: float) -> float | None:
    return None if old == 0.0 else (new - old) / old


def _curve_metrics(old: np.ndarray, new: np.ndarray) -> dict[str, float | None]:
    old_peak = float(np.max(old))
    new_peak = float(np.max(new))
    denominator = float(np.sum(np.abs(old)))
    return {
        "old_peak": old_peak,
        "new_peak": new_peak,
        "peak_fractional_change": _fractional_change(old_peak, new_peak),
        "relative_L1_spectrum_change": (
            None if denominator == 0.0
            else float(np.sum(np.abs(new - old)) / denominator)
        ),
    }


def _line_metrics(
    old_curve: np.ndarray,
    new_curve: np.ndarray,
    old_luminosity: np.ndarray,
    new_luminosity: np.ndarray,
) -> dict[str, dict[str, float | None]]:
    metrics = {
        regime: {
            **_curve_metrics(old_curve[branch], new_curve[branch]),
            "old_input_luminosity_erg_s": float(old_luminosity[branch]),
            "new_input_luminosity_erg_s": float(new_luminosity[branch]),
            "input_luminosity_fractional_change": _fractional_change(
                float(old_luminosity[branch]),
                float(new_luminosity[branch]),
            ),
        }
        for branch, regime in enumerate(("T_lt_3000", "T_ge_3000"))
    }
    old_total_luminosity = float(np.sum(old_luminosity))
    new_total_luminosity = float(np.sum(new_luminosity))
    metrics["total"] = {
        **_curve_metrics(np.sum(old_curve, axis=0), np.sum(new_curve, axis=0)),
        "old_input_luminosity_erg_s": old_total_luminosity,
        "new_input_luminosity_erg_s": new_total_luminosity,
        "input_luminosity_fractional_change": _fractional_change(
            old_total_luminosity,
            new_total_luminosity,
        ),
    }
    return metrics


def _plot_curves(
    output_stem: Path,
    velocity: np.ndarray,
    curves: list[tuple[str, np.ndarray, str, str]],
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    shared_max = max(float(np.nanmax(values)) for _, values, _, _ in curves)
    for branch, axis in enumerate(axes):
        for label, values, color, style in curves:
            axis.plot(
                velocity,
                values[branch],
                color=color,
                linestyle=style,
                linewidth=1.8,
                drawstyle="steps-mid",
                label=label,
                # Draw the legacy dashed curve last/on top.  Most old/new
                # differences are too small to see if the solid curve covers it.
                zorder=4 if style == "--" else 3,
            )
        axis.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
        axis.set_xlabel(r"Velocity [km s$^{-1}$]")
        axis.set_ylabel(dsigma_dv_ylabel(DSIGMA_DV_UNIT))
        axis.set_title(
            r"$T_{\rm QUOKKA}<3000\,$K" if branch == 0
            else r"$T_{\rm QUOKKA}\geq3000\,$K"
        )
        axis.set_ylim(0.0, 1.05 * shared_max if shared_max > 0.0 else 1.0)
        axis.ticklabel_format(
            style="sci", axis="y", scilimits=(0, 0), useMathText=True
        )
        axis.tick_params(axis="y", labelleft=True)
        axis.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        axis.legend(fontsize=7.5, frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(output_stem.with_suffix(suffix), dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {"output_root": str(output_root), "los": {}}
    for los in ("y", "z"):
        old_path = _bundle_path(output_root, "legacy", los)
        new_path = _bundle_path(output_root, "extended", los)
        with np.load(old_path, allow_pickle=False) as old_blob:
            old = {name: np.asarray(old_blob[name]) for name in old_blob.files}
        with np.load(new_path, allow_pickle=False) as new_blob:
            new = {name: np.asarray(new_blob[name]) for name in new_blob.files}
        if not np.array_equal(old["velocity_kms"], new["velocity_kms"]):
            raise ValueError(f"velocity grids differ for LOS {los}")
        velocity = old["velocity_kms"]
        line_keys = tuple(str(value) for value in old["line_keys"])
        co_keys = tuple(str(value) for value in old["co_line_keys"])
        los_metrics: dict[str, object] = {
            "legacy_bundle": str(old_path),
            "extended_bundle": str(new_path),
            "atomic_cloudy": {},
            "reference": {},
            "co_despotic": {},
            "sampling": {
                "legacy": json.loads(
                    _report_path(output_root, "legacy", los).read_text()
                )["counts"],
                "extended": json.loads(
                    _report_path(output_root, "extended", los).read_text()
                )["counts"],
            },
        }

        for index, key in enumerate(line_keys):
            old_curve = old["dsigma_dv"][index]
            new_curve = new["dsigma_dv"][index]
            old_lum = old["input_luminosity_erg_s"][index]
            new_lum = new["input_luminosity_erg_s"][index]
            los_metrics["atomic_cloudy"][key] = _line_metrics(
                old_curve, new_curve, old_lum, new_lum
            )
            curves = [
                ("Cloudy, legacy dV/dr", old_curve, "#D55E00", "--"),
                ("Cloudy, extended dV/dr", new_curve, "#D55E00", "-"),
            ]
            if index < 3:
                old_reference = old["reference_dsigma_dv"][index]
                new_reference = new["reference_dsigma_dv"][index]
                reference_key = ("cii", "halpha", "hi21")[index]
                old_ref_lum = old["reference_input_luminosity_erg_s"][index]
                new_ref_lum = new["reference_input_luminosity_erg_s"][index]
                los_metrics["reference"][reference_key] = _line_metrics(
                    old_reference, new_reference, old_ref_lum, new_ref_lum
                )
                reference_label = (
                    "DESPOTIC" if reference_key == "cii" else "pipeline"
                )
                curves = [
                    (f"{reference_label}, legacy dV/dr", old_reference, "#0072B2", "--"),
                    (f"{reference_label}, extended dV/dr", new_reference, "#0072B2", "-"),
                    *curves,
                ]
            _plot_curves(
                comparison_dir / f"{key}_legacy_vs_extended_dvdr_LOS{los}",
                velocity,
                curves,
                f"{ATOMIC_TITLES[key]}, LOS {los}, " + r"$R=\infty$",
            )

        for index, key in enumerate(co_keys):
            old_curve = old["co_dsigma_dv"][index]
            new_curve = new["co_dsigma_dv"][index]
            old_lum = old["co_input_luminosity_erg_s"][index]
            new_lum = new["co_input_luminosity_erg_s"][index]
            los_metrics["co_despotic"][key] = _line_metrics(
                old_curve, new_curve, old_lum, new_lum
            )
            _plot_curves(
                comparison_dir / f"{key}_legacy_vs_extended_dvdr_LOS{los}",
                velocity,
                [
                    ("DESPOTIC, legacy dV/dr", old_curve, "#0072B2", "--"),
                    ("DESPOTIC, extended dV/dr", new_curve, "#0072B2", "-"),
                ],
                f"{CO_TITLES[key]}, LOS {los}, " + r"$R=\infty$",
            )
        report["los"][los] = los_metrics

    report_path = comparison_dir / "despotic_dvdr_extension_comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
