"""Plot QUOKKA/Saha, DESPOTIC, and two Cloudy-state spectra.

Each of CII, H-alpha, and H I 21-cm is shown below and above the
``T_QUOKKA=3000 K`` boundary.  Existing intrinsic-spectrum intermediates are
reused for CII and H I.  The two all-temperature H-alpha diagnostics are
streamed once from the simulation:

* QUOKKA-mu recombination uses the mu-derived electron fraction at T_QUOKKA.
* DESPOTIC recombination uses DESPOTIC e-/H+ and T_DESPOTIC.

The two Cloudy curves come from the completed four-state ablation-spectrum
intermediate: baseline and molecular+charge-transfer.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt
from yt.units.physical_constants import kb, mh
from yt.units.yt_array import YTArray

from plot_cloudy_line_physics_ablation_spectra import (
    N_CHANNELS,
    REGIME_SPLIT_K,
    V_RANGE_KMS,
    accumulate_velocity_spectra,
)
from quokka2s.line_regimes import electron_fraction_from_mean_molecular_weight
from quokka2s.pipeline.cache import (
    cache_root_for_dataset,
    compute_cache_key,
    field_cache_key,
    field_cache_path,
)
from quokka2s.pipeline.prep import config as cfg
from quokka2s.pipeline.prep.physics_fields import (
    _clip_to_table_domain,
    c,
    effective_halpha_recombination_coefficient,
    h,
    lambda_Halpha,
)
from quokka2s.pipeline.spectrum_units import (
    DSIGMA_DV_UNIT,
    SPEED_OF_LIGHT_CGS,
    dsigma_dv_ylabel,
)
from quokka2s.tables import load_table
from quokka2s.tables.lookup import TableLookup


COLUMN_FIELD = ("gas", "column_density_H")
DVDR_FIELD = ("gas", "dVdr_lvg")
TDSP_FIELD = ("gas", "temperature_despotic")
MODEL_KEYS = ("quokka", "despotic", "cloudy_baseline", "cloudy_mol_ct")
MODEL_STYLES = {
    "quokka": ("#009E73", "-"),
    "despotic": ("#0072B2", "--"),
    "cloudy_baseline": ("#D55E00", "-."),
    "cloudy_mol_ct": ("#CC79A7", ":"),
}
SPECIES_LABELS = {
    "cii": "[C II] 158 μm",
    "halpha": r"H$\alpha$",
    "hi21": "H I 21 cm",
}
MODEL_LABELS = {
    "cii": {
        "quokka": r"QUOKKA $\mu$ (Saha/LTE)",
        "despotic": "DESPOTIC",
        "cloudy_baseline": "Cloudy baseline",
        "cloudy_mol_ct": "Cloudy molecular + charge transfer",
    },
    "halpha": {
        "quokka": r"QUOKKA $\mu$ (recombination)",
        "despotic": "DESPOTIC chemistry",
        "cloudy_baseline": "Cloudy baseline",
        "cloudy_mol_ct": "Cloudy molecular + charge transfer",
    },
    "hi21": {
        "quokka": r"QUOKKA $\mu$",
        "despotic": "DESPOTIC",
        "cloudy_baseline": "Cloudy baseline",
        "cloudy_mol_ct": "Cloudy molecular + charge transfer",
    },
}
OUTPUT_FILENAMES = {
    "cii": "CII_four_model_CloudyBaseline_MolCT_Tsplit_targeted_zooms_Rinf.png",
    "halpha": "Halpha_four_model_CloudyBaseline_MolCT_Tsplit_targeted_zooms_Rinf.png",
    "hi21": "HI21_four_model_CloudyBaseline_MolCT_Tsplit_targeted_zooms_Rinf.png",
}
NO_LOW_QUOKKA_OUTPUT_FILENAMES = {
    "cii": "CII_CloudyBaseline_MolCT_DESPOTIC_noLowQUOKKA_Tsplit_zooms_Rinf.png",
    "halpha": "Halpha_CloudyBaseline_MolCT_DESPOTIC_noLowQUOKKA_Tsplit_zooms_Rinf.png",
    "hi21": "HI21_CloudyBaseline_MolCT_DESPOTIC_noLowQUOKKA_Tsplit_zooms_Rinf.png",
}

# These panels need a branch-scale inset even when no single curve dominates
# within that branch.  C II-low excludes its dominant QUOKKA/Saha curve so the
# other three can be compared; H-alpha-high and H I-high show all four curves.
FORCED_ZOOMS = {
    ("cii", 0): "exclude_dominant",
    ("halpha", 1): "all_models",
    ("hi21", 1): "all_models",
}


def _read_curve(
    path: Path,
    group: str,
    *,
    los_group: bool,
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        base = handle["spectra"][group]
        if los_group:
            base = base["y"]
        velocity = np.asarray(base["v_axis"], dtype=float)
        spectrum = np.asarray(base["dsigma_dv"], dtype=float)
    if velocity.shape != (N_CHANNELS,) or spectrum.shape != (N_CHANNELS,):
        raise ValueError(f"unexpected spectrum shape in {path}: {group}")
    return velocity, spectrum


def _validated_cache(
    path: Path,
    expected_key: str,
    expected_field: tuple[str, str],
    dimensions: tuple[int, int, int],
    *,
    legacy_key: str | None = None,
) -> h5py.File:
    handle = h5py.File(path, "r")
    actual_key = str(handle.attrs.get("cache_key", ""))
    actual_field = (
        str(handle.attrs.get("field_type", "")),
        str(handle.attrs.get("field_name", "")),
    )
    valid_keys = {expected_key}
    if legacy_key is not None:
        valid_keys.add(legacy_key)
    if actual_key not in valid_keys or actual_field != expected_field:
        handle.close()
        raise RuntimeError(f"stale or mismatched field cache: {path}")
    if tuple(handle["data"].shape) != dimensions:
        handle.close()
        raise ValueError(f"cache shape does not match simulation: {path}")
    return handle


def _build_halpha_diagnostics(args: argparse.Namespace) -> Path:
    output = args.output_dir / "Halpha_QUOKKAmu_DESPOTIC_Tsplit_Rinf.npz"
    if output.exists() and not args.force_halpha:
        with np.load(output, allow_pickle=False) as table:
            if bool(np.asarray(table["completed_full_domain"]).item()):
                print(f"Reusing: {output}")
                return output

    ds = yt.load(str(args.dataset))
    ds.force_periodicity()
    dimensions = tuple(int(value) for value in ds.domain_dimensions)
    nx, ny, nz = dimensions
    base_key = compute_cache_key(
        dataset_path=args.dataset,
        despotic_table_path=args.despotic_table,
        downsample_factor=cfg.DOWNSAMPLE_FACTOR,
        column_extension_lateral_kpc=cfg.COLUMN_EXTENSION_LATERAL_KPC,
    )
    legacy_base_key = compute_cache_key(
        dataset_path=args.dataset,
        despotic_table_path=args.despotic_table,
        downsample_factor=cfg.DOWNSAMPLE_FACTOR,
        column_extension_lateral_kpc=cfg.COLUMN_EXTENSION_LATERAL_KPC,
        schema_version=19,
    )
    cache_root = cache_root_for_dataset(args.dataset)
    cache_specs = (
        (COLUMN_FIELD, args.column_cache),
        (DVDR_FIELD, args.dvdr_cache),
        (TDSP_FIELD, args.tdsp_cache),
    )
    handles: dict[tuple[str, str], h5py.File] = {}
    paths: dict[tuple[str, str], Path] = {}
    for field, explicit in cache_specs:
        path = (
            explicit.resolve()
            if explicit is not None
            else field_cache_path(cache_root, field)
        )
        handles[field] = _validated_cache(
            path,
            field_cache_key(base_key, field),
            field,
            dimensions,
            legacy_key=field_cache_key(legacy_base_key, field),
        )
        paths[field] = path

    lookup = TableLookup(load_table(args.despotic_table))
    cell_width_cm = np.asarray(
        ds.domain_width.to("cm") / ds.domain_dimensions, dtype=float,
    )
    volume_cm3 = float(np.prod(cell_width_cm))
    area_cm2 = float(nx * nz * cell_width_cm[0] * cell_width_cm[2])
    hydrogen_mass_g = float(mh.to_value("g"))
    boltzmann_cgs = float(kb.to_value("erg/K"))
    h_mass_g = 1.00794 * 1.66053906660e-24
    photon_energy = float(((h * c) / lambda_Halpha).in_cgs().value)
    c_kms = float(SPEED_OF_LIGHT_CGS.to_value("cm/s")) / 1.0e5
    velocity_edges = np.linspace(-V_RANGE_KMS, V_RANGE_KMS, N_CHANNELS + 1)
    velocity_axis = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])
    accumulated = np.zeros((2, 2, N_CHANNELS), dtype=float)
    n_slabs = (nz + args.slab_nz - 1) // args.slab_nz
    started = time.perf_counter()

    try:
        for slab_number, iz in enumerate(range(0, nz, args.slab_nz), start=1):
            local_nz = min(args.slab_nz, nz - iz)
            left_edge = ds.domain_left_edge.copy()
            left_edge[2] += iz * (ds.domain_width[2] / ds.domain_dimensions[2])
            grid = ds.covering_grid(
                level=ds.max_level,
                left_edge=left_edge,
                dims=(nx, ny, local_nz),
            )
            temperature_qk = np.asarray(
                grid[("boxlib", "temperature")], dtype=float,
            ).reshape(-1)
            density = np.asarray(
                grid[("gas", "density")].to("g/cm**3"), dtype=float,
            ).reshape(-1)
            velocity_y = np.asarray(
                grid[("gas", "velocity_y")].to("km/s"), dtype=float,
            ).reshape(-1)
            total_energy = np.asarray(
                grid[("gas", "total_energy_density")].to("erg/cm**3"),
                dtype=float,
            ).reshape(-1)
            kinetic_energy = np.asarray(
                grid[("gas", "kinetic_energy_density")].to("erg/cm**3"),
                dtype=float,
            ).reshape(-1)
            del grid

            n_H = density * float(cfg.X_H) / hydrogen_mass_g
            column = np.asarray(
                handles[COLUMN_FIELD]["data"][:, :, iz:iz + local_nz],
                dtype=float,
            ).reshape(-1)
            dvdr = np.asarray(
                handles[DVDR_FIELD]["data"][:, :, iz:iz + local_nz],
                dtype=float,
            ).reshape(-1)
            temperature_dsp = np.asarray(
                handles[TDSP_FIELD]["data"][:, :, iz:iz + local_nz],
                dtype=float,
            ).reshape(-1)
            internal_energy = total_energy - kinetic_energy
            low = temperature_qk < REGIME_SPLIT_K

            # QUOKKA-mu recombination diagnostic, applied on both sides of the
            # split for this comparison rather than as a production policy.
            x_e = electron_fraction_from_mean_molecular_weight(
                internal_energy,
                density,
                temperature_qk,
                hydrogen_mass_g=hydrogen_mass_g,
                boltzmann_erg_K=boltzmann_cgs,
            )
            n_e_qk = x_e * n_H
            n_hp_qk = np.where(x_e <= 1.0, x_e, 1.0) * n_H
            alpha_qk = effective_halpha_recombination_coefficient(
                temperature_qk,
            )
            epsilon_qk = photon_energy * alpha_qk * n_e_qk * n_hp_qk

            # All-temperature DESPOTIC-chemistry recombination diagnostic.
            n_safe, column_safe, dvdr_safe = _clip_to_table_domain(
                lookup, n_H, column, dvdr,
            )
            densities = lookup.number_densities(
                ("e-", "H+"), n_safe, column_safe, dvdr_safe,
            )
            n_e_dsp = np.nan_to_num(densities["e-"], nan=0.0)
            n_hp_dsp = np.nan_to_num(densities["H+"], nan=0.0)
            alpha_dsp = effective_halpha_recombination_coefficient(
                temperature_dsp,
            )
            epsilon_dsp = photon_energy * alpha_dsp * n_e_dsp * n_hp_dsp

            for model_index, (epsilon, temperature_width) in enumerate((
                (epsilon_qk, temperature_qk),
                (epsilon_dsp, temperature_dsp),
            )):
                cell_luminosity = epsilon * volume_cm3
                luminosities = np.column_stack((
                    np.where(low, cell_luminosity, 0.0),
                    np.where(low, 0.0, cell_luminosity),
                ))
                thermal_width = np.sqrt(
                    boltzmann_cgs * temperature_width / h_mass_g,
                ) / 1.0e5
                thermal_width *= 1.0 - velocity_y / c_kms
                slab_spectra = accumulate_velocity_spectra(
                    velocity_y,
                    thermal_width,
                    luminosities,
                    velocity_edges,
                    cell_chunk=args.cell_chunk,
                    workers=args.workers,
                )
                accumulated[model_index, 0] += slab_spectra[:, 0]
                accumulated[model_index, 1] += slab_spectra[:, 1]

            elapsed = time.perf_counter() - started
            rate = slab_number / elapsed
            eta = (n_slabs - slab_number) / rate if rate > 0.0 else np.nan
            print(
                f"Halpha [{slab_number:02d}/{n_slabs:02d}] "
                f"z={iz}:{iz + local_nz} elapsed={elapsed / 60.0:.1f} min "
                f"ETA={eta / 60.0:.1f} min",
                flush=True,
            )
    finally:
        for handle in handles.values():
            handle.close()

    spectra = YTArray(
        accumulated / area_cm2,
        "erg/s/cm**2/(km/s)",
    ).to(DSIGMA_DV_UNIT).value
    np.savez_compressed(
        output,
        velocity_kms=velocity_axis,
        dsigma_dv=spectra,
        model_keys=np.asarray(("quokka", "despotic")),
        regime_keys=np.asarray(("T_lt_3000K", "T_ge_3000K")),
        dsigma_dv_units=np.asarray(DSIGMA_DV_UNIT),
        completed_full_domain=np.asarray(True),
        dataset=np.asarray(str(args.dataset)),
        despotic_table=np.asarray(str(args.despotic_table)),
        column_cache=np.asarray(str(paths[COLUMN_FIELD])),
        dvdr_cache=np.asarray(str(paths[DVDR_FIELD])),
        temperature_despotic_cache=np.asarray(str(paths[TDSP_FIELD])),
        workers=np.asarray(args.workers),
        elapsed_minutes=np.asarray((time.perf_counter() - started) / 60.0),
    )
    print(f"Saved: {output}")
    return output


def _orient_like(
    source_velocity: np.ndarray,
    source_spectrum: np.ndarray,
    target_velocity: np.ndarray,
) -> np.ndarray:
    if np.allclose(source_velocity, target_velocity, rtol=0.0, atol=1.0e-10):
        return source_spectrum
    if np.allclose(source_velocity[::-1], target_velocity, rtol=0.0, atol=1.0e-10):
        return source_spectrum[::-1]
    raise ValueError("spectrum velocity grids do not match")


def _load_curves(args: argparse.Namespace, halpha_path: Path) -> dict[str, dict]:
    with np.load(args.cloudy_spectra, allow_pickle=False) as table:
        cloud_velocity = np.asarray(table["velocity_kms"], dtype=float)
        cloud_spectra = np.asarray(table["dsigma_dv"], dtype=float)
        states = tuple(str(value) for value in table["state_labels"].tolist())
        lines = tuple(str(value) for value in table["line_keys"].tolist())
    baseline_index = states.index("baseline")
    mol_ct_index = states.index("mol_ct")

    curves: dict[str, dict] = {species: {} for species in ("cii", "halpha", "hi21")}

    cii_low_saha = _read_curve(
        args.cii_low_saha, "CPLUS_SAHA_TQK_LT3000", los_group=True,
    )
    cii_low_dsp = _read_curve(
        args.cii_low_despotic,
        "CPLUS_DESPOTIC_TQK_LT3000_DIAGNOSTIC",
        los_group=False,
    )
    cii_high_saha = _read_curve(
        args.cii_high_models, "CPLUS_SAHA_TQK_GE3000", los_group=True,
    )
    cii_high_dsp = _read_curve(
        args.cii_high_models, "CPLUS_DESPOTIC_TQK_GE3000", los_group=True,
    )
    curves["cii"]["quokka"] = (cii_low_saha, cii_high_saha)
    curves["cii"]["despotic"] = (cii_low_dsp, cii_high_dsp)

    with np.load(halpha_path, allow_pickle=False) as table:
        halpha_velocity = np.asarray(table["velocity_kms"], dtype=float)
        halpha_spectra = np.asarray(table["dsigma_dv"], dtype=float)
    curves["halpha"]["quokka"] = tuple(
        (halpha_velocity, halpha_spectra[0, branch]) for branch in (0, 1)
    )
    curves["halpha"]["despotic"] = tuple(
        (halpha_velocity, halpha_spectra[1, branch]) for branch in (0, 1)
    )

    hi_groups = {
        "quokka": ("HI_QUOKKA_LOW", "HI_QUOKKA_HIGH"),
        "despotic": ("HI_DESPOTIC_LOW", "HI_DESPOTIC_HIGH"),
    }
    for model, groups in hi_groups.items():
        curves["hi21"][model] = tuple(
            _read_curve(args.hi_models, group, los_group=True)
            for group in groups
        )

    for species in curves:
        line_index = lines.index(species)
        curves[species]["cloudy_baseline"] = tuple(
            (cloud_velocity, cloud_spectra[line_index, baseline_index, branch])
            for branch in (0, 1)
        )
        curves[species]["cloudy_mol_ct"] = tuple(
            (cloud_velocity, cloud_spectra[line_index, mol_ct_index, branch])
            for branch in (0, 1)
        )
    return curves


def _validate_halpha_known_branches(
    args: argparse.Namespace,
    curves: dict[str, dict],
) -> None:
    pipeline_low = _read_curve(
        args.halpha_pipeline, "HALPHA_PIPELINE_LOW", los_group=True,
    )
    pipeline_high = _read_curve(
        args.halpha_pipeline, "HALPHA_PIPELINE_HIGH", los_group=True,
    )
    dsp_velocity, dsp_low = curves["halpha"]["despotic"][0]
    qk_velocity, qk_high = curves["halpha"]["quokka"][1]
    expected_dsp = _orient_like(pipeline_low[0], pipeline_low[1], dsp_velocity)
    expected_qk = _orient_like(pipeline_high[0], pipeline_high[1], qk_velocity)
    # The new accumulator works directly on an ascending velocity grid while
    # the historical pipeline cache was built in frequency space and then
    # reversed.  Their Gaussian integrals agree to sub-ppm precision; allow
    # that harmless coordinate-rounding difference while still catching any
    # physical-field or normalization mismatch.
    np.testing.assert_allclose(dsp_low, expected_dsp, rtol=1.0e-6, atol=0.0)
    np.testing.assert_allclose(qk_high, expected_qk, rtol=1.0e-6, atol=0.0)
    print("Validated Halpha: DESPOTIC-low and QUOKKA-high match pipeline caches")


def _plot_species(
    species: str,
    curves: dict[str, tuple],
    output: Path,
    *,
    omit_low_quokka: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    active_models = {
        branch: tuple(
            model for model in MODEL_KEYS
            if not (omit_low_quokka and branch == 0 and model == "quokka")
        )
        for branch in (0, 1)
    }
    all_peaks = [
        float(np.nanmax(curves[model][branch][1]))
        for branch in (0, 1) for model in active_models[branch]
    ]
    shared_max = max(all_peaks)
    if not np.isfinite(shared_max) or shared_max <= 0.0:
        raise ValueError(f"{species} curves contain no positive spectrum")

    for branch, axis in enumerate(axes):
        branch_peaks = []
        for model in active_models[branch]:
            velocity, spectrum = curves[model][branch]
            color, linestyle = MODEL_STYLES[model]
            branch_peaks.append((float(np.nanmax(spectrum)), model))
            axis.plot(
                velocity,
                spectrum,
                color=color,
                linestyle=linestyle,
                linewidth=1.6,
                drawstyle="steps-mid",
                label=MODEL_LABELS[species][model],
            )
        axis.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
        axis.set_xlabel(r"Velocity [km s$^{-1}$]")
        axis.set_ylabel(dsigma_dv_ylabel(DSIGMA_DV_UNIT))
        axis.set_title(
            r"$T_{\rm QUOKKA}<3000\,$K"
            if branch == 0 else r"$T_{\rm QUOKKA}\geq3000\,$K"
        )
        axis.set_ylim(0.0, 1.05 * shared_max)
        axis.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        axis.legend(fontsize=8.1, frameon=False)
        axis.ticklabel_format(
            style="sci", axis="y", scilimits=(0, 0), useMathText=True,
        )
        axis.tick_params(axis="y", labelleft=True)

        ordered = sorted(branch_peaks, reverse=True)
        dominant_peak, dominant_model = ordered[0]
        second_peak = ordered[1][0]
        zoom_mode = FORCED_ZOOMS.get((species, branch))
        if omit_low_quokka and branch == 0 and species in ("cii", "halpha"):
            zoom_mode = "all_models"
        if (
            zoom_mode is not None
            or (second_peak > 0.0 and dominant_peak > 5.0 * second_peak)
        ):
            if zoom_mode == "all_models":
                zoom_models = active_models[branch]
                zoom_peak = dominant_peak
                zoom_title = f"zoom: all {len(zoom_models)} models"
            else:
                zoom_models = tuple(
                    model for model in active_models[branch]
                    if model != dominant_model
                )
                zoom_peak = second_peak
                zoom_title = f"zoom: other {len(zoom_models)} models"
            if zoom_peak <= 0.0:
                continue
            inset = axis.inset_axes([0.08, 0.52, 0.48, 0.40])
            for model in zoom_models:
                velocity, spectrum = curves[model][branch]
                color, linestyle = MODEL_STYLES[model]
                inset.plot(
                    velocity,
                    spectrum,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.2,
                    drawstyle="steps-mid",
                )
            inset.set_xlim(-35.0, 35.0)
            inset.set_ylim(0.0, 1.08 * zoom_peak)
            inset.ticklabel_format(
                style="sci", axis="y", scilimits=(0, 0), useMathText=True,
            )
            inset.tick_params(labelsize=7)
            inset.grid(True, alpha=0.2, linestyle="--", linewidth=0.4)
            inset.set_title(zoom_title, fontsize=8)

    fig.suptitle(
        f"Comparison of the {SPECIES_LABELS[species]} spectrum, "
        r"LOS y, $R=\infty$"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    output = Path(cfg.OUTPUT_DIR)
    task = output / "task_intermediates"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(cfg.YT_DATASET_PATH))
    parser.add_argument("--despotic-table", type=Path, default=Path(cfg.DESPOTIC_TABLE_PATH))
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument(
        "--cloudy-spectra", type=Path,
        default=output / "cloudy_line_physics_ablation_spectra_Rinf_linear_failure_fill.npz",
    )
    parser.add_argument(
        "--cii-low-saha", type=Path,
        default=task / "Build_CplusColdSahaComparison_2e214350.h5",
    )
    parser.add_argument(
        "--cii-low-despotic", type=Path,
        default=task / "Build_CplusLowCloudyComparison_c389c117.h5",
    )
    parser.add_argument(
        "--cii-high-models", type=Path,
        default=task / "Build_CplusHighModelComparison_3a471490.h5",
    )
    parser.add_argument(
        "--halpha-pipeline", type=Path,
        default=task / "Build_HalphaCloudyComparison_50c67088.h5",
    )
    parser.add_argument(
        "--hi-models", type=Path,
        default=task / "Build_HICloudyComparison_9bca5349.h5",
    )
    parser.add_argument("--column-cache", type=Path, default=None)
    parser.add_argument("--dvdr-cache", type=Path, default=None)
    parser.add_argument("--tdsp-cache", type=Path, default=None)
    parser.add_argument("--slab-nz", type=int, default=32)
    parser.add_argument("--cell-chunk", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--force-halpha", action="store_true")
    parser.add_argument(
        "--halpha-only",
        action="store_true",
        help="rebuild only the QUOKKA/DESPOTIC H-alpha diagnostic intermediate",
    )
    parser.add_argument("--force-plots", action="store_true")
    parser.add_argument(
        "--omit-low-quokka",
        action="store_true",
        help="omit the QUOKKA-mu/Saha curve only for T_QUOKKA < 3000 K",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for name in (
        "dataset", "despotic_table", "output_dir", "cloudy_spectra",
        "cii_low_saha", "cii_low_despotic", "cii_high_models",
        "halpha_pipeline", "hi_models",
    ):
        value = getattr(args, name)
        setattr(args, name, value.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.slab_nz <= 0 or args.cell_chunk <= 0 or args.workers <= 0:
        raise ValueError("slab, chunk, and worker counts must be positive")

    if args.halpha_only:
        _build_halpha_diagnostics(args)
        return

    filenames = (
        NO_LOW_QUOKKA_OUTPUT_FILENAMES
        if args.omit_low_quokka else OUTPUT_FILENAMES
    )
    outputs = {
        species: args.output_dir / filename
        for species, filename in filenames.items()
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.force_plots:
        raise FileExistsError(
            "refusing to overwrite existing figures; pass --force-plots:\n"
            + "\n".join(str(path) for path in existing)
        )

    halpha_path = _build_halpha_diagnostics(args)
    curves = _load_curves(args, halpha_path)
    _validate_halpha_known_branches(args, curves)
    for species in ("cii", "halpha", "hi21"):
        _plot_species(
            species,
            curves[species],
            outputs[species],
            omit_low_quokka=args.omit_low_quokka,
        )

    provenance_name = (
        "CloudyBaseline_MolCT_DESPOTIC_noLowQUOKKA_Tsplit_zooms_Rinf.json"
        if args.omit_low_quokka
        else "four_model_CloudyBaseline_MolCT_Tsplit_targeted_zooms_Rinf.json"
    )
    provenance_path = args.output_dir / provenance_name
    provenance = {
        "models": {
            species: MODEL_LABELS[species] for species in MODEL_LABELS
        },
        "temperature_split_K": REGIME_SPLIT_K,
        "los": "y",
        "spectral_resolution_R": "infinity",
        "omitted_models": (
            {"T_QUOKKA_lt_3000K": ["quokka"]}
            if args.omit_low_quokka else {}
        ),
        "cloudy_failure_policy": (
            "uses the previously generated bracketed linear-failure-fill spectra"
        ),
        "inputs": {
            "cloudy_spectra": str(args.cloudy_spectra),
            "halpha_diagnostics": str(halpha_path),
            "cii_low_saha": str(args.cii_low_saha),
            "cii_low_despotic": str(args.cii_low_despotic),
            "cii_high_models": str(args.cii_high_models),
            "hi_models": str(args.hi_models),
        },
        "figures": {species: str(path) for species, path in outputs.items()},
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved: {provenance_path}")


if __name__ == "__main__":
    main()
