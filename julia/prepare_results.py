"""Prepare the standard figures from Julia's saved numerical products.

This stage reads only products.npz and report.json. It converts array order,
prepares display values, and saves the three products consumed by the existing
Python renderers. Snapshot cells and emission tables are never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from unyt import unyt_quantity

from quokka2s.prepare_emission_results import save_prepared_arrays
from quokka2s.products.line_luminosity_images import (
    add_image_display_fields,
    prepare_coarser_image,
)
from quokka2s.products.profile_preparation import (
    add_gas_phase_display_fields,
    add_spectral_display_fields,
)


def file_sha256(path: Path) -> str:
    """Hash a saved product without retaining its compressed contents in memory."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def checked_array(saved, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Read one numeric product and reject an unexpected native axis shape."""
    values = saved[name]
    if values.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {values.shape}")
    return values


def prepare_run_metadata(report: dict, *, jobname: str | None) -> dict:
    """Prepare title identity and simulation age from saved raw metadata only.

    time_s is the Mera header time multiplied by its physical time unit.
    unyt uses the same seconds-to-Myr convention as the reference yt pipeline.
    Older diagnostic reports can lack time or jobname; missing values remain
    explicit rather than being inferred from output or snapshot parent folders.
    """
    saved_jobname = report.get("jobname")
    for value in (saved_jobname, jobname):
        if value is not None and (
            not isinstance(value, str) or not value.strip()
            or value in (".", "..") or "/" in value or "\\" in value
        ):
            raise ValueError("jobname must be a nonempty label without path separators")
    if saved_jobname is not None and jobname is not None and saved_jobname != jobname:
        raise ValueError("--jobname conflicts with the saved jobname")
    selected_jobname = saved_jobname if saved_jobname is not None else jobname
    dataset_name = Path(report["dataset"]).name
    snapshot_name = report.get("snapshot_name", dataset_name)
    if snapshot_name != dataset_name:
        raise ValueError("Saved snapshot_name conflicts with the saved dataset path")
    time_seconds = report.get("time_s")
    simulation_time_myr = None
    if time_seconds is not None:
        time_seconds = float(time_seconds)
        if not np.isfinite(time_seconds) or time_seconds < 0:
            raise ValueError("Saved time_s must be finite and nonnegative")
        simulation_time_myr = float(unyt_quantity(time_seconds, "s").to_value("Myr"))
    return {
        "jobname": selected_jobname,
        "snapshot_name": snapshot_name,
        "simulation_time_myr": simulation_time_myr,
        "time_s": time_seconds,
        "time_conversion": "unyt physical seconds to Myr (million years)",
    }


def prepare_results(
    directory: Path,
    *,
    output: Path | None = None,
    image_downsample_factor: int = 1,
    jobname: str | None = None,
) -> Path:
    """Save standard image, spectrum and gas-phase figure arrays.

    Native Julia images are (x, y, line, dust), spectra are
    (channel, line, dust, regime), and gas histograms are (channel, phase).
    Prepared files use the existing Python reader's documented axis orders.
    Optional coarse pixels are summed and saved separately, never averaged.
    The source numerical products and source report are left unchanged.
    """
    started = time.perf_counter()
    directory = directory.expanduser().resolve()
    output = directory / "prepared" if output is None else output.expanduser().resolve()
    report_path = directory / "report.json"
    products_path = directory / "products.npz"
    report = json.loads(report_path.read_text())
    run_metadata = prepare_run_metadata(report, jobname=jobname)
    if not (report.get("raw_processing_complete") or report.get("processing_complete")):
        raise ValueError("Julia's raw numerical calculation is not complete")
    if report["axis_order"]["images"] != ["x", "y", "line", "dust"]:
        raise ValueError("Unsupported Julia image axis order")
    if report["axis_order"]["spectra"] != ["channel", "line", "dust", "regime"]:
        raise ValueError("Unsupported Julia spectrum axis order")

    line_keys = np.asarray(report["physics"]["line_keys"])
    dust_keys = np.asarray(report["dust_states"])
    regime_keys = np.asarray(report["regimes"])
    phase_keys = np.asarray([*report["physics"]["phase_keys"], "total"])
    nx, ny, _ = report["shape"]
    line_count = len(line_keys)
    dust_count = len(dust_keys)
    regime_count = len(regime_keys)
    phase_count = len(phase_keys)
    if list(dust_keys) != ["intrinsic", "attenuated"]:
        raise ValueError("Expected intrinsic and attenuated dust states")
    if list(regime_keys) != ["cold", "hot"]:
        raise ValueError("Expected cold and hot emission regimes")
    print(f"Preparing saved Julia products: {directory}", flush=True)

    with np.load(products_path, allow_pickle=False) as saved:
        velocity_edges = saved["velocity_edges_kms"]
        channel_count = len(velocity_edges) - 1
        velocity_centers = checked_array(saved, "velocity_centers_kms", (channel_count,))
        area = float(checked_array(saved, "projected_area_cm2", (1,))[0])
        if not np.isfinite(area) or area <= 0:
            raise ValueError("Projected area must be finite and positive [cm^2]")
        if not np.all(np.isfinite(velocity_edges)) or not np.all(np.diff(velocity_edges) > 0):
            raise ValueError("Saved velocity edges must be finite and increasing")

        images = {
            "line_keys": line_keys,
            "dust_state_keys": dust_keys,
            "axis_order": np.asarray("dust_state,line,x_pixel,y_pixel"),
            "native_xy_shape": np.asarray([nx, ny]),
            "image_shape": np.asarray([nx, ny]),
            "image_xy_origin": checked_array(saved, "image_xy_origin", (2,)),
            "line_luminosity_image_erg_s": checked_array(
                saved, "images_luminosity_erg_s", (nx, ny, line_count, dust_count),
            ).transpose(3, 2, 0, 1),
            "total_luminosity_erg_s": checked_array(
                saved, "image_total_luminosity_erg_s", (line_count, dust_count),
            ).T,
            "x_edges_kpc": checked_array(saved, "x_edges_kpc", (nx + 1,)),
            "y_edges_kpc": checked_array(saved, "y_edges_kpc", (ny + 1,)),
            "processing_complete": np.asarray(True),
            "full_snapshot": np.asarray(report["full_snapshot"]),
        }
        spectra = {
            "line_keys": line_keys,
            "dust_state_keys": dust_keys,
            "regime_keys": regime_keys,
            "axis_order": np.asarray("dust_state,line,temperature_regime,velocity_channel"),
            "velocity_edges_kms": velocity_edges,
            "velocity_kms": velocity_centers,
            "projected_area_cm2": np.asarray(area),
            "dL_dv_erg_s_per_kms": checked_array(
                saved, "spectra_dL_dv_erg_s_per_kms",
                (channel_count, line_count, dust_count, regime_count),
            ).transpose(2, 1, 3, 0),
            "total_dL_dv_erg_s_per_kms": checked_array(
                saved, "spectra_total_dL_dv_erg_s_per_kms",
                (channel_count, line_count, dust_count),
            ).transpose(2, 1, 0),
        }
        for name in (
            "line_centroid_window_kms", "line_sigma_window_kms",
            "line_centroid_full_kms", "line_sigma_full_kms",
        ):
            spectra[name] = checked_array(saved, name, (line_count, dust_count)).T
        for name in ("line_centroid_window_by_regime_kms", "line_sigma_window_by_regime_kms"):
            spectra[name] = checked_array(
                saved, name, (line_count, dust_count, regime_count),
            ).transpose(1, 0, 2)
        phases = {
            "phase_keys": phase_keys,
            "phase_boundaries_K": checked_array(saved, "phase_boundaries_K", (phase_count - 2,)),
            "velocity_edges_kms": velocity_edges,
            "histogram_mass_g": checked_array(
                saved, "phase_histogram_mass_g", (channel_count, phase_count),
            ).T,
            "sigma_about_global_mean_kms": checked_array(
                saved, "phase_sigma_about_global_mean_kms", (phase_count,),
            ),
            "sigma_internal_kms": checked_array(saved, "phase_sigma_internal_kms", (phase_count,)),
        }

    for values in (
        images["line_luminosity_image_erg_s"], spectra["dL_dv_erg_s_per_kms"],
        spectra["total_dL_dv_erg_s_per_kms"], phases["histogram_mass_g"],
    ):
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Accumulated light and mass must be finite and nonnegative")
    if not np.allclose(
        images["line_luminosity_image_erg_s"].sum(axis=(-2, -1)),
        images["total_luminosity_erg_s"], rtol=1e-10, atol=0,
    ):
        raise ValueError("Saved images do not conserve the recorded line luminosities")
    if not np.allclose(
        spectra["dL_dv_erg_s_per_kms"].sum(axis=2),
        spectra["total_dL_dv_erg_s_per_kms"], rtol=1e-10, atol=0,
    ):
        raise ValueError("Saved cold/hot spectra do not sum to their recorded totals")
    add_image_display_fields(payload=images)
    add_spectral_display_fields(payload=spectra)
    add_gas_phase_display_fields(payload=phases)
    payloads = {"images.npz": images, "spectra.npz": spectra, "phase_velocity.npz": phases}
    if image_downsample_factor != 1:
        payloads[f"images_factor_{image_downsample_factor}.npz"] = prepare_coarser_image(
            payload=images,
            factor=image_downsample_factor,
        )

    manifest_path = output / "preparation.json"
    if output.exists() and any(output.iterdir()):
        if not manifest_path.exists():
            raise FileExistsError(f"Refusing to replace unrelated files in {output}")
        previous = json.loads(manifest_path.read_text())
        if previous.get("implementation") != "Julia saved-product preparation":
            raise ValueError(f"Unrecognized preparation directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "implementation": "Julia saved-product preparation",
        "schema_version": 2,
        "processing_complete": False,
        "full_snapshot": report["full_snapshot"],
        "source_products": str(products_path),
        "source_products_sha256": file_sha256(products_path),
        "source_report_sha256": file_sha256(report_path),
        "dataset": report["dataset"],
        "shape": report["shape"],
        "line_keys": line_keys.tolist(),
        "image_downsample_factor": image_downsample_factor,
        "source_report": report,
        "run_metadata": run_metadata,
    }
    # Mark an interrupted preparation unusable while keeping raw products intact.
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for name, payload in payloads.items():
        save_prepared_arrays(path=output / name, payload=payload)
        print(f"  Prepared {name}", flush=True)
    metadata_path = output / "emission_report.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n")
    prepared_names = [*payloads, "emission_report.json"]
    manifest["prepared_file_sha256"] = {name: file_sha256(output / name) for name in prepared_names}
    manifest["processing_complete"] = True
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Preparation complete: {output} ({manifest['elapsed_seconds']:.2f} s)", flush=True)
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("products", type=Path, help="Julia directory containing products.npz and report.json")
    parser.add_argument("--output", type=Path, help="Prepared directory; default: PRODUCTS/prepared")
    parser.add_argument("--image-downsample-factor", type=int, default=1)
    parser.add_argument("--jobname", help="Add an explicit job label when missing from an older saved report")
    args = parser.parse_args(argv)
    prepare_results(
        directory=args.products,
        output=args.output,
        image_downsample_factor=args.image_downsample_factor,
        jobname=args.jobname,
    )


if __name__ == "__main__":
    main()
