"""Plot saved line-luminosity images and integrated spectra.

The plotting stage reads only ``images.npz`` and ``spectra.npz``. It never
opens a simulation snapshot or recalculates an emissivity or line profile.
Run it with ``quokka2s plot --config emission_plot.yaml``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np

from .emission_config import load_plot_config

VARIANT_KEYS = ("intrinsic", "attenuated")
SHORT_VELOCITY_LIMITS_KMS = (-50.0, 50.0)
CIII_VELOCITY_LIMITS_KMS = (-200.0, 200.0)


def _read_npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = set(required) - set(data.files)
        if missing:
            raise ValueError(f"{path.name} is missing {', '.join(sorted(missing))}")
        return {key: np.array(data[key], copy=True) for key in required}


def _strings(array: np.ndarray, name: str) -> tuple[str, ...]:
    if array.ndim != 1 or array.dtype.kind not in ("U", "S"):
        raise ValueError(f"{name} must be a one-dimensional string array")
    values = tuple(value.decode() if isinstance(value, bytes) else str(value)
                   for value in array)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must be nonempty and unique")
    if any(re.fullmatch(r"[a-zA-Z0-9_]+", value) is None for value in values):
        raise ValueError(f"{name} contains an invalid key")
    return values


def _edges(array: np.ndarray, size: int, name: str) -> np.ndarray:
    edges = np.asarray(array, dtype=float)
    if (edges.shape != (size + 1,) or not np.isfinite(edges).all()
            or not np.all(np.diff(edges) > 0)):
        raise ValueError(f"{name} must have {size + 1} increasing finite edges")
    return edges


def load_emission_products(products_dir: str | Path) -> tuple[dict, dict]:
    """Load and validate the two portable, plotting-only output bundles."""
    directory = Path(products_dir)
    images = _read_npz(directory / "images.npz", (
        "line_keys", "variant_keys", "line_luminosity_image_erg_s",
        "total_luminosity_erg_s", "x_edges_kpc", "y_edges_kpc",
        "input_fingerprint_sha256", "full_snapshot",
    ))
    spectra = _read_npz(directory / "spectra.npz", (
        "line_keys", "variant_keys", "regime_keys", "velocity_edges_kms",
        "velocity_kms", "dL_dv_erg_s_per_kms", "projected_area_cm2",
        "input_luminosity_erg_s", "input_fingerprint_sha256", "full_snapshot",
    ))
    image_keys = _strings(images["line_keys"], "image line_keys")
    spectrum_keys = _strings(spectra["line_keys"], "spectrum line_keys")
    if image_keys != spectrum_keys:
        raise ValueError("Image and spectrum line keys must have the same order")
    for name, bundle in (("image", images), ("spectrum", spectra)):
        digest = bundle["input_fingerprint_sha256"]
        full = bundle["full_snapshot"]
        if digest.shape != () or digest.dtype.kind not in ("U", "S"):
            raise ValueError(f"{name} input_fingerprint_sha256 must be a SHA-256 digest")
        digest_text = digest.item()
        if isinstance(digest_text, bytes):
            digest_text = digest_text.decode()
        if re.fullmatch(r"[0-9a-fA-F]{64}", str(digest_text)) is None:
            raise ValueError(f"{name} input_fingerprint_sha256 must be a SHA-256 digest")
        if full.shape != () or full.dtype.kind != "b":
            raise ValueError(f"{name} full_snapshot must be a Boolean scalar")
    if (not np.array_equal(images["input_fingerprint_sha256"], spectra["input_fingerprint_sha256"])
            or not np.array_equal(images["full_snapshot"], spectra["full_snapshot"])):
        raise ValueError("Image and spectrum products have different provenance")
    for name, bundle in (("image", images), ("spectrum", spectra)):
        if _strings(bundle["variant_keys"], f"{name} variant_keys") != VARIANT_KEYS:
            raise ValueError(f"{name} variants must be intrinsic, attenuated")
    if _strings(spectra["regime_keys"], "regime_keys") != (
            "T_QUOKKA_lt_3000K", "T_QUOKKA_ge_3000K"):
        raise ValueError("Expected cold and hot temperature regimes")

    image_values = np.asarray(images["line_luminosity_image_erg_s"], dtype=float)
    if (image_values.shape != (2, len(image_keys), 128, 128)
            or not np.isfinite(image_values).all() or np.any(image_values < 0)):
        raise ValueError("Images must be nonnegative with shape (variant,line,x=128,y=128)")
    images["line_luminosity_image_erg_s"] = image_values
    images["x_edges_kpc"] = _edges(images["x_edges_kpc"], 128, "x_edges_kpc")
    images["y_edges_kpc"] = _edges(images["y_edges_kpc"], 128, "y_edges_kpc")
    image_total = np.asarray(images["total_luminosity_erg_s"], dtype=float)
    if (image_total.shape != (2, len(image_keys)) or not np.isfinite(image_total).all()
            or np.any(image_total < 0)):
        raise ValueError("Image total_luminosity_erg_s must have shape (variant,line)")
    if not np.allclose(image_values.sum(axis=(-2, -1)), image_total, rtol=1e-12, atol=0):
        raise ValueError("Image pixels do not sum to saved image luminosities")

    velocity = np.asarray(spectra["velocity_kms"], dtype=float)
    if velocity.shape != (400,) or not np.isfinite(velocity).all():
        raise ValueError("Expected 400 finite velocity channels")
    edges = _edges(spectra["velocity_edges_kms"], 400, "velocity_edges_kms")
    if edges[0] != -200.0 or edges[-1] != 200.0:
        raise ValueError("Velocity edges must span exactly -200 to 200 km/s")
    if not np.allclose(velocity, (edges[:-1] + edges[1:]) / 2, rtol=0, atol=1e-10):
        raise ValueError("Velocity channels must be centres of their saved edges")
    if not np.allclose(np.diff(edges), np.diff(edges)[0], rtol=1e-12, atol=0):
        raise ValueError("Velocity channels must be uniform")
    values = np.asarray(spectra["dL_dv_erg_s_per_kms"], dtype=float)
    if (values.shape != (2, len(image_keys), 2, 400)
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("Spectra must be nonnegative with shape (variant,line,regime,400)")
    spectra["dL_dv_erg_s_per_kms"] = values
    spectrum_input = np.asarray(spectra["input_luminosity_erg_s"], dtype=float)
    if (spectrum_input.shape != (2, len(image_keys), 2)
            or not np.isfinite(spectrum_input).all() or np.any(spectrum_input < 0)):
        raise ValueError("Spectrum input_luminosity_erg_s must have shape (variant,line,regime)")
    if not np.allclose(image_total, spectrum_input.sum(axis=-1), rtol=1e-12, atol=0):
        raise ValueError("Image and spectrum input luminosities differ")
    area = np.asarray(spectra["projected_area_cm2"], dtype=float)
    if area.shape != () or not np.isfinite(area) or area <= 0:
        raise ValueError("projected_area_cm2 must be a finite positive scalar")
    return images, spectra


def _save_figure(figure, stem: Path, formats: tuple[str, ...]) -> list[Path]:
    paths = []
    for extension in formats:
        path = stem.with_suffix("." + extension)
        figure.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)
    return paths


def plot_emission_products(
    products_dir: str | Path,
    output_dir: str | Path,
    *,
    per_projected_area: bool = True,
    formats: tuple[str, ...] = ("png", "pdf"),
    allow_partial: bool = False,
) -> dict[str, list[Path]]:
    """Make one spectrum per line and one image per line/variant, except H I.

    Intrinsic and dust-attenuated images of a given line share colour limits.
    Spectrum views use the manuscript's ±50 km/s range except for C III, but
    all saved channels remain intact. H I 21 cm has one curve because the dust
    prescription leaves that line unchanged.
    """
    if not formats or any(extension not in ("png", "pdf") for extension in formats):
        raise ValueError("formats must contain png and/or pdf")
    images, spectra = load_emission_products(products_dir)
    full_snapshot = bool(images["full_snapshot"])
    if not full_snapshot and not allow_partial:
        raise ValueError("Partial diagnostic products require allow_partial=True")
    diagnostic_suffix = "_partial_diagnostic" if not full_snapshot else ""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    spectrum_paths: list[Path] = []
    keys = _strings(images["line_keys"], "line_keys")
    image_values = images["line_luminosity_image_erg_s"]
    velocity = spectra["velocity_kms"]
    spectrum_values = spectra["dL_dv_erg_s_per_kms"]
    area = float(spectra["projected_area_cm2"]) if per_projected_area else 1.0
    ylabel = (r"$d\Sigma_L/dv$ [erg s$^{-1}$ cm$^{-2}$ (km s$^{-1}$)$^{-1}$]"
              if per_projected_area else
              r"$dL/dv$ [erg s$^{-1}$ (km s$^{-1}$)$^{-1}$]")

    for line_index, key in enumerate(keys):
        pair = image_values[:, line_index]
        if key == "hi21" and not np.array_equal(pair[0], pair[1]):
            raise ValueError("H I 21 cm images differ between intrinsic and attenuated variants")
        positive = pair[pair > 0]
        norm = (LogNorm(vmin=float(positive.min()) / 2, vmax=float(positive.max()))
                if positive.size else None)
        image_variants = ((0, ""),) if key == "hi21" else tuple(enumerate(VARIANT_KEYS))
        for variant_index, variant in image_variants:
            figure, axis = plt.subplots(figsize=(5.2, 4.5))
            image = pair[variant_index].T  # saved x,y -> Matplotlib rows=y, columns=x
            if norm is None:
                axis.set_facecolor("white")
                axis.text(0.5, 0.5, "No emission", ha="center", va="center",
                          transform=axis.transAxes)
            else:
                mesh = axis.pcolormesh(images["x_edges_kpc"], images["y_edges_kpc"],
                                       np.ma.masked_less_equal(image, 0.0), cmap="magma",
                                       norm=norm, shading="flat")
            axis.set_xlabel("x [kpc]")
            axis.set_ylabel("y [kpc]")
            axis.set_xlim(images["x_edges_kpc"][[0, -1]])
            axis.set_ylim(images["y_edges_kpc"][[0, -1]])
            axis.set_aspect("equal")
            if norm is not None:
                figure.colorbar(mesh, ax=axis, label=r"Pixel luminosity [erg s$^{-1}$]")
            figure.tight_layout()
            image_stem = f"line_luminosity_{key}"
            if variant:
                image_stem += f"_{variant}"
            image_paths.extend(_save_figure(figure, output / (image_stem + diagnostic_suffix), formats))
            plt.close(figure)

        figure, axis = plt.subplots(figsize=(6.2, 3.5))
        intrinsic = spectrum_values[0, line_index].sum(axis=0) / area
        attenuated = spectrum_values[1, line_index].sum(axis=0) / area
        if key == "hi21":
            if not np.allclose(intrinsic, attenuated, rtol=1e-12, atol=0):
                plt.close(figure)
                raise ValueError("H I 21 cm differs between intrinsic and attenuated variants")
            axis.plot(velocity, intrinsic, color="#242424", lw=1.8)
        else:
            axis.plot(velocity, intrinsic, color="#242424", lw=1.5, ls="--",
                      label="Intrinsic")
            axis.plot(velocity, attenuated, color="#C53D46", lw=1.8,
                      label="Dust attenuated")
            axis.legend(frameon=False, fontsize=9)
        axis.set_xlim(*(CIII_VELOCITY_LIMITS_KMS if key.startswith("ciii_")
                        else SHORT_VELOCITY_LIMITS_KMS))
        axis.set_ylim(bottom=0)
        axis.set_xlabel(r"$v_z$ [km s$^{-1}$]")
        axis.set_ylabel(ylabel)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        axis.grid(alpha=0.18)
        figure.tight_layout()
        spectrum_paths.extend(_save_figure(figure, output / f"spectrum_{key}{diagnostic_suffix}", formats))
        plt.close(figure)
    return {"images": image_paths, "spectra": spectrum_paths}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog='quokka2s plot', description=__doc__)
    parser.add_argument('--config', required=True, type=Path,
                        help='YAML file containing the saved-products and figure paths')
    config_path = parser.parse_args(argv).config
    try:
        args = load_plot_config(config_path)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    plot_emission_products(args.products, args.output_dir,
                           per_projected_area=not args.raw_luminosity,
                           allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
