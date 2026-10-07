"""Prepare figure data from saved products without reading a snapshot.

Future process runs save the normalized spectra and image limits directly.
Use this command to prepare existing products or an optional image resolution:
python -m quokka2s.prepare_emission_results --image-downsample-factor 2
The native images.npz is retained; the coarse image is images_factor_2.npz.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np

from quokka2s.paths import resolve_path
from quokka2s.products.line_luminosity_images import (
    add_image_display_fields,
    prepare_coarser_image,
)
from quokka2s.products.profile_preparation import (
    add_gas_phase_display_fields,
    add_spectral_display_fields,
)
from quokka2s.run_settings import DEFAULT_PLOT_CONFIG, load_plot_config


def read_product_arrays(path: Path) -> dict[str, np.ndarray]:
    """Read all named arrays from one processed NPZ; no scientific lookup."""
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def save_prepared_arrays(path: Path, payload: dict[str, np.ndarray]) -> None:
    """Replace one product only after its complete NPZ has been written."""
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as file:
        temporary = Path(file.name)
    try:
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_saved_emission_results(directory: Path, image_downsample_factor: int = 1) -> None:
    """Add prepared figure arrays to native saved products and optional image.

    directory contains images.npz, spectra.npz and phase_velocity.npz.
    Raw luminosities, histograms and moments remain unchanged. All preparation
    uses these small accumulated arrays; it never reads simulation cells.
    A factor of 2 also writes the separately saved 2x2-summed luminosity image.
    """
    images = read_product_arrays(path=directory / "images.npz")
    spectra = read_product_arrays(path=directory / "spectra.npz")
    gas_phases = read_product_arrays(path=directory / "phase_velocity.npz")

    add_image_display_fields(payload=images)
    add_spectral_display_fields(payload=spectra)
    add_gas_phase_display_fields(payload=gas_phases)
    coarse_images = None
    if image_downsample_factor != 1:
        coarse_images = prepare_coarser_image(
            payload=images,
            factor=image_downsample_factor,
        )

    save_prepared_arrays(path=directory / "images.npz", payload=images)
    save_prepared_arrays(path=directory / "spectra.npz", payload=spectra)
    save_prepared_arrays(path=directory / "phase_velocity.npz", payload=gas_phases)
    if coarse_images is not None:
        save_prepared_arrays(
            path=directory / f"images_factor_{image_downsample_factor}.npz",
            payload=coarse_images,
        )


def main(argv=None) -> None:
    """Use plot YAML's products directory, or an explicit --products path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PLOT_CONFIG)
    parser.add_argument("--products", type=Path)
    parser.add_argument("--image-downsample-factor", type=int, default=1)
    args = parser.parse_args(argv)
    if args.products is None:
        directory = load_plot_config(args.config).products
    else:
        directory = resolve_path(args.products)
    prepare_saved_emission_results(
        directory=directory,
        image_downsample_factor=args.image_downsample_factor,
    )
    print(f"Prepared saved figure data in {directory}")


if __name__ == "__main__":
    main()
