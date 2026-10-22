"""Draw all standard emission figures from Julia's prepared saved products.

Run prepare_results.py once after numerical processing. This renderer opens
only the prepared directory; snapshots, tables, and raw products are unnecessary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from quokka2s.emission_results import read_emission_results
from quokka2s.plot_emission_results import draw_emission_products


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("products", type=Path, help="Julia result directory or its prepared/ directory")
    parser.add_argument("--output", type=Path, help="Untitled figure directory; default: RESULT/figures")
    parser.add_argument("--format", choices=("png", "pdf"), nargs="+", default=["png"])
    parser.add_argument("--style", choices=("titled", "untitled"), nargs="+", default=["titled"])
    parser.add_argument("--image-downsample-factor", type=int, default=1)
    parser.add_argument("--raw-luminosity", action="store_true", help="Select saved erg/s/(km/s) instead of light per projected cm^2")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    directory = args.products.expanduser().resolve()
    prepared = directory if (directory / "preparation.json").exists() else directory / "prepared"
    manifest_path = prepared / "preparation.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prepare saved figure arrays first with julia/prepare_results.py {directory}")
    manifest = json.loads(manifest_path.read_text())
    if not manifest["processing_complete"]:
        raise ValueError("Saved figure preparation is incomplete")
    image_name = "images.npz"
    if args.image_downsample_factor != 1:
        image_name = f"images_factor_{args.image_downsample_factor}.npz"
    for name in (image_name, "spectra.npz", "phase_velocity.npz", "emission_report.json"):
        expected_hash = manifest["prepared_file_sha256"].get(name)
        if expected_hash is None:
            raise ValueError(f"{name} was not prepared; rerun prepare_results.py with the requested metadata and image factor")
        with (prepared / name).open("rb") as source:
            actual_hash = hashlib.file_digest(source, "sha256").hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Saved prepared product has changed: {name}")
    output = prepared.parent / "figures" if args.output is None else args.output.expanduser().resolve()
    print(f"Plot: reading prepared products from {prepared}", flush=True)
    products = read_emission_results(
        directory=prepared,
        image_downsample_factor=args.image_downsample_factor,
    )
    paths = draw_emission_products(
        products=products,
        output_dir=output,
        per_projected_area=not args.raw_luminosity,
        formats=tuple(dict.fromkeys(args.format)),
        styles=tuple(dict.fromkeys(args.style)),
    )
    file_count = sum(len(files) for groups in paths.values() for files in groups.values())
    print(f"Plot complete: {file_count} files in {time.perf_counter() - started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
