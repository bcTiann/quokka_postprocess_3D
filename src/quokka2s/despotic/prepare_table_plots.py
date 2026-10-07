"""Prepare selected DESPOTIC table heatmaps and contours without drawing."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from quokka2s.despotic.table_files import load_table
from quokka2s.despotic.table_figure_data import (
    DEFAULT_FIELDS,
    DEFAULT_FIGURE_DATA_PATH,
    prepare_table_figure_data,
)
from quokka2s.paths import resolve_path


def select_dvdr_indices(
    axis_size: int,
    *,
    explicit_indices: list[int] | None,
    all_slices: bool,
    slice_count: int,
) -> list[int]:
    """Choose explicit indices, all indices, or the existing evenly spaced set."""
    if explicit_indices is not None:
        return explicit_indices
    if all_slices:
        return list(range(axis_size))
    if slice_count <= 0:
        raise ValueError("n-slices must be positive")
    indices = np.linspace(0, axis_size - 1, slice_count).astype(int)
    return sorted(set(indices.tolist()))


def main(argv=None) -> None:
    """Read explicit table/sample inputs and save complete plotting arrays.

    --samples selects one NPY array with 2/3/4 columns: log10(nH), log10(NH),
    optional log10(dVdr), optional cell mass [g]. Without it no sampling overlay
    is prepared; environment variables and repository sample files are ignored.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        type=Path,
        required=True,
        help="DESPOTIC table NPZ to analyze",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        help="Explicit NPY array of logarithmic query coordinates and optional mass",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Prepare every dVdr slice")
    selection.add_argument("--indices", nargs="+", type=int, help="Original dVdr indices to prepare")
    parser.add_argument(
        "-n", "--n-slices",
        type=int,
        default=5,
        help="Number of evenly spaced slices, including endpoints (default: 5)",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_FIELDS,
        help="Table field tokens (default: the existing eleven temperature/species fields)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=DEFAULT_FIGURE_DATA_PATH,
        help=f"Prepared numerical NPZ (default: {DEFAULT_FIGURE_DATA_PATH})",
    )
    args = parser.parse_args(argv)
    # Keep the supplied table parent for the established default figure directory.
    source_table_path = str(args.table.expanduser())
    source_samples_path = "" if args.samples is None else str(args.samples.expanduser())
    args.table = resolve_path(args.table)
    args.output = resolve_path(args.output)
    if args.samples is not None:
        args.samples = resolve_path(args.samples)
    table = load_table(args.table)
    samples = None
    if args.samples is not None:
        samples = np.load(args.samples, allow_pickle=False)
    indices = select_dvdr_indices(
        axis_size=table.dVdr_values.size,
        explicit_indices=args.indices,
        all_slices=args.all,
        slice_count=args.n_slices,
    )
    payload = prepare_table_figure_data(
        table=table,
        field_tokens=args.fields,
        dvdr_indices=indices,
        samples=samples,
    )
    payload["source_table_path"] = np.asarray(source_table_path)
    payload["source_samples_path"] = np.asarray(source_samples_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"Prepared {len(args.fields)} fields × {len(indices)} dVdr slices: {args.output}")
    print(f"Failure display policy: {payload['failure_display_policy']}")


if __name__ == "__main__":
    main()
