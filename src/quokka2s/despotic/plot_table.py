"""Draw DESPOTIC heatmaps from prepared numerical products only."""
from __future__ import annotations

import argparse
from pathlib import Path

from quokka2s.despotic.table_figure_data import (
    DEFAULT_FIGURE_DATA_PATH,
    read_table_figure_data,
)
from quokka2s.despotic.table_plots import plot_table_overview
from quokka2s.paths import resolve_path


def default_output_directory(source_table: str) -> Path:
    """Retain the established TablePlots_<table-parent> output naming."""
    tag = Path(source_table).parent.name
    for prefix in ("output_tables_3D_", "output_tables_"):
        if tag.startswith(prefix):
            tag = tag[len(prefix):]
            break
    return Path(f"TablePlots_{tag}")


def main(argv=None) -> None:
    """Load a prepared NPZ and draw its selected slices, without table/sample reads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_FIGURE_DATA_PATH,
        help=f"Prepared figure NPZ (default: {DEFAULT_FIGURE_DATA_PATH})",
    )
    parser.add_argument(
        "-o", "--out-root",
        type=Path,
        help="Output directory; default retains the source table's TablePlots_<parent> name",
    )
    args = parser.parse_args(argv)
    args.data = resolve_path(args.data)
    figure_data = read_table_figure_data(path=args.data)
    output_root = args.out_root
    if output_root is None:
        output_root = default_output_directory(
            source_table=str(figure_data["source_table_path"]),
        )
    output_root = resolve_path(output_root)
    tokens = tuple(str(token) for token in figure_data["field_tokens"])
    indices = figure_data["dvdr_indices"]
    axis_size = int(figure_data["dvdr_axis_size"])
    print(f"Drawing {len(tokens)} fields × {len(indices)} dVdr slices into {output_root}/")

    import matplotlib.pyplot as plt

    for slice_index, original_index in enumerate(indices):
        gradient_s = figure_data["dvdr_values_s"][slice_index]
        figures = plot_table_overview(
            figure_data=figure_data,
            fields=tokens,
            ncols=3,
            figsize=(14, 10),
            separate=True,
            slice_index=slice_index,
        )
        output_directory = output_root / f"dVdr_{gradient_s:.2e}"
        output_directory.mkdir(parents=True, exist_ok=True)
        for token, figure in zip(tokens, figures):
            for axis in figure.axes:
                title = axis.get_title()
                if title:
                    axis.set_title(
                        f"{title}  |  frame {original_index + 1:02d}/{axis_size}  "
                        f"dVdr = {gradient_s:.2e} s$^{{-1}}$",
                    )
            filename = token.replace(":", "_") + ".png"
            figure.savefig(output_directory / filename, dpi=200)
            plt.close(figure)
        print(f"  done dVdr = {gradient_s:.2e} s^-1  →  {output_directory}/")


if __name__ == "__main__":
    main()
