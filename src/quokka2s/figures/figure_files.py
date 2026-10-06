"""Save plot files in separate PNG and PDF subdirectories."""
from pathlib import Path


def save_figure_formats(
    figure,
    stem: Path,
    formats: tuple[str, ...],
    *,
    bbox_inches: str | None = "tight",
) -> list[Path]:
    """Save one figure under a subdirectory for each requested format.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Completed figure from an image, spectrum or gas-phase plot.
    stem : pathlib.Path
        Output directory and filename without an extension.
    formats : tuple of str
        Extensions such as ("png", "pdf").
    bbox_inches : str or None
        "tight" trims empty margins; None retains the figure's fixed layout.

    Returns
    -------
    list of pathlib.Path
        Saved files, in format order.

    Examples
    --------
    stem = Path("figures/spectrum_halpha") and formats = ("png", "pdf")
    write figures/png/spectrum_halpha.png and figures/pdf/spectrum_halpha.pdf.
    """
    paths = []
    for extension in formats:
        format_directory = stem.parent / extension
        format_directory.mkdir(parents=True, exist_ok=True)
        path = format_directory / stem.with_suffix("." + extension).name
        figure.savefig(path, dpi=200, bbox_inches=bbox_inches)
        paths.append(path)
    return paths
