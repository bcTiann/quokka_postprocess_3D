"""Draw the existing 8 eV guide on saved Cloudy radiation figures."""

from matplotlib.axes import Axes


def add_eight_ev_marker(axis: Axes, *, marker_Ryd: float, label_line: bool = False) -> None:
    """Draw the existing 8 eV guide at its position on a Rydberg x axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Radiation plot with photon energy [Ryd] on its x axis.
    marker_Ryd : float
        Saved eight_ev_marker_Ryd value from the prepared radiation NPZ.
    label_line : bool
        True adds the existing rotated label at 4 percent of the axis height.

    Returns
    -------
    None
        Adds the guide and optional label to axis using the original styles.

    Example
    -------
    add_eight_ev_marker(axis=component_axis, marker_Ryd=saved_marker,
                       label_line=True) labels one panel.
    """
    axis.axvline(marker_Ryd, color="0.35", linestyle=":", linewidth=1.2)
    if label_line:
        axis.text(
            marker_Ryd,
            0.04,
            "8 eV",
            rotation=90,
            transform=axis.get_xaxis_transform(),
            ha="right",
            va="bottom",
            color="0.35",
            fontsize=8,
        )
