"""Shared display settings for the existing Cloudy radiation figures."""

from matplotlib.axes import Axes
import numpy as np

from quokka2s.constants import EV_PER_RYD


RADIATION_X_MIN_EV = 7.0
RADIATION_X_MIN_RYD = RADIATION_X_MIN_EV / EV_PER_RYD
EIGHT_EV_RYD = 8.0 / EV_PER_RYD
RADIATION_X_MAX_RYD = 1.0e3
RADIATION_Y_DYNAMIC_RANGE_DEX = 8.0


def positive(values: np.ndarray) -> np.ndarray:
    """Mask nonpositive samples for the current logarithmic radiation plots.

    Parameters
    ----------
    values : ndarray, shape (N,)
        Incident intensities, or their linear sum [erg cm^-2 s^-1].

    Returns
    -------
    ndarray, shape (N,)
        Positive values are unchanged; other values become NaN. The original
        physical arrays are unchanged and remain available for saved products.

    Example
    -------
    values=[0.0, 2e-6, -1e-6] gives [NaN, 2e-6, NaN].
    """
    return np.where(values > 0.0, values, np.nan)


def add_eight_ev_marker(axis: Axes, *, label_line: bool = False) -> None:
    """Draw the existing 8 eV guide at its position on a Rydberg x axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Radiation plot with photon energy [Ryd] on its x axis.
    label_line : bool
        True adds the existing rotated label at 4 percent of the axis height.

    Returns
    -------
    None
        Adds the guide and optional label to axis using the original styles.

    Example
    -------
    add_eight_ev_marker(axis=component_axis, label_line=True) labels one panel.
    """
    axis.axvline(EIGHT_EV_RYD, color="0.35", linestyle=":", linewidth=1.2)
    if label_line:
        axis.text(
            EIGHT_EV_RYD,
            0.04,
            "8 eV",
            rotation=90,
            transform=axis.get_xaxis_transform(),
            ha="right",
            va="bottom",
            color="0.35",
            fontsize=8,
        )
