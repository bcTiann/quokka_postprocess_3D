"""Prepare numerical panels for the five-field simulation input slice."""
from __future__ import annotations

import numpy as np


TABLE_INPUT_PANEL_KEYS = (
    "nH_slice",
    "NH_slice",
    "dVdr_slice",
    "T_qk_slice",
    "T_dsp_slice",
)
TEMPERATURE_SLICE_KEYS = ("T_qk_slice", "T_dsp_slice")


def prepare_slice_plot_data(payload):
    """Keep raw fields and add the selected/log-valued panels before saving.

    Parameters
    ----------
    payload : dict[str, ndarray]
        The five TABLE_INPUT_PANEL_KEYS arrays and the mixed-temperature
        availability mask 'valid', each shape (Ny, Nz). Raw fields use CGS
        units or kelvin; this function retains them unchanged.

    Returns
    -------
    dict[str, ndarray]
        Adds {key}_selected with unavailable cells marked NaN, {key}_log10
        with nonpositive values also marked NaN, a scalar {key}_is_empty,
        two-entry {key}_log_limits, and {key}_colorbar_ticks. All panel arrays
        remain (Ny, Nz); the renderer transposes them only for display.

    Example: prepared['NH_slice_log10'] is the final selected log10(NH)
    panel. The renderer neither applies 'valid' again nor takes a logarithm.
    """
    prepared = dict(payload)
    for key in TABLE_INPUT_PANEL_KEYS:
        selected_values = np.where(payload["valid"], payload[key], np.nan)
        positive_values = selected_values > 0.0
        is_empty = not positive_values.any()
        log_values = np.where(
            positive_values,
            np.log10(np.where(positive_values, selected_values, 1.0)),
            np.nan,
        )
        if is_empty:
            log_limits = np.asarray([np.nan, np.nan])
        elif key in TEMPERATURE_SLICE_KEYS:
            log_limits = np.asarray([2.0, 8.0])
        else:
            log_limits = np.asarray([
                float(np.nanmin(log_values)),
                float(np.nanmax(log_values)),
            ])
        colorbar_ticks = np.asarray([], dtype=float)
        if key in TEMPERATURE_SLICE_KEYS and not is_empty:
            colorbar_ticks = np.asarray([2.0, 4.0, 6.0, 8.0])
        prepared[key + "_selected"] = selected_values
        prepared[key + "_log10"] = log_values
        prepared[key + "_is_empty"] = np.asarray(is_empty)
        prepared[key + "_log_limits"] = log_limits
        prepared[key + "_colorbar_ticks"] = colorbar_ticks
    return prepared
