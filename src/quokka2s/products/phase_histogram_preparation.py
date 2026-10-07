"""Prepare numerical arrays and saved descriptors for the manuscript phase panels."""

from __future__ import annotations

import numpy as np

from quokka2s.products.emission_phase_histograms import PANELS


# Keep all fourteen raw panels; these ten fill the manuscript's five-by-two grid.
DISPLAY_PANEL_KEYS = (
    "mass_T_QK", "mass_T_DSP", "mass_T_2R", "NH_rho",
    "halpha", "hi21", "cii", "ciii_977", "civ_1548", "co21",
)
COLORBAR_DYNAMIC_RANGE = 1.0e6


def add_phase_histogram_display_fields(panels: dict[str, dict[str, np.ndarray]]) -> None:
    """Add logarithmic bins, color limits and axis descriptors before saving.

    Parameters
    ----------
    panels : dict of dict
        All PANELS from DexHistogram.result(). Each H has shape (Nx, Ny),
        with mass [g] or luminosity [erg/s]; x_edges/y_edges are log10
        density, temperature or NH coordinates with one extra edge per axis.

    Returns
    -------
    None
        Updates each panel without changing H or its edges. log10_H has
        NaN in nonpositive bins. Each panel also stores its color and axis
        limits, temperature description, quantity name and physical weight unit.

    Examples
    --------
    panels["halpha"]["log10_H"] stores log10 luminosity for each density-T bin.
    """
    definitions = {key: (temperature, group) for key, temperature, group in PANELS}
    group_maxima = calculate_display_group_maxima(
        panels=panels,
        definitions=definitions,
    )
    rho_limits_log10, temperature_limits_log10 = calculate_shared_phase_limits(
        panels=panels,
    )

    for key, panel in panels.items():
        temperature, group = definitions[key]
        positive_bin_mask = panel["H"] > 0.0
        log10_H = np.full(panel["H"].shape, np.nan, dtype=float)
        np.log10(panel["H"], out=log10_H, where=positive_bin_mask)
        panel["log10_H"] = log10_H

        group_peak = group_maxima.get(group, 0.0)
        if group_peak > 0.0:
            log10_peak = np.log10(group_peak)
            color_limits_log10 = (
                log10_peak - np.log10(COLORBAR_DYNAMIC_RANGE),
                log10_peak,
            )
        else:
            color_limits_log10 = (0.0, 1.0)
        panel["color_limits_log10"] = np.asarray(color_limits_log10)
        panel["temperature_descriptor"] = np.asarray(temperature or "")

        is_mass = key.startswith("mass") or key == "NH_rho"
        unit = "g" if is_mass else "erg/s"
        quantity = "mass" if is_mass else "luminosity"
        panel["quantity"] = np.asarray(quantity)
        panel["weight_unit"] = np.asarray(unit)
        if key == "NH_rho":
            panel["x_limits_log10"] = np.asarray(panel["x_edges"][[0, -1]])
            panel["y_limits_log10"] = np.asarray(panel["y_edges"][[0, -1]])
        else:
            panel["x_limits_log10"] = np.asarray(rho_limits_log10)
            panel["y_limits_log10"] = np.asarray(temperature_limits_log10)


def calculate_display_group_maxima(*, panels, definitions):
    """Return peaks for the saved ten-panel selection's shared colorbar groups."""
    group_maxima = {}
    for key in DISPLAY_PANEL_KEYS:
        _, group = definitions[key]
        panel_peak = float(np.max(panels[key]["H"]))
        group_maxima[group] = max(group_maxima.get(group, 0.0), panel_peak)
    return group_maxima


def calculate_shared_phase_limits(*, panels):
    """Return shared density and temperature limits [dex] for displayed T panels."""
    temperature_panels = [
        panels[key] for key in DISPLAY_PANEL_KEYS if key != "NH_rho"
    ]
    rho_limits_log10 = (
        min(panel["x_edges"][0] for panel in temperature_panels),
        max(panel["x_edges"][-1] for panel in temperature_panels),
    )
    temperature_limits_log10 = (
        min(panel["y_edges"][0] for panel in temperature_panels),
        max(panel["y_edges"][-1] for panel in temperature_panels),
    )
    return rho_limits_log10, temperature_limits_log10
