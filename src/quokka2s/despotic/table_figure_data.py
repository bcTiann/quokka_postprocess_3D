"""Prepare DESPOTIC heatmap values and contours before drawing any figures."""
from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import numpy as np

from quokka2s.despotic.table_data import DespoticTable


DEFAULT_FIELDS = (
    "tg_final",
    "species:CO:abundance",
    "species:C+:abundance",
    "species:C:abundance",
    "species:HCO+:abundance",
    "species:e-:abundance",
    "species:CO:lumPerH",
    "species:CO21:lumPerH",
    "species:C+:lumPerH",
    "species:C:lumPerH",
    "species:HCO+:lumPerH",
)
ENCLOSURE_FRACTIONS = (0.50, 0.90, 0.99, 1.00)
FAILURE_DISPLAY_POLICY = "original solver failures hidden"
DEFAULT_FIGURE_DATA_PATH = Path("output/despotic_table_figures/prepared.npz")


def logarithmic_cell_edges(values: np.ndarray) -> np.ndarray:
    """Return log10 midpoint edges (N+1,) for a positive table axis (N,)."""
    if values.size < 2:
        raise ValueError("Need at least two grid points to compute edges.")
    logarithmic_values = np.log10(values)
    differences = np.diff(logarithmic_values)
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = logarithmic_values[:-1] + differences / 2.0
    edges[0] = logarithmic_values[0] - differences[0] / 2.0
    edges[-1] = logarithmic_values[-1] + differences[-1] / 2.0
    return edges


def blocky_contour_segments(
    values: np.ndarray,
    threshold: float,
    hydrogen_density_edges: np.ndarray,
    column_edges: np.ndarray,
) -> np.ndarray:
    """Return (Nsegment, 2, 2) boundaries between values above/below a threshold.

    values is (NnH, NNH); endpoints use x=NH [cm^-2], y=nH [cm^-3]. NaN
    lies below the threshold. Interior table-cell boundaries are retained
    exactly, without marching-squares interpolation or an outer-frame contour.
    """
    above_threshold = values >= threshold
    segments = []
    density_count, column_count = above_threshold.shape
    for density_index in range(density_count):
        for column_index in range(column_count - 1):
            if (
                above_threshold[density_index, column_index]
                != above_threshold[density_index, column_index + 1]
            ):
                x = column_edges[column_index + 1]
                segments.append((
                    (x, hydrogen_density_edges[density_index]),
                    (x, hydrogen_density_edges[density_index + 1]),
                ))
    for density_index in range(density_count - 1):
        for column_index in range(column_count):
            if (
                above_threshold[density_index, column_index]
                != above_threshold[density_index + 1, column_index]
            ):
                y = hydrogen_density_edges[density_index + 1]
                segments.append((
                    (column_edges[column_index], y),
                    (column_edges[column_index + 1], y),
                ))
    return np.asarray(segments, dtype=float).reshape(-1, 2, 2)


def read_table_field(table: DespoticTable, token: str) -> tuple[np.ndarray, str]:
    """Select a (NnH, NNH, NdVdr) field and its established heatmap label."""
    if token == "tg_final":
        return table.tg_final, "T_g (K)"
    if token == "failure_mask":
        return table.failure_mask.astype(float), "Failure Mask"
    if token.startswith("energy:"):
        key = token.split(":", 1)[1]
        if not table.energy_terms or key not in table.energy_terms:
            raise ValueError(f"Energy term '{key}' not found in the table.")
        return table.energy_terms[key], f"Energy Term: {key}"
    if token.startswith("species:"):
        _, species, field = token.split(":")
        record = table.require_species(species)
        if field == "abundance":
            values = record.abundance
        else:
            if record.line is None:
                raise ValueError(f"Species '{species}' has no line data; cannot plot '{field}'")
            values = getattr(record.line, field)
        return values, f"{species}:{field}"
    raise ValueError(f"Unknown field token: {token}")


def prepare_selected_panels(
    table: DespoticTable,
    field_tokens: Sequence[str],
    dvdr_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Prepare raw values, availability, display masks and limits for every panel.

    Saved panels have shape (Nslice, Nfield, NnH, NNH). Numerical availability
    is finite and positive, independently of original solver failure. Display
    retains the existing policy of blanking original failures even when a
    filled value is available. Color limits have shape (Nslice, Nfield, 2).
    """
    field_values = []
    field_titles = []
    for token in field_tokens:
        values, title = read_table_field(table=table, token=token)
        field_values.append(values[:, :, dvdr_indices])
        field_titles.append(title)
    panel_values = np.stack(field_values).transpose(3, 0, 1, 2)
    field_is_available = np.isfinite(panel_values) & (panel_values > 0.0)
    original_failure = np.zeros(panel_values.shape[0:1] + panel_values.shape[2:], dtype=bool)
    if table.failure_mask is not None:
        original_failure = table.failure_mask[:, :, dvdr_indices].transpose(2, 0, 1).copy()
    panel_is_hidden = ~field_is_available | original_failure[:, np.newaxis]
    color_limits = np.full(panel_values.shape[:2] + (2,), np.nan)
    panel_has_values = np.zeros(panel_values.shape[:2], dtype=bool)
    for slice_index in range(panel_values.shape[0]):
        for field_index in range(panel_values.shape[1]):
            values = panel_values[slice_index, field_index]
            hidden = panel_is_hidden[slice_index, field_index]
            visible_values = values[~hidden]
            if visible_values.size:
                color_limits[slice_index, field_index] = (visible_values.min(), visible_values.max())
                panel_has_values[slice_index, field_index] = True
    return {
        "field_tokens": np.asarray(field_tokens),
        "field_titles": np.asarray(field_titles),
        "panel_values": panel_values,
        "field_is_available": field_is_available,
        "original_solver_failure": original_failure,
        "panel_is_hidden": panel_is_hidden,
        "color_limits": color_limits,
        "panel_has_values": panel_has_values,
        "failure_display_policy": np.asarray(FAILURE_DISPLAY_POLICY),
    }


def prepare_sample_histogram(
    samples: np.ndarray | None,
    density_log_edges: np.ndarray,
    column_log_edges: np.ndarray,
    dvdr_log_edges: np.ndarray | None,
    dvdr_index: int,
) -> tuple[np.ndarray, bool]:
    """Bin one slice's explicit samples once, independently of the plotted field.

    samples is (N,2): log10(nH), log10(NH); (N,3) adds log10(dVdr); (N,4)
    adds cell mass [g]. Three/four-column data use a lower-inclusive and
    upper-exclusive dVdr bin. Returns (NnH, NNH) counts/mass and mass-mode flag.
    None produces an empty count histogram and no overlays.
    """
    histogram_shape = (density_log_edges.size - 1, column_log_edges.size - 1)
    if samples is None:
        return np.zeros(histogram_shape), False
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] not in (2, 3, 4):
        raise ValueError("samples must have 2, 3 or 4 columns: log10(nH), log10(NH), optional log10(dVdr), optional cell mass [g]")
    mass_mode = samples.shape[1] == 4
    weights = None
    if samples.shape[1] >= 3:
        lower = dvdr_log_edges[dvdr_index]
        upper = dvdr_log_edges[dvdr_index + 1]
        selected = (samples[:, 2] >= lower) & (samples[:, 2] < upper)
        if mass_mode:
            weights = samples[selected, 3]
        samples = samples[selected, :2]
    histogram, _, _ = np.histogram2d(
        samples[:, 0],
        samples[:, 1],
        bins=[density_log_edges, column_log_edges],
        weights=weights,
    )
    return histogram, mass_mode


def enclosure_levels(histogram: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Select distinct bin thresholds enclosing 50, 90, 99 and 100 percent.

    Fractions refer to total counts/mass in this slice's histogram, rather than
    the percentile of occupied bins. Duplicate thresholds keep the first fraction.
    """
    positive_values = histogram[histogram > 0.0]
    if positive_values.size == 0:
        return np.empty(0), np.empty(0), 0.0
    sorted_values = np.sort(positive_values)[::-1]
    cumulative_values = np.cumsum(sorted_values)
    total = float(cumulative_values[-1])
    fractions = []
    thresholds = []
    seen_thresholds = set()
    for fraction in ENCLOSURE_FRACTIONS:
        position = int(np.searchsorted(cumulative_values, fraction * total))
        position = min(position, sorted_values.size - 1)
        threshold = float(sorted_values[position])
        if threshold in seen_thresholds:
            continue
        seen_thresholds.add(threshold)
        fractions.append(fraction)
        thresholds.append(threshold)
    return np.asarray(fractions), np.asarray(thresholds), total


def enclosure_legend_labels(
    fractions: np.ndarray,
    thresholds: np.ndarray,
    total: float,
    mass_mode: bool,
) -> np.ndarray:
    """Prepare the established count/mass legend text for stored contours."""
    labels = []
    unit = "mass" if mass_mode else "cells"
    for fraction, threshold in zip(fractions, thresholds):
        percent = int(fraction * 100)
        if fraction >= 1.0:
            if mass_mode:
                label = f"{percent}% of {unit} (total {total:.3e} g)"
            else:
                label = f"{percent}% of {unit} (total {int(total):,} cells)"
        elif mass_mode:
            label = f"{percent}% of {unit} (≥ {threshold:.1e} g / bin)"
        else:
            label = f"{percent}% of {unit} (≥ {int(threshold):,} cells / bin)"
        labels.append(label)
    return np.asarray(labels, dtype=str)


def add_sample_contours(
    payload: dict[str, np.ndarray],
    samples: np.ndarray | None,
    dvdr_log_edges: np.ndarray | None,
) -> None:
    """Save one histogram and enclosure segment set per selected dVdr slice.

    Segments are variable-length numeric arrays under slice/level keys, so NPZ
    reading needs no pickle. Every field in a slice reuses these same arrays.
    """
    histograms = []
    for slice_index, dvdr_index in enumerate(payload["dvdr_indices"]):
        histogram, mass_mode = prepare_sample_histogram(
            samples=samples,
            density_log_edges=payload["density_log_edges"],
            column_log_edges=payload["column_log_edges"],
            dvdr_log_edges=dvdr_log_edges,
            dvdr_index=int(dvdr_index),
        )
        histograms.append(histogram)
        fractions, thresholds, total = enclosure_levels(histogram=histogram)
        prefix = f"samples_slice_{slice_index}"
        payload[f"{prefix}_fractions"] = fractions
        payload[f"{prefix}_thresholds"] = thresholds
        payload[f"{prefix}_total"] = np.asarray(total)
        payload[f"{prefix}_labels"] = enclosure_legend_labels(
            fractions=fractions,
            thresholds=thresholds,
            total=total,
            mass_mode=mass_mode,
        )
        unit = "mass" if mass_mode else "cells"
        payload[f"{prefix}_legend_title"] = np.asarray(f"sim {unit} enclosed")
        for level_index, threshold in enumerate(thresholds):
            payload[f"{prefix}_segments_{level_index}"] = blocky_contour_segments(
                values=histogram,
                threshold=float(threshold),
                hydrogen_density_edges=payload["density_edges_cm3"],
                column_edges=payload["column_edges_cm2"],
            )
    payload["sample_histograms"] = np.stack(histograms)


def longest_segment_midpoint(segments: np.ndarray) -> tuple[float, float]:
    """Return the established label position on the longest physical segment.

    segments has shape (Nsegment, 2, 2), x=NH and y=nH. Preserve the original
    squared-distance comparison and first-longest selection without smoothing.
    """
    squared_lengths = []
    for segment in segments:
        x_difference = segment[1][0] - segment[0][0]
        y_difference = segment[1][1] - segment[0][1]
        squared_lengths.append(x_difference ** 2 + y_difference ** 2)
    longest = int(np.argmax(squared_lengths))
    xmid = 0.5 * (segments[longest][0][0] + segments[longest][1][0])
    ymid = 0.5 * (segments[longest][0][1] + segments[longest][1][1])
    return xmid, ymid


def add_temperature_contours(payload: dict[str, np.ndarray]) -> None:
    """Save the raw temperature panel's 100/200 K contours and label positions.

    As in the original renderer, contour geometry uses finite positive raw T,
    including original-failure locations; heatmap blanking remains independent.
    Labels sit at the midpoint of the longest physical-coordinate segment.
    """
    temperature_fields = np.flatnonzero(payload["field_tokens"] == "tg_final")
    for slice_index in range(payload["dvdr_indices"].size):
        prefix = f"temperature_slice_{slice_index}"
        levels = []
        positions = []
        if temperature_fields.size:
            values = payload["panel_values"][slice_index, temperature_fields[0]]
            temperature = np.where(np.isfinite(values) & (values > 0.0), values, np.nan)
            finite_temperature = temperature[np.isfinite(temperature)]
            if finite_temperature.size:
                minimum = float(finite_temperature.min())
                maximum = float(finite_temperature.max())
                for threshold in (100.0, 200.0):
                    if not minimum <= threshold <= maximum:
                        continue
                    segments = blocky_contour_segments(
                        values=temperature,
                        threshold=threshold,
                        hydrogen_density_edges=payload["density_edges_cm3"],
                        column_edges=payload["column_edges_cm2"],
                    )
                    if segments.size == 0:
                        continue
                    xmid, ymid = longest_segment_midpoint(segments=segments)
                    level_index = len(levels)
                    payload[f"{prefix}_segments_{level_index}"] = segments
                    levels.append(threshold)
                    positions.append((xmid, ymid))
        payload[f"{prefix}_levels_K"] = np.asarray(levels)
        payload[f"{prefix}_label_positions"] = np.asarray(positions).reshape(-1, 2)


def prepare_table_figure_data(
    table: DespoticTable,
    *,
    field_tokens: Sequence[str] = DEFAULT_FIELDS,
    dvdr_indices: Sequence[int],
    samples: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build complete NPZ-ready panels and reusable contours for selected slices.

    No table values are changed. Plot consumes only this prepared dictionary;
    original failure and current numerical availability remain separate records.
    """
    indices = np.asarray(dvdr_indices, dtype=int)
    if indices.size == 0 or not field_tokens:
        raise ValueError("Choose at least one dVdr slice and table field")
    if np.any(indices < 0) or np.any(indices >= table.dVdr_values.size):
        raise ValueError("dVdr indices must be within the table axis")
    payload = prepare_selected_panels(
        table=table,
        field_tokens=field_tokens,
        dvdr_indices=indices,
    )
    density_log_edges = logarithmic_cell_edges(values=table.nH_values)
    column_log_edges = logarithmic_cell_edges(values=table.col_density_values)
    payload["density_log_edges"] = density_log_edges
    payload["column_log_edges"] = column_log_edges
    payload["density_edges_cm3"] = np.power(10.0, density_log_edges)
    payload["column_edges_cm2"] = np.power(10.0, column_log_edges)
    payload["dvdr_indices"] = indices
    payload["dvdr_values_s"] = table.dVdr_values[indices]
    payload["dvdr_axis_size"] = np.asarray(table.dVdr_values.size)
    dvdr_log_edges = None
    if samples is not None:
        samples = np.asarray(samples, dtype=float)
        if samples.ndim == 2 and samples.shape[1] >= 3:
            dvdr_log_edges = logarithmic_cell_edges(values=table.dVdr_values)
    add_sample_contours(
        payload=payload,
        samples=samples,
        dvdr_log_edges=dvdr_log_edges,
    )
    add_temperature_contours(payload=payload)
    return payload


def read_table_figure_data(path: str | Path) -> dict[str, np.ndarray]:
    """Read prepared numeric/string arrays, with no table access or calculation."""
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}
