"""Accumulate cold QUOKKA gas in hydrogen-density/shielding-column bins."""
from __future__ import annotations

import numpy as np

from quokka2s.physics.gas_fields import hydrogen_number_density_cm3
from quokka2s.physics.settings import EMISSION_TEMPERATURE_BOUNDARY_K


class ColdGasPhaseAccumulator:
    """Keep count and mass histograms; retain no cell arrays or lookup tables.

    Parameters
    ----------
    edges_nH_log10, edges_NH_log10 : array-like, shape (Nx+1,), (Ny+1,)
        Increasing log10 bin edges for nH [cm^-3] and shielding NH [cm^-2].
        Bins are [left, right), including the last bin. Choose the final edge
        above the largest coordinate; an uncovered cold cell raises an error.
    query_chunk : int
        Cells selected and binned at once, e.g. 1_000_000. This changes temporary
        array sizes, not the native file reads or complete z-column calculation.

    Example: cold_nH_NH_count[2, 7] counts cold cells in nH bin 2 and NH bin 7.
    Cold means T_QUOKKA < 3000 K; it does not use a DESPOTIC temperature.
    """

    def __init__(
        self,
        *,
        edges_nH_log10,
        edges_NH_log10,
        query_chunk=1_000_000,
    ):
        self.edges_nH_log10 = np.asarray(edges_nH_log10, dtype=float).copy()
        self.edges_NH_log10 = np.asarray(edges_NH_log10, dtype=float).copy()
        for edges in (self.edges_nH_log10, self.edges_NH_log10):
            if edges.ndim != 1 or edges.size < 2:
                raise ValueError("Phase edges must be one-dimensional bin boundaries")
            if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0.0):
                raise ValueError("Phase edges must be finite and increasing")
        self.query_chunk = int(query_chunk)
        if self.query_chunk <= 0:
            raise ValueError("query_chunk must be positive")

        histogram_shape = (
            self.edges_nH_log10.size - 1,
            self.edges_NH_log10.size - 1,
        )
        self.cold_nH_NH_count = np.zeros(histogram_shape, dtype=np.int64)
        self.cold_nH_NH_mass_g = np.zeros(histogram_shape, dtype=float)
        self.all_cell_count = 0
        self.cold_cell_count = 0
        self.all_mass_g = 0.0
        self.cold_mass_g = 0.0
        self.snapshot_count = 0
        self.header_only_cell_count = 0
        self.cold_ranges = {
            "nH_cm3": [float("inf"), float("-inf")],
            "NH_cm2": [float("inf"), float("-inf")],
            "temperature_QUOKKA_K": [float("inf"), float("-inf")],
        }

    def add_layer(self, *, layer, cell_volume_cm3):
        """Select cold cells from one native layer and add their count/mass.

        layer : ColdGasLayer
            From ColdGasColumnReader.iter_layers(). Its density_g_cm3,
            temperature_QUOKKA_K and shielding_NH_cm2 are equal-shape XYZ
            arrays, e.g. (256, 256, 128), in g/cm^3, K and cm^-2.
        cell_volume_cm3 : float
            One native cell's volume [cm^3], from Snapshot.cell_volume_cm3.

        Returns None. Only histograms, extrema and scalar counts survive.
        Density/NH must be positive and finite; temperature must be finite
        and nonnegative. Invalid inputs raise rather than silently losing cells.
        """
        density = layer.density_g_cm3.reshape(-1)
        temperature = layer.temperature_QUOKKA_K.reshape(-1)
        shielding_column = layer.shielding_NH_cm2.reshape(-1)
        volume = float(cell_volume_cm3)
        if not np.isfinite(volume) or volume <= 0.0:
            raise ValueError("Cell volume must be finite and positive [cm^3]")
        if density.shape != temperature.shape or density.shape != shielding_column.shape:
            raise ValueError("Cold-gas layer field shapes must match")

        self.snapshot_count = max(self.snapshot_count, 1)
        for start in range(0, density.size, self.query_chunk):
            stop = min(start + self.query_chunk, density.size)
            batch_density = density[start:stop]
            batch_temperature = temperature[start:stop]
            batch_column = shielding_column[start:stop]
            if not np.isfinite(batch_density).all() or np.any(batch_density <= 0.0):
                raise ValueError("Cold-gas density must be finite and positive [g/cm^3]")
            if not np.isfinite(batch_column).all() or np.any(batch_column <= 0.0):
                raise ValueError("Shielding NH must be finite and positive [cm^-2]")
            if not np.isfinite(batch_temperature).all() or np.any(batch_temperature < 0.0):
                raise ValueError("QUOKKA temperature must be finite and nonnegative [K]")

            self.all_cell_count += batch_density.size
            self.all_mass_g += float(np.sum(batch_density)) * volume
            cold_cells = batch_temperature < EMISSION_TEMPERATURE_BOUNDARY_K
            cold_count = int(np.count_nonzero(cold_cells))
            if cold_count == 0:
                continue

            hydrogen_density = hydrogen_number_density_cm3(
                density_g_cm3=batch_density[cold_cells],
            )
            cold_column = batch_column[cold_cells]
            cold_temperature = batch_temperature[cold_cells]
            cold_mass = batch_density[cold_cells] * volume
            self._add_cold_cells(
                hydrogen_density_cm3=hydrogen_density,
                shielding_column_cm2=cold_column,
                temperature_QUOKKA_K=cold_temperature,
                mass_g=cold_mass,
            )

    def record_header_proven_no_cold_cells(self, *, cell_count):
        """Record an all-hot snapshot after a fresh header proves Tmin >= 3000 K.

        cell_count is the complete native grid's population. The caller owns
        the header proof; no payload is read here. Cold counts/mass are exactly
        zero, but total gas mass is unknown, so it is saved as NaN, not zero.
        """
        self.all_cell_count += int(cell_count)
        self.header_only_cell_count += int(cell_count)
        self.all_mass_g = float("nan")
        self.snapshot_count += 1

    def _add_cold_cells(
        self,
        *,
        hydrogen_density_cm3,
        shielding_column_cm2,
        temperature_QUOKKA_K,
        mass_g,
    ):
        """Bin one already selected cold batch, each input shape (N,).

        Called by add_layer() after its physical-input checks. Coordinates are
        physical nH [cm^-3] and NH [cm^-2]; weights are native gas mass [g].
        Uses the same right-sided bin search as the validated range benchmark.
        """
        density_bins = np.searchsorted(
            self.edges_nH_log10,
            np.log10(hydrogen_density_cm3),
            side="right",
        ) - 1
        column_bins = np.searchsorted(
            self.edges_NH_log10,
            np.log10(shielding_column_cm2),
            side="right",
        ) - 1
        outside_bins = (
            (density_bins < 0)
            | (density_bins >= self.cold_nH_NH_count.shape[0])
            | (column_bins < 0)
            | (column_bins >= self.cold_nH_NH_count.shape[1])
        )
        if np.any(outside_bins):
            raise ValueError("Configured nH–NH bins do not cover every cold cell")

        flat_bins = np.ravel_multi_index(
            (density_bins, column_bins),
            self.cold_nH_NH_count.shape,
        )
        batch_counts = np.bincount(
            flat_bins,
            minlength=self.cold_nH_NH_count.size,
        )
        batch_mass = np.bincount(
            flat_bins,
            weights=mass_g,
            minlength=self.cold_nH_NH_mass_g.size,
        )
        self.cold_nH_NH_count += batch_counts.reshape(self.cold_nH_NH_count.shape)
        self.cold_nH_NH_mass_g += batch_mass.reshape(self.cold_nH_NH_mass_g.shape)
        self.cold_cell_count += hydrogen_density_cm3.size
        self.cold_mass_g += float(np.sum(mass_g))
        for name, values in (
            ("nH_cm3", hydrogen_density_cm3),
            ("NH_cm2", shielding_column_cm2),
            ("temperature_QUOKKA_K", temperature_QUOKKA_K),
        ):
            self.cold_ranges[name][0] = min(self.cold_ranges[name][0], float(values.min()))
            self.cold_ranges[name][1] = max(self.cold_ranges[name][1], float(values.max()))

    def add_snapshot(self, *, saved_arrays):
        """Sum one saved snapshot's raw bins, counts, masses and cold ranges.

        saved_arrays is the NPZ-compatible dict from build_outputs(). Bin edges
        must match exactly. Each snapshot contributes once: this is a sum of
        snapshot populations, without time weighting or averaging fractions.
        Display arrays are rebuilt from the combined raw histograms afterward.
        """
        for key in ("edges_nH_log10", "edges_NH_log10"):
            if not np.array_equal(getattr(self, key), saved_arrays[key]):
                raise ValueError("Snapshot phase bins must match before combining")
        self.cold_nH_NH_count += saved_arrays["cold_nH_NH_count"]
        self.cold_nH_NH_mass_g += saved_arrays["cold_nH_NH_mass_g"]
        self.all_cell_count += int(saved_arrays["all_cell_count"])
        self.cold_cell_count += int(saved_arrays["cold_cell_count"])
        self.all_mass_g += float(saved_arrays["all_mass_g"])
        self.cold_mass_g += float(saved_arrays["cold_mass_g"])
        self.snapshot_count += int(saved_arrays["snapshot_count"])
        self.header_only_cell_count += int(saved_arrays["header_only_cell_count"])
        if int(saved_arrays["cold_cell_count"]) > 0:
            for name, bounds in self.cold_ranges.items():
                saved_bounds = saved_arrays["cold_range_" + name]
                bounds[0] = min(bounds[0], float(saved_bounds[0]))
                bounds[1] = max(bounds[1], float(saved_bounds[1]))

    def summary(self):
        """Return JSON-ready counts, physical cold ranges and selection meaning."""
        return {
            "all_cell_count": self.all_cell_count,
            "cold_cell_count": self.cold_cell_count,
            "all_mass_g": self.all_mass_g if np.isfinite(self.all_mass_g) else None,
            "cold_mass_g": self.cold_mass_g,
            "snapshot_count": self.snapshot_count,
            "header_only_cell_count": self.header_only_cell_count,
            "invalid_cell_count": None if self.header_only_cell_count else 0,
            "out_of_bin_cell_count": 0,
            "cold_temperature_rule": f"T_QUOKKA < {EMISSION_TEMPERATURE_BOUNDARY_K:g} K",
            "cold_ranges": {
                name: bounds.copy() if self.cold_cell_count else None
                for name, bounds in self.cold_ranges.items()
            },
            "aggregation": "Snapshot populations summed once each; no time weighting",
        }

    def build_outputs(self, *, title, original_table_bounds, extended_table_bounds):
        """Return raw histograms and every numerical array needed for plotting.

        title : str
            Snapshot or job title, e.g. "sigma13 / plt0008703".
        original_table_bounds, extended_table_bounds : dict
            Physical axis endpoints: {"nH": [min, max], "NH": [min, max]},
            in cm^-3 and cm^-2. Read these from saved table coordinates, without
            interpolation or a solver call. They describe a 2D projection only.

        Returns dict[str, ndarray]. Histograms have shape (Nx, Ny); saved
        logarithmic display arrays retain NaN in empty bins. The renderer only
        selects these arrays and uses the saved zoom/full limits and outlines.
        """
        if int(np.sum(self.cold_nH_NH_count)) != self.cold_cell_count:
            raise ValueError("Cold histogram lost cells")
        if not np.isclose(
            np.sum(self.cold_nH_NH_mass_g),
            self.cold_mass_g,
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError("Cold histogram lost gas mass")
        arrays = {
            "edges_nH_log10": self.edges_nH_log10.copy(),
            "edges_NH_log10": self.edges_NH_log10.copy(),
            "cold_nH_NH_count": self.cold_nH_NH_count.copy(),
            "cold_nH_NH_mass_g": self.cold_nH_NH_mass_g.copy(),
            "all_cell_count": np.asarray(self.all_cell_count, dtype=np.int64),
            "cold_cell_count": np.asarray(self.cold_cell_count, dtype=np.int64),
            "all_mass_g": np.asarray(self.all_mass_g),
            "cold_mass_g": np.asarray(self.cold_mass_g),
            "snapshot_count": np.asarray(self.snapshot_count, dtype=np.int64),
            "header_only_cell_count": np.asarray(self.header_only_cell_count, dtype=np.int64),
            "title": np.asarray(title),
            "has_cold_cells": np.asarray(self.cold_cell_count > 0),
        }
        for name, bounds in self.cold_ranges.items():
            arrays["cold_range_" + name] = np.asarray(
                bounds if self.cold_cell_count else [np.nan, np.nan],
            )
        prepare_distribution_display_arrays(arrays=arrays)
        prepare_table_outlines(
            arrays=arrays,
            original_table_bounds=original_table_bounds,
            extended_table_bounds=extended_table_bounds,
        )
        return arrays


def prepare_distribution_display_arrays(*, arrays):
    """Save normalization, logarithms, masks, marginals and occupied-bin limits."""
    counts = arrays["cold_nH_NH_count"]
    mass_g = arrays["cold_nH_NH_mass_g"]
    count_total = int(arrays["cold_cell_count"])
    mass_total = float(arrays["cold_mass_g"])
    count_fraction = np.zeros(counts.shape, dtype=float)
    mass_fraction = np.zeros(mass_g.shape, dtype=float)
    if count_total:
        count_fraction = counts / count_total
    if mass_total:
        mass_fraction = mass_g / mass_total
    arrays["cold_nH_NH_count_fraction"] = count_fraction
    arrays["cold_nH_NH_mass_fraction"] = mass_fraction

    for name, values in (("count", counts), ("mass", mass_fraction)):
        positive_bins = values > 0.0
        log_values = np.full(values.shape, np.nan)
        np.log10(values, out=log_values, where=positive_bins)
        arrays[name + "_display_log10"] = log_values
        arrays[name + "_display_mask"] = ~positive_bins
        if positive_bins.any():
            low = float(np.min(log_values[positive_bins]))
            high = float(np.max(log_values[positive_bins]))
            if low == high:
                low -= 0.5
                high += 0.5
            arrays[name + "_color_limits_log10"] = np.asarray([low, high])
        else:
            arrays[name + "_color_limits_log10"] = np.asarray([0.0, 1.0])
        fractions = count_fraction if name == "count" else mass_fraction
        arrays[name + "_nH_marginal_fraction"] = np.sum(fractions, axis=1)
        arrays[name + "_NH_marginal_fraction"] = np.sum(fractions, axis=0)

    arrays["count_colorbar_label"] = np.asarray(r"$\log_{10}$ cold-cell count per bin")
    arrays["mass_colorbar_label"] = np.asarray(r"$\log_{10}$ fraction of cold gas mass per bin")
    occupied_density, occupied_column = np.nonzero(counts)
    for axis, occupied in (("nH", occupied_density), ("NH", occupied_column)):
        edges = arrays["edges_" + axis + "_log10"]
        if occupied.size:
            limits = edges[[int(occupied.min()), int(occupied.max()) + 1]]
            padding = max(0.1, 0.05 * float(limits[1] - limits[0]))
            limits = limits + np.asarray([-padding, padding])
        else:
            limits = edges[[0, -1]]
        arrays["zoom_" + axis + "_limits_log10"] = limits

    total = int(arrays["all_cell_count"])
    fraction = count_total / total if total else 0.0
    arrays["population_label"] = np.asarray(
        f"{count_total:,} cold / {total:,} total cells ({fraction:.2%})",
    )
    snapshots = int(arrays["snapshot_count"])
    temperature_boundary = f"{EMISSION_TEMPERATURE_BOUNDARY_K:g}"
    selection = r"$T_{\rm QUOKKA}<" + temperature_boundary + "$ K"
    arrays["empty_population_label"] = np.asarray("No cells with " + selection)
    if snapshots > 1:
        sampling = selection + f"; sum of {snapshots} snapshot populations; no time weighting"
    else:
        sampling = "Cold selection: " + selection
    arrays["sampling_label"] = np.asarray(sampling)


def prepare_table_outlines(*, arrays, original_table_bounds, extended_table_bounds):
    """Save physical table endpoints, log-coordinate rectangles and full limits."""
    for prefix, bounds in (
        ("original", original_table_bounds),
        ("extended", extended_table_bounds),
    ):
        density_bounds = np.asarray(bounds["nH"], dtype=float)
        column_bounds = np.asarray(bounds["NH"], dtype=float)
        for values in (density_bounds, column_bounds):
            if values.shape != (2,) or not np.isfinite(values).all():
                raise ValueError("Table bounds must contain two finite endpoints")
            if values[0] <= 0.0 or values[1] <= values[0]:
                raise ValueError("Table bounds must be positive and increasing")
        arrays[prefix + "_table_nH_bounds_cm3"] = density_bounds.copy()
        arrays[prefix + "_table_NH_bounds_cm2"] = column_bounds.copy()
        x_min, x_max = np.log10(density_bounds)
        y_min, y_max = np.log10(column_bounds)
        arrays[prefix + "_table_outline_log10"] = np.asarray([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
            [x_min, y_min],
        ])

    for axis, coordinate in (("nH", 0), ("NH", 1)):
        original = arrays["original_table_outline_log10"][:, coordinate]
        extended = arrays["extended_table_outline_log10"][:, coordinate]
        zoom_limits = arrays["zoom_" + axis + "_limits_log10"]
        low = min(float(original.min()), float(extended.min()), float(zoom_limits[0]))
        high = max(float(original.max()), float(extended.max()), float(zoom_limits[1]))
        padding = max(0.1, 0.03 * (high - low))
        arrays["full_" + axis + "_limits_log10"] = np.asarray([
            low - padding,
            high + padding,
        ])
