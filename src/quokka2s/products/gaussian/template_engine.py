"""Integrate original-cell Gaussians using immutable preintegrated templates.

The compiled kernel locates four phase/log-sigma corners, reads their stored
channel masses and immediately accumulates luminosity-weighted spectra. Valid
outside widths use stable direct bin integrals with the same five-sigma cutoff.
Each call owns its output; concurrent workers share only the immutable table.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from time import perf_counter, perf_counter_ns

import numpy as np

from . import _template_integrate as native
from .intervals import interval_probability
from .template_coverage import TemplateGroupCoverage, sigma_range
from .template_table import FractionalGaussianTable


@dataclass(frozen=True)
class TemplateGroupResult:
    """Channel densities[K,P], original-cell window totals[P] and query timing."""

    profiles: np.ndarray
    full_window: np.ndarray
    clipped_window: np.ndarray
    statistics: dict
    query_interval_ns: tuple[int, int]
    coverage: TemplateGroupCoverage


class TemplateEngine:
    """Read four templates for each original cell, without accumulating nodes.

    Gaussian means and widths use km/s. Luminosities have shape[N,P] in erg/s;
    P is the existing width group's one to twelve luminosity accounts. Results
    have shape[K,P] in erg/s/(km/s), with K at most 400 velocity channels.
    """

    coordinate_rule = "original-cell"
    per_cell_probability_buffer_bytes = 3200
    maximum_direct_probability_rows = 1024
    cutoff_applies_to = "Stored template corners in range; original-cell Gaussians for direct integration and references"

    def __init__(self, *, table):
        if not isinstance(table, FractionalGaussianTable):
            raise TypeError("Gaussian templates require FractionalGaussianTable")
        # Verify the full immutable state before passing metadata-indexed arrays
        # to the compiled kernel, whose inner loops omit bounds checks.
        table.validate_prepared_state()
        self.lookup_tables = table
        self.portable_backend = "compiled-original-cell-templates"
        self.native_parameters = {
            "sigma_nodes": table.sigma_nodes,
            "log_sigma_nodes": table.log_sigma_nodes,
            "phase_starts": table.phase_starts,
            "phase_values": table.phase_axis_values,
            "row_min": table.row_min_bins,
            "row_count": table.row_bin_counts,
            "row_offset": table.row_value_offsets,
            "table_values": table.flat_values,
            "output_bins": table.output_bin_indices,
        }
        native.validate_table(**self.native_parameters)
        # Reuse exact results from the existing reference, not a new tail rule.
        self._clipped_interior_mass = float(interval_probability(-table.Z, table.Z))
        self._full_window_saturation = 12.0
        probe_lower = np.array([-np.inf, -12.0, np.nextafter(-12.0, -np.inf)])
        probe_upper = np.array([np.inf, 12.0, np.nextafter(12.0, np.inf)])
        probe = interval_probability(probe_lower, probe_upper)
        if probe.tobytes() != np.ones(probe.shape, dtype=np.float64).tobytes():
            raise RuntimeError("Original-cell interval reference does not saturate at +/-12")
        self.gate_result = self._check_prepared_arithmetic()

    def validate_prepared_state(self):
        """Refuse table mutation after preparation outside the process clock."""
        self.lookup_tables.validate_prepared_state()

    def _prepare_coordinates(self, centers, sigmas, *, require_coverage=True):
        """Create contiguous float64 vectors and retain NumPy's logarithm."""
        centers = np.ascontiguousarray(centers, dtype=np.float64)
        sigmas = np.ascontiguousarray(sigmas, dtype=np.float64)
        if centers.ndim != 1 or sigmas.shape != centers.shape:
            raise ValueError("Gaussian means and widths must be matching vectors")
        if not np.isfinite(sigmas).all() or np.any(sigmas <= 0.0):
            raise ValueError("Gaussian sigma must be finite and positive")
        minimum = self.lookup_tables.sigma_nodes[0]
        maximum = self.lookup_tables.sigma_nodes[-1]
        if require_coverage and (np.any(sigmas < minimum) or np.any(sigmas > maximum)):
            raise ValueError(
                "Gaussian sigma lies outside the prepared template coverage: "
                f"required [{sigmas.min():.17g}, {sigmas.max():.17g}] km/s, "
                f"prepared [{minimum:.17g}, {maximum:.17g}] km/s. "
                "Set gaussian_grid sigma_min/sigma_max before preparing the table."
            )
        log_sigmas = np.log(sigmas)
        return centers, sigmas, log_sigmas

    def evaluate(self, centers, sigmas):
        """Return template probabilities[N,K] for bounded diagnostic inputs."""
        centers, sigmas, logarithms = self._prepare_coordinates(centers, sigmas)
        return native.probabilities(
            centers=centers,
            sigmas=sigmas,
            log_sigmas=logarithms,
            delta_v=self.lookup_tables.delta_v,
            **self.native_parameters,
        )

    def _check_prepared_arithmetic(self):
        """Check compiled lookup and ordered accumulation before process timing.

        The independent NumPy implementation reads the same prepared table. This
        is an arithmetic-equivalence check, not a claim of exact Gaussian-CDF
        integration or a validation of interpolation accuracy on arbitrary cells.
        """
        table = self.lookup_tables
        layer_indices = np.unique(
            np.linspace(0, table.sigma_nodes.size - 1, 8, dtype=np.int64)
        )
        sigmas = table.sigma_nodes[layer_indices]
        phases = np.asarray([0.0, 0.1, 0.499999999, 0.5, 0.999999999, 1.0])
        shifts = np.asarray([-201.0, -1.0, 0.0, 1.0, 199.0, 201.0])
        centers = table.delta_v * (phases + shifts)
        centers = np.tile(centers, sigmas.size)
        sigmas = np.repeat(sigmas, phases.size)
        expected = table.evaluate(centers, sigmas)
        actual = self.evaluate(centers, sigmas)
        if actual.tobytes() != expected.tobytes():
            raise RuntimeError("Compiled template probabilities differ from the prepared NumPy reference")
        light = np.arange(1, centers.size * 3 + 1, dtype=np.float64)
        light = light.reshape(centers.size, 3)
        light[::5, 1] = 0.0
        expected_mass = np.zeros((table.K, light.shape[1]), dtype=np.float64)
        # Only the small preparation gate uses a Python cell loop. The process
        # kernel performs the same ascending-cell accumulation in compiled code.
        for row in range(centers.size):
            expected_mass += expected[row, :, None] * light[row, None, :]
        centers, sigmas, logarithms = self._prepare_coordinates(centers, sigmas)
        actual_mass, _ = native.accumulate(
            centers=centers,
            sigmas=sigmas,
            log_sigmas=logarithms,
            luminosity=light,
            delta_v=table.delta_v,
            **self.native_parameters,
        )
        actual_mass = np.ascontiguousarray(actual_mass.T)
        if actual_mass.tobytes() != expected_mass.tobytes():
            raise RuntimeError("Compiled template luminosity accumulation changed ordered arithmetic")
        digest = hashlib.sha256()
        digest.update(actual.tobytes())
        digest.update(actual_mass.tobytes())
        return {
            "scope": "Compiled probability lookup and ordered float64 accumulation against the prepared NumPy table",
            "completed_arithmetic_equivalence_gate": True,
            "probe_rows": int(centers.size),
            "probabilities_byte_exact": True,
            "luminosity_accumulation_byte_exact": True,
            "table_sha256": table.info()["table_sha256"],
            "probe_sha256": digest.hexdigest(),
            "kernel_info": native.kernel_info(),
        }

    def _integrate_mixed(self, centers, sigmas, logarithms, luminosity, outside_rows):
        """Bound only direct probabilities while retaining original cell order.

        Every chunk continues the same small output[P,K], including intervening
        table-covered cells. Separate table/direct sums would reorder floating
        additions, so neither a split spectrum nor a final two-path sum is used.
        """
        native.validate_mixed_inputs(
            centers=centers,
            sigmas=sigmas,
            log_sigmas=logarithms,
            luminosity=luminosity,
            delta_v=self.lookup_tables.delta_v,
        )
        channel_mass = np.zeros((luminosity.shape[1], self.lookup_tables.K), dtype=np.float64)
        counters = {
            name: 0 for name in (
                "kernel_rows", "active_rows", "support_bins_visited",
                "template_corner_values_read", "product_updates",
                "direct_active_rows", "direct_support_bins_visited", "direct_product_updates",
            )
        }
        probability_seconds = 0.0
        first_cell = 0
        for first_direct in range(0, outside_rows.size, self.maximum_direct_probability_rows):
            stop_direct = min(first_direct + self.maximum_direct_probability_rows, outside_rows.size)
            selected_rows = outside_rows[first_direct:stop_direct]
            stop_cell = int(outside_rows[stop_direct]) if stop_direct < outside_rows.size else centers.size
            probability_began = perf_counter()
            with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                boundaries = (
                    self.lookup_tables.edges[None, :] - centers[selected_rows, None]
                ) / sigmas[selected_rows, None]
            boundaries = np.clip(boundaries, -self.lookup_tables.Z, self.lookup_tables.Z)
            probabilities = np.ascontiguousarray(
                interval_probability(boundaries[:, :-1], boundaries[:, 1:]),
            )
            probability_seconds += perf_counter() - probability_began
            counts = native.accumulate_mixed_into(
                centers=centers,
                sigmas=sigmas,
                log_sigmas=logarithms,
                luminosity=luminosity,
                direct_rows=selected_rows,
                direct_probabilities=probabilities,
                output=channel_mass,
                start_row=first_cell,
                stop_row=stop_cell,
                delta_v=self.lookup_tables.delta_v,
                **self.native_parameters,
            )
            for name in counters:
                counters[name] += counts[name]
            first_cell = stop_cell
        return channel_mass, counters, probability_seconds

    def _original_window_probabilities(self, lower, upper):
        """Reuse saturated/full-cutoff masses; retain stable boundary integrals."""
        saturated = (lower <= -self._full_window_saturation) & (upper >= self._full_window_saturation)
        if np.any(saturated):
            full = np.ones(lower.shape, dtype=np.float64)
            remaining = ~saturated
            if np.any(remaining):
                full[remaining] = interval_probability(lower[remaining], upper[remaining])
        else:
            full = interval_probability(lower, upper)
        cutoff = self.lookup_tables.Z
        interior = (lower <= -cutoff) & (upper >= cutoff)
        if np.any(interior):
            clipped = np.full(lower.shape, self._clipped_interior_mass, dtype=np.float64)
            boundary = ~interior
            if np.any(boundary):
                clipped[boundary] = interval_probability(
                    np.clip(lower[boundary], -cutoff, cutoff),
                    np.clip(upper[boundary], -cutoff, cutoff),
                )
        else:
            clipped = interval_probability(
                np.clip(lower, -cutoff, cutoff),
                np.clip(upper, -cutoff, cutoff),
            )
        return full, clipped

    def integrate(self, centers, sigmas, luminosity):
        """Query original cells, multiply luminosities and accumulate profiles.

        query_interval_ns covers validation, coordinate location, template reads,
        interpolation, multiplication, reduction and channel-density conversion.
        Independent analytic original-cell window accounts are evaluated after
        that interval; both timings are retained separately.
        """
        query_started_ns = perf_counter_ns()
        owned_account_inputs = all(
            isinstance(values, np.ndarray) and values.dtype == np.float64
            and values.flags.c_contiguous and values.flags.owndata
            for values in (centers, sigmas, luminosity)
        )
        centers, sigmas, logarithms = self._prepare_coordinates(
            centers, sigmas, require_coverage=False,
        )
        luminosity = np.ascontiguousarray(luminosity, dtype=np.float64)
        if luminosity.ndim != 2 or luminosity.shape[0] != centers.size:
            raise ValueError("Luminosity must have shape(cell count, profile count)")
        below = sigmas < self.lookup_tables.sigma_nodes[0]
        above = sigmas > self.lookup_tables.sigma_nodes[-1]
        outside = below | above
        outside_rows = np.flatnonzero(outside).astype(np.int64, copy=False)
        probability_seconds = 0.0
        if outside_rows.size:
            channel_mass, counts, probability_seconds = self._integrate_mixed(
                centers, sigmas, logarithms, luminosity, outside_rows,
            )
        else:
            channel_mass, counts = native.accumulate(
                centers=centers,
                sigmas=sigmas,
                log_sigmas=logarithms,
                luminosity=luminosity,
                delta_v=self.lookup_tables.delta_v,
                **self.native_parameters,
            )
        channel_mass = np.ascontiguousarray(channel_mass.T)
        channel_mass /= self.lookup_tables.delta_v
        if not np.isfinite(channel_mass).all():
            raise FloatingPointError("Compiled template accumulation produced a nonfinite profile")
        query_finished_ns = perf_counter_ns()

        accounting_began = perf_counter()
        positive = luminosity > 0.0
        active = np.any(positive, axis=1)
        if owned_account_inputs and np.all(active):
            account_centers = centers
            account_sigmas = sigmas
            account_luminosity = luminosity
        else:
            account_centers = centers[active]
            account_sigmas = sigmas[active]
            account_luminosity = luminosity[active]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            lower = (self.lookup_tables.edges[0] - account_centers) / account_sigmas
            upper = (self.lookup_tables.edges[-1] - account_centers) / account_sigmas
        full_probability, clipped_probability = self._original_window_probabilities(
            lower, upper,
        )
        full_window = np.einsum("i,ip->p", full_probability, account_luminosity, optimize=False)
        clipped_window = np.einsum("i,ip->p", clipped_probability, account_luminosity, optimize=False)
        accounting_seconds = perf_counter() - accounting_began
        if not np.isfinite(full_window).all() or not np.isfinite(clipped_window).all():
            raise FloatingPointError("Original-cell analytic window accounts became nonfinite")
        coverage_began = perf_counter()
        outside_accounts = outside[active]
        coverage = TemplateGroupCoverage(
            rows=int(centers.size),
            emitting_rows=int(np.count_nonzero(active)),
            outside_rows=int(outside_rows.size),
            outside_emitting_rows=int(np.count_nonzero(outside & active)),
            below_range_rows=int(np.count_nonzero(below)),
            above_range_rows=int(np.count_nonzero(above)),
            below_range_emitting_rows=int(np.count_nonzero(below & active)),
            above_range_emitting_rows=int(np.count_nonzero(above & active)),
            sigma_range=sigma_range(sigmas),
            emitting_sigma_range=sigma_range(account_sigmas),
            outside_sigma_range=sigma_range(sigmas[outside]),
            profile_emitting_rows=np.count_nonzero(positive, axis=0),
            outside_profile_emitting_rows=np.count_nonzero(positive[outside], axis=0),
            outside_input_luminosity=np.sum(luminosity[outside], axis=0, dtype=np.float64),
            outside_window_luminosity_reference=np.einsum(
                "i,ip->p", clipped_probability[outside_accounts],
                account_luminosity[outside_accounts], optimize=False,
            ),
        )
        coverage_seconds = perf_counter() - coverage_began
        statistics = {
            "group_calls": 1,
            "kernel_rows": int(counts["kernel_rows"]),
            "active_rows": int(counts["active_rows"]),
            "support_bins_visited": int(counts["support_bins_visited"]),
            "template_corner_values_gathered": int(counts["template_corner_values_read"]),
            "product_updates": int(counts["product_updates"]),
            "integration_seconds": (query_finished_ns - query_started_ns) * 1e-9,
            "independent_window_accounting_seconds": accounting_seconds,
            "template_probability_values_generated": int(counts["support_bins_visited"] - counts.get("direct_support_bins_visited", 0)),
            "template_subbatches": max(1, (outside_rows.size + self.maximum_direct_probability_rows - 1) // self.maximum_direct_probability_rows),
            "template_online_cdf_queries": 0,
            "fallback_rows": int(outside_rows.size),
            "fallback_active_rows": int(counts.get("direct_active_rows", 0)),
            "fallback_probability_intervals": int(outside_rows.size * self.lookup_tables.K),
            "fallback_probability_batches": int((outside_rows.size + self.maximum_direct_probability_rows - 1) // self.maximum_direct_probability_rows),
            "fallback_probability_seconds": probability_seconds,
            "fallback_product_updates": int(counts.get("direct_product_updates", 0)),
            "coverage_statistics_seconds": coverage_seconds,
        }
        return TemplateGroupResult(
            profiles=channel_mass,
            full_window=full_window,
            clipped_window=clipped_window,
            statistics=statistics,
            query_interval_ns=(query_started_ns, query_finished_ns),
            coverage=coverage,
        )


def prepare_template_engine(grid_parameters=None):
    """Build and verify the shared probability table before the process clock."""
    began = perf_counter()
    parameters = {} if grid_parameters is None else dict(grid_parameters)
    table = FractionalGaussianTable(**parameters)
    engine = TemplateEngine(table=table)
    return engine, perf_counter() - began
