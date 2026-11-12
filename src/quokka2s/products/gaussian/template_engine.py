"""Integrate original-cell Gaussians using immutable preintegrated templates.

NumPy locates four phase/log-sigma corners, reads their stored channel masses
in bounded subbatches and reduces luminosity-weighted spectra. Valid
outside widths use stable direct bin integrals with the same five-sigma cutoff.
Each call owns its output; concurrent workers share only the immutable table.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
from time import perf_counter, perf_counter_ns

import numpy as np

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
    maximum_probability_rows = 1024
    maximum_direct_probability_rows = maximum_probability_rows
    cutoff_applies_to = "Stored template corners in range; original-cell Gaussians for direct integration and references"

    def __init__(self, *, table):
        if not isinstance(table, FractionalGaussianTable):
            raise TypeError("Gaussian templates require FractionalGaussianTable")
        table.validate_prepared_state()
        if not 1 <= table.K <= 400:
            raise ValueError("Require one to 400 template output channels")
        self.lookup_tables = table
        self.portable_backend = "numpy-original-cell-templates"
        self._corner_specs = {}
        self._corner_spec_lock = threading.Lock()
        self._corner_worker_context = threading.local()
        # Reuse exact results from the existing reference, not a new tail rule.
        self._clipped_interior_mass = float(interval_probability(-table.Z, table.Z))
        self._full_window_saturation = 12.0
        probe_lower = np.array([-np.inf, -12.0, np.nextafter(-12.0, -np.inf)])
        probe_upper = np.array([np.inf, 12.0, np.nextafter(12.0, np.inf)])
        probe = interval_probability(probe_lower, probe_upper)
        if probe.tobytes() != np.ones(probe.shape, dtype=np.float64).tobytes():
            raise RuntimeError("Original-cell interval reference does not saturate at +/-12")

    def prepare_corner_store(self, profile_count=40):
        """Share immutable corner/shift indexing; allocate no snapshot luminosity."""
        if type(profile_count) is not int or profile_count < 1:
            raise ValueError("Corner-store profile_count must be a positive integer")
        if profile_count not in self._corner_specs:
            with self._corner_spec_lock:
                if profile_count not in self._corner_specs:
                    from .corner_store import RaggedCornerSpec
                    self._corner_specs[profile_count] = RaggedCornerSpec(
                        self.lookup_tables,
                        profile_count=profile_count,
                    )
        return self._corner_specs[profile_count]

    def corner_deposit_engine(self, profile_count=40):
        """Reuse call-local deposit engines per worker and profile layout."""
        if not hasattr(self._corner_worker_context, "engines"):
            self._corner_worker_context.engines = {}
        engines = self._corner_worker_context.engines
        if profile_count not in engines:
            from .corner_store import CornerDepositEngine
            engines[profile_count] = CornerDepositEngine(
                spec=self.prepare_corner_store(profile_count=profile_count),
            )
        return engines[profile_count]

    def validate_prepared_state(self):
        """Refuse table mutation after preparation outside the process clock."""
        self.lookup_tables.validate_prepared_state()

    def _prepare_coordinates(self, centers, sigmas, *, require_coverage=True):
        """Prepare contiguous float64 coordinates and validate safe shifts."""
        centers = np.ascontiguousarray(centers, dtype=np.float64)
        sigmas = np.ascontiguousarray(sigmas, dtype=np.float64)
        if centers.ndim != 1 or sigmas.shape != centers.shape:
            raise ValueError("Gaussian means and widths must be matching vectors")
        if not np.isfinite(centers).all():
            raise ValueError("Gaussian means must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            coordinates = centers / self.lookup_tables.delta_v
        if not np.isfinite(coordinates).all() or np.any(np.abs(coordinates) >= 2**62):
            raise ValueError("Gaussian integer-bin shifts must fit safe signed int64 coordinates")
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
        return centers, sigmas

    def evaluate(self, centers, sigmas):
        """Return template probabilities[N,K] for bounded diagnostic inputs."""
        centers, sigmas = self._prepare_coordinates(centers, sigmas)
        return self.lookup_tables.evaluate(
            centers,
            sigmas,
            chunk_size=self.maximum_probability_rows,
        )

    def _accumulate_probabilities(self, centers, sigmas, luminosity, outside):
        """Reduce bounded [cell,channel] probabilities without a [cell,K,P] array.

        In-range cells retain the prepared four-corner interpolation and each
        corner's stored five-sigma support. Outside widths use the existing
        stable SciPy interval integral, clipped at the original cell's five
        sigma. Only the floating-point luminosity summation order changes.
        """
        table = self.lookup_tables
        output = np.zeros((luminosity.shape[1], table.K), dtype=np.float64)
        counters = {
            "kernel_rows": centers.size,
            "active_rows": 0,
            "support_bins_visited": 0,
            "template_corner_values_read": 0,
            "product_updates": 0,
            "direct_active_rows": 0,
            "direct_support_bins_visited": 0,
            "direct_product_updates": 0,
            "probability_subbatches": 0,
            "direct_probability_batches": 0,
        }
        probability_seconds = 0.0
        for start in range(0, centers.size, self.maximum_probability_rows):
            stop = min(start + self.maximum_probability_rows, centers.size)
            active = np.any(luminosity[start:stop] > 0.0, axis=1)
            if not np.any(active):
                continue
            selected_centers = centers[start:stop][active]
            selected_sigmas = sigmas[start:stop][active]
            selected_luminosity = luminosity[start:stop][active]
            direct = outside[start:stop][active]
            probabilities = np.empty((selected_centers.size, table.K), dtype=np.float64)
            covered = ~direct
            if np.any(covered):
                probabilities[covered] = table.evaluate(
                    selected_centers[covered],
                    selected_sigmas[covered],
                    chunk_size=self.maximum_probability_rows,
                )
            direct_count = int(np.count_nonzero(direct))
            if direct_count:
                began = perf_counter()
                with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                    boundaries = (
                        table.edges[None, :] - selected_centers[direct, None]
                    ) / selected_sigmas[direct, None]
                np.clip(boundaries, -table.Z, table.Z, out=boundaries)
                probabilities[direct] = interval_probability(
                    boundaries[:, :-1],
                    boundaries[:, 1:],
                )
                probability_seconds += perf_counter() - began
                counters["direct_probability_batches"] += 1
            output += np.einsum("ik,ip->pk", probabilities, selected_luminosity, optimize=False)
            rows = selected_centers.size
            counters["active_rows"] += rows
            counters["support_bins_visited"] += rows * table.K
            counters["template_corner_values_read"] += (rows - direct_count) * 4 * table.K
            counters["product_updates"] += rows * table.K * luminosity.shape[1]
            counters["direct_active_rows"] += direct_count
            counters["direct_support_bins_visited"] += direct_count * table.K
            counters["direct_product_updates"] += direct_count * table.K * luminosity.shape[1]
            counters["probability_subbatches"] += 1
        return output, counters, probability_seconds

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
        centers, sigmas = self._prepare_coordinates(
            centers, sigmas, require_coverage=False,
        )
        luminosity = np.ascontiguousarray(luminosity, dtype=np.float64)
        if luminosity.ndim != 2 or luminosity.shape[0] != centers.size:
            raise ValueError("Luminosity must have shape(cell count, profile count)")
        if not 1 <= luminosity.shape[1] <= 12:
            raise ValueError("Require one to twelve luminosity accounts")
        if not np.isfinite(luminosity).all() or np.any(luminosity < 0.0):
            raise ValueError("Luminosity must be finite and nonnegative")
        below = sigmas < self.lookup_tables.sigma_nodes[0]
        above = sigmas > self.lookup_tables.sigma_nodes[-1]
        outside = below | above
        outside_rows = np.flatnonzero(outside).astype(np.int64, copy=False)
        channel_mass, counts, probability_seconds = self._accumulate_probabilities(
            centers, sigmas, luminosity, outside,
        )
        channel_mass = np.ascontiguousarray(channel_mass.T)
        channel_mass /= self.lookup_tables.delta_v
        if not np.isfinite(channel_mass).all():
            raise FloatingPointError("NumPy template accumulation produced a nonfinite profile")
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
            "template_subbatches": int(counts["probability_subbatches"]),
            "template_online_cdf_queries": 0,
            "fallback_rows": int(outside_rows.size),
            "fallback_active_rows": int(counts.get("direct_active_rows", 0)),
            "fallback_probability_intervals": int(counts["direct_support_bins_visited"]),
            "fallback_probability_batches": int(counts["direct_probability_batches"]),
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
    engine.prepare_corner_store()
    return engine, perf_counter() - began
