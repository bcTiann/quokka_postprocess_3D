"""Share immutable templates, worker preparation and original-cell accounts.

Workers prepare covered-cell corner contributions for the snapshot luminosity
store. Only valid outside-table widths use direct channel integration.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
from time import perf_counter, perf_counter_ns

import numpy as np

from .intervals import (
    _NARROW_INTERVAL_THRESHOLD,
    _QUADRATURE_NODES,
    _QUADRATURE_WEIGHTS,
    _SQRT_TWO,
    _SQRT_TWO_PI,
    interval_probability,
)
from ._gaussian_kernels import original_window_probabilities
from .template_coverage import TemplateGroupCoverage, sigma_range
from .template_table import FractionalGaussianTable


@dataclass(frozen=True)
class TemplateGroupResult:
    """Route profiles[K,P], original-cell window totals[P] and activity timing.

    Profiles use erg/s/(km/s), and window totals use erg/s. Profiles contain
    only directly integrated outside-width light; covered contributions are
    carried by the separate corner increment.
    """

    profiles: np.ndarray
    full_window: np.ndarray
    clipped_window: np.ndarray
    statistics: dict
    query_interval_ns: tuple[int, int]
    coverage: TemplateGroupCoverage


class TemplateEngine:
    """Coordinate immutable tables and thread-local original-cell preparation.

    N is the current width group's original-cell count. Means[N] and widths[N]
    use km/s; luminosity[N,P] uses erg/s, with one to twelve local accounts P.
    Output profiles[K,P] use erg/s/(km/s), with at most 400 velocity channels K.
    CornerPreparationEngine owns each worker's mutable contribution buffers.
    Shared validation and analytic accounting remain here.
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

    def prepare_corner_layout(self, profile_count=40):
        """Share immutable template-row/shift indexing without allocating H.

        Each row's K+B-1 valid integer shifts map into a flat slot range.
        The main RaggedCornerStore later owns H[global profile, flat slot].
        """
        if type(profile_count) is not int or profile_count < 1:
            raise ValueError("Corner-store profile_count must be a positive integer")
        if profile_count not in self._corner_specs:
            with self._corner_spec_lock:
                if profile_count not in self._corner_specs:
                    from .corner_store import RaggedCornerSpec
                    self._corner_specs[profile_count] = RaggedCornerSpec(
                        table=self.lookup_tables,
                        profile_count=profile_count,
                    )
        return self._corner_specs[profile_count]

    def corner_preparation_engine(self, profile_count=40):
        """Reuse one mutable preparation engine per worker and account layout."""
        if not hasattr(self._corner_worker_context, "engines"):
            self._corner_worker_context.engines = {}
        engines = self._corner_worker_context.engines
        if profile_count not in engines:
            from .corner_store import CornerPreparationEngine
            engines[profile_count] = CornerPreparationEngine(
                spec=self.prepare_corner_layout(profile_count=profile_count),
            )
        return engines[profile_count]

    def validate_prepared_state(self):
        """Refuse table mutation after preparation outside the process clock."""
        self.lookup_tables.validate_prepared_state()

    def _prepare_coordinates(self, centers, sigmas):
        """Prepare contiguous float64 coordinates and validate safe shifts."""
        centers = np.ascontiguousarray(centers, dtype=np.float64)
        sigmas = np.ascontiguousarray(sigmas, dtype=np.float64)
        if centers.ndim != 1 or sigmas.shape != centers.shape:
            raise ValueError("Gaussian means and widths must be matching vectors")
        if not np.isfinite(centers).all():
            raise ValueError("Gaussian means must be finite")
        if self.lookup_tables.delta_v == 1.0:
            # The usual 1-km/s channels need no second coordinate allocation.
            coordinates = centers
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                coordinates = centers / self.lookup_tables.delta_v
            if not np.isfinite(coordinates).all():
                raise ValueError("Gaussian integer-bin shifts must fit safe signed int64 coordinates")
        if np.any(coordinates <= -(2**62)) or np.any(coordinates >= 2**62):
            raise ValueError("Gaussian integer-bin shifts must fit safe signed int64 coordinates")
        if not np.isfinite(sigmas).all() or np.any(sigmas <= 0.0):
            raise ValueError("Gaussian sigma must be finite and positive")
        return centers, sigmas

    def _integrate_outside_widths(self, centers, sigmas, luminosity):
        """Integrate selected outside-table widths in bounded channel arrays.

        The caller supplies only valid out-of-coverage rows, in original order.
        Standardized channel edges are clipped at five sigma without restoring
        discarded tails. Each output account keeps its existing reduction order.
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
            direct_count = selected_centers.size
            began = perf_counter()
            with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                boundaries = (
                    table.edges[None, :] - selected_centers[:, None]
                ) / selected_sigmas[:, None]
            np.clip(boundaries, -table.Z, table.Z, out=boundaries)
            probabilities = interval_probability(
                boundaries[:, :-1],
                boundaries[:, 1:],
            )
            probability_seconds += perf_counter() - began
            counters["direct_probability_batches"] += 1
            output += np.einsum("ik,ip->pk", probabilities, selected_luminosity, optimize=False)
            rows = selected_centers.size
            counters["active_rows"] += rows
            counters["support_bins_visited"] += rows * table.K
            counters["product_updates"] += rows * table.K * luminosity.shape[1]
            counters["direct_active_rows"] += direct_count
            counters["direct_support_bins_visited"] += direct_count * table.K
            counters["direct_product_updates"] += direct_count * table.K * luminosity.shape[1]
            counters["probability_subbatches"] += 1
        return output, counters, probability_seconds

    def _original_window_probabilities(self, centers, sigmas):
        """Standardize original cells and retain stable full/clipped probabilities."""
        return original_window_probabilities(
            centers=centers,
            sigmas=sigmas,
            lower_edge=self.lookup_tables.edges[0],
            upper_edge=self.lookup_tables.edges[-1],
            cutoff_sigma=self.lookup_tables.Z,
            full_saturation=self._full_window_saturation,
            clipped_interior_mass=self._clipped_interior_mass,
            quadrature_nodes=_QUADRATURE_NODES,
            quadrature_weights=_QUADRATURE_WEIGHTS,
            sqrt_two=_SQRT_TWO,
            sqrt_two_pi=_SQRT_TWO_PI,
            narrow_threshold=_NARROW_INTERVAL_THRESHOLD,
        )

    def _prepare_group_result(self, centers, sigmas, luminosity):
        """Validate one worker group and retain its original-cell accounts.

        CornerPreparationEngine.prepare_group() returns raw covered-corner terms
        separately from the directly integrated outside-width profiles here.
        query_interval_ns covers validation and contribution preparation;
        independent original-cell endpoint accounts are timed separately after it.
        """
        query_started_ns = perf_counter_ns()
        centers, sigmas = self._prepare_coordinates(centers, sigmas)
        luminosity = np.asfortranarray(luminosity, dtype=np.float64)
        if luminosity.ndim != 2 or luminosity.shape[0] != centers.size:
            raise ValueError("Luminosity must have shape(cell count, profile count)")
        if not 1 <= luminosity.shape[1] <= 12:
            raise ValueError("Require one to twelve luminosity accounts")
        if not np.isfinite(luminosity).all() or np.any(luminosity < 0.0):
            raise ValueError("Luminosity must be finite and nonnegative")
        below = sigmas < self.lookup_tables.sigma_nodes[0]
        above = sigmas > self.lookup_tables.sigma_nodes[-1]
        outside = below | above
        outside_count = int(np.count_nonzero(outside))
        positive = luminosity > 0.0
        active = np.any(positive, axis=1)
        all_active = bool(np.all(active))
        channel_mass, counts, probability_seconds = self._prepare_contributions(
            centers, sigmas, luminosity, outside, active,
        )
        channel_mass = np.ascontiguousarray(channel_mass.T)
        channel_mass /= self.lookup_tables.delta_v
        if not np.isfinite(channel_mass).all():
            raise FloatingPointError("NumPy template accumulation produced a nonfinite profile")
        query_finished_ns = perf_counter_ns()

        accounting_began = perf_counter()
        if all_active:
            account_centers = centers
            account_sigmas = sigmas
            account_luminosity = luminosity
        else:
            account_centers = centers[active]
            account_sigmas = sigmas[active]
            account_luminosity = luminosity[active]
        full_probability, clipped_probability = self._original_window_probabilities(
            account_centers, account_sigmas,
        )
        full_window = np.dot(full_probability, account_luminosity)
        clipped_window = np.dot(clipped_probability, account_luminosity)
        accounting_seconds = perf_counter() - accounting_began
        if not np.isfinite(full_window).all() or not np.isfinite(clipped_window).all():
            raise FloatingPointError("Original-cell analytic window accounts became nonfinite")
        coverage_began = perf_counter()
        all_sigma_range = sigma_range(sigmas)
        emitting_sigma_range = all_sigma_range if all_active else sigma_range(account_sigmas)
        if outside_count:
            outside_accounts = outside[active]
            outside_emitting_rows = int(np.count_nonzero(outside & active))
            below_range_rows = int(np.count_nonzero(below))
            above_range_rows = int(np.count_nonzero(above))
            below_range_emitting_rows = int(np.count_nonzero(below & active))
            above_range_emitting_rows = int(np.count_nonzero(above & active))
            outside_sigma_range = sigma_range(sigmas[outside])
            outside_profile_emitting_rows = np.count_nonzero(positive[outside], axis=0)
            outside_input_luminosity = np.sum(luminosity[outside], axis=0, dtype=np.float64)
            outside_window_luminosity_reference = np.einsum(
                "i,ip->p", clipped_probability[outside_accounts],
                account_luminosity[outside_accounts], optimize=False,
            )
        else:
            # Covered-only groups have no outside cells to gather or integrate.
            outside_emitting_rows = 0
            below_range_rows = 0
            above_range_rows = 0
            below_range_emitting_rows = 0
            above_range_emitting_rows = 0
            outside_sigma_range = None
            outside_profile_emitting_rows = np.zeros(luminosity.shape[1], dtype=np.int64)
            outside_input_luminosity = np.zeros(luminosity.shape[1], dtype=np.float64)
            outside_window_luminosity_reference = np.zeros(luminosity.shape[1], dtype=np.float64)
        coverage = TemplateGroupCoverage(
            rows=int(centers.size),
            emitting_rows=int(np.count_nonzero(active)),
            outside_rows=outside_count,
            outside_emitting_rows=outside_emitting_rows,
            below_range_rows=below_range_rows,
            above_range_rows=above_range_rows,
            below_range_emitting_rows=below_range_emitting_rows,
            above_range_emitting_rows=above_range_emitting_rows,
            sigma_range=all_sigma_range,
            emitting_sigma_range=emitting_sigma_range,
            outside_sigma_range=outside_sigma_range,
            profile_emitting_rows=np.count_nonzero(positive, axis=0),
            outside_profile_emitting_rows=outside_profile_emitting_rows,
            outside_input_luminosity=outside_input_luminosity,
            outside_window_luminosity_reference=outside_window_luminosity_reference,
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
            "fallback_rows": outside_count,
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
    from .corner_scatter import prepare_corner_scatter
    prepare_corner_scatter()
    parameters = {} if grid_parameters is None else dict(grid_parameters)
    table = FractionalGaussianTable(**parameters)
    engine = TemplateEngine(table=table)
    engine.prepare_corner_layout()
    return engine, perf_counter() - began
