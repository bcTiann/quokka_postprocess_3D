"""Accumulate original-cell spectra from immutable integrated-bin templates.

Workers keep one bounded batch's owned mu/sigma/luminosity vectors until its
Gaussian wave finishes. Only small profiles, endpoint accounts and complete
moments are merged into the main accumulator, in the original batch order.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter, perf_counter_ns
from types import SimpleNamespace

import numpy as np

from .accounting import add_cutoff_accounts, update_cutoff_report
from .spectra import HistogramIntegratedSpectra
from .template_coverage import TemplateCoverageTotals, print_template_coverage
from ..integrated_spectra import IntegratedSpectra
from ..line_velocity_moments import MINIMUM_GAUSSIAN_WIDTH_KMS


def elapsed_union_ns(intervals):
    """Count overlapping half-open clock intervals once, retaining exact ns."""
    total = 0
    previous_stop = None
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("Gaussian timing interval is reversed")
        if previous_stop is None or start > previous_stop:
            total += stop - start
            previous_stop = stop
        elif stop > previous_stop:
            total += stop - previous_stop
            previous_stop = stop
    return total


@dataclass(frozen=True)
class GaussianBatchGroup:
    """Owned km/s vectors[N] and luminosity accounts[N,P] in erg/s."""

    profile_ids: np.ndarray
    centers: np.ndarray
    sigmas: np.ndarray
    luminosity: np.ndarray

    @property
    def array_bytes(self):
        return self.profile_ids.nbytes + self.centers.nbytes + self.sigmas.nbytes + self.luminosity.nbytes


class TemplateIntegratedSpectra(HistogramIntegratedSpectra):
    """Reuse established cell grouping and moments without node deposition."""

    def __init__(self, line_keys, velocity_edges_kms, *, engine, batch_partial=False,
                 workers=1, cell_chunk=8192, merge_threads=None):
        if merge_threads is not None and (type(merge_threads) is not int or merge_threads != 1):
            raise ValueError("Direct templates merge ordered batch profiles with one main thread")
        if workers != 1:
            raise ValueError("Direct templates use one native integration thread per batch")
        IntegratedSpectra.__init__(
            self,
            line_keys=line_keys,
            velocity_edges_kms=velocity_edges_kms,
            workers=workers,
            cell_chunk=cell_chunk,
            spectral_method="direct-template",
        )
        if not np.array_equal(self.velocity_edges_kms, engine.lookup_tables.edges):
            raise ValueError("Spectral edges differ from the prepared integrated templates")
        self.engine = engine
        self.batch_partial = batch_partial
        self.profile_count = 2 * len(self.line_keys) * 2
        self._finalized = False
        self._full_window_luminosity = np.zeros(self.profile_count, dtype=np.float64)
        self._clipped_window_luminosity = np.zeros(self.profile_count, dtype=np.float64)
        self._coverage = TemplateCoverageTotals(
            profile_count=self.profile_count,
            table_minimum=engine.lookup_tables.sigma_nodes[0],
            table_maximum=engine.lookup_tables.sigma_nodes[-1],
        )
        self._groups = []
        self._owned_input_bytes = 0
        self._groups_prepared = 0
        self._groups_finished = 0
        self._queue_finished = False
        self._queue_failed = False
        self._query_intervals = []
        self._addition_intervals = []
        self._timing = {
            "query_and_profile_addition_ns": 0,
            "query_call_elapsed_sum_ns": 0,
            "query_calls": 0,
            "physical_preparation_wave_ns": 0,
            "gaussian_wave_ns": 0,
            "ordered_product_merge_wave_ns": 0,
            "maximum_retained_gaussian_input_bytes": 0,
            "waves": 0,
        }
        self.direct_statistics = {
            "group_calls": 0,
            "kernel_rows": 0,
            "integration_seconds": 0.0,
            "independent_window_accounting_seconds": 0.0,
            "spectral_add_batch_seconds": 0.0,
            "ordered_profile_merge_seconds": 0.0,
            "fallback_probability_seconds": 0.0,
            "fallback_rows": 0,
            "fallback_active_rows": 0,
            "fallback_probability_intervals": 0,
            "fallback_probability_batches": 0,
            "fallback_product_updates": 0,
            "coverage_statistics_seconds": 0.0,
        }

    def add_batch(self, cells, emission):
        """Prepare unchanged masks, full moments and a bounded Gaussian queue."""
        if not self.batch_partial or self._queue_finished or self._queue_failed:
            raise RuntimeError("Prepare each template batch through a fresh worker partial")
        began = perf_counter()
        try:
            return IntegratedSpectra.add_batch(self, cells=cells, emission=emission)
        finally:
            self.direct_statistics["spectral_add_batch_seconds"] += perf_counter() - began

    def accumulate_group_histogram(
        self, *, line_keys, branch_index, velocity_kms, thermal_width_kms,
        luminosity_matrix, velocity_edges_kms, cell_chunk, workers, spectral_method,
    ):
        """Implement the shared width-group hook by retaining original cells.

        The hook name is inherited from the shared optimized grouping code; no
        histogram or interpolation-node luminosities are allocated here.
        """
        if not self.batch_partial or self._queue_finished or self._queue_failed:
            raise RuntimeError("Cannot append Gaussians after this worker queue finishes")
        widths = np.maximum(thermal_width_kms, MINIMUM_GAUSSIAN_WIDTH_KMS)
        group = GaussianBatchGroup(
            profile_ids=self._profile_ids(line_keys, branch_index),
            centers=np.array(velocity_kms, dtype=np.float64, order="C", copy=True),
            sigmas=np.array(widths, dtype=np.float64, order="C", copy=True),
            luminosity=np.array(luminosity_matrix, dtype=np.float64, order="C", copy=True),
        )
        self._groups.append(group)
        self._owned_input_bytes += group.array_bytes
        self._groups_prepared += 1
        # Shared preparation still computes every original-cell moment and L sum.
        return np.zeros((self.velocity_edges_kms.size - 1, luminosity_matrix.shape[1]))

    def finish_gaussians(self):
        """Query each prepared original-cell group once, then release its inputs."""
        if not self.batch_partial or self._queue_finished or self._queue_failed:
            raise RuntimeError("Finish each prepared Gaussian worker queue exactly once")
        try:
            for group in self._groups:
                result = self.engine.integrate(group.centers, group.sigmas, group.luminosity)
                expected_shape = (self.velocity_edges_kms.size - 1, group.profile_ids.size)
                if result.profiles.shape != expected_shape:
                    raise ValueError("Gaussian result does not match its original profile IDs")
                self._query_intervals.append(result.query_interval_ns)
                addition_started = perf_counter_ns()
                profiles = self.dL_dv.reshape(self.profile_count, -1)
                profiles[group.profile_ids] += result.profiles.T
                self._addition_intervals.append((addition_started, perf_counter_ns()))
                self._full_window_luminosity[group.profile_ids] += result.full_window
                self._clipped_window_luminosity[group.profile_ids] += result.clipped_window
                for name, value in result.statistics.items():
                    self.direct_statistics[name] = self.direct_statistics.get(name, 0) + value
                self._coverage.add_group(result.coverage, group.profile_ids)
                self._groups_finished += 1
            if self._groups_finished != self._groups_prepared:
                raise AssertionError("A prepared original-cell Gaussian group was omitted")
            self._queue_finished = True
        except BaseException:
            self._queue_failed = True
            raise
        finally:
            self.release_gaussian_inputs()
        return self

    def release_gaussian_inputs(self):
        """Drop owned input vectors after success or failure of their bounded wave."""
        self._groups.clear()
        self._owned_input_bytes = 0

    def merge(self, other):
        """Preserve ordered profile, luminosity, count, moment and endpoint sums."""
        if self.batch_partial or self._finalized or other._finalized:
            raise RuntimeError("Merge unfinalized template partials into the main accumulator")
        if self.engine is not other.engine or self.line_keys != other.line_keys:
            raise ValueError("Cannot merge different template engines or line orders")
        if not other._queue_finished or other._queue_failed or other._groups:
            raise RuntimeError("Finish every Gaussian group before the ordered global merge")
        if self.spectral_method != other.spectral_method:
            raise ValueError("Cannot merge spectra calculated with different methods")
        began = perf_counter()
        addition_started = perf_counter_ns()
        self.dL_dv += other.dL_dv
        self._addition_intervals.append((addition_started, perf_counter_ns()))
        self.input_luminosity += other.input_luminosity
        self.cell_counts += other.cell_counts
        self.merge_full_line_moments(moments=other.full_line_moments, line_keys=self.line_keys)
        self._full_window_luminosity += other._full_window_luminosity
        self._clipped_window_luminosity += other._clipped_window_luminosity
        self._coverage.merge(other._coverage)
        for name, value in other.direct_statistics.items():
            self.direct_statistics[name] = self.direct_statistics.get(name, 0) + value
        self.direct_statistics["ordered_profile_merge_seconds"] += perf_counter() - began
        self._query_intervals.extend(other._query_intervals)
        self._addition_intervals.extend(other._addition_intervals)
        other._query_intervals.clear()
        other._addition_intervals.clear()
        return self

    def record_completed_wave(self, *, physical_ns, gaussian_ns, merge_ns, owned_input_bytes):
        """Summarize nonoverlapping waves and release their small timing lists."""
        self._timing["query_and_profile_addition_ns"] += elapsed_union_ns(
            self._query_intervals + self._addition_intervals,
        )
        self._timing["query_call_elapsed_sum_ns"] += sum(
            stop - start for start, stop in self._query_intervals
        )
        self._timing["query_calls"] += len(self._query_intervals)
        self._timing["physical_preparation_wave_ns"] += physical_ns
        self._timing["gaussian_wave_ns"] += gaussian_ns
        self._timing["ordered_product_merge_wave_ns"] += merge_ns
        self._timing["maximum_retained_gaussian_input_bytes"] = max(
            self._timing["maximum_retained_gaussian_input_bytes"], owned_input_bytes,
        )
        self._timing["waves"] += 1
        self._query_intervals.clear()
        self._addition_intervals.clear()

    def timing_summary(self):
        """Return actual activity union, broader wave wall and diagnostic call sums."""
        if self._query_intervals or self._addition_intervals:
            raise RuntimeError("Summarize template timing after its ordered wave merge")
        timing = self._timing
        return {
            "query_and_profile_addition_seconds": timing["query_and_profile_addition_ns"] / 1e9,
            "query_and_profile_addition_ns": timing["query_and_profile_addition_ns"],
            "query_call_elapsed_sum_seconds": timing["query_call_elapsed_sum_ns"] / 1e9,
            "query_calls": timing["query_calls"],
            "width_group_rows": int(self.direct_statistics["kernel_rows"]),
            "gaussian_wave_seconds_including_analytic_accounts_and_scheduling": timing["gaussian_wave_ns"] / 1e9,
            "physical_preparation_wave_seconds": timing["physical_preparation_wave_ns"] / 1e9,
            "ordered_product_merge_wave_seconds": timing["ordered_product_merge_wave_ns"] / 1e9,
            "maximum_retained_gaussian_input_bytes": timing["maximum_retained_gaussian_input_bytes"],
            "waves": timing["waves"],
            "all_deferred_groups_finished": timing["query_calls"] == self.direct_statistics["group_calls"],
            "activity_scope": "mu/sigma/L checks, template lookup/interpolation, L multiplication, group and main spectrum additions; overlapping calls count once",
            "wave_scope": "Gaussian task dispatch, queries, independent original-cell analytic endpoint accounts, result bookkeeping and input release",
            "worker_call_sum_is_not_wall_time": True,
            "whole_snapshot_cell_arrays_retained": False,
        }

    def build_output(self, projected_area_cm2):
        """Save standard products and independent original-cell cutoff references."""
        if self.batch_partial:
            raise RuntimeError("Merge worker profiles before building template output")
        self.engine.validate_prepared_state()
        self._finalized = True
        payload, report = IntegratedSpectra.build_output(self, projected_area_cm2=projected_area_cm2)
        coverage = self._coverage.summary(
            line_keys=self.line_keys,
            input_luminosity=self.input_luminosity,
            clipped_window_luminosity=self._clipped_window_luminosity,
        )
        statistics = {
            **self.direct_statistics,
            "coordinate_rule": "original-cell-template-query",
            "histogram_allocated": False,
            "node_records_created": 0,
            "cell_by_channel_probability_matrix_allocated": self._coverage.counts["outside_rows"] > 0,
            "probability_matrix_scope": "Only bounded out-of-table rows; never table-covered cells or a whole snapshot",
            "maximum_direct_probability_rows": self.engine.maximum_direct_probability_rows,
            "per_cell_probability_buffer_bytes": 400 * 8,
            "integration": "Original-cell four-corner template query; valid out-of-table widths use stable direct bin integrals",
            "cutoff_sigma": 5.0,
            "cutoff_applies_to": self.engine.cutoff_applies_to,
            "renormalized": False,
            "accounting_scope": "Independent original-cell analytic endpoint intervals",
            "timing_scope": "integration/accounting/add_batch durations sum overlapping worker calls; see processing_timings.gaussian for elapsed union and wave wall",
            "spectral_add_batch_scope": "Shared availability masks, full moments, luminosity packing and owned queue copies; Gaussian queries occur in the next wave",
            "lookup_table": self.engine.lookup_tables.info(),
            "portable_backend": self.engine.portable_backend,
        }
        accounting_engine = SimpleNamespace(
            coordinate_rule="original-cell",
            cutoff_applies_to=self.engine.cutoff_applies_to,
            lookup_tables=self.engine.lookup_tables,
            expansion_statistics=statistics,
            untruncated_window_luminosity=self._full_window_luminosity,
            clipped_window_luminosity=self._clipped_window_luminosity,
        )
        audit = add_cutoff_accounts(payload, engine=accounting_engine, cutoff_sigma=5.0)
        update_cutoff_report(report, payload, engine=accounting_engine, audit=audit)
        for dust_state, dust_report in report.items():
            dust_report["spectral_method"] = "direct-template"
            dust_report["direct_statistics"] = statistics
            dust_report["template_coverage"] = {
                **coverage,
                "profiles": [entry for entry in coverage["profiles"] if entry["dust_state"] == dust_state],
            }
            dust_report["profile"] = (
                "Original-cell four-corner phase/log-sigma templates inside coverage; "
                "original mu/sigma stable bin integration outside sigma coverage"
            )
            dust_report["cutoff_applies_to"] = self.engine.cutoff_applies_to
            dust_report["cell_integration"] = "preintegrated-templates-with-direct-outside-sigma-integration"
            dust_report["fallback"] = (
                "All valid out-of-table widths use stable original-cell bin integrals; "
                "same +/-5sigma cutoff, no renormalization or luminosity omission threshold"
            )
            dust_report["merge_backend"] = {
                "selected_threads": 1,
                "batch_group_order_unchanged": True,
                "storage": "Small channel profiles and endpoint vectors; no parameter-node accumulator",
            }
        print_template_coverage(coverage, self.direct_statistics)
        return payload, report
