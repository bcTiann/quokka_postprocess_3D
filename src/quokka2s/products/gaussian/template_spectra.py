"""Scatter original prepared-template corners into profile-major NumPy accounts.

Workers release means/widths after preparation; L survives only until all
profile writers join in the same bounded wave.
Original-cell accounts, complete moments and physical products still merge in
batch order. The covered corner rows convolve once before output preparation.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter, perf_counter_ns
from types import SimpleNamespace

import numpy as np

from .accounting import add_cutoff_accounts, update_cutoff_report
from .corner_store import RaggedCornerStore, check_corner_memory
from .corner_scatter import prepare_corner_scatter
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
    """Read-only owned vectors[N] and accounts[N,P]; groups may share centers."""

    profile_ids: np.ndarray
    centers: np.ndarray
    sigmas: np.ndarray
    luminosity: np.ndarray

class TemplateIntegratedSpectra(HistogramIntegratedSpectra):
    """Keep original Gaussian rules while summing identical template corners."""

    def __init__(self, line_keys, velocity_edges_kms, *, engine, batch_partial=False,
                 workers=1, cell_chunk=8192, merge_threads=None):
        if merge_threads is not None and (type(merge_threads) is not int or merge_threads != 1):
            raise ValueError("Direct templates merge ordered batch profiles with one main thread")
        if workers != 1:
            raise ValueError("Direct templates use one NumPy query caller per batch")
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
        self._input_owner_ids = set()
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
            "gaussian_worker_wave_ns": 0,
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

        self._corner_spec = engine.prepare_corner_store(profile_count=self.profile_count)
        self._corner_increments = []
        self._corner_store = None
        self._corner_failed = False
        self._corner_finalized = False
        self._corner_final_metrics = {}
        self._corner_scatter_ns = 0
        self._scatter_statistics = {
            "waves": 0,
            "workers": 0,
            "kernel_calls": 0,
            "contiguous_X_logical_bytes": 0,
            "raw_corner_terms": 0,
            "maximum_profile_tasks": 0,
            "maximum_pending_raw_owned_bytes": 0,
            "maximum_pending_luminosity_owner_bytes": 0,
        }
        self._corner_final_ns = 0
        self._corner_init_ns = 0
        self._maximum_worker_increment_bytes = 0
        if not batch_partial:
            began = perf_counter_ns()
            self._corner_store = RaggedCornerStore(self._corner_spec)
            self._corner_init_ns = perf_counter_ns() - began

    def add_batch(self, cells, emission):
        """Prepare unchanged masks, full moments and a bounded original-cell queue."""
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

        The shared preparation hook retains owned inputs only until the next
        bounded worker wave deposits their existing four template corners.
        """
        if not self.batch_partial or self._queue_finished or self._queue_failed:
            raise RuntimeError("Cannot append Gaussians after this worker queue finishes")
        widths = np.maximum(thermal_width_kms, MINIMUM_GAUSSIAN_WIDTH_KMS)
        for values in (velocity_kms, widths, luminosity_matrix):
            if (
                not isinstance(values, np.ndarray)
                or values.dtype != np.float64
                or not values.flags.c_contiguous
                or not values.flags.owndata
            ):
                raise ValueError("Template queues require owned C-order float64 inputs")
            values.flags.writeable = False
        group = GaussianBatchGroup(
            profile_ids=self._profile_ids(line_keys, branch_index),
            centers=velocity_kms,
            sigmas=widths,
            luminosity=luminosity_matrix,
        )
        self._groups.append(group)
        for values in (group.profile_ids, group.centers, group.sigmas, group.luminosity):
            if id(values) not in self._input_owner_ids:
                self._input_owner_ids.add(id(values))
                self._owned_input_bytes += values.nbytes
        self._groups_prepared += 1
        # Shared preparation still computes every original-cell moment and L sum.
        return np.zeros((self.velocity_edges_kms.size - 1, luminosity_matrix.shape[1]))

    def finish_gaussians(self):
        """Prepare raw corners and original accounts; keep L only until wave scattering."""
        if not self.batch_partial or self._queue_finished or self._queue_failed:
            raise RuntimeError("Finish each prepared Gaussian worker exactly once")
        try:
            worker = self.engine.corner_deposit_engine(profile_count=self.profile_count)
            for group in self._groups:
                result, increment = worker.deposit(
                    group.centers,
                    group.sigmas,
                    group.luminosity,
                    group.profile_ids,
                )
                expected_shape = (self.velocity_edges_kms.size - 1, group.profile_ids.size)
                if result.profiles.shape != expected_shape:
                    raise ValueError("Fallback profiles do not match original profile IDs")
                self._query_intervals.append(result.query_interval_ns)
                began = perf_counter_ns()
                profiles = self.dL_dv.reshape(self.profile_count, -1)
                profiles[group.profile_ids] += result.profiles.T
                self._corner_increments.append(increment)
                del increment
                self._addition_intervals.append((began, perf_counter_ns()))
                self._full_window_luminosity[group.profile_ids] += result.full_window
                self._clipped_window_luminosity[group.profile_ids] += result.clipped_window
                self._coverage.add_group(result.coverage, group.profile_ids)
                for name, value in result.statistics.items():
                    self.direct_statistics[name] = self.direct_statistics.get(name, 0) + value
                self._groups_finished += 1
            if self._groups_finished != self._groups_prepared:
                raise AssertionError("A prepared original-cell width group was omitted")
            self._queue_finished = True
        except BaseException:
            self._queue_failed = True
            self._corner_failed = True
            self._corner_increments.clear()
            raise
        finally:
            self._release_original_cell_inputs()
        return self

    def _release_original_cell_inputs(self):
        """Drop coordinate queues; pending raw increments retain only wave-local L."""
        self._groups.clear()
        self._input_owner_ids.clear()
        self._owned_input_bytes = 0

    def release_gaussian_inputs(self):
        """Release inputs after their workers/writers have joined, including pending L."""
        self._release_original_cell_inputs()
        if self.batch_partial:
            self._corner_increments.clear()

    def scatter_corner_wave(self, partials, *, executor, workers):
        """Hold raw L until all writers join, consume increments, then merge small data."""
        if self.batch_partial or self._corner_finalized or self._corner_failed:
            raise RuntimeError('Cannot scatter into a failed main spectrum')
        started = perf_counter_ns()
        increments = []
        try:
            if any(
                partial.spectra.engine is not self.engine
                or partial.spectra.line_keys != self.line_keys
                for partial in partials
            ):
                raise ValueError("Raw partials must share the main engine and line order")
            if any(
                partial.spectra._corner_failed or partial.spectra._queue_failed
                or not partial.spectra._queue_finished or partial.spectra._groups
                for partial in partials
            ):
                raise RuntimeError("Finish every raw Gaussian group before scattering")
            increments.extend(
                increment for partial in partials
                for increment in partial.spectra._corner_increments
            )
            owner_ids = set()
            luminosity_bytes = 0
            raw_bytes = 0
            maximum_worker_bytes = 0
            for partial in partials:
                worker_bytes = 0
                for increment in partial.spectra._corner_increments:
                    own_bytes = sum(
                        values.nbytes for values in (
                            increment.terms.rows, increment.terms.cells,
                            increment.terms.weights, increment.profiles,
                            increment.touched_rows,
                        )
                    )
                    raw_bytes += own_bytes
                    worker_bytes += own_bytes
                    owner = increment.luminosity
                    while isinstance(owner.base, np.ndarray):
                        owner = owner.base
                    if id(owner) not in owner_ids:
                        owner_ids.add(id(owner))
                        luminosity_bytes += owner.nbytes
                maximum_worker_bytes = max(maximum_worker_bytes, worker_bytes)
            metrics = self._corner_store.scatter(
                increments,
                executor=executor,
                workers=workers,
            )
        except BaseException:
            # The store cancels and joins all writers before it raises. The main
            # numerical accumulator is terminal after any failed wave, never saved.
            self._corner_failed = True
            raise
        finally:
            for partial in partials:
                partial.spectra._corner_increments.clear()
            increments.clear()
        finished = perf_counter_ns()
        elapsed = finished - started
        self._maximum_worker_increment_bytes = max(
            self._maximum_worker_increment_bytes, maximum_worker_bytes,
        )
        self._addition_intervals.append((started, finished))
        self._corner_scatter_ns += elapsed
        observations = self._scatter_statistics
        observations['waves'] += 1
        observations['workers'] = max(observations['workers'], workers)
        observations['kernel_calls'] += metrics['kernel_calls']
        observations['contiguous_X_logical_bytes'] += metrics['contiguous_X_logical_bytes']
        observations['raw_corner_terms'] += metrics['raw_corner_terms']
        observations['maximum_profile_tasks'] = max(
            observations['maximum_profile_tasks'], metrics['profile_tasks'],
        )
        observations['maximum_pending_raw_owned_bytes'] = max(
            observations['maximum_pending_raw_owned_bytes'], raw_bytes,
        )
        observations['maximum_pending_luminosity_owner_bytes'] = max(
            observations['maximum_pending_luminosity_owner_bytes'], luminosity_bytes,
        )

    def merge(self, other):
        """Preserve ordered profile, luminosity, count, moment and endpoint sums."""
        if self._corner_finalized or self._corner_failed or other._corner_failed:
            raise RuntimeError("Cannot merge failed or finalized corner spectra")
        if self.batch_partial or self._finalized or other._finalized:
            raise RuntimeError("Merge unfinalized template partials into the main accumulator")
        if self.engine is not other.engine or self.line_keys != other.line_keys:
            raise ValueError("Cannot merge different template engines or line orders")
        if not other._queue_finished or other._queue_failed or other._groups:
            raise RuntimeError("Finish every Gaussian group before the ordered global merge")
        if self.spectral_method != other.spectral_method:
            raise ValueError("Cannot merge spectra calculated with different methods")
        if other._corner_increments:
            self._corner_failed = True
            raise RuntimeError("Scatter the complete raw wave before ordered product merge")
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

    def record_completed_wave(
        self, *, physical_ns, gaussian_ns, gaussian_worker_ns,
        merge_ns, owned_input_bytes,
    ):
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
        self._timing["gaussian_worker_wave_ns"] += gaussian_worker_ns
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
        result = {
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

        result.update({
            "original_cell_deposit_calls": timing["query_calls"],
            "original_cell_deposit_groups_finished": result["all_deferred_groups_finished"],
            "final_corner_expansion_pending": not self._corner_finalized,
            "all_deferred_groups_finished": result["all_deferred_groups_finished"] and self._corner_finalized,
            "snapshot_corner_store_initialization_seconds": self._corner_init_ns / 1e9,
            "gaussian_worker_wave_seconds": timing["gaussian_worker_wave_ns"] / 1e9,
            "inplace_corner_scatter_wave_seconds": self._corner_scatter_ns / 1e9,
            "inplace_scatter_included_in_gaussian_wave": True,
            "raw_wave_statistics": dict(self._scatter_statistics),
            "snapshot_corner_finalization_seconds": self._corner_final_ns / 1e9,
            "maximum_single_worker_increment_bytes": self._maximum_worker_increment_bytes,
            "maximum_single_worker_increment_bytes_scope": "Owned raw term, touched-row and profile arrays only; unique pending luminosity owners are reported separately. Neither is process RSS.",
            "gaussian_total_seconds_including_corner_store": result["gaussian_wave_seconds_including_analytic_accounts_and_scheduling"] + (self._corner_init_ns + self._corner_final_ns) / 1e9,
            "activity_scope": "Original-cell validation/raw corner preparation, guarded inplace profile scattering and outside-width fallback; final convolution is separate",
            "finalization_scope": "Touched-row selection, stored-template convolution, profile addition and memory guards",
            "snapshot_corner_store": self._corner_final_metrics if self._corner_finalized else self._corner_store.statistics(),
            "timing_note": "Inplace scattering is already included in Gaussian wave wall and activity union; do not add it again. Ordered product merge contains small products/accounts. Final convolution is inside output construction.",
        })
        return result

    def finalize_corner_spectra(self):
        """Convolve accumulated corner rows once; original outside profiles exist."""
        if self.batch_partial or self._corner_failed:
            raise RuntimeError("Finalize only a successful main corner store")
        if self._corner_finalized:
            return self._corner_final_metrics
        if self._groups or self._query_intervals or self._addition_intervals:
            raise RuntimeError("Finish and record every worker wave before finalization")
        began = perf_counter_ns()
        try:
            check_corner_memory("before final snapshot corner expansion")
            self.engine.validate_prepared_state()
            covered, metrics = self._corner_store.finalize()
            expected_shape = (self.velocity_edges_kms.size - 1, self.profile_count)
            if covered.shape != expected_shape or not np.isfinite(covered).all():
                raise FloatingPointError("Final corner profiles must have finite[channel,account] values")
            self.dL_dv.reshape(self.profile_count, -1)[:] += covered.T
            del covered
            self._corner_store = None
            check_corner_memory("after final snapshot corner expansion")
        except BaseException:
            self._corner_failed = True
            raise
        self._corner_final_ns = perf_counter_ns() - began
        self._corner_final_metrics = dict(metrics)
        self._corner_finalized = True
        return self._corner_final_metrics

    def build_output(self, projected_area_cm2):
        """Save standard products and independent original-cell cutoff references."""
        if self.batch_partial:
            raise RuntimeError("Merge worker profiles before building template output")
        self.finalize_corner_spectra()
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
            "histogram_allocated": True,
            "new_parameter_grid_allocated": False,
            "node_records_created": self._corner_final_metrics["unique_snapshot_corner_nodes"],
            "node_records_created_scope": "Occupied row/shift keys; allocated dense capacity is corner_account_slots",
            "corner_account_capacity_bytes": self._corner_spec.account_bytes,
            "corner_account_slots": self._corner_spec.slot_count,
            "corner_account_layout": "profile-major [original account, row-specific valid integer shift]",
            "cell_by_channel_probability_matrix_allocated": self.direct_statistics.get("fallback_probability_intervals", 0) > 0,
            "probability_matrix_scope": "Only outside-width fallback uses cell/channel matrices; covered rows keep raw terms and read-only luminosity for one bounded scattering wave",
            "maximum_probability_rows": self.engine.maximum_probability_rows,
            "maximum_direct_probability_rows": self.engine.maximum_direct_probability_rows,
            "maximum_single_fallback_probability_array_bytes": self.engine.maximum_probability_rows * (self.velocity_edges_kms.size - 1) * 8,
            "probability_memory_scope": "Fallback probability-block bound only; raw term buffers, pending unique luminosity owners and dense H capacity are separate. None is process RSS.",
            "work_counter_scope": "Original raw corner terms and direct fallback; raw_corner_terms and weighted_profile_updates count contributions, not grouped unique keys. Final occupied nodes and convolution work are separate.",
            "snapshot_corner_expansion": self._corner_final_metrics,
            "integration": "Original four template corners scatter into disjoint profile-major NumPy planes before stored-row convolution; outside widths use stable original-cell integrals",
            "cutoff_sigma": 5.0,
            "cutoff_applies_to": self.engine.cutoff_applies_to,
            "renormalized": False,
            "accounting_scope": "Independent original-cell analytic endpoint intervals",
            "timing_scope": "integration/accounting/add_batch durations sum overlapping worker calls; see processing_timings.gaussian for elapsed union and wave wall",
            "spectral_add_batch_scope": "Shared availability masks, direct owned luminosity queues and stable full moments; Gaussian queries occur in the next wave",
            "lookup_table": self.engine.lookup_tables.info(),
            "portable_backend": "numpy-scipy-private-coo-corner-convolution",
            "prebuilt_corner_backend": prepare_corner_scatter(),
            "raw_wave_statistics": dict(self._scatter_statistics),
            "raw_luminosity_retention": "Read-only packed owners survive for one bounded wave; every profile writer joins before release.",
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
                "selected_threads": self._scatter_statistics["workers"],
                "maximum_profile_tasks": self._scatter_statistics["maximum_profile_tasks"],
                "thread_scope": "Independent profile tasks use the existing worker pool; small products merge on one main thread.",
                "batch_group_order_unchanged": True,
                "storage": "Profile-major NumPy rows hold only overlapping integer shifts; raw terms and packed luminosity are bounded by one wave",
                "floating_sum_tree_changed": True,
                "new_interpolation_grid": False,
            }
        print_template_coverage(coverage, self.direct_statistics)
        return payload, report
