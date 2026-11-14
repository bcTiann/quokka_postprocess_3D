"""Prepare original Gaussian terms and own their bounded snapshot accounts.

Each template row stores only integer shifts overlapping the observation
window. Workers prepare checked raw corner terms; disjoint profile tasks add
into existing NumPy account planes through the guarded prebuilt SciPy kernel.
Final NumPy convolution, analytic accounts and outside-width integration keep
their existing scientific rules.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import resource
import sys
from time import perf_counter

import numpy as np

from .template_corners import locate_corners, prepare_corner_increment
from .corner_scatter import scatter_corner_profiles
from .template_engine import TemplateEngine

CORNER_MEMORY_LIMIT_BYTES = 20 * 1024**3


def check_corner_memory(stage, additional_bytes=0):
    """Refuse a planned allocation above the conservative 20-GiB process limit.

    macOS reports high-water RSS in bytes; Linux reports KiB. High-water RSS
    includes caller-owned queued inputs and can overestimate current residency
    after buffers are freed. This is an allocation check, not a cross-thread
    reservation; the outer whole-process RSS guard remains required.
    """
    high_water = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    high_water_bytes = int(high_water if sys.platform == "darwin" else high_water * 1024)
    additional_bytes = max(0, int(additional_bytes))
    if high_water_bytes + additional_bytes > CORNER_MEMORY_LIMIT_BYTES:
        raise MemoryError(
            f"Corner accumulation exceeds the 20-GiB process budget at {stage}: "
            f"high-water RSS {high_water_bytes} bytes + planned allocation "
            f"{additional_bytes} bytes"
        )
    return high_water_bytes


@dataclass(frozen=True)
class RaggedCornerSpec:
    """Immutable per-row shift bounds and offsets for the prepared templates."""

    table: object
    profile_count: int = 40

    def __post_init__(self):
        if self.profile_count < 1:
            raise ValueError("A corner store requires positive profile count")
        counts = self.table.row_bin_counts
        row_spans = np.where(counts > 0, self.table.K + counts - 1, 0).astype(np.int64)
        first = int(self.table.output_bin_indices[0])
        # A B-bin template overlaps K observation bins at K+B-1 integer shifts.
        minimum_shifts = first - self.table.row_min_bins - counts + 1
        offsets = np.r_[np.int64(0), np.cumsum(row_spans, dtype=np.int64)]
        for values in (row_spans, minimum_shifts, offsets):
            values.flags.writeable = False
        object.__setattr__(self, "row_shift_counts", row_spans)
        object.__setattr__(self, "row_minimum_shifts", minimum_shifts)
        object.__setattr__(self, "row_offsets", offsets)
        object.__setattr__(self, "slot_count", int(offsets[-1]))
        object.__setattr__(self, "account_bytes", int(offsets[-1]) * self.profile_count * 8)


class CornerPreparationEngine(TemplateEngine):
    """Worker-local original-cell checks/accounts/fallback and raw corner terms.

    prepare_group() returns (TemplateGroupResult, CornerIncrement). It does not
    write H or convolve templates. The result holds outside-width profiles and
    unchanged analytic original-cell accounts. The increment owns term indices
    and weights, and borrows read-only luminosity[N,P] until all wave writers
    join. This mutable engine belongs to one worker thread.
    """

    def __init__(self, *, spec):
        super().__init__(table=spec.table)
        self.spec = spec
        self.portable_backend = "numpy-original-corner-preparation"
        self._profile_ids = None
        self._increment = None
        self._metrics = None

    def prepare_group(
        self,
        centers,
        sigmas,
        luminosity,
        profile_ids,
    ):
        """Prepare one original-cell group's terms, fallback and endpoint accounts."""
        check_corner_memory("corner deposit")
        if self._profile_ids is not None:
            raise RuntimeError("CornerPreparationEngine calls must not overlap")
        profile_ids = np.asarray(profile_ids)
        if profile_ids.ndim != 1 or profile_ids.dtype.kind not in "iu":
            raise ValueError("Profile IDs must be an integer vector")
        if (
            np.asarray(luminosity).ndim != 2
            or profile_ids.size != np.asarray(luminosity).shape[1]
        ):
            raise ValueError("Profile IDs must match luminosity columns")
        if np.any(profile_ids < 0) or np.any(profile_ids >= self.spec.profile_count):
            raise ValueError("Profile IDs lie outside the corner store")
        if np.unique(profile_ids).size != profile_ids.size:
            raise ValueError("A width group cannot repeat a profile ID")
        self._profile_ids = profile_ids.astype(np.int64, copy=False)
        self._increment = None
        self._metrics = None
        try:
            result = super().integrate(
                centers=centers,
                sigmas=sigmas,
                luminosity=luminosity,
            )
            check_corner_memory("finished corner deposit")
            metrics = dict(self._metrics)
            result = replace(result, statistics={**result.statistics, **metrics})
            return result, self._increment
        finally:
            self._profile_ids = None
            self._increment = None
            self._metrics = None

    def _accumulate_probabilities(self, centers, sigmas, luminosity, outside):
        """Prepare covered terms and directly integrate only outside-table widths."""
        if self._profile_ids is None:
            raise RuntimeError("Use prepare_group with explicit original profile IDs")
        table = self.lookup_tables
        began = perf_counter()
        active = np.any(luminosity > 0.0, axis=1)
        covered = active & ~outside
        output = np.zeros((luminosity.shape[1], table.K))
        counts = {
            "kernel_rows": centers.size,
            "active_rows": int(np.count_nonzero(covered)),
            "support_bins_visited": 0,
            "template_corner_values_read": 0,
            "product_updates": 0,
            "direct_active_rows": 0,
            "direct_support_bins_visited": 0,
            "direct_product_updates": 0,
            "probability_subbatches": 0,
            "direct_probability_batches": 0,
        }
        if np.all(covered):
            selected_centers = centers
            selected_sigmas = sigmas
            selected_light = luminosity
        else:
            selected_centers = centers[covered]
            selected_sigmas = sigmas[covered]
            selected_light = luminosity[covered]
        metrics = {"covered_selection_seconds": perf_counter() - began}
        check_corner_memory(
            "raw corner term preparation",
            additional_bytes=selected_centers.size * 512,
        )
        began = perf_counter()
        shifts, rows, fractions = locate_corners(
            table=table,
            centers=selected_centers,
            sigmas=selected_sigmas,
        )
        metrics["coordinate_location_seconds"] = perf_counter() - began
        began = perf_counter()
        increment, preparation = prepare_corner_increment(
            spec=self.spec,
            shifts=shifts,
            rows=rows,
            fractions=fractions,
            luminosity=selected_light,
            profiles=self._profile_ids,
        )
        metrics.update(preparation)
        metrics["raw_term_preparation_seconds"] = perf_counter() - began
        probability_seconds = 0.0
        if np.any(outside):
            direct, direct_counts, probability_seconds = TemplateEngine._accumulate_probabilities(
                self,
                centers=centers[outside],
                sigmas=sigmas[outside],
                luminosity=luminosity[outside],
                outside=np.ones(np.count_nonzero(outside), dtype=np.bool_),
            )
            output += direct
            for name in counts:
                if name != "kernel_rows":
                    counts[name] += direct_counts[name]
        self._increment = increment
        self._metrics = metrics
        return output, counts, probability_seconds


class RaggedCornerStore:
    """One main owner of float64 luminosity[profile, valid integer-shift slot]."""

    def __init__(self, spec):
        self.spec = spec
        check_corner_memory(
            "ragged dense corner accounts",
            spec.account_bytes + spec.row_shift_counts.size,
        )
        began = perf_counter()
        self.light = np.zeros((spec.profile_count, spec.slot_count))
        self.touched_rows = np.zeros(spec.row_shift_counts.size, dtype=bool)
        check_corner_memory("finished ragged account allocation")
        self.allocation_seconds = perf_counter() - began
        self.failed = False
        self.scatter_seconds = 0.0
        self.scatter_waves = 0
        self.raw_corner_terms = 0
        self.weighted_profile_updates = 0

    def scatter(self, increments, *, executor, workers):
        """Add one checked wave into disjoint profile planes, then join all writers."""
        began = perf_counter()
        if self.failed or self.light is None:
            raise RuntimeError("Cannot scatter into a failed or consumed corner store")
        try:
            if type(workers) is not int or workers < 1:
                raise ValueError("Corner scattering requires a positive worker count")
            maximum_rows = max(
                (increment.terms.input_length for increment in increments),
                default=0,
            )
            check_corner_memory(
                "contiguous corner profile inputs",
                additional_bytes=maximum_rows * min(workers, self.spec.profile_count) * 8,
            )
            metrics = scatter_corner_profiles(
                store=self,
                increments=increments,
                executor=executor,
            )
            check_corner_memory("finished corner profile scattering")
        except BaseException:
            # The helper joins every submitted writer before propagating an
            # error. Drop the partial H so it can never be reused or finalized.
            self.failed = True
            self.light = None
            self.touched_rows = None
            raise
        self.scatter_seconds += perf_counter() - began
        self.scatter_waves += 1
        self.raw_corner_terms += metrics["raw_corner_terms"]
        self.weighted_profile_updates += sum(
            increment.terms.destination_slots.size * increment.profiles.size
            for increment in increments
        )
        return metrics

    def statistics(self):
        """Report array capacity and measured timings, without equating bytes to RSS."""
        return {
            "dense_account_capacity_bytes": self.spec.account_bytes,
            "retained_dense_account_bytes": self.spec.account_bytes if self.light is not None else 0,
            "slot_count": self.spec.slot_count,
            "allocation_seconds": self.allocation_seconds,
            "scatter_waves": self.scatter_waves,
            "scatter_seconds": self.scatter_seconds,
            "raw_corner_terms": self.raw_corner_terms,
            "weighted_profile_updates": self.weighted_profile_updates,
            "array_layout": "profile-major-ragged",
        }

    def finalize(self):
        """Convolve each occupied template row with its accumulated shift light.

        A row owns K+B-1 consecutive shift slots, with B stored template bins.
        Direct NumPy convolution in valid mode returns exactly K channels. This
        is the same stored-corner linear sum; there is no FFT or new cutoff.
        """
        if self.failed or self.light is None:
            raise RuntimeError("Cannot finalize failed or consumed corner accounts")
        output = np.zeros((self.spec.table.K, self.spec.profile_count))
        metrics = {
            **self.statistics(),
            "row_selection_seconds": 0.0,
            "convolution_seconds": 0.0,
            "candidate_template_rows": int(np.count_nonzero(self.touched_rows)),
            "occupied_template_rows": 0,
            "unique_snapshot_corner_nodes": 0,
            "convolution_calls": 0,
            "logical_products": 0,
        }
        for row in np.flatnonzero(self.touched_rows):
            began = perf_counter()
            first = int(self.spec.row_offsets[row])
            stop = int(self.spec.row_offsets[row + 1])
            check_corner_memory(
                "ragged row convolution buffers",
                (stop - first) * self.spec.profile_count * 4
                + (stop - first + self.spec.table.K) * 32,
            )
            light = self.light[:, first:stop]
            positive = light != 0.0
            active = np.flatnonzero(np.any(positive, axis=1))
            if active.size == 0:
                metrics["row_selection_seconds"] += perf_counter() - began
                continue
            metrics["occupied_template_rows"] += 1
            metrics["unique_snapshot_corner_nodes"] += int(
                np.count_nonzero(np.any(positive, axis=0))
            )
            width = int(self.spec.table.row_bin_counts[row])
            offset = int(self.spec.table.row_value_offsets[row])
            template = self.spec.table.flat_values[offset:offset + width]
            metrics["row_selection_seconds"] += perf_counter() - began
            began = perf_counter()
            for profile in active:
                output[:, profile] += np.convolve(
                    light[profile],
                    template,
                    mode="valid",
                )
            metrics["convolution_seconds"] += perf_counter() - began
            metrics["convolution_calls"] += int(active.size)
            metrics["logical_products"] += int(self.spec.table.K * width * active.size)
        output /= self.spec.table.delta_v
        if not np.isfinite(output).all():
            raise FloatingPointError("Ragged final profile became nonfinite")
        self.light = None
        self.touched_rows = None
        metrics["retained_dense_account_bytes"] = 0
        check_corner_memory("finished ragged final expansion")
        return output, metrics
