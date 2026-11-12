"""Accumulate original Gaussian corners in standard NumPy snapshot accounts.

Each template row keeps only integer shifts whose stored support overlaps the
observation window. Workers return compact unique-node luminosities; one main
owner adds them into profile-major ragged rows. At the end, NumPy convolution
expands each active row/account into the fixed velocity channels. Original-cell
analytic accounts and outside-width integration remain in TemplateEngine.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import resource
import sys
from time import perf_counter

import numpy as np

from .template_corners import combine_corner_light, locate_corners
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


class CornerDepositEngine(TemplateEngine):
    """Worker-local validation/accounts/fallback plus compact corner deposits.

    Each call returns (TemplateGroupResult, increment), where the increment is
    (template row IDs, integer shifts, luminosity[U,P], global profile IDs).
    The result contains fallback-only profiles and unchanged original-cell
    analytic accounts. This mutable engine belongs to one worker thread; only
    its immutable table and indexing spec are shared.
    """

    def __init__(self, *, spec):
        super().__init__(table=spec.table)
        self.spec = spec
        self.portable_backend = "numpy-scipy-compact-corner-deposit"
        self._profile_ids = None
        self._increment = None
        self._metrics = None

    def deposit(self, centers, sigmas, luminosity, profile_ids):
        check_corner_memory("corner deposit")
        if self._profile_ids is not None:
            raise RuntimeError("CornerDepositEngine calls must not overlap")
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
            result = super().integrate(centers, sigmas, luminosity)
            check_corner_memory("finished corner deposit")
            metrics = dict(self._metrics)
            result = replace(result, statistics={**result.statistics, **metrics})
            return result, self._increment
        finally:
            self._profile_ids = None
            self._increment = None
            self._metrics = None

    def _accumulate_probabilities(self, centers, sigmas, luminosity, outside):
        if self._profile_ids is None:
            raise RuntimeError("Use deposit with explicit original profile IDs")
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
        metrics = {
            "covered_selection_seconds": 0.0,
            "coordinate_location_seconds": 0.0,
            "key_and_light_grouping_seconds": 0.0,
            "compact_increment_seconds": 0.0,
            "positive_corner_entries": 0,
            "window_corner_entries": 0,
            "unique_corner_keys": 0,
        }
        if np.all(covered):
            selected_centers = centers
            selected_sigmas = sigmas
            selected_light = luminosity
        else:
            selected_centers = centers[covered]
            selected_sigmas = sigmas[covered]
            selected_light = luminosity[covered]
        metrics["covered_selection_seconds"] = perf_counter() - began
        if selected_centers.size:
            check_corner_memory(
                "compact template grouping buffers",
                selected_centers.size * (768 + selected_light.shape[1] * 40),
            )
            began = perf_counter()
            shifts, rows, fractions = locate_corners(
                table=table,
                centers=selected_centers,
                sigmas=selected_sigmas,
            )
            metrics["coordinate_location_seconds"] = perf_counter() - began
            began = perf_counter()
            row_ids, shifts, light, grouping = combine_corner_light(
                table=table,
                shifts=shifts,
                rows=rows,
                fractions=fractions,
                luminosity=selected_light,
            )
            metrics["key_and_light_grouping_seconds"] = perf_counter() - began
            metrics.update(grouping)
        else:
            row_ids = np.empty(0, dtype=np.int64)
            shifts = np.empty(0, dtype=np.int64)
            light = np.empty((0, luminosity.shape[1]))
        began = perf_counter()
        increment = (row_ids, shifts, light, self._profile_ids.copy())
        metrics["compact_increment_bytes"] = int(sum(values.nbytes for values in increment))
        metrics["compact_increment_seconds"] = perf_counter() - began
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
        self.add_seconds = 0.0
        self.add_calls = 0
        self.node_updates = 0
        self.account_updates = 0

    def add(self, row_ids, shifts, luminosity, profile_ids):
        """Add one independent unique-corner increment without retaining it."""
        began = perf_counter()
        if self.light is None:
            raise RuntimeError("Cannot add after final corner expansion")
        if (
            luminosity.shape != (row_ids.size, profile_ids.size)
            or shifts.shape != row_ids.shape
        ):
            raise ValueError("Corner coordinates, profiles and luminosity shapes disagree")
        check_corner_memory(
            "ragged indexed corner addition",
            2 * luminosity.nbytes + row_ids.size * 32,
        )
        if np.any(row_ids < 0) or np.any(row_ids >= self.spec.row_shift_counts.size):
            raise ValueError("Corner template row is outside the prepared table")
        relative = shifts - self.spec.row_minimum_shifts[row_ids]
        if np.any(relative < 0) or np.any(relative >= self.spec.row_shift_counts[row_ids]):
            raise ValueError("Corner shift is outside its row's observation overlap")
        slots = self.spec.row_offsets[row_ids] + relative
        if np.any(slots[1:] <= slots[:-1]):
            raise ValueError("A dense increment requires sorted unique corner nodes")
        if np.any(profile_ids < 0) or np.any(profile_ids >= self.spec.profile_count):
            raise ValueError("Corner profile is outside the store")
        if np.unique(profile_ids).size != profile_ids.size:
            raise ValueError("Corner profiles must be unique")
        # At most twelve accounts, each using one vectorized unique-slot update.
        for local, profile in enumerate(profile_ids):
            plane = self.light[profile]
            plane[slots] += luminosity[:, local]
        self.touched_rows[row_ids] = True
        self.add_calls += 1
        self.node_updates += int(row_ids.size)
        self.account_updates += int(luminosity.size)
        self.add_seconds += perf_counter() - began
        check_corner_memory("finished ragged indexed addition")

    def statistics(self):
        """Report array capacity and measured timings, without equating bytes to RSS."""
        return {
            "dense_account_capacity_bytes": self.spec.account_bytes,
            "retained_dense_account_bytes": self.spec.account_bytes if self.light is not None else 0,
            "slot_count": self.spec.slot_count,
            "allocation_seconds": self.allocation_seconds,
            "add_calls": self.add_calls,
            "add_seconds": self.add_seconds,
            "node_updates": self.node_updates,
            "account_updates": self.account_updates,
            "array_layout": "profile-major-ragged",
        }

    def finalize(self):
        """Convolve each occupied template row with its accumulated shift light.

        A row owns K+B-1 consecutive shift slots, with B stored template bins.
        Direct NumPy convolution in valid mode returns exactly K channels. This
        is the same stored-corner linear sum; there is no FFT or new cutoff.
        """
        if self.light is None:
            raise RuntimeError("Ragged corner accounts have already been consumed")
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
