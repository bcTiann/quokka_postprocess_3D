"""Snapshot-wide corner accounts built only with standard SciPy CSR.

Covered cells keep their original four template corners.  Each worker deposits
one width group's luminosities into a sparse [template row, shift/profile]
matrix; a serial owner merges matrices in binary levels.  Cell arrays are never
retained.  Stored finite-support templates are expanded once at finalization.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import resource
import sys
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix

from .template_corners import combine_corner_light, expand_unique_corners, locate_corners
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


def csr_sum_allocation_bytes(first, second):
    """Bound new CSR sum buffers, including a possible pruned result copy."""
    maximum_nnz = int(first.nnz) + int(second.nnz)
    index_bytes = max(first.indices.dtype.itemsize, second.indices.dtype.itemsize)
    if maximum_nnz > np.iinfo(np.int32).max:
        index_bytes = 8
    output_bytes = (
        maximum_nnz * (8 + index_bytes)
        + (first.shape[0] + 1) * index_bytes
    )
    return 2 * output_bytes


def sparse_bytes(matrix):
    """Return owned CSR numeric/index buffer sizes, not measured process RSS."""
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


@dataclass(frozen=True)
class CornerStoreSpec:
    """Immutable finite-support indexing rules shared by independent workers."""

    table: object
    profile_count: int = 40

    def __post_init__(self):
        table = self.table
        if self.profile_count < 1:
            raise ValueError("A corner store requires positive account count")
        valid = table.row_bin_counts > 0
        if not np.any(valid):
            raise ValueError("A corner store requires a nonempty template")
        first = int(table.output_bin_indices[0])
        last = first + table.K - 1
        row_minimum = table.row_min_bins[valid]
        row_counts = table.row_bin_counts[valid]
        minimum_shift = int(np.min(first - row_minimum - row_counts + 1))
        maximum_shift = int(np.max(last - row_minimum))
        shift_span = maximum_shift - minimum_shift + 1
        rows = int(table.phase_axis_values.size)
        columns = shift_span * self.profile_count
        if rows * shift_span > np.iinfo(np.int64).max:
            raise ValueError("Sparse corner node identifiers exceed int64")
        object.__setattr__(self, "minimum_shift", minimum_shift)
        object.__setattr__(self, "maximum_shift", maximum_shift)
        object.__setattr__(self, "shift_span", shift_span)
        object.__setattr__(self, "shape", (rows, columns))

class CornerDepositEngine(TemplateEngine):
    """Worker-local deposit with inherited cell validation/accounts/fallback.

    ``deposit`` returns original-cell analytic accounts plus fallback-only
    profiles, an independent CSR increment, and detailed deposit metrics.
    This instance is mutable during one call and must not be shared by workers.
    The table and immutable spec are safe to share.
    """

    def __init__(self, *, spec):
        super().__init__(table=spec.table)
        self.spec = spec
        self.portable_backend = "numpy-scipy-snapshot-corner-deposit"
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
            return result, self._increment, metrics
        finally:
            self._profile_ids = None
            self._increment = None
            self._metrics = None

    def _accumulate_probabilities(self, centers, sigmas, luminosity, outside):
        if self._profile_ids is None:
            raise RuntimeError("Use deposit with explicit global profile IDs")
        table = self.lookup_tables
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
            "coordinate_location_seconds": 0.0,
            "key_and_light_grouping_seconds": 0.0,
            "csr_increment_seconds": 0.0,
            "positive_corner_entries": 0,
            "window_corner_entries": 0,
            "unique_corner_keys": 0,
            "increment_nnz": 0,
            "increment_csr_bytes": 0,
        }
        if np.any(covered):
            began = perf_counter()
            selected_centers = centers[covered]
            selected_sigmas = sigmas[covered]
            selected_light = luminosity[covered]
            shifts, rows, fractions = locate_corners(
                table=table,
                centers=selected_centers,
                sigmas=selected_sigmas,
            )
            metrics["coordinate_location_seconds"] = perf_counter() - began
            began = perf_counter()
            row_ids, shifts, light, grouped = combine_corner_light(
                table=table,
                shifts=shifts,
                rows=rows,
                fractions=fractions,
                luminosity=selected_light,
            )
            metrics["key_and_light_grouping_seconds"] = perf_counter() - began
            metrics.update(grouped)
            began = perf_counter()
            maximum_entries = int(np.count_nonzero(light))
            check_corner_memory(
                "corner CSR increment",
                maximum_entries * 96 + (self.spec.shape[0] + 1) * 16,
            )
            node_indices, local_profiles = np.nonzero(light)
            columns = (
                (shifts[node_indices] - self.spec.minimum_shift) * self.spec.profile_count
                + self._profile_ids[local_profiles]
            )
            if np.any(columns < 0) or np.any(columns >= self.spec.shape[1]):
                raise RuntimeError("A retained corner escaped the finite-support shift bounds")
            increment = csr_matrix(
                (light[node_indices, local_profiles], (row_ids[node_indices], columns)),
                shape=self.spec.shape,
            )
            if not np.isfinite(increment.data).all():
                raise FloatingPointError("Corner luminosity deposit became nonfinite")
            metrics["csr_increment_seconds"] = perf_counter() - began
        else:
            increment = csr_matrix(self.spec.shape, dtype=np.float64)
        metrics["increment_nnz"] = int(increment.nnz)
        metrics["increment_csr_bytes"] = sparse_bytes(increment)
        probability_seconds = 0.0
        if np.any(outside):
            direct_output, direct_counts, probability_seconds = super()._accumulate_probabilities(
                centers=centers[outside],
                sigmas=sigmas[outside],
                luminosity=luminosity[outside],
                outside=np.ones(np.count_nonzero(outside), dtype=np.bool_),
            )
            output += direct_output
            for name in counts:
                if name != "kernel_rows":
                    counts[name] += direct_counts[name]
        self._increment = increment
        self._metrics = metrics
        return output, counts, probability_seconds


class BinaryCornerStore:
    """One serial owner's O(log batches) CSR subtotals; no cell vectors."""

    def __init__(self, spec):
        self.spec = spec
        self.levels = []
        self.add_calls = 0
        self.additions = 0
        self.merge_seconds = 0.0
        self.maximum_retained_csr_bytes = 0
        self.maximum_estimated_merge_live_csr_bytes = 0
        self.maximum_retained_nnz = 0

    def _record_sizes(self):
        present = [matrix for matrix in self.levels if matrix is not None]
        self.maximum_retained_csr_bytes = max(
            self.maximum_retained_csr_bytes, sum(sparse_bytes(matrix) for matrix in present),
        )
        self.maximum_retained_nnz = max(
            self.maximum_retained_nnz, sum(matrix.nnz for matrix in present),
        )

    def statistics(self):
        """Describe currently retained CSR buffers, without treating them as RSS."""
        present = [matrix for matrix in self.levels if matrix is not None]
        return {
            "occupied_levels": len(present),
            "allocated_level_slots": len(self.levels),
            "current_csr_nnz": int(sum(matrix.nnz for matrix in present)),
            "current_csr_bytes": int(sum(sparse_bytes(matrix) for matrix in present)),
            "maximum_retained_nnz": self.maximum_retained_nnz,
            "maximum_retained_csr_bytes": self.maximum_retained_csr_bytes,
            "maximum_estimated_merge_live_csr_bytes": self.maximum_estimated_merge_live_csr_bytes,
            "add_calls": self.add_calls,
            "binary_csr_additions": self.additions,
            "binary_merge_seconds": self.merge_seconds,
        }

    def add(self, increment):
        """Consume an owned canonical CSR increment without retaining its caller."""
        if not isinstance(increment, csr_matrix) or increment.shape != self.spec.shape:
            raise ValueError("Corner increments must be CSR with the store shape")
        check_corner_memory("incoming sparse increment validation", 2 * int(increment.nnz))
        if not increment.has_canonical_format:
            raise ValueError("Corner increments must have sorted unique columns")
        if not np.isfinite(increment.data).all() or np.any(increment.data < 0.0):
            raise ValueError("Corner increments must be finite and nonnegative")
        check_corner_memory("sparse level merge")
        began = perf_counter()
        self.add_calls += 1
        if increment.nnz == 0:
            self.merge_seconds += perf_counter() - began
            return
        carry = increment
        level = 0
        while level < len(self.levels) and self.levels[level] is not None:
            older = self.levels[level]
            retained_other = sum(
                sparse_bytes(matrix) for index, matrix in enumerate(self.levels)
                if index != level and matrix is not None
            )
            check_corner_memory(
                stage="CSR level addition",
                additional_bytes=csr_sum_allocation_bytes(older, carry),
            )
            merged = older + carry
            check_corner_memory("finished CSR level addition")
            self.maximum_estimated_merge_live_csr_bytes = max(
                self.maximum_estimated_merge_live_csr_bytes,
                retained_other + sparse_bytes(older) + sparse_bytes(carry) + sparse_bytes(merged),
            )
            self.levels[level] = None
            carry = merged
            del older, merged
            self.additions += 1
            level += 1
        if level == len(self.levels):
            self.levels.append(carry)
        else:
            self.levels[level] = carry
        if not np.isfinite(carry.data).all():
            raise FloatingPointError("Merged corner luminosity became nonfinite")
        self._record_sizes()
        self.merge_seconds += perf_counter() - began


    def finalize(self, *, maximum_dense_nodes=4096):
        """Return covered density[K,40] after one final expansion per corner.

        Dense buffers hold at most maximum_dense_nodes times 40 luminosities.
        COO row decoding is O(nnz), while its data/column buffers may be views;
        measured RSS and numeric-buffer estimates are reported separately.
        """
        if maximum_dense_nodes < 1:
            raise ValueError("Finalization requires a positive node-block size")
        check_corner_memory("final corner expansion")
        merge_began = perf_counter()
        # Higher occupied levels contain older inputs.  CSR additions determine
        # a different floating-point reduction tree from per-cell expansion.
        final = None
        for level in range(len(self.levels) - 1, -1, -1):
            matrix = self.levels[level]
            if matrix is None:
                continue
            if final is None:
                final = matrix
            else:
                check_corner_memory(
                    stage="final CSR level addition",
                    additional_bytes=csr_sum_allocation_bytes(final, matrix),
                )
                merged = final + matrix
                check_corner_memory("finished final CSR level addition")
                self.maximum_estimated_merge_live_csr_bytes = max(
                    self.maximum_estimated_merge_live_csr_bytes,
                    sum(sparse_bytes(item) for item in self.levels if item is not None)
                    + sparse_bytes(final) + sparse_bytes(merged),
                )
                final = merged
                self.additions += 1
            self.levels[level] = None
        final_merge_seconds = perf_counter() - merge_began
        output = np.zeros((self.spec.table.K, self.spec.profile_count))
        expansion = {
            "support_entries_gathered": 0,
            "nonzero_support_entries": 0,
            "support_blocks": 0,
            "logical_product_updates": 0,
        }
        decode_seconds = 0.0
        expand_seconds = 0.0
        dense_blocks = 0
        unique_nodes = 0
        nnz = 0 if final is None else int(final.nnz)
        final_bytes = 0 if final is None else sparse_bytes(final)
        maximum_dense_bytes = 0
        decode_array_bytes = 0
        profile_columns = 0
        minimum_profiles = self.spec.profile_count
        maximum_profiles = 0
        if final is not None and final.nnz:
            # COO row decoding, int64 node IDs, temporary comparisons, unique
            # boundaries and stops are all O(nnz). Check before allocating them.
            check_corner_memory("global CSR node decoding", 48 * int(final.nnz))
            began = perf_counter()
            coordinate = final.tocoo(copy=False)
            encoded = (
                coordinate.row.astype(np.int64) * self.spec.shift_span
                + coordinate.col // self.spec.profile_count
            )
            boundaries = np.flatnonzero(np.r_[True, encoded[1:] != encoded[:-1]])
            stops = np.r_[boundaries[1:], encoded.size]
            unique_nodes = int(boundaries.size)
            decode_array_bytes = int(
                coordinate.row.nbytes + encoded.nbytes
                + boundaries.nbytes + stops.nbytes
            )
            decode_seconds += perf_counter() - began
            for begin in range(0, unique_nodes, maximum_dense_nodes):
                check_corner_memory(
                    stage="bounded template contraction",
                    additional_bytes=(
                        maximum_dense_nodes * self.spec.profile_count * 32
                        + 262144 * 96
                        + self.spec.table.K * self.spec.profile_count * 16
                    ),
                )
                end = min(begin + maximum_dense_nodes, unique_nodes)
                began = perf_counter()
                first_entry = int(boundaries[begin])
                stop_entry = int(stops[end - 1])
                node_codes = encoded[boundaries[begin:end]]
                row_ids = node_codes // self.spec.shift_span
                shifts = node_codes % self.spec.shift_span + self.spec.minimum_shift
                counts = stops[begin:end] - boundaries[begin:end]
                owners = np.repeat(np.arange(end - begin), counts)
                profiles = coordinate.col[first_entry:stop_entry] % self.spec.profile_count
                light = np.zeros((end - begin, self.spec.profile_count))
                light[owners, profiles] = coordinate.data[first_entry:stop_entry]
                maximum_dense_bytes = max(maximum_dense_bytes, int(light.nbytes))
                active_profiles = np.flatnonzero(np.any(light != 0.0, axis=0))
                profile_columns += int(active_profiles.size)
                minimum_profiles = min(minimum_profiles, int(active_profiles.size))
                maximum_profiles = max(maximum_profiles, int(active_profiles.size))
                if active_profiles.size != self.spec.profile_count:
                    light = np.ascontiguousarray(light[:, active_profiles])
                decode_seconds += perf_counter() - began
                began = perf_counter()
                part, metrics = expand_unique_corners(
                    table=self.spec.table,
                    row_ids=row_ids,
                    shifts=shifts,
                    luminosity=light,
                )
                output[:, active_profiles] += part
                expand_seconds += perf_counter() - began
                for name, value in metrics.items():
                    expansion[name] += value
                dense_blocks += 1
        output /= self.spec.table.delta_v
        if not np.isfinite(output).all():
            raise FloatingPointError("Final corner expansion became nonfinite")
        check_corner_memory("finished final corner expansion")
        metrics = {
            "add_calls": self.add_calls,
            "binary_csr_additions": self.additions,
            "binary_merge_seconds": self.merge_seconds,
            "final_level_merge_seconds": final_merge_seconds,
            "final_csr_nnz": nnz,
            "final_csr_bytes": final_bytes,
            "unique_snapshot_corner_nodes": unique_nodes,
            "maximum_retained_nnz": self.maximum_retained_nnz,
            "maximum_retained_csr_bytes": self.maximum_retained_csr_bytes,
            "maximum_estimated_merge_live_csr_bytes": self.maximum_estimated_merge_live_csr_bytes,
            "final_decode_array_bytes": decode_array_bytes,
            "maximum_dense_account_bytes": maximum_dense_bytes,
            "final_decode_seconds": decode_seconds,
            "final_support_expansion_seconds": expand_seconds,
            "dense_node_blocks": dense_blocks,
            "summed_profile_columns_across_dense_blocks": profile_columns,
            "minimum_profiles_per_dense_block": minimum_profiles if dense_blocks else 0,
            "maximum_profiles_per_dense_block": maximum_profiles,
            **expansion,
        }
        return output, metrics
