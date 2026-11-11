"""deferred two-dimensional spectral accumulation.

The production line masks, physical widths, luminosity sums and full Gaussian
moments are retained. Worker batches own only sparse histogram increments;
the main accumulator merges them in original batch order and expands once.
"""
from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np

from .merge import NodeMajorMerger
from .accounting import add_cutoff_accounts, update_cutoff_report
from ...constants import ATOMIC_MASS_UNIT_G, SPEED_OF_LIGHT_KMS
from ...line_definitions import LINE_DEFINITIONS
from .. import DUST_STATES
from ._spectral_fields import pack_width_group, prepare_available_group
from ..integrated_spectra import (
    IntegratedSpectra,
    calculate_thermal_width_kms,
)
from ..line_velocity_moments import LineVelocityMoments, MINIMUM_GAUSSIAN_WIDTH_KMS


def shared_width_moments(*, velocity_kms, thermal_width_kms, luminosity_by_dust, shared_lines):
    """Return the original full moments, reusing only identical input vectors.

    A profile keeps its own positive-luminosity mask and first positive velocity
    anchor. NumPy receives the same contiguous reduction vectors as before;
    no C reduction or raw-second-moment subtraction replaces those operations.
    """
    effective_width = np.maximum(thermal_width_kms, MINIMUM_GAUSSIAN_WIDTH_KMS)
    contexts = []
    result = {}
    for dust_state, luminosity in luminosity_by_dust.items():
        for line_index in shared_lines:
            emitting_cells = luminosity[line_index] > 0.0
            if not np.any(emitting_cells):
                result[dust_state, line_index] = LineVelocityMoments()
                continue
            selected_luminosity = luminosity[line_index][emitting_cells]
            total_luminosity = float(np.sum(selected_luminosity, dtype=np.float64))
            context = None
            for previous_cells, previous_context in contexts:
                if np.array_equal(previous_cells, emitting_cells):
                    context = previous_context
                    break
            if context is None:
                selected_velocity = velocity_kms[emitting_cells]
                selected_width = effective_width[emitting_cells]
                velocity_anchor = selected_velocity[0]
                velocity_offset = selected_velocity - velocity_anchor
                thermal_variance = selected_width ** 2
                context = (velocity_anchor, velocity_offset, thermal_variance)
                contexts.append((emitting_cells, context))
            velocity_anchor, velocity_offset, thermal_variance = context
            mean_offset = float(
                np.sum(selected_luminosity * velocity_offset) / total_luminosity
            )
            centroid = float(velocity_anchor + mean_offset)
            bulk_variance = (velocity_offset - mean_offset) ** 2
            centered_second = float(
                np.sum(selected_luminosity * (bulk_variance + thermal_variance))
            )
            result[dust_state, line_index] = LineVelocityMoments(
                luminosity_erg_s=total_luminosity,
                centroid_kms=centroid,
                centered_second_moment=centered_second,
            )
    return result



class HistogramIntegratedSpectra(IntegratedSpectra):
    """Keep existing small sums; defer new spectral templates until final output."""

    def __init__(self, line_keys, velocity_edges_kms, *, engine, batch_partial=False,
                 workers=1, cell_chunk=8192, merge_threads=None):
        super().__init__(
            line_keys=line_keys,
            velocity_edges_kms=velocity_edges_kms,
            workers=workers,
            cell_chunk=cell_chunk,
            spectral_method="histogram-2d",
        )
        self.spectral_method = "histogram-2d"
        self.engine = engine
        self.batch_partial = batch_partial
        self._merger = None if batch_partial else NodeMajorMerger(threads=merge_threads)
        self.profile_count = 2 * len(self.line_keys) * 2
        self._sparse_updates = []
        self._histogram = None
        self._finalized = False
        self._fallback_input = np.zeros(self.profile_count)
        self._registered_input = np.zeros(self.profile_count)
        self.histogram_statistics = {
            "deposit_seconds": 0.0,
            "sparse_merge_seconds": 0.0,
            "fallback_seconds": 0.0,
            "expansion_seconds": 0.0,
            "spectral_add_batch_seconds": 0.0,
            "kernel_rows": 0,
            "fallback_rows": 0,
            "histogram_updates": 0,
        }

    def add_batch(self, cells, emission):
        if self._finalized:
            raise RuntimeError("Cannot add cells after final spectral expansion")
        began = time.perf_counter()
        try:
            return super().add_batch(cells=cells, emission=emission)
        finally:
            self.histogram_statistics["spectral_add_batch_seconds"] += time.perf_counter() - began

    def _profile_ids(self, line_keys, branch_index):
        """Map intrinsic-lines then attenuated-lines to (dust,line,branch)."""
        return np.array([
            (dust_index * len(self.line_keys) + self.line_keys.index(line_key)) * 2 + branch_index
            for dust_index in range(2)
            for line_key in line_keys
        ], dtype=np.intp)

    def _merge_sparse(self, profile_ids, nodes, weights):
        """Merge unique-node weights into the main owned matrix, once per batch."""
        if nodes.size == 0:
            return
        began = time.perf_counter()
        if self._histogram is None:
            self._histogram = np.zeros((self.engine.node_count, self.profile_count))
        # deposit() coalesces repeated IDs; each profile ID is unique too.
        self._merger.merge(self._histogram, nodes, profile_ids, weights)
        self.histogram_statistics["sparse_merge_seconds"] += time.perf_counter() - began

    def accumulate_group_histogram(
        self, *, line_keys, branch_index, velocity_kms, thermal_width_kms,
        luminosity_matrix, velocity_edges_kms, cell_chunk, workers, spectral_method,
    ):
        """Deposit all emitting physical Gaussians; no one-dimensional fallback."""
        effective_sigma = np.maximum(thermal_width_kms, MINIMUM_GAUSSIAN_WIDTH_KMS)
        began = time.perf_counter()
        increment = self.engine.deposit(velocity_kms, effective_sigma, luminosity_matrix)
        self.histogram_statistics["deposit_seconds"] += time.perf_counter() - began
        self.histogram_statistics["kernel_rows"] += int(velocity_kms.size)
        self.histogram_statistics["fallback_rows"] += int(np.count_nonzero(increment.fallback_mask))
        self.histogram_statistics["histogram_updates"] += int(increment.nodes.shape[0])
        profile_ids = self._profile_ids(line_keys, branch_index)
        self._registered_input[profile_ids] += increment.weights.sum(axis=0)
        if self.batch_partial:
            # The increment owns these arrays. No cell mask or scratch view survives.
            self._sparse_updates.append((profile_ids, increment.nodes, increment.weights))
        else:
            self._merge_sparse(profile_ids, increment.nodes, increment.weights)
        if np.any(increment.fallback_mask):
            raise AssertionError("A pure 2D deposit returned fallback rows")
        return np.zeros((velocity_edges_kms.size - 1, luminosity_matrix.shape[1]))

    def merge(self, other):
        if self.batch_partial or self._finalized or other._finalized:
            raise RuntimeError("Merge unfinalized worker partials into the main accumulator")
        if self.engine.grid != other.engine.grid or self.line_keys != other.line_keys:
            raise ValueError("Cannot merge different histogram grids or line orders")
        super().merge(other)
        for name, value in other.histogram_statistics.items():
            self.histogram_statistics[name] += value
        for profile_ids, nodes, weights in other._sparse_updates:
            self._merge_sparse(profile_ids, nodes, weights)
        if other._histogram is not None:
            raise RuntimeError("A parallel partial must not retain a dense histogram")
        self._registered_input += other._registered_input
        self._fallback_input += other._fallback_input
        return self

    def build_output(self, projected_area_cm2):
        if self.batch_partial:
            raise RuntimeError("Merge worker partials before building spectral output")
        if not self._finalized:
            expected = self.input_luminosity.reshape(-1)
            histogram_light = np.zeros(self.profile_count)
            if self._histogram is not None:
                histogram_light = self._histogram.sum(axis=0)
            accounted = histogram_light + self._fallback_input
            if not np.allclose(accounted, expected, rtol=1e-10, atol=0.0):
                raise AssertionError("Histogram plus fallback lost input luminosity")
            if not np.allclose(histogram_light, self._registered_input, rtol=1e-10, atol=0.0):
                raise AssertionError("Sparse histogram merging lost deposited luminosity")
            relative = np.divide(
                np.abs(accounted - expected), expected,
                out=np.zeros_like(expected), where=expected > 0.0,
            )
            self.histogram_statistics["maximum_input_conservation_relative_error"] = float(relative.max(initial=0.0))
            self.histogram_statistics["histogram_luminosity_by_profile"] = histogram_light.tolist()
            self.histogram_statistics["fallback_luminosity_by_profile"] = self._fallback_input.tolist()
            self.histogram_statistics["input_luminosity_by_profile"] = expected.tolist()
            if self._histogram is not None:
                began = time.perf_counter()
                nodes = np.flatnonzero(np.any(self._histogram, axis=1))
                expanded = self.engine.expand(nodes, self._histogram[nodes], self.velocity_edges_kms)
                # Existing profile order is dust,line,branch,channel.
                self.dL_dv += expanded.T.reshape(self.dL_dv.shape)
                self.histogram_statistics["expansion_seconds"] += time.perf_counter() - began
                self.histogram_statistics["occupied_nodes"] = int(nodes.size)
                self._histogram = None
            else:
                self.histogram_statistics["occupied_nodes"] = 0
            self._finalized = True
        payload, report = super().build_output(projected_area_cm2=projected_area_cm2)
        for dust_report in report.values():
            dust_report["histogram_grid"] = asdict(self.engine.grid)
            dust_report["template_contraction"] = self.engine.expansion_statistics["contraction"]
            dust_report["template_expansion_statistics"] = self.engine.expansion_statistics
            dust_report["deposition_backend"] = "compiled original-row selection, indexed corner-major scatter, stable radix node sorting"
            dust_report["merge_backend"] = self._merger.statistics
            dust_report["histogram_statistics"] = self.histogram_statistics
            dust_report["fallback"] = "Disabled: all emitting Gaussians use 2D; out-of-grid inputs raise"
        if not self.histogram_statistics['occupied_nodes']:
            self.engine.untruncated_window_luminosity = np.zeros(self.profile_count)
            self.engine.clipped_window_luminosity = np.zeros(self.profile_count)
        audit = add_cutoff_accounts(payload, engine=self.engine, cutoff_sigma=5.0)
        update_cutoff_report(report, payload, engine=self.engine, audit=audit)
        return payload, report

    def calculate_available_line_group(
        self, *, cells, emission, line_keys, available_cells, cell_volume_cm3,
    ):
        """Build the two final F-order branch arrays directly from original fields.

        Original line emissivities/temperatures have shape (B,). The native helper
        selects the shared availability mask, then fills (Nline, Nbranch) matrices
        directly in cell order. It checks the same physical finite/nonnegative
        rules before any histogram deposit, without keeping full-group copies.
        """
        lines = tuple(emission.lines[line_key] for line_key in line_keys)
        available_rows = np.flatnonzero(available_cells)
        branches = prepare_available_group(
            velocity_kms=np.asarray(cells.velocity_z_kms, dtype=np.float64),
            cold_cells=np.asarray(emission.cold_cells, dtype=np.bool_),
            available_rows=available_rows,
            intrinsic_fields=tuple(
                np.asarray(line.intrinsic_emissivity_erg_s_cm3, dtype=np.float64)
                for line in lines
            ),
            attenuated_fields=tuple(
                np.asarray(line.attenuated_emissivity_erg_s_cm3, dtype=np.float64)
                for line in lines
            ),
            temperature_fields=tuple(
                np.asarray(line.temperature_K, dtype=np.float64) for line in lines
            ),
            cell_volume_cm3=cell_volume_cm3,
            speed_of_light_kms=SPEED_OF_LIGHT_KMS,
        )
        profile_shape = (2, len(line_keys), 2, self.velocity_edges_kms.size - 1)
        profiles = np.zeros(profile_shape)
        luminosities = np.zeros(profile_shape[:3])
        counts = np.zeros((len(line_keys), 2), dtype=np.int64)
        moments = {
            dust_state: {
                line_key: {"cold": LineVelocityMoments(), "hot": LineVelocityMoments()}
                for line_key in line_keys
            }
            for dust_state in DUST_STATES
        }
        masses_g = np.array([
            LINE_DEFINITIONS[line_key].emitter_mass_amu for line_key in line_keys
        ]) * ATOMIC_MASS_UNIT_G
        for branch_index, branch_fields in enumerate(branches):
            velocity, temperature, intrinsic, attenuated = branch_fields
            branch_count = velocity.size
            if branch_count == 0:
                continue
            branch_luminosities = dict(zip(DUST_STATES, (intrinsic, attenuated)))
            branch_profiles, branch_moments = self.calculate_branch_histogram_and_moments(
                branch_index=branch_index,
                line_keys=line_keys,
                masses_g=masses_g,
                velocity_kms=velocity,
                temperature_K=temperature,
                luminosity_by_dust=branch_luminosities,
                velocity_edges_kms=self.velocity_edges_kms,
                cell_chunk=self.cell_chunk,
                workers=self.workers,
                spectral_method=self.spectral_method,
            )
            profiles[:, :, branch_index] = branch_profiles
            counts[:, branch_index] = branch_count
            branch_name = ("cold", "hot")[branch_index]
            for dust_index, dust_state in enumerate(DUST_STATES):
                # F order is essential: preserve the frozen axis-1 sum order.
                luminosities[dust_index, :, branch_index] = branch_luminosities[dust_state].sum(axis=1)
                for line_key in line_keys:
                    moments[dust_state][line_key][branch_name] = branch_moments[dust_state][line_key]
        return profiles, luminosities, counts, moments


    def calculate_branch_histogram_and_moments(
        self, *, branch_index, line_keys, masses_g, velocity_kms, temperature_K,
        luminosity_by_dust, velocity_edges_kms, cell_chunk, workers, spectral_method="exact",
    ):
        """Keep the frozen width-group and native deposit order without repacking.

        The only changed moment work is sharing velocity offsets and sigma squared
        between profiles whose positive masks match. Each line's sums stay separate.
        """
        profiles = np.zeros((2, len(line_keys), velocity_edges_kms.size - 1))
        moments = {
            dust_state: {line_key: LineVelocityMoments() for line_key in line_keys}
            for dust_state in DUST_STATES
        }
        emitting_lines = np.zeros(len(line_keys), dtype=bool)
        for luminosity in luminosity_by_dust.values():
            emitting_lines |= np.any(luminosity != 0.0, axis=1)
        remaining_lines = list(np.flatnonzero(emitting_lines))
        while remaining_lines:
            first_line = remaining_lines[0]
            shared_lines = [
                line_index for line_index in remaining_lines
                if masses_g[line_index] == masses_g[first_line]
                and np.array_equal(temperature_K[line_index], temperature_K[first_line])
            ]
            remaining_lines = [
                line_index for line_index in remaining_lines if line_index not in shared_lines
            ]
            thermal_width = calculate_thermal_width_kms(
                velocity_kms=velocity_kms,
                temperature_K=temperature_K[first_line],
                mass_g=masses_g[first_line],
            )
            group_moments = shared_width_moments(
                velocity_kms=velocity_kms,
                thermal_width_kms=thermal_width,
                luminosity_by_dust=luminosity_by_dust,
                shared_lines=shared_lines,
            )
            for dust_state in DUST_STATES:
                for line_index in shared_lines:
                    moments[dust_state][line_keys[line_index]] = group_moments[dust_state, line_index]
            # Fill the exact final layout once, without two advanced-index copies.
            luminosity_matrix = pack_width_group(
                intrinsic_luminosity=luminosity_by_dust[DUST_STATES[0]],
                attenuated_luminosity=luminosity_by_dust[DUST_STATES[1]],
                shared_lines=np.asarray(shared_lines, dtype=np.int64),
                cell_major=self.spectral_method == "direct-template",
            )
            channel_profiles = self.accumulate_group_histogram(
                line_keys=tuple(line_keys[index] for index in shared_lines),
                branch_index=branch_index,
                velocity_kms=velocity_kms,
                thermal_width_kms=thermal_width,
                luminosity_matrix=luminosity_matrix,
                velocity_edges_kms=velocity_edges_kms,
                cell_chunk=cell_chunk,
                workers=workers,
                spectral_method=spectral_method,
            )
            group_size = len(shared_lines)
            for dust_index in range(2):
                first_column = dust_index * group_size
                last_column = first_column + group_size
                profiles[dust_index, shared_lines] = channel_profiles[:, first_column:last_column].T
        return profiles, moments
