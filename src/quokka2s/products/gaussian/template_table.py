"""Preintegrated fractional-mean templates with sigma-specific compact supports.

Prepare the immutable probability table before processing a snapshot.
Online evaluation only gathers stored bin masses; it never evaluates a CDF.
Velocities and standard deviations use km/s. Fractional phases are dimensionless.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from time import perf_counter

import numpy as np

from quokka2s.products.gaussian.intervals import interval_probability
from quokka2s.products.line_velocity_moments import MINIMUM_GAUSSIAN_WIDTH_KMS
from quokka2s.products.velocity_grid import velocity_channel_edges_kms


@dataclass(frozen=True)
class TemplateParameters:
    """Set interpolation spacing, finite sigma coverage and fixed node cutoff."""

    sigma_min: float = MINIMUM_GAUSSIAN_WIDTH_KMS
    sigma_max: float = 1024.0
    log_sigma_step: float = 0.03125
    mu_spacing_ratio: float = 0.1
    maximum_phase_step: float = 0.1
    cutoff_sigma: float = 5.0

    def __post_init__(self):
        values = np.asarray(tuple(asdict(self).values()), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Template parameters must be finite")
        if self.sigma_min < MINIMUM_GAUSSIAN_WIDTH_KMS:
            raise ValueError("Template sigma_min cannot lie below the physical width floor")
        if self.sigma_max <= self.sigma_min:
            raise ValueError("Require sigma_max greater than sigma_min")
        if self.log_sigma_step <= 0.0 or self.mu_spacing_ratio <= 0.0:
            raise ValueError("Template logarithmic and mean spacing must be positive")
        if self.maximum_phase_step <= 0.0 or self.maximum_phase_step > 1.0:
            raise ValueError("maximum_phase_step must lie in (0,1]")
        if self.cutoff_sigma != 5.0:
            raise ValueError("Integrated templates retain the five-sigma node cutoff")


def _readonly(array, dtype=None):
    """Own one immutable contiguous numerical buffer."""
    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


class FractionalGaussianTable:
    """Store integrated bin masses for phase/log-sigma interpolation.

    A physical mean is split as mu/delta_v = integer + phase. Layer-specific
    phase nodes cover [0,1], with small spacings near narrow Gaussian bin edges.
    A narrow layer's exactly flat interior needs only its two boundary knots.
    Each layer stores the entire union of five-sigma supports for all phases,
    plus known-zero guard bins. Thus shifts are not restricted to the observed
    400-channel window. Values outside a layer's stored support are exactly zero
    under that layer's cutoff, rather than extrapolated or silently discarded.
    """

    method = "fractional-phase-log-sigma-template"

    def __init__(
        self,
        *,
        sigma_min=MINIMUM_GAUSSIAN_WIDTH_KMS,
        sigma_max=1024.0,
        log_sigma_step=0.03125,
        mu_spacing_ratio=0.1,
        maximum_phase_step=0.1,
        edges=None,
    ):
        began = perf_counter()
        self.parameters = TemplateParameters(
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            log_sigma_step=float(log_sigma_step),
            mu_spacing_ratio=float(mu_spacing_ratio),
            maximum_phase_step=float(maximum_phase_step),
        )
        selected_edges = velocity_channel_edges_kms() if edges is None else edges
        selected_edges = np.asarray(selected_edges, dtype=np.float64)
        if selected_edges.ndim != 1 or selected_edges.size < 2:
            raise ValueError("Template edges must be a nonempty velocity boundary vector")
        widths = np.diff(selected_edges)
        if not np.isfinite(selected_edges).all() or np.any(widths <= 0.0):
            raise ValueError("Template velocity edges must be finite and strictly increasing")
        delta_v = float(widths[0])
        if not np.allclose(widths, delta_v, rtol=1e-12, atol=0.0):
            raise ValueError("Integer-shift templates require uniformly spaced channels")
        first_bin = float(selected_edges[0] / delta_v)
        if first_bin != round(first_bin):
            raise ValueError("Template channel edges must align with the zero-anchored shift grid")
        self.edges = _readonly(selected_edges)
        self.delta_v = delta_v
        self.Z = self.parameters.cutoff_sigma
        self.K = widths.size
        self.output_bin_indices = _readonly(
            int(round(first_bin)) + np.arange(self.K, dtype=np.int64),
        )
        log_minimum = math.log(self.parameters.sigma_min)
        log_maximum = math.log(self.parameters.sigma_max)
        layer_count = int(math.ceil((log_maximum - log_minimum) / self.parameters.log_sigma_step)) + 1
        sigma_nodes = np.exp(log_minimum + self.parameters.log_sigma_step * np.arange(layer_count))
        sigma_nodes[0] = self.parameters.sigma_min
        sigma_nodes[-1] = self.parameters.sigma_max
        if not np.isfinite(sigma_nodes).all() or np.any(np.diff(sigma_nodes) <= 0.0):
            raise ValueError("Requested sigma layers are not distinguishable finite values")
        self.sigma_nodes = _readonly(sigma_nodes)
        self.log_sigma_nodes = _readonly(np.log(sigma_nodes))
        phase_layers = [self._build_phase_nodes(sigma) for sigma in sigma_nodes]
        phase_counts = np.asarray([nodes.size for nodes in phase_layers], dtype=np.int64)
        phase_starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(phase_counts)))
        phase_values = np.concatenate(phase_layers)
        phase_layer_ids = np.repeat(np.arange(layer_count, dtype=np.int64), phase_counts)
        encoded_phases = 2.0 * phase_layer_ids + phase_values
        if np.any(np.diff(encoded_phases) <= 0.0):
            raise ValueError("Layer/phase coordinates are not distinguishable float64 values")
        radii = self.Z * sigma_nodes / self.delta_v
        minimum_bins = np.floor(-radii).astype(np.int64) - 1
        stop_bins = np.ceil(1.0 + radii).astype(np.int64) + 1
        bin_counts = stop_bins - minimum_bins
        row_bin_counts = np.repeat(bin_counts, phase_counts)
        row_starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(row_bin_counts)))
        if row_starts[-1] < 0 or row_starts[-1] > np.iinfo(np.intp).max // 8:
            raise ValueError("Requested template table exceeds platform allocation limits")
        values = np.empty(int(row_starts[-1]), dtype=np.float64)
        for layer, sigma in enumerate(sigma_nodes):
            first_row = int(phase_starts[layer])
            stop_row = int(phase_starts[layer + 1])
            phase = phase_layers[layer]
            bins = minimum_bins[layer] + np.arange(bin_counts[layer] + 1)
            standardized = self.delta_v * (bins[None, :] - phase[:, None]) / sigma
            np.clip(standardized, -self.Z, self.Z, out=standardized)
            probabilities = interval_probability(standardized[:, :-1], standardized[:, 1:])
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
                raise FloatingPointError("Integrated templates must be finite and nonnegative")
            if np.any(probabilities[:, 0] != 0.0) or np.any(probabilities[:, -1] != 0.0):
                raise AssertionError("Template boundary bins must be known-zero sentinels")
            first_value = int(row_starts[first_row])
            stop_value = int(row_starts[stop_row])
            values[first_value:stop_value] = probabilities.ravel()
        self.phase_counts = _readonly(phase_counts)
        self.phase_starts = _readonly(phase_starts)
        self.phase_axis_values = _readonly(phase_values)
        self.phase_layer_ids = _readonly(phase_layer_ids)
        self.encoded_phase_axis = _readonly(encoded_phases)
        self.layer_min_bins = _readonly(minimum_bins)
        self.layer_bin_counts = _readonly(bin_counts)
        self.row_min_bins = _readonly(np.repeat(minimum_bins, phase_counts))
        self.row_bin_counts = _readonly(row_bin_counts)
        self.row_value_offsets = _readonly(row_starts[:-1])
        self.flat_values = _readonly(values)
        self.build_seconds = perf_counter() - began
        self._signature = self.signature()

    def _build_phase_nodes(self, sigma):
        """Resolve bin-edge transitions while omitting only an exact flat gap."""
        radius = self.Z * sigma / self.delta_v
        maximum_step = min(
            self.parameters.maximum_phase_step,
            self.parameters.mu_spacing_ratio * sigma / self.delta_v,
        )
        if radius < 0.5:
            transition_intervals = max(1, int(math.ceil(radius / maximum_step)))
            left = np.linspace(0.0, radius, transition_intervals + 1)
            phases = np.concatenate((left, 1.0 - left[::-1]))
        else:
            intervals = max(1, int(math.ceil(1.0 / maximum_step)))
            phases = np.linspace(0.0, 1.0, intervals + 1)
        if phases[0] != 0.0 or phases[-1] != 1.0 or np.any(np.diff(phases) <= 0.0):
            raise ValueError("Phase nodes must distinguish a closed [0,1] interval")
        return phases

    def phase_nodes(self, layer):
        """Return a read-only phase vector for one stored integer sigma layer."""
        layer = int(layer)
        if layer < 0 or layer >= self.sigma_nodes.size:
            raise IndexError("Sigma layer is outside the template table")
        return self.phase_axis_values[self.phase_starts[layer]:self.phase_starts[layer + 1]]

    def signature(self):
        """Hash immutable parameters, axes, indexing metadata and template values."""
        digest = hashlib.sha256()
        digest.update(repr(asdict(self.parameters)).encode("ascii"))
        digest.update(repr((self.delta_v, self.Z, self.K)).encode("ascii"))
        for array in (
            self.edges,
            self.output_bin_indices,
            self.sigma_nodes,
            self.log_sigma_nodes,
            self.phase_counts,
            self.phase_starts,
            self.phase_axis_values,
            self.phase_layer_ids,
            self.encoded_phase_axis,
            self.layer_min_bins,
            self.layer_bin_counts,
            self.row_min_bins,
            self.row_bin_counts,
            self.row_value_offsets,
            self.flat_values,
        ):
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(memoryview(array))
        return digest.hexdigest()

    def validate_prepared_state(self):
        """Refuse mutation after the table was prepared outside process timing."""
        if self.signature() != self._signature:
            raise ValueError("The prepared integrated template table was modified")

    def _validate_inputs(self, centers, sigmas):
        centers = np.asarray(centers, dtype=np.float64)
        sigmas = np.asarray(sigmas, dtype=np.float64)
        if centers.ndim != 1 or sigmas.shape != centers.shape:
            raise ValueError("Template Gaussian means and widths must be equal vectors")
        if not np.isfinite(centers).all() or not np.isfinite(sigmas).all():
            raise ValueError("Template means and widths must be finite")
        if np.any(sigmas < self.sigma_nodes[0]) or np.any(sigmas > self.sigma_nodes[-1]):
            raise ValueError("Gaussian sigma lies outside the stored template coverage")
        coordinates = centers / self.delta_v
        if not np.isfinite(coordinates).all() or np.any(np.abs(coordinates) >= 2**62):
            raise ValueError("Gaussian integer-bin shifts must fit safe signed int64 coordinates")
        shifts_float = np.floor(coordinates)
        phases = coordinates - shifts_float
        shifts = shifts_float.astype(np.int64)
        return centers, sigmas, shifts, phases

    def _phase_brackets(self, layer_ids, phases):
        """Find per-row phase neighbors without Python iteration over cells."""
        encoded = 2.0 * layer_ids + phases
        lower = np.searchsorted(self.encoded_phase_axis, encoded, side="right") - 1
        # Encoding may move a query by one ulp. Correct against actual phase
        # coordinates, so exact/nearly exact phase nodes retain their meaning.
        lower = lower - (phases < self.phase_axis_values[lower])
        upper = lower + 1
        terminal = phases >= 1.0
        upper = np.where(terminal, lower, upper)
        too_low = (~terminal) & (phases > self.phase_axis_values[upper])
        lower = lower + too_low
        upper = upper + too_low
        lower_phase = self.phase_axis_values[lower]
        upper_phase = self.phase_axis_values[upper]
        fractions = np.divide(
            phases - lower_phase,
            upper_phase - lower_phase,
            out=np.zeros_like(phases),
            where=upper != lower,
        )
        if np.any(fractions < -1e-12) or np.any(fractions > 1.0 + 1e-12):
            raise AssertionError("Vectorized phase search failed to bracket an input")
        np.clip(fractions, 0.0, 1.0, out=fractions)
        return lower, upper, fractions

    def _evaluate_chunk(self, sigmas, shifts, phases):
        """Gather four templates into [cell,channel], with bounded local scratch."""
        upper_layers = np.searchsorted(self.sigma_nodes, sigmas, side="left")
        lower_layers = np.maximum(upper_layers - 1, 0)
        lower_log = self.log_sigma_nodes[lower_layers]
        upper_log = self.log_sigma_nodes[upper_layers]
        sigma_fraction = np.divide(
            np.log(sigmas) - lower_log,
            upper_log - lower_log,
            out=np.zeros_like(sigmas),
            where=upper_layers != lower_layers,
        )
        np.clip(sigma_fraction, 0.0, 1.0, out=sigma_fraction)
        low_left, low_right, low_fraction = self._phase_brackets(lower_layers, phases)
        high_left, high_right, high_fraction = self._phase_brackets(upper_layers, phases)
        rows = (low_left, low_right, high_left, high_right)
        corner_weights = (
            (1.0 - sigma_fraction) * (1.0 - low_fraction),
            (1.0 - sigma_fraction) * low_fraction,
            sigma_fraction * (1.0 - high_fraction),
            sigma_fraction * high_fraction,
        )
        base_bins = self.output_bin_indices[None, :] - shifts[:, None]
        probabilities = np.zeros((sigmas.size, self.K), dtype=np.float64)
        gathered = np.empty_like(probabilities)
        for node_rows, weights in zip(rows, corner_weights):
            relative_bins = base_bins - self.row_min_bins[node_rows, None]
            np.clip(relative_bins, 0, self.row_bin_counts[node_rows, None] - 1, out=relative_bins)
            relative_bins += self.row_value_offsets[node_rows, None]
            # First/last stored bins are zero sentinels. Clipping an integer
            # index therefore returns known-zero support outside the template;
            # it never clips a Gaussian parameter or extrapolates a profile.
            np.take(self.flat_values, relative_bins, out=gathered, mode="clip")
            gathered *= weights[:, None]
            probabilities += gathered
        return probabilities

    def evaluate(self, mu, sigma, *, chunk_size=2048):
        """Return stored-template masses [N,K], not luminosity densities.

        Work is vectorized over a subbatch and all channels. The only Python
        loops select bounded subbatches and the fixed four interpolation corners.
        """
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("Template chunk_size must be a positive integer")
        _, sigmas, shifts, phases = self._validate_inputs(mu, sigma)
        output = np.empty((sigmas.size, self.K), dtype=np.float64)
        for start in range(0, sigmas.size, chunk_size):
            stop = min(start + chunk_size, sigmas.size)
            output[start:stop] = self._evaluate_chunk(
                sigmas=sigmas[start:stop],
                shifts=shifts[start:stop],
                phases=phases[start:stop],
            )
        return output

    def info(self):
        """Describe actual storage, finite coverage and interpolation semantics."""
        buffers = (
            self.edges,
            self.output_bin_indices,
            self.sigma_nodes,
            self.log_sigma_nodes,
            self.phase_counts,
            self.phase_starts,
            self.phase_axis_values,
            self.phase_layer_ids,
            self.encoded_phase_axis,
            self.layer_min_bins,
            self.layer_bin_counts,
            self.row_min_bins,
            self.row_bin_counts,
            self.row_value_offsets,
            self.flat_values,
        )
        return {
            "method": self.method,
            "build_seconds": self.build_seconds,
            "table_bytes": int(self.flat_values.nbytes),
            "memory_bytes": int(sum(buffer.nbytes for buffer in buffers)),
            "configuration": {
                **asdict(self.parameters),
                "v_min": float(self.edges[0]),
                "v_max": float(self.edges[-1]),
                "delta_v": self.delta_v,
                "K": self.K,
                "sigma_node_count": int(self.sigma_nodes.size),
                "phase_node_count": int(self.phase_axis_values.size),
                "minimum_phase_count_per_layer": int(self.phase_counts.min()),
                "maximum_phase_count_per_layer": int(self.phase_counts.max()),
                "minimum_extended_bin_count": int(self.layer_bin_counts.min()),
                "maximum_extended_bin_count": int(self.layer_bin_counts.max()),
                "mean_decomposition": "mu/delta_v=floor(mu/delta_v)+phase; negative shifts use floor",
                "phase_interpolation": "linear dimensionless phase separately on each sigma layer",
                "sigma_interpolation": "linear in natural log(sigma); node placement has the same logarithmic spacing",
                "phase_spacing": "At most min(maximum_phase_step,mu_spacing_ratio*sigma/delta_v) in nonflat transition regions",
                "flat_interior": "When 10*sigma < delta_v, only exact five-sigma-flat interior knots are omitted",
                "velocity_storage": "Each layer spans the union of all phase five-sigma supports, with zero guard bins",
                "outside_behavior": "Table-only queries require sigma coverage; TemplateEngine directly integrates valid outside widths; unsafe shifts raise and outside node support is known zero",
                "cutoff_sigma": self.Z,
                "cutoff_applies_to": "interpolated template-node Gaussians, not an exact original-cell support mask",
                "renormalized": False,
                "online_cdf_queries": False,
                "prefix_used": False,
            },
            "table_sha256": self._signature,
        }
