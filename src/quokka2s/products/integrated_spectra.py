"""Prepare available line Gaussians, complete moments and saved spectrum arrays.

The template accumulator supplies nearest-node accumulation and convolution.
This shared preparation preserves per-line missingness, temperature branches,
original-cell luminosities and complete Gaussian moments.
"""
from __future__ import annotations

import numpy as np

from quokka2s.constants import ATOMIC_MASS_UNIT_G, BOLTZMANN_ERG_K, SPEED_OF_LIGHT_KMS
from quokka2s.line_definitions import LINE_DEFINITIONS
from quokka2s.products import DUST_STATES, REGIME_KEYS
from quokka2s.products.gaussian.spectral_fields import (
    calculate_centered_full_moments,
    pack_width_group,
    prepare_available_group,
    prepare_branch_rows,
    prepare_direct_group_luminosity,
    selected_temperatures_equal,
    validate_branch_emission,
)
from quokka2s.products.line_velocity_moments import (
    LineVelocityMoments,
    MINIMUM_GAUSSIAN_WIDTH_KMS,
)
from quokka2s.products.profile_preparation import add_spectral_display_fields


# Allow measured float64 summation-order differences between independent sums.
SPECTRUM_LUMINOSITY_RTOL = 1e-10


def shared_width_moments(*, velocity_kms, thermal_width_kms, luminosity_by_dust, shared_lines):
    """Return stable full moments, reusing only identical input vectors.

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


class IntegratedSpectra:
    """Prepare bounded batches and own the common saved spectral products.

    dL_dv is (2, L, 2, K) [erg/s/(km/s)]: intrinsic/attenuated, line,
    cold/hot gas, velocity channel. input_luminosity is (2, L, 2) [erg/s].
    The template subclass queues Gaussian groups and accumulates their corners.
    """

    def __init__(
        self,
        line_keys,
        velocity_edges_kms,
    ):
        """Create empty profiles on the run's fixed velocity-channel grid.

        line_keys follows CellEmissionCalculator.line_keys. Channel edges are
        (K + 1,) [km/s]; batch ownership bounds temporary Gaussian preparation.
        Complete Gaussian moments retain their original cell calculation.
        Example: ten lines and 400 channels give dL_dv.shape == (2, 10, 2, 400).
        """
        self.line_keys = tuple(line_keys)
        self.velocity_edges_kms = np.asarray(velocity_edges_kms, dtype=float).copy()
        self.spectral_method = "direct-template"
        profile_shape = (2, len(self.line_keys), 2, self.velocity_edges_kms.size - 1)
        self.dL_dv = np.zeros(profile_shape)
        self.input_luminosity = np.zeros(profile_shape[:3])
        self.cell_counts = np.zeros((len(self.line_keys), 2), dtype=np.int64)
        self.full_line_moments = {
            dust_state: {
                line_key: {"cold": LineVelocityMoments(), "hot": LineVelocityMoments()}
                for line_key in self.line_keys
            }
            for dust_state in DUST_STATES
        }

    def add_batch(self, cells, emission):
        """Add each line's available cells to both dust states and gas branches.

        cells is CellBatch: velocity_z_kms is (B,) [km/s], cell_volume_cm3 is
        the uniform scalar volume [cm^3]. emission is BatchEmission: each named
        line supplies (B,) intrinsic/attenuated epsilon [erg/s/cm^3], thermal
        temperature [K], and its own missingness mask. Physical zeros remain.
        All group increments are calculated before the accumulated sums change.
        """
        volume = float(cells.cell_volume_cm3)
        if not np.isfinite(volume) or volume <= 0.:
            raise ValueError("cell_volume_cm3 must be finite and positive")
        increments = []
        for line_keys, available_cells in self.group_lines_by_available_cells(emission=emission):
            profiles, luminosities, counts, moments = self.calculate_available_line_group(
                cells=cells,
                emission=emission,
                line_keys=line_keys,
                available_cells=available_cells,
                cell_volume_cm3=volume,
            )
            increments.append((line_keys, profiles, luminosities, counts, moments))
        for line_keys, profiles, luminosities, counts, moments in increments:
            line_indices = [self.line_keys.index(line_key) for line_key in line_keys]
            self.dL_dv[:, line_indices] += profiles
            self.input_luminosity[:, line_indices] += luminosities
            self.cell_counts[line_indices] += counts
            self.merge_full_line_moments(
                moments=moments,
                line_keys=line_keys,
            )
        return self

    def group_lines_by_available_cells(self, *, emission):
        """Return line-name groups whose (B,) available-cell masks match.

        A missing CO result must not remove an available Halpha result. Groups
        preserve declared line order and share only their temporary kernel input.
        """
        groups = []
        for line_key in self.line_keys:
            available_cells = ~emission.lines[line_key].emissivity_is_missing
            for line_keys, group_cells in groups:
                if np.array_equal(group_cells, available_cells):
                    line_keys.append(line_key)
                    break
            else:
                groups.append(([line_key], available_cells))
        return groups

    def calculate_available_line_group(
        self, *, cells, emission, line_keys, available_cells, cell_volume_cm3,
    ):
        """Prepare one group of lines sharing exactly the same available cells.

        Normal processing builds profile-contiguous queues directly. A custom NumPy
        error policy retains the original Fortran-order moment preparation.
        """
        default_numpy_errors = {
            "divide": "warn",
            "over": "warn",
            "under": "ignore",
            "invalid": "warn",
        }
        if np.geterr() == default_numpy_errors:
            return self.calculate_direct_line_group(
                cells=cells,
                emission=emission,
                line_keys=line_keys,
                available_cells=available_cells,
                cell_volume_cm3=cell_volume_cm3,
            )
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
            branch_profiles, branch_moments = self.prepare_branch_gaussians_and_moments(
                branch_index=branch_index,
                line_keys=line_keys,
                masses_g=masses_g,
                velocity_kms=velocity,
                temperature_K=temperature,
                luminosity_by_dust=branch_luminosities,
                velocity_edges_kms=self.velocity_edges_kms,
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

    def calculate_direct_line_group(
        self, *, cells, emission, line_keys, available_cells, cell_volume_cm3,
    ):
        """Build final template queues directly from selected source fields.

        No full branch temperature or luminosity matrix is retained. Lines may
        share a queue only when their emitter masses and selected branch
        temperatures are identical. Stable two-pass NumPy full moments reuse
        the final queue; all masks and the Gaussian integration are unchanged.
        """
        lines = tuple(emission.lines[line_key] for line_key in line_keys)
        intrinsic_fields = tuple(
            np.asarray(line.intrinsic_emissivity_erg_s_cm3, dtype=np.float64)
            for line in lines
        )
        attenuated_fields = tuple(
            np.asarray(line.attenuated_emissivity_erg_s_cm3, dtype=np.float64)
            for line in lines
        )
        temperature_fields = tuple(
            np.asarray(line.temperature_K, dtype=np.float64) for line in lines
        )
        available_rows = np.flatnonzero(available_cells)
        branches = prepare_branch_rows(
            available_rows=available_rows,
            cold_cells=np.asarray(emission.cold_cells, dtype=np.bool_),
            velocity_kms=np.asarray(cells.velocity_z_kms, dtype=np.float64),
            speed_of_light_kms=SPEED_OF_LIGHT_KMS,
        )
        line_count = len(line_keys)
        profile_shape = (2, line_count, 2, self.velocity_edges_kms.size - 1)
        profiles = np.zeros(profile_shape)
        luminosities = np.zeros(profile_shape[:3])
        counts = np.zeros((line_count, 2), dtype=np.int64)
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
        for branch_index, (rows, velocity) in enumerate(branches):
            counts[:, branch_index] = rows.size
            if rows.size == 0:
                continue
            emitting_lines = validate_branch_emission(
                rows=rows,
                intrinsic_fields=intrinsic_fields,
                attenuated_fields=attenuated_fields,
                temperature_fields=temperature_fields,
                cell_volume_cm3=cell_volume_cm3,
                original_count=cells.cell_count,
            )
            remaining_lines = list(np.flatnonzero(emitting_lines))
            while remaining_lines:
                first_line = remaining_lines[0]
                shared_lines = [
                    line_index for line_index in remaining_lines
                    if masses_g[line_index] == masses_g[first_line]
                    and selected_temperatures_equal(
                        rows=rows,
                        first_temperature_K=temperature_fields[line_index],
                        second_temperature_K=temperature_fields[first_line],
                    )
                ]
                remaining_lines = [
                    line_index for line_index in remaining_lines
                    if line_index not in shared_lines
                ]
                thermal_width = calculate_thermal_width_kms(
                    velocity_kms=velocity,
                    temperature_K=temperature_fields[first_line][rows],
                    mass_g=masses_g[first_line],
                )
                luminosity_matrix = prepare_direct_group_luminosity(
                    rows=rows,
                    intrinsic_fields=intrinsic_fields,
                    attenuated_fields=attenuated_fields,
                    shared_lines=np.asarray(shared_lines, dtype=np.int64),
                    cell_volume_cm3=cell_volume_cm3,
                )
                full_moments = calculate_centered_full_moments(
                    velocity_kms=velocity,
                    thermal_width_kms=thermal_width,
                    luminosity=luminosity_matrix,
                    minimum_width_kms=MINIMUM_GAUSSIAN_WIDTH_KMS,
                )
                group_size = len(shared_lines)
                branch_name = ("cold", "hot")[branch_index]
                for dust_index, dust_state in enumerate(DUST_STATES):
                    for shared_index, line_index in enumerate(shared_lines):
                        profile = dust_index * group_size + shared_index
                        values = full_moments[profile]
                        luminosities[dust_index, line_index, branch_index] = values[0]
                        moments[dust_state][line_keys[line_index]][branch_name] = LineVelocityMoments(
                            luminosity_erg_s=float(values[0]),
                            centroid_kms=float(values[1]),
                            centered_second_moment=float(values[2]),
                        )
                channel_profiles = self.queue_gaussian_group(
                    line_keys=tuple(line_keys[index] for index in shared_lines),
                    branch_index=branch_index,
                    velocity_kms=velocity,
                    thermal_width_kms=thermal_width,
                    luminosity_matrix=luminosity_matrix,
                )
                for dust_index in range(2):
                    first_column = dust_index * group_size
                    last_column = first_column + group_size
                    profiles[dust_index, shared_lines, branch_index] = (
                        channel_profiles[:, first_column:last_column].T
                    )
        return profiles, luminosities, counts, moments

    def prepare_branch_gaussians_and_moments(
        self, *, branch_index, line_keys, masses_g, velocity_kms, temperature_K,
        luminosity_by_dust, velocity_edges_kms,
    ):
        """Keep the frozen width-group and deposit order without repacking.

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
            )
            channel_profiles = self.queue_gaussian_group(
                line_keys=tuple(line_keys[index] for index in shared_lines),
                branch_index=branch_index,
                velocity_kms=velocity_kms,
                thermal_width_kms=thermal_width,
                luminosity_matrix=luminosity_matrix,
            )
            group_size = len(shared_lines)
            for dust_index in range(2):
                first_column = dust_index * group_size
                last_column = first_column + group_size
                profiles[dust_index, shared_lines] = channel_profiles[:, first_column:last_column].T
        return profiles, moments

    def merge_full_line_moments(self, *, moments, line_keys):
        """Merge complete Gaussian moments in dust, line and cold/hot order.

        moments is keyed by dust state, line name and gas branch, with one
        LineVelocityMoments value per branch. line_keys may be a batch group.
        """
        for dust_state in DUST_STATES:
            for line_key in line_keys:
                for branch in ("cold", "hot"):
                    previous = self.full_line_moments[dust_state][line_key][branch]
                    increment = moments[dust_state][line_key][branch]
                    self.full_line_moments[dust_state][line_key][branch] = previous.merged(increment)

    def build_output(self, projected_area_cm2):
        """Build the final NPZ fields and dust-state luminosity reports once.

        Profiles use (dust, line, cold/hot, channel). Full and window moments
        use (dust, line), combining both branches. Empty lines have NaN moments.
        Surface-luminosity and peak-normalized profiles are calculated here.
        projected_area_cm2 is the processed region's x-y area [cm^2]; full-box
        processing uses the original snapshot area.
        """
        edges = self.velocity_edges_kms
        centers = .5 * (edges[:-1] + edges[1:])
        total_profiles = self.dL_dv.sum(axis=2)
        captured = np.sum(self.dL_dv * np.diff(edges), axis=-1)
        outside = np.maximum(0., self.input_luminosity - captured)
        centroid, sigma = calculate_channel_line_moments(
            total_profiles=total_profiles,
            velocity_edges_kms=edges,
            velocity_centers_kms=centers,
        )
        regime_centroid, regime_sigma = calculate_channel_line_moments(
            total_profiles=self.dL_dv,
            velocity_edges_kms=edges,
            velocity_centers_kms=centers,
        )
        payload = {
            'schema_version': np.asarray(1),
            'line_keys': np.asarray(self.line_keys),
            'dust_state_keys': np.asarray(DUST_STATES),
            'regime_keys': np.asarray(REGIME_KEYS),
            'axis_order': np.asarray('dust_state,line,regime,velocity_channel'),
            'velocity_edges_kms': edges.copy(),
            'velocity_kms': centers,
            'dL_dv_erg_s_per_kms': self.dL_dv.copy(),
            'total_dL_dv_erg_s_per_kms': total_profiles,
            'input_luminosity_erg_s': self.input_luminosity.copy(),
            'captured_luminosity_erg_s': captured,
            'outside_velocity_luminosity_erg_s': outside,
            'line_centroid_window_kms': centroid,
            'line_sigma_window_kms': sigma,
            'line_centroid_window_by_regime_kms': regime_centroid,
            'line_sigma_window_by_regime_kms': regime_sigma,
            'cell_counts_by_regime': self.cell_counts.copy(),
            'projected_area_cm2': np.asarray(projected_area_cm2),
            'line_of_sight': np.asarray('z'),
        }
        payload.update(self.build_full_line_moment_payload())
        add_spectral_display_fields(payload=payload)
        report = self.build_luminosity_report(captured=captured, outside=outside)
        return payload, report

    def build_full_line_moment_payload(self):
        """Combine cold/hot complete Gaussians into (2, L) moment fields."""
        combined = []
        for dust_state in DUST_STATES:
            combined.append([
                self.full_line_moments[dust_state][line_key]["cold"].merged(
                    self.full_line_moments[dust_state][line_key]["hot"],
                )
                for line_key in self.line_keys
            ])
        return {
            "full_line_luminosity_erg_s": np.array([
                [moment.luminosity_erg_s for moment in dust_moments]
                for dust_moments in combined
            ]),
            "line_centroid_full_kms": np.array([
                [moment.centroid_kms if moment.luminosity_erg_s > 0. else np.nan
                 for moment in dust_moments]
                for dust_moments in combined
            ]),
            "line_sigma_full_kms": np.array([
                [moment.sigma_kms for moment in dust_moments]
                for dust_moments in combined
            ]),
            "line_second_raw_moment_full_kms2": np.array([
                [moment.raw_second_moment_kms2 for moment in dust_moments]
                for dust_moments in combined
            ]),
        }

    def build_luminosity_report(self, *, captured, outside):
        """Describe complete and saved-window luminosities [erg/s] per line."""
        outside_fraction = np.divide(
            outside,
            self.input_luminosity,
            out=np.zeros_like(outside),
            where=self.input_luminosity > 0.,
        )
        total_input = self.input_luminosity.sum(axis=2)
        total_captured = captured.sum(axis=2)
        total_outside = np.maximum(0., total_input - total_captured)
        total_outside_fraction = np.divide(
            total_outside,
            total_input,
            out=np.zeros_like(total_input),
            where=total_input > 0.,
        )
        report = {}
        for dust_index, dust_state in enumerate(DUST_STATES):
            dust_report = {
                "line_keys": list(self.line_keys),
                "regime_keys": list(REGIME_KEYS),
                "axis_order": "line,regime,velocity_channel",
                "velocity_range_kms": self.velocity_edges_kms[[0, -1]].tolist(),
                "velocity_channels": int(self.velocity_edges_kms.size - 1),
                "cell_counts_by_regime": self.cell_counts.tolist(),
                "profile": "analytic Gaussian integral over channel edges; existing Doppler width convention",
                "spectral_method": self.spectral_method,
                "transfer": "none added; supplied emissivities are summed as epsilon times cell volume",
                "lines": {},
            }
            for line_index, line_key in enumerate(self.line_keys):
                line_report = {
                    "total": make_luminosity_summary(
                        input_erg_s=total_input[dust_index, line_index],
                        captured_erg_s=total_captured[dust_index, line_index],
                        outside_erg_s=total_outside[dust_index, line_index],
                        outside_fraction=total_outside_fraction[dust_index, line_index],
                    ),
                }
                for branch_index, branch in enumerate(REGIME_KEYS):
                    line_report[branch] = make_luminosity_summary(
                        input_erg_s=self.input_luminosity[dust_index, line_index, branch_index],
                        captured_erg_s=captured[dust_index, line_index, branch_index],
                        outside_erg_s=outside[dust_index, line_index, branch_index],
                        outside_fraction=outside_fraction[dust_index, line_index, branch_index],
                    )
                dust_report["lines"][line_key] = line_report
            report[dust_state] = dust_report
        return report


def make_luminosity_summary(input_erg_s, captured_erg_s, outside_erg_s, outside_fraction):
    """Convert one line's window-accounting scalars into a JSON report entry."""
    return {
        "input_luminosity_erg_s": float(input_erg_s),
        "captured_luminosity_erg_s": float(captured_erg_s),
        "outside_velocity_luminosity_erg_s": float(outside_erg_s),
        "outside_velocity_fraction": float(outside_fraction),
    }


def calculate_thermal_width_kms(velocity_kms, temperature_K, mass_g):
    """Return sqrt(k_B*T/m)/1e5 * (1-v/c), shape (R,), in km/s.

    velocity_kms and temperature_K are the retained cells for one thermal group;
    mass_g is that line's particle mass [g]. The channel kernel applies the
    existing 1 cm/s minimum width after this calculation.
    """
    thermal_width = np.sqrt(BOLTZMANN_ERG_K * temperature_K / mass_g) / 1.e5
    thermal_width *= 1. - velocity_kms / SPEED_OF_LIGHT_KMS
    return thermal_width


def calculate_channel_line_moments(total_profiles, velocity_edges_kms, velocity_centers_kms):
    """Return channel-luminosity-weighted line centroids and dispersions [km/s].

    total_profiles is (..., Nchannel) [erg/s/(km/s)]: summed profiles or
    separate cold/hot profiles. Channel widths give luminosity weights [erg/s].
    Returned arrays have the leading input shape; a zero-light line has NaN.
    For example, sigma[1, 1] is the attenuated Halpha dispersion.
    """
    weights = total_profiles * np.diff(velocity_edges_kms)
    captured_luminosity = weights.sum(axis=-1)
    centers = velocity_centers_kms
    with np.errstate(divide="ignore", invalid="ignore"):
        centroid = np.divide(
            np.sum(weights * centers, axis=-1),
            captured_luminosity,
            out=np.full_like(captured_luminosity, np.nan),
            where=captured_luminosity > 0,
        )
        variance = np.divide(
            np.sum(weights * (centers - centroid[..., None])**2, axis=-1),
            captured_luminosity,
            out=np.full_like(captured_luminosity, np.nan),
            where=captured_luminosity > 0,
        )
    return centroid, np.sqrt(variance)


def check_spectrum_luminosity(payload, expected_luminosity):
    """Compare generated spectra with independent cell luminosities [erg/s].

    payload comes from IntegratedSpectra.build_output(); expected_luminosity
    has shape (2, Nline, 2): dust state, line, temperature regime. Returns None;
    raises ValueError when the channel integral differs from captured light,
    or input light, captured + outside light, or full-moment luminosity differs
    from the independent cell sums. Full moments combine both branches.
    Different float64 summation orders are allowed at rtol=1e-10; an expected
    zero must remain exactly zero. The error identifies the dust state, line,
    and cold/hot branch. Neither input is modified.
    """
    expected = np.asarray(expected_luminosity, dtype=float)
    input_luminosity = np.asarray(payload['input_luminosity_erg_s'], dtype=float)
    window_and_outside_luminosity = (
        np.asarray(payload['captured_luminosity_erg_s'], dtype=float)
        + np.asarray(payload['outside_velocity_luminosity_erg_s'], dtype=float)
    )
    channel_widths_kms = np.diff(payload['velocity_edges_kms'])
    integrated_channel_luminosity = np.sum(
        payload['dL_dv_erg_s_per_kms'] * channel_widths_kms,
        axis=-1,
    )
    comparisons = (
        ('Spectrum input differs from independent cell luminosities', input_luminosity, expected),
        ('Spectral window and outside luminosity do not sum to cell luminosities',
         window_and_outside_luminosity, expected),
        ('Full-profile moments do not include the independent cell luminosities',
         np.asarray(payload['full_line_luminosity_erg_s']), expected.sum(axis=-1)),
        ('Integrated velocity channels differ from captured luminosity',
         integrated_channel_luminosity, np.asarray(payload['captured_luminosity_erg_s'])),
    )
    for description, actual, expected_totals in comparisons:
        if (np.isfinite(actual).all() and np.isfinite(expected_totals).all()
                and np.allclose(
            actual,
            expected_totals,
            rtol=SPECTRUM_LUMINOSITY_RTOL,
            atol=0,
        )):
            continue
        difference = np.abs(actual - expected_totals)
        relative_difference = np.divide(
            difference,
            np.abs(expected_totals),
            out=np.zeros_like(difference),
            where=expected_totals != 0,
        )
        relative_difference[(expected_totals == 0) & (difference != 0)] = np.inf
        worst = np.unravel_index(
            np.argmax(relative_difference),
            relative_difference.shape,
        )
        dust_index, line_index = worst[:2]
        line_name = str(payload['line_keys'][line_index])
        dust_state = DUST_STATES[dust_index]
        location = f'{dust_state} {line_name}'
        if len(worst) == 3:
            regime = ('cold', 'hot')[worst[2]]
            location += f' {regime}'
        raise ValueError(
            f'{description}: {location} has '
            f'relative difference {relative_difference[worst]:.3e} '
            f'(spectrum={actual[worst]:.6e}, cells={expected_totals[worst]:.6e}; '
            f'rtol={SPECTRUM_LUMINOSITY_RTOL:.1e})'
        )
