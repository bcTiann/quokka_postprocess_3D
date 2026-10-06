"""Streaming LOS spectra using the existing Gaussian channel integrals.

This module receives each named line's emissivities and gas temperature,
selects its available cells, and accumulates both dust states. It does not
select a chemistry model, interpolate tables, or apply transfer.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.special import erf as scipy_erf

from quokka2s.constants import ATOMIC_MASS_UNIT_G, BOLTZMANN_ERG_K, SPEED_OF_LIGHT_KMS
from quokka2s.products import DUST_STATES
from quokka2s.products.line_velocity_moments import LineVelocityMoments


REGIME_KEYS = ("T_QUOKKA_lt_3000K", "T_QUOKKA_ge_3000K")
# Allow measured float64 summation-order differences between independent sums.
SPECTRUM_LUMINOSITY_RTOL = 1e-10
# Bound temporary (velocity channel, emitting cell) arrays per integration task.
# This controls memory and task size, not the physical Gaussian calculation.
# Measurements: docs/validation/spectral_cell_chunks_20261006.md.
DEFAULT_SPECTRAL_CELL_CHUNK = 8192
LINE_MASSES_AMU = {
    "cii": 12.01, "halpha": 1.00794, "hi21": 1.00794,
    "ciii_977": 12.01, "ciii_1907": 12.01, "ciii_1909": 12.01,
    "civ_1548": 12.01, "civ_1551": 12.01,
    "co10": 28.009, "co21": 28.009,
}



def accumulate_velocity_spectra(
    velocity_kms: np.ndarray,
    thermal_width_kms: np.ndarray,
    luminosity_matrix: np.ndarray,
    velocity_edges_kms: np.ndarray,
    *,
    cell_chunk: int = DEFAULT_SPECTRAL_CELL_CHUNK,
    workers: int = 1,
) -> np.ndarray:
    """Integrate each cell's Gaussian over channel edges and sum dL/dv.

    Parameters
    ----------
    velocity_kms, thermal_width_kms : array-like, shape (R,)
        Cell LOS velocities and Gaussian standard deviations [km/s].
    luminosity_matrix : array-like, shape (R, Nprofile)
        Retained cell luminosities [erg/s]; columns share thermal widths.
        For one line with both dust states, columns are intrinsic and attenuated.
    velocity_edges_kms : array-like, shape (Nchannel + 1,)
        Increasing, uniformly spaced channel edges [km/s].
    cell_chunk, workers : int, optional
        Maximum cells per kernel chunk and number of integration threads.

    Returns
    -------
    numpy.ndarray, shape (Nchannel, Nprofile)
        Channel-average dL/dv [erg/s/(km/s)], in input profile order.

    Notes
    -----
    Each Gaussian is integrated analytically over the velocity-channel edges.
    Several lines with identical widths share one Gaussian evaluation. The
    1 cm/s minimum width is retained; flux outside the window is not restored.

    Examples
    --------
    profile = accumulate_velocity_spectra(velocity, sigma, cell_light, edges)
    """
    velocity = np.asarray(velocity_kms, dtype=float).reshape(-1)
    thermal = np.asarray(thermal_width_kms, dtype=float).reshape(-1)
    luminosity = np.asarray(luminosity_matrix, dtype=float)
    velocity_edges_kms = np.asarray(velocity_edges_kms, dtype=float)
    # The process command constructs a uniform channel grid once. Inputs here
    # have already been selected and prepared by IntegratedSpectra.

    emitting = np.any(luminosity != 0.0, axis=1)
    velocity = velocity[emitting]
    thermal = thermal[emitting]
    luminosity = luminosity[emitting]
    output_shape = (velocity_edges_kms.size - 1, luminosity.shape[1])
    delta_v = float(velocity_edges_kms[1] - velocity_edges_kms[0])
    channel_chunk = 150

    def accumulate_chunk(cell0: int) -> np.ndarray:
        """Return this cell chunk's (channel, profile) dL/dv [erg/s/(km/s)]."""
        cell1 = min(cell0 + cell_chunk, velocity.size)
        centers = velocity[cell0:cell1][None, :]
        values = luminosity[cell0:cell1]
        sigma = np.maximum(thermal[cell0:cell1], 1.0e-5)[None, :]
        denominator = np.sqrt(2.0) * sigma
        partial = np.zeros(output_shape, dtype=float)
        for channel0 in range(0, output_shape[0], channel_chunk):
            channel1 = min(channel0 + channel_chunk, output_shape[0])
            edges = velocity_edges_kms[channel0:channel1 + 1, None]
            erf_edges = (edges - centers) / denominator
            scipy_erf(erf_edges, out=erf_edges)
            fractions = 0.5 * (erf_edges[1:] - erf_edges[:-1])
            partial[channel0:channel1] = np.einsum(
                "kc,cm->km", fractions, values, optimize=False,
            ) / delta_v
        return partial

    chunk_starts = range(0, velocity.size, cell_chunk)
    output = np.zeros(output_shape, dtype=float)
    if workers <= 1:
        for cell0 in chunk_starts:
            output += accumulate_chunk(cell0)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for partial in executor.map(accumulate_chunk, chunk_starts):
                output += partial
    return output


class IntegratedSpectra:
    """Whole-box line profiles and complete moments from bounded cell batches.

    dL_dv is (2, L, 2, K) [erg/s/(km/s)]: intrinsic/attenuated, line,
    cold/hot gas, velocity channel. input_luminosity is (2, L, 2) [erg/s].
    Only these small sums and full-profile moments survive each add_batch().
    """

    def __init__(
        self,
        line_keys,
        velocity_edges_kms,
        *,
        workers=1,
        cell_chunk=DEFAULT_SPECTRAL_CELL_CHUNK,
    ):
        """Create empty profiles on the run's fixed velocity-channel grid.

        line_keys follows CellEmissionCalculator.line_keys. Channel edges are
        (K + 1,) [km/s]; workers and cell_chunk bound Gaussian integration.
        Example: ten lines and 400 channels give dL_dv.shape == (2, 10, 2, 400).
        """
        self.line_keys = tuple(line_keys)
        self.velocity_edges_kms = np.asarray(velocity_edges_kms, dtype=float).copy()
        self.workers = int(workers)
        self.cell_chunk = int(cell_chunk)
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
            for dust_state in DUST_STATES:
                for line_key in line_keys:
                    for branch in ("cold", "hot"):
                        previous = self.full_line_moments[dust_state][line_key][branch]
                        increment = moments[dust_state][line_key][branch]
                        self.full_line_moments[dust_state][line_key][branch] = previous.merged(increment)
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

    def pack_available_line_fields(self, *, emission, line_keys, available_cells):
        """Select three (Lgroup, R) inputs in original available-cell order.

        Returns intrinsic/attenuated epsilon [erg/s/cm^3] and temperature [K].
        Fortran layout preserves the existing per-line luminosity reduction.
        Example: a group's Halpha row comes from emission.lines["halpha"].
        """
        available_count = int(np.count_nonzero(available_cells))
        shape = (len(line_keys), available_count)
        intrinsic_emissivity = np.empty(shape, dtype=float, order="F")
        attenuated_emissivity = np.empty(shape, dtype=float, order="F")
        thermal_temperature_K = np.empty(shape, dtype=float, order="F")
        for line_index, line_key in enumerate(line_keys):
            line = emission.lines[line_key]
            intrinsic_emissivity[line_index] = line.intrinsic_emissivity_erg_s_cm3[available_cells]
            attenuated_emissivity[line_index] = line.attenuated_emissivity_erg_s_cm3[available_cells]
            thermal_temperature_K[line_index] = line.temperature_K[available_cells]
        return intrinsic_emissivity, attenuated_emissivity, thermal_temperature_K

    def calculate_available_line_group(
        self, *, cells, emission, line_keys, available_cells, cell_volume_cm3,
    ):
        """Calculate one availability group's small profile/moment increments.

        The selected cell arrays are temporary. Returned profiles are
        (2, Lgroup, 2, K); luminosities are (2, Lgroup, 2) [erg/s], and counts
        are (Lgroup, 2). Both dust states share velocities and Gaussian widths.
        """
        intrinsic, attenuated, temperature = self.pack_available_line_fields(
            emission=emission,
            line_keys=line_keys,
            available_cells=available_cells,
        )
        velocity = np.asarray(cells.velocity_z_kms[available_cells], dtype=float)
        cold = np.asarray(emission.cold_cells[available_cells], dtype=bool)
        if not np.isfinite(velocity).all():
            raise ValueError("velocity_kms must be finite")
        if np.any(np.abs(velocity) >= SPEED_OF_LIGHT_KMS):
            raise ValueError("The nonrelativistic profile requires abs(velocity) < c")
        if not np.isfinite(temperature).all() or np.any(temperature < 0.):
            raise ValueError("temperatures_K must be finite and nonnegative")
        luminosity_by_dust = {}
        for dust_state, emissivity in zip(DUST_STATES, (intrinsic, attenuated)):
            if not np.isfinite(emissivity).all() or np.any(emissivity < 0.):
                raise ValueError("emissivity must be finite and nonnegative")
            with np.errstate(over="ignore", invalid="ignore"):
                luminosity = emissivity * cell_volume_cm3
            if not np.isfinite(luminosity).all():
                raise ValueError("emissivity times volume must remain finite")
            luminosity_by_dust[dust_state] = luminosity
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
            LINE_MASSES_AMU[line_key] for line_key in line_keys
        ]) * ATOMIC_MASS_UNIT_G
        for branch_index, branch_cells in enumerate((cold, ~cold)):
            if not np.any(branch_cells):
                continue
            branch_luminosities = {
                dust_state: luminosity[:, branch_cells]
                for dust_state, luminosity in luminosity_by_dust.items()
            }
            branch_profiles, branch_moments = calculate_branch_spectra_and_moments(
                line_keys=line_keys,
                masses_g=masses_g,
                velocity_kms=velocity[branch_cells],
                temperature_K=temperature[:, branch_cells],
                luminosity_by_dust=branch_luminosities,
                velocity_edges_kms=self.velocity_edges_kms,
                cell_chunk=self.cell_chunk,
                workers=self.workers,
            )
            profiles[:, :, branch_index] = branch_profiles
            counts[:, branch_index] = np.count_nonzero(branch_cells)
            branch = ("cold", "hot")[branch_index]
            for dust_index, dust_state in enumerate(DUST_STATES):
                luminosities[dust_index, :, branch_index] = branch_luminosities[dust_state].sum(axis=1)
                for line_key in line_keys:
                    moments[dust_state][line_key][branch] = branch_moments[dust_state][line_key]
        return profiles, luminosities, counts, moments

    def merge(self, other):
        """Add a completed worker batch in the caller's chosen input order."""
        self.dL_dv += other.dL_dv
        self.input_luminosity += other.input_luminosity
        self.cell_counts += other.cell_counts
        for dust_state in DUST_STATES:
            for line_key in self.line_keys:
                for branch in ("cold", "hot"):
                    previous = self.full_line_moments[dust_state][line_key][branch]
                    increment = other.full_line_moments[dust_state][line_key][branch]
                    self.full_line_moments[dust_state][line_key][branch] = previous.merged(increment)
        return self

    def build_output(self, projected_area_cm2):
        """Build the final NPZ fields and dust-state luminosity reports once.

        Profiles use (dust, line, cold/hot, channel). Full and window moments
        use (dust, line), combining both branches. Empty lines have NaN moments.
        projected_area_cm2 is snapshot x-y area [cm^2], stored as metadata.
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


def calculate_branch_spectra_and_moments(
    *, line_keys, masses_g, velocity_kms, temperature_K, luminosity_by_dust,
    velocity_edges_kms, cell_chunk, workers,
):
    """Share one thermal Gaussian calculation across lines and dust states.

    Selected temperatures and luminosities are (Lgroup, R); velocities are (R,).
    Lines with identical masses and temperature arrays share channel fractions.
    Returns (2, Lgroup, K) profiles and each line's complete Gaussian moments.
    """
    profiles = np.zeros((2, len(line_keys), velocity_edges_kms.size - 1))
    moments = {
        dust_state: {line_key: LineVelocityMoments() for line_key in line_keys}
        for dust_state in DUST_STATES
    }
    emitting_lines = np.zeros(len(line_keys), dtype=bool)
    for luminosity in luminosity_by_dust.values():
        emitting_lines |= np.any(luminosity != 0., axis=1)
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
        for dust_state, luminosity in luminosity_by_dust.items():
            for line_index in shared_lines:
                line_key = line_keys[line_index]
                moments[dust_state][line_key] = LineVelocityMoments.from_cells(
                    velocity_kms=velocity_kms,
                    thermal_width_kms=thermal_width,
                    luminosity_erg_s=luminosity[line_index],
                )
        # Column order matches the existing reduction: shared intrinsic lines,
        # then shared attenuated lines. Evaluate each Gaussian only once.
        luminosity_matrix = np.concatenate([
            luminosity[shared_lines].T for luminosity in luminosity_by_dust.values()
        ], axis=1)
        channel_profiles = accumulate_velocity_spectra(
            velocity_kms=velocity_kms,
            thermal_width_kms=thermal_width,
            luminosity_matrix=luminosity_matrix,
            velocity_edges_kms=velocity_edges_kms,
            cell_chunk=cell_chunk,
            workers=workers,
        )
        group_size = len(shared_lines)
        for dust_index in range(2):
            first_column = dust_index * group_size
            last_column = first_column + group_size
            profiles[dust_index, shared_lines] = channel_profiles[:, first_column:last_column].T
    return profiles, moments


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
