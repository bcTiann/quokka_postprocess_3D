"""Streaming mass-weighted LOS velocity distributions of gas with a known temperature.

The batch interface uses DESPOTIC temperature where T_QUOKKA < 3000 K and
QUOKKA temperature otherwise. Missing line emissivities do not remove gas.
The low-level ``add`` interface accepts an already chosen phase temperature.
This module does not solve chemistry or calculate emissivity.
Velocities are not recentered, thermally broadened, or clipped to the plot window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quokka2s.physics.gas_fields import mixed_gas_temperature_K
from quokka2s.physics.settings import EMISSION_TEMPERATURE_BOUNDARY_K
from quokka2s.products.profile_preparation import add_gas_phase_display_fields


# These are the established cuts in the CNM/UNM/WNM/WIM/HIM phase definition.
# Keeping this numerical helper independent avoids loading the yt pipeline.
PHASE_ORDER = ("CNM", "UNM", "WNM", "WIM", "HIM")
PHASE_BOUNDS_K = (200.0, EMISSION_TEMPERATURE_BOUNDARY_K, 1.0e4, 10.0**5.5)
GROUP_ORDER = (*PHASE_ORDER, "total")
PHASE_MASS_RTOL = 1e-10


@dataclass(frozen=True)
class WeightedVelocityMoments:
    """Mergeable mass-weighted velocity moments for one population.

    Attributes
    ----------
    cell_count : int
        Number of cells.
    mass_g : float
        Sum of cell masses [g].
    mean_velocity_kms : float
        Mass-weighted LOS velocity [km/s].
    centered_second_moment : float
        Sum of mass * (velocity - mean)^2 [g * (km/s)^2].
    """
    cell_count: int = 0
    mass_g: float = 0.0
    mean_velocity_kms: float = 0.0
    centered_second_moment: float = 0.0

    @classmethod
    def from_arrays(cls, velocity, mass):
        """Summarize matching 1D velocity [km/s] and positive mass [g] arrays.

        Returns a WeightedVelocityMoments object, or the empty state for zero cells.
        Subtracting a velocity offset before summing preserves narrow dispersions
        when the gas has a large common bulk velocity.
        """
        if velocity.size == 0:
            return cls()
        # Shift before summing: E[v^2] - E[v]^2 loses the dispersion when
        # a narrow velocity distribution has a large common bulk offset.
        with np.errstate(over="ignore", invalid="ignore"):
            weight = float(np.sum(mass, dtype=np.float64))
            offset = velocity - velocity[0]
            mean_offset = float(np.sum(mass * offset, dtype=np.float64) / weight)
            mean = float(velocity[0] + mean_offset)
            centered_second = float(np.sum(
                mass * (offset - mean_offset)**2, dtype=np.float64))
        if not np.isfinite([weight, mean, centered_second]).all() or weight <= 0.0:
            raise ValueError("Mass or weighted velocity moments overflowed")
        return cls(
            cell_count=int(velocity.size),
            mass_g=weight,
            mean_velocity_kms=mean,
            centered_second_moment=centered_second,
        )

    def merged(self, other):
        """Return moments for both populations without changing either input.

        other is a WeightedVelocityMoments with the same velocity/mass units. The weighted
        parallel-variance formula retains the offset between the two group means.
        """
        if self.cell_count == 0:
            return other
        if other.cell_count == 0:
            return self
        weight = self.mass_g + other.mass_g
        delta = other.mean_velocity_kms - self.mean_velocity_kms
        fraction = other.mass_g / weight
        mean = self.mean_velocity_kms + delta * fraction
        # Parallel/streaming form of the weighted centered second moment.
        second = (self.centered_second_moment + other.centered_second_moment
                  + delta**2 * (self.mass_g * fraction))
        if not np.isfinite([weight, mean, second]).all():
            raise ValueError("Accumulated mass or weighted moments overflowed")
        return WeightedVelocityMoments(
            cell_count=self.cell_count + other.cell_count,
            mass_g=weight,
            mean_velocity_kms=mean,
            centered_second_moment=second,
        )

    def report(self, global_mean):
        """Return JSON-safe mass [g], velocity and two dispersions [km/s].

        global_mean is the full-range all-gas mean [km/s]. The internal dispersion
        is about this population's own mean; sigma_about_global_mean_kms uses
        global_mean instead. Empty populations return None for means/widths.
        """
        if not self.cell_count:
            return {
                "count": 0, "mass_g": 0.0, "mean_velocity_kms": None,
                "sigma_internal_kms": None,
                "sigma_about_global_mean_kms": None,
            }
        variance = self.centered_second_moment / self.mass_g
        return {
            "count": self.cell_count,
            "mass_g": self.mass_g,
            "mean_velocity_kms": self.mean_velocity_kms,
            "sigma_internal_kms": float(np.sqrt(variance)),
            "sigma_about_global_mean_kms": float(np.sqrt(
                variance + (self.mean_velocity_kms - global_mean)**2)),
        }


@dataclass(frozen=True)
class GasVelocityPopulation:
    """Keep full/window moments and a velocity-bin mass histogram.

    histogram_mass_g is (Nchannel,) [g]. full_range_moments includes all
    selected gas velocities; window_moments uses only the channel window. The
    below/above fields retain outside-window counts and masses [g].
    """
    full_range_moments: WeightedVelocityMoments
    window_moments: WeightedVelocityMoments
    histogram_mass_g: np.ndarray
    below_window_count: int = 0
    below_window_mass_g: float = 0.0
    above_window_count: int = 0
    above_window_mass_g: float = 0.0

    @classmethod
    def empty(cls, n_bins):
        """Return zero counts/moments and n_bins zero-filled mass bins [g]."""
        return cls(
            full_range_moments=WeightedVelocityMoments(),
            window_moments=WeightedVelocityMoments(),
            histogram_mass_g=np.zeros(n_bins),
        )

    @classmethod
    def from_arrays(cls, velocity, mass, edges):
        """Build group moments and a mass histogram from selected cells.

        Parameters
        ----------
        velocity, mass : numpy.ndarray, shape (R,)
            LOS velocities [km/s] and positive cell masses [g].
        edges : numpy.ndarray, shape (Nchannel + 1,)
            Increasing velocity edges [km/s]; both outer edges belong to the window.

        Returns
        -------
        GasVelocityPopulation
            Full/window moments, (Nchannel,) histogram [g], and outside-window
            counts/masses. Bins sum directly to preserve very low-mass contributions.
        """
        below = velocity < edges[0]
        above = velocity > edges[-1]
        inside = ~(below | above)
        # Match np.histogram's inclusive rightmost edge, without its generic
        # weighted cumulative-sum subtraction (which can lose low-mass bins
        # when the chunk spans many decades in mass). Accumulate each bin
        # directly instead. Values outside the window are never clamped.
        indices = np.searchsorted(edges, velocity[inside], side="right") - 1
        indices[indices == edges.size - 1] = edges.size - 2
        histogram = np.bincount(indices, weights=mass[inside], minlength=edges.size - 1)
        if not np.isfinite(histogram).all():
            raise ValueError("Mass histogram overflowed")
        return cls(
            full_range_moments=WeightedVelocityMoments.from_arrays(velocity, mass),
            window_moments=WeightedVelocityMoments.from_arrays(velocity[inside], mass[inside]),
            histogram_mass_g=histogram,
            below_window_count=int(np.count_nonzero(below)),
            below_window_mass_g=float(np.sum(mass[below])),
            above_window_count=int(np.count_nonzero(above)),
            above_window_mass_g=float(np.sum(mass[above])),
        )

    def merged(self, other):
        """Return the sum of two groups on the same channel grid.

        other is a GasVelocityPopulation; neither input changes. Moments merge stably and
        histograms [g] and outside-window counts/masses add element by element.
        """
        return GasVelocityPopulation(
            full_range_moments=self.full_range_moments.merged(other.full_range_moments),
            window_moments=self.window_moments.merged(other.window_moments),
            histogram_mass_g=self.histogram_mass_g + other.histogram_mass_g,
            below_window_count=self.below_window_count + other.below_window_count,
            below_window_mass_g=self.below_window_mass_g + other.below_window_mass_g,
            above_window_count=self.above_window_count + other.above_window_count,
            above_window_mass_g=self.above_window_mass_g + other.above_window_mass_g,
        )


def check_phase_accounting(
    groups: dict[str, dict],
    mass_by_bin: np.ndarray,
    gas_cell_count: int,
    gas_mass_g: float,
) -> None:
    """Check phase/window totals against gas with an available mixed temperature.

    ``groups`` is report()['groups']; ``mass_by_bin`` is (6, Nchannel) [g],
    in GROUP_ORDER. gas_cell_count and gas_mass_g [g] are independent batch
    sums over available mixed temperatures; line-emissivity masks are not used.
    Return None when the histograms, five phases, total and windows agree;
    raise ValueError on a mismatch. Inputs and accumulated state are unchanged.

    Example: check_phase_accounting(groups, mass_by_bin, gas_cell_count,
                                     gas_mass_g)
    """
    if not np.allclose(mass_by_bin[:-1].sum(axis=0), mass_by_bin[-1],
                       rtol=PHASE_MASS_RTOL, atol=0):
        raise ValueError('Phase histograms do not add to the all-gas histogram')
    if (sum(groups[key]['count'] for key in PHASE_ORDER) != gas_cell_count
            or groups['total']['count'] != gas_cell_count):
        raise ValueError('Phase cell counts differ from available-temperature gas cells')
    phase_masses = np.array([groups[key]['mass_g'] for key in PHASE_ORDER])
    all_mass = groups['total']['mass_g']
    if (not np.isclose(phase_masses.sum(), all_mass, rtol=PHASE_MASS_RTOL, atol=0)
            or not np.isclose(all_mass, gas_mass_g, rtol=PHASE_MASS_RTOL, atol=0)):
        raise ValueError('Phase masses differ from available-temperature gas mass')
    for key in GROUP_ORDER:
        group = groups[key]
        outside = group['below_window']['mass_g'] + group['above_window']['mass_g']
        window_mass = group['histogram_mass_g']
        if (group['in_window']['count'] + group['below_window']['count']
                + group['above_window']['count'] != group['count']):
            raise ValueError(f'{key} velocity-window counts are inconsistent')
        if not np.isclose(window_mass + outside, group['mass_g'],
                          rtol=PHASE_MASS_RTOL, atol=0):
            raise ValueError(f'{key} velocity-window mass is inconsistent')


class GasPhaseVelocityAccumulator:
    """Accumulate five gas phases and all gas using mass = density * volume.

    ``add`` takes finite, already-selected one-dimensional cell arrays; volume
    may be a positive scalar or a same-shaped array. Phase temperature must be
    positive, and is intentionally chosen by the caller. Thresholds are lower
    inclusive (e.g. 3000 K belongs to WNM). State scales with the bin count,
    while temporary arrays are bounded by the caller's input chunk.

    ``report`` returns JSON-safe full-range moments plus separate in-window
    moments and counts/masses below and above the histogram window. Both sets
    of ``sigma_about_global_mean_kms`` use the full-range all-gas mean. Empty
    populations have null means/widths, rather than an artificial zero width.
    """

    def __init__(self, velocity_edges_kms):
        """Create empty mass histograms and moments for five phases and all gas.

        Parameters
        ----------
        velocity_edges_kms : array-like, shape (Nchannel + 1,)
            Increasing channel edges [km/s]; copied into this accumulator.

        Notes
        -----
        Group order is CNM, UNM, WNM, WIM, HIM, total. Full-range moments also
        include velocities beyond these edges; the histogram includes both edges.

        Examples
        --------
        phases = GasPhaseVelocityAccumulator(np.linspace(-200., 200., 401))
        """
        edges = np.asarray(velocity_edges_kms, dtype=float)
        self.velocity_edges_kms = edges.copy()
        self.velocity_edges_kms.flags.writeable = False
        self.populations = {key: GasVelocityPopulation.empty(edges.size - 1) for key in GROUP_ORDER}

    @property
    def histogram_mass_g(self):
        """Copy each group's mass in the saved velocity channels.

        Returns
        -------
        dict of numpy.ndarray
            (Nchannel,) arrays [g], keyed in CNM, UNM, WNM, WIM, HIM, total order.
            These copies can be changed without modifying accumulated state.

        Examples
        --------
        cnm_mass_per_channel = phases.histogram_mass_g['CNM']
        """
        return {key: group.histogram_mass_g.copy() for key, group in self.populations.items()}

    def merge(self, other):
        """Merge another batch's histograms and stable mass-weighted moments.

        Parameters
        ----------
        other : GasPhaseVelocityAccumulator
            Independent batch with exactly the same velocity channel edges.

        Returns
        -------
        GasPhaseVelocityAccumulator
            This accumulator; all phase groups and the all-gas total are updated.

        Examples
        --------
        phases.merge(batch_phases)
        """
        if not np.array_equal(self.velocity_edges_kms, other.velocity_edges_kms):
            raise ValueError("Cannot merge phase histograms with different velocity bins")
        self.populations = {key: self.populations[key].merged(other.populations[key])
                        for key in GROUP_ORDER}
        return self

    def add(self, velocity_kms, phase_temperature_K, density_g_cm3, cell_volume_cm3):
        """Add already selected cells using the supplied phase temperature.

        Parameters
        ----------
        velocity_kms : array-like, shape (R,)
            Cell LOS velocities [km/s], without recentering or broadening.
        phase_temperature_K : array-like, shape (R,)
            Chosen phase temperatures [K]; add_batch() constructs the mixed values.
        density_g_cm3 : array-like, shape (R,)
            Gas mass densities [g/cm^3], from CellBatch.density_g_cm3.
        cell_volume_cm3 : float or array-like, shape (R,)
            Cell volumes [cm^3]; mass weights are density times volume.

        Returns
        -------
        GasPhaseVelocityAccumulator
            This accumulator. Phase boundaries 200, 3000, 1e4 and 10^5.5 K
            belong to the hotter phase. Failures leave accumulated state unchanged.

        Examples
        --------
        phases.add(velocity, phase_temperature, density, volume)
        """
        velocity = np.asarray(velocity_kms, dtype=float)
        temperature = np.asarray(phase_temperature_K, dtype=float)
        density = np.asarray(density_g_cm3, dtype=float)
        volume = np.asarray(cell_volume_cm3, dtype=float)
        if velocity.ndim != 1 or not np.isfinite(velocity).all():
            raise ValueError("velocity_kms must be finite with shape (cells,)")
        for name, value in (("phase_temperature_K", temperature),
                            ("density_g_cm3", density)):
            if (value.shape != velocity.shape or not np.isfinite(value).all()
                    or np.any(value <= 0.0)):
                raise ValueError(f"{name} must be finite and positive with shape (cells,)")
        if (volume.shape not in ((), velocity.shape) or not np.isfinite(volume).all()
                or np.any(volume <= 0.0)):
            raise ValueError("cell_volume_cm3 must be finite and positive, scalar or (cells,)")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            mass = density * volume
        if not np.isfinite(mass).all() or np.any(mass <= 0.0):
            raise ValueError("Cell mass density * volume must be finite and positive")
        phase_index = np.searchsorted(PHASE_BOUNDS_K, temperature, side="right")
        # Compute everything before replacing state, including all-gas moments
        # independently of the five phase populations for cross-checking.
        updated = {}
        for index, key in enumerate(GROUP_ORDER):
            if key == "total":
                group_velocity = velocity
                group_mass = mass
            else:
                selected = phase_index == index
                group_velocity = velocity[selected]
                group_mass = mass[selected]
            delta = GasVelocityPopulation.from_arrays(
                velocity=group_velocity,
                mass=group_mass,
                edges=self.velocity_edges_kms,
            )
            updated[key] = self.populations[key].merged(delta)
        self.populations = updated
        return self

    def add_batch(self, cells, emission):
        """Add gas using its mixed temperature, independently of line availability.

        Parameters
        ----------
        cells : CellBatch
            From SlabArrays.batch(); provides (B,) QUOKKA temperatures [K],
            z velocities [km/s], densities [g/cm^3] and a common volume [cm^3].
        emission : BatchEmission
            From CellEmissionCalculator.calculate(); cold_cells and DESPOTIC
            temperatures [K] are (B,). Only cold cells need a usable DESPOTIC
            temperature; hot cells use their own QUOKKA temperature.

        Returns
        -------
        GasPhaseVelocityAccumulator
            This accumulator, with unavailable mixed temperatures omitted.

        Examples
        --------
        A hot cell with missing DESPOTIC temperature still contributes gas mass:
            phases.add_batch(cells=cells, emission=emission)
        """
        phase_temperature, gas_cells = self.prepare_mixed_temperature_cells(
            cells=cells,
            emission=emission,
        )
        return self.add(
            velocity_kms=cells.velocity_z_kms[gas_cells],
            phase_temperature_K=phase_temperature[gas_cells],
            density_g_cm3=cells.density_g_cm3[gas_cells],
            cell_volume_cm3=cells.cell_volume_cm3,
        )

    def prepare_mixed_temperature_cells(self, *, cells, emission):
        """Return mixed temperatures [K] and their own available-cell selection.

        cells supplies T_QUOKKA; emission supplies cold_cells and T_DESPOTIC,
        each (B,) in original order. Returns two (B,) arrays: temperature and
        a bool selection. A missing cold T_DESPOTIC is omitted; a hot cell uses
        T_QUOKKA even when every line emissivity is missing.
        Example: T_Q=[100, 1e6], T_D=[NaN, NaN] gives [NaN, 1e6], [False, True].
        """
        phase_temperature = mixed_gas_temperature_K(
            cold_cells=emission.cold_cells,
            temperature_despotic_K=emission.despotic_temperature_K,
            temperature_quokka_K=cells.temperature_QUOKKA_K,
        )
        gas_cells = np.isfinite(phase_temperature) & (phase_temperature > 0.0)
        return phase_temperature, gas_cells

    def report(self):
        """Summarize each phase's full-range and window-restricted statistics.

        Returns
        -------
        dict
            groups maps CNM, UNM, WNM, WIM, HIM and total to counts, masses [g],
            fractions, mean velocities and dispersions [km/s]. Each group has
            in_window moments and below_window/above_window counts/masses.
            Both sigma_about_global_mean_kms values use the full-range all-gas
            mean. Empty groups have None means/widths; metadata records the cuts.

        Examples
        --------
        report = phases.report()
        cnm_sigma = report['groups']['CNM']['sigma_about_global_mean_kms']
        """
        total = self.populations["total"].full_range_moments
        global_mean = total.mean_velocity_kms if total.cell_count else None
        groups = {}
        for key, group in self.populations.items():
            result = group.full_range_moments.report(global_mean)
            result.update({
                "mass_fraction": group.full_range_moments.mass_g / total.mass_g if total.cell_count else None,
                "cell_fraction": group.full_range_moments.cell_count / total.cell_count if total.cell_count else None,
                "in_window": group.window_moments.report(global_mean),
                "below_window": {"count": group.below_window_count, "mass_g": group.below_window_mass_g},
                "above_window": {"count": group.above_window_count, "mass_g": group.above_window_mass_g},
                "histogram_mass_g": float(np.sum(group.histogram_mass_g)),
            })
            groups[key] = result
        return {
            "phase_order": list(PHASE_ORDER),
            "phase_boundaries_K": list(PHASE_BOUNDS_K),
            "boundary_rule": "lower inclusive; upper exclusive; HIM has no upper bound",
            "velocity_window_kms": [float(self.velocity_edges_kms[0]),
                                    float(self.velocity_edges_kms[-1])],
            "histogram_boundary_rule": "both outer edges included; no clipping",
            "global_mean_velocity_kms": global_mean,
            "sigma_reference": "gas with available mixed temperature over the full velocity range",
            "groups": groups,
        }

    def build_output(self):
        """Build phase histograms/moments without comparing independent cell totals.

        Returns
        -------
        payload : dict
            histogram_mass_g is (6, Nchannel) [g]; group_count, group_mass_g and
            both dispersion arrays are (6,), ordered CNM, UNM, WNM, WIM, HIM, total.
            Dispersions [km/s] use full-range cells; empty groups are NaN.
            Velocity centres, normalized profiles and comparison sigmas are
            included so figures only select and draw saved values.
        report : dict
            The detailed groups/window summary returned by report().

        Examples
        --------
        payload, report = phases.build_output()
        check_phase_accounting(report['groups'], payload['histogram_mass_g'],
                               gas_cell_count, gas_mass_g)
        """
        report = self.report()
        groups = report['groups']
        mass_by_bin = np.stack([
            self.populations[key].histogram_mass_g for key in GROUP_ORDER
        ])
        payload = {
            'schema_version': np.asarray(1),
            'phase_keys': np.asarray(GROUP_ORDER),
            'phase_boundaries_K': np.asarray(PHASE_BOUNDS_K),
            'velocity_edges_kms': self.velocity_edges_kms.copy(),
            'histogram_mass_g': mass_by_bin,
            'sigma_about_global_mean_kms': np.asarray([
                groups[key]['sigma_about_global_mean_kms']
                if groups[key]['count'] else np.nan for key in GROUP_ORDER]),
            'sigma_internal_kms': np.asarray([
                groups[key]['sigma_internal_kms']
                if groups[key]['count'] else np.nan for key in GROUP_ORDER]),
            'group_count': np.asarray([groups[key]['count'] for key in GROUP_ORDER]),
            'group_mass_g': np.asarray([groups[key]['mass_g'] for key in GROUP_ORDER]),
            'phase_temperature_method': np.asarray('mixed'),
            'bundle_source': np.asarray('process'),
        }
        add_gas_phase_display_fields(payload=payload)
        return payload, report
