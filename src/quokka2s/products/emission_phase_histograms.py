"""Accumulate gas-mass and line-luminosity histograms in density and temperature."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quokka2s.physics.gas_fields import mixed_gas_temperature_K


PANELS = (
    ('mass_T_QK', 'QUOKKA', 'mass'),
    ('mass_T_DSP', 'DESPOTIC', 'mass'),
    ('mass_T_2R', 'mixed', 'mass'),
    ('NH_rho', None, 'NH_rho'),
    ('halpha', 'mixed', 'halpha'),
    ('hi21', 'mixed', 'hi21'),
    ('cii', 'mixed', 'cii'),
    ('co10', 'DESPOTIC', 'co10'),
    ('co21', 'DESPOTIC', 'co21'),
    ('ciii_977', 'QUOKKA', 'ciii_977'),
    ('ciii_1907', 'QUOKKA', 'ciii_1907'),
    ('ciii_1909', 'QUOKKA', 'ciii_1909'),
    ('civ_1548', 'QUOKKA', 'civ_1548'),
    ('civ_1551', 'QUOKKA', 'civ_1551'),
)



@dataclass(frozen=True)
class PhasePanelValues:
    """One phase panel's log-coordinate inputs and physical bin weights.

    x, y and weights are equal-length arrays of selected cells. x/y are still
    physical values here; accumulate_emission_phase_histograms() takes log10
    when adding them to the histogram. Weights are gas mass [g] or light [erg/s].
    """

    panel_key: str
    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray


def accumulate_emission_phase_histograms(
    histograms,
    rho,
    tq,
    td,
    column,
    emission,
    volume,
):
    """Add gas mass and line luminosity to density-temperature phase panels.

    Parameters
    ----------
    histograms : dict of DexHistogram
        Accumulators keyed by PANELS, e.g. histograms["halpha"].
    rho, tq, td, column : array-like, shape (B,)
        Original batch density [g/cm^3], QUOKKA/DESPOTIC temperatures [K],
        and shielding hydrogen column [cm^-2].
    emission : BatchEmission
        From CellEmissionCalculator.calculate(). Each named line contains
        intrinsic emissivity [erg/s/cm^3] and its emitting-state temperature_K [K], both
        (B,). Missing results in one line do not remove another line's cells.
    volume : float or array-like, shape (B,)
        Original cell volume [cm^3]; weights are density/epsilon times volume.

    Returns
    -------
    None
        Updates histograms; physical weights are not peak-normalized here.

    Examples
    --------
    emission.lines["halpha"].intrinsic_emissivity_erg_s_cm3[7] supplies cell 7's
    Halpha weight before multiplication by cell volume.
    """
    missing = {key for key, _, _ in PANELS} - histograms.keys()
    if missing:
        raise ValueError(f"Missing phase histograms: {sorted(missing)}")
    rows = prepare_phase_panel_values(
        rho=rho,
        tq=tq,
        td=td,
        column=column,
        emission=emission,
        volume=volume,
    )
    # Check every weight before changing any histogram, including overflows
    # in the conversion from density/epsilon to mass/luminosity.
    if any(not np.isfinite(row.weights).all() for row in rows):
        raise ValueError("Nonfinite phase mass or luminosity")
    for row in rows:
        if row.weights.size:
            histograms[row.panel_key].add(
                x=np.log10(row.x),
                y=np.log10(row.y),
                weight=row.weights,
            )


def prepare_phase_panel_values(rho, tq, td, column, emission, volume):
    """Prepare each panel using only the quantities that panel requires.

    Parameters
    ----------
    rho, tq, td, column : ndarray, shape (B,)
        Original batch density [g/cm^3], QUOKKA/DESPOTIC temperature [K], and
        shielding column [cm^-2].
    emission : BatchEmission
        Named LineEmission results and cold_cells from the cell calculator.
        Each line's fields and cold_cells have shape (B,) in original order.
    volume : float or ndarray, shape (B,)
        Cell volume [cm^3].

    Returns
    -------
    list of PhasePanelValues
        Selected coordinates and mass [g] or luminosity [erg/s]. Raw QUOKKA/NH
        panels use all cells; DESPOTIC and mixed panels require their own
        temperatures. Each line requires its own emissivity and temperature.
        Example: a hot cell with missing T_D remains in mixed-temperature mass.
    """
    rho = np.asarray(rho, dtype=float)
    tq = np.asarray(tq, dtype=float)
    td = np.asarray(td, dtype=float)
    column = np.asarray(column, dtype=float)
    shape = rho.shape
    if any(value.shape != shape for value in (tq, td, column)):
        raise ValueError("Phase coordinate shapes must match")
    cold_cells = np.asarray(emission.cold_cells)
    if cold_cells.dtype.kind != "b" or cold_cells.shape != shape:
        raise ValueError("Emission cold_cells must be boolean with the cell shape")
    volume = np.asarray(volume, dtype=float)
    if volume.shape not in ((), shape):
        raise ValueError("Volume must be scalar or match the cell shape")
    for name, value in (("density", rho), ("QUOKKA temperature", tq),
                        ("column", column), ("volume", volume)):
        if not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError(f"Invalid {name}: expected finite positive values")
    rows = prepare_gas_mass_panel_values(
        density_g_cm3=rho,
        temperature_QUOKKA_K=tq,
        temperature_DESPOTIC_K=td,
        shielding_NH_cm2=column,
        cold_cells=cold_cells,
        cell_volume_cm3=volume,
    )
    rows.extend(prepare_line_luminosity_panel_values(
        density_g_cm3=rho,
        line_emissions=emission.lines,
        cell_volume_cm3=volume,
    ))
    return rows


def prepare_gas_mass_panel_values(
    *,
    density_g_cm3,
    temperature_QUOKKA_K,
    temperature_DESPOTIC_K,
    shielding_NH_cm2,
    cold_cells,
    cell_volume_cm3,
):
    """Return four mass panels with independent temperature availability.

    Density, temperatures, NH and cold_cells are (B,) arrays from
    prepare_phase_panel_values(); volume is scalar or (B,) [cm^3]. The result
    contains density-temperature and NH-density coordinates with mass [g].
    A hot cell with missing T_D belongs to raw and mixed mass, but not T_D mass.
    """
    rho = density_g_cm3
    tq = temperature_QUOKKA_K
    td = temperature_DESPOTIC_K
    column = shielding_NH_cm2
    volume = cell_volume_cm3
    mass = rho * volume
    despotic_temperature_cells = np.isfinite(td) & (td > 0.0)
    mixed_temperature_K = mixed_gas_temperature_K(
        cold_cells=cold_cells,
        temperature_despotic_K=td,
        temperature_quokka_K=tq,
    )
    mixed_temperature_cells = np.isfinite(mixed_temperature_K) & (mixed_temperature_K > 0.0)
    return [
        PhasePanelValues(
            panel_key="mass_T_QK",
            x=rho,
            y=tq,
            weights=mass,
        ),
        PhasePanelValues(
            panel_key="mass_T_DSP",
            x=rho[despotic_temperature_cells],
            y=td[despotic_temperature_cells],
            weights=mass[despotic_temperature_cells],
        ),
        PhasePanelValues(
            panel_key="mass_T_2R",
            x=rho[mixed_temperature_cells],
            y=mixed_temperature_K[mixed_temperature_cells],
            weights=mass[mixed_temperature_cells],
        ),
        PhasePanelValues(
            panel_key="NH_rho",
            x=column,
            y=rho,
            weights=mass,
        ),
    ]


def prepare_line_luminosity_panel_values(*, density_g_cm3, line_emissions, cell_volume_cm3):
    """Return each line's available density-temperature points and luminosities.

    density_g_cm3 is (B,) [g/cm^3]; line_emissions maps names to LineEmission
    from the calculator; volume is scalar or (B,) [cm^3]. Returned panels use
    epsilon * volume [erg/s], with their own epsilon and thermal temperature.
    Example: missing CO cannot remove a usable hot Halpha point.
    """
    shape = density_g_cm3.shape
    volume_by_cell = np.broadcast_to(cell_volume_cm3, shape)
    lines = tuple(key for key, _, _ in PANELS if not key.startswith("mass") and key != "NH_rho")
    if not set(lines).issubset(line_emissions):
        raise ValueError("Emission must contain each plotted line")
    rows = []
    for line_key in lines:
        line_emission = line_emissions[line_key]
        epsilon = np.asarray(line_emission.intrinsic_emissivity_erg_s_cm3, dtype=float)
        # Preserve float64 log coordinates near dex-bin boundaries.
        line_temperature_K = np.asarray(
            line_emission.temperature_K,
            dtype=float,
        )
        if epsilon.shape != shape or line_temperature_K.shape != shape:
            raise ValueError(f"{line_key} emission and temperature must match the cell shape")
        line_cells = (
            ~line_emission.emissivity_is_missing
            & np.isfinite(line_temperature_K)
            & (line_temperature_K > 0.0)
        )
        if not np.isfinite(epsilon[line_cells]).all() or np.any(epsilon[line_cells] < 0.0):
            raise ValueError(f"Invalid available {line_key} emissivity")
        rows.append(PhasePanelValues(
            panel_key=line_key,
            x=density_g_cm3[line_cells],
            y=line_temperature_K[line_cells],
            weights=epsilon[line_cells] * volume_by_cell[line_cells],
        ))
    return rows


class DexHistogram:
    """Streaming absolute sums in globally aligned, fixed-width dex bins.

    Grow the small histogram as new extrema arrive, not the cell arrays.
    Bins are left-closed/right-open, including exact dex-boundary values.
    """
    def __init__(self, step=0.2):
        """Start an empty 2D histogram with a fixed bin width step [dex].

        Example: step=0.2 places bins at ..., -0.2, 0.0, 0.2, ... on each axis.
        The histogram expands only when the next batch extends its coordinate range.
        """
        self.step = float(step)
        if self.step <= 0:
            raise ValueError('Bin width must be positive')
        self.H = None
        self.origin = None
        self.total = 0.0
        self.count = 0

    def add(self, x, y, weight):
        """Sum physical weights into aligned two-dimensional log-coordinate bins.

        Parameters
        ----------
        x, y, weight : array-like, matching or broadcastable shapes
            log10 coordinates and physical weights [g or erg/s]. These come
            from one prepared PhasePanelValues; no normalization is applied.

        Examples
        --------
        histogram.add(x=np.log10(density), y=np.log10(temperature), weight=mass)
        """
        x, y, weight = np.broadcast_arrays(x, y, weight)
        if not (np.isfinite(x).all() and np.isfinite(y).all()
                and np.isfinite(weight).all()) or np.any(weight < 0):
            raise ValueError("Nonfinite coordinates/weights or negative weights")
        x_bin_indices = np.floor(x.ravel() / self.step).astype(np.int64)
        y_bin_indices = np.floor(y.ravel() / self.step).astype(np.int64)
        grown, origin = self.grow_histogram_for_bins(x_bin_indices, y_bin_indices)
        flat_bin_indices = (
            (x_bin_indices - origin[0]) * grown.shape[1]
            + (y_bin_indices - origin[1])
        )
        grown += np.bincount(
            flat_bin_indices,
            weights=weight.ravel(),
            minlength=int(np.prod(grown.shape)),
        ).reshape(grown.shape)
        self.H = grown
        self.origin = origin
        self.total += float(np.sum(weight, dtype=np.float64))
        self.count += weight.size

    def grow_histogram_for_bins(self, x_bin_indices, y_bin_indices):
        """Return an expanded histogram and its first global (x, y) bin indices.

        The existing sums are copied to their unchanged physical bins. For
        example, origin (-3, 2) with step 0.2 starts at x=-0.6 and y=0.4 dex.
        Neither the old histogram nor its origin changes until add() commits.
        """
        lower = np.array([x_bin_indices.min(), y_bin_indices.min()])
        upper = np.array([x_bin_indices.max(), y_bin_indices.max()]) + 1
        if self.H is not None:
            lower = np.minimum(lower, self.origin)
            upper = np.maximum(upper, self.origin + self.H.shape)
        shape = tuple(upper - lower)
        grown = np.zeros(shape)
        if self.H is not None:
            offset = self.origin - lower
            grown[offset[0]:offset[0] + self.H.shape[0],
                  offset[1]:offset[1] + self.H.shape[1]] = self.H
        return grown, lower

    def result(self):
        """Return physical bin sums H and their log10 coordinate edges.

        H has shape (Nx_bin, Ny_bin); x_edges/y_edges each have one extra entry.
        Units of H match the weights passed to add(): gas mass [g] or light [erg/s].
        """
        if self.H is None:
            raise ValueError('Empty histogram')
        return dict(H=self.H,
                    x_edges=(self.origin[0] + np.arange(self.H.shape[0] + 1)) * self.step,
                    y_edges=(self.origin[1] + np.arange(self.H.shape[1] + 1)) * self.step)
