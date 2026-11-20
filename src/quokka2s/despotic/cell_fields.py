"""Prepare DESPOTIC queries, read fields, and restore original batch positions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from quokka2s.despotic.lookup import DespoticTemperatureCO


# Relative endpoint tolerance shared by processing and table-range diagnostics.
COORDINATE_BOUNDARY_RTOL = 1e-6

if TYPE_CHECKING:
    from quokka2s.despotic.lookup import DespoticLookup, DespoticQueries
    from quokka2s.snapshot_reader import CellBatch


@dataclass(frozen=True)
class DespoticQueryInputs:
    """DESPOTIC lookup coordinates for all B original batch cells.

    Attributes
    ----------
    hydrogen_density_cm3, shielding_NH_cm2, velocity_gradient_s : ndarray, shape (B,)
        Prepared physical coordinates [cm^-3], [cm^-2], [s^-1], after the
        permitted endpoint clipping. Original physical cell arrays are unchanged.
    coordinate_clipped : dict of str to ndarray of bool, shape (B,)
        'nH', 'NH' and 'dVdr' flags recording which coordinate entries changed.
        Skipped cells have no clipping flags.
    below_hydrogen_density_domain : ndarray of bool, shape (B,), or None
        Cells skipped by the low-density emission rule. These have NaN query
        nH. None means every cell is queried.

    Returned by DespoticCellReader.prepare_query_inputs(). Example: hydrogen_density_cm3[7]
    is the eighth cell's lookup nH, also used when converting DESPOTIC lumPerH.
    """

    hydrogen_density_cm3: np.ndarray
    shielding_NH_cm2: np.ndarray
    velocity_gradient_s: np.ndarray
    coordinate_clipped: dict[str, np.ndarray]
    below_hydrogen_density_domain: np.ndarray | None


@dataclass(frozen=True)
class DespoticColdFields:
    """DESPOTIC fields queried only for Q eligible cold cells.

    Attributes
    ----------
    cii_luminosity_per_H : ndarray, shape (Q,)
        CII luminosity [erg/s/H nucleus] from the C+ line's lumPerH field.
    electron_density_cm3, ionized_hydrogen_density_cm3, neutral_hydrogen_density_cm3 : ndarray, shape (Q,)
        Electron, H+ and neutral H number densities [cm^-3], from the corresponding
        abundance multiplied by the clipped query hydrogen density.

    DespoticCellReader.read_cii_and_hydrogen_densities() returns these in selected order.
    For cold_cells_to_query=[True, False, True], entries 0 and 1 describe original
    batch cells 0 and 2; the reader restores them to those original positions.
    """

    cii_luminosity_per_H: np.ndarray
    electron_density_cm3: np.ndarray
    ionized_hydrogen_density_cm3: np.ndarray
    neutral_hydrogen_density_cm3: np.ndarray


@dataclass(frozen=True)
class DespoticCellFields:
    """DESPOTIC lookup results in original batch order, each array shape (B,).

    Attributes
    ----------
    temperature_K : ndarray
        T_DESPOTIC [K], queried for covered cells at all T_QUOKKA.
    co10_luminosity_per_H, co21_luminosity_per_H : ndarray
        CO(1-0)/(2-1) luminosities [erg/s/H nucleus], queried for covered cells.
    cii_luminosity_per_H : ndarray
        CII luminosity [erg/s/H nucleus], queried only for eligible cold cells.
    electron_density_cm3, ionized_hydrogen_density_cm3, neutral_hydrogen_density_cm3 : ndarray
        Number densities [cm^-3], queried only for eligible cold cells. Their
        unqueried positions, like unqueried CII entries, remain NaN.
    query_hydrogen_density_cm3 : ndarray
        Clipped nH [cm^-3] used to convert DESPOTIC luminosities per H to emissivity.
    invalid_temperature_cells : ndarray of bool
        True for nonfinite or nonpositive T_DESPOTIC. This flag checks temperature
        only; other invalid fields with usable temperature are rejected later.
    coordinate_clipped : dict of str to ndarray of bool
        'nH', 'NH', 'dVdr' flags from the reader's coordinate preparation, each (B,).
    below_hydrogen_density_domain : ndarray of bool, shape (B,), or None
        Skipped low-density cells for which only the CO emissivity is prescribed
        as zero. None means every cell was queried.

    DespoticCellReader.read_fields() returns this object. Every returned field
    keeps the original batch shape (B,) and cell order; unqueried values are NaN.
    """

    temperature_K: np.ndarray
    co10_luminosity_per_H: np.ndarray
    co21_luminosity_per_H: np.ndarray
    cii_luminosity_per_H: np.ndarray
    electron_density_cm3: np.ndarray
    ionized_hydrogen_density_cm3: np.ndarray
    neutral_hydrogen_density_cm3: np.ndarray
    query_hydrogen_density_cm3: np.ndarray
    invalid_temperature_cells: np.ndarray
    coordinate_clipped: dict[str, np.ndarray]
    below_hydrogen_density_domain: np.ndarray | None = None



class DespoticCellReader:
    """Read the DESPOTIC fields needed by each original cell batch.

    Parameters
    ----------
    lookup : DespoticLookup
        Loaded filled-table lookup from load_despotic_lookup(). Stored as
        self.lookup and shared across batches; no batch data is retained.

    Examples
    --------
    reader = DespoticCellReader(lookup=lookup)
    fields = reader.read_fields(cells=cells, cold_cells=cold)
    fields.temperature_K[7] is the eighth original cell's DESPOTIC temperature.
    """

    def __init__(self, *, lookup: DespoticLookup):
        self.lookup = lookup

    def read_temperature(self, *, cells: CellBatch) -> np.ndarray:
        """Read only T_DESPOTIC [K] for every cell in the supplied batch.

        cells comes from slab.batch(); its physical nH, shielding NH and
        velocity gradient each have shape (B,). This action requires coverage
        of all coordinates within the permitted endpoint tolerance, then
        queries the existing temperature interpolator without reading CO,
        CII or particle densities. Returned values keep shape (B,) and the
        original cell order; missing temperature values remain NaN.
        Example: temperature_K[7] is original batch cell 7's T_DESPOTIC.
        """
        _query_inputs, queries = self.prepare_cell_queries(cells=cells)
        return self.lookup.temperature(queries=queries)

    def prepare_cell_queries(
        self,
        *,
        cells: CellBatch,
        skip_below_density_domain: bool = False,
    ) -> tuple[DespoticQueryInputs, DespoticQueries | None]:
        """Check/clip cell coordinates and prepare their reusable logarithmic rows.

        Returns full-batch inputs and queries in covered-cell order. With the
        low-density emission rule enabled, an entirely skipped batch has no queries.
        """
        query_inputs = self.prepare_query_inputs(
            cells=cells,
            skip_below_density_domain=skip_below_density_domain,
        )
        coordinates = (
            query_inputs.hydrogen_density_cm3,
            query_inputs.shielding_NH_cm2,
            query_inputs.velocity_gradient_s,
        )
        if query_inputs.below_hydrogen_density_domain is not None:
            queried_cells = ~query_inputs.below_hydrogen_density_domain
            if not np.any(queried_cells):
                return query_inputs, None
            coordinates = tuple(values[queried_cells] for values in coordinates)
        density, column, gradient = coordinates
        queries = self.lookup.prepare_queries(
            hydrogen_density_cm3=density,
            shielding_NH_cm2=column,
            velocity_gradient_s=gradient,
        )
        return query_inputs, queries

    def read_fields(
        self,
        *,
        cells: CellBatch,
        cold_cells: np.ndarray,
    ) -> DespoticCellFields:
        """Read T/CO for covered cells and CII/e-/H+/H for eligible cold cells.

        Parameters
        ----------
        cells : CellBatch
            From slab.batch(); physical nH [cm^-3], shielding NH [cm^-2] and
            gradient [s^-1] each have shape (B,). nH is cached on cells.
        cold_cells : ndarray of bool, shape (B,)
            Caller-selected T_QUOKKA < 3000 K branch, including unavailable cells.

        Returns
        -------
        DespoticCellFields
            Returned physical fields keep the input cell-array shape and order.
            Unqueried or missing values are NaN.
        """
        query_inputs, queries = self.prepare_cell_queries(
            cells=cells,
            skip_below_density_domain=True,
        )

        # Read temperature and CO on the covered subset at any QUOKKA temperature.
        if queries is None:
            empty = np.empty(0)
            temperature_and_co = DespoticTemperatureCO(
                temperature_K=empty,
                co10_luminosity_per_H=empty,
                co21_luminosity_per_H=empty,
            )
        else:
            temperature_and_co = self.lookup.temperature_and_co(queries=queries)

        queried_cells = None
        queried_cold_cells = cold_cells
        if query_inputs.below_hydrogen_density_domain is not None:
            queried_cells = ~query_inputs.below_hydrogen_density_domain
            queried_cold_cells = cold_cells[queried_cells]
        temperature_K = temperature_and_co.temperature_K
        invalid_temperature_cells = ~np.isfinite(temperature_K) | (temperature_K <= 0)
        cold_cells_to_query = queried_cold_cells & ~invalid_temperature_cells

        cold_fields = self.read_cii_and_hydrogen_densities(
            queries=queries,
            cold_cells_to_query=cold_cells_to_query,
        )

        if queried_cells is not None:
            # Restore table data to original rows, leaving skipped states unknown.
            def restore_values(values):
                complete = np.full(cold_cells.shape, np.nan)
                complete[queried_cells] = values
                return complete

            temperature_and_co = DespoticTemperatureCO(
                temperature_K=restore_values(temperature_K),
                co10_luminosity_per_H=restore_values(temperature_and_co.co10_luminosity_per_H),
                co21_luminosity_per_H=restore_values(temperature_and_co.co21_luminosity_per_H),
            )
            batch_invalid_temperature = np.ones(cold_cells.shape, dtype=bool)
            batch_invalid_temperature[queried_cells] = invalid_temperature_cells
            invalid_temperature_cells = batch_invalid_temperature
            batch_cold_cells_to_query = np.zeros(cold_cells.shape, dtype=bool)
            batch_cold_cells_to_query[queried_cells] = cold_cells_to_query
            cold_cells_to_query = batch_cold_cells_to_query

        return self.restore_batch_positions(
            query_inputs=query_inputs,
            temperature_and_co=temperature_and_co,
            cold_fields=cold_fields,
            cold_cells_to_query=cold_cells_to_query,
            invalid_temperature_cells=invalid_temperature_cells,
        )

    def prepare_query_inputs(
        self,
        *,
        cells: CellBatch,
        skip_below_density_domain: bool = False,
    ) -> DespoticQueryInputs:
        """Check and prepare lookup coordinates without changing physical cell arrays.

        cells comes from slab.batch(); cells.hydrogen_density_cm3 is physical nH
        [cm^-3], computed once from its rho and reused by both readers.
        Returns DespoticQueryInputs with nH [cm^-3], NH [cm^-2], dV/dr [s^-1] and
        clipping flags, each shape (B,). Example: a tiny endpoint rounding offset
        changes only the lookup coordinate and its clipping flag.
        """
        hydrogen_density_cm3 = cells.hydrogen_density_cm3
        below_hydrogen_density_domain = self.check_coordinate_bounds(
            hydrogen_density_cm3=hydrogen_density_cm3,
            shielding_NH_cm2=cells.shielding_NH_cm2,
            velocity_gradient_s=cells.velocity_gradient_s,
            skip_below_density_domain=skip_below_density_domain,
        )
        (
            query_hydrogen_density_cm3,
            query_shielding_NH_cm2,
            query_velocity_gradient_s,
        ) = self.lookup.clip_coordinates(
            hydrogen_density_cm3=hydrogen_density_cm3,
            shielding_NH_cm2=cells.shielding_NH_cm2,
            velocity_gradient_s=cells.velocity_gradient_s,
        )
        coordinate_clipped = {
            'nH': hydrogen_density_cm3 != query_hydrogen_density_cm3,
            'NH': cells.shielding_NH_cm2 != query_shielding_NH_cm2,
            'dVdr': cells.velocity_gradient_s != query_velocity_gradient_s,
        }
        if below_hydrogen_density_domain is not None:
            query_hydrogen_density_cm3 = np.where(
                below_hydrogen_density_domain, np.nan, query_hydrogen_density_cm3,
            )
            for clipped_cells in coordinate_clipped.values():
                clipped_cells[below_hydrogen_density_domain] = False
        return DespoticQueryInputs(
            hydrogen_density_cm3=query_hydrogen_density_cm3,
            shielding_NH_cm2=query_shielding_NH_cm2,
            velocity_gradient_s=query_velocity_gradient_s,
            coordinate_clipped=coordinate_clipped,
            below_hydrogen_density_domain=below_hydrogen_density_domain,
        )

    def check_coordinate_bounds(
        self,
        *,
        hydrogen_density_cm3: np.ndarray,
        shielding_NH_cm2: np.ndarray,
        velocity_gradient_s: np.ndarray,
        skip_below_density_domain: bool = False,
    ) -> np.ndarray | None:
        """Reject materially uncovered coordinates before endpoint clipping.

        The three input arrays have shape (B,) and units cm^-3, cm^-2 and s^-1;
        their bounds come from self.lookup.table. Nonfinite/nonpositive values
        or relative endpoint offsets >1e-6 raise ValueError. With the emission
        rule enabled, positive nH below the allowed range instead returns a
        mask of those cells. All other checks still apply. Returns None when
        every cell is covered within tolerance.
        """
        boundary_rtol = COORDINATE_BOUNDARY_RTOL
        coordinates = (hydrogen_density_cm3, shielding_NH_cm2, velocity_gradient_s)
        axes = (
            self.lookup.table.nH_values,
            self.lookup.table.col_density_values,
            self.lookup.table.dVdr_values,
        )
        below_hydrogen_density_domain = None
        for name, values, axis in zip(('nH', 'NH', 'dVdr'), coordinates, axes):
            if np.size(values) == 0:
                continue
            # Extrema enforce the same per-cell bounds without three full masks.
            minimum = np.min(values)
            maximum = np.max(values)
            if (
                not np.isfinite(minimum)
                or not np.isfinite(maximum)
                or minimum <= 0
                or maximum > axis[-1] * (1 + boundary_rtol)
            ):
                raise ValueError(f'DESPOTIC {name} outside the table domain')
            lower_bound = axis[0] * (1 - boundary_rtol)
            if minimum < lower_bound:
                if name != 'nH' or not skip_below_density_domain:
                    raise ValueError(f'DESPOTIC {name} outside the table domain')
                below_hydrogen_density_domain = values < lower_bound
        return below_hydrogen_density_domain

    def read_cii_and_hydrogen_densities(
        self,
        *,
        queries: DespoticQueries | None,
        cold_cells_to_query: np.ndarray,
    ) -> DespoticColdFields:
        """Read CII luminosity per H and e-/H+/H densities for selected cold cells.

        queries contains the checked coordinates in covered-cell order.
        cold_cells_to_query selects cold cells in that order.
        Returns four (Q,) arrays: CII
        [erg/s/H nucleus] and particle densities [cm^-3], using clipped query nH.
        Example: [True, False, True] reads cells 0 and 2; an empty selection returns
        empty arrays without accessing the table.
        """
        if queries is None or not np.any(cold_cells_to_query):
            return DespoticColdFields(
                cii_luminosity_per_H=np.empty(0),
                electron_density_cm3=np.empty(0),
                ionized_hydrogen_density_cm3=np.empty(0),
                neutral_hydrogen_density_cm3=np.empty(0),
            )

        cold_queries = queries.select(selected_cells=cold_cells_to_query)
        cii_and_hydrogen = self.lookup.cii_and_hydrogen_abundances(
            queries=cold_queries,
        )
        hydrogen_density_cm3 = cold_queries.hydrogen_density_cm3.reshape(
            cold_queries.query_shape,
        )
        return DespoticColdFields(
            cii_luminosity_per_H=cii_and_hydrogen[..., 0],
            electron_density_cm3=hydrogen_density_cm3 * cii_and_hydrogen[..., 1],
            ionized_hydrogen_density_cm3=hydrogen_density_cm3 * cii_and_hydrogen[..., 2],
            neutral_hydrogen_density_cm3=hydrogen_density_cm3 * cii_and_hydrogen[..., 3],
        )

    def restore_batch_positions(
        self,
        *,
        query_inputs: DespoticQueryInputs,
        temperature_and_co: DespoticTemperatureCO,
        cold_fields: DespoticColdFields,
        cold_cells_to_query: np.ndarray,
        invalid_temperature_cells: np.ndarray,
    ) -> DespoticCellFields:
        """Combine the queried DESPOTIC fields into full-batch arrays.

        The earlier queries provide temperature and both CO luminosities per H
        in original batch order. CII luminosity per H and the three particle
        densities are obtained only for cold cells with usable DESPOTIC temperature.

        For each cold-only field, create an array with the input cell-array
        shape, initially filled with NaN. Write the queried values into the
        positions where cold_cells_to_query is True; other positions stay NaN.
        Temperature and CO already have the full shape and are used directly.

        Returns DespoticCellFields. All returned physical fields have the same
        shape and cell order as the input cell arrays.
        """
        batch_shape = cold_cells_to_query.shape
        cii_luminosity_per_H = np.full(batch_shape, np.nan)
        electron_density_cm3 = np.full(batch_shape, np.nan)
        ionized_hydrogen_density_cm3 = np.full(batch_shape, np.nan)
        neutral_hydrogen_density_cm3 = np.full(batch_shape, np.nan)

        cii_luminosity_per_H[cold_cells_to_query] = cold_fields.cii_luminosity_per_H
        electron_density_cm3[cold_cells_to_query] = cold_fields.electron_density_cm3
        ionized_hydrogen_density_cm3[cold_cells_to_query] = cold_fields.ionized_hydrogen_density_cm3
        neutral_hydrogen_density_cm3[cold_cells_to_query] = cold_fields.neutral_hydrogen_density_cm3

        return DespoticCellFields(
            temperature_K=temperature_and_co.temperature_K,
            co10_luminosity_per_H=temperature_and_co.co10_luminosity_per_H,
            co21_luminosity_per_H=temperature_and_co.co21_luminosity_per_H,
            cii_luminosity_per_H=cii_luminosity_per_H,
            electron_density_cm3=electron_density_cm3,
            ionized_hydrogen_density_cm3=ionized_hydrogen_density_cm3,
            neutral_hydrogen_density_cm3=neutral_hydrogen_density_cm3,
            query_hydrogen_density_cm3=query_inputs.hydrogen_density_cm3,
            invalid_temperature_cells=invalid_temperature_cells,
            coordinate_clipped=query_inputs.coordinate_clipped,
            below_hydrogen_density_domain=query_inputs.below_hydrogen_density_domain,
        )
