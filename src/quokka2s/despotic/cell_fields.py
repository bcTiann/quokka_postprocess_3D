"""Prepare DESPOTIC queries, read fields, and restore original batch positions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quokka2s.despotic.lookup import DespoticLookup, DespoticTemperatureCO
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
        These are clipping flags, not failure flags.

    Returned by DespoticCellReader.prepare_query_inputs(). Example: hydrogen_density_cm3[7]
    is the eighth cell's lookup nH, also used when converting DESPOTIC lumPerH.
    """

    hydrogen_density_cm3: np.ndarray
    shielding_NH_cm2: np.ndarray
    velocity_gradient_s: np.ndarray
    coordinate_clipped: dict[str, np.ndarray]


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
        T_DESPOTIC [K], queried for every cell, independently of its T_QUOKKA branch.
    co10_luminosity_per_H, co21_luminosity_per_H : ndarray
        CO(1-0)/(2-1) luminosities [erg/s/H nucleus], queried for every cell.
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

    DespoticCellReader.read_fields() returns this object. Example: electron_density_cm3[7]
    describes the eighth original cell, even if only two cold cells were queried.
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

    def read_fields(
        self,
        *,
        cells: CellBatch,
        cold_cells: np.ndarray,
    ) -> DespoticCellFields:
        """Read T/CO for all cells and CII/e-/H+/H for eligible cold cells.

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
        # 1. Prepare nH, shielding NH and dV/dr; record permitted endpoint clipping.
        query_inputs = self.prepare_query_inputs(
            cells=cells,
        )

        # 2. Read temperature and both CO lines at the same coordinates for all cells.
        temperature_and_co = self.lookup.temperature_and_co(
            hydrogen_density_cm3=query_inputs.hydrogen_density_cm3,
            shielding_NH_cm2=query_inputs.shielding_NH_cm2,
            velocity_gradient_s=query_inputs.velocity_gradient_s,
        )

        # 3. Keep the caller's temperature branch; require usable T_D for cold queries.
        temperature_K = temperature_and_co.temperature_K
        invalid_temperature_cells = ~np.isfinite(temperature_K) | (temperature_K <= 0)
        cold_cells_to_query = cold_cells & ~invalid_temperature_cells

        # 4. Read CII and e-/H+/H densities only for those selected cold cells.
        cold_fields = self.read_cii_and_hydrogen_densities(
            query_inputs=query_inputs,
            cold_cells_to_query=cold_cells_to_query,
        )

        # 5. Return cold-only results to their original cell indices; leave all
        # unqueried positions as NaN. Temperature and CO already include every cell.
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
    ) -> DespoticQueryInputs:
        """Check and prepare lookup coordinates without changing physical cell arrays.

        cells comes from slab.batch(); cells.hydrogen_density_cm3 is physical nH
        [cm^-3], computed once from its rho and reused by both readers.
        Returns DespoticQueryInputs with nH [cm^-3], NH [cm^-2], dV/dr [s^-1] and
        clipping flags, each shape (B,). Example: a tiny endpoint rounding offset
        changes only the lookup coordinate and its clipping flag.
        """
        hydrogen_density_cm3 = cells.hydrogen_density_cm3
        self.check_coordinate_bounds(
            hydrogen_density_cm3=hydrogen_density_cm3,
            shielding_NH_cm2=cells.shielding_NH_cm2,
            velocity_gradient_s=cells.velocity_gradient_s,
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
        return DespoticQueryInputs(
            hydrogen_density_cm3=query_hydrogen_density_cm3,
            shielding_NH_cm2=query_shielding_NH_cm2,
            velocity_gradient_s=query_velocity_gradient_s,
            coordinate_clipped={
                'nH': hydrogen_density_cm3 != query_hydrogen_density_cm3,
                'NH': cells.shielding_NH_cm2 != query_shielding_NH_cm2,
                'dVdr': cells.velocity_gradient_s != query_velocity_gradient_s,
            },
        )

    def check_coordinate_bounds(
        self,
        *,
        hydrogen_density_cm3: np.ndarray,
        shielding_NH_cm2: np.ndarray,
        velocity_gradient_s: np.ndarray,
    ) -> None:
        """Reject materially uncovered coordinates before endpoint clipping.

        The three input arrays have shape (B,) and units cm^-3, cm^-2 and s^-1;
        their bounds come from self.lookup.table. Nonfinite/nonpositive values
        or relative endpoint offsets >1e-6 raise ValueError. Example: for an nH
        upper bound of 100, 100.00001 may be clipped, whereas 101 is rejected.
        """
        boundary_rtol = 1e-6
        coordinates = (hydrogen_density_cm3, shielding_NH_cm2, velocity_gradient_s)
        axes = (
            self.lookup.table.nH_values,
            self.lookup.table.col_density_values,
            self.lookup.table.dVdr_values,
        )
        for name, values, axis in zip(('nH', 'NH', 'dVdr'), coordinates, axes):
            invalid = ~np.isfinite(values) | (values <= 0)
            below = values < axis[0] * (1 - boundary_rtol)
            above = values > axis[-1] * (1 + boundary_rtol)
            if np.any(invalid | below | above):
                raise ValueError(f'DESPOTIC {name} outside the table domain')

    def read_cii_and_hydrogen_densities(
        self,
        *,
        query_inputs: DespoticQueryInputs,
        cold_cells_to_query: np.ndarray,
    ) -> DespoticColdFields:
        """Read CII luminosity per H and e-/H+/H densities for selected cold cells.

        query_inputs contains all B prepared coordinates. cold_cells_to_query is
        the (B,) selection from read_fields(). Returns four (Q,) arrays: CII
        [erg/s/H nucleus] and particle densities [cm^-3], using clipped query nH.
        Example: [True, False, True] reads cells 0 and 2; an empty selection returns
        empty arrays without accessing the table.
        """
        if not np.any(cold_cells_to_query):
            return DespoticColdFields(
                cii_luminosity_per_H=np.empty(0),
                electron_density_cm3=np.empty(0),
                ionized_hydrogen_density_cm3=np.empty(0),
                neutral_hydrogen_density_cm3=np.empty(0),
            )

        cold_hydrogen_density_cm3 = query_inputs.hydrogen_density_cm3[cold_cells_to_query]
        cold_shielding_NH_cm2 = query_inputs.shielding_NH_cm2[cold_cells_to_query]
        cold_velocity_gradient_s = query_inputs.velocity_gradient_s[cold_cells_to_query]
        cii_luminosity_per_H = self.lookup.line_field(
            species='C+',
            field_name='lumPerH',
            hydrogen_density_cm3=cold_hydrogen_density_cm3,
            shielding_NH_cm2=cold_shielding_NH_cm2,
            velocity_gradient_s=cold_velocity_gradient_s,
        )
        number_densities = self.lookup.number_densities(
            species=('e-', 'H+', 'H'),
            hydrogen_density_cm3=cold_hydrogen_density_cm3,
            shielding_NH_cm2=cold_shielding_NH_cm2,
            velocity_gradient_s=cold_velocity_gradient_s,
        )
        return DespoticColdFields(
            cii_luminosity_per_H=cii_luminosity_per_H,
            electron_density_cm3=number_densities['e-'],
            ionized_hydrogen_density_cm3=number_densities['H+'],
            neutral_hydrogen_density_cm3=number_densities['H'],
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
        for all cells. CII luminosity per H and electron, ionized-hydrogen and
        neutral-hydrogen densities are obtained only for cold cells with a
        usable DESPOTIC temperature.

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
        )
