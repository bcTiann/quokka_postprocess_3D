"""Read the current 3D Cloudy table and restore results to batch cell positions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quokka2s.cloudy.lookup import CloudyLookup, CloudyQueryResult
    from quokka2s.snapshot_reader import CellBatch


@dataclass(frozen=True)
class CloudyCellFields:
    """Cloudy lookup results restored to every original position in the batch.

    Attributes
    ----------
    emissivity_per_nH2 : dict of str to ndarray, each shape (B,)
        Named values [erg cm^3/s] from the actual lookup.line_keys. Arrays are
        row views of one restored matrix. Unqueried or failed cells remain NaN;
        physical zero emission remains zero.
    temperature_K : ndarray, shape (B,)
        The temperature supplied to Cloudy for each queried cell [K]. Unqueried
        positions are NaN. Line calculations reuse this same temperature.
    failed_cells : ndarray of bool, shape (B,)
        True only for queried cells that depend on failed table nodes. False
        at unqueried positions does not mean those cells have been validated.
    column_clipped_cells : dict of str to int
        Counts under 'below'/'above' for successful queries whose physical NH
        lies outside the attenuation axis. No simulation column is overwritten.

    Returned by CloudyCellReader.read_fields(). Example: emissivity_per_nH2['halpha'][7]
    is cell 7's Halpha value, independent of the table's row order.
    """

    emissivity_per_nH2: dict[str, np.ndarray]
    temperature_K: np.ndarray
    failed_cells: np.ndarray
    column_clipped_cells: dict[str, int]



class CloudyCellReader:
    """Read named line fields from the fixed 3D Cloudy table for selected cells.

    Parameters
    ----------
    lookup : CloudyLookup
        CloudyLookup from load_emission_calculator(). Stored as self.lookup and
        shared across batches; no selected cells or query results are retained.
        Model depth was set when this table was built.

    Examples
    --------
    reader = CloudyCellReader(lookup=lookup)
    fields = reader.read_fields(cells=cells, selected_cells=hot)
    fields.emissivity_per_nH2['halpha'][7] is original cell 7's queried Halpha.
    """

    def __init__(self, *, lookup: CloudyLookup):
        self.lookup = lookup

    def read_fields(
        self,
        *,
        cells: CellBatch,
        selected_cells: np.ndarray,
    ) -> CloudyCellFields:
        """Read selected cells and return named fields in original batch order.

        Parameters
        ----------
        cells : CellBatch
            From slab.batch(); supplies cached physical nH [cm^-3], T_QUOKKA [K]
            and shielding NH [cm^-2], each shape (B,).
        selected_cells : ndarray of bool, shape (B,)
            Caller-selected hot cells (T_QUOKKA >= 3000 K), independent of
            DESPOTIC temperature or CO availability.

        Returns
        -------
        CloudyCellFields
            Named emissivity_per_nH2 [erg cm^3/s] and the temperature actually
            supplied to Cloudy [K]. Returned fields keep the input cell shape
            and order; unqueried fields and failed emissivities are NaN.
        """
        # 1. Take T, nH and shielding NH for the cells selected by the caller.
        query_temperature_K = cells.temperature_QUOKKA_K[selected_cells]
        query_hydrogen_density_cm3 = cells.hydrogen_density_cm3[selected_cells]
        query_shielding_NH_cm2 = cells.shielding_NH_cm2[selected_cells]

        # 2. Read the three table axes; raw lookup handles NH-axis clipping.
        queried_fields = self.lookup.interpolate_available(
            temperature_K=query_temperature_K,
            hydrogen_density_cm3=query_hydrogen_density_cm3,
            shielding_NH_cm2=query_shielding_NH_cm2,
        )

        # 3. Count successful queries whose physical NH lies outside the lookup axis.
        column_clipped_cells = self.count_column_clipping(
            shielding_NH_cm2=query_shielding_NH_cm2,
            failed_queries=queried_fields.failed_queries,
        )

        # 4. Put each named result back at its original cell position in the batch.
        return self.restore_batch_positions(
            queried_fields=queried_fields,
            query_temperature_K=query_temperature_K,
            selected_cells=selected_cells,
            column_clipped_cells=column_clipped_cells,
        )

    def count_column_clipping(
        self,
        *,
        shielding_NH_cm2: np.ndarray,
        failed_queries: np.ndarray,
    ) -> dict[str, int]:
        """Count successful queries whose physical NH lies outside the lookup axis.

        shielding_NH_cm2 is the unchanged physical column [cm^-2], shape (Q,).
        failed_queries is the (Q,) bool array from self.lookup.interpolate_available().
        Returns integer 'below'/'above' counts using the loaded lookup's bounds.
        Example: [1e17, 1e19, 1e22], bounds (1e18, 1e21), and no failures give
        {'below': 1, 'above': 1}. No simulation column is changed.

        These physical-NH statistics differ from the lookup's log-NH flags at
        nextafter endpoints: log10 may round a tiny excursion back to the bound.
        """
        successful_queries = ~failed_queries
        lower_column, upper_column = self.lookup.attenuation_column_bounds_cm2
        below = successful_queries & (shielding_NH_cm2 < lower_column)
        above = successful_queries & (shielding_NH_cm2 > upper_column)
        return {
            'below': int(np.count_nonzero(below)),
            'above': int(np.count_nonzero(above)),
        }

    def restore_batch_positions(
        self,
        *,
        queried_fields: CloudyQueryResult,
        query_temperature_K: np.ndarray,
        selected_cells: np.ndarray,
        column_clipped_cells: dict[str, int],
    ) -> CloudyCellFields:
        """Put each selected result back at its original batch cell index.

        The lookup returned emissivity_per_nH2 for Q selected cells;
        query_temperature_K (Q,) [K] holds their actual query temperatures.
        We create full-batch NaN arrays
        and put these values at the True positions of selected_cells, shape (B,).
        Returned fields keep the original cell shape and order. Unqueried fields
        remain NaN; failure flags are False there. column_clipped_cells holds
        integer counts and is returned unchanged.
        """
        line_count = queried_fields.emissivity_per_nH2.shape[0]
        cell_count = selected_cells.size
        restored_emissivity_per_nH2 = np.full((line_count, cell_count), np.nan)
        temperature_K = np.full(cell_count, np.nan)
        failed_cells = np.zeros(cell_count, dtype=bool)

        restored_emissivity_per_nH2[:, selected_cells] = queried_fields.emissivity_per_nH2
        temperature_K[selected_cells] = query_temperature_K
        failed_cells[selected_cells] = queried_fields.failed_queries

        # Each named array is a row view; naming the lines creates no data copies.
        emissivity_per_nH2 = {}
        for line_index, line_key in enumerate(self.lookup.line_keys):
            emissivity_per_nH2[line_key] = restored_emissivity_per_nH2[line_index]

        return CloudyCellFields(
            emissivity_per_nH2=emissivity_per_nH2,
            temperature_K=temperature_K,
            failed_cells=failed_cells,
            column_clipped_cells=column_clipped_cells,
        )
