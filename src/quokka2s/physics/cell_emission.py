"""Calculate each line's emissivity and gas temperature together, then apply dust."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quokka2s.despotic.cell_fields import DespoticCellFields, DespoticCellReader
from quokka2s.cloudy.cell_fields import CloudyCellFields, CloudyCellReader
from quokka2s.physics.line_emissivity import (
    ATOMIC_LINE_KEYS,
    CO_LINE_KEYS,
    HYDROGEN_CII_LINE_KEYS,
    CIII_CIV_LINE_KEYS,
    calculate_co_emissivities,
    calculate_cold_cii_emissivity,
    calculate_cold_halpha_emissivity,
    calculate_cold_hi21_emissivity,
    calculate_hot_atomic_emissivities,
    check_field_values,
)
from quokka2s.physics.dust_attenuation import attenuate_emissivities
from quokka2s.snapshot_reader import CellBatch


@dataclass(frozen=True)
class IntrinsicLineEmission:
    """One line's emissivity and corresponding gas temperature before dust.

    Attributes
    ----------
    emissivity_erg_s_cm3 : ndarray, shape (B,)
        Volume emissivity [erg/s/cm^3]. Missing values are NaN; zero is physical.
    temperature_K : ndarray, shape (B,)
        Temperature [K] from the same gas state as the emissivity. Missing
        emissivity positions also have NaN temperature. Both arrays keep the
        input cell shape and order.

    Example: lines['halpha'].temperature_K[7] is the temperature associated
    with cell 7's Halpha, before and after foreground dust attenuation.
    """

    emissivity_erg_s_cm3: np.ndarray
    temperature_K: np.ndarray


@dataclass(frozen=True)
class LineEmission:
    """One line's results for the original B-cell batch.

    Attributes
    ----------
    intrinsic_emissivity_erg_s_cm3, attenuated_emissivity_erg_s_cm3 : ndarray (B,)
        Epsilon [erg/s/cm^3] before/after foreground dust. A missing result for
        this line is NaN; prescribed zero emission remains zero.
    temperature_K : ndarray (B,)
        Temperature of the gas state used for this line [K]. Reused for
        thermal broadening and the emission-weighted phase plot.
    emissivity_is_missing : ndarray of bool (B,), computed property
        True where this line's intrinsic emissivity is nonfinite. No additional mask
        is stored; a physical zero is not missing.

    Example: emission.lines['halpha'].intrinsic_emissivity_erg_s_cm3[7]
    is cell 7's intrinsic Halpha. These are arrays, not one object per cell.
    """

    intrinsic_emissivity_erg_s_cm3: np.ndarray
    attenuated_emissivity_erg_s_cm3: np.ndarray
    temperature_K: np.ndarray

    @property
    def emissivity_is_missing(self) -> np.ndarray:
        """Return this line's missing-result flags in original cell order."""
        return ~np.isfinite(self.intrinsic_emissivity_erg_s_cm3)


@dataclass(frozen=True)
class BatchEmission:
    """Named line results and the QUOKKA temperature branch for one batch.

    Attributes
    ----------
    lines : dict of str to LineEmission
        The ten lines from CellEmissionCalculator.line_keys. Each line contains
        three (B,) arrays; select Halpha as lines['halpha'], without remembering a row.
    despotic_temperature_K : ndarray, shape (B,)
        Temperature from DespoticCellReader.read_fields() [K].
    cold_cells : ndarray of bool, shape (B,)
        True where T_QUOKKA < 3000 K, including unavailable results.
        Classification is independent of table availability.
    despotic_coordinate_clipped_cells : dict of str to int
        Counts of usable DESPOTIC-temperature queries clipped in nH, NH and dVdr.
    cloudy_column_clipped_cells : dict of str to int
        Counts of successful hot queries below/above the Cloudy attenuation grid.

    Examples
    --------
    emission.lines['halpha'].temperature_K[7] is cell 7's Halpha T.
    emission.lines['co10'].emissivity_is_missing[7] checks only that line.
    emission.cold_cells[7] describes its QUOKKA branch, even if CO is missing.
    """
    lines: dict[str, LineEmission]
    despotic_temperature_K: np.ndarray
    cold_cells: np.ndarray
    despotic_coordinate_clipped_cells: dict[str, int]
    cloudy_column_clipped_cells: dict[str, int]


class CellEmissionCalculator:
    """Calculate batch emission using two shared readers and named dust values.

    Attributes
    ----------
    despotic_reader : DespoticCellReader
        Reads T_DESPOTIC, CO, cold CII and cold particle densities from one table.
    cloudy_reader : CloudyCellReader
        Reads named atomic-line emissivity_per_nH2 from the current 3D table.
    line_keys : tuple of str, length 10
        Output order selected once by load_emission_calculator():
        ("cii", "halpha", "hi21", "ciii_977", "ciii_1907", "ciii_1909",
         "civ_1548", "civ_1551", "co10", "co21") for the current tables.
    dust_cross_section_cm2_H : dict of str to float
        One extinction cross-section [cm^2/H] per named line; HI is zero.

    This object is shared across worker threads. Only fixed readers, line names
    and cross-sections are stored on self; each calculate() keeps its batch's
    coordinates, masks and results in local variables.
    """

    def __init__(
        self,
        *,
        despotic_reader: DespoticCellReader,
        cloudy_reader: CloudyCellReader,
        line_keys: tuple[str, ...],
        dust_cross_section_cm2_H: dict[str, float],
    ):
        """Keep the readers and dust settings created by load_emission_calculator().

        Readers already own loaded tables. line_keys fixes output order, while
        dust_cross_section_cm2_H selects sigma by name, e.g. ['halpha'] [cm^2/H].
        No cells are queried and no batch arrays are allocated here.
        """
        self.despotic_reader = despotic_reader
        self.cloudy_reader = cloudy_reader
        self.line_keys = tuple(line_keys)
        self.dust_cross_section_cm2_H = dust_cross_section_cm2_H
        if set(self.line_keys) != set(ATOMIC_LINE_KEYS + CO_LINE_KEYS):
            raise ValueError('Emission requires the ten supported lines')

    def calculate(self, *, cells: CellBatch) -> BatchEmission:
        """Read one batch and calculate each line's emissivity and temperature together.

        Parameters
        ----------
        cells : CellBatch
            From slab.batch(); physical fields have shape (B,) in original order.
            Readers use nH [cm^-3], shielding NH [cm^-2], dV/dr [s^-1] and T_Q [K].

        Returns
        -------
        BatchEmission
            Named LineEmission records with intrinsic/attenuated epsilon
            [erg/s/cm^3] and the corresponding gas temperature [K], all (B,).
            Unavailable line results remain NaN; prescribed zero remains zero.

        Example: emission.lines['halpha'].temperature_K[7] is cell 7's Halpha T.
        No cell arrays are retained on this shared calculator after returning.
        """
        # 1. Choose the QUOKKA temperature branch once for this batch.
        cold_cells = cells.temperature_QUOKKA_K < 3000.0

        # 2. Read DESPOTIC T and CO for all cells, plus cold CII and H/e- densities.
        despotic_fields = self.despotic_reader.read_fields(
            cells=cells,
            cold_cells=cold_cells,
        )

        # 3. Read Cloudy emission and retain its query temperature for every hot cell.
        cloudy_fields = self.cloudy_reader.read_fields(
            cells=cells,
            selected_cells=~cold_cells,
        )

        # 4. Each group returns emissivity and temperature from the same gas state.
        intrinsic_lines = self.calculate_intrinsic_lines(
            cells=cells,
            cold_cells=cold_cells,
            despotic_fields=despotic_fields,
            cloudy_fields=cloudy_fields,
        )

        # 5. Attenuate the light; each line keeps its existing gas temperature.
        lines = self.apply_foreground_dust(
            intrinsic_lines=intrinsic_lines,
            foreground_column_cm2=cells.foreground_NH_cm2,
        )

        # 6. Return named lines together with the branch and lookup counts.
        return self.build_batch_emission(
            lines=lines,
            despotic_fields=despotic_fields,
            cloudy_fields=cloudy_fields,
            cold_cells=cold_cells,
        )

    def calculate_intrinsic_lines(
        self,
        *,
        cells: CellBatch,
        cold_cells: np.ndarray,
        despotic_fields: DespoticCellFields,
        cloudy_fields: CloudyCellFields,
    ) -> dict[str, IntrinsicLineEmission]:
        """Calculate emissivity and temperature together under one set of line rules.

        Line             T_QUOKKA < 3000 K        T_QUOKKA >= 3000 K
        Halpha, HI, CII   DESPOTIC state and T_D    Cloudy state and query T_Q
        CIII, CIV         Zero emission            Cloudy state and query T_Q
        CO(1-0), CO(2-1)  DESPOTIC state and T_D    DESPOTIC state and T_D

        Parameters
        ----------
        cells : CellBatch
            Original (B,) cell arrays. Supplies physical nH [cm^-3] and T_Q [K].
        cold_cells : ndarray of bool, shape (B,)
            T_QUOKKA < 3000 K; table availability does not change this mask.
        despotic_fields, cloudy_fields : DespoticCellFields, CloudyCellFields
            From the two readers. Each carries its own gas temperature and
            queried emission fields in the original cell shape and order.

        Returns
        -------
        dict of str to IntrinsicLineEmission
            One (B,) emissivity array [erg/s/cm^3] and one (B,) temperature array
            [K] per line, in self.line_keys order. No table calls happen here.
            Missing emission and temperature remain NaN for that line only.

        Example: lines['co10'] uses T_D even when the cell's T_Q is 10000 K.
        Cold CIII/CIV retain T_Q as a placeholder for their zero emission;
        those entries contribute no broadened light.
        """
        hydrogen_and_cii_lines = self.calculate_cold_hydrogen_and_cii_lines(
            cold_cells=cold_cells,
            despotic_fields=despotic_fields,
        )
        ciii_and_civ_lines = self.initialize_ciii_and_civ_lines(
            cells=cells,
            cold_cells=cold_cells,
        )

        lines = {}
        lines.update(hydrogen_and_cii_lines)
        lines.update(ciii_and_civ_lines)

        # All eight hot atomic lines use the same Cloudy state and physical nH.
        self.fill_hot_cloudy_lines(
            lines=lines,
            cells=cells,
            cold_cells=cold_cells,
            cloudy_fields=cloudy_fields,
        )
        co_lines = self.calculate_co_lines(
            despotic_fields=despotic_fields,
        )
        lines.update(co_lines)

        # Restore the requested output order without copying the result arrays.
        ordered_lines = {}
        for line_key in self.line_keys:
            ordered_lines[line_key] = lines[line_key]
        self.check_intrinsic_lines(lines=ordered_lines)
        return ordered_lines

    def calculate_cold_hydrogen_and_cii_lines(
        self,
        *,
        cold_cells: np.ndarray,
        despotic_fields: DespoticCellFields,
    ) -> dict[str, IntrinsicLineEmission]:
        """Use the DESPOTIC state for cold Halpha/HI/CII; leave hot entries NaN.

        Inputs are the original (B,) cold mask and DESPOTIC fields.
        Returns the three named lines, each with (B,) epsilon [erg/s/cm^3]
        and temperature [K]. A DESPOTIC failure only removes cold results.
        fill_hot_cloudy_lines() supplies the hot entries afterward.
        """
        lines = self.create_empty_intrinsic_lines(
            line_keys=HYDROGEN_CII_LINE_KEYS,
            cell_array_shape=cold_cells.shape,
        )
        cold_despotic_cells = cold_cells & ~despotic_fields.invalid_temperature_cells
        cold_temperature_K = despotic_fields.temperature_K[cold_despotic_cells]

        # The cold line formulae use fields from this same DESPOTIC state.
        cold_cii_emissivity = calculate_cold_cii_emissivity(
            despotic_fields=despotic_fields,
            selected_cells=cold_despotic_cells,
        )
        self.assign_emission_from_gas_state(
            line=lines['cii'],
            selected_cells=cold_despotic_cells,
            emissivity_erg_s_cm3=cold_cii_emissivity,
            temperature_K=cold_temperature_K,
        )
        cold_halpha_emissivity = calculate_cold_halpha_emissivity(
            despotic_fields=despotic_fields,
            selected_cells=cold_despotic_cells,
        )
        self.assign_emission_from_gas_state(
            line=lines['halpha'],
            selected_cells=cold_despotic_cells,
            emissivity_erg_s_cm3=cold_halpha_emissivity,
            temperature_K=cold_temperature_K,
        )
        cold_hi21_emissivity = calculate_cold_hi21_emissivity(
            despotic_fields=despotic_fields,
            selected_cells=cold_despotic_cells,
        )
        self.assign_emission_from_gas_state(
            line=lines['hi21'],
            selected_cells=cold_despotic_cells,
            emissivity_erg_s_cm3=cold_hi21_emissivity,
            temperature_K=cold_temperature_K,
        )

        return lines

    def initialize_ciii_and_civ_lines(
        self,
        *,
        cells: CellBatch,
        cold_cells: np.ndarray,
    ) -> dict[str, IntrinsicLineEmission]:
        """Set cold CIII/CIV to zero; leave hot entries for the Cloudy calculation.

        Inputs are the original CellBatch and its (B,) cold mask.
        Returns five named lines with (B,) epsilon [erg/s/cm^3] and temperature
        [K]. Cold zeros do not require DESPOTIC; their stored T_Q is a placeholder.
        """
        lines = self.create_empty_intrinsic_lines(
            line_keys=CIII_CIV_LINE_KEYS,
            cell_array_shape=cold_cells.shape,
        )
        cold_zero_emissivity = np.zeros(np.count_nonzero(cold_cells))
        cold_placeholder_temperature_K = cells.temperature_QUOKKA_K[cold_cells]
        for line in lines.values():
            self.assign_emission_from_gas_state(
                line=line,
                selected_cells=cold_cells,
                emissivity_erg_s_cm3=cold_zero_emissivity,
                temperature_K=cold_placeholder_temperature_K,
            )

        return lines

    def calculate_co_lines(
        self,
        *,
        despotic_fields: DespoticCellFields,
    ) -> dict[str, IntrinsicLineEmission]:
        """Calculate both CO lines from DESPOTIC for cold and hot cells alike.

        despotic_fields comes from DespoticCellReader.read_fields(). Returns
        two named results with the original (B,) epsilon [erg/s/cm^3] and T_D [K].
        Missing DESPOTIC temperature leaves both results NaN at that cell.
        The lumPerH conversion uses DESPOTIC's clipped query nH.
        """
        lines = self.create_empty_intrinsic_lines(
            line_keys=CO_LINE_KEYS,
            cell_array_shape=despotic_fields.temperature_K.shape,
        )
        available_cells = ~despotic_fields.invalid_temperature_cells
        temperature_K = despotic_fields.temperature_K[available_cells]
        check_field_values('DESPOTIC temperature', temperature_K, positive=True)
        co10_emissivity, co21_emissivity = calculate_co_emissivities(
            despotic_fields=despotic_fields,
            selected_cells=available_cells,
        )
        self.assign_emission_from_gas_state(
            line=lines['co10'],
            selected_cells=available_cells,
            emissivity_erg_s_cm3=co10_emissivity,
            temperature_K=temperature_K,
        )
        self.assign_emission_from_gas_state(
            line=lines['co21'],
            selected_cells=available_cells,
            emissivity_erg_s_cm3=co21_emissivity,
            temperature_K=temperature_K,
        )
        return lines

    def fill_hot_cloudy_lines(
        self,
        *,
        lines: dict[str, IntrinsicLineEmission],
        cells: CellBatch,
        cold_cells: np.ndarray,
        cloudy_fields: CloudyCellFields,
    ) -> None:
        """Write hot emissivities and the exact temperature used to query Cloudy.

        lines contains one full-shape result for each of the eight atomic lines.
        cells supplies physical nH [cm^-3]; cold_cells is the original (B,) mask.
        cloudy_fields supplies emissivity_per_nH2 and its query temperatures [K].
        Updates both arrays at successful hot positions; returns None.
        """
        hot_cloudy_cells = ~cold_cells & ~cloudy_fields.failed_cells
        hot_emissivities = calculate_hot_atomic_emissivities(
            cloudy_fields=cloudy_fields,
            hydrogen_density_cm3=cells.hydrogen_density_cm3,
            selected_cells=hot_cloudy_cells,
            line_keys=tuple(lines),
        )
        temperature_K = cloudy_fields.temperature_K[hot_cloudy_cells]
        for line_key, emissivity in hot_emissivities.items():
            self.assign_emission_from_gas_state(
                line=lines[line_key],
                selected_cells=hot_cloudy_cells,
                emissivity_erg_s_cm3=emissivity,
                temperature_K=temperature_K,
            )

    def create_empty_intrinsic_lines(
        self,
        *,
        line_keys: tuple[str, ...],
        cell_array_shape: tuple[int, ...],
    ) -> dict[str, IntrinsicLineEmission]:
        """Allocate paired NaN arrays for the named lines in this group.

        cell_array_shape is the original cell shape, e.g. (1000000,).
        Returns one emissivity [erg/s/cm^3] and temperature [K] array per line.
        """
        lines = {}
        for line_key in line_keys:
            lines[line_key] = IntrinsicLineEmission(
                emissivity_erg_s_cm3=np.full(cell_array_shape, np.nan),
                temperature_K=np.full(cell_array_shape, np.nan),
            )
        return lines

    def assign_emission_from_gas_state(
        self,
        *,
        line: IntrinsicLineEmission,
        selected_cells: np.ndarray,
        emissivity_erg_s_cm3: np.ndarray,
        temperature_K: np.ndarray,
    ) -> None:
        """Put the selected gas state's emissivity and temperature into one line.

        line holds two original (B,) destination arrays; selected_cells is (B,).
        The supplied epsilon [erg/s/cm^3] and T [K] each have R entries, where
        R is the number of selected cells. Updates both arrays; returns None.
        Example: mask [True, False, True] puts epsilon [2, 7] and T [50, 100]
        at cells 0 and 2; cell 1 keeps NaN in both arrays.
        """
        line.emissivity_erg_s_cm3[selected_cells] = emissivity_erg_s_cm3
        line.temperature_K[selected_cells] = temperature_K

    def check_intrinsic_lines(self, *, lines: dict[str, IntrinsicLineEmission]) -> None:
        """Check paired line values without choosing their gas state again.

        Each line has original (B,) epsilon [erg/s/cm^3] and temperature [K].
        Returns None; checks nonnegative available epsilon and positive matching T.
        Missing positions remain NaN, and physical zeros remain available.
        """
        for line in lines.values():
            available_emissivity = ~np.isnan(line.emissivity_erg_s_cm3)
            check_field_values(
                'Volume emissivity',
                line.emissivity_erg_s_cm3[available_emissivity],
            )
            check_field_values(
                'Thermal temperature',
                line.temperature_K[available_emissivity],
                positive=True,
            )

    def apply_foreground_dust(
        self,
        *,
        intrinsic_lines: dict[str, IntrinsicLineEmission],
        foreground_column_cm2: np.ndarray,
    ) -> dict[str, LineEmission]:
        """Apply dust and keep each line's existing emissivity/temperature pair.

        Parameters
        ----------
        intrinsic_lines : dict of str to IntrinsicLineEmission
            From calculate_intrinsic_lines(); each epsilon [erg/s/cm^3] and
            temperature [K] array has the original (B,) shape and order.
        foreground_column_cm2 : ndarray, shape (B,)
            cells.foreground_NH_cm2 [cm^-2], from each cell center to the -z face.

        Returns
        -------
        dict of str to LineEmission
            Original intrinsic epsilon, newly attenuated epsilon, and unchanged
            temperature arrays. Missing values stay NaN; HI's sigma is zero.

        Example: lines['halpha'].temperature_K is the same array as before dust.
        """
        lines = {}
        for line_key, intrinsic_line in intrinsic_lines.items():
            intrinsic_emissivity = intrinsic_line.emissivity_erg_s_cm3
            available_cells = np.isfinite(intrinsic_emissivity)
            attenuated_emissivity = intrinsic_emissivity.copy()
            cross_section = self.dust_cross_section_cm2_H[line_key]
            attenuated_values = attenuate_emissivities(
                emissivity=intrinsic_emissivity[available_cells],
                foreground_NH=foreground_column_cm2[available_cells],
                sigma_cm2_H=cross_section,
            )
            attenuated_emissivity[available_cells] = attenuated_values
            lines[line_key] = LineEmission(
                intrinsic_emissivity_erg_s_cm3=intrinsic_emissivity,
                attenuated_emissivity_erg_s_cm3=attenuated_emissivity,
                temperature_K=intrinsic_line.temperature_K,
            )
        return lines

    def build_batch_emission(
        self,
        *,
        lines: dict[str, LineEmission],
        despotic_fields: DespoticCellFields,
        cloudy_fields: CloudyCellFields,
        cold_cells: np.ndarray,
    ) -> BatchEmission:
        """Return the calculated lines with the branch and lookup statistics.

        lines comes from apply_foreground_dust(); each record holds original
        (B,) arrays. The two readers supply T_D and clipping metadata.
        cold_cells is the independent (B,) T_QUOKKA < 3000 K classification.
        Returns BatchEmission; line records and arrays are reused without copying.
        """
        despotic_clipped_counts = {}
        for coordinate, clipped_cells in despotic_fields.coordinate_clipped.items():
            retained_clipped_cells = clipped_cells & ~despotic_fields.invalid_temperature_cells
            despotic_clipped_counts[coordinate] = int(np.count_nonzero(retained_clipped_cells))

        return BatchEmission(
            lines=lines,
            despotic_temperature_K=despotic_fields.temperature_K,
            cold_cells=cold_cells,
            despotic_coordinate_clipped_cells=despotic_clipped_counts,
            cloudy_column_clipped_cells=cloudy_fields.column_clipped_cells,
        )
