"""Measure cold-branch Halpha, HI and CII light at high DESPOTIC temperatures.

This diagnostic reuses calculated cell emission. It never filters a product,
queries a table, or retains cell arrays.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quokka2s.physics.line_emissivity import HYDROGEN_CII_LINE_KEYS
from quokka2s.physics.settings import EMISSION_TEMPERATURE_BOUNDARY_K

if TYPE_CHECKING:
    from quokka2s.physics.cell_emission import BatchEmission
    from quokka2s.snapshot_reader import CellBatch


DESPOTIC_TEMPERATURE_THRESHOLDS_K = (1e4, 4e4, 1e5, 1e6)


class ColdGasLuminosityByDespoticTemperature:
    """Keep intrinsic/attenuated luminosity sums above four TD thresholds.

    line_keys is the process line order. Only Halpha, HI and CII are recorded;
    each line has two arrays of shape (4,) [erg/s], in threshold order. For
    example, intrinsic_luminosity_erg_s['halpha'][1] sums cold-branch Halpha
    from cells with TD >= 40000 K. Threshold selections overlap.
    """

    def __init__(self, line_keys):
        self.line_indices = {}
        self.intrinsic_luminosity_erg_s = {}
        self.attenuated_luminosity_erg_s = {}
        for line_index, line_key in enumerate(line_keys):
            if line_key not in HYDROGEN_CII_LINE_KEYS:
                continue
            self.line_indices[line_key] = line_index
            self.intrinsic_luminosity_erg_s[line_key] = np.zeros(
                len(DESPOTIC_TEMPERATURE_THRESHOLDS_K),
            )
            self.attenuated_luminosity_erg_s[line_key] = np.zeros(
                len(DESPOTIC_TEMPERATURE_THRESHOLDS_K),
            )

    def add_batch(self, *, cells: CellBatch, emission: BatchEmission) -> None:
        """Add cold cells' light using existing (B,) temperatures and emissivities.

        cells supplies the common cell_volume_cm3 [cm^3]. emission supplies
        cold_cells (T_QUOKKA < 3000 K), despotic_temperature_K [K] and named
        line emissivities [erg/s/cm^3], all in the original batch order.
        Each line skips only its own missing emission. Zero emission is valid.
        No cell arrays survive this call; only luminosity sums are updated.
        """
        despotic_temperature_K = emission.despotic_temperature_K
        cold_cells_with_temperature = (
            emission.cold_cells & np.isfinite(despotic_temperature_K)
        )
        for threshold_index, threshold_K in enumerate(DESPOTIC_TEMPERATURE_THRESHOLDS_K):
            cells_above_threshold = (
                cold_cells_with_temperature & (despotic_temperature_K >= threshold_K)
            )
            if not np.any(cells_above_threshold):
                continue
            for line_key in self.line_indices:
                line = emission.lines[line_key]
                selected_cells = cells_above_threshold & ~line.emissivity_is_missing
                self.intrinsic_luminosity_erg_s[line_key][threshold_index] += (
                    line.intrinsic_emissivity_erg_s_cm3[selected_cells].sum()
                    * cells.cell_volume_cm3
                )
                self.attenuated_luminosity_erg_s[line_key][threshold_index] += (
                    line.attenuated_emissivity_erg_s_cm3[selected_cells].sum()
                    * cells.cell_volume_cm3
                )

    def merge(self, other: ColdGasLuminosityByDespoticTemperature) -> None:
        """Add another batch's sums; EmissionProducts preserves input batch order."""
        for line_key in self.line_indices:
            self.intrinsic_luminosity_erg_s[line_key] += other.intrinsic_luminosity_erg_s[line_key]
            self.attenuated_luminosity_erg_s[line_key] += other.attenuated_luminosity_erg_s[line_key]

    def build_report(
        self,
        *,
        intrinsic_luminosity_by_branch_erg_s: np.ndarray,
        attenuated_luminosity_by_branch_erg_s: np.ndarray,
    ) -> dict:
        """Return JSON-ready light fractions without modifying accumulated values.

        Inputs are EmissionProducts' existing total-light arrays, each (L, 2)
        [erg/s], in process line order with cold/hot columns. Their sum is the
        denominator for that line and dust state over all processed cells and
        all velocities. A zero denominator gives None (JSON null), not zero.
        Example: luminosities [2, 1, 0, 0] with total 10 give [.2, .1, 0, 0].
        """
        lines = {}
        for line_key, line_index in self.line_indices.items():
            lines[line_key] = {}
            dust_luminosities = (
                (
                    "intrinsic",
                    self.intrinsic_luminosity_erg_s[line_key],
                    intrinsic_luminosity_by_branch_erg_s[line_index],
                ),
                (
                    "attenuated",
                    self.attenuated_luminosity_erg_s[line_key],
                    attenuated_luminosity_by_branch_erg_s[line_index],
                ),
            )
            for dust_state, luminosity_above_threshold, branch_luminosities in dust_luminosities:
                total_luminosity = float(branch_luminosities.sum())
                if total_luminosity > 0:
                    fractions = (luminosity_above_threshold / total_luminosity).tolist()
                else:
                    fractions = [None] * len(DESPOTIC_TEMPERATURE_THRESHOLDS_K)
                lines[line_key][dust_state] = {
                    "total_luminosity_erg_s": total_luminosity,
                    "luminosity_above_threshold_erg_s": luminosity_above_threshold.tolist(),
                    "fraction_of_total_luminosity": fractions,
                }

        return {
            "cold_cell_selection": f"T_QUOKKA < {EMISSION_TEMPERATURE_BOUNDARY_K:g} K",
            "despotic_temperature_thresholds_K": list(DESPOTIC_TEMPERATURE_THRESHOLDS_K),
            "threshold_selection": "finite T_DESPOTIC >= threshold; selections overlap",
            "denominator": (
                "This line's available luminosity in the same dust state, summed over "
                "cold and hot branches, all processed cells and all velocities. "
                "Fractions are between 0 and 1; a zero total gives null."
            ),
            "lines": lines,
        }
