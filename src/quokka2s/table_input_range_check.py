"""Compare saved physical ranges with current table axes, without cell queries."""
from __future__ import annotations

import numpy as np

from quokka2s.cloudy.lookup import CloudyLookup, COORDINATE_LOG_BOUNDARY_RTOL
from quokka2s.despotic.cell_fields import COORDINATE_BOUNDARY_RTOL
from quokka2s.despotic.lookup import DespoticLookup


def compare_axis_range(
    *,
    measured: dict,
    table_bounds: tuple[float, float],
    permitted_bounds: tuple[float, float],
    logarithmic_comparison: bool = False,
    allow_lookup_clipping: bool = False,
) -> dict:
    """Compare one saved range with exact table bounds and processing tolerance.

    measured comes from FieldRange.as_dict() and retains physical units.
    permitted_bounds use physical units, or log10 units when requested.
    Only Cloudy NH allows arbitrary lookup clipping. Actual physical ranges
    are never replaced. Empty query selections pass; invalid values do not.
    """
    minimum = measured["minimum"]
    maximum = measured["maximum"]
    below = False
    above = False
    lookup_range = None
    if minimum is not None:
        comparison_minimum = np.log10(minimum) if logarithmic_comparison else minimum
        comparison_maximum = np.log10(maximum) if logarithmic_comparison else maximum
        below = bool(comparison_minimum < permitted_bounds[0])
        above = bool(comparison_maximum > permitted_bounds[1])
        if allow_lookup_clipping:
            lookup_range = {
                "minimum": float(np.clip(minimum, *table_bounds)),
                "maximum": float(np.clip(maximum, *table_bounds)),
            }
    covered = measured["invalid_cell_count"] == 0
    if not allow_lookup_clipping:
        covered = covered and not below and not above
    return {
        "unit": measured["unit"],
        "physical_minimum": minimum,
        "physical_maximum": maximum,
        "table_minimum": table_bounds[0],
        "table_maximum": table_bounds[1],
        "below_permitted_range": below,
        "above_permitted_range": above,
        "invalid_cell_count": measured["invalid_cell_count"],
        "lookup_clipping_allowed": allow_lookup_clipping,
        "clipped_lookup_range": lookup_range,
        "input_range_covered": bool(covered),
    }


def compare_despotic_ranges(*, ranges: dict, lookup: DespoticLookup) -> dict:
    """Use the same physical endpoint tolerance as DespoticCellReader."""
    table_axes = {
        "nH": lookup.table.nH_values,
        "NH": lookup.table.col_density_values,
        "dVdr": lookup.table.dVdr_values,
    }
    checks = {}
    for name, axis in table_axes.items():
        minimum = float(axis[0])
        maximum = float(axis[-1])
        checks[name] = compare_axis_range(
            measured=ranges[name],
            table_bounds=(minimum, maximum),
            permitted_bounds=(
                minimum * (1.0 - COORDINATE_BOUNDARY_RTOL),
                maximum * (1.0 + COORDINATE_BOUNDARY_RTOL),
            ),
        )
    return checks


def compare_cloudy_ranges(*, ranges: dict, lookup: CloudyLookup) -> dict:
    """Check hot nH/T in log coordinates; permit NH lookup clipping only."""
    checks = {}
    for name, axis in (("nH", lookup.log_nH), ("T_QUOKKA", lookup.log_T)):
        tolerance = COORDINATE_LOG_BOUNDARY_RTOL * max(1.0, abs(axis[0]), abs(axis[-1]))
        checks[name] = compare_axis_range(
            measured=ranges[name],
            table_bounds=(float(10.0 ** axis[0]), float(10.0 ** axis[-1])),
            permitted_bounds=(float(axis[0] - tolerance), float(axis[-1] + tolerance)),
            logarithmic_comparison=True,
        )
    column_bounds = tuple(float(bound) for bound in lookup.attenuation_column_bounds_cm2)
    checks["NH"] = compare_axis_range(
        measured=ranges["NH"],
        table_bounds=column_bounds,
        permitted_bounds=column_bounds,
        allow_lookup_clipping=True,
    )
    return checks


def build_table_range_report(
    *,
    statistics: dict,
    despotic: DespoticLookup,
    cloudy: CloudyLookup,
) -> dict:
    """Build a range-only report, explicitly excluding table-hole certification.

    statistics is a completed or partial scan from scan_snapshot_input_ranges().
    Lookups contain the current table axes; no interpolation is performed here.
    Changing table bounds can therefore be checked against saved statistics.
    """
    despotic_checks = compare_despotic_ranges(ranges=statistics["despotic"], lookup=despotic)
    cloudy_checks = compare_cloudy_ranges(ranges=statistics["cloudy"], lookup=cloudy)
    axes_covered = all(
        check["input_range_covered"]
        for checks in (despotic_checks, cloudy_checks)
        for check in checks.values()
    )
    return {
        "check": "coordinate ranges only; not interpolation availability or physical-setup suitability",
        "scan_status": statistics["scan_status"],
        "full_snapshot": statistics["full_snapshot"],
        "selected_input_ranges_covered": bool(
            statistics["scan_status"] == "complete"
            and statistics["invalid_quokka_temperature_count"] == 0
            and axes_covered
        ),
        "despotic": despotic_checks,
        "cloudy": cloudy_checks,
        "table_holes_checked": False,
    }
