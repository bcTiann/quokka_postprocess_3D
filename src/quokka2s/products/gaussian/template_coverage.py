"""Aggregate template coverage without retaining cells or mixing min/max sums."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import DUST_STATES, REGIME_KEYS


COUNT_FIELDS = (
    "rows", "emitting_rows", "outside_rows", "outside_emitting_rows",
    "below_range_rows", "above_range_rows",
    "below_range_emitting_rows", "above_range_emitting_rows",
)
RANGE_FIELDS = ("sigma_range", "emitting_sigma_range", "outside_sigma_range")


@dataclass(frozen=True)
class TemplateGroupCoverage:
    """Coverage for one original-cell width group, with profile arrays[P].

    Rows count width-group Gaussians, not distinct physical cells. An emitting
    row has at least one positive luminosity account. Window light is the
    independently integrated original-cell five-sigma window reference [erg/s].
    """

    rows: int
    emitting_rows: int
    outside_rows: int
    outside_emitting_rows: int
    below_range_rows: int
    above_range_rows: int
    below_range_emitting_rows: int
    above_range_emitting_rows: int
    sigma_range: tuple[float, float] | None
    emitting_sigma_range: tuple[float, float] | None
    outside_sigma_range: tuple[float, float] | None
    profile_emitting_rows: np.ndarray
    outside_profile_emitting_rows: np.ndarray
    outside_input_luminosity: np.ndarray
    outside_window_luminosity_reference: np.ndarray


def sigma_range(values):
    """Return finite input extrema in km/s, or None for an empty selection."""
    if values.size == 0:
        return None
    return float(np.min(values)), float(np.max(values))


def combined_range(first, second):
    """Combine observed extrema; empty groups do not introduce infinities."""
    if first is None:
        return second
    if second is None:
        return first
    return min(first[0], second[0]), max(first[1], second[1])


class TemplateCoverageTotals:
    """Own worker/main totals indexed as (dust,line,temperature branch)."""

    def __init__(self, *, profile_count, table_minimum, table_maximum):
        self.table_minimum = float(table_minimum)
        self.table_maximum = float(table_maximum)
        self.counts = {name: 0 for name in COUNT_FIELDS}
        self.ranges = {name: None for name in RANGE_FIELDS}
        self.profile_emitting_rows = np.zeros(profile_count, dtype=np.int64)
        self.outside_profile_emitting_rows = np.zeros(profile_count, dtype=np.int64)
        self.outside_input_luminosity = np.zeros(profile_count, dtype=np.float64)
        self.outside_window_luminosity_reference = np.zeros(profile_count, dtype=np.float64)

    def add_group(self, coverage, profile_ids):
        """Insert one original width group's own accounts in its profile IDs."""
        for name in COUNT_FIELDS:
            self.counts[name] += int(getattr(coverage, name))
        for name in RANGE_FIELDS:
            self.ranges[name] = combined_range(self.ranges[name], getattr(coverage, name))
        self.profile_emitting_rows[profile_ids] += coverage.profile_emitting_rows
        self.outside_profile_emitting_rows[profile_ids] += coverage.outside_profile_emitting_rows
        self.outside_input_luminosity[profile_ids] += coverage.outside_input_luminosity
        self.outside_window_luminosity_reference[profile_ids] += coverage.outside_window_luminosity_reference

    def merge(self, other):
        """Merge a finished worker in the same order as its numerical products."""
        if (self.table_minimum, self.table_maximum) != (other.table_minimum, other.table_maximum):
            raise ValueError("Cannot merge coverage from different template bounds")
        for name in COUNT_FIELDS:
            self.counts[name] += other.counts[name]
        for name in RANGE_FIELDS:
            self.ranges[name] = combined_range(self.ranges[name], other.ranges[name])
        self.profile_emitting_rows += other.profile_emitting_rows
        self.outside_profile_emitting_rows += other.outside_profile_emitting_rows
        self.outside_input_luminosity += other.outside_input_luminosity
        self.outside_window_luminosity_reference += other.outside_window_luminosity_reference

    def summary(self, *, line_keys, input_luminosity, clipped_window_luminosity):
        """Return JSON-ready counters and each line's luminosity-weighted coverage."""
        input_light = np.asarray(input_luminosity).reshape(-1)
        window_light = np.asarray(clipped_window_luminosity).reshape(-1)
        profiles = []
        profile_id = 0
        for dust_state in DUST_STATES:
            for line_key in line_keys:
                for regime in REGIME_KEYS:
                    outside_input = float(self.outside_input_luminosity[profile_id])
                    outside_window = float(self.outside_window_luminosity_reference[profile_id])
                    input_total = float(input_light[profile_id])
                    window_total = float(window_light[profile_id])
                    profiles.append({
                        "dust_state": dust_state,
                        "line_key": line_key,
                        "regime": regime,
                        "emitting_rows": int(self.profile_emitting_rows[profile_id]),
                        "fallback_emitting_rows": int(self.outside_profile_emitting_rows[profile_id]),
                        "fallback_input_luminosity_erg_s": outside_input,
                        "fallback_window_luminosity_reference_erg_s": outside_window,
                        "fallback_fraction_of_input_luminosity": outside_input / input_total if input_total > 0.0 else 0.0,
                        "fallback_fraction_of_window_luminosity_reference": outside_window / window_total if window_total > 0.0 else 0.0,
                    })
                    profile_id += 1
        return {
            "table_sigma_range_kms": [self.table_minimum, self.table_maximum],
            **self.counts,
            **{name + "_kms": value for name, value in self.ranges.items()},
            "outside_emitting_fraction": self.counts["outside_emitting_rows"] / self.counts["emitting_rows"] if self.counts["emitting_rows"] else 0.0,
            "row_scope": "Original width-group Gaussian rows; one physical cell may occur in several width groups",
            "counter_scope": "Whole-run counters are repeated in each dust report; do not sum the dust copies",
            "emitting_definition": "At least one positive luminosity account; no brightness threshold is applied",
            "window_luminosity_scope": "Independent original-cell +/-5sigma window reference, not a second spectrum reduction",
            "profiles": profiles,
        }


def print_template_coverage(summary, statistics):
    """Write one final summary and only the accounts with fallback emission."""
    print(
        "Gaussian template coverage: "
        f"table_sigma_kms={summary['table_sigma_range_kms']}; "
        f"Gaussian_rows={summary['rows']}; emitting_rows={summary['emitting_rows']}; "
        f"outside_rows={summary['outside_rows']}; "
        f"outside_emitting_rows={summary['outside_emitting_rows']}; "
        f"outside_emitting_fraction={summary['outside_emitting_fraction']:.8g}; "
        f"below={summary['below_range_rows']}; above={summary['above_range_rows']}; "
        f"emitting_sigma_kms={summary['emitting_sigma_range_kms']}; "
        f"outside_sigma_kms={summary['outside_sigma_range_kms']}; "
        "fallback_probability_worker_seconds="
        f"{statistics['fallback_probability_seconds']:.8g}; "
        "integration_worker_seconds="
        f"{statistics['integration_seconds']:.8g}",
        flush=True,
    )
    for profile in summary["profiles"]:
        if profile["fallback_emitting_rows"] == 0:
            continue
        print(
            "Gaussian fallback account: "
            f"line={profile['line_key']}; dust={profile['dust_state']}; "
            f"regime={profile['regime']}; rows={profile['fallback_emitting_rows']}; "
            f"input_erg_s={profile['fallback_input_luminosity_erg_s']:.12g}; "
            "window_reference_erg_s="
            f"{profile['fallback_window_luminosity_reference_erg_s']:.12g}; "
            "fraction_of_window_reference="
            f"{profile['fallback_fraction_of_window_luminosity_reference']:.8g}",
            flush=True,
        )
