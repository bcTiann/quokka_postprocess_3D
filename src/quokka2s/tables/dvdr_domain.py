"""Snapshot-matched DESPOTIC ``dV/dr`` grid definition.

The original production table used 35 logarithmic nodes from 1e-19 to
1e-12 s^-1.  Keep those nodes exactly and extend only beyond them so a
before/after comparison does not also change interpolation throughout the
original domain.
"""
from __future__ import annotations

import numpy as np


LEGACY_DVDR_MIN_S = 1.0e-19
LEGACY_DVDR_MAX_S = 1.0e-12
LEGACY_DVDR_POINTS = 35

# Full-resolution plt0655228, evaluated as abs(div(v))/3 with the same
# finite-difference operator as the production derived field.
SIMULATION_DVDR_MIN_S = 1.25685313685528378e-22
SIMULATION_DVDR_MAX_S = 2.78718511000726292e-12


def legacy_dvdr_values() -> np.ndarray:
    """Return the exact logarithmic axis used by the existing 35^3 table."""
    return np.logspace(
        np.log10(LEGACY_DVDR_MIN_S),
        np.log10(LEGACY_DVDR_MAX_S),
        LEGACY_DVDR_POINTS,
    )


def extended_dvdr_values() -> np.ndarray:
    """Return the old axis plus snapshot-covering nodes at both ends.

    The number of new intervals is chosen so their log spacing is no larger
    than the legacy spacing.  The legacy values themselves are inserted
    unchanged.
    """
    legacy = legacy_dvdr_values()
    legacy_spacing = float(np.diff(np.log10(legacy))[0])

    lower_intervals = int(np.ceil(
        (np.log10(legacy[0]) - np.log10(SIMULATION_DVDR_MIN_S))
        / legacy_spacing
    ))
    upper_intervals = int(np.ceil(
        (np.log10(SIMULATION_DVDR_MAX_S) - np.log10(legacy[-1]))
        / legacy_spacing
    ))

    lower = np.logspace(
        np.log10(SIMULATION_DVDR_MIN_S),
        np.log10(legacy[0]),
        lower_intervals + 1,
    )[:-1]
    upper = np.logspace(
        np.log10(legacy[-1]),
        np.log10(SIMULATION_DVDR_MAX_S),
        upper_intervals + 1,
    )[1:]
    values = np.concatenate((lower, legacy, upper))
    values[0] = SIMULATION_DVDR_MIN_S
    values[-1] = SIMULATION_DVDR_MAX_S
    if np.any(np.diff(values) <= 0.0):
        raise RuntimeError("extended dV/dr axis is not strictly increasing")
    return values


def added_dvdr_values() -> np.ndarray:
    """Return only the nodes absent from the legacy table."""
    values = extended_dvdr_values()
    return values[
        (values < LEGACY_DVDR_MIN_S) | (values > LEGACY_DVDR_MAX_S)
    ]
