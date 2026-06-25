"""Common helpers for working with axis specifications."""

from typing import Union

AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
INDEX_TO_AXIS = {value: key for key, value in AXIS_TO_INDEX.items()}


def axis_index(axis: Union[str, int]) -> int:
    """Return the integer index (0, 1, 2) for an axis specification."""
    if isinstance(axis, str):
        try:
            return AXIS_TO_INDEX[axis.lower()]
        except KeyError as exc:
            raise ValueError("axis must be 'x', 'y', or 'z'") from exc

    if isinstance(axis, int):
        if axis in INDEX_TO_AXIS:
            return axis
        raise ValueError("axis index must be 0, 1, or 2")

    raise TypeError("axis must be a string ('x','y','z') or an int (0,1,2)")


def axis_label(axis: Union[str, int]) -> str:
    """Return the canonical axis label ('x', 'y', 'z') for an axis specification."""
    if isinstance(axis, str):
        axis_lower = axis.lower()
        if axis_lower in AXIS_TO_INDEX:
            return axis_lower
        raise ValueError("axis must be 'x', 'y', or 'z'")

    if isinstance(axis, int):
        try:
            return INDEX_TO_AXIS[axis]
        except KeyError as exc:
            raise ValueError("axis index must be 0, 1, or 2") from exc

    raise TypeError("axis must be a string ('x','y','z') or an int (0,1,2)")


__all__ = [
    "AXIS_TO_INDEX",
    "INDEX_TO_AXIS",
    "axis_index",
    "axis_label",
]
