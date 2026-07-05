from typing import Union

import numpy as np
from yt.units import mh

from .utils.axes import axis_index

def get_attenuation_factor(
    number_column_density,
    A_lambda_over_NH=8e-22,
):
    A_lambda_3d = number_column_density * A_lambda_over_NH

    print(f"Max A_lambda: {A_lambda_3d.max():.2f} mag")

    attenuation_factor_3d = 10.0 ** (-A_lambda_3d / 2.5)
    print(
        "Attenuation factor range: min="
        f"{attenuation_factor_3d.min():.2e}, max={attenuation_factor_3d.max():.2e}"
    )

    return attenuation_factor_3d



def along_sight_cumulation(
    data: np.ndarray,
    axis: Union[str, int],
    sign: str,
):
    """Cumulative sum along a requested axis and direction."""
    axis = axis_index(axis)

    if sign == "+":
        return np.flip(np.cumsum(np.flip(data, axis=axis), axis=axis), axis=axis)

    if sign == "-":
        return np.cumsum(data, axis=axis)

    raise ValueError("Direction must be '+' or '-'.")


def calculate_cumulative_column_density(
    density_3d: np.ndarray,
    dx_3d: np.ndarray,
    axis: Union[str, int],
    X_H: float,
    sign: str,
):
    """Calculates the cumulative hydrogen column density along a given axis."""
    m_H = mh.in_cgs()
    n_H_3d = (density_3d * X_H) / m_H
    N_H_cell_3d = n_H_3d * dx_3d

    N_H_cumulative = along_sight_cumulation(N_H_cell_3d, axis=axis, sign=sign)
    return N_H_cumulative


def calculate_attenuation(
    column_density_3d: np.ndarray,
    A_lambda_over_NH: float,
):
    """Calculates the dust attenuation factor from column density."""
    A_lambda_3d = column_density_3d * A_lambda_over_NH
    attenuation_factor = 10.0 ** (-A_lambda_3d / 2.5)
    return attenuation_factor, A_lambda_3d
