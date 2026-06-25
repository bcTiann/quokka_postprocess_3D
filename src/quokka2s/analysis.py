from typing import Optional, Union
import time

import numpy as np
from tqdm import tqdm
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg

from .utils.axes import axis_index


def run_despotic_on_map(
    nH_map: np.ndarray,
    colDen_map: np.ndarray,
    Tg_map: Optional[np.ndarray] = None,
    dVdr_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Iterates over a 2D slice and runs DESPOTIC on each pixel.

    Args:
        nH_map (np.ndarray): 2D map of Volume density of H nuclei [cm^-3].
        Tg_map (np.ndarray): 2D map of gas temperature [K].
        colDen_map (np.ndarray): 2D map of Column density of H nuclei [cm^-2].
        dVdr_map (np.ndarray, optional): 2D map of LVG velocity gradient [s^-1].
                  Defaults to 1e-14 s^-1 (median ISM value) if not given.
    """
    # despotic is an optional, table-building-only dependency.  Import it lazily
    # here so the runtime pipeline (which never calls this function) does not
    # require despotic to be installed — see also tables/solver.py, tables/
    # builder.py, despotic_tables.py.
    from despotic import cloud
    from despotic.chemistry import NL99

    shape = nH_map.shape
    print(f"nH_map.shape = {nH_map.shape}")

    co_line_map = np.zeros(shape)
    Tg_map_final = np.zeros(shape)

    print(f"\n--- Running DESPOTIC on a {shape[0]}x{shape[1]} map ---")
    start_time = time.time()

    Tg_working = []
    # Single-guess fallback list — bistability test (2026-05-09) showed final
    # Tg is unique per (nH, NH), so the long historical guess ladder is
    # unnecessary. Kept short for occasional convergence retries.
    tg_guesses = [100.0, 1000.0, 10000.0]
    working = 0
    for i in tqdm(range(shape[0]), desc="DESPOTIC Processing Rows"):
        for j in range(shape[1]):
            success = False
            dvdr_ij = float(dVdr_map[i, j]) if dVdr_map is not None else 1e-14
            for guess in tg_guesses:
                try:
                    cell = cloud()
                    cell.nH = nH_map[i, j]
                    cell.colDen = colDen_map[i, j]
                    cell.Tg = guess
                    cell.dVdr = dvdr_ij

                    cell.sigmaNT = 2.0e5
                    cell.comp.xoH2 = 0.1
                    cell.comp.xpH2 = 0.4
                    cell.comp.xHe = 0.1
                    cell.comp.mu = 0.6

                    cell.dust.alphaGD = 3.2e-34
                    cell.dust.sigma10 = 2.0e-25
                    cell.dust.sigmaPE = 1.0e-21
                    cell.dust.sigmaISRF = 3.0e-22
                    cell.dust.beta = 2.0
                    cell.dust.Zd = 1.0
                    cell.Td = 10.0
                    cell.rad.TCMB = 2.73
                    cell.rad.TradDust = 0.0
                    cell.rad.ionRate = 2.0e-17
                    cell.rad.chi = 1.0

                    cell.addEmitter("CO", 8.0e-9)
                    co_abundance = cell.emitters["CO"].abundance
                    print("++++++++++++++++++++++++\n")
                    print(f"initial CO abundance = {co_abundance}")
                    print(f"initial Tg = {cell.Tg}")
                    print("haven't pass")
                    cell.setChemEq(network=NL99, evolveTemp="iterateDust")
                    print("pass!!")
                    lines = cell.lineLum("CO", escapeProbGeom='LVG')
                    co_int_TB = lines[0]["intTB"]

                    co_line_map[i, j] = co_int_TB
                    Tg_map_final[i, j] = cell.Tg

                    Tg_working.append(guess)

                    co_abundance = cell.emitters["CO"].abundance

                    print(f"after ChemEq CO abundance = {co_abundance}")
                    print(f"final Tg = {cell.Tg}")
                    print("++++++++++++++++++++++++\n")

                    success = True
                    print(f"guess T = {guess} successed at ({i}, {j})")
                    working += 1
                    break

                except Exception:
                    print(f"guess T = {guess} failed at ({i}, {j})")
                    continue

            if not success:
                print(f"Cell ({i}, {j}) failed for all guesses.")

    end_time = time.time()
    total_time = end_time - start_time
    num_pixels = shape[0] * shape[1]
    time_per_pixel = total_time / num_pixels if num_pixels > 0 else 0

    print(f"\n--- DESPOTIC run complete ---")
    print(f"Processed {num_pixels} pixels in {total_time:.2f} seconds.")
    print(f"Average time per pixel: {time_per_pixel*1000:.2f} ms.")
    print(f"Tg_working = {Tg_working}")
    print(f"working times = {working}")
    return co_line_map, Tg_map_final


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
