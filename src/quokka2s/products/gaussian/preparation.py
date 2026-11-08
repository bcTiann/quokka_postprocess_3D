"""Prepare one standard-Gaussian prefix table before process timing."""
from time import perf_counter

import numpy as np

from .engine import GaussianHistogramEngine as GeometryEngine
from .prefix_engine import GaussianHistogramEngine
from .prefix_arithmetic import run_gate
from .geometry import GridSpec
from .prefix_tables import PrefixGaussianTable
from ..velocity_grid import velocity_channel_edges_kms


DEFAULT_GRID = {
    'mu_min': -3072.0,
    'mu_max': 3328.0,
    'mu_step': 0.125,
    'sigma_min': 0.025,
    'sigma_max': 8192.0,
    'log_sigma_step': 0.03125,
    'mu_core_min': -256.0,
    'mu_core_max': 256.0,
    'mu_wing_step': 0.5,
    'narrow_sigma_threshold': 0.5,
    'narrow_mu_step': 0.015625,
}


def prepare_gaussian_engine(grid_parameters=None):
    """Return one fresh 2D accumulator engine and its separate preparation cost.

    Bounds, spacings, means and widths use km/s. No snapshot fields or emission
    tables are read here. Every emitting cell must later fit this grid; outside
    values raise rather than clip, disappear, or invoke a 1D spectrum kernel.
    """
    began = perf_counter()
    values = DEFAULT_GRID if grid_parameters is None else grid_parameters
    grid = GridSpec(**values)
    edges = velocity_channel_edges_kms()
    geometry = GeometryEngine(grid, contraction='einsum')
    table = PrefixGaussianTable(edges=edges)
    gate = run_gate(grid, edges, engines=geometry.engines, table=table)
    engine = GaussianHistogramEngine(grid, table=table, gate_result=gate)
    if any(not np.array_equal(actual.mu_nodes, prepared.mu_nodes)
           or not np.array_equal(actual.sigma_nodes, prepared.sigma_nodes)
           for actual, prepared in zip(engine.engines, geometry.engines)):
        raise AssertionError('Prepared prefix gate and final Gaussian axes differ')
    return engine, perf_counter() - began
