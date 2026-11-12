"""A templates: fractional-mu by actual-sigma templates clipped at fixed standardized support, shifted by whole bins.

The permanent table T[S,R+1,M_A] contains integrated bin probabilities, not CDF
samples. Evaluation only reads/interpolates table rows and never queries a CDF.
"""
from __future__ import annotations

import math
import operator
import time

import numpy as np

from .intervals import VelocityGrid, interval_probability, validate_inputs


_BUILD_SCRATCH_BYTES = 32 * 1024**2


class MethodA:
    """Tabulate fractional shifts, then shift each interpolated template slice.

    sigma_nodes are positive standard deviations in the velocity unit. Their
    interpolation weights are linear in sigma itself, including when their
    placement is geometric. R subdivides one channel into R fractional shifts.
    All evaluation coordinates must lie inside the configured closed bounds.
    """

    def __init__(self, grid=None, *, sigma_nodes=None, mu_min=-4.0, mu_max=4.0, R=10, cutoff_sigma=5.0):
        began = time.perf_counter()
        self.grid = VelocityGrid() if grid is None else grid
        if not isinstance(self.grid, VelocityGrid):
            raise TypeError("grid must be a VelocityGrid")
        if self.grid.K < 1:
            raise ValueError("The velocity grid must contain at least one channel")
        if isinstance(R, (bool, np.bool_)):
            raise TypeError("R must be a positive integer")
        try:
            R = operator.index(R)
        except TypeError as error:
            raise TypeError("R must be a positive integer") from error
        if R < 1:
            raise ValueError("R must be a positive integer")
        self.R = R
        self.cutoff_sigma = None if cutoff_sigma is None else float(cutoff_sigma)
        if self.cutoff_sigma is not None:
            if not np.isfinite(self.cutoff_sigma) or self.cutoff_sigma <= 0.0:
                raise ValueError("cutoff_sigma must be finite and positive, or None")
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        if not np.isfinite(self.mu_min) or not np.isfinite(self.mu_max) or self.mu_min > self.mu_max:
            raise ValueError("Require finite mu_min <= mu_max")
        if sigma_nodes is None:
            sigma_nodes = np.geomspace(0.1, 4.0, 65)
        nodes = np.array(sigma_nodes, dtype=np.float64, order="C", copy=True)
        if nodes.ndim != 1 or nodes.size < 1:
            raise ValueError("sigma_nodes must be a nonempty one-dimensional array")
        if not np.isfinite(nodes).all() or np.any(nodes <= 0.0) or np.any(np.diff(nodes) <= 0.0):
            raise ValueError("sigma_nodes must be finite, positive and strictly increasing")
        nodes.flags.writeable = False
        self.sigma_nodes = nodes
        self.n_min = math.floor(self.mu_min / self.grid.delta_v)
        self.n_max = math.floor(self.mu_max / self.grid.delta_v)
        self.L = self.grid.v_min - self.n_max * self.grid.delta_v
        self.M_A = self.grid.K + self.n_max - self.n_min
        shape = (nodes.size, R + 1, self.M_A)
        if math.prod(shape) > np.iinfo(np.intp).max // np.dtype(np.float64).itemsize:
            raise ValueError("Requested template dimensions exceed the platform index range")
        self.T = np.empty(shape, dtype=np.float64, order="C")
        template_edges = self.L + self.grid.delta_v * np.arange(self.M_A + 1, dtype=np.float64)
        row_count = nodes.size * (R + 1)
        # The normalized edges plus masks/probability temporaries stay bounded;
        # the 512-MiB target table is never duplicated during construction.
        bytes_per_row = max(1, (self.M_A + 1) * 8 * 8)
        build_chunk = min(16384, max(1, _BUILD_SCRATCH_BYTES // bytes_per_row))
        rows = self.T.reshape(row_count, self.M_A)
        for start in range(0, row_count, build_chunk):
            stop = min(start + build_chunk, row_count)
            ids = np.arange(start, stop, dtype=np.int64)
            sigma = nodes[ids // (R + 1)]
            fractional_mu = (ids % (R + 1)) / R * self.grid.delta_v
            z = (template_edges[None, :] - fractional_mu[:, None]) / sigma[:, None]
            if self.cutoff_sigma is not None:
                # Clip each endpoint, retaining only the inside portion of a
                # channel crossing support. There is no finite-mass renormalization.
                np.clip(z, -self.cutoff_sigma, self.cutoff_sigma, out=z)
            probabilities = interval_probability(z[:, :-1], z[:, 1:])
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
                raise FloatingPointError("Template probabilities must be finite and nonnegative")
            rows[start:stop] = probabilities
        self.T.flags.writeable = False
        self.build_chunk_rows = build_chunk
        self.build_seconds = time.perf_counter() - began
        self.last_evaluation_sigma_mode = None

    @property
    def fraction_nodes(self):
        """Return the R+1 fractional-channel nodes; this is not a stored buffer."""
        return np.arange(self.R + 1, dtype=np.float64) / self.R

    def evaluate(self, mu, sigma, *, sigma_id=None, chunk_size=16384):
        """Return C-contiguous bin masses[N,K], including validation/allocation.

        sigma_id optionally selects an exact stored sigma node per row. It may
        be an integer scalar or integer vector[N]; sigma must exactly equal the
        indicated node value. This explicitly labelled fast path skips searching
        sigma but still validates the supplied IDs inside the timed call.
        """
        mu, sigma = validate_inputs(mu, sigma)
        if np.any(mu < self.mu_min) or np.any(mu > self.mu_max):
            raise ValueError("mu is outside the configured closed bounds")
        if np.any(sigma < self.sigma_nodes[0]) or np.any(sigma > self.sigma_nodes[-1]):
            raise ValueError("sigma is outside the stored sigma-node bounds")
        if isinstance(chunk_size, (bool, np.bool_)):
            raise TypeError("chunk_size must be a positive integer")
        try:
            chunk_size = operator.index(chunk_size)
        except TypeError as error:
            raise TypeError("chunk_size must be a positive integer") from error
        if chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        has_sigma_id = sigma_id is not None
        if has_sigma_id:
            ids = np.asarray(sigma_id)
            if not np.issubdtype(ids.dtype, np.integer):
                raise TypeError("sigma_id must contain integers")
            if ids.ndim == 0:
                ids = np.full(mu.size, ids.item(), dtype=ids.dtype)
            if ids.shape != mu.shape:
                raise ValueError("sigma_id must be an integer scalar or a vector matching mu")
            if np.any(ids < 0) or np.any(ids >= self.sigma_nodes.size):
                raise ValueError("sigma_id is outside the stored sigma-node indices")
            ids = np.ascontiguousarray(ids, dtype=np.int64)
            if not np.array_equal(sigma, self.sigma_nodes[ids]):
                raise ValueError("sigma_id does not exactly match the supplied sigma values")
            self.last_evaluation_sigma_mode = "validated_exact_sigma_id"
        else:
            ids = np.empty(0, dtype=np.int64)
            self.last_evaluation_sigma_mode = "sigma_search_linear_actual_value"
        output = np.empty((mu.size, self.grid.K), dtype=np.float64, order="C")
        chunk_size = min(chunk_size, max(1, _BUILD_SCRATCH_BYTES // (self.grid.K * 8 * 4)))
        for start in range(0, mu.size, chunk_size):
            stop = min(start + chunk_size, mu.size)
            coordinate = mu[start:stop] / self.grid.delta_v
            integer_mu = np.floor(coordinate).astype(np.int64)
            fractional_coordinate = (coordinate - integer_mu) * self.R
            fraction_lower = np.minimum(np.floor(fractional_coordinate).astype(np.int64), self.R)
            fraction_upper = np.minimum(fraction_lower + 1, self.R)
            mu_weight = fractional_coordinate - fraction_lower
            exact_lower = coordinate == integer_mu + fraction_lower / self.R
            exact_upper = coordinate == integer_mu + fraction_upper / self.R
            fraction_lower = np.where(exact_upper, fraction_upper, fraction_lower)
            fraction_upper = np.where(exact_lower, fraction_lower, fraction_upper)
            mu_weight[exact_lower | exact_upper] = 0.0
            if has_sigma_id:
                sigma_lower = ids[start:stop]
                sigma_upper = sigma_lower
                sigma_weight = np.zeros(stop - start)
            else:
                sigma_upper = np.searchsorted(self.sigma_nodes, sigma[start:stop], side="left")
                exact_sigma = self.sigma_nodes[sigma_upper] == sigma[start:stop]
                sigma_lower = np.where(exact_sigma, sigma_upper, sigma_upper - 1)
                sigma_weight = np.zeros(stop - start)
                between = ~exact_sigma
                sigma_weight[between] = (
                    (sigma[start:stop][between] - self.sigma_nodes[sigma_lower[between]])
                    / (self.sigma_nodes[sigma_upper[between]] - self.sigma_nodes[sigma_lower[between]])
                )
            channels = self.n_max - integer_mu[:, None] + np.arange(self.grid.K)
            lower = self.T[sigma_lower[:, None], fraction_lower[:, None], channels]
            interpolate_mu = fraction_lower != fraction_upper
            if np.any(interpolate_mu):
                upper = self.T[sigma_lower[interpolate_mu, None], fraction_upper[interpolate_mu, None], channels[interpolate_mu]]
                lower[interpolate_mu] = (
                    (1.0 - mu_weight[interpolate_mu, None]) * lower[interpolate_mu]
                    + mu_weight[interpolate_mu, None] * upper
                )
            interpolate_sigma = sigma_lower != sigma_upper
            if np.any(interpolate_sigma):
                upper = self.T[sigma_upper[:, None], fraction_lower[:, None], channels]
                if np.any(interpolate_mu):
                    upper_mu = self.T[sigma_upper[interpolate_mu, None], fraction_upper[interpolate_mu, None], channels[interpolate_mu]]
                    upper[interpolate_mu] = (
                        (1.0 - mu_weight[interpolate_mu, None]) * upper[interpolate_mu]
                        + mu_weight[interpolate_mu, None] * upper_mu
                    )
                lower[interpolate_sigma] = (
                    (1.0 - sigma_weight[interpolate_sigma, None]) * lower[interpolate_sigma]
                    + sigma_weight[interpolate_sigma, None] * upper[interpolate_sigma]
                )
            output[start:stop] = lower
        return output

    def info(self):
        """Return construction cost and exact owned persistent NumPy bytes."""
        return {
            "method": "A_fractional_mu_sigma_shift",
            "implementation": "numpy",
            "build_seconds": float(self.build_seconds),
            "table_bytes": int(self.T.nbytes),
            "axis_bytes": int(self.sigma_nodes.nbytes),
            "memory_bytes": int(self.T.nbytes + self.sigma_nodes.nbytes),
            "memory_scope": "Owned persistent NumPy buffers T and sigma_nodes; views counted once",
            "last_evaluation_sigma_mode": self.last_evaluation_sigma_mode,
            "configuration": {
                "v_min": float(self.grid.v_min),
                "v_max": float(self.grid.v_max),
                "delta_v": float(self.grid.delta_v),
                "K": int(self.grid.K),
                "mu_min": self.mu_min,
                "mu_max": self.mu_max,
                "n_min": self.n_min,
                "n_max": self.n_max,
                "L": float(self.L),
                "M_A": self.M_A,
                "R": self.R,
                "fraction_node_count": self.R + 1,
                "sigma_node_count": int(self.sigma_nodes.size),
                "sigma_min": float(self.sigma_nodes[0]),
                "sigma_max": float(self.sigma_nodes[-1]),
                "T_shape": list(self.T.shape),
                "table_order": "C; last extended-channel axis contiguous",
                "sigma_interpolation": "linear_actual_sigma",
                "fraction_interpolation": "linear_fractional_channel",
                "outside_behavior": "raise; no silent clipping or extrapolation",
                "sigma_id_supported": True,
                "build_chunk_rows": self.build_chunk_rows,
                "build_scratch_budget_bytes": _BUILD_SCRATCH_BYTES,
                "online_cdf_queries": False,
                "cutoff_sigma": self.cutoff_sigma,
                "cutoff_mode": "clip standardized channel endpoints during construction",
                "renormalized": False,
            },
        }
