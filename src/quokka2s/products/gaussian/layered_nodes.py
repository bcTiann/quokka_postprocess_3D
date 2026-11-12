"""NumPy deposition and sorted storage for occupied (sigma layer,mu index) keys."""
from __future__ import annotations

import threading

import numpy as np

_KEY_DTYPE = np.dtype([("layer", np.int64), ("mu", np.int64)])
_INDEX_LIMIT = 9223372036854774784.0


def key_rows(keys):
    """View contiguous signed key pairs as sortable/searchable NumPy records."""
    return np.ascontiguousarray(keys).view(_KEY_DTYPE).reshape(-1)


def mean_brackets(centers, spacing, anchor):
    """Find bracketing integer means, correcting represented endpoint roundoff."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        coordinate = (centers - anchor) / spacing
    if not np.isfinite(coordinate).all() or np.any(np.abs(coordinate) >= _INDEX_LIMIT):
        raise ValueError("A layered mean index is outside representable integer capacity")
    lower = np.floor(coordinate).astype(np.int64)
    left = anchor + lower * spacing
    right = anchor + (lower + 1) * spacing
    while True:
        if not np.isfinite(left).all() or not np.isfinite(right).all() or np.any(right <= left):
            raise ValueError("Adjacent layered mean coordinates are indistinguishable")
        below = centers < left
        above = centers >= right
        if not np.any(below | above):
            break
        lower[below] -= 1
        lower[above] += 1
        left = anchor + lower * spacing
        right = anchor + (lower + 1) * spacing
    previous = anchor + (lower - 1) * spacing
    following = anchor + (lower + 2) * spacing
    if not np.isfinite(previous).all() or not np.isfinite(following).all() or np.any(previous >= left) or np.any(following <= right):
        raise ValueError("Neighbouring layered means are not distinguishable")
    fraction = (centers - left) / (right - left)
    return lower, fraction


def deposit_layered(grid, centers, sigmas, luminosity):
    """Select active rows, find four layer-specific corners and scatter light.

    Four fixed corner passes use np.add.at in original cell/profile order.
    Logical extent never allocates a rectangular histogram.
    """
    if not np.isfinite(luminosity).all() or np.any(luminosity < 0.0):
        raise ValueError("Layered luminosities must be finite and nonnegative")
    active = np.any(luminosity != 0.0, axis=1)
    centers = centers[active]
    sigmas = sigmas[active]
    light = luminosity[active]
    if not np.isfinite(centers).all() or not np.isfinite(sigmas).all() or np.any(sigmas < grid.sigma_floor):
        raise ValueError("Active layered means/widths must be finite and sigma >= the physical floor")
    if centers.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, luminosity.shape[1]))
    coordinate = (np.log(sigmas) - np.log(grid.sigma_floor)) / grid.log_sigma_step
    if not np.isfinite(coordinate).all() or np.any(coordinate < -1.0) or np.any(coordinate >= _INDEX_LIMIT):
        raise ValueError("A layered sigma index is unrepresentable")
    layers = np.maximum(np.floor(coordinate).astype(np.int64), 0)
    while True:
        sigma_lower = grid.sigma_values(layers)
        sigma_upper = grid.sigma_values(layers + 1)
        if np.any(sigma_upper <= sigma_lower):
            raise ValueError("Adjacent sigma layers are indistinguishable")
        below = (sigmas < sigma_lower) & (layers > 0)
        above = sigmas >= sigma_upper
        if not np.any(below | above):
            break
        layers[below] -= 1
        layers[above] += 1
    beta = np.zeros(sigmas.size)
    between = sigmas != sigma_lower
    beta[between] = np.log(sigmas[between] / sigma_lower[between]) / np.log(sigma_upper[between] / sigma_lower[between])
    if not np.isfinite(beta).all() or np.any(beta < 0.0) or np.any(beta > 1.0):
        raise ValueError("Invalid layered sigma interpolation fraction")
    low_ids, low_fraction = mean_brackets(centers, grid.mu_spacing_ratio * sigma_lower, grid.mu_anchor)
    high_ids, high_fraction = mean_brackets(centers, grid.mu_spacing_ratio * sigma_upper, grid.mu_anchor)
    corners = (
        np.column_stack((layers, low_ids)),
        np.column_stack((layers, low_ids + 1)),
        np.column_stack((layers + 1, high_ids)),
        np.column_stack((layers + 1, high_ids + 1)),
    )
    fractions = (
        (1.0 - beta) * (1.0 - low_fraction),
        (1.0 - beta) * low_fraction,
        beta * (1.0 - high_fraction),
        beta * high_fraction,
    )
    unique = np.unique(np.concatenate([key_rows(keys[fraction != 0.0]) for keys, fraction in zip(corners, fractions)]))
    weights = np.zeros((unique.size, luminosity.shape[1]))
    for keys, fraction in zip(corners, fractions):
        retained = fraction != 0.0
        positions = np.searchsorted(unique, key_rows(keys[retained]))
        np.add.at(weights, positions, light[retained] * fraction[retained, None])
    if not np.isfinite(weights).all():
        raise FloatingPointError("Layered histogram luminosity accumulation overflowed")
    retained = np.any(weights != 0.0, axis=1)
    keys = unique.view(np.int64).reshape(-1, 2)
    return np.ascontiguousarray(keys[retained]), np.ascontiguousarray(weights[retained])


class SparseNodeStore:
    """Keep only occupied sorted keys and forty accounts, merged on one thread."""

    def __init__(self, profile_count=40):
        if type(profile_count) is not int or not 1 <= profile_count <= 40:
            raise ValueError("profile_count must be an integer from one to forty")
        self.profile_count = profile_count
        self.keys = np.empty((0, 2), dtype=np.int64)
        self.weights = np.empty((0, profile_count))
        self.owner_thread = threading.get_ident()
        self.merge_calls = 0
        self.increment_rows = 0

    def merge(self, keys, profile_ids, weights):
        """Union sorted keys and add one increment in original batch order."""
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("A sparse node store belongs to one main thread")
        keys = np.asarray(keys)
        profile_ids = np.asarray(profile_ids)
        weights = np.asarray(weights, dtype=np.float64)
        if keys.ndim != 2 or keys.shape[1] != 2 or keys.dtype != np.int64 or profile_ids.ndim != 1 or profile_ids.dtype != np.int64:
            raise ValueError("Sparse keys must be int64[N,2] and profile IDs an int64 vector")
        if weights.shape != (keys.shape[0], profile_ids.size):
            raise ValueError("Sparse increment requires keys[U,2] and weights[U,P]")
        records = key_rows(keys)
        increasing = (keys[1:, 0] > keys[:-1, 0]) | ((keys[1:, 0] == keys[:-1, 0]) & (keys[1:, 1] > keys[:-1, 1]))
        if np.any(keys[:, 0] < 0) or not np.all(increasing):
            raise ValueError("Sparse logical keys must be nonnegative-layer, sorted and unique")
        if np.any(profile_ids < 0) or np.any(profile_ids >= self.profile_count) or np.unique(profile_ids).size != profile_ids.size:
            raise ValueError("Sparse profile IDs must be in bounds and unique")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("Sparse luminosities must be finite and nonnegative")
        if np.shares_memory(keys, self.keys) or np.shares_memory(weights, self.weights):
            raise ValueError("Sparse increments must not alias the main node store")
        if keys.shape[0] == 0 or profile_ids.size == 0:
            return
        old_records = key_rows(self.keys)
        combined = np.union1d(old_records, records)
        combined_weights = np.zeros((combined.size, self.profile_count))
        combined_weights[np.searchsorted(combined, old_records)] = self.weights
        positions = np.searchsorted(combined, records)
        combined_weights[np.ix_(positions, profile_ids)] += weights
        self.keys = np.ascontiguousarray(combined.view(np.int64).reshape(-1, 2))
        self.weights = combined_weights
        self.merge_calls += 1
        self.increment_rows += keys.shape[0]

    def arrays(self):
        if not np.isfinite(self.weights).all():
            raise FloatingPointError("Accumulated node luminosity is nonfinite")
        retained = np.any(self.weights != 0.0, axis=1)
        return self.keys[retained].copy(), self.weights[retained].copy()

    def statistics(self):
        return {
            "implementation": "numpy",
            "occupied_nodes": self.keys.shape[0],
            "key_bytes": self.keys.nbytes,
            "payload_bytes": self.weights.nbytes,
            "allocated_bytes": self.keys.nbytes + self.weights.nbytes,
            "merge_calls": self.merge_calls,
            "increment_rows": self.increment_rows,
            "selected_threads": 1,
            "logical_enclosing_grid_allocated": False,
            "memory_scope": "Current owned key/luminosity buffers, not RSS or union temporary peak",
        }
