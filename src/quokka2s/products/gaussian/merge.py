"""Merge unique node/profile increments with NumPy indexed addition."""
from __future__ import annotations

import numpy as np


class NodeMajorMerger:
    """Finish each original batch/group on the main thread before its successor."""

    def __init__(self, *, threads=None):
        if threads is not None and (type(threads) is not int or threads != 1):
            raise ValueError("NumPy histogram merging supports histogram_merge_threads=None or 1")
        self.threads = 1
        self.statistics = {
            "implementation": "numpy",
            "selected_threads": 1,
            "automatic_selection": threads is None,
            "merge_calls": 0,
            "increment_rows": 0,
            "batch_group_order_unchanged": True,
            "dense_H_layout_unchanged": True,
            "per_location_arithmetic_unchanged": True,
        }

    def merge(self, target, nodes, profile_ids, weights):
        """Add weights[N,P] once at unique target[node,profile] locations."""
        nodes = np.asarray(nodes)
        profile_ids = np.asarray(profile_ids)
        weights = np.asarray(weights)
        if target.ndim != 2 or target.shape[1] != 40 or target.dtype != np.float64:
            raise ValueError("The accepted spectral accumulator requires float64 target[Nnode,40]")
        if nodes.ndim != 1 or profile_ids.ndim != 1 or nodes.dtype != np.int64 or profile_ids.dtype != np.int64:
            raise ValueError("Node/profile IDs must be int64 vectors")
        if weights.shape != (nodes.size, profile_ids.size):
            raise ValueError("Sparse weights must match node/profile dimensions")
        if np.any(nodes < 0) or np.any(nodes >= target.shape[0]) or np.any(nodes[1:] <= nodes[:-1]):
            raise ValueError("Sparse nodes must be in bounds, increasing and unique")
        if np.any(profile_ids < 0) or np.any(profile_ids >= 40) or np.unique(profile_ids).size != profile_ids.size:
            raise ValueError("Sparse profile IDs must be in bounds and unique")
        if any(np.shares_memory(target, values) for values in (nodes, profile_ids, weights)):
            raise ValueError("Sparse merge inputs must not alias target")
        target[np.ix_(nodes, profile_ids)] += weights
        self.statistics["merge_calls"] += 1
        self.statistics["increment_rows"] += int(nodes.size)
