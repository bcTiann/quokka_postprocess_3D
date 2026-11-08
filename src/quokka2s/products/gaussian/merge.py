"""Owned node-major merge with verified optional OpenMP support."""
from __future__ import annotations

from ._histogram_merge import kernel_info, merge_node_major


class NodeMajorMerger:
    """Merge each original batch/group before starting its successor.

    Distinct node rows are independent; each node/profile retains one ordinary
    float64 addition per increment. This object belongs to the main accumulator,
    so workers neither create OpenMP teams nor allocate a dense histogram.
    """

    def __init__(self, *, threads=None):
        if threads is not None and (type(threads) is not int or threads < 1):
            raise ValueError("histogram_merge_threads must be a positive integer or None")
        requested = 4 if threads is None else threads
        info = kernel_info(requested)
        if requested > 1 and info["observed_team_size"] != requested:
            if threads is not None:
                raise RuntimeError(
                    "The installed histogram merge binary/runtime cannot create "
                    f"the explicitly requested OpenMP team of {threads} threads"
                )
            requested = 1
            info = kernel_info(requested)
        self.threads = requested
        self.statistics = {
            **info,
            "selected_threads": requested,
            "automatic_selection": threads is None,
            "prefetch_distance_rows": 32,
            "parallel_threshold_rows": 16384,
            "merge_calls": 0,
            "increment_rows": 0,
            "batch_group_order_unchanged": True,
            "dense_H_layout_unchanged": True,
            "per_location_arithmetic_unchanged": True,
        }

    def merge(self, target, nodes, profile_ids, weights):
        """Update target[Nnode,40] in place and finish before returning."""
        merge_node_major(
            target,
            nodes,
            profile_ids,
            weights,
            threads=self.threads,
            prefetch_distance=32,
            parallel_threshold_rows=16384,
        )
        self.statistics["merge_calls"] += 1
        self.statistics["increment_rows"] += int(nodes.size)
