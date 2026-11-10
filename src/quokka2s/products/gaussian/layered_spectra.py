"""Preserve cell products while storing only occupied sigma-layer nodes."""
from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import numpy as np

from . import _layered_deposit as native
from .accounting import add_cutoff_accounts, update_cutoff_report
from .spectra import HistogramIntegratedSpectra
from ..integrated_spectra import IntegratedSpectra


class LayeredIntegratedSpectra(HistogramIntegratedSpectra):
    """Reuse emissivity masks/full moments and replace only histogram storage."""

    def __init__(self, line_keys, velocity_edges_kms, *, engine, batch_partial=False,
                 workers=1, cell_chunk=8192, merge_threads=None):
        if merge_threads is not None and (type(merge_threads) is not int or merge_threads != 1):
            raise ValueError("The layered node store supports one main merge thread")
        IntegratedSpectra.__init__(
            self,
            line_keys=line_keys,
            velocity_edges_kms=velocity_edges_kms,
            workers=workers,
            cell_chunk=cell_chunk,
            spectral_method="histogram-layered",
        )
        self.engine = engine
        self.batch_partial = batch_partial
        self.profile_count = 2 * len(self.line_keys) * 2
        if self.profile_count != 40:
            raise ValueError("The layered process retains the adopted ten-line 40-account layout")
        self._store = None if batch_partial else native.SparseNodeStore(profile_count=self.profile_count)
        self._sparse_updates = []
        self._histogram = None
        self._finalized = False
        self._fallback_input = np.zeros(self.profile_count, dtype=np.float64)
        self._registered_input = np.zeros(self.profile_count, dtype=np.float64)
        self.histogram_statistics = {
            "deposit_seconds": 0.0,
            "sparse_merge_seconds": 0.0,
            "fallback_seconds": 0.0,
            "expansion_seconds": 0.0,
            "spectral_add_batch_seconds": 0.0,
            "kernel_rows": 0,
            "fallback_rows": 0,
            "histogram_updates": 0,
        }
        self._merge_statistics = {
            "selected_threads": 1,
            "automatic_selection": merge_threads is None,
            "batch_group_order_unchanged": True,
            "per_location_arithmetic_unchanged": True,
            "merge_calls": 0,
            "increment_rows": 0,
            "coordinate_keys": "signed int64 (sigma_layer_id,mu_index)",
            "storage": "compiled hash index and contiguous occupied-node luminosity rows",
        }

    def _merge_sparse(self, profile_ids, nodes, weights):
        """Finish each original batch/group update before its successor starts."""
        if nodes.shape[0] == 0:
            return
        if self._store is None:
            raise RuntimeError("Worker partials cannot merge into a global layered store")
        began = perf_counter()
        self._store.merge(nodes, profile_ids, weights)
        self.histogram_statistics["sparse_merge_seconds"] += perf_counter() - began
        self._merge_statistics["merge_calls"] += 1
        self._merge_statistics["increment_rows"] += int(nodes.shape[0])

    def merge(self, other):
        """Merge numerical cell products and logical-node batches in input order."""
        if self.batch_partial or self._finalized or other._finalized:
            raise RuntimeError("Merge unfinalized layered worker partials into the main accumulator")
        if self.engine.grid != other.engine.grid or self.line_keys != other.line_keys:
            raise ValueError("Cannot merge different layered coordinate rules or line orders")
        IntegratedSpectra.merge(self, other)
        for name, value in other.histogram_statistics.items():
            self.histogram_statistics[name] += value
        for profile_ids, nodes, weights in other._sparse_updates:
            self._merge_sparse(profile_ids, nodes, weights)
        if other._store is not None:
            raise RuntimeError("A layered worker partial must not own a global node store")
        self._registered_input += other._registered_input
        self._fallback_input += other._fallback_input
        return self

    def build_output(self, projected_area_cm2):
        """Finalize sorted occupied nodes and preserve the numerical product schema."""
        if self.batch_partial:
            raise RuntimeError("Merge layered worker partials before building output")
        if not self._finalized:
            began = perf_counter()
            keys, weights = self._store.arrays()
            self.histogram_statistics["node_sort_and_gather_seconds"] = perf_counter() - began
            expected = self.input_luminosity.reshape(-1)
            histogram_light = weights.sum(axis=0)
            accounted = histogram_light + self._fallback_input
            if not np.allclose(accounted, expected, rtol=1e-10, atol=0.0):
                raise AssertionError("Layered nodes plus explicit fallback lost input luminosity")
            if not np.allclose(histogram_light, self._registered_input, rtol=1e-10, atol=0.0):
                raise AssertionError("Sparse logical-node merging lost deposited luminosity")
            relative = np.divide(
                np.abs(accounted - expected),
                expected,
                out=np.zeros_like(expected),
                where=expected > 0.0,
            )
            self.histogram_statistics["maximum_input_conservation_relative_error"] = float(relative.max(initial=0.0))
            self.histogram_statistics["histogram_luminosity_by_profile"] = histogram_light.tolist()
            self.histogram_statistics["fallback_luminosity_by_profile"] = self._fallback_input.tolist()
            self.histogram_statistics["input_luminosity_by_profile"] = expected.tolist()
            self.histogram_statistics["occupied_nodes"] = int(keys.shape[0])
            self.histogram_statistics["main_node_store"] = self._store.statistics()
            self.histogram_statistics["worker_node_stores"] = self.engine.worker_statistics()
            # arrays() returns owned compact copies. Release the hash and its
            # capacity buffers before allocating final integration scratch.
            self._store = None
            began = perf_counter()
            expanded = self.engine.expand(keys, weights, self.velocity_edges_kms)
            self.dL_dv += expanded.T.reshape(self.dL_dv.shape)
            self.histogram_statistics["expansion_seconds"] += perf_counter() - began
            self._finalized = True
        payload, report = IntegratedSpectra.build_output(self, projected_area_cm2=projected_area_cm2)
        for dust_report in report.values():
            dust_report["histogram_grid"] = asdict(self.engine.grid)
            dust_report["histogram_coordinate_rule"] = self.engine.coordinate_rule
            dust_report["template_contraction"] = self.engine.expansion_statistics["contraction"]
            dust_report["template_expansion_statistics"] = self.engine.expansion_statistics
            dust_report["deposition_backend"] = "compiled layered corner-major scatter and sparse integer-key hash"
            dust_report["merge_backend"] = self._merge_statistics
            dust_report["histogram_statistics"] = self.histogram_statistics
            dust_report["fallback"] = "None: all active rows use fixed layer coordinates; unrepresentable nodes raise"
        audit = add_cutoff_accounts(payload, engine=self.engine, cutoff_sigma=5.0)
        update_cutoff_report(report, payload, engine=self.engine, audit=audit)
        return payload, report
