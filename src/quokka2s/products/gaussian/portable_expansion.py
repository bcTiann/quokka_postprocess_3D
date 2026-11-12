"""Expand A-template node luminosities using the original NumPy contraction."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

from ...file_provenance import file_sha256
from .engine import GaussianHistogramEngine as OriginalABCEngine

BACKENDS = ("numpy_einsum",)


def array_sha256(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def runtime_signature():
    return {
        "implementation": "numpy", "workers": 1,
        "system": platform.system(), "machine": platform.machine(),
        "python_version": list(sys.version_info[:3]),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "A_source_sha256": file_sha256(Path(__file__).with_name("a_templates.py")),
        "contraction_source_sha256": file_sha256(Path(__file__).with_name("engine.py")),
        "custom_compiled_kernel": False,
    }


def band_geometry(engines):
    return [{
        "mu_nodes": int(band.mu_nodes.size), "sigma_nodes": int(band.sigma_nodes.size),
        "mu_sha256": array_sha256(band.mu_nodes), "sigma_sha256": array_sha256(band.sigma_nodes),
        "log_sigma_sha256": array_sha256(band.log_sigma_nodes),
        "template_scratch_bytes": int(band.template_scratch_bytes),
    } for band in engines]


def validate_gate(result, grid, engines):
    expected = {
        "schema_version": 2, "complete": True, "selected_backend": "numpy_einsum",
        "supported_profiles": 40, "cutoff_sigma": 5.0,
        "runtime_signature": runtime_signature(), "grid": asdict(grid),
        "table_geometry": band_geometry(engines),
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("A-template validation belongs to different NumPy code or geometry")
    if not result.get("cases") or not all(case["passed"] for case in result["cases"]):
        raise ValueError("Independent A-template interpolation fixtures did not pass")
    return "numpy_einsum"


class GaussianHistogramEngine(OriginalABCEngine):
    """Bind immutable A-table geometry, then use NumPy einsum for every block."""

    def __init__(self, grid, *, contraction="einsum", gate_json=None, gate_result=None):
        if (gate_json is None) == (gate_result is None):
            raise ValueError("Supply exactly one completed gate_json or gate_result")
        if gate_json is not None:
            gate_result = json.loads(Path(gate_json).read_text())
        super().__init__(grid, contraction=contraction)
        self.portable_backend = validate_gate(gate_result, grid, self.engines)
        self._gate_edges_sha256 = gate_result["edges_sha256"]
        self.expansion_statistics.update({"implementation": "numpy", "portable_backend": self.portable_backend, "portable_gate": gate_result})

    def _expand_band(self, band_index, nodes, weights, edges):
        if self.lookup_tables is None or self.lookup_tables.method != "A":
            raise ValueError("A expansion requires its prepared integral-template bundle")
        if self.lookup_tables.cutoff_sigma != 5.0 or self.gaussian_cutoff_sigma != 5.0:
            raise ValueError("The five-sigma support rule cannot change after preparation")
        if array_sha256(edges) != self._gate_edges_sha256:
            raise ValueError("Velocity edges differ from the validated A table")
        return super()._expand_band(band_index, nodes, weights, edges)
