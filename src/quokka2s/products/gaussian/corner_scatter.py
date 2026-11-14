"""Guarded original-corner accumulation through SciPy's prebuilt private COO API.

The project never compiles this kernel. Process initialization pins SciPy
1.16.3 and verifies nonzero-output addition. Generic imports and plot readers
perform no backend initialization. Trusted wave-local buffers are validated
before any unchecked library pointers are used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
from time import perf_counter

import numpy as np


_PREPARATION_LOCK = threading.Lock()
_COO_MATVEC = None
_BACKEND_INFO = None


def prepare_corner_scatter():
    """Validate the private prebuilt backend once; return disclosed metadata."""
    global _COO_MATVEC, _BACKEND_INFO
    if _COO_MATVEC is None:
        with _PREPARATION_LOCK:
            if _COO_MATVEC is None:
                import scipy
                from scipy.sparse import _sparsetools

                if scipy.__version__ != "1.16.3":
                    raise RuntimeError("Corner scattering requires exactly SciPy 1.16.3")
                kernel = getattr(_sparsetools, "coo_matvec", None)
                if kernel is None:
                    raise RuntimeError("Pinned SciPy build has no private coo_matvec")
                began = perf_counter()
                rows = np.array([0, 1, 0], dtype=np.int32)
                cells = np.array([0, 1, 0], dtype=np.int32)
                weights = np.array([0.5, 2.0, 0.25])
                values = np.array([2.0, 4.0])
                output = np.array([3.0, 5.0])
                returned = kernel(3, rows, cells, weights, values, output)
                if (
                    returned is not None
                    or output.tobytes() != np.array([4.5, 13.0]).tobytes()
                ):
                    raise RuntimeError("SciPy COO kernel failed its nonzero-output semantic probe")
                _BACKEND_INFO = {
                    "private_api": "scipy.sparse._sparsetools.coo_matvec",
                    "scipy_version_pin": scipy.__version__,
                    "prebuilt_module": _sparsetools.__file__,
                    "semantic_probe_seconds": perf_counter() - began,
                    "project_compilation_required": False,
                    "buffer_contract": "Trusted owned read-only terms; bounded wave-local luminosity; disjoint writable profile planes",
                }
                _COO_MATVEC = kernel
    return dict(_BACKEND_INFO)


def _require_vector(name, values, dtype, *, writable=False):
    """Refuse buffers that could trigger implicit casts or output writebacks."""
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be an existing NumPy array")
    if values.ndim != 1 or values.dtype != dtype or not values.dtype.isnative:
        raise ValueError(f"{name} requires one-dimensional native {dtype}")
    if not values.flags.c_contiguous or not values.flags.aligned:
        raise ValueError(f"{name} must be aligned and C-contiguous")
    if writable and not values.flags.writeable:
        raise ValueError(f"{name} must be writable")


@dataclass(frozen=True)
class CornerTerms:
    """Validated immutable term vectors, safe for repeated disjoint profiles.

    E is the number of retained original-corner terms. destination_slots[E]
    and source_cells[E] use matching signed int32/int64 buffers; weights[E]
    uses native float64. A destination is a flat (template row, integer shift)
    slot in one H profile plane, not a template row number. A source is a local
    row in the selected luminosity[N,P], not a global snapshot cell ID.
    Each term adds weights[e] * X[source_cells[e]] to Y[destination_slots[e]].
    Existing Y is preserved, and explicit zeros are not removed.
    This is trusted wave-local ownership: owners must never reopen or mutate
    term buffers after construction, even between calls. Cheap metadata checks
    reject reopened/nonmatching buffers; they do not repeat bounds scans after
    arbitrary hostile mutation. Concurrent calls must have disjoint outputs.
    """

    destination_slots: np.ndarray
    source_cells: np.ndarray
    weights: np.ndarray
    input_length: int
    output_length: int
    _term_count: int = field(init=False, repr=False)
    _index_dtype: np.dtype = field(init=False, repr=False)

    def __post_init__(self):
        prepare_corner_scatter()
        if type(self.input_length) is not int or self.input_length < 0:
            raise ValueError("COO input length must be a nonnegative integer")
        if type(self.output_length) is not int or self.output_length < 0:
            raise ValueError("COO output length must be a nonnegative integer")
        if (
            not isinstance(self.destination_slots, np.ndarray)
            or self.destination_slots.dtype not in (np.dtype("int32"), np.dtype("int64"))
        ):
            raise ValueError("COO row indices require signed int32 or int64")
        _require_vector(
            name="destination_slots",
            values=self.destination_slots,
            dtype=self.destination_slots.dtype,
        )
        _require_vector(
            name="source_cells",
            values=self.source_cells,
            dtype=self.destination_slots.dtype,
        )
        _require_vector("weights", self.weights, np.dtype("float64"))
        if any(
            not values.flags.owndata
            for values in (self.destination_slots, self.source_cells, self.weights)
        ):
            raise ValueError("COO terms require independent owned buffers")
        if (
            self.destination_slots.size != self.source_cells.size
            or self.destination_slots.size != self.weights.size
        ):
            raise ValueError("COO term arrays must have equal lengths")
        if self.destination_slots.size > np.iinfo(np.int64).max:
            raise ValueError("COO term count exceeds the kernel's signed int64")
        if self.destination_slots.size:
            if (
                self.destination_slots.min() < 0
                or self.destination_slots.max() >= self.output_length
            ):
                raise ValueError("COO row index is outside the output")
            if self.source_cells.min() < 0 or self.source_cells.max() >= self.input_length:
                raise ValueError("COO cell index is outside the input")
        object.__setattr__(self, "_term_count", int(self.destination_slots.size))
        object.__setattr__(self, "_index_dtype", self.destination_slots.dtype)
        for values in (self.destination_slots, self.source_cells, self.weights):
            values.flags.writeable = False

    def accumulate(self, values, output):
        """Add one contiguous profile into existing writable output in place."""
        _require_vector(
            name="destination_slots",
            values=self.destination_slots,
            dtype=self._index_dtype,
        )
        _require_vector(
            name="source_cells",
            values=self.source_cells,
            dtype=self._index_dtype,
        )
        _require_vector("weights", self.weights, np.dtype("float64"))
        if any(
            values.flags.writeable or not values.flags.owndata
            or values.size != self._term_count
            for values in (self.destination_slots, self.source_cells, self.weights)
        ):
            raise ValueError("Trusted COO term buffers must remain owned, read-only and unchanged")
        _require_vector("values", values, np.dtype("float64"))
        _require_vector("output", output, np.dtype("float64"), writable=True)
        if values.size != self.input_length or output.size != self.output_length:
            raise ValueError("COO input/output lengths differ from validated bounds")
        if any(
            np.may_share_memory(output, source)
            for source in (values, self.destination_slots, self.source_cells, self.weights)
        ):
            raise ValueError("COO output cannot overlap a kernel input")
        _COO_MATVEC(
            int(self.destination_slots.size),
            self.destination_slots,
            self.source_cells,
            self.weights,
            values,
            output,
        )


@dataclass(frozen=True)
class CornerIncrement:
    """One spec-tagged checked operator and read-only borrowed L[N,P] for a wave.

    profiles[P] maps local luminosity columns to global spectrum accounts.
    Terms own their index/weight arrays; luminosity[N,P] is a borrowed view.
    Neither the view nor its source owner may be mutated until all profile
    writers have joined. The source is newly packed product data, never H.
    """

    terms: CornerTerms
    luminosity: np.ndarray
    profiles: np.ndarray
    touched_rows: np.ndarray
    spec: object


def scatter_corner_profiles(store, increments, executor):
    """Write disjoint H[profile] planes in original increment order per profile.

    Validation, touched-row updates, plan creation, every contiguous X copy and
    all private-library calls are included in this function's measured interval.
    The shared executor is the existing Gaussian pool.
    """
    began = perf_counter()
    light = store.light
    if getattr(store, "failed", False):
        raise RuntimeError("Cannot reuse an accumulator after failed raw scattering")
    if light is None or light.shape != (store.spec.profile_count, store.spec.slot_count):
        raise ValueError("Raw scattering requires the live matching ragged store")
    if (
        light.dtype != np.dtype("float64") or not light.dtype.isnative
        or not light.flags.c_contiguous or not light.flags.aligned
        or not light.flags.writeable
    ):
        raise ValueError("Ragged H must be native aligned writable C float64")
    plan = [[] for _ in range(store.spec.profile_count)]
    for increment in increments:
        if increment.spec is not store.spec:
            raise ValueError("Raw increment belongs to a different prepared corner spec")
        matrix = increment.luminosity
        if (
            not isinstance(matrix, np.ndarray) or matrix.ndim != 2
            or matrix.dtype != np.dtype("float64") or not matrix.dtype.isnative
            or not matrix.flags.c_contiguous or not matrix.flags.aligned
        ):
            raise ValueError("Raw luminosity requires native aligned C float64[N,P]")
        if matrix.flags.writeable:
            raise ValueError("Borrowed raw luminosity must remain read-only for the wave")
        if any(
            np.may_share_memory(buffer, light)
            for buffer in (
                increment.terms.destination_slots, increment.terms.source_cells,
                increment.terms.weights,
            )
        ):
            raise ValueError("Raw term buffers cannot overlap any H profile plane")
        if np.may_share_memory(matrix, light):
            raise ValueError("Raw luminosity cannot overlap any H profile plane")
        if matrix.shape != (increment.terms.input_length, increment.profiles.size):
            raise ValueError("Raw luminosity shape differs from validated terms/profiles")
        if increment.terms.output_length != store.spec.slot_count:
            raise ValueError("Raw terms were validated against a different H size")
        if (
            increment.profiles.ndim != 1
            or increment.profiles.dtype.kind not in "iu"
            or np.any(increment.profiles < 0)
            or np.any(increment.profiles >= store.spec.profile_count)
        ):
            raise ValueError("Raw profile IDs are outside this H")
        if np.unique(increment.profiles).size != increment.profiles.size:
            raise ValueError("Raw profile IDs cannot repeat")
        if (
            increment.touched_rows.shape != store.touched_rows.shape
            or increment.touched_rows.dtype != np.dtype("bool")
        ):
            raise ValueError("Raw touched-row metadata differs from this table")
        store.touched_rows |= increment.touched_rows
        if not increment.terms.destination_slots.size:
            continue
        for local, profile in enumerate(increment.profiles):
            plan[int(profile)].append((increment, local))
    increment = None

    def one_profile(profile):
        plane = light[profile]
        copied_bytes = 0
        kernel_calls = 0
        for increment, local in plan[profile]:
            values = np.ascontiguousarray(increment.luminosity[:, local])
            copied_bytes += values.nbytes
            increment.terms.accumulate(values=values, output=plane)
            # Free this profile copy before allocating the next group's copy.
            del values
            kernel_calls += 1
        return copied_bytes, kernel_calls

    active_profiles = [profile for profile, tasks in enumerate(plan) if tasks]
    futures = []
    try:
        for profile in active_profiles:
            futures.append(executor.submit(one_profile, profile))
        results = [future.result() for future in futures]
    except BaseException:
        for future in futures:
            future.cancel()
        # All running writers must stop before a caller releases the borrowed
        # luminosity or the partially modified, now unusable H accumulator.
        for future in futures:
            if not future.cancelled():
                try:
                    future.result()
                except BaseException:
                    pass
        raise
    finally:
        # Completed Future bookkeeping can briefly outlive its worker call;
        # empty the captured plan so it cannot retain this wave's luminosity.
        for tasks in plan:
            tasks.clear()
    return {
        "profile_scatter_seconds": perf_counter() - began,
        "profile_tasks": len(active_profiles),
        "contiguous_X_logical_bytes": sum(row[0] for row in results),
        "kernel_calls": sum(row[1] for row in results),
        "raw_corner_terms": sum(increment.terms.destination_slots.size for increment in increments),
    }
