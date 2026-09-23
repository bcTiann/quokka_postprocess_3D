"""Bounded-memory projections and central slices of a uniform simulation grid.

The caller supplies the mixed temperature and adopted validity mask. This
module makes no choices about chemistry, interpolation, or hydrogen abundance.
Array axes are always (x, y, z), without any plotting transpose or recentering.
"""
from __future__ import annotations

import operator

import numpy as np


_FIELDS = ("vz_kms", "T_quokka_K", "T_mixed_K")
_TOTAL_KEYS = ("mass_g", "momentum_z_g_kms", "T_quokka_mass_g_K", "T_mixed_mass_g_K")


class MultiviewAccumulator:
    """Stream contiguous x slabs, retaining only two-dimensional products.

    ``shape`` is the full (nx, ny, nz) grid shape. ``widths_cm`` gives the
    individual cell widths (dx, dy, dz), not the domain widths. An edge-on
    projection integrates along x and has native shape (ny, nz); a face-on
    projection integrates along z and has native shape (nx, ny).

    Density panels are central slices at x=nx//2 and z=nz//2. Surface mass
    densities are sum(rho * dl); projected velocity and temperatures are
    sum(rho * field * dl) / sum(rho * dl). Both all-cell and valid-cell
    versions are accumulated. Empty valid sightlines have zero surface density
    and count, but NaN weighted means. Invalid central-slice pixels are NaN.
    """

    def __init__(self, shape, widths_cm):
        try:
            grid_shape = tuple(operator.index(value) for value in shape)
        except (TypeError, ValueError) as exc:
            raise ValueError("shape must contain three positive integers") from exc
        if len(grid_shape) != 3 or any(value <= 0 for value in grid_shape):
            raise ValueError("shape must contain three positive integers")
        widths = np.asarray(widths_cm, dtype=np.float64)
        if widths.shape != (3,) or not np.isfinite(widths).all() or np.any(widths <= 0):
            raise ValueError("widths_cm must contain three finite positive cell widths")
        with np.errstate(over="ignore", under="ignore"):
            volume = float(np.prod(widths))
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError("Cell volume must be finite and positive")
        self.shape = grid_shape
        self.widths_cm = widths.copy()
        self.widths_cm.flags.writeable = False
        self.cell_volume_cm3 = volume
        self.next_ix = 0
        nx, ny, nz = grid_shape
        self._maps = {}
        self._totals = {}
        for selection in ("all", "valid"):
            self._totals[selection] = {"cell_count": 0, **{key: 0. for key in _TOTAL_KEYS}}
            for view, view_shape in (("edge", (ny, nz)), ("face", (nx, ny))):
                self._maps[selection, view] = {
                    "sigma_g_cm2": np.zeros(view_shape),
                    "cell_count": np.zeros(view_shape, dtype=np.int64),
                    "rho_slice_g_cm3": np.full(view_shape, np.nan),
                    **{field: np.zeros(view_shape) for field in _FIELDS},
                }

    def add(self, ix, rho, vz_kms, tq_K, mixed_K, valid):
        """Add a full-y, full-z slab starting at ``ix``; inputs are not mutated.

        All physical inputs must be finite, including excluded cells, because
        the all-cell products are computed independently. In particular a NaN
        DESPOTIC result must be replaced with QUOKKA temperature by the caller
        when that cell belongs to the hot branch, before forming mixed_K.
        """
        try:
            start = operator.index(ix)
        except TypeError as exc:
            raise ValueError("ix must be an integer x index") from exc
        if start != self.next_ix or start >= self.shape[0]:
            raise ValueError("Slabs must be ordered, contiguous, and inside the grid")
        density = np.asarray(rho, dtype=np.float64)
        if (density.ndim != 3 or density.shape[1:] != self.shape[1:]
                or density.shape[0] == 0 or start + density.shape[0] > self.shape[0]):
            raise ValueError("Slab shape must be (positive nslab, ny, nz) inside the grid")
        fields = dict(zip(_FIELDS, (np.asarray(value, dtype=np.float64)
                                    for value in (vz_kms, tq_K, mixed_K))))
        for name, values in (("rho", density), *fields.items()):
            if values.shape != density.shape or not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite and have the same slab shape")
            if name != "vz_kms" and np.any(values <= 0):
                raise ValueError(f"{name} must be positive")
        valid = np.asarray(valid)
        if valid.shape != density.shape or valid.dtype != np.bool_:
            raise ValueError("valid must be a boolean array with the same slab shape")

        # Complete all potentially failing calculations before changing state.
        stop = start + density.shape[0]
        x_slice = slice(start, stop)
        deltas, totals = {}, {}
        for selection, selected in (("all", np.ones(density.shape, dtype=bool)),
                                    ("valid", valid)):
            selected_density = np.where(selected, density, 0.)
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                mass = selected_density * self.cell_volume_cm3
                total = {"cell_count": int(np.count_nonzero(selected)),
                         "mass_g": float(np.sum(mass, dtype=np.float64))}
                for view, axis, dl in (("edge", 0, self.widths_cm[0]),
                                       ("face", 2, self.widths_cm[2])):
                    deltas[selection, view] = {
                        "sigma_g_cm2": np.sum(selected_density, axis=axis) * dl,
                        "cell_count": np.sum(selected, axis=axis, dtype=np.int64),
                    }
                for field, total_key in zip(_FIELDS, _TOTAL_KEYS[1:]):
                    # Select before multiplication: an excluded value never
                    # contributes through an IEEE 0 * NaN intermediate.
                    selected_field = np.where(selected, fields[field], 0.)
                    numerator = selected_density * selected_field
                    total[total_key] = float(np.sum(mass * selected_field, dtype=np.float64))
                    for view, axis, dl in (("edge", 0, self.widths_cm[0]),
                                           ("face", 2, self.widths_cm[2])):
                        deltas[selection, view][field] = np.sum(numerator, axis=axis) * dl
            if not all(np.isfinite(value) for value in total.values()):
                raise ValueError("Full-volume totals overflowed")
            totals[selection] = {
                key: self._totals[selection][key] + value for key, value in total.items()
            }
            if not all(np.isfinite(value) for value in totals[selection].values()):
                raise ValueError("Accumulated full-volume totals overflowed")
            for view in ("edge", "face"):
                for key, delta in deltas[selection, view].items():
                    previous = self._maps[selection, view][key]
                    previous = previous if view == "edge" else previous[x_slice]
                    with np.errstate(over="ignore", invalid="ignore"):
                        updated = previous + delta
                    if not np.isfinite(updated).all():
                        raise ValueError("Projected sums overflowed")
                    deltas[selection, view][key] = updated

        for selection, selected in (("all", None), ("valid", valid)):
            self._totals[selection] = totals[selection]
            for view in ("edge", "face"):
                for key, values in deltas[selection, view].items():
                    target = self._maps[selection, view][key]
                    if view == "edge":
                        target[...] = values
                    else:
                        target[x_slice] = values
            z_center = self.shape[2] // 2
            face_density = density[:, :, z_center]
            self._maps[selection, "face"]["rho_slice_g_cm3"][x_slice] = (
                face_density if selected is None
                else np.where(selected[:, :, z_center], face_density, np.nan))
            x_center = self.shape[0] // 2
            if start <= x_center < stop:
                offset = x_center - start
                edge_density = density[offset]
                self._maps[selection, "edge"]["rho_slice_g_cm3"][...] = (
                    edge_density if selected is None
                    else np.where(selected[offset], edge_density, np.nan))
        self.next_ix = stop
        return self

    def _require_complete(self):
        if self.next_ix != self.shape[0]:
            raise ValueError("All x slabs must be added before requesting final products")

    def payload(self):
        """Return independent arrays for saving; all view arrays use native axes.

        Keys follow ``{all|valid}_{edge|face}_{field}``, with fields
        ``sigma_g_cm2``, ``cell_count``, ``rho_g_cm3``, ``vz_kms``,
        ``T_quokka_K`` and ``T_mixed_K``. No hydrogen conversion is applied.
        """
        self._require_complete()
        output = {"shape": np.asarray(self.shape, dtype=np.int64),
                  "cell_widths_cm": self.widths_cm.copy()}
        for (selection, view), maps in self._maps.items():
            prefix = f"{selection}_{view}_"
            for field in ("sigma_g_cm2", "cell_count", "rho_slice_g_cm3"):
                output_field = "rho_g_cm3" if field == "rho_slice_g_cm3" else field
                output[prefix + output_field] = maps[field].copy()
            for field in _FIELDS:
                average = np.full(maps["sigma_g_cm2"].shape, np.nan)
                np.divide(maps[field], maps["sigma_g_cm2"], out=average,
                          where=maps["sigma_g_cm2"] > 0.)
                output[prefix + field] = average
        return output

    def report(self):
        """JSON-safe independent volume totals for numerical cross-checks."""
        self._require_complete()
        return {
            "shape": list(self.shape),
            "cell_widths_cm": self.widths_cm.tolist(),
            "cell_volume_cm3": self.cell_volume_cm3,
            "edge_projection_axis": "x",
            "face_projection_axis": "z",
            "edge_native_axes": ["y", "z"],
            "face_native_axes": ["x", "y"],
            "edge_slice_x_index": self.shape[0] // 2,
            "face_slice_z_index": self.shape[2] // 2,
            "selections": {key: value.copy() for key, value in self._totals.items()},
        }
