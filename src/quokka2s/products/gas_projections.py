"""Bounded-memory projections and central slices of a uniform simulation grid.

The caller supplies the mixed temperature and retained-cell mask. This
module makes no choices about chemistry, interpolation, or hydrogen abundance.
Array axes are always (x, y, z), without any plotting transpose or recentering.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass

import numpy as np


_FIELDS = ("vz_kms", "T_quokka_K", "T_mixed_K")
_TOTAL_KEYS = ("mass_g", "momentum_z_g_kms", "T_quokka_mass_g_K", "T_mixed_mass_g_K")


@dataclass(frozen=True)
class ProjectionSlab:
    """Temporary full-y/full-z slab used for gas projections.

    start/stop locate its global x layers. density_g_cm3, valid_cells and the
    three physical field arrays have shape (slab_x, Ny, Nz). fields maps
    vz_kms, T_quokka_K and T_mixed_K to their values; no yt arrays are retained.
    """

    start: int
    stop: int
    density_g_cm3: np.ndarray
    fields: dict[str, np.ndarray]
    valid_cells: np.ndarray

    @property
    def x_slice(self):
        """Global x layers occupied by this slab, e.g. slice(8, 16)."""
        return slice(self.start, self.stop)


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
        """Create empty native edge-on and face-on projection containers.

        Parameters
        ----------
        shape : tuple of int, shape (3,)
            Full snapshot cell counts (Nx, Ny, Nz), currently (256, 256, 2048).
        widths_cm : array-like, shape (3,)
            Original cell widths (dx, dy, dz) [cm], not the domain widths.

        Examples
        --------
        maps = MultiviewAccumulator(shape=snapshot.shape, widths_cm=cell_widths)
        """
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
        """Add one contiguous x slab to edge-on and face-on gas products.

        Parameters
        ----------
        ix : int
            Global starting x index; slabs are added in their original order.
        rho, vz_kms, tq_K, mixed_K : array-like, shape (slab_x, Ny, Nz)
            Gas density [g/cm^3], z velocity [km/s], QUOKKA temperature [K],
            and caller-selected mixed temperature [K]. All must be finite,
            including excluded cells, for the independent all-cell products.
        valid : array-like of bool, shape (slab_x, Ny, Nz)
            Retained cells for the separate valid-cell projections.

        Returns
        -------
        MultiviewAccumulator
            This accumulator, retaining 2D sums and central slices only.

        Examples
        --------
        maps.add(ix=8, rho=density, vz_kms=velocity, tq_K=tq,
                 mixed_K=mixed_temperature, valid=valid_cells)
        """
        slab = self.prepare_slab_inputs(
            ix=ix,
            rho=rho,
            vz_kms=vz_kms,
            tq_K=tq_K,
            mixed_K=mixed_K,
            valid=valid,
        )
        projection_sums, volume_totals = self.calculate_slab_projection_sums(slab)
        updated_maps, updated_totals = self.combine_and_check_projection_sums(
            slab=slab,
            projection_sums=projection_sums,
            volume_totals=volume_totals,
        )
        self.store_projection_sums_and_slices(
            slab=slab,
            updated_maps=updated_maps,
            updated_totals=updated_totals,
        )
        self.next_ix = slab.stop
        return self

    def prepare_slab_inputs(self, ix, rho, vz_kms, tq_K, mixed_K, valid):
        """Validate the contiguous slab and keep named arrays in a ProjectionSlab."""
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
        fields = {
            "vz_kms": np.asarray(vz_kms, dtype=np.float64),
            "T_quokka_K": np.asarray(tq_K, dtype=np.float64),
            "T_mixed_K": np.asarray(mixed_K, dtype=np.float64),
        }
        for name, values in (("rho", density), *fields.items()):
            if values.shape != density.shape or not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite and have the same slab shape")
            if name != "vz_kms" and np.any(values <= 0):
                raise ValueError(f"{name} must be positive")
        valid_cells = np.asarray(valid)
        if valid_cells.shape != density.shape or valid_cells.dtype != np.bool_:
            raise ValueError("valid must be a boolean array with the same slab shape")
        return ProjectionSlab(
            start=start,
            stop=start + density.shape[0],
            density_g_cm3=density,
            fields=fields,
            valid_cells=valid_cells,
        )

    def calculate_slab_projection_sums(self, slab: ProjectionSlab):
        """Calculate this slab's surface sums and independent full-volume totals.

        Returns dictionaries keyed by (selection, view) and selection. The
        selection is all or valid; edge integrates along x onto (Ny, Nz),
        while face integrates along z onto (slab_x, Ny). No state changes yet.
        """
        projection_sums = {}
        volume_totals = {}
        selections = (
            ("all", np.ones(slab.density_g_cm3.shape, dtype=bool)),
            ("valid", slab.valid_cells),
        )
        for selection, selected_cells in selections:
            selected_density = np.where(selected_cells, slab.density_g_cm3, 0.)
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                mass = selected_density * self.cell_volume_cm3
                total = {
                    "cell_count": int(np.count_nonzero(selected_cells)),
                    "mass_g": float(np.sum(mass, dtype=np.float64)),
                }
                for view, axis, dl in (("edge", 0, self.widths_cm[0]),
                                       ("face", 2, self.widths_cm[2])):
                    projection_sums[selection, view] = {
                        "sigma_g_cm2": np.sum(selected_density, axis=axis) * dl,
                        "cell_count": np.sum(selected_cells, axis=axis, dtype=np.int64),
                    }
                for field, total_key in zip(_FIELDS, _TOTAL_KEYS[1:]):
                    # Select before multiplication so excluded values cannot
                    # contribute through a 0 * NaN intermediate.
                    selected_field = np.where(selected_cells, slab.fields[field], 0.)
                    numerator = selected_density * selected_field
                    total[total_key] = float(np.sum(mass * selected_field, dtype=np.float64))
                    for view, axis, dl in (("edge", 0, self.widths_cm[0]),
                                           ("face", 2, self.widths_cm[2])):
                        projection_sums[selection, view][field] = np.sum(numerator, axis=axis) * dl
            if not all(np.isfinite(value) for value in total.values()):
                raise ValueError("Full-volume totals overflowed")
            volume_totals[selection] = total
        return projection_sums, volume_totals

    def combine_and_check_projection_sums(self, slab, projection_sums, volume_totals):
        """Add temporary slab sums to previous sums, checking overflow before storage."""
        updated_maps = {}
        updated_totals = {}
        for selection in ("all", "valid"):
            updated_totals[selection] = {
                key: self._totals[selection][key] + value
                for key, value in volume_totals[selection].items()
            }
            if not all(np.isfinite(value) for value in updated_totals[selection].values()):
                raise ValueError("Accumulated full-volume totals overflowed")
            for view in ("edge", "face"):
                updated_maps[selection, view] = {}
                for key, delta in projection_sums[selection, view].items():
                    previous = self._maps[selection, view][key]
                    if view == "face":
                        previous = previous[slab.x_slice]
                    with np.errstate(over="ignore", invalid="ignore"):
                        updated = previous + delta
                    if not np.isfinite(updated).all():
                        raise ValueError("Projected sums overflowed")
                    updated_maps[selection, view][key] = updated
        return updated_maps, updated_totals

    def store_projection_sums_and_slices(self, slab, updated_maps, updated_totals):
        """Store validated 2D sums and the central density slices for this slab."""
        for selection, selected_cells in (("all", None), ("valid", slab.valid_cells)):
            self._totals[selection] = updated_totals[selection]
            for view in ("edge", "face"):
                for key, values in updated_maps[selection, view].items():
                    target = self._maps[selection, view][key]
                    if view == "edge":
                        target[...] = values
                    else:
                        target[slab.x_slice] = values
            z_center = self.shape[2] // 2
            face_density = slab.density_g_cm3[:, :, z_center]
            self._maps[selection, "face"]["rho_slice_g_cm3"][slab.x_slice] = (
                face_density if selected_cells is None
                else np.where(selected_cells[:, :, z_center], face_density, np.nan)
            )
            x_center = self.shape[0] // 2
            if slab.start <= x_center < slab.stop:
                offset = x_center - slab.start
                edge_density = slab.density_g_cm3[offset]
                self._maps[selection, "edge"]["rho_slice_g_cm3"][...] = (
                    edge_density if selected_cells is None
                    else np.where(selected_cells[offset], edge_density, np.nan)
                )

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
