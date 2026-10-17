"""Parallel construction of the canonical 3D GOW/LVG table."""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm

from quokka2s.despotic.table_data import (
    AttemptRecord,
    DespoticTable,
    ExplicitGrid,
    LINE_RESULT_FIELDS,
    NAN_LINE_RESULT,
    SpeciesLineGrid,
    SpeciesRecord,
)
from quokka2s.despotic.checkpoint import PointCheckpoints, source_metadata
from quokka2s.despotic.cell_solver import (
    CO21_TABLE_TOKEN,
    solve_gow_lvg_point,
    validated_solver_metadata,
)
from quokka2s.despotic.solver_settings import (
    DEFAULT_ABUNDANCE_SPECIES,
    DEFAULT_EMITTERS,
    INITIAL_GAS_TEMPERATURE_K,
)
from quokka2s.despotic.table_extension import (
    map_old_axis_positions,
    read_raw_table_for_extension,
    validate_reuse_settings,
)
from quokka2s.file_provenance import file_sha256
from quokka2s.paths import resolve_path


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeciesSpec:
    name: str
    is_emitter: bool


GOW_LVG_SPECIES: tuple[SpeciesSpec, ...] = (
    *(SpeciesSpec(name=name, is_emitter=True) for name in DEFAULT_EMITTERS),
    *(SpeciesSpec(name=name, is_emitter=False) for name in DEFAULT_ABUNDANCE_SPECIES),
)


def build_gow_lvg_table(
    nH_grid: ExplicitGrid,
    col_grid: ExplicitGrid,
    dVdr_grid: ExplicitGrid,
    *,
    species_specs: Sequence[SpeciesSpec] = GOW_LVG_SPECIES,
    show_progress: bool = True,
    workers: int | None = None,
    checkpoint_dir: Path | str | None = None,
    checkpoint_context: dict | None = None,
    reuse_raw_table: Path | str | None = None,
) -> DespoticTable:
    """Build a true 3-input ``(nH, N_H, dVdr)`` GOW/LVG table.

    Each grid cell independently solves GOW chemistry and dust/gas thermal
    equilibrium.  Emitters are present during the equilibrium solve, so LVG
    line cooling contributes to the converged temperature. A persistent process
    pool schedules individual points with one inner native-library thread per
    worker, and completed results are assembled by their grid indices.

    ``species_specs`` is exposed only so the sparse smoke test can exercise a
    cheaper subset.  The production CLI always uses :data:`GOW_LVG_SPECIES`.

    ``checkpoint_dir`` optionally commits every completed point, including
    failures, for reuse after interruption. Reuse requires identical grid,
    species, source/data hashes, solver metadata and ``checkpoint_context``
    (for example, the source snapshot domain). Worker count may change.

    reuse_raw_table optionally names an existing raw NPZ. Its exact axis
    coordinates must all occur in the requested grids. Successful nodes are
    copied, including their physical fields and remapped attempt records.
    Old failed nodes remain failed; only coordinates outside the old grid
    are solved. Interpolated source tables and different solver settings
    are rejected. An extension uses its own new checkpoint directory.
    """
    build_metadata = validated_solver_metadata()
    specs = tuple(species_specs)
    nH_vals = nH_grid.sample()
    col_vals = col_grid.sample()
    dvdr_vals = dVdr_grid.sample()
    source_table = None
    source_identity = None
    if reuse_raw_table is not None:
        source_path = resolve_path(reuse_raw_table)
        source_digest = file_sha256(source_path)
        source_table = read_raw_table_for_extension(source_path)
        if file_sha256(source_path) != source_digest:
            raise ValueError("Raw source changed while loading; no extension was started")
        validate_reuse_settings(
            source_table=source_table,
            current_metadata=build_metadata,
            species_specs=specs,
        )
        source_identity = {
            "raw_table_sha256": source_digest,
            "policy": "copy raw successes; keep old failures; solve new coordinates only",
        }
    options = {
        "specs": specs,
        "build_metadata": build_metadata,
        "show_progress": show_progress,
        "workers": workers,
        "source_table": source_table,
        "source_identity": source_identity,
    }
    if checkpoint_dir is not None:
        manifest = {
            "axes": [nH_vals.tolist(), col_vals.tolist(), dvdr_vals.tolist()],
            "species": [[spec.name, spec.is_emitter] for spec in specs],
            "solver_metadata": build_metadata,
            "source_metadata": source_metadata(),
            "configuration": {
                "Tg_init": INITIAL_GAS_TEMPERATURE_K,
                "context": checkpoint_context,
            },
        }
        if source_identity is not None:
            manifest["reuse_raw_table"] = source_identity
        with PointCheckpoints.open(Path(checkpoint_dir), manifest) as checkpoints:
            return _build_sampled_table(
                nH_vals,
                col_vals,
                dvdr_vals,
                checkpoints=checkpoints,
                **options,
            )
    return _build_sampled_table(nH_vals, col_vals, dvdr_vals, **options)


def _solve_point(indices, coordinates, emitter_names, abundance_only, checkpoints, solver):
    """Solve or restore one independent point in a persistent worker process."""
    saved = checkpoints.load(indices, coordinates) if checkpoints else None
    if saved is not None:
        result, attempts = saved
    else:
        attempts: list[AttemptRecord] = []
        row, col, dvdr = indices
        result = solver(
            nH_val=coordinates[0],
            colDen_val=coordinates[1],
            dvdr_val=coordinates[2],
            species=emitter_names,
            abundance_only=abundance_only,
            row_idx=row,
            col_idx=col,
            dvdr_idx=dvdr,
            Tg_init=INITIAL_GAS_TEMPERATURE_K,
            log_failures=True,
            attempt_log=attempts,
        )
        if checkpoints:
            checkpoints.save(indices, coordinates, result, attempts)
    return indices, result, attempts


class TableBuildArrays:
    """Own one table's writable arrays and its pending-coordinate mask.

    axes are nH [cm^-3], NH [cm^-2] and dV/dr [s^-1]. Every physical field
    has shape (len(nH), len(NH), len(dVdr)). Pending nodes are new physical
    coordinates, not old failed points. No source arrays are modified.
    """

    def __init__(self, *, axes, specs):
        self.axes = axes
        self.specs = specs
        self.shape = tuple(len(axis) for axis in axes)
        self.temperature_K = np.full(self.shape, np.nan)
        self.failure_mask = np.zeros(self.shape, dtype=bool)
        self.nodes_to_solve = np.ones(self.shape, dtype=bool)
        self.abundances = {
            spec.name: np.full(self.shape, np.nan)
            for spec in specs
        }
        self.mu_values = np.full(self.shape, np.nan)
        self.cv_values = np.full(self.shape, np.nan)
        self.internal_energy_values = np.full(self.shape, np.nan)
        self.emitter_names = tuple(spec.name for spec in specs if spec.is_emitter)
        self.abundance_only_names = tuple(spec.name for spec in specs if not spec.is_emitter)
        self.line_names = self.emitter_names
        if "CO" in self.emitter_names:
            self.line_names = self.line_names + (CO21_TABLE_TOKEN,)
            self.abundances[CO21_TABLE_TOKEN] = np.full(self.shape, np.nan)
        self.line_fields = {
            name: {
                field: np.full(self.shape, np.nan)
                for field in LINE_RESULT_FIELDS
            }
            for name in self.line_names
        }
        self.energy_fields = {}
        self.energy_order = {}
        self.attempts_by_point = {}
        self.reused_success_count = 0
        self.retained_failure_count = 0

    def copy_raw_nodes(self, *, source_table):
        """Copy complete direct solutions; retain old failures without retry.

        Exact coordinate maps account for lower-end additions. For example,
        old nH index 0 may become new nH index 8. All three source axes are
        mapped independently before copying any physical fields.
        """
        positions = tuple(
            map_old_axis_positions(old_values=old_axis, new_values=new_axis)
            for old_axis, new_axis in zip(
                (source_table.nH_values, source_table.col_density_values, source_table.dVdr_values),
                self.axes,
            )
        )
        old_region = np.ix_(*positions)
        successful = ~source_table.failure_mask
        self.nodes_to_solve[old_region] = False
        self.failure_mask[old_region] = source_table.failure_mask
        self.reused_success_count = int(np.count_nonzero(successful))
        self.retained_failure_count = int(np.count_nonzero(source_table.failure_mask))

        # Source and destination fields have matching physical meanings. The
        # old failed positions stay NaN even if a malformed raw file stored a
        # diagnostic value there. No interpolated values seed this new table.
        physical_fields = [
            (self.temperature_K, source_table.tg_final),
            (self.mu_values, source_table.mu_values),
            (self.cv_values, source_table.cv_values),
            (self.internal_energy_values, source_table.Eint_values),
        ]
        for spec in self.specs:
            source_species = source_table.require_species(spec.name)
            physical_fields.append((self.abundances[spec.name], source_species.abundance))
        if CO21_TABLE_TOKEN in self.line_names:
            physical_fields.append((
                self.abundances[CO21_TABLE_TOKEN],
                source_table.require_species(CO21_TABLE_TOKEN).abundance,
            ))
        for name in self.line_names:
            source_line = source_table.require_species(name).require_line()
            for field in LINE_RESULT_FIELDS:
                physical_fields.append((self.line_fields[name][field], getattr(source_line, field)))
        for destination, original in physical_fields:
            if not np.isfinite(original[successful]).all():
                raise ValueError("Raw successful nodes must contain complete finite physical fields")
            copied = np.full(original.shape, np.nan)
            copied[successful] = original[successful]
            destination[old_region] = copied

        for position, (name, original) in enumerate((source_table.energy_terms or {}).items()):
            if not np.isfinite(original[successful]).all():
                raise ValueError("Raw successful nodes must contain complete finite energy fields")
            destination = np.full(self.shape, np.nan)
            copied = np.full(original.shape, np.nan)
            copied[successful] = original[successful]
            destination[old_region] = copied
            self.energy_fields[name] = destination
            present = np.flatnonzero(successful & np.isfinite(original))
            if present.size:
                old_index = np.unravel_index(present[0], original.shape)
                new_index = tuple(int(axis[index]) for axis, index in zip(positions, old_index))
            else:
                new_index = self.shape
            self.energy_order[name] = (new_index, position)

        # Preserve the old attempts as history, with their indices translated
        # to the new grid. Their temperatures, durations and messages are not
        # fabricated anew. They are not new checkpoint records.
        for attempt in source_table.attempts:
            old_index = (attempt.row_idx, attempt.col_idx, attempt.dvdr_idx)
            if attempt.dvdr_idx is None:
                raise ValueError("Raw attempt records must identify their dV/dr index")
            if any(index < 0 or index >= len(axis) for axis, index in zip(positions, old_index)):
                raise ValueError("Raw attempt index lies outside its source grid")
            new_index = tuple(int(axis[index]) for axis, index in zip(positions, old_index))
            expected = tuple(float(axis[index]) for axis, index in zip(self.axes, new_index))
            if (attempt.nH, attempt.colDen, attempt.dvdr) != expected:
                raise ValueError("Raw attempt coordinates differ from their stored axis positions")
            remapped = replace(
                attempt,
                row_idx=new_index[0],
                col_idx=new_index[1],
                dvdr_idx=new_index[2],
            )
            self.attempts_by_point.setdefault(new_index, []).append(remapped)

    def pending_points(self):
        """Yield only new coordinates, in canonical nH/NH/dVdr index order."""
        for indices in np.ndindex(self.shape):
            if not self.nodes_to_solve[indices]:
                continue
            coordinates = tuple(float(axis[index]) for axis, index in zip(self.axes, indices))
            yield indices, coordinates

    def record_solved_point(self, *, indices, result, attempts):
        """Store one new point or its completed new-build checkpoint record."""
        self.attempts_by_point[indices] = attempts
        line_results, chem_abundances, mu, cv, internal_energy, temperature, energy_terms, failed = result
        self.failure_mask[indices] = failed
        if failed:
            return
        self.temperature_K[indices] = temperature
        self.mu_values[indices] = mu
        self.cv_values[indices] = cv
        self.internal_energy_values[indices] = internal_energy
        for spec in self.specs:
            self.abundances[spec.name][indices] = chem_abundances.get(spec.name, np.nan)
        if CO21_TABLE_TOKEN in self.line_names:
            self.abundances[CO21_TABLE_TOKEN][indices] = chem_abundances.get("CO", np.nan)
        for name in self.line_names:
            line = line_results.get(name, NAN_LINE_RESULT)
            for field in LINE_RESULT_FIELDS:
                self.line_fields[name][field][indices] = getattr(line, field)
        for position, (name, value) in enumerate(energy_terms.items()):
            if name not in self.energy_fields:
                self.energy_fields[name] = np.full(self.shape, np.nan)
                self.energy_order[name] = (indices, position)
            self.energy_fields[name][indices] = value
            self.energy_order[name] = min(self.energy_order[name], (indices, position))

    def build_table(self, *, metadata):
        """Return the normal raw-table record, preserving named field arrays."""
        species_data = {}
        for spec in self.specs:
            line = None
            if spec.is_emitter:
                line = SpeciesLineGrid(**self.line_fields[spec.name])
            species_data[spec.name] = SpeciesRecord(
                name=spec.name,
                abundance=self.abundances[spec.name],
                line=line,
                is_emitter=spec.is_emitter,
            )
        if CO21_TABLE_TOKEN in self.line_fields:
            species_data[CO21_TABLE_TOKEN] = SpeciesRecord(
                name=CO21_TABLE_TOKEN,
                abundance=self.abundances[CO21_TABLE_TOKEN],
                line=SpeciesLineGrid(**self.line_fields[CO21_TABLE_TOKEN]),
                is_emitter=True,
            )
        attempts = tuple(
            attempt
            for indices in sorted(self.attempts_by_point)
            for attempt in self.attempts_by_point[indices]
        )
        energy_terms = {
            name: self.energy_fields[name]
            for name in sorted(self.energy_fields, key=self.energy_order.__getitem__)
        }
        return DespoticTable(
            species_data=species_data,
            tg_final=self.temperature_K,
            nH_values=self.axes[0],
            col_density_values=self.axes[1],
            dVdr_values=self.axes[2],
            mu_values=self.mu_values,
            cv_values=self.cv_values,
            Eint_values=self.internal_energy_values,
            failure_mask=self.failure_mask,
            energy_terms=energy_terms or None,
            attempts=attempts,
            build_metadata=metadata,
        )


def _build_sampled_table(
    nH_vals,
    col_vals,
    dvdr_vals,
    *,
    specs,
    build_metadata,
    show_progress,
    workers,
    checkpoints=None,
    source_table=None,
    source_identity=None,
) -> DespoticTable:
    arrays = TableBuildArrays(
        axes=(nH_vals, col_vals, dvdr_vals),
        specs=specs,
    )
    if source_table is not None:
        arrays.copy_raw_nodes(source_table=source_table)
    new_node_count = int(np.count_nonzero(arrays.nodes_to_solve))
    if checkpoints is not None and new_node_count:
        checkpoints.prepare_rows(arrays.shape[0])

    def tasks():
        for indices, coordinates in arrays.pending_points():
            yield delayed(_solve_point)(
                indices,
                coordinates,
                arrays.emitter_names,
                arrays.abundance_only_names,
                checkpoints,
                solve_gow_lvg_point,
            )

    if new_node_count:
        with (
            parallel_config(backend="loky", inner_max_num_threads=1),
            Parallel(
                n_jobs=-1 if workers is None else workers,
                batch_size=1,
                pre_dispatch="2*n_jobs",
                return_as="generator_unordered",
            ) as pool,
            tqdm(
                total=new_node_count,
                desc="DESPOTIC new points" if source_table is not None else "DESPOTIC points",
                unit="point",
                disable=not show_progress,
            ) as progress,
        ):
            results = pool(tasks())
            try:
                for indices, result, point_attempts in results:
                    arrays.record_solved_point(
                        indices=indices,
                        result=result,
                        attempts=point_attempts,
                    )
                    progress.update(1)
            finally:
                # Cancel in-flight new work before releasing the checkpoint
                # lock when collection is interrupted. Copied old nodes have
                # never been submitted to the process pool.
                results.close()

    failed_nodes = int(np.count_nonzero(arrays.failure_mask))
    if failed_nodes:
        LOGGER.warning("GOW/LVG table: %s/%s nodes failed", failed_nodes, arrays.failure_mask.size)
    metadata = dict(build_metadata)
    if source_table is not None:
        # The measured snapshot remains historical evidence. Extended table
        # bounds are chosen coordinates, not newly measured snapshot extrema.
        if "snapshot_domain" in source_table.build_metadata:
            metadata["snapshot_domain"] = dict(source_table.build_metadata["snapshot_domain"])
        metadata["table_domain"] = {
            "origin": "outward extension of an existing raw table",
            "axes": {
                name: {"minimum": float(axis[0]), "maximum": float(axis[-1])}
                for name, axis in zip(("nH", "NH", "dVdr"), arrays.axes)
            },
        }
        metadata["table_extension"] = {
            **source_identity,
            "reused_successful_nodes": arrays.reused_success_count,
            "retained_old_failed_nodes": arrays.retained_failure_count,
            "new_coordinate_nodes": new_node_count,
            "source_grid_shape": list(source_table.tg_final.shape),
            "source_build_metadata": dict(source_table.build_metadata),
        }
    return arrays.build_table(metadata=metadata)
