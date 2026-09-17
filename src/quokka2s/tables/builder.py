"""Parallel construction of the canonical 3D GOW/LVG table."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm

from .models import AttemptRecord, DespoticTable, LineLumResult, SpeciesLineGrid, SpeciesRecord
from .checkpoint import PointCheckpoints, source_metadata
from .solver import CO21_TABLE_TOKEN, LINE_RESULT_FIELDS, solve_gow_lvg_point, validated_solver_metadata


LOGGER = logging.getLogger(__name__)
DEFAULT_LINE_RESULT = LineLumResult(*([float("nan")] * len(LINE_RESULT_FIELDS)))


class GridSpec(Protocol):
    def sample(self) -> np.ndarray: ...


@dataclass(frozen=True)
class SpeciesSpec:
    name: str
    is_emitter: bool


GOW_LVG_SPECIES: tuple[SpeciesSpec, ...] = (
    SpeciesSpec("CO", True),
    SpeciesSpec("C", True),
    SpeciesSpec("C+", True),
    SpeciesSpec("HCO+", True),
    SpeciesSpec("O", True),
    SpeciesSpec("e-", False),
    SpeciesSpec("H+", False),
    SpeciesSpec("H2", False),
    SpeciesSpec("H", False),
)


def build_gow_lvg_table(
    nH_grid: GridSpec,
    col_grid: GridSpec,
    dVdr_grid: GridSpec,
    *,
    species_specs: Sequence[SpeciesSpec] = GOW_LVG_SPECIES,
    show_progress: bool = True,
    workers: int | None = None,
    checkpoint_dir: Path | str | None = None,
    checkpoint_context: dict | None = None,
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
    """
    build_metadata = validated_solver_metadata()
    specs = tuple(species_specs)
    nH_vals = nH_grid.sample()
    col_vals = col_grid.sample()
    dvdr_vals = dVdr_grid.sample()
    options = dict(specs=specs, build_metadata=build_metadata,
                   show_progress=show_progress, workers=workers)
    if checkpoint_dir is not None:
        manifest = {
            "axes": [nH_vals.tolist(), col_vals.tolist(), dvdr_vals.tolist()],
            "species": [[spec.name, spec.is_emitter] for spec in specs],
            "solver_metadata": build_metadata,
            "source_metadata": source_metadata(),
            "configuration": {"Tg_init": 100.0, "context": checkpoint_context},
        }
        with PointCheckpoints.open(Path(checkpoint_dir), manifest) as checkpoints:
            return _build_sampled_table(nH_vals, col_vals, dvdr_vals,
                                        checkpoints=checkpoints, **options)
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
            nH_val=coordinates[0], colDen_val=coordinates[1], dvdr_val=coordinates[2],
            species=emitter_names, abundance_only=abundance_only,
            row_idx=row, col_idx=col, dvdr_idx=dvdr, Tg_init=100.0,
            log_failures=True, attempt_log=attempts,
        )
        if checkpoints:
            checkpoints.save(indices, coordinates, result, attempts)
    return indices, result, attempts


def _build_sampled_table(nH_vals, col_vals, dvdr_vals, *, specs, build_metadata,
                         show_progress, workers, checkpoints=None) -> DespoticTable:
    shape = (len(nH_vals), len(col_vals), len(dvdr_vals))
    num_rows = shape[0]

    tg_table = np.full(shape, np.nan)
    failure_mask = np.zeros(shape, dtype=bool)
    abundance_map = {spec.name: np.full(shape, np.nan) for spec in specs}
    mu_grid = np.full(shape, np.nan)
    cv_grid = np.full(shape, np.nan)
    eint_grid = np.full(shape, np.nan)
    energy_fields: dict[str, np.ndarray] = {}

    emitter_names = tuple(spec.name for spec in specs if spec.is_emitter)
    abundance_only = tuple(spec.name for spec in specs if not spec.is_emitter)
    line_output_names = emitter_names + (
        (CO21_TABLE_TOKEN,) if "CO" in emitter_names else ()
    )
    line_buffers = {
        name: {field: np.full(shape, np.nan) for field in LINE_RESULT_FIELDS}
        for name in line_output_names
    }

    if checkpoints:
        # All points in a row can now save concurrently. Prepare directories
        # before dispatch so no worker races to create the same row directory.
        checkpoints.prepare_rows(num_rows)

    def tasks():
        for indices in np.ndindex(shape):
            row, col, dvdr = indices
            coordinates = (float(nH_vals[row]), float(col_vals[col]), float(dvdr_vals[dvdr]))
            yield delayed(_solve_point)(
                indices, coordinates, emitter_names, abundance_only, checkpoints,
                solve_gow_lvg_point,
            )

    # Preserve canonical diagnostic order despite out-of-order completion.
    attempts_by_point = {}
    energy_order = {}
    with (
        parallel_config(backend="loky", inner_max_num_threads=1),
        Parallel(n_jobs=-1 if workers is None else workers, batch_size=1,
                 pre_dispatch="2*n_jobs", return_as="generator_unordered") as pool,
        tqdm(total=int(np.prod(shape)), desc="DESPOTIC points", unit="point",
             disable=not show_progress) as progress,
    ):
        results = pool(tasks())
        try:
            for indices, result, point_attempts in results:
                attempts_by_point[indices] = point_attempts
                line_results, chem_abunds, mu, cv, eint, tg, energy_terms, failed = result
                failure_mask[indices] = failed
                progress.update(1)
                if failed:
                    # Failed attempts remain diagnostic records, never inputs
                    # to interpolation or the valid-node cleaning support.
                    continue
                tg_table[indices] = tg
                mu_grid[indices] = mu
                cv_grid[indices] = cv
                eint_grid[indices] = eint
                for spec in specs:
                    abundance_map[spec.name][indices] = chem_abunds.get(spec.name, np.nan)
                for name in line_output_names:
                    line = line_results.get(name, DEFAULT_LINE_RESULT)
                    for field in LINE_RESULT_FIELDS:
                        line_buffers[name][field][indices] = getattr(line, field)
                for position, (term, value) in enumerate(energy_terms.items()):
                    if term not in energy_fields:
                        energy_fields[term] = np.full(shape, np.nan)
                    energy_fields[term][indices] = value
                    order = (indices, position)
                    energy_order[term] = min(energy_order.get(term, order), order)
        finally:
            # Also cancel outstanding work before releasing the checkpoint
            # lock if consumption fails or the user interrupts the build.
            results.close()

    attempts = [attempt for indices in sorted(attempts_by_point)
                for attempt in attempts_by_point[indices]]
    energy_fields = {term: energy_fields[term]
                     for term in sorted(energy_fields, key=energy_order.__getitem__)}

    failed_cells = int(np.count_nonzero(failure_mask))
    if failed_cells:
        LOGGER.warning("GOW/LVG table: %s/%s cells failed", failed_cells, failure_mask.size)

    species_data: dict[str, SpeciesRecord] = {}
    for spec in specs:
        abundance = abundance_map[spec.name]
        line = None
        if spec.is_emitter:
            fields = line_buffers[spec.name]
            line = SpeciesLineGrid(
                freq=fields["freq"], intIntensity=fields["intIntensity"], intTB=fields["intTB"],
                lumPerH=fields["lumPerH"], tau=fields["tau"], tauDust=fields["tauDust"],
                abundance=abundance,
            )
        species_data[spec.name] = SpeciesRecord(spec.name, abundance, line, spec.is_emitter)

    if CO21_TABLE_TOKEN in line_buffers:
        fields = line_buffers[CO21_TABLE_TOKEN]
        co_abundance = abundance_map["CO"]
        line = SpeciesLineGrid(
            freq=fields["freq"], intIntensity=fields["intIntensity"], intTB=fields["intTB"],
            lumPerH=fields["lumPerH"], tau=fields["tau"], tauDust=fields["tauDust"],
            abundance=co_abundance,
        )
        species_data[CO21_TABLE_TOKEN] = SpeciesRecord(
            CO21_TABLE_TOKEN, co_abundance, line, True,
        )

    return DespoticTable(
        species_data=species_data,
        tg_final=tg_table,
        nH_values=nH_vals,
        col_density_values=col_vals,
        dVdr_values=dvdr_vals,
        mu_values=mu_grid,
        cv_values=cv_grid,
        Eint_values=eint_grid,
        failure_mask=failure_mask,
        energy_terms=energy_fields or None,
        attempts=tuple(attempts),
        build_metadata=build_metadata,
    )
