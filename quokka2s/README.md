# quokka2s

Post-processing pipeline for QUOKKA R-MHD simulation snapshots — produces
synthetic line emission (CO, [C II] 158 μm, Hα, H I 21 cm), temperature
diagnostics, σ–Σ_SFR overlays, and multi-phase ISM analyses on top of a
pre-built DESPOTIC chemistry / cooling table.

---

## Quickstart (5 min)

```bash
# 1. Point the config at a QUOKKA plt directory and a DESPOTIC table.
$EDITOR quokka2s/src/quokka2s/pipeline/prep/config.py
#   YT_DATASET_PATH     = "/path/to/plt263168"
#   DESPOTIC_TABLE_PATH = "/path/to/despotic_table.npz"   (auto-selected by DESPOTIC_GEOM env var)
#   DOWNSAMPLE_FACTOR   = 2     # see "Why downsample?" below

# 2. Run the whole pipeline.
python -m quokka2s.pipeline.tasks.run_pipeline

# 3. Outputs land in:
#       <OUTPUT_DIR>/                        — PNGs from each task
#       <OUTPUT_DIR>/task_intermediates/     — task compute() results (HDF5)
#       <dataset>/intermediates/<dataset_name>/fields/
#                                            — derived-field intermediates (HDF5)
```

The first cold run takes ~5–15 min depending on resolution. Subsequent
runs reuse the saved intermediates and finish in seconds–minutes.

---

## CLI cheat sheet

```bash
# Full run (compute + plot)
python -m quokka2s.pipeline.tasks.run_pipeline

# Re-plot only (use saved task intermediates; iterate on figure styling)
python -m quokka2s.pipeline.tasks.run_pipeline --mode plot

# Compute only — populate intermediates without writing figures (CI / batch use)
python -m quokka2s.pipeline.tasks.run_pipeline --mode compute

# Ignore saved intermediates and recompute everything from scratch
python -m quokka2s.pipeline.tasks.run_pipeline --force

# Run only specific task class(es)
python -m quokka2s.pipeline.tasks.run_pipeline --task EmitterTask --task PhaseSigmaVTask

# Wipe both intermediate stores
python -m quokka2s.pipeline.tasks.run_pipeline --clean-intermediates
```

---

## Intermediates (the daily-iteration speedup)

Two kinds of intermediate data get saved to disk so repeat runs are cheap.

### Field intermediates — shared 3D derived fields

Expensive yt derived fields (`column_density_H`, `temperature_despotic`,
all four line luminosities, …) are written once to HDF5 and reloaded on
subsequent runs:

```
<dataset_path>/intermediates/<dataset_name>/fields/
    field_gas_temperature_despotic.h5
    field_gas_CO_luminosity.h5
    field_gas_C+_luminosity.h5
    field_gas_H_alpha_luminosity.h5
    field_gas_HI_luminosity.h5
```

The full list lives in `pipeline/cache.py::CACHED_FIELDS`. Cheap fields
(constants, simple math) are not stored — yt's in-memory layer handles
them per process.

### Task intermediates — per-task compute() outputs

Each task's `compute()` dict is serialised to HDF5:

```
<output_dir>/task_intermediates/
    EmitterTask_<hash>.h5
    PhaseSigmaVTask_<hash>.h5
    PhaseResolvedSpectrumTask_<hash>.h5
    ...
```

The 8-char hash suffix encodes the task's `__init__` args, so e.g.
`PhaseSpectrumOverlayTask(R=1e5)` and `PhaseSpectrumOverlayTask(R=inf)`
get independent files.

### Invalidation

Each saved file stores a `cache_key = sha1(snapshot path + snapshot
mtime + table path + table mtime + downsample factor + schema version)`.
On load, if the recomputed key doesn't match, the file is silently
ignored and rebuilt. To force a rebuild without changing inputs, pass
`--force`. If you edit a derived-field definition in `physics_fields.py`,
bump `CACHE_SCHEMA_VERSION` in `pipeline/cache.py` to invalidate all
existing intermediates.

### Runtime spectrum store (memory only)

`SpectrumStore` (`pipeline/services/spectrum_service.py`) memoises 1D
emission spectra **within a single pipeline run**. Any task that needs a
spectrum calls `context.spectrum_store.get_spectrum(species, los, phase,
R)`; the first call builds it, subsequent calls return the same array.
This is what lets IntegratedSpectrum, PhaseSpectrumOverlay, and
PhaseResolvedSpectrum stop re-building the same cubes. Not persisted to
disk — the task intermediates already cover the `--mode plot` case.

---

## Adding a new task

1. Copy `quokka2s/src/quokka2s/pipeline/tasks/_template.py`. Rename the
   class.
2. Implement `prepare → compute → plot`. Use `context.provider.get_slab_z`
   for data; keep numerical work in `compute()` and pure rendering in
   `plot()`. Anything you return from `compute()` will be HDF5-saved.
3. Export the new class from `tasks/__init__.py`.
4. Register it in `run_pipeline.py::build_pipeline()`.

That's it — the task intermediate is picked up automatically based on
the class name + init args.

---

## Tables (DESPOTIC chemistry)

`build_table.py` runs DESPOTIC's `setChemEq(evolveTemp='iterateDust')` over
a 35×35×35 grid of `(n_H, N_H, dV/dr)` and saves the result as
`despotic_table.npz`. The pipeline interpolates this table for every cell
at runtime via `tables/lookup.py`.

```bash
# Build a table (slow — hours).
python -m quokka2s.tables.build_table

# Stress-test DESPOTIC convergence on a sparse grid (minutes).
python check_convergence_sparse.py --points 10

# Plot the table on (n_H, N_H) slices at several dV/dr values.
python -m quokka2s.tables.view_table -n 5
```

The two geometries (LVG vs sphere escape probability) live in separate
files; select with `DESPOTIC_GEOM=LVG` (default) or `DESPOTIC_GEOM=sphere`.

---

## Why downsample? (the cumsum constraint)

`column_density_H` integrates `n_H × dx` along all three axes by cumulative
sum (six passes — forward + backward on each axis, then harmonic-mean
symmetrised). `np.cumsum` is **fundamentally non-streaming**: the value at
cell *k* depends on every cell from 0 to *k*.

A streaming version is possible (slab-by-slab accumulator), but rewriting
column-density + `dVdr_lvg` + spectral-cube builder to stream is a
~1–2 week project (see *Phase 2* in the planning notes). Until then, with
`DOWNSAMPLE_FACTOR=2` the native 256×256×2048 cube becomes 128×128×1024,
which fits comfortably in 16 GB of RAM. Increase the factor if you hit
memory limits on a smaller machine.

The downsample is implemented in `data_handling.py::make_downsampled_dataset`
using slab-by-slab block-mean (memory-efficient itself).

---

## Repo layout

```
quokka2s/src/quokka2s/
├── data_handling.py             — YTDataProvider, dataset downsample, field-intermediate hook
├── analysis.py                  — generic numerical helpers
├── pipeline/
│   ├── base.py                  — Pipeline / AnalysisTask / mode handling
│   ├── cache.py                 — HDF5 helpers + CACHED_FIELDS + cache-key hashing
│   ├── services/
│   │   └── spectrum_service.py  — SpectrumStore (in-memory shared spectrum builder)
│   ├── prep/
│   │   ├── config.py            — paths, downsample factor, T cutoffs
│   │   ├── physics_fields.py    — all yt derived-field definitions
│   │   └── …
│   └── tasks/
│       ├── _template.py         — copy-and-rename starting point
│       ├── run_pipeline.py      — CLI entry (--mode, --force, --task, …)
│       ├── emitter.py           — CO / C+ / Hα / HI surface-brightness panels
│       ├── phase_sigmaV.py      — σ_v split by 5 ISM phases
│       ├── phase_spectrum_overlay.py
│       ├── phase_resolved_spectrum.py
│       ├── spaxel_sigma.py
│       ├── sigma_sfr_overlay.py — comparison with Lenkić+24 Fig 2
│       ├── temperature_slices.py
│       └── …
└── tables/
    ├── build_table.py           — DESPOTIC table generation
    ├── view_table.py            — heatmaps of (n_H, N_H) at several dV/dr
    ├── lookup.py                — runtime trilinear interpolation
    └── …
```

---

## Installation

```bash
# In a fresh env with Python 3.11+, yt 4.x, numpy, scipy, matplotlib,
# h5py (for intermediates), unyt, despotic (for table builds only):
pip install -e quokka2s/
```
