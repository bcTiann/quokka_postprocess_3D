# Repository directories and reading order

The repository has one current emission workflow. Process reads the snapshot
and fixed lookup tables; plot reads the saved numerical products. Table-building
tools and additional paper figures have their own commands.

`pyproject.toml` declares dependencies; `uv.lock` records the resolved versions
and source identities. uv creates `.venv` with `uv sync` and selects it for
`uv run`. Sync automatically installs `quokka2s` in editable mode from `src/`
and registers `process` and `plot`. No `PYTHONPATH` export or manual
package installation is needed. The repository remains the source of bundled
tables and configuration; this workflow uses an editable checkout, not a
standalone wheel with embedded input data. The default dependencies support
both process and plot; the `tables` extra adds the optional chemistry solver
and table-building tools.

## Where each kind of file belongs

| Directory | Contents | In Git? |
|---|---|---|
| `configs/` | Shared `emission.yaml` for process/plot and `table_input_survey.yaml` for job surveys; paths are relative to their YAML file | Yes |
| `src/quokka2s/` | Reusable calculations and process/plot entry points | Yes |
| `tools/cloudy/` | Build radiation inputs, run Cloudy, package its emission table | Yes |
| `tools/despotic/` | Install required GOW corrections, measure table axes, check coverage | Yes |
| `tools/snapshots/` | Survey stored field extrema from snapshot text headers | Yes |
| `tools/figures/` | Additional slice, phase, gas-projection, dust and radiation figures | Yes |
| `docs/` | Usage instructions, physical methods and code walkthrough | Yes |
| `vendor/` | Pinned external source, dust table, molecular data and their notes | Yes |
| `inputs/snapshots/` | User-provided QUOKKA snapshots | No |
| `inputs/tables/` | Cloudy `emission.npz`, expanded DESPOTIC `interpolated.npz`, and previous `interpolated_old.npz` | Yes; raw tables and other build files stay local |
| `output/` | Processed numerical products and figures | No |
| `runtime/` | Cloudy build parameters, `.inc` exports, raw outputs and logs | No |

## Start reading the current pipeline

1. [shared configuration](../configs/emission.yaml): choose the simulation job,
   snapshot and results root. Both commands can override `dataset` and x/y ranges
   explicitly on the command line. The bundled emission tables are defaults.
2. [process_snapshot.py](../src/quokka2s/process_snapshot.py): read inputs, process
   slabs, build/check/save results.
3. [snapshot_reader.py](../src/quokka2s/snapshot_reader.py): derive six copied
   core arrays from one slab. [native_snapshot_reader.py](../src/quokka2s/native_snapshot_reader.py)
   uses yt block/file metadata to copy selected x/y cells plus one periodic
   neighbour on each side, keeping full z. Full-x, partial-y/z selections read
   contiguous y bands in each z plane. Full-y/z selections read temporary field
   buffers; other selections use read-only NumPy maps. It caches metadata.
   During process, [slab_field_cache.py](../src/quokka2s/slab_field_cache.py)
   shares five converted halo-field arrays among up to four adjacent selected
   slabs, under a 3 GiB retained-value limit. The reading thread owns this scoped
   cache; each slab still derives its six copied core arrays in the same order.
   Groups shrink to fit, an unshared slab uses the direct reader, and the
   context releases retained values on success or failure. It retains no open
   maps or full-snapshot field arrays.
   `slab_windows(x_start=..., x_stop=..., slab_nx=...)` yields global core bounds;
   `slab.iter_batches(batch_size=...)` yields native cell views. Consume batches
   before releasing the slab: a retained view keeps its underlying arrays alive.
4. [cell_emission.py](../src/quokka2s/physics/cell_emission.py): obtain each named
   line's emissivity and temperature, then apply foreground dust.
5. [emission_products.py](../src/quokka2s/products/emission_products.py): send the
   same calculated batch to image, spectrum and gas-phase accumulators.
6. [plot_emission_results.py](../src/quokka2s/plot_emission_results.py): draw the
   saved products selected by the same [configuration](../configs/emission.yaml).

[line_definitions.py](../src/quokka2s/line_definitions.py) is the shared list of
line wavelengths and emitter masses.
[emission_results.py](../src/quokka2s/emission_results.py) reads completed
products and selects their arrays by line, dust state, or gas-phase name.
Renderers use this interface and each product's own saved coordinates.

The [code walkthrough](code_walkthrough.md) explains concrete arrays, units and
examples at each step. The [method reference](emission_method.md) states the
current scientific rules.

## Package names describe their jobs

| Directory | Main files and purpose |
|---|---|
| `physics/` | `settings.py`: shared physical choices; `gas_fields.py`: derived fields and mixed temperature; `line_emissivity.py`, `hydrogen_emissivity.py`, `dust_attenuation.py`: emission calculations |
| `cloudy/` | `lookup.py`: interpolation; `cell_fields.py`: batch queries; `table_definition.py`: ordered build lines, grid and radiation recipe; `incident_spectrum.py`: continuum-export reader |
| `despotic/` | `solver_settings.py`: adopted solver defaults/source identities; `table_data.py`: table records/field names; `snapshot_domain.py`: physical settings/bounds checks and recorded build-snapshot metadata |
| `despotic/` | `table_files.py`: NPZ serialization; `cell_solver.py`: equilibrium at one state; `fixed_temperature_chemistry.py`: fixed-temperature GOW callable; `abundance_workspace.py`, `two_body_rates.py`, `reaction_terms.py`: its composition and reaction helpers; `table_builder.py`: grid calculation |
| `despotic/` | `lookup.py` and `cell_fields.py`: process-time queries; `build_table.py`, `extend_table.py`, `interpolate_failed.py`, `list_failures.py`: table commands; `table_extension.py`: exact axis extension and raw-source validation; `prepare_table_plots.py`/`table_figure_data.py`: numerical figure preparation; `plot_table.py`/`table_plots.py`: draw saved panels |
| `products/` | Image, spectrum, gas-phase and auxiliary numerical accumulation; `profile_preparation.py` prepares saved normalized/per-area curves; auxiliary preparation modules save masks, logarithmic panels, limits and radiation/dust samples |
| `figures/` | `line_luminosity_images.py`, `line_spectra.py`, `gas_phase_spectra.py`: saved-product renderers; `line_labels.py`: titles; `figure_files.py`: PNG/PDF paths and saving; other modules: additional figures |

`gas_fields.mixed_gas_temperature_K()` supplies the cold-DESPOTIC/hot-QUOKKA
temperature rule to gas-phase statistics, projection maps and phase histograms.
`products/gas_phase_velocity.py` owns the phase names and temperature bounds;
`GasPhaseResults` reads the saved definitions for plotting. Presentation titles
belong to `figures/line_labels.py`. `file_provenance.py` provides file hashes for
table-build and diagnostic identities; process-time table checks use the
physical domain, not file hashes.

`configs/despotic_extension.yaml` selects an existing raw table and new bounds.
`extend_table.py` prints a plan or runs the extension. `table_builder.py` owns
the writable table arrays, copies raw successes by exact coordinate maps, and
submits only new coordinates to the solver. Old failed nodes stay missing for
the separate interpolation step. Newly solved points use their own checkpoint
directory; the old input table is never overwritten.

The root modules also separate the final processing steps:
`result_metadata.py` extracts selected pixel geometry and ordered dust values;
`processing_report.py` builds the readable summary; `result_files.py` writes
already-built arrays and the report. `run_settings.py` returns `ProcessSettings`
or `PlotSettings` from the same YAML and explicit command-line overrides.
Both derive the snapshot directory from `results_dir/jobname/snapshot_name`.
`--dataset ABS_PATH` replaces the YAML dataset; `--x` and `--y` each replace
only their matching YAML range. Omitted options keep YAML values. CLI dataset
paths must be absolute after expanding `~`; YAML relative paths still start at
the YAML file. Plot uses the dataset folder name without accessing the original
snapshot. `jobname` is only a manually chosen output group. Process writes its
`processed/` subdirectory; plot reads that subdirectory and writes figures.
Explicit x/y ranges add a shared region subdirectory, for example
`region_x64-66_y64-66/`; an omitted or `null` axis is named `all`. Both axes
omitted or `null` keep the full-box result directory.
The effective process settings are saved as `run_configuration` in the report.
`input_paths.py` stores bundled input locations without importing calculations.
`check_table_input_ranges.py` reuses the same configuration and physical field
definitions to survey input coordinates. `table_input_reader.py` reads complete
native fields and shares them among XY columns at the same z. Its ascending
density pass saves exact cumulative columns at z faces; its descending pass
visits every selected column at each z before moving down and clearing that
layer's native-field cache. Four owned range fields preserve complete z-column
physics without retaining full-snapshot arrays. It reuses block metadata,
file reads and unit stages in `native_snapshot_reader.py`. Emission processing retains its six fields,
x slabs and existing reduction order; `slab_nx` and `max_slabs` remain emission
settings.
`table_input_statistics.py` accumulates named scalar ranges;
`table_input_range_check.py` compares them with current
axes. `table_input_files.py` owns safe output and completed-scan reuse. The
statistics are table-independent and are separate from emission products.
`check_job_table_input_ranges.py` selects snapshots from the jobs in
`configs/table_input_survey.yaml` and runs that same scan with one worker by
default. Its YAML controls `query_chunk` and diagnostic `max_columns`;
the single-snapshot command accepts `--max-columns`. The limit counts native
XY block columns, each using its complete z depth; omitted columns make a scan
partial.
`snapshot_workers` enables independent concurrent snapshot scans;
optional invocation limits cap new scans and stop dispatch after a time budget.
Already running scans finish. The command saves per-snapshot statistics plus
CSV/JSON summaries in the configured output
directory. `table_input_jobs.py` owns job selection, scan order and summaries;
`table_input_survey.py` owns the shared scan-or-reuse action for both commands.
Completed scans can be reused; summaries distinguish `complete`, `partial`,
`failed` and `not_selected` snapshots.

`prepare_cold_gas_phase.py` prepares the cold nH–NH distribution for the jobs
in `configs/cold_gas_phase.yaml`. `cold_gas_reader.py` reads density and
temperature and reuses exact complete-z shielding columns; it does not read
momentum or calculate velocity gradients. `cold_gas_survey.py` owns source
protection, complete-result reuse and independent snapshot workers.
`products/cold_gas_phase.py` prepares histograms and every figure array;
`plot_cold_gas_phase.py` and `figures/cold_gas_phase.py` draw saved products.
`snapshot_headers.py` shares the text-header reader with the header-survey
tool, allowing fresh headers to prove a snapshot contains no cold cells.
`paths.py` resolves user paths once at configuration and command-line entry
points, expanding `~` and using the appropriate relative-path base.
`prepare_emission_results.py` prepares figure arrays from existing numerical
products, including optional coarse images. Normal snapshot processing saves
native figure arrays directly. `figures/display_settings.py` shares the
velocity display window; these choices do not change saved channels.

## Data and results

```text
inputs/
  snapshots/plt0655228/
  tables/despotic/raw.npz       # Local table-building output; not included in Git
  tables/despotic/interpolated.npz
  tables/cloudy/emission.npz

output/sigma13.v2/plt0655228/
  processed/
    images.npz
    images_factor_2.npz       # Optional, separately prepared 2x2-summed image
    spectra.npz
    phase_velocity.npz
    emission_report.json
    status.json
  figures/
    png/
    pdf/
  figures_titled/
    png/
    pdf/
  region_x64-66_y64-66/       # Only when x/y ranges are selected
    processed/
    figures/
    figures_titled/
```

Jobs such as `sigma13.v2` and `sigma50` have separate directories, so both can
contain a snapshot named `plt0655228`. More snapshots of one job sit alongside
each other under that job's directory. Set `jobname` to a distinct output group
when switching jobs or runs: no identity is inferred from a dataset path.

Processing writes a new directory per run. Plotting may redraw from the saved
products many times. Each figure version keeps PNG and PDF files in its own
`png/` and `pdf/` subdirectories. Raw Cloudy calculation files belong in `runtime/`;
process and plot use only the job and snapshot selected in the shared configuration.
