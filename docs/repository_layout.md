# Repository directories and reading order

Two root scripts are independent entry points:

- `uv run python process_snapshot.py --dataset /absolute/path/plt0008703`
  computes, checks and saves one explicitly selected snapshot.
- `uv run python plot_emission_results.py --dataset /absolute/path/plt0008703`
  draws that snapshot's saved products without reading the original plt.

No process/plot aliases are registered in `pyproject.toml`. That file describes
the package and dependencies; `uv.lock` fixes their versions. `uv sync` creates
an editable environment, so the root scripts import the current `src/` files.
No custom Cython build or import-path export is required.

A parent run directory is rejected. Shared YAML holds output, tables and worker
settings, and optionally one absolute snapshot path. Grid dimensions come from
metadata; arbitrary valid x/y index ranges retain full z columns. Original
level-zero FABs can have shorter edge blocks. No converted field copy is used.

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
| `inputs/tables/` | Cloudy `emission.npz`, DESPOTIC `interpolated.npz`, and their previous `emission_old.npz` / `interpolated_old.npz` | Yes; raw tables and other build files stay local |
| `output/` | Processed numerical products and figures, including the Setonix results; diagnostics and temporary field copies remain local | Results only |
| `runtime/` | Cloudy build parameters, `.inc` exports, raw outputs and logs | No |

## Start reading the current pipeline

1. [process_snapshot.py](../process_snapshot.py) calls the package's process
   `main()`. [run_settings.py](../src/quokka2s/run_settings.py) resolves one
   absolute snapshot path, optional x/y ranges and output/table settings.
2. [process_snapshot.py](../src/quokka2s/process_snapshot.py) checks reusable
   results and calls `run_one_snapshot()`. Gaussian template preparation is
   outside the process clock. Metadata, physical table loading and all later
   work stay inside it. Process does not invoke plot.
3. [processing_inputs.py](../src/quokka2s/processing_inputs.py) opens metadata
   with yt and loads the Cloudy, DESPOTIC and dust inputs. It owns one
   [native_column_reader.py](../src/quokka2s/native_column_reader.py), which
   uses `np.fromfile` for original FAB fields and copies selected values into
   z-contiguous RAM tiles. Global neighbour faces support unchanged gradients.
4. [native_column_process.py](../src/quokka2s/native_column_process.py) validates
   exact selected-region coverage and memory planning. Each selected tile is
   derived by [native_column_fields.py](../src/quokka2s/native_column_fields.py),
   then processed in bounded worker waves. Partial final batches are valid.
5. [cell_emission.py](../src/quokka2s/physics/cell_emission.py) calculates named
   emissivities and thermal temperatures, keeping per-line missingness, then
   foreground dust attenuation. [emission_products.py](../src/quokka2s/products/emission_products.py)
   owns image, spectrum and gas-phase accumulators and independent accounting.
6. [template_spectra.py](../src/quokka2s/products/gaussian/template_spectra.py)
   prepares each Gaussian's four template-corner contributions.
   [template_batches.py](../src/quokka2s/products/gaussian/template_batches.py)
   finishes physical preparation, corner preparation and accumulation in bounded
   waves. [corner_scatter.py](../src/quokka2s/products/gaussian/corner_scatter.py)
   uses guarded prebuilt SciPy sparse arithmetic to add weighted luminosity into
   disjoint profile planes. [corner_store.py](../src/quokka2s/products/gaussian/corner_store.py)
   convolves completed rows into 400-bin spectra. No other Gaussian engine is
   selected or maintained. Stable direct integrals for valid out-of-template
   sigma values remain an essential part of this route.
7. `build_outputs()` prepares all numerical figure fields, `check_outputs()`
   validates conservation, and `write_emission_results()` saves three NPZs and
   the report. Status completion follows successful saving.
8. [plot_emission_results.py](../plot_emission_results.py) independently calls
   the package plot entry point, reading saved arrays via `emission_results.py`.
   It selects formats/styles and renders without recomputing physical products.

[code_walkthrough.md](code_walkthrough.md) follows the arrays in more detail.
[emission_method.md](emission_method.md) records the scientific definitions.

## Package names describe their jobs

| Directory | Responsibility |
|---|---|
| `physics/` | Derived gas fields, emission, thermal widths and dust attenuation |
| `cloudy/`, `despotic/` | Table data, validated interpolation and independent table-building tools |
| `products/` | Images, spectra, gas phases, moments and prepared display arrays |
| `products/gaussian/` | Stored templates, four-corner preparation/accumulation, convolution, direct width fallback and accounting |
| `figures/` | Render prepared numerical arrays; select stored fields and format axes/layout |

`products/integrated_spectra.py` supplies shared line masks, full moments and
saved-spectrum preparation; it does not select an alternative integration method.
`result_metadata.py`, `processing_report.py` and `result_files.py` respectively
prepare interpretation fields, assemble descriptive metadata, and save results.
`process_reuse.py` verifies completion, exact configuration/input identity and
NPZ integrity before reusing results. Older/conflicting outputs remain untouched.

`gas_fields.mixed_gas_temperature_K()` supplies the cold-DESPOTIC/hot-QUOKKA
temperature rule to gas-phase statistics, projection maps and phase histograms.
`products/gas_phase_velocity.py` owns the phase names and temperature bounds;
`GasPhaseResults` reads the saved definitions for plotting. Presentation titles
belong to `figures/line_labels.py`. `file_provenance.py` provides file hashes for
table-build, diagnostic and completed-result identities. Table loading checks
the physical domain; the separate result-reuse check also requires matching
table-file hashes.

`configs/despotic_extension.yaml` selects an existing raw table and new bounds.
`extend_table.py` prints a plan or runs the extension. `table_builder.py` owns
the writable table arrays, copies raw successes by exact coordinate maps, and
submits only new coordinates to the solver. Old failed nodes stay missing for
the separate interpolation step. Newly solved points use their own checkpoint
directory; the old input table is never overwritten.

`input_paths.py` stores bundled input locations without importing calculations.
`check_table_input_ranges.py` reuses the same configuration and physical field
definitions to survey input coordinates. `table_input_reader.py` reads complete
native fields and shares them among XY columns at the same z. Its ascending
density pass saves exact cumulative columns at z faces; its descending pass
visits every selected column at each z before moving down and clearing that
layer's native-field cache. Four owned range fields preserve complete z-column
physics without retaining full-snapshot arrays. It reuses block metadata,
file reads and unit stages in `native_snapshot_reader.py`. Emission processing
retains six physical fields for the current selected tile and deterministic
batch reduction order. Native x/y ranges select the work; z remains complete.
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
±50 and ±200 km/s spectral display windows; these choices do not change saved
channels. Plot settings select PNG/PDF and titled/untitled output; the default
is titled PNG files.

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
    png/                    # Only when untitled PNG output is selected
    pdf/                    # Only when untitled PDF output is selected
  figures_titled/
    png/                    # Default plot output
    pdf/                    # Only when titled PDF output is selected
  region_x64-66_y64-66/       # Only when x/y ranges are selected
    processed/
    figures/
    figures_titled/
```

Jobs such as `sigma13.v2` and `sigma50` have separate directories, so both can
contain a snapshot named `plt0655228`. More snapshots of one job sit alongside
each other under that job's directory. Set `jobname` to a distinct output group
when switching jobs or runs: no identity is inferred from a dataset path.

Processing writes a new directory for each selected snapshot/region, or reuses
matching completed results. Plotting may redraw from the saved products many
times, selecting one snapshot each time. Each selected figure style keeps its chosen formats in
`png/` and/or `pdf/` subdirectories. Spectral filenames end with `_v50` or
`_v200` to distinguish the two display windows. Images are drawn once per style.
Existing files from unselected styles or formats remain in place.
Raw Cloudy calculation files belong in `runtime/`. Each emission output group
should correspond to one simulation run. Each invocation selects one saved
snapshot and the configured or overridden x/y region.
