# Julia snapshot processing and saved-product plotting

The independent implementation in `julia/` reads Quokka plotfiles through Mera
and calculates the adopted emission model in Julia. Its command processes one
complete snapshot by default, prepares the numerical figure arrays, and draws
the standard emission figures. Optional x/y selections use the same workflow.
The Python `process` command remains the production reference. Both use the
physical rules in [emission_method.md](emission_method.md), including the existing
fixed table compositions. A dataset name does not change abundances or scale
metallicity automatically.

For copyable local/Setonix installation, an independent experimental checkout,
YAML configuration and saved-result transfer, start with
[julia/README.md](../julia/README.md). This document describes the implementation
and numerical contract.

## Supported inputs and reader layout

Inputs must be three-dimensional Cartesian, uniform level-0 Quokka plotfiles,
named `plt` followed by digits. The pinned Mera frontend requires cubic cells;
the physical domain may be rectangular or offset. Stored density, three momentum
components, and `temperature` or `gasTemperature` are required. `metadata.yaml`
must declare finite positive length, mass, time and temperature unit factors.
Temperature reconstruction and AMR processing are not enabled.

The streaming reader also requires native FAB boxes to form aligned z layers.
After duplicate z intervals are removed, those intervals must partition the
original z domain without gaps or overlaps. Multiple boxes may tile x/y within
one layer, but staggered or overlapping z extents across boxes are rejected.
Uniform resolution alone does not guarantee this supported file layout.

Every selected x/y sightline includes the complete original z depth. Command-line
ranges use 0-based half-open native indices. Each omitted CLI axis retains its
YAML range; a `null` range selects the full corresponding axis. Julia APIs use
1-based inclusive ranges. Periodic x/y neighbours refer to the original box,
including for a region; only the two physical z faces use one-sided velocity
differences. Subdivision into numerical batches does not introduce new physical
boundaries.

## Environment and fixed assets

Use Julia 1.13.2 with the checked-in `Project.toml` and `Manifest.toml`. Mera is
fixed to commit `db277419e05a45e8d4921fb5f3a0a4d15792cf7f`, which contains the
Quokka frontend on `multicode`. Python numerical preparation and rendering use
the repository's locked uv environment. The `process-julia` project entry point
selects its current Python interpreter and passes the Julia depot, project,
thread limits and Matplotlib cache directly to subprocesses; no shell exports
or temporary configuration files are needed.

From the repository root:

```bash
uv sync --locked
uv run process-julia --setup
```

`--setup` requires an installed Julia 1.13.2 selected by YAML. It instantiates
and precompiles the checked-in Julia manifest and calls the one-time exporter
if the configured assets are absent. Existing assets must match their recorded
source inputs and numeric hash; conflicts are preserved and reported. Routine
processing never performs Julia package installation automatically.

The exporter reads the fixed Cloudy, DESPOTIC and dust inputs without running
either chemistry solver. It writes numeric-only `tables.npz` and
`physics.json`, containing axes, coefficients, missing/zero masks, line
definitions, physical constants and input hashes. Original tables remain
untouched. The standalone `uv run python julia/export_tables.py --output PATH`
also accepts `--despotic`, `--cloudy` and `--dust` for explicit asset sources.
The Julia loader verifies the exported numeric asset hash before processing.
Launcher-managed defaults stay under ignored `runtime/julia/`: `depot/`,
`assets/` and `matplotlib/`. uv's project cache is `runtime/uv-cache/`; an
existing `UV_CACHE_DIR` overrides that uv setting.

## Run one snapshot or region

Edit [`configs/emission_julia.yaml`](../configs/emission_julia.yaml):

```yaml
dataset: /absolute/path/to/run/plt0000000
jobname: my-run-julia-validation
output: ../output/julia-experimental/my-run/plt0000000-full-v1
x_index_range: null
y_index_range: null
threads: 1
```

Then run from the repository root:

```bash
uv run process-julia --dry-run
uv run process-julia
```

`--dry-run` reads and validates only the configuration, then prints resolved
settings, the child environment overrides and argument vector. It does not
access Julia, snapshots or assets or create directories. Normal execution uses
the same settings; the selected YAML path/hash and effective configuration are
retained in the report. Duplicate/unknown YAML keys and invalid types are
rejected instead of being silently ignored.

Precedence is CLI over YAML over built-in defaults. Relative YAML paths start
at the YAML file; relative CLI paths start at the invoking directory. `~` is
expanded, but shell variables in YAML are not. A bare `julia_executable: julia`
uses PATH; a path with a slash is YAML-relative. Julia subprocesses run from
the repository root, including when a local Juliaup directory override is used.

This selects the full x/y domain when both YAML ranges are `null`. For a
region, add `--x 64 68 --y 64 72`, supply either axis alone, or set the
corresponding YAML pair. `--all-x` and `--all-y` independently clear a YAML
selection. Every explicit `--dataset` must be absolute after `~` expansion and
requires an explicit `--jobname`, even when YAML has a name. The job label is
manually chosen and is never inferred from the snapshot path.

`output` is the exact destination and must not exist or overlap the snapshot
or assets. Existing completed and failed outputs are preserved; choose a fresh
destination for another run. Changing dataset/jobname does not change `output`
automatically: update all three fields or also pass `--output` for another case.
This command handles one snapshot. Run discovery, result reuse and checkpoint
recovery are not implemented. It reads the Julia YAML independently of Python's
`configs/emission.yaml`.

| YAML setting / CLI option | Meaning and default |
|---|---|
| `dataset`, `jobname`, `output` / `--dataset`, `--jobname`, `--output` | Required input, manually selected simulation label and exact result directory |
| `x_index_range`, `y_index_range` / `--x START STOP`, `--y START STOP` | Independent native half-open ranges; default `null` selects each whole axis |
| `julia_executable` / `--julia-executable PATH` | Installed Julia 1.13.2; default `julia` on PATH |
| `julia_depot`, `assets` / `--julia-depot`, `--assets` | Default repository `runtime/julia/depot` and `runtime/julia/assets` |
| `batch_cells` / `--batch-cells N` | Target cells per emission batch; default `131072`, at least one selected x/y plane |
| `reader_cache_mib` / `--reader-cache-mib N` | Estimated retained reader-array budget in MiB; default `1024` |
| `threads` / `--threads N` | Julia worker and product task count; default `1`; BLAS remains single-threaded |
| `no_plot` / `--no-plot`, `--plot` | Save raw and prepared numerical products with optional rendering; default `false` |
| `figure_formats` / `--format png pdf` | One or both figure formats; default `[png]` |
| `figure_styles` / `--style titled untitled` | One or both layouts; default `[titled]`; `untitled` is the paper layout |
| `image_downsample_factor` / `--image-downsample-factor N` | Prepare and select N×N summed image pixels; default `1`; must divide both selected image dimensions |
| `raw_luminosity` / `--raw-luminosity`, `--per-area` | Draw saved dL/dv instead of dΣL/dv for individual line spectra; default `false` |
| `save_cells` / `--save-cells`, `--no-save-cells` | Large per-batch verification arrays under `cells/`; default `false` |

Normalized gas-phase and all-line overlays keep their established full-channel
peak normalization regardless of `raw_luminosity`. The lower-level
`julia/bin/process.jl --help` remains available for direct Julia callers; it
accepts explicit resolved options and has no YAML parser. The uv launcher
supplies `--python` automatically, creates the selected Julia thread pool, and
passes the same task count to that script.

The reader budget is **not an RSS cap**. Before field reads, the planner estimates
its retained adjacent-layer fluid arrays, column boundaries, reverse columns,
velocity face, workspaces, batch arrays and fill mask. It rejects a plan exceeding
`--reader-cache-mib`. Mera's temporary FAB/conversion arrays, Julia's runtime,
emission arrays, accumulated products and Python preparation/rendering require
additional memory. Mera reads complete intersecting FAB components; a periodic
region's bounding window can visit more cells than requested. Reducing
`--batch-cells` affects batch arrays but does not eliminate the adjacent-layer
cache. Each batch contains complete selected x/y planes, so the target cannot be
smaller than `Nx_selected * Ny_selected`.

## Data flow and complete-column physics

[`bin/process.jl`](../julia/bin/process.jl) calls
[`run_snapshot()`](../julia/src/pipeline.jl). The stages are:

1. `reader.jl` opens geometry, stored-field definitions and physical units with
   Mera. `stream_reader.jl` validates the native z-layer layout and estimates
   retained reader memory.
2. A density-only prepass obtains each layer's hydrogen column. Positive sums
   in ascending and descending z independently provide the columns outside
   each layer; this avoids recovering a small column by subtracting a prefix
   from a nearly equal total.
3. The main reader visits z layers in ascending order, retaining at most two
   adjacent layers of density, stored temperature and three velocities, plus
   the preceding velocity face. It emits thin-z batches in native Julia
   `(x,y,z)` order, with x fastest. Each callback consumes its arrays before
   the next batch; complete snapshot fluid fields are not retained.
4. `emission.jl` interpolates fixed tables and calculates ten intrinsic
   emissivities and their emitting temperatures. Missing values remain missing;
   prescribed physical zeros remain available.
5. `stream_products.jl` retains one `ProductAccumulator` across batches.
   `products.jl` applies dust, integrates thermal Gaussians over velocity
   channels, and accumulates images, moments and gas-phase statistics. Tasks
   own fixed disjoint x/y columns and private spectra/moments. Batches must
   cover the selected z interval exactly once in increasing order.
   Each batch computes bounded raw partials; after all tasks succeed, these
   merge into persistent task states in a fixed order. Weighted moments retain
   weights, means and centered second moments, including the correction between
   group means. Dispersions and normalized profiles are calculated only at finalization.
   Images remain direct sums within disjoint x/y columns. Changing the batch
   size or task count can change floating-point summation order.
6. `pipeline.jl` checks final luminosities, masses and counts against independent
   cell sums, prepares native display coordinates, verifies source identity,
   and saves `products.npz` and the raw-complete report.
7. `prepare_results.py` reads only these saved arrays and report, converts array
   order, and prepares the existing renderer schema. Optional rendering then
   reads only `prepared/` through `plot_results.py`.

The spectral kernel shares each Gaussian probability across lines with identical
thermal widths and both dust states. `gaussian_channel_range` omits only channels
whose two edge arguments are on the same already-saturated `product_erf` tail
(both at or below -8, or both at or above +8). It searches using the original
Float64 quotient rather than a rounded inverse velocity bound. Retained channels
keep their original arithmetic and accumulation order; this introduces no
finite-width physical cutoff. The innermost line updates omit repeated bounds
checks after the accumulator's shapes and task-owned line indices establish the
valid ranges. Changes to those construction invariants must preserve this safety
contract.

The density conversion is `nH = rho * X_H / mH` in cm⁻³ using the exported
constants. For a cell at z index k, the two shielding columns are
`N_minus = sum(j <= k, nH[j] * dz)` and
`N_plus = sum(j >= k, nH[j] * dz)`, each including the full current cell.
The lookup column is their harmonic mean,
`NH = 2 / (1/N_minus + 1/N_plus)` in cm⁻². Foreground dust toward the low-z face
uses `N_minus - 0.5*nH[k]*dz`, with half of the current cell. The prepass and
layer carries preserve these complete-sightline definitions across batch and
FAB boundaries. Velocity derivatives use original neighbours and the adopted
`max(abs(div(v))/3, floor)` definition in s⁻¹.

Console and `status.json` progress distinguish the density prepass, layer reads,
processed batches, finalization, numerical figure preparation and rendering.
Processed-cell counts advance after emission, product accumulation and independent
accounting complete. ETA is estimated during the main pass; timings record
initialization, streaming including compute, emission, products, preparation and
rendering separately. These stage timings can overlap and should not all be
added together.

## Saved results and completion

```text
new-julia-results/
  products.npz
  report.json
  status.json
  cells/batch_000001.npz        # Optional --save-cells verification data
  prepared/
    images.npz
    images_factor_2.npz        # Optional requested image factor
    spectra.npz
    phase_velocity.npz
    emission_report.json       # Prepared job, snapshot and simulation-time metadata
    preparation.json
  figures/png/                # Selected untitled formats
  figures/pdf/
  figures_titled/png/          # Selected titled formats
  figures_titled/pdf/
```

`products.npz` contains numeric arrays only. Images use `(x,y,line,dust)` and
component spectra use `(channel,line,dust,regime)`. Dust states are intrinsic
and attenuated; regimes are cold and hot. Images store erg/s per pixel. Spectra
store dL/dv in erg/s/(km/s), with the exported velocity channels (the current
assets use 400 channels from −200 to +200 km/s). Luminosity outside the window
is recorded separately, and full Gaussian moments include the entire profile.
The `products.jl` schema documents the remaining moments and gas statistics.

The native report is not a field-for-field replacement for Python's
`emission_report.json`. Per-line missing counts and masses, cold/hot counts,
luminosity moments and phase statistics are retained in the native products.
The additional Python diagnostic report grouping cold-gas luminosity by
DESPOTIC-temperature thresholds is not generated by this implementation; it is
not used by the standard figures.

`report.json` records the job name, snapshot name, physical time in seconds,
region, geometry, asset metadata, implementation identity, reader statistics,
timing and independent conservation residuals. When launched through
`process-julia`, `run_configuration` also records the YAML path/hash, resolved
settings, selected Python and explicit child environment overrides.
After raw arrays are saved, it has `raw_processing_complete=true`,
`processing_complete=false` and `status="preparing"`. Successful preparation
writes all required prepared arrays and their `preparation.json` manifest;
only then does `run_snapshot()` mark numerical processing complete.
`full_snapshot` separately records whether the selection covers the original
box. A completed region therefore has `processing_complete=true` and
`full_snapshot=false`.

Preparation converts physical time to Myr and saves the renderer's
`prepared/emission_report.json`. Titled figures use the saved job name, snapshot
and simulation time, matching the current Python presentation. Untitled figures
retain their existing layout. Plot verifies this metadata file against the
prepared manifest along with the numerical arrays.

The prepared manifest includes raw-product and report hashes, a snapshot of
source metadata, and hashes of its own prepared files. Its source-report snapshot
may still say `preparing`: `run_snapshot()` subsequently updates the original
report's status and timings. Plot validates the prepared files themselves and
does not reopen or compare the source report. The raw product and prepared
schema remain separate; no per-cell recomputation is required to prepare figures.

`status.json` tracks numerical state and a separate `plot_status`. Numerical
products remain complete if optional rendering fails; plotting can be rerun from
`prepared/`. With `--no-plot`, the prepared products are complete and
`plot_status` is `not_requested`. These status files are progress records,
not resumable numerical checkpoints.

## Redraw the standard figures

```bash
uv run python julia/plot_results.py /absolute/path/to/new-julia-results \
  --format png pdf --style titled untitled
```

The positional path may also point directly at `prepared/`; original snapshots,
emission tables, `products.npz` and the original `report.json` are unnecessary
for plotting. `--output DIR` selects the untitled figure directory and the
titled layout uses sibling `DIR_titled`. The renderer supports the same
`--format`, `--style`, `--image-downsample-factor` and `--raw-luminosity` options
listed above, with the same defaults. A requested image factor must already
be present in the prepared manifest.

All four standard figure types reuse the existing Python renderers:

| Figure type | Files per style and format |
|---|---:|
| Intrinsic and attenuated luminosity images for ten lines; HI has one image because its dust opacity is zero | 19 |
| Individual line spectra at ±50 and ±200 km/s | 20 |
| Gas-phase mass profiles overlaid with each line at both velocity windows | 20 |
| All-line normalized spectra for both dust states at both windows | 4 |
| **Total** | **63** |

Selecting both styles and both formats writes 252 files. Image colour limits
are shared between dust states of each line. Per-area spectra, peak-normalized
profiles, image masks, limits and optional pixel sums are calculated before
saving. Plot readers only select stored values; rendering sets axes, labels,
colours and layout. Auxiliary slice, contour and phase-diagram commands outside
the original `plot` entry point are separate tools.

To prepare an existing raw result, or add a different image factor, run:

```bash
uv run python julia/prepare_results.py /absolute/path/to/new-julia-results \
  --image-downsample-factor 2
```

`--output PREPARED_DIR` changes the preparation destination. This command
accepts a raw-complete staged report or an already completed region report.
It leaves raw products and the original report unchanged. Repreparing an owned
prepared directory updates its manifest; plotting selects only factors listed
in that manifest.

To upgrade older prepared results with the new title metadata, rerun preparation
from the raw result and pass `--jobname NAME` if the original report lacks it.
A supplied name must agree with an already saved name. Missing time in an older
diagnostic report remains unavailable; no time is inferred from a `plt` number.

## The retained region diagnostic API

`QuokkaProcess.run_region(...)` remains available for small complete-z numerical
comparisons. It requires explicit Julia ranges, for example `x_range=65:68` and
`y_range=65:72`, retains the entire region in memory, and defaults to
`max_cells=2_000_000`. Its optional verification output is one `cells.npz`.
It saves raw products and marks its own raw calculation complete, but does not
invoke figure preparation or rendering. Run `prepare_results.py` and
`plot_results.py` explicitly afterward if figures are needed.

The main command now uses `run_snapshot()` for both full snapshots and regions.
`--max-cells` is not a command-line option; the diagnostic API's size guard does
not control the streaming batch target or retained-reader budget.

## Validation and performance interpretation

Use independent analytic examples as well as differential comparisons with
Python. Check geometry and units, derived fields, table interpolation and line
results before comparing images, spectra, moments and phase populations. NaN
masks, physical zeros, classifications and counts are separate checks. Small
floating-point differences can arise from summation order and special-function
libraries; compare errors relative to each physical quantity's scale.
Conservation alone does not establish correct image or spectral structure.

Measure full selected inputs, effective CPU resources, peak memory and filesystem
conditions together. Separate first-use loading/compilation, warm computation,
reader I/O, numerical preparation and plotting. Region compute-only timings do
not establish an end-to-end snapshot or Setonix speedup. Validation scripts and
run-specific results belong under ignored diagnostic output directories.
