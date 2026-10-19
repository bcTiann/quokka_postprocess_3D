# Usage settings and table building

Use the [README](../README.md) for cloning, environment setup, inputs, and the
two main commands. Run the commands below from the repository root.

## Source environment and dependencies

uv manages `.venv` from `pyproject.toml` and the committed `uv.lock`. It installs
`quokka2s` in editable mode automatically, so imports use the current checkout's
`src/`. Source edits take effect on the next run. No activation, `PYTHONPATH`
export, or manual `pip install -e .` is required. From the repository root:

```bash
uv sync --locked --extra process
```

`.python-version` selects Python 3.11. On Setonix, load the Python module and
add `--python "$(command -v python)"` to sync, as shown in the Setonix guide.
In later shells, return to the repository and use `uv run --no-sync process`
or `uv run --no-sync plot`. These commands use the prepared environment without
installing or updating dependencies during scientific work. The other Python
tools below also work in this environment without an import-path export. Repeat
sync after pulling changes to project metadata or the lockfile. For Setonix
environment commands, see the [Setonix guide](setonix.md).

The base dependencies support saved-products plotting; `process` adds the
pinned yt reader, and `tables` adds the optional DESPOTIC solver and build tools.
For an environment that only plots saved products, install just the base:

```bash
uv sync --locked
```

For table building, use `uv sync --locked --extra process --extra tables`.
Sync installs exactly the selected dependencies and removes unselected extras;
keep the extras you need when syncing that environment again. The table-building
section below includes the required solver integration steps.

`uv.lock` replaces the previous requirements.txt lock. Maintain direct
dependencies in `pyproject.toml`; after an intentional edit, update the lock:

```bash
uv lock
```

Review the lock diff and validate any changed versions. Existing locked versions
are preferred unless constraints change or an upgrade is explicitly requested.
Normal users use `--locked` instead of regenerating the lock. Do not manually
edit `uv.lock`.

## Shared configuration

Process and plot both read [configs/emission.yaml](../configs/emission.yaml).
The first three settings select the simulation job, its snapshot, and the
root directory for saved results:

| Setting | Meaning |
|---|---|
| `jobname` | Simulation setup, for example `sigma13.v2` or `sigma50` |
| `dataset` | Complete QUOKKA snapshot directory |
| `results_dir` | Root containing all jobs and snapshots |

```yaml
jobname: sigma13.v2
dataset: ../inputs/snapshots/plt0655228
results_dir: ../output
```

The snapshot name comes from the supplied `dataset` path, here `plt0655228`.
Both commands select `results_dir/jobname/snapshot_name/`: process writes
`processed/`, while plot reads `processed/` and writes `figures/` and,
by default, `figures_titled/`. They do not search for the latest snapshot.
If either x/y index range is set, these directories go inside the corresponding
`region_xSTART-STOP_ySTART-STOP/` folder. An omitted or `null` axis uses `all`;
with both axes omitted or `null`, no region folder is added.

```text
output/
├── sigma13.v2/
│   ├── plt0655228/
│   │   ├── processed/
│   │   ├── figures/
│   │   ├── figures_titled/
│   │   └── region_x64-66_y64-66/
│   │       ├── processed/
│   │       ├── figures/
│   │       └── figures_titled/
│   └── plt0916720/
└── sigma50/
    └── plt0655228/
```

The two bundled tables are selected automatically:
`inputs/tables/despotic/interpolated.npz` and
`inputs/tables/cloudy/emission.npz`. DESPOTIC now supplies the validated expanded
55 × 60 × 64 interpolated table; the previous table is retained unchanged at
`inputs/tables/despotic/interpolated_old.npz`. Cloudy and the dust table are
unchanged. The expanded DESPOTIC table still has 184 unavailable nodes outside
the finite-support hull. Coordinate coverage alone does not guarantee a usable
interpolation stencil or physical accuracy.

The supplied YAML explicitly names `despotic_table`, `cloudy_table`, and
`dust_opacity_table` using paths relative to `configs/`. If you copy that YAML
elsewhere, update these paths, use absolute paths, or remove those three entries
to use the source checkout's bundled defaults. Omitted table settings stay
anchored to the checkout even when the YAML is elsewhere. To reproduce an old
run, explicitly select `interpolated_old.npz` and use a new results root; changing
the table does not update existing saved products.

The supplied YAML uses repository-local input and output paths. To store data
elsewhere, set `dataset` and `results_dir` to the desired locations. The
[Setonix guide](setonix.md) keeps the repository and bundled tables under
`$MYSOFTWARE`, and snapshots and results under `$MYSCRATCH`.

Relative paths are resolved from the YAML file's directory, so `../inputs/` in
`configs/emission.yaml` refers to this clone's `inputs/` directory.
`dataset` must point to the individual snapshot containing `Header`, rather
than its parent `inputs/snapshots/` directory. Processing refuses to overwrite
an existing `processed/` directory; choose another results root for a rerun
of the same job, snapshot and region selection.
An interrupted emission run starts again from the beginning in a new directory.
Table-building checkpoints are a separate mechanism.

Existing numerical products can be reused without rerunning process. Place
their complete directory at `results_dir/jobname/snapshot_name/processed/`,
then run plot with that job and snapshot selected. For region results, retain
the region folder and set the same x/y ranges in the shared YAML.

### Path rules

| Where a path is supplied | Base for a relative path |
|---|---|
| Inside a YAML configuration | Directory containing that YAML file |
| Command-line options, including `--config` | Current terminal working directory |
| Bundled reference data | Source checkout containing `src/quokka2s/` |

Paths used by each command are resolved to absolute `Path` objects at the entry
point; calculations receive those resolved paths. The YAML file retains its
relative paths, so moving or cloning the repository does not require changing
the defaults.
The program does not search for replacement inputs when a file is missing.
Resolution does not create files or require output files to exist; input checks
and output-overwrite rules remain in force.

Absolute paths and `~/...` are supported in the supplied configurations too.
Use absolute paths for snapshot and result locations outside the clone.
YAML does not expand `$VARIABLE` expressions.

## Processing settings

Optional execution settings control memory and concurrency, not the physics.
Omitted settings use the same defaults shown in the shared YAML:

| Setting | Meaning | Default |
|---|---|---|
| `slab_nx` | Number of x cells read per slab, retaining selected y and full z | 8 |
| `query_chunk` | Maximum cells in one table-query batch | 1,000,000 |
| `chunk_workers` | Concurrent query batches within the loaded slab | 2 |
| `spectral_workers` | Spectral-integration threads per query batch | 3 |

Emission processing internally shares native fields across up to four adjacent
selected slabs. This leaves `slab_nx`, batch order and numerical products
unchanged. The shared converted fields retain at most 3 GiB of value arrays;
this is separate from total process RSS, address-space limits and any
cache-inclusive user memory quota. Smaller groups or a direct single-slab read
are used when the selected geometry would exceed that value limit. Only actual
selected windows are grouped, including a diagnostic `max_slabs` selection.

The worker counts multiply: this configuration permits up to 2 × 3 spectral
threads. Reading slabs remains sequential. More workers do not necessarily
make a complete run faster; select them using end-to-end timing and memory
measurements. `max_slabs: 1` in a separate YAML makes a labelled partial
run for debugging. Omit it to finish the selected region or full snapshot.

The terminal prints the selected job, snapshot path, output path, native grid,
cell widths, and selected region before processing. It identifies each table
as it loads. Each slab prints separate reading and emission-calculation
messages, followed by percentage, elapsed time, and ETA. Result construction,
conservation checks, and saving also print their stages and the final output
directory. Messages are flushed immediately, including when output is redirected
to a log file.

## Select an x-y region

Set native cell-index ranges in the [shared YAML](../configs/emission.yaml).
Keep the same `results_dir`; process and plot automatically select a region folder:

```yaml
x_index_range: [64, 128]
y_index_range: [32, 96]
```

Indices start at zero and the stop is excluded: this example selects
`64 × 64 × Nz` cells, where `Nz` is read from the snapshot (`2048` for the
reference snapshot). An omitted or `null` axis uses its full extent; z always spans
the original box. No cell averaging is performed. Gradients use neighbours
from the original simulation, and both shielding and dust columns use the
complete original z column.

Run both commands with the same YAML:

```bash
uv run --no-sync process
uv run --no-sync plot
```

This example saves under
`output/sigma13.v2/plt0655228/region_x64-128_y32-96/`.
Setting only x gives `region_x64-128_yall/`; setting only y gives
`region_xall_y32-96/`. With both ranges omitted or `null`, outputs remain directly
under the snapshot folder. Explicit ranges use their supplied indices in the
folder name, even if they cover an entire axis.
The region's images retain their physical x/y coordinates.
Its spectra, full-profile sigma, and gas-phase statistics use only selected
cells; spectra per projected area use the region's x-y area. A finished region
is a complete result and does not require `allow_partial`.
Image-resolution preparation must divide both selected image dimensions; the
default native resolution works for any region. The separate slice, multiview-map and emission
phase-histogram tools still use full-box configurations.

## Saved products

For a three-dimensional uniform level-0 snapshot of shape `(Nx, Ny, Nz)`
and the ten transitions:

| File | Main contents | Shape and units |
|---|---|---|
| `images.npz` | Intrinsic and dust-attenuated LOS-z luminosity images | `(2, 10, Nx, Ny)`, erg/s per pixel |
| `spectra.npz` | Whole-box spectra, cold/hot contributions, full-profile and window line moments | Spectra `(2, 10, 2, 400)`, erg/s/(km/s); full and window σ `(2, 10)`, km/s |
| `phase_velocity.npz` | Mass velocity histograms and moments for five gas phases and all gas | Histograms `(6, 400)`, g per channel |
| `emission_report.json` | Input paths, snapshot grid, physical settings, cell counts, omitted mass, and luminosity checks | Human-readable run record |
| `status.json` | Running, failed, partial, or completed status | Progress and elapsed time |

The reference snapshot has shape `(256, 256, 2048)`. A full
`(512, 512, 4096)` snapshot instead produces `(2, 10, 512, 512)` images;
the spectral channels do not change with spatial resolution. Cell widths
and volumes are obtained from the snapshot, not the reference grid.
For a region, the last two image axes are its selected x/y cell counts.
The report records the selected indices, shape, cell count and projected area;
the original simulation geometry is also retained. The other array dimensions
remain unchanged.

The two dust states are intrinsic and attenuated; the two regimes are selected
by `T_QUOKKA < 3000 K` and `T_QUOKKA >= 3000 K`. Spectra have 400 channels
covering −200 to +200 km/s. `line_sigma_full_kms` includes each cell's complete
thermal Gaussian, including emission outside those channels.
`line_sigma_window_kms` uses the saved channels only. Both are saved with their
centroids; figure legends use the full-profile sigma. Luminosity outside the
window is recorded separately.

`emission_report.json` also saves `cold_gas_luminosity_by_despotic_temperature`
for Halpha (`halpha`), HI (`hi21`) and CII (`cii`). Each threshold selects cells
with `T_QUOKKA < 3000 K` and finite `T_DESPOTIC >= threshold`, using thresholds
of 10,000, 40,000, 100,000 and 1,000,000 K. These selections overlap. For each
line and dust state (`intrinsic`, `attenuated`), it saves the selected luminosity,
the total luminosity and their ratio, as a fraction from 0 to 1. The denominator
includes both cold and hot branches and all velocities in the processed region;
it is not restricted to the saved spectral window. Each line skips only its own
missing emission. A zero total luminosity gives a `null` fraction. This diagnostic
does not exclude high-temperature cells or change their emission. Older results
do not contain this section; plotting does not require it.

Processing reads one x slab at a time, queries table batches within it, and
adds their luminosities to the image and spectrum arrays. It discards the
cell arrays after each slab. No full-snapshot emissivity cube is saved.
Before saving, it also prepares per-area spectra, peak-normalized line/gas
profiles, gas-channel centres, image display masks and colour limits. Raw arrays and both
full/window moments remain available alongside these fields.

Each line omits only cells with unavailable results required by that line;
image and spectrum use the same selection for that line. The processor reports
the missing count and mass separately for each line. Failed lookups
are not interpreted as physical zero emission. Before saving, it checks
cell accounting, luminosity sums, channel integrals and full-profile moments. The report
records the DESPOTIC, Cloudy, and dust table paths under `input_tables`, the
grid shape and cell widths under `snapshot_grid`, and the density/column
settings under `physical_settings`. See the
[physics reference](emission_method.md).

## Plot settings

Copy the completed `processed/` directory to the laptop, preserving its job,
snapshot and, if present, region folders. In the same shared YAML, set
`results_dir` to the local results root and `jobname` to the selected job.
The basename of `dataset`
selects the saved snapshot. Its original Setonix path can remain unchanged;
plot does not require that path or the tables to exist locally. Retain the same
x/y ranges to select a downloaded region; omit them to select the full-box result.

| Plot setting | Meaning |
|---|---|
| `titled_figures` | Also save titled figures in `figures_titled/`; default true |
| `image_downsample_factor` | Select an already prepared image resolution; default 1, native |
| `raw_luminosity` | Display spectra in luminosity units rather than luminosity per projected area; default false |
| `allow_partial` | Permit figures from an intentionally incomplete diagnostic run; default false |

To use a factor of 2, prepare that resolution once from the saved native pixels:

```bash
uv run --no-sync python -m quokka2s.prepare_emission_results --image-downsample-factor 2
```

This writes `images_factor_2.npz` in the configured products directory, summing
each 2 × 2 group rather than averaging. It retains `images.npz` at native
resolution and does not read the snapshot. Set `image_downsample_factor: 2`
in the shared YAML to select it. For another product directory, use
`--products PATH` when preparing it.

Intrinsic and attenuated images of one line share a colour range from
`vmax/1e5` to `vmax`, saved for that resolution. Plot only selects saved fields
and draws them; normalization, physical conversions and pixel sums are completed
before rendering. Products from an earlier code version can be prepared with
the same command without the optional factor, using only their saved arrays.

The command draws individual line images, intrinsic/dust spectrum comparisons,
and gas-phase/line-profile comparisons as PNG and PDF. Each figure directory
contains `png/` and `pdf/` subdirectories, including `figures_titled/` when
enabled. For example, `figures/png/spectrum_co10.png` and
`figures/pdf/spectrum_co10.pdf` contain the two formats of the same figure.
The displayed velocity range is −50 to +50 km/s; all saved channels remain
intact. H I 21 cm appears once because this dust prescription leaves it unchanged. Set
`allow_partial: true` only when intentionally plotting diagnostic subsets.

## Build tables

Table building is separate from snapshot processing. Install the optional
DESPOTIC dependency, apply the checked patches, measure the snapshot's table
coordinates, then build and interpolate:

```bash
uv sync --locked --extra process --extra tables
uv run --no-sync python tools/despotic/patch_gow_composition.py
uv run --no-sync python tools/despotic/install_checked_gow_integration.py
uv run --no-sync python tools/despotic/measure_snapshot_domain.py
uv run --no-sync python -m quokka2s.despotic.build_table \
  --snapshot-domain output/plt0655228/table_build/snapshot_domain.json \
  --checkpoint-dir output/plt0655228/table_build/checkpoints \
  --workers 1
uv run --no-sync python -m quokka2s.despotic.interpolate_failed
```

Run the integration installer again after updating an existing table-building
environment. It upgrades the reviewed checked loop to prepare one reaction
callable per fixed-temperature integration attempt, including retries.
The callable prepares fixed temperature, density and radiation terms once,
reuses private abundance mappings and evaluates the current composition and
shielding on every call. The initial state, solver tolerances and convergence
checks stay unchanged.
Prebuilt-table processing does not import this optional chemistry code.

`--snapshot-domain` is required: all three axis endpoints come from the
snapshot measurement. The current builder uses 35 × 35 × 53 logarithmic
nodes. The builder writes `inputs/tables/despotic/raw.npz`; interpolation writes
`inputs/tables/despotic/interpolated.npz`. A failed solver node remains marked
in the original `failure_mask` even if interpolation supplies a value.
`remaining_unavailable_*` masks describe availability after filling. Finite
high-temperature solutions are retained; there is no `10^6 K` deletion rule.

Use the same build command and checkpoint directory to resume completed
DESPOTIC nodes after an interruption. Inputs, solver settings and source
identity must match. Use a new checkpoint directory when changing them.
Set `--workers` to the CPUs allocated to the build job.

### Extend an existing raw DESPOTIC table

Use [configs/despotic_extension.yaml](../configs/despotic_extension.yaml) to
select the old raw table, requested bounds, a separate output and a new
checkpoint directory. Paths are relative to that YAML. The raw source is
local build data; only the interpolated process table is bundled in Git.

Inspect the plan without solving nodes or creating output directories:

```bash
uv run --no-sync python -m quokka2s.despotic.extend_table --dry-run
```

The plan lists the actual extended axes and counts of reused successes,
retained old failures and new coordinates. Each old coordinate is kept
exactly. New nodes are added outside the old bounds at the original log
spacing, so the outer endpoints may go slightly beyond the requested bounds.
Omit an axis or endpoint to retain its old range. `--config PATH` selects
another extension YAML.

When ready to solve the planned new nodes, run the same command without
`--dry-run`. The extension refuses existing output files; it has no overwrite
option. It copies complete directly solved raw states, including temperature,
abundances, line fields and diagnostics. It does not accept an interpolated
source, retry old failed coordinates, or treat missing values as zero.
Source and current solver settings/species must match.
The reviewed native and prepared fixed-temperature implementations may reuse
the same raw successes. Actual implementation hashes remain recorded; unknown
implementations and changes to physical or numerical settings are rejected.

The new checkpoint directory saves newly calculated points, including failed
attempts. Repeating the same command resumes them without solving them again.
The old source table and old checkpoint directory are not modified. Different
axes, source-table contents or solver settings require a new checkpoint
directory. The original measured snapshot domain remains historical metadata;
the chosen extended endpoints are saved separately as `table_domain`.
Changing the reaction implementation also requires a new checkpoint directory:
checkpoint reuse checks the actual source identity, not only the physical model.

After building, apply the normal convex-hull filling to the new raw table:

```bash
uv run --no-sync python -m quokka2s.despotic.interpolate_failed \
  --source output/tables/despotic_extended/raw.npz \
  --output output/tables/despotic_extended/interpolated.npz
```

Directly solved old nodes remain unchanged. Old failed positions may receive
different interpolated values because the set of real supporting nodes has
grown. Keep the existing process tables until the new pair has been reviewed.

Cloudy requires a separately installed Cloudy 17.02 executable:

```bash
uv run --no-sync python tools/cloudy/build_cloudy_emission_table.py \
  --cloudy-exe /absolute/path/to/cloudy.exe --workers 1
```

This builder handles eight atomic lines.
Runtime inputs/logs go to `runtime/cloudy_eightline/`; the final table is
`inputs/tables/cloudy/emission.npz`. Both table builders require `--force` to replace
existing final tables. The [physics reference](emission_method.md)
summarizes the table-query rules used during emission processing.

## Check table-input ranges

Select a job/snapshot in the same `configs/emission.yaml`, then run:

```bash
uv run --no-sync python -m quokka2s.check_table_input_ranges
```

The command uses `NativeBlockRangeReader`. An ascending density pass records
exact cumulative columns at each selected XY column's z-block faces. The
second pass visits z from top to bottom, processing all selected XY columns
at each z. Native fields are read once per z layer and reused for neighbouring
columns, then released. Each layer's inputs use the original column carries
and periodic velocity neighbours. Every sightline retains complete z-column
physics without a full-snapshot field array. The `nH` conversion, column and
velocity-gradient definitions match process, including the order of cumulative
additions. Emission
processing retains its six slab fields and x-slab accumulation order; its YAML
settings `slab_nx` and `max_slabs` do not control the range scan.
The range scan records DESPOTIC inputs for all cells
(including hot CO) and Cloudy inputs for
`T_QUOKKA >= 3000 K`. It does not calculate emission products or query table
values. Inputs outside the current table are reported rather than rejected.

Two files are saved in the selected run's `table_input_check/` directory,
alongside `processed/`:

| File | Contents |
|---|---|
| `input_ranges.json` | Physical extrema, units, invalid counts, hot-cell count, scan completeness and field definitions; independent of tables |
| `table_range_report.json` | Comparison with the currently configured table axes, using processing's endpoint tolerances |

Cloudy NH records both the unchanged physical range and the permitted clipped
lookup range. Only NH permits arbitrary lookup clipping; DESPOTIC coordinates
and Cloudy nH/T must be covered. Range coverage alone does not certify remaining
table holes or the suitability of composition/radiation assumptions.

Rerunning the command reuses a completed scan when source file paths/sizes/
modification times, physical definitions and the requested region match.
It reloads the current tables and rebuilds their comparison. Changing the
tables therefore does not require rereading cells. Changed simulation data or
field definitions require a new directory, selected with `--output-dir PATH`.
An interrupted snapshot is scanned again; no full-snapshot field cache is saved.

Optional x/y ranges restrict the survey just as they restrict process; such a
report describes that region only. `--max-columns 1` limits a diagnostic to one
native XY block column, still using its complete z depth. If other selected
columns remain, the result is explicitly partial and cannot certify complete
range coverage. For example:

```bash
uv run --no-sync python -m quokka2s.check_table_input_ranges --max-columns 1
```

Omit x/y restrictions and `--max-columns` for a full-snapshot survey.
Every output is checked against the resolved source directories, including
symlink destinations. Snapshot files are only read.

### Survey several simulation jobs

Use [configs/table_input_survey.yaml](../configs/table_input_survey.yaml) to select
directories containing snapshots:

```yaml
source_root: ../inputs/snapshots
jobs:
  reference: .
output_dir: ../output/table_input_ranges
snapshot_workers: 1
query_chunk: 1000000
```

Each `jobs` key is a label for the saved results. Its value is a directory
relative to `source_root`; here `.` selects snapshots directly inside
`inputs/snapshots/`. For several simulation setups, use entries such as
`jobA: jobA/run1` and `jobB: jobB/run1`. The scanner selects child directories
named `plt` followed by digits, in numeric snapshot order. Configured job
directories must exist inside `source_root`; an empty job is reported as
incomplete. Paths in this YAML follow the same rules as the emission
configuration.

```bash
uv run --no-sync python -m quokka2s.check_job_table_input_ranges
```

The command scans full snapshots using the same shared-z native-block reader and
statistics as the single-snapshot command. By default, one snapshot is scanned
at a time.
Set `snapshot_workers: 2` to scan two independent snapshots concurrently;
each worker uses one calculation thread and saves its own statistics. The
coordinator reuses completed scans, compares ranges with the tables, and writes
the shared summaries in inventory order. It does not calculate emission.
The bundled tables are defaults; optional `despotic_table` and `cloudy_table`
settings select other tables. `query_chunk` sets cells per in-memory statistics
batch. Batches are views of the current native layer; the command does not
reread files for each batch.

To scan a small batch with a soft time limit, add:

```yaml
snapshot_workers: 1
max_new_snapshots: 2
max_runtime_minutes: 120
```

`max_new_snapshots` counts newly launched scans, including failed attempts;
reusing completed statistics does not consume this limit. The command stops
launching new scans after `max_runtime_minutes`, but lets running scans finish
and save their results. It can therefore finish later than the requested time.
The command-line `--max-new-snapshots N` overrides the YAML limit. Omit both
limits to continue through all configured snapshots.

```text
output/table_input_ranges/
├── snapshots/
│   └── reference/
│       └── plt0655228/
│           ├── input_ranges.json
│           └── table_range_report.json
├── snapshots.csv
├── jobs.csv
└── summary.json
```

`snapshots.csv` records each snapshot's status and ranges: `complete`, `partial`,
`failed` or `not_selected`. `jobs.csv` and `summary.json` combine extrema only
from successfully completed full snapshots. They also report
whether the complete survey finished. The entire resolved `source_root` is
protected from output writes, and resolved output targets must remain inside
`output_dir`, including when directories contain symlinks.

Rerunning reuses completed snapshot statistics and compares them with the
current tables. An interrupted snapshot restarts its density prepass and scans all selected
z layers again.
Changed source files or physical definitions are rejected for reuse; choose a
new output directory rather than mixing old and new statistics.
Snapshot read failures are recorded and skipped; broken snapshot links, unsafe
outputs, stale checkpoints and output-write failures stop the survey.

To measure one full snapshot first:

```bash
uv run --no-sync python -m quokka2s.check_job_table_input_ranges --max-snapshots 1
```

This selects the first snapshot in the inventory, even if its statistics have
already been saved. To scan the next unfinished snapshot instead, use
`--max-new-snapshots 1`. Unscanned snapshots remain visible in the summary,
leaving the survey incomplete. Rerun with the same configuration to continue;
completed statistics are reused. `max_columns: 1` instead limits each selected
snapshot to one native XY block column. A scan that omits other columns is
partial; its extrema cannot certify full-snapshot coverage.

## Header ranges across multiple snapshots

To summarize stored extrema without reading cell payloads or querying tables:

```bash
uv run --no-sync python tools/snapshots/measure_header_ranges.py \
  --source-root /path/to/simulations \
  --runs jobA/run1 jobB/run1 \
  --output-dir output/header_ranges
```

The tool reads `Header`, `Level_0/Cell_H` and `metadata.yaml` for each `plt` plus
digits directory. It combines stored block extrema and converts density to
`nH` using the processing workflow's `X_H` and hydrogen mass. `snapshots.csv`
lists snapshot ranges; `jobs.csv` and `summary.json` give combined ranges and
report unreadable headers. Outputs must be new and outside the source tree;
the tool resolves symlinks and never writes to snapshots. A readable header
does not confirm binary payload completeness. This survey supports uniform
3-D level-zero snapshots with the cgs/K metadata and stored extrema format
used here. `NH`, `dV/dr` and hot-only input ranges still require a cell scan.

## Additional figures

Figure 1 and emission phase histograms are separate tools using the same
snapshot reader. Slice and projection maps query DESPOTIC temperature only;
emission phase histograms reuse the full cell-emission calculator:

```bash
uv run --no-sync python tools/figures/build_table_input_slice.py --config configs/emission.yaml \
  --output-dir output/sigma13.v2/plt0655228/table_input_slice --slice-index 216 --no-plot
uv run --no-sync python tools/figures/build_emission_phase_histograms.py --config configs/emission.yaml \
  --output-dir output/sigma13.v2/plt0655228/emission_phase_histograms --no-plot
```

Run either script again with the same `--output-dir` and `--plot-only` to draw
its saved arrays. Gas projection maps use `tools/figures/build_gas_projection_maps.py`
with the same config/output-dir/no-plot/plot-only pattern.

Additional dust and radiation-field figures:

```bash
uv run --no-sync python tools/figures/prepare_dust_extinction.py --output-dir output/dust_extinction
uv run --no-sync python tools/figures/plot_dust_extinction.py --output-dir output/dust_extinction
uv run --no-sync python tools/figures/prepare_radiation_fields.py --recipe components --components-only
uv run --no-sync python tools/figures/plot_radiation_components.py --components-only
uv run --no-sync python tools/figures/prepare_radiation_fields.py --recipe unattenuated
uv run --no-sync python tools/figures/plot_unattenuated_radiation.py
```

The radiation preparation command reads Cloudy `.inc` exports from `runtime/cloudy_eightline/sed/`.
Supply the required Cloudy `save incident continuum` exports before running
these tools. The exports are separate inputs, not included with the emission
table or Git repository; they are needed only for radiation-field figures.
Preparation saves the curves, sums, samples and numerical display ranges in
NPZ files. The plot commands read these NPZs only; `--data PATH` selects a
prepared file at another location. Add `--include-cmb` to both unattenuated
commands to use that separate recipe.
The prepared radiation file records its CMB choice; when using `--data`, labels
follow that saved recipe.

DESPOTIC table diagnostics follow the same preparation/drawing split:

```bash
uv run --no-sync python -m quokka2s.despotic.prepare_table_plots \
  --table inputs/tables/despotic/interpolated.npz
uv run --no-sync python -m quokka2s.despotic.plot_table
```

Preparation saves selected fields, masks, edges and contour segments in
`output/despotic_table_figures/prepared.npz`. Use `--indices 0 17` to select
dVdr slices, `--fields tg_final species:CO:lumPerH` to select fields, or
`--samples PATH.npy` for an explicit query-coordinate overlay. Drawing uses
`--data PATH` to select another prepared file. These diagnostic heatmaps
retain the original display policy of hiding solver-failure nodes, including
nodes subsequently filled by interpolation. This display mask does not change
the numerical table used for emission processing.


## Cold-gas nH–NH distributions across snapshot jobs

Edit `configs/cold_gas_phase.yaml`: `source_root` names the common simulation
root, and `jobs` maps readable output names to run directories below that root.
For example, `sigma13: job41101509-sigma13/run1` selects that run's direct
`plt` followed by digits directories, in numeric order. `output_dir` belongs
outside the simulation tree. Paths are relative to the YAML file.

```bash
uv run --no-sync python -m quokka2s.prepare_cold_gas_phase
uv run --no-sync python -m quokka2s.plot_cold_gas_phase --products output/cold_gas_phases
```

The preparation command accepts `--config PATH` and `--extension-config PATH`.
The latter defaults to `configs/despotic_extension.yaml`; its bounds extend the
original DESPOTIC coordinates at their existing spacing. These two-dimensional
outlines describe original and expanded table coordinate coverage, including
an expanded grid that is still being built. They do not claim that every node
converges or that every simulation cell has reliable emission.

Each worker reads only density and QUOKKA temperature, preserving the exact
complete positive/negative z columns and their harmonic mean. It selects
`T_QUOKKA < 3000 K`, saves count/mass histograms and all figure arrays, and never
queries an emission table. `snapshot_workers` controls independent snapshot
readers; numerical libraries use one thread per reader. Choose this count from
actual CPU/memory limits rather than the machine's total CPU count.

Fresh text headers can prove that a snapshot has zero cold cells. Such results
explicitly report zero payload cells scanned, and leave the unread total gas
mass and payload validity unknown. Other snapshots retain the full z depth.
Input opens are read-only, source writes are rejected, and all original file
sizes/mtimes are checked before/after. Complete caches require the same source,
physics and bin edges; changing table outlines only rebuilds figure arrays.

Numerical results are saved in `snapshots/JOB/PLT/`, with a separate
`jobs/JOB/` histogram that sums snapshot populations once each without time
weighting. Plotting reads these saved arrays and writes English figures in
`figures/png/` and `figures/pdf/` beneath each result. Each figure shows the cold
distribution at useful scale and a second view containing both table outlines.
Use `--weight mass` to draw the saved cold-gas mass fractions instead of counts.
