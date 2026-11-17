# Usage settings and table building

Use the [README](../README.md) for cloning, environment setup, inputs, and the
two main commands. Run the commands below from the repository root.

## Source environment and dependencies

uv manages `.venv` from `pyproject.toml` and the committed `uv.lock`. It installs
`quokka2s` in editable mode automatically, so imports use the current checkout's
`src/`. Source edits take effect on the next run. No activation, `PYTHONPATH`
export, or manual `pip install -e .` is required. From the repository root:

```bash
uv sync
```

Sync automatically builds three small Cython extensions using a C compiler.
Build dependencies are isolated and pinned in `pyproject.toml`. Grid arrays,
physical settings and configuration remain Python inputs; editing them needs
no recompilation. Editing a `.pyx` file triggers rebuilding on the next sync/run.

`.python-version` selects Python 3.11. On Setonix, load the Python module and
add `--python "$(command -v python)"` to sync, as shown in the Setonix guide.
In later shells, return to the repository and use `uv run python process_snapshot.py`
or `uv run python plot_emission_results.py`. uv checks and synchronizes the environment automatically,
reusing the locked versions rather than reinstalling everything on each run.
The other Python tools below also work without an import-path export. For Setonix
environment commands, see the [Setonix guide](setonix.md).

The default dependencies support both process and plot, including the pinned
yt reader. `tables` adds the optional DESPOTIC solver and build tools.
For table building, use `uv sync --extra tables`.
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
Use `uv sync --locked` when an outdated lockfile should cause an error instead
of being updated. `uv run --no-sync ...` skips environment synchronization and
assumes it is already correct; ordinary process and plot runs do not need it.
Do not manually edit `uv.lock`.

## Shared configuration

Run the two independent scripts from the repository root:

```bash
uv run python process_snapshot.py --dataset /absolute/path/run1/plt0008703
uv run python plot_emission_results.py --dataset /absolute/path/run1/plt0008703
```

Process accepts exactly one snapshot containing `Header`. A run directory is
an error and is never expanded. Process does not import or invoke the plot
entry point. Plot reads saved numerical arrays and uses only the dataset folder
name to locate them; it does not open the original snapshot or emission tables.

[configs/emission.yaml](../configs/emission.yaml) provides shared defaults:

```yaml
jobname: reference_job
dataset: null
results_dir: ../output
x_index_range: null
y_index_range: null
```

With `dataset: null`, supply `--dataset` on the command line. A non-null YAML
dataset must also be an absolute path to one snapshot. `~` expands to the home
directory; shell variables are not expanded. `--jobname my-run` optionally
replaces the output group. Keep a distinct group for each simulation run.
Relative table and result paths start at the YAML file. Bundled tables remain
defaults when their path entries are omitted.

The scripts select `results_dir/jobname/snapshot_name/`. Process writes
`processed/`; plot writes the chosen `figures/` or `figures_titled/` styles.
Explicit x/y selections place both under the same region directory, for example
`region_x64-66_y64-66/`. They do not search for a latest snapshot or scan a run.

### Path rules

| Path | Resolution |
|---|---|
| CLI or YAML `dataset` | Absolute snapshot path after expanding `~` |
| Other YAML paths | Relative to the YAML file, or absolute |
| CLI `--config` | Relative to the terminal working directory, or absolute |
| Omitted bundled table paths | Anchored to the source checkout |

### Select an x-y region

```bash
uv run python process_snapshot.py \
  --dataset /absolute/path/run1/plt0008703 --jobname my-run \
  --x 64 66 --y 64 66
uv run python plot_emission_results.py \
  --dataset /absolute/path/run1/plt0008703 --jobname my-run \
  --x 64 66 --y 64 66
```

Each range includes START and excludes STOP. These commands select four full-z
columns. An omitted axis retains its YAML range; `null` means the full axis.
All selections must lie inside the actual grid. Repeat CLI selection overrides
when plotting; they do not persist or change YAML. Column densities still use
complete z columns, and gradients use neighbours in the original global grid.
Halo cells do not enter images or spectra.

Grid dimensions come from snapshot metadata. Rectangular level-zero layouts
and shorter edge FABs are supported; grid dimensions do not have to divide a
hardcoded tile or query size. For example, Nz=2028 is valid. Batches can end
with fewer than `query_chunk` cells. Multilevel AMR is a separate unsupported
layout, not another grid size. The native x and z dimensions need at least two cells for the existing
finite-difference implementation; a one-column selected region is valid.

### Completed-result reuse

Process reuses only completed, intact results with matching configuration,
input file metadata, table/source hashes and numerical runtime identity.
A selected region can be complete while `full_snapshot` is false. Conflicting,
incomplete and legacy outputs are preserved and reported. Use a fresh
`results_dir` for another calculation. Existing products remain drawable even
when process refuses to reuse them. There is no partial snapshot checkpoint.

## Processing settings

| Setting | Meaning | Default |
|---|---|---|
| `query_chunk` | Maximum cells per physical/Gaussian batch | 131072 |
| `chunk_workers` | Concurrent batches in one bounded wave | 16 |
| `z_read_workers` | Concurrent original FAB-field reads | 5 |
| `z_reader_threads` | Complete-column derivation tasks | 32 |
| `gaussian_grid` | Gaussian template geometry overrides | `{sigma_max: 8192.0}` |

Process sets one native BLAS/OpenMP thread by default before importing NumPy;
independent batches provide the outer parallelism. Explicit environment thread
settings remain respected. This avoids nesting a BLAS team inside every batch
when reducing column-contiguous luminosity arrays.

### Original native-column fields

The sole process reader reads selected original full-z XY tiles into RAM.
`np.fromfile` reads a complete intersecting FAB field, then NumPy copies the
selected values to z-contiguous RAM arrays. Required global x/y neighbour faces
are read separately and reused. A very small region can still read a complete
intersecting FAB field; it does not trigger conversion of a complete snapshot.

Only the current tile and bounded worker wave stay resident. Six derived
physical arrays replace/reuse raw allocations, and each tile is released before
the next read. Preflight estimates raw, derived, worker, neighbour and headroom
storage. The limit is 30,000,000,000 bytes (30 GB, approximately 27.94 GiB).
The process RSS guard catches excess high-water usage; lower
`query_chunk` or `chunk_workers` if needed. This is a memory guard, not proof
that every possible grid or FAB layout fits at every worker setting.

### Spectral integration

The only production method accumulates original Gaussian luminosities at four
stored fractional-mean/sigma template corners, then convolves completed rows.
There is no algorithm selector. Means use integer-channel shifts and fractional
phase; sigma nodes are logarithmically placed with **linear sigma weights**.
The observed grid remains 400 channels over [-200,200] km/s.

The supplied configuration covers sigma=[1e-5,8192] km/s at natural-log node
spacing 1/32, matching the validated compiled run. Omitting Gaussian settings
uses the engine fallback range [1e-5,1024] km/s. Set coverage before preparation:

```yaml
gaussian_grid:
  sigma_max: 8192.0
```

Each template stores its own five-sigma support plus zero guards, including
bins outside the final observation window that can shift into it. Valid
out-of-coverage sigmas use stable original-cell bin integration with the same
five-sigma cutoff. No nonzero luminosity is discarded by a brightness threshold,
and no truncated tail is renormalized. Logs/reports retain coverage counts,
width ranges and fallback luminosity by line, dust state and temperature branch.

Gaussian template preparation/validation is measured separately before process
timing. Metadata, original reads, physical lookups, four-corner accumulation,
convolution, checks and file saving are timed. Worker-call durations can overlap;
do not add their sums to wall-clock stages. The final status includes saving.
Environment synchronization builds the numerical extensions before processing;
there is no compilation inside the timed snapshot calculation.

### Memory, concurrency and progress

Each tile finishes physical preparation, corner preparation and profile-plane
accumulation in bounded waves. One writer owns each spectrum profile plane;
small products merge in input order. Original-cell luminosities are released
only after all wave writers finish. Final convolution runs after the last tile.
Logs print selected tile bounds and update status after completed tiles.

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
the original simulation geometry and effective `run_configuration` are also
retained. The other array dimensions
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

Default processing reads one native full-z tile, derives physical fields in
reused RAM allocations, then queries table batches and accumulates images,
spectra and gas phases. Cell arrays are released before the next tile; only
finished waves merge immediately in deterministic native-tile order. The explicit
z source reads bounded slabs after cache validation/conversion.
No full-snapshot emissivity cube is saved.
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
`results_dir` to the local results root and `jobname` to the selected job, or
override the group with `--jobname` for that invocation.
For a single plot, the basename of `dataset` selects the saved snapshot. Its
original Setonix path can remain unchanged;
plot does not require that path or the tables to exist locally. Retain the same
x/y ranges to select a downloaded region; omit them to select the full-box result.
Use the plotting script to redraw the entire output group without a dataset selection.

| Plot setting | Meaning |
|---|---|
| `figure_formats` | Nonempty list containing `png`, `pdf`, or both; default `[png]` |
| `figure_styles` | Nonempty list containing `titled`, `untitled`, or both; default `[titled]` |
| `image_downsample_factor` | Select an already prepared image resolution; default 1, native |
| `raw_luminosity` | Display spectra in luminosity units rather than luminosity per projected area; default false |
| `allow_partial` | Permit figures from an intentionally incomplete diagnostic run; default false |

For example, the default plotting selection is:

```yaml
figure_formats: [png]
figure_styles: [titled]
```

Replace the old `titled_figures` setting with `figure_styles`. These choices
apply to figures from `uv run python plot_emission_results.py`, including
luminosity images. Processing never draws figures.
The command-line equivalents are:

| Command | Formats | Styles |
|---|---|---|
| `uv run python plot_emission_results.py` | YAML value, otherwise PNG | YAML value, otherwise titled |
| `uv run python plot_emission_results.py --pdf` | PDF only | YAML value, otherwise titled |
| `uv run python plot_emission_results.py --png --pdf` | PNG and PDF | YAML value, otherwise titled |
| `uv run python plot_emission_results.py --untitled` | YAML value, otherwise PNG | Untitled only |
| `uv run python plot_emission_results.py --titled --untitled` | YAML value, otherwise PNG | Titled and untitled |

An explicitly selected category replaces its YAML list. For example,
`--pdf` does not also save PNG just because PNG is the default. Combine
`--png --pdf --titled --untitled` to write both formats of both styles.

Every titled figure includes the saved `jobname`, snapshot name and simulation
time, for example `sigma13.v2 | plt0655228 | time = 256.00 Myr`, where Myr means
million years. Process records the actual snapshot time as
`simulation_time_myr` in `processed/emission_report.json`. Plot reads only
this saved metadata; the untitled style keeps its existing appearance.

Older products without a saved time display `time = unavailable`. To add the
actual time without repeating the emission calculation, prepare the metadata
once from the matching original snapshot, then redraw only the titled style:

```bash
uv run python -m quokka2s.prepare_plot_metadata \
  --products /absolute/path/to/output/sigma13.v2/plt0655228/processed \
  --dataset /absolute/path/to/simulations/plt0655228
uv run python plot_emission_results.py --dataset /absolute/path/to/simulations/plt0655228 \
  --jobname sigma13.v2 --titled
```

The preparation command reads snapshot metadata and updates the saved report;
it preserves the numerical NPZ files. Add `--jobname sigma13.v2` to preparation
only when the legacy report has no saved jobname. Use the exact region's
`processed/` directory and repeat its x/y selection when redrawing a region.
For legacy results, prepare and plot each explicitly selected snapshot separately. Snapshot numbers and file dates are
never used to guess simulation time. Normal plotting continues to work without
the original snapshots after metadata preparation.

To use a factor of 2, prepare that resolution once from the saved native pixels:

```bash
uv run python -m quokka2s.prepare_emission_results --products /absolute/path/to/processed --image-downsample-factor 2
```

This writes `images_factor_2.npz` in the configured products directory, summing
each 2 × 2 group rather than averaging. It retains `images.npz` at native
resolution and does not read the snapshot. Set `image_downsample_factor: 2`
in the shared YAML to select it. For another product directory, use
`--products PATH` when preparing it.
This preparation tool does not accept `--dataset` or CLI x/y selections.
For results selected with process overrides, pass their complete
`processed/` directory with `--products`; otherwise it uses the YAML selection.

Intrinsic and attenuated images of one line share a colour range from
`vmax/1e5` to `vmax`, saved for that resolution. Plot only selects saved fields
and draws them; normalization, physical conversions and pixel sums are completed
before rendering. Products from an earlier code version can be prepared with
the same command without the optional factor, using only their saved arrays.

The command draws individual line images, intrinsic/dust spectrum comparisons,
gas-phase/line-profile comparisons, and all-lines spectrum comparisons.
Every spectral figure has separate −50 to +50 and −200 to +200 km/s views;
the existing 400 saved channels already cover the wider range. These views
do not change the stored arrays or recalculate spectra.

The all-lines comparison draws all ten cold+hot total profiles, with separate
intrinsic and dust-attenuated figures. Each line is divided by its own peak
across the full saved −200 to +200 km/s window before the products are saved.
The same normalized curve is used in both views; a peak outside ±50 km/s
can therefore leave the narrower view below one. Each line retains its colour
across views and dust states, and the legend sits outside the axes.

Selected PNG files go in `png/` and PDF files in `pdf/` under `figures_titled/`
or `figures/`, depending on the requested style. Examples include:

```text
figures_titled/png/spectrum_co10_v50.png
figures_titled/png/spectrum_co10_v200.png
figures_titled/png/gas_phase_spectrum_co10_v200.png
figures_titled/png/spectrum_all_lines_intrinsic_v200.png
figures_titled/png/spectrum_all_lines_attenuated_v200.png
```

Images are drawn once per selected style, rather than once per velocity view.
For the standard ten-line products, an empty output directory receives 63 PNG
files with the defaults, or 252 files when both formats and styles are selected.
Plot writes only the selected formats and styles; it does not remove files
from earlier runs. H I 21 cm appears once in the individual image/spectrum
comparisons because this dust prescription leaves it unchanged. Set
`allow_partial: true` only when intentionally plotting diagnostic subsets.

## Build tables

Table building is separate from snapshot processing. Install the optional
DESPOTIC dependency, apply the checked patches, measure the snapshot's table
coordinates, then build and interpolate:

```bash
uv sync --extra tables
uv run python tools/despotic/patch_gow_composition.py
uv run python tools/despotic/install_checked_gow_integration.py
uv run python tools/despotic/measure_snapshot_domain.py
uv run python -m quokka2s.despotic.build_table \
  --snapshot-domain output/plt0655228/table_build/snapshot_domain.json \
  --checkpoint-dir output/plt0655228/table_build/checkpoints \
  --workers 1
uv run python -m quokka2s.despotic.interpolate_failed
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
uv run python -m quokka2s.despotic.extend_table --dry-run
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
uv run python -m quokka2s.despotic.interpolate_failed \
  --source output/tables/despotic_extended/raw.npz \
  --output output/tables/despotic_extended/interpolated.npz
```

Directly solved old nodes remain unchanged. Old failed positions may receive
different interpolated values because the set of real supporting nodes has
grown. Keep the existing process tables until the new pair has been reviewed.

Cloudy requires a separately installed Cloudy 17.02 executable:

```bash
uv run python tools/cloudy/build_cloudy_emission_table.py \
  --cloudy-exe /absolute/path/to/cloudy.exe --workers 1
```

This builder handles eight atomic lines.
Runtime inputs/logs go to `runtime/cloudy_eightline/`; the final table is
`inputs/tables/cloudy/emission.npz`. Both table builders require `--force` to replace
existing final tables. The [physics reference](emission_method.md)
summarizes the table-query rules used during emission processing.

The bundled Cloudy table includes two lower-density planes; its previous
version is kept as `inputs/tables/cloudy/emission_old.npz`. To reproduce this
extension, use the incremental tool. It continues the source table's first
density interval, keeps its NH and temperature axes, and calculates only the
294 new states. Supply the parameter file recorded by the source table and
its existing incident-spectrum directory:

```bash
uv run python tools/cloudy/extend_cloudy_density.py plan \
  --source-table inputs/tables/cloudy/emission_old.npz \
  --output-table output/tables/cloudy_extended.npz \
  --recipe-parameter /path/to/original_7x10x21.par \
  --cloudy-exe /absolute/path/to/cloudy.exe \
  --radiation-directory /path/to/HM12_ATTENUATION_ISM_NH21 \
  --runtime-dir runtime/cloudy_density_extension --workers 4
```

Repeat the same arguments with `smoke`, then `build`, then `merge` in place of
`plan`. The smoke covers both new densities and every NH value at the source
temperature endpoints and 100,000 K. The build requires that smoke to pass for
the same input hashes; merge requires complete, successful production states.
Raw Cloudy outputs, exit statuses and warnings remain in the runtime directory.
The output path must be new. Existing node arrays, masks and metadata are
copied with exact dtype and byte checks; extension provenance describes only
the new planes. The original table is never overwritten by this tool.

## Check table-input ranges

This diagnostic still selects `dataset`, `jobname` and x/y ranges from YAML;
it does not inherit the last process command's dataset or CLI region.
Set the intended target in `configs/emission.yaml` (or a separate YAML passed
with `--config PATH`), then run:

```bash
uv run python -m quokka2s.check_table_input_ranges
```

The command uses `NativeBlockRangeReader`. An ascending density pass records
exact cumulative columns at each selected XY column's z-block faces. The
second pass visits z from top to bottom, processing all selected XY columns
at each z. Native fields are read once per z layer and reused for neighbouring
columns, then released. Each layer's inputs use the original column carries
and periodic velocity neighbours. Every sightline retains complete z-column
physics without a full-snapshot field array. The `nH` conversion, column and
velocity-gradient definitions match process, including the order of cumulative
additions. Emission processing retains six fields for one full-z XY tile and
merges its batches in input order. Its x/y selection and the range survey's
`max_columns` limit are independent.
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
uv run python -m quokka2s.check_table_input_ranges --max-columns 1
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
uv run python -m quokka2s.check_job_table_input_ranges
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
uv run python -m quokka2s.check_job_table_input_ranges --max-snapshots 1
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
uv run python tools/snapshots/measure_header_ranges.py \
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
uv run python tools/figures/build_table_input_slice.py --config configs/emission.yaml --dataset /absolute/path/to/plt0655228 \
  --output-dir output/sigma13.v2/plt0655228/table_input_slice --slice-index 216 --no-plot
uv run python tools/figures/build_emission_phase_histograms.py --config configs/emission.yaml --dataset /absolute/path/to/plt0655228 \
  --output-dir output/sigma13.v2/plt0655228/emission_phase_histograms --no-plot
```

Run either script again with the same `--output-dir` and `--plot-only` to draw
its saved arrays. Gas projection maps use `tools/figures/build_gas_projection_maps.py`
with the same config/output-dir/no-plot/plot-only pattern.

Additional dust and radiation-field figures:

```bash
uv run python tools/figures/prepare_dust_extinction.py --output-dir output/dust_extinction
uv run python tools/figures/plot_dust_extinction.py --output-dir output/dust_extinction
uv run python tools/figures/prepare_radiation_fields.py --recipe components --components-only
uv run python tools/figures/plot_radiation_components.py --components-only
uv run python tools/figures/prepare_radiation_fields.py --recipe unattenuated
uv run python tools/figures/plot_unattenuated_radiation.py
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
uv run python -m quokka2s.despotic.prepare_table_plots \
  --table inputs/tables/despotic/interpolated.npz
uv run python -m quokka2s.despotic.plot_table
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
uv run python -m quokka2s.prepare_cold_gas_phase
uv run python -m quokka2s.plot_cold_gas_phase --products output/cold_gas_phases
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
