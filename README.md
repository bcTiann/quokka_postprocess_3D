# quokka2s — QUOKKA emission post-processing

`quokka2s` reads a QUOKKA simulation snapshot, uses prebuilt DESPOTIC and
Cloudy tables to calculate line emission, and saves numerical products.
A separate command draws figures from those products.

```text
snapshot + tables → process → numerical files → plot → PNG/PDF
```

## Repository layout

```text
configs/       process and plot YAML settings
src/quokka2s/  reusable calculations and the two main entry points
tools/         table-building and additional manuscript-figure commands
tests/         tests for the current workflow
docs/          current method, code walkthrough, and validation records
vendor/        pinned external source and physical data
inputs/        snapshots and final lookup tables (local, outside Git)
output/        processed products, figures, and manuscripts (local)
runtime/       intermediate Cloudy build files and logs (local)
archive/       inactive code/notes; old large local data in archive/local/
```

See the [directory guide](docs/repository_layout.md) for individual files and
where historical material was moved. The [cleanup validation](docs/validation/repository_reorganization_validation.md)
records complete numerical and figure comparisons.

The current workflow makes LOS-z images, whole-box spectra, and gas-phase
velocity distributions. It keeps all native simulation cells during processing.
Images may be binned later for plotting.

## Set up a new environment and run with existing tables

### 1. Clone the code

On Setonix, use `/scratch/pawsey0807/btian` as the working directory for
this run. Clone before uploading the data, so the upload destination is
already the repository directory:

```bash
mkdir -p /scratch/pawsey0807/btian
cd /scratch/pawsey0807/btian
git clone --branch codex/cloudy-emission-backup \
  https://github.com/bcTiann/quokka_postprocess_3D.git
cd quokka_postprocess_3D
```

On another machine, clone the same branch into your chosen working directory.

### 2. Create the Python environment

Use Python 3.11. On Setonix, run `module spider python` and load an available
Python 3.11 module using `module load python/VERSION`, replacing `VERSION`
with the module version actually listed. If Python 3.11 is already active,
continue directly below.

Create a new environment on Setonix and install the dependencies once:

```bash
python -m venv /software/projects/pawsey0807/btian/venvs/quokka2s-py311
source /software/projects/pawsey0807/btian/venvs/quokka2s-py311/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

For another machine, choose a local path for the environment; the two
installation commands are the same. On later Setonix logins, load the same
Python module and activate the existing environment again.

`requirements.txt` pins the numerical dependencies and the exact yt source
commit used to read QUOKKA snapshots. The editable installation (`-e`) uses
this checkout directly, so keep it. Processing also reads the dust table
already included in `vendor/draine/`.

Processing existing tables does not require DESPOTIC itself, a Cloudy
executable or RADMC-3D. These solvers are not invoked by the process command.

### 3. Upload the snapshot and the two tables

These inputs are outside Git. Run the following on the laptop, after cloning
on Setonix. The SSH alias `setonix` uses the existing Pawsey login/key settings;
the file transfers use Pawsey's data-mover with the same key:

```bash
cd /Users/tianbaochen/quokka_postprocess_3D
setonix_repo=/scratch/pawsey0807/btian/quokka_postprocess_3D

ssh setonix "mkdir -p \
  ${setonix_repo}/inputs/snapshots/plt0655228 \
  ${setonix_repo}/inputs/tables/despotic \
  ${setonix_repo}/inputs/tables/cloudy"

# Complete simulation snapshot, including its Header and data subdirectories.
rsync -rvP --exclude='.DS_Store' \
  -e "ssh -i $HOME/.ssh/pawsey_ed25519_key" \
  inputs/snapshots/plt0655228/ \
  "btian@data-mover.pawsey.org.au:${setonix_repo}/inputs/snapshots/plt0655228/"

# DESPOTIC table after filling gaps within the finite-data convex hull.
rsync -rvP \
  -e "ssh -i $HOME/.ssh/pawsey_ed25519_key" \
  inputs/tables/despotic/interpolated.npz \
  "btian@data-mover.pawsey.org.au:${setonix_repo}/inputs/tables/despotic/"

# Cloudy emission table.
rsync -rvP \
  -e "ssh -i $HOME/.ssh/pawsey_ed25519_key" \
  inputs/tables/cloudy/emission.npz \
  "btian@data-mover.pawsey.org.au:${setonix_repo}/inputs/tables/cloudy/"
```

`-r` copies subdirectories, `-v` prints transferred files, and `-P` shows
progress and retains partial files after an interruption. Re-run the same
command if a transfer is interrupted. These commands intentionally do not
preserve old modification times, following
[Pawsey's scratch-transfer guidance](https://pawsey.atlassian.net/wiki/spaces/US/pages/51925882).

The resulting inputs are:

```text
inputs/
  snapshots/plt0655228/
  tables/despotic/interpolated.npz
  tables/cloudy/emission.npz
```

The raw DESPOTIC table, checkpoints and historical caches are not needed.
The supplied [process YAML](configs/emission_process.yaml) already points to
these relative paths. Its `output_dir` must be a new directory.

### 4. Run processing

On Setonix, use the interactive debug allocation supplied by the supervisor:

```bash
alias cpush="salloc --nodes=1 --time=00:59:00 -A pawsey0807 --mem=222GB -p debug"
cpush --ntasks=1 --cpus-per-task=8
```

The alias is unchanged. The added options request one Python process with
eight CPUs for the current two query workers and three spectral threads per
worker. After the allocation is granted, run:

```bash
cd /scratch/pawsey0807/btian/quokka_postprocess_3D
source /software/projects/pawsey0807/btian/venvs/quokka2s-py311/bin/activate

# The Python worker settings control parallelism; avoid extra native BLAS threads.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --distribution=block:block:block --cpu-bind=cores \
  python -u -m quokka2s.process_snapshot --config configs/emission_process.yaml
```

`salloc` reserves the resources; `srun` launches the process using them.
See [Pawsey's interactive-job instructions](https://pawsey.atlassian.net/wiki/spaces/US/pages/51925964).
When processing finishes, `exit` releases the interactive allocation.

On the laptop, run the same process command directly in its Python environment:

```bash
python -m quokka2s.process_snapshot --config configs/emission_process.yaml
```

### 5. Download the results and plot locally

After the Setonix process finishes, run on the laptop:

```bash
cd /Users/tianbaochen/quokka_postprocess_3D
mkdir -p output/plt0655228/processed_setonix
rsync -rvP \
  -e "ssh -i $HOME/.ssh/pawsey_ed25519_key" \
  btian@data-mover.pawsey.org.au:/scratch/pawsey0807/btian/quokka_postprocess_3D/output/plt0655228/processed/ \
  output/plt0655228/processed_setonix/
```

Set `products: ../output/plt0655228/processed_setonix` in
[configs/emission_plot.yaml](configs/emission_plot.yaml), then run:

```bash
python -m quokka2s.plot_emission_results --config configs/emission_plot.yaml
```

Plotting reads the downloaded products; it does not reopen the snapshot or
tables. The separate download directory keeps the existing laptop results.

The process YAML specifies these input and output paths:

| Setting | Meaning |
|---|---|
| `dataset` | Complete QUOKKA snapshot directory |
| `despotic_table` | Interpolated DESPOTIC table |
| `cloudy_table` | Eight-line Cloudy table |
| `output_dir` | New directory for the numerical products |

Relative paths are resolved from the YAML file's directory. Processing refuses
to overwrite an existing output directory; choose a new name for each run.
An interrupted emission run starts again from the beginning in a new directory.
Table-building checkpoints are a separate mechanism.

Optional execution settings control memory and concurrency, not the physics:

| Setting | Meaning | Supplied YAML |
|---|---|---|
| `slab_nx` | Number of x cells read per slab, retaining full y and z | 8, by default |
| `query_chunk` | Maximum cells in one table-query batch | 1,000,000 |
| `chunk_workers` | Concurrent query batches within the loaded slab | 2 |
| `spectral_workers` | Spectral-integration threads per query batch | 3 |

The worker counts multiply: this configuration permits up to 2 × 3 spectral
threads. Reading slabs remains sequential. More workers do not necessarily
make a complete run faster; select them using end-to-end timing and memory
measurements. `max_slabs: 1` in a separate YAML makes a labelled partial
run for debugging. Omit it for the full snapshot.

## What processing saves

For the current 256 × 256 × 2048 snapshot and ten transitions:

| File | Main contents | Shape and units |
|---|---|---|
| `images.npz` | Intrinsic and dust-attenuated LOS-z luminosity images | `(2, 10, 256, 256)`, erg/s per pixel |
| `spectra.npz` | Whole-box spectra, cold/hot contributions, full-profile and window line moments | Spectra `(2, 10, 2, 400)`, erg/s/(km/s); full and window σ `(2, 10)`, km/s |
| `phase_velocity.npz` | Mass velocity histograms and moments for five gas phases and all gas | Histograms `(6, 400)`, g per channel |
| `emission_report.json` | Input paths, snapshot grid, physical settings, cell counts, omitted mass, and luminosity checks | Human-readable run record |
| `status.json` | Running, failed, partial, or completed status | Progress and elapsed time |

The two dust states are intrinsic and attenuated; the two regimes are selected
by `T_QUOKKA < 3000 K` and `T_QUOKKA >= 3000 K`. Spectra have 400 channels
covering −200 to +200 km/s. `line_sigma_full_kms` includes each cell's complete
thermal Gaussian, including emission outside those channels.
`line_sigma_window_kms` uses the saved channels only. Both are saved with their
centroids; figure legends use the full-profile sigma. Luminosity outside the
window is recorded separately.

Processing reads one x slab at a time, queries table batches within it, and
adds their luminosities to the image and spectrum arrays. It discards the
cell arrays after each slab. No full-snapshot emissivity cube is saved.

Each line omits only cells with unavailable results required by that line;
image and spectrum use the same selection for that line. The processor reports
the missing count and mass separately for each line. Failed lookups
are not interpreted as physical zero emission. Before saving, it checks
cell accounting, luminosity sums, channel integrals and full-profile moments. The report
records the DESPOTIC, Cloudy, and dust table paths under `input_tables`, the
grid shape and cell widths under `snapshot_grid`, and the density/column
settings under `physical_settings`. See the
[physics reference](docs/emission_method.md).

## Plot locally or on another machine

Copy the completed numerical-product directory to the laptop. Plotting needs
those NPZ files and the package; it does not reopen the snapshot or tables.

| Plot setting | Meaning |
|---|---|
| `products` | Directory containing the three NPZ products |
| `output_dir` | Figures without titles, for manuscript captions |
| `titled_output_dir` | Optional second directory of titled figures |
| `image_downsample_factor` | Optional pixel binning factor; default 1 |

A factor of 2 sums each 2 × 2 group of image pixels into one displayed pixel.
It does not average luminosities or change the saved native-resolution data.
Intrinsic and attenuated images of one line share a colour range from
`vmax/1e5` to `vmax`, measured after any display binning.

The command draws individual line images, intrinsic/dust spectrum comparisons,
and gas-phase/line-profile comparisons as PNG and PDF. The displayed velocity
range is −50 to +50 km/s; all saved channels remain intact. H I 21 cm appears
once because this dust prescription leaves it unchanged. Set
`allow_partial: true` only when intentionally plotting diagnostic subsets.

## Build tables

Table building is separate from snapshot processing. Install the optional
DESPOTIC dependency, apply the checked patches, measure the snapshot's table
coordinates, then build and interpolate:

```bash
python -m pip install -e ".[tables]"
python tools/despotic/patch_gow_composition.py
python tools/despotic/install_checked_gow_integration.py
python tools/despotic/measure_snapshot_domain.py
python -m quokka2s.despotic.build_table \
  --snapshot-domain output/plt0655228/table_build/snapshot_domain.json \
  --checkpoint-dir output/plt0655228/table_build/checkpoints \
  --workers 1
python -m quokka2s.despotic.interpolate_failed
```

`--snapshot-domain` is required: all three axis endpoints come from the
snapshot measurement. The current builder uses 35 × 35 × 53 logarithmic
nodes. The builder writes `inputs/tables/despotic/raw.npz`; interpolation writes
`inputs/tables/despotic/interpolated.npz`. A failed solver node remains marked
in the original `failure_mask` even if interpolation supplies a value.
`remaining_unavailable_*` masks describe availability after filling. Finite
high-temperature solutions are retained; there is no `10^6 K` deletion rule.

Use the same build command and checkpoint directory to resume completed
DESPOTIC nodes after an interruption. Inputs, solver settings, and source identity must match. Existing checkpoints
made before a source reorganization may be rejected; use their archived source
for a historical build, or start a new checkpoint directory for the current code.
Set `--workers` to the CPUs allocated to the build job.

Cloudy requires a separately installed Cloudy 17.02 executable:

```bash
python tools/cloudy/build_cloudy_emission_table.py \
  --cloudy-exe /absolute/path/to/cloudy.exe --workers 1
```

This builder handles eight atomic lines.
Runtime inputs/logs go to `runtime/cloudy_eightline/`; the final table is
`inputs/tables/cloudy/emission.npz`. Both table builders require `--force` to replace
existing final tables. The [physics reference](docs/emission_method.md)
summarizes the table-query rules used during emission processing.

## Read the code

Start with the [step-by-step code guide](docs/code_walkthrough.md). The main files are:

```text
src/quokka2s/
  process_snapshot.py       runs the processing stages in order
  plot_emission_results.py  runs plotting from saved products
  run_settings.py           reads process/plot YAML settings
  processing_inputs.py      opens the snapshot and emission tables
  snapshot_reader.py        snapshot → slabs → cell batches
  result_files.py            saves numerical products and reports
  constants.py              physical constants in the chosen units
  physics/                  derived fields, line emissivities, dust
  cloudy/                   three-dimensional Cloudy lookup and cell queries
  despotic/                 DESPOTIC building, interpolation and lookup
  products/                 numerical images, spectra and gas statistics
  figures/                  plotting functions
```

Figure 1 and emission phase histograms are separate tools using the same
snapshot reader and cell-emission functions:

```bash
python tools/figures/build_table_input_slice.py --config configs/emission_process.yaml \
  --output-dir output/plt0655228/table_input_slice --slice-index 216 --no-plot
python tools/figures/build_emission_phase_histograms.py --config configs/emission_process.yaml \
  --output-dir output/plt0655228/emission_phase_histograms --no-plot
```

Run either script again with the same `--output-dir` and `--plot-only` to draw
its saved arrays. Gas projection maps use `tools/figures/build_gas_projection_maps.py`
with the same config/output-dir/no-plot/plot-only pattern. These tools are
independent of the old Task/Context and intermediate-cache framework, whose
source is preserved in the [history archive](archive/README.md).


Additional dust and radiation-field figures:

```bash
python tools/figures/plot_dust_extinction.py
python tools/figures/plot_radiation_components.py --components-only
python tools/figures/plot_unattenuated_radiation.py
```

The radiation tools read Cloudy `.inc` exports from `runtime/cloudy_eightline/sed/`.
The unattenuated and wider-column exports used in the manuscript are preserved
there locally; they are separate from the production emission grid. These files
are needed only for the additional radiation figures, not emission processing.
