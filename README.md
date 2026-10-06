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
tools/         table-building and additional figure commands
docs/          usage instructions, physical methods and code walkthrough
vendor/        pinned external source and physical data
inputs/        user-provided snapshots and the two bundled lookup tables
output/        processed numerical products and figures (local)
runtime/       intermediate Cloudy build files and logs (local)
```

See the [directory guide](docs/repository_layout.md) for file locations and
a suggested reading order.

The current workflow makes LOS-z images, integrated spectra, and gas-phase
velocity distributions for the full box or a selected x-y region. It keeps the
native cell resolution and full z depth. Images may be binned later for plotting.

## Set up a new environment and run with existing tables

You need Git and Python 3.11. The shell commands below work on macOS and Linux.

### 1. Clone the code

Choose a working directory and clone the repository:

```bash
git clone https://github.com/bcTiann/quokka_postprocess_3D.git
cd quokka_postprocess_3D
```

The two prebuilt emission tables are included in this clone. Supply your own
QUOKKA snapshot as described in step 3.

### 2. Create the Python environment

From the repository root, create a project-local Python environment and
install the package:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

`.venv` is the environment directory; its leading dot makes it hidden from
plain `ls`. `source .venv/bin/activate` makes the current terminal use that
environment's Python and packages. When opening a new terminal, return to the
repository and activate it again:

```bash
cd /path/to/quokka_postprocess_3D
source .venv/bin/activate
```

Run `deactivate` to leave the environment.

`requirements.txt` pins the numerical dependencies and the yt source commit
used to read QUOKKA snapshots. The editable installation (`-e`) uses this
checkout directly, so keep it. The dust-opacity table is included in
`vendor/draine/`. Processing prebuilt tables does not invoke DESPOTIC,
Cloudy or RADMC-3D executables.

### 3. Configure your existing snapshot

The default input paths, relative to the repository root, are:

```text
inputs/
  snapshots/plt0655228/
  tables/despotic/interpolated.npz
  tables/cloudy/emission.npz
```

| Input | Path | Provided by |
|---|---|---|
| Simulation snapshot | `inputs/snapshots/plt0655228/` | The user |
| DESPOTIC table | `inputs/tables/despotic/interpolated.npz` | This repository |
| Cloudy table | `inputs/tables/cloudy/emission.npz` | This repository |

The simulation is assumed to already exist on your machine. Either place its
complete directory at the default path or set `dataset` in
[configs/emission_process.yaml](configs/emission_process.yaml) to its existing
absolute path:

```yaml
dataset: /absolute/path/to/plt0655228
```

Point to the snapshot directory containing `Header`, `metadata.yaml`, and the
data subdirectories. The two table paths already work after cloning; processing
uses the filled DESPOTIC table, so the raw table and builder checkpoints are not
required.

These fixed tables were prepared for the `plt0655228` reference simulation.
DESPOTIC records its original grid and measured coordinate range. For a different
simulation, confirm table coverage and use the [table-building commands](#build-tables)
when new tables are needed.

### 4. Run processing

From the repository root, with `.venv` active:

```bash
python -m quokka2s.process_snapshot
```

Run these commands from the repository root. Processing uses
`configs/emission_process.yaml` by default, and plotting uses
`configs/emission_plot.yaml`. Use `--config path/to/settings.yaml` to select
another configuration. After installing the package, the equivalent short
commands are `quokka2s-process` and `quokka2s-plot`.

### 5. Plot the saved results

After processing finishes, run:

```bash
python -m quokka2s.plot_emission_results
```

The default plot config reads `output/plt0655228/processed/` and writes figures
to `output/plt0655228/figures/` and `output/plt0655228/figures_titled/`, each with
`png/` and `pdf/` subdirectories. To plot on another machine, copy the completed
processed directory there and set `products` in
[configs/emission_plot.yaml](configs/emission_plot.yaml) to that directory.
Plotting reads the saved products and does not reopen the snapshot or tables.

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
run for debugging. Omit it to finish the selected region or full snapshot.

### Select an x-y region

Copy the process YAML and add native cell-index ranges:

```yaml
x_index_range: [64, 128]
y_index_range: [32, 96]
output_dir: ../output/plt0655228/region_x64_128_y32_96/processed
```

Indices start at zero and the stop is excluded: this example selects
`64 × 64 × 2048` cells. An omitted axis uses its full extent; z always spans
the original box. No cell averaging is performed. Gradients use neighbours
from the original simulation, and both shielding and dust columns use the
complete original z column.

Run the copied YAML with the same process command. In a copied plot YAML,
point `products` at this region's processed directory and choose new figure
directories. The region's images retain their physical x/y coordinates.
Its spectra, full-profile sigma, and gas-phase statistics use only selected
cells; spectra per projected area use the region's x-y area. A finished region
is a complete result and does not require `allow_partial`.
Plot-time pixel binning must divide both selected image dimensions; the default
factor of 1 works for any region. The separate slice, multiview-map and emission
phase-histogram tools still use full-box configurations.

## What processing saves

For the current 256 × 256 × 2048 snapshot and ten transitions:

| File | Main contents | Shape and units |
|---|---|---|
| `images.npz` | Intrinsic and dust-attenuated LOS-z luminosity images | `(2, 10, 256, 256)`, erg/s per pixel |
| `spectra.npz` | Whole-box spectra, cold/hot contributions, full-profile and window line moments | Spectra `(2, 10, 2, 400)`, erg/s/(km/s); full and window σ `(2, 10)`, km/s |
| `phase_velocity.npz` | Mass velocity histograms and moments for five gas phases and all gas | Histograms `(6, 400)`, g per channel |
| `emission_report.json` | Input paths, snapshot grid, physical settings, cell counts, omitted mass, and luminosity checks | Human-readable run record |
| `status.json` | Running, failed, partial, or completed status | Progress and elapsed time |

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
and gas-phase/line-profile comparisons as PNG and PDF. Each figure directory
contains `png/` and `pdf/` subdirectories, including `titled_output_dir` when set.
For example, `output_dir/png/spectrum_co10.png` and
`output_dir/pdf/spectrum_co10.pdf` contain the two formats of the same figure.
The displayed velocity range is −50 to +50 km/s; all saved channels remain
intact. H I 21 cm appears once because this dust prescription leaves it unchanged. Set
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
DESPOTIC nodes after an interruption. Inputs, solver settings and source
identity must match. Use a new checkpoint directory when changing them.
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
  line_definitions.py       the ten lines' wavelengths and emitter masses
  result_metadata.py        pixel geometry and ordered dust metadata
  processing_report.py      builds the readable processing report
  result_files.py           writes finished products and run status
  emission_results.py       reads saved products by line and phase name
  constants.py              physical constants in the chosen units
  physics/                  derived fields, line emissivities, dust
  cloudy/                   three-dimensional Cloudy lookup and cell queries
  despotic/                 DESPOTIC building, interpolation and lookup
  products/                 numerical images, spectra and gas statistics
  figures/                  plotting functions
```

The ten lines share one definition file. Dust attenuation, thermal widths, and
UV plot labels read it instead of maintaining separate wavelength/mass lists.
Cloudy input tokens remain in its table-building definition.

Saved results can also be read without opening the snapshot or lookup tables:

```python
from quokka2s.emission_results import read_emission_results

results = read_emission_results("output/plt0655228/processed")
image = results.images.for_line(line="halpha", dust_state="attenuated")
spectrum = results.spectra.for_line(line="halpha", dust_state="attenuated")
cnm = results.gas_phases.for_phase(phase="CNM")
```

`image` contains luminosity per native pixel in erg/s. `spectrum` contains its
saved velocity channels, profile, and separate full/window moments. `cnm` uses
the gas-phase file's own velocity coordinates. Selecting a single temperature
regime uses its exact saved name, such as `T_QUOKKA_ge_3000K`; full moments are
saved for total lines only.

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
with the same config/output-dir/no-plot/plot-only pattern.


Additional dust and radiation-field figures:

```bash
python tools/figures/plot_dust_extinction.py
python tools/figures/plot_radiation_components.py --components-only
python tools/figures/plot_unattenuated_radiation.py
```

The radiation tools read Cloudy `.inc` exports from `runtime/cloudy_eightline/sed/`.
Supply the required Cloudy `save incident continuum` exports before running
these tools. The exports are separate inputs, not included with the emission
table or Git repository; they are needed only for radiation-field figures.
