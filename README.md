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

## Run with existing tables

From the repository root, install once in the chosen Python environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Use Python 3.11 for this pinned environment. `requirements.txt` includes the exact
yt source commit used to read QUOKKA
snapshots. The editable install (`-e`) uses this checkout directly; keep the
checkout because processing also reads the pinned dust-opacity file in
`vendor/draine/`. DESPOTIC itself and a Cloudy executable are needed only to
build tables, not to process an existing pair of tables.

For running processing on Setonix and plotting on the laptop, follow the
[Setonix guide](docs/setonix_process.md). The
[batch script](tools/setonix/run_process.sbatch) runs the same process command
with the existing tables; it does not build tables or draw figures.

Copy the snapshot and tables separately from the code. They are not in Git:

```text
inputs/
  snapshots/plt0655228/
  tables/despotic/raw.npz
  tables/despotic/interpolated.npz
  tables/cloudy/emission.npz
```

The raw DESPOTIC table preserves the solver results. `process` queries
`interpolated.npz`, which fills unavailable nodes only inside the finite-data
convex hull. The emission run reads only the interpolated DESPOTIC table.

1. Edit [emission_process.yaml](configs/emission_process.yaml).
2. Run processing:

   ```bash
   python -m quokka2s.process_snapshot --config configs/emission_process.yaml
   ```

3. Set `products` in [emission_plot.yaml](configs/emission_plot.yaml) to the completed
   process output directory. Run plotting:

   ```bash
   python -m quokka2s.plot_emission_results --config configs/emission_plot.yaml
   ```

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
