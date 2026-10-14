# Usage settings and table building

Use the [README](../README.md) for cloning, environment setup, inputs, and the
two main commands. Run the commands below from the repository root.

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

```text
output/
├── sigma13.v2/
│   ├── plt0655228/
│   │   ├── processed/
│   │   ├── figures/
│   │   └── figures_titled/
│   └── plt0916720/
└── sigma50/
    └── plt0655228/
```

The two bundled tables are selected automatically:
`inputs/tables/despotic/interpolated.npz` and
`inputs/tables/cloudy/emission.npz`. These defaults refer to the installed
repository, even when the YAML file is elsewhere. Advanced configurations
may set `despotic_table`, `cloudy_table`, or `dust_opacity_table` explicitly.

The supplied YAML uses repository-local input and output paths. To store data
elsewhere, set `dataset` and `results_dir` to the desired locations. The
[Setonix guide](setonix.md) keeps the repository and bundled tables under
`$MYSOFTWARE`, and snapshots and results under `$MYSCRATCH`.

Relative paths are resolved from the YAML file's directory, so `../inputs/` in
`configs/emission.yaml` refers to this clone's `inputs/` directory.
`dataset` must point to the individual snapshot containing `Header`, rather
than its parent `inputs/snapshots/` directory. Processing refuses to overwrite
an existing `processed/` directory; choose another results root for a rerun
of the same job and snapshot.
An interrupted emission run starts again from the beginning in a new directory.
Table-building checkpoints are a separate mechanism.

Existing numerical products can be reused without rerunning process. Place
their complete directory at `results_dir/jobname/snapshot_name/processed/`,
then run plot with that job and snapshot selected.

### Path rules

| Where a path is supplied | Base for a relative path |
|---|---|
| Inside a YAML configuration | Directory containing that YAML file |
| Command-line options, including `--config` | Current terminal working directory |
| Bundled reference data | Repository containing the installed editable package |

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

The worker counts multiply: this configuration permits up to 2 × 3 spectral
threads. Reading slabs remains sequential. More workers do not necessarily
make a complete run faster; select them using end-to-end timing and memory
measurements. `max_slabs: 1` in a separate YAML makes a labelled partial
run for debugging. Omit it to finish the selected region or full snapshot.

## Select an x-y region

Copy the [shared YAML](../configs/emission.yaml) and set native cell-index ranges
and a separate results root:

```yaml
x_index_range: [64, 128]
y_index_range: [32, 96]
results_dir: ../output/region_x64_128_y32_96
```

Indices start at zero and the stop is excluded: this example selects
`64 × 64 × Nz` cells, where `Nz` is read from the snapshot (`2048` for the
reference snapshot). An omitted or `null` axis uses its full extent; z always spans
the original box. No cell averaging is performed. Gradients use neighbours
from the original simulation, and both shielding and dust columns use the
complete original z column.

Pass the same copied YAML to both commands:

```bash
python -m quokka2s.process_snapshot --config configs/emission_region.yaml
python -m quokka2s.plot_emission_results --config configs/emission_region.yaml
```

This example saves under
`output/region_x64_128_y32_96/sigma13.v2/plt0655228/`.
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

Copy the completed `processed/` directory to the laptop, preserving its job
and snapshot folders. In the same shared YAML, set `results_dir` to the local
results root and `jobname` to the selected job. The basename of `dataset`
selects the saved snapshot. Its original Setonix path can remain unchanged;
plot does not require that path or the tables to exist locally.

| Plot setting | Meaning |
|---|---|
| `titled_figures` | Also save titled figures in `figures_titled/`; default true |
| `image_downsample_factor` | Select an already prepared image resolution; default 1, native |
| `raw_luminosity` | Display spectra in luminosity units rather than luminosity per projected area; default false |
| `allow_partial` | Permit figures from an intentionally incomplete diagnostic run; default false |

To use a factor of 2, prepare that resolution once from the saved native pixels:

```bash
python -m quokka2s.prepare_emission_results --image-downsample-factor 2
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
existing final tables. The [physics reference](emission_method.md)
summarizes the table-query rules used during emission processing.

## Additional figures

Figure 1 and emission phase histograms are separate tools using the same
snapshot reader. Slice and projection maps query DESPOTIC temperature only;
emission phase histograms reuse the full cell-emission calculator:

```bash
python tools/figures/build_table_input_slice.py --config configs/emission.yaml \
  --output-dir output/sigma13.v2/plt0655228/table_input_slice --slice-index 216 --no-plot
python tools/figures/build_emission_phase_histograms.py --config configs/emission.yaml \
  --output-dir output/sigma13.v2/plt0655228/emission_phase_histograms --no-plot
```

Run either script again with the same `--output-dir` and `--plot-only` to draw
its saved arrays. Gas projection maps use `tools/figures/build_gas_projection_maps.py`
with the same config/output-dir/no-plot/plot-only pattern.

Additional dust and radiation-field figures:

```bash
python tools/figures/prepare_dust_extinction.py --output-dir output/dust_extinction
python tools/figures/plot_dust_extinction.py --output-dir output/dust_extinction
python tools/figures/prepare_radiation_fields.py --recipe components --components-only
python tools/figures/plot_radiation_components.py --components-only
python tools/figures/prepare_radiation_fields.py --recipe unattenuated
python tools/figures/plot_unattenuated_radiation.py
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
python -m quokka2s.despotic.prepare_table_plots \
  --table inputs/tables/despotic/interpolated.npz
python -m quokka2s.despotic.plot_table
```

Preparation saves selected fields, masks, edges and contour segments in
`output/despotic_table_figures/prepared.npz`. Use `--indices 0 17` to select
dVdr slices, `--fields tg_final species:CO:lumPerH` to select fields, or
`--samples PATH.npy` for an explicit query-coordinate overlay. Drawing uses
`--data PATH` to select another prepared file. These diagnostic heatmaps
retain the original display policy of hiding solver-failure nodes, including
nodes subsequently filled by interpolation. This display mask does not change
the numerical table used for emission processing.
