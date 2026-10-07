# Repository directories and reading order

The repository has one current emission workflow. Process reads the snapshot
and fixed lookup tables; plot reads the saved numerical products. Table-building
tools and additional paper figures have their own commands.

## Where each kind of file belongs

| Directory | Contents | In Git? |
|---|---|---|
| `configs/` | Process and plot YAML files; paths are relative to these files | Yes |
| `src/quokka2s/` | Reusable calculations and process/plot entry points | Yes |
| `tools/cloudy/` | Build radiation inputs, run Cloudy, package its emission table | Yes |
| `tools/despotic/` | Install required GOW corrections, measure table axes, check coverage | Yes |
| `tools/figures/` | Additional slice, phase, gas-projection, dust and radiation figures | Yes |
| `docs/` | Usage instructions, physical methods and code walkthrough | Yes |
| `vendor/` | Pinned external source, dust table, molecular data and their notes | Yes |
| `inputs/snapshots/` | User-provided QUOKKA snapshots | No |
| `inputs/tables/` | Fixed Cloudy `emission.npz` and DESPOTIC `interpolated.npz` | Yes; raw tables and other build files stay local |
| `output/` | Processed numerical products and figures | No |
| `runtime/` | Cloudy build parameters, `.inc` exports, raw outputs and logs | No |

## Start reading the current pipeline

1. [process configuration](../configs/emission_process.yaml): choose the snapshot,
   two tables and numerical output directory.
2. [process_snapshot.py](../src/quokka2s/process_snapshot.py): read inputs, process
   slabs, build/check/save results.
3. [snapshot_reader.py](../src/quokka2s/snapshot_reader.py): read one slab with
   periodic x/y neighbours. `slab_windows(x_start=..., x_stop=..., slab_nx=...)`
   yields global core bounds; `slab.iter_batches(batch_size=...)` yields native
   cell views. Consume batches before releasing the slab: a retained view keeps
   its underlying arrays alive.
4. [cell_emission.py](../src/quokka2s/physics/cell_emission.py): obtain each named
   line's emissivity and temperature, then apply foreground dust.
5. [emission_products.py](../src/quokka2s/products/emission_products.py): send the
   same calculated batch to image, spectrum and gas-phase accumulators.
6. [plot_emission_results.py](../src/quokka2s/plot_emission_results.py): draw the
   saved products using the [plot configuration](../configs/emission_plot.yaml).

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
| `despotic/` | `solver_settings.py`: adopted solver defaults/source identities; `table_data.py`: table records/field names; `snapshot_domain.py`: recorded geometry/settings/bounds checks |
| `despotic/` | `table_files.py`: NPZ serialization; `cell_solver.py`: equilibrium at one state; `table_builder.py`: grid calculation |
| `despotic/` | `lookup.py` and `cell_fields.py`: process-time queries; `build_table.py`, `interpolate_failed.py`, `plot_table.py`, `list_failures.py`: table commands |
| `products/` | `line_luminosity_images.py`, `integrated_spectra.py`, `line_velocity_moments.py`, `gas_phase_velocity.py`: numerical accumulation |
| `figures/` | `line_luminosity_images.py`, `line_spectra.py`, `gas_phase_spectra.py`: saved-product renderers; `line_labels.py`: titles; `figure_files.py`: PNG/PDF paths and saving; other modules: additional figures |

`gas_fields.mixed_gas_temperature_K()` supplies the cold-DESPOTIC/hot-QUOKKA
temperature rule to gas-phase statistics, projection maps and phase histograms.
`products/gas_phase_velocity.py` owns the phase names and temperature bounds;
`GasPhaseResults` reads the saved definitions for plotting. Presentation titles
belong to `figures/line_labels.py`. `file_provenance.py` provides file hashes for
table-build and diagnostic identities; process-time table checks use the
physical domain, not file hashes.

The root modules also separate the final processing steps:
`result_metadata.py` extracts selected pixel geometry and ordered dust values;
`processing_report.py` builds the readable summary; `result_files.py` writes
already-built arrays and the report. `run_settings.py` returns `ProcessSettings`
or `PlotSettings` from YAML, without changing the command-line defaults.

## Data and results

```text
inputs/
  snapshots/plt0655228/
  tables/despotic/raw.npz       # Local table-building output; not included in Git
  tables/despotic/interpolated.npz
  tables/cloudy/emission.npz

output/plt0655228/
  processed/
    images.npz
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
```

Processing writes a new directory per run. Plotting may redraw from the saved
products many times. Each figure version keeps PNG and PDF files in its own
`png/` and `pdf/` subdirectories. Raw Cloudy calculation files belong in `runtime/`;
process and plot read only the paths specified in their configuration files.
