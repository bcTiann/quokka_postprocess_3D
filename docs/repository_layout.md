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
| `tests/` | Current calculation and rendering tests; a small frozen numerical fixture | Yes |
| `docs/` | Usage instructions, physical methods and code walkthrough | Yes |
| `vendor/` | Pinned external source, dust table, molecular data and their notes | Yes |
| `inputs/` | QUOKKA snapshots and final lookup tables | No; transfer separately |
| `output/` | Processed numerical products and figures | No |
| `runtime/` | Cloudy build parameters, `.inc` exports, raw outputs and logs | No |

## Start reading the current pipeline

1. [process configuration](../configs/emission_process.yaml): choose the snapshot,
   two tables and numerical output directory.
2. [process_snapshot.py](../src/quokka2s/process_snapshot.py): read inputs, process
   slabs, build/check/save results.
3. [snapshot_reader.py](../src/quokka2s/snapshot_reader.py): read one slab with
   periodic x/y neighbours; produce batches of native cells.
4. [cell_emission.py](../src/quokka2s/physics/cell_emission.py): obtain each named
   line's emissivity and temperature, then apply foreground dust.
5. [emission_products.py](../src/quokka2s/products/emission_products.py): send the
   same calculated batch to image, spectrum and gas-phase accumulators.
6. [plot_emission_results.py](../src/quokka2s/plot_emission_results.py): draw the
   saved products using the [plot configuration](../configs/emission_plot.yaml).

The [code walkthrough](code_walkthrough.md) explains concrete arrays, units and
examples at each step. The [method reference](emission_method.md) states the
current scientific rules.

## Package names describe their jobs

| Directory | Main files and purpose |
|---|---|
| `physics/` | `gas_fields.py`, `line_emissivity.py`, `hydrogen_emissivity.py`, `dust_attenuation.py`: physical calculations |
| `cloudy/` | `lookup.py`: interpolation; `cell_fields.py`: prepare and restore batch queries; `table_definition.py`: line/grid settings |
| `despotic/` | `table_data.py`: table structures/field names; `table_files.py`: NPZ serialization; `cell_solver.py`: equilibrium at one state; `table_builder.py`: grid calculation |
| `despotic/` | `lookup.py` and `cell_fields.py`: process-time queries; `build_table.py`, `interpolate_failed.py`, `plot_table.py`, `list_failures.py`: table commands |
| `products/` | `line_luminosity_images.py`, `integrated_spectra.py`, `line_velocity_moments.py`, `gas_phase_velocity.py`: numerical accumulation |
| `figures/` | `emission_results.py`, `gas_phase_spectra.py`, `line_labels.py` and the additional figure modules: rendering; `figure_files.py`: PNG/PDF paths and saving |

## Data and results

```text
inputs/
  snapshots/plt0655228/
  tables/despotic/raw.npz
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
