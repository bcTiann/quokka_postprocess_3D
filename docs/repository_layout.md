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
| `docs/` | Current instructions, physics, code walkthrough, validation records | Yes |
| `vendor/` | Pinned external source, dust table, molecular data and their notes | Yes |
| `inputs/` | QUOKKA snapshots and final lookup tables | No; transfer separately |
| `output/` | Processed products, figures, manuscripts and run evidence | No |
| `runtime/` | Cloudy build parameters, `.inc` exports, raw outputs and logs | No |
| `archive/code/`, `archive/documents/` | Compressed historical source and notes | Yes |
| `archive/local/` | Old large caches, experiments, results and transfer files | No |

There is no active `scripts/`, `notebook/`, `documentation/`, `data/`, or
intermediate-cache directory at the repository root. Archived Python files are
inside compressed files, outside the import path.

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
| `figures/` | `emission_results.py`, `gas_phase_spectra.py`, `line_labels.py` and the additional figure modules: rendering |

The former `io.py`, `models.py`, `solver.py`, `builder.py`, `plotting.py` and
`products/accumulation.py` names were replaced by the specific names above.
Callers use the new names directly; old-name wrappers were not added.

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
  figures_titled/
```

The shorter Cloudy filename is the existing table with unchanged bytes, not
a rebuilt table. The current manuscript directories remain in `output/` so
their open source files and figure paths continue to work.

Processing writes a new directory per run. Plotting may redraw from it many
times. Raw Cloudy calculation files belong in `runtime/`; neither process nor
plot searches historical results or caches in `archive/`.

## Cleanup and verification plan

The 2026-10-06 cleanup follows these steps:

1. Freeze the dirty working tree, completed numerical products and figures.
2. Audit actual source imports, tool callers, tests and current manuscript
   figure tools; retain code with a current job.
3. Move active files and update all callers/configurations/instructions.
4. Pack obsolete Saha/CII, CIE, old temperature masks, fixed-axis migration,
   old single-line table and plotting experiments into recoverable archives.
5. Compare real cells at both periodic x boundaries and the middle of the box.
6. Run the current test suite and the complete process/plot commands; compare
   every saved numerical field and rendered figure with the frozen baseline.

The [archive index](../archive/README.md) records what was retired and where
old local data went. The [completed validation record](validation/repository_reorganization_validation.md)
reports exact numerical and image comparisons, including the separately excluded
pre-existing checkpoint-interruption test.
