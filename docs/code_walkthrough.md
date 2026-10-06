# Reading `process` → saved data → `plot`

Start at `main()` near the bottom of [process_snapshot.py](../src/quokka2s/process_snapshot.py).
Read what each call receives and returns before opening the function itself.
Process and plot have independent entry points:

```bash
python -m quokka2s.process_snapshot
python -m quokka2s.plot_emission_results
```

From the repository root, each reads its own default YAML:
`configs/emission_process.yaml` or `configs/emission_plot.yaml`.
Use `--config PATH` to select another file. Each runs directly;
there is no shared dispatcher.

The two commands and table-build instructions are in the
[root README](../README.md). The scientific choices are in the
[physics reference](emission_method.md).

The package has five purpose-specific folders:

| Folder | Responsibility |
|---|---|
| `physics/` | Derived gas fields, emissivity branches, and dust attenuation |
| `cloudy/` | Three-dimensional Cloudy table lookup and cell queries |
| `despotic/` | DESPOTIC building, filling unavailable nodes, and table lookup |
| `products/` | Numerical image, spectrum, gas-phase, and projection accumulation |
| `figures/` | Draw figures from already calculated numerical arrays |

## 1. Follow the seven stages in `main()`

| Stage | Function | What it produces |
|---|---|---|
| Read settings | `load_process_config()` | Input/output paths and execution settings from YAML |
| Open inputs | `processing_inputs.load_processing_inputs(config)` | `snapshot`, `emission_calculator` |
| Create output arrays | `EmissionProducts(...)` | Initially empty image, spectrum, and gas-phase accumulators |
| Process cells | `process_snapshot()` | The accumulated numerical results |
| Build outputs | `products.build_outputs()` | Image, spectrum and gas-phase arrays and summaries |
| Check outputs | `products.check_outputs(outputs)` | Conservation/accounting checks; image-conservation diagnostics |
| Save | `result_files.write_products_and_report()` | Three NPZ files plus a JSON report and status |

The orchestration stays in [process_snapshot.py](../src/quokka2s/process_snapshot.py).
Input loading belongs to [processing_inputs.py](../src/quokka2s/processing_inputs.py), and output writing
belongs to [result_files.py](../src/quokka2s/result_files.py);
[products/emission_products.py](../src/quokka2s/products/emission_products.py) groups the accumulated products.
The objects group related variables; they are not extra copies of the snapshot:

- `snapshot` holds the yt dataset and an optional `xy_region` of native index
  bounds. `shape`, cell widths/volume, `cell_count`, and `projected_area_cm2`
  describe the original box. `processing_shape`, `processing_cell_count`,
  `processing_area_cm2`, and `processing_xy_origin` describe the selected region.
  `snapshot.read_slab()` loads cell fields; physical settings and function
  modules are not attributes.
- `emission_calculator` is a `CellEmissionCalculator`. It holds the DESPOTIC
  and Cloudy readers, ten line names, and named dust cross-sections. Each reader
  holds its fixed lookup table. These objects are created once and shared across
  batches; they do not store queried cell fields or emissivities.
- `slab` owns six field arrays, its global x position, local shape, and cell
  volume. `slab.batch(start, stop)` uses these to locate its zero-copy cell views.
- `cells` carries those batch views and their position. Its first/last full-grid
  cell IDs are calculated properties, so callers cannot provide conflicting IDs.
- `products` holds the running sums: luminosity images, spectra, gas-phase
  statistics, total luminosities, cell counts, and mass totals. It saves the
  expected selected-cell count and projected area at construction, so `build_outputs()`
  needs no arguments and products retain no dataset/table references.

Slab reads, table queries, and spectrum accumulation remain explicit actions.
Hydrogen density is cached on the current batch. A line's
`emissivity_is_missing` property is derived from its emissivity array, so there
is no second stored mask that can disagree with the values.

Inside `load_processing_inputs(config)`, follow three named steps:

1. `check_input_paths()` checks the configured input and output locations.
2. `open_snapshot()` opens yt, obtains the original grid geometry, and records
   any x/y selection from the YAML.
3. `load_emission_calculator()` loads the tables, creates their cell readers,
   and prepares the per-line dust cross-sections. It derives line names once
   from Cloudy plus `co10` and `co21`, then constructs `CellEmissionCalculator`.

The function creates the output directory after loading succeeds and returns:

```python
snapshot, emission_calculator = load_processing_inputs(config)
```

Physical constants come from [constants.py](../src/quokka2s/constants.py), which converts
Astropy constants once to the units used by NumPy arrays. Functions import
these constants directly; they are not passed through the calculator or constructors.
The adopted hydrogen mass is `1.007947 * const.u`, using Astropy's atomic mass
unit. The old CIAOLoop constants in the table-reuse
check describe how that table was built and remain historical values.

Opening checks cover missing inputs and numerical compatibility, including the
DESPOTIC table's recorded snapshot grid, density/column settings, and coordinate
bounds. Final checks verify cell accounting and luminosity conservation. Follow
the slab and batch calculations below before studying each check.

## 2. Read one slab: `snapshot_reader.read_slab()`

For the current grid, one slab contains eight x cells and all y,z cells:
`(8, 256, 2048)`, or 4,194,304 cells. There are 32 such slabs. The code reads
one extra x cell on each side where available so velocity derivatives can
use neighbouring cells. Only the eight core x cells contribute to products.
The x/y boundaries are periodic. For the first and last slabs, the reader
also loads the opposite x face's `velocity_x` plane. The full y axis is already
present in each slab. Both directions use centred differences with wrapped
neighbours; z retains one-sided differences at its two outer faces.
Full z columns are kept together for both column-density calculations.

Call `snapshot.read_slab(x_start=0, x_stop=8)` for the first eight x layers.
The caller supplies only the desired cells; the reader determines all neighbour
indices and removes those neighbours from its returned arrays.

Read `read_slab()` from top to bottom as these steps:

1. `read_slab_grid()` opens the slab with its neighbouring x layers.
2. `read_simulation_fields()` copies density, QUOKKA temperature and vz.
3. `calculate_dust_foreground_column()` calculates the column to the observer.
4. `calculate_shielding_column()` calculates the +z/-z harmonic-mean column.
5. `calculate_core_velocity_gradient()` calculates dV/dr using neighbours.
6. Release the yt grid; for a region, copy only the selected y rows after all
   column and gradient calculations. Flatten the six retained arrays into
   `SlabArrays`.

All six numerical field arrays retain their three-dimensional core-slab shape
until the final flattening in `read_slab()`.

From yt, the code obtains density, QUOKKA temperature, velocities, and cell
widths. It derives the shielding column and velocity gradient using
[gas_fields.py](../src/quokka2s/physics/gas_fields.py), and the foreground dust
column using [physics/dust_attenuation.py](../src/quokka2s/physics/dust_attenuation.py).

In [snapshot_reader.py](../src/quokka2s/snapshot_reader.py), `Snapshot` holds the dataset
handle and its geometry. `read_slab()` returns `SlabArrays` with exactly these
six arrays:

| Array | Meaning | Unit |
|---|---|---|
| `density_g_cm3` | Gas mass density | g/cm³ |
| `temperature_QUOKKA_K` | QUOKKA temperature | K |
| `shielding_NH_cm2` | Harmonic mean of the +z and −z hydrogen columns; used for table queries | cm⁻² |
| `foreground_NH_cm2` | Column from the emitting cell centre to the outer −z face; used for dust attenuation | cm⁻² |
| `velocity_gradient_s` | LVG velocity-gradient estimate | s⁻¹ |
| `velocity_z_kms` | Cell velocity along the selected line of sight | km/s |

Each is flattened from `(8, 256, 2048)` to `(4194304,)` in NumPy C order
(z changes fastest). Flattening preserves all cell values and their ordering;
it makes a contiguous batch easy to select. It is not averaging/downsampling.

Density, columns, gradient, and velocity are converted to the listed units
before becoming plain NumPy arrays. The raw QUOKKA temperature values are
interpreted in kelvin. Plain arrays do not retain a yt unit object; names and
interfaces state their units. Cell widths and volume come from yt conversions.
Hydrogen density `nH` is calculated later for each query batch, not retained
as a seventh slab array.

For `x_index_range: [64, 128]` and `y_index_range: [32, 96]`, slabs cover
global x ranges `64:72`, `72:80`, ..., `120:128`. Each read first calculates
fields with all 256 original y rows and full z, then retains y=32:96.
The returned slab shape is `(8, 64, 2048)`. Its `y_start=32` and
`native_y_size=256` retain the original positions. Images subtract the region's
origin `(64, 32)` to map these cells to local pixels. Original full-grid cell
IDs skip the unselected y rows between x layers; they are not a contiguous
range from first to last.

## 3. Calculate one batch: `emission_calculator.calculate()`

With `query_chunk: 1000000`, this slab is processed in four batches of one
million cells and one batch of 194,304 cells. `slab.batch(...)` creates a
`CellBatch`: six array views plus the original grid position, cell volume,
and first/last cell IDs. These views share the slab memory; selecting a batch
does not copy its six arrays. Let `N` be its cell count.

`accumulate_batch()` first passes the current cells to the shared calculator
from [physics/cell_emission.py](../src/quokka2s/physics/cell_emission.py):

```python
emission = emission_calculator.calculate(cells=cells)
```

`CellEmissionCalculator` owns the batch calculation steps and the fixed readers,
line names, and dust values they use. Each method below performs its named action.
The query results, masks, and emissivity arrays remain local to `calculate()`.

1. `cells.hydrogen_density_cm3` supplies physical nH, shape `(N,)` [cm⁻³],
   using `rho * X_H / m_H`. It is computed on first access, cached on this batch,
   and read by both readers. The same read-only array is reused until the batch
   is released; nH is not passed beside `cells` to the readers.
2. `cold_cells = cells.temperature_QUOKKA_K < 3000.0` classifies the batch.
   This `(N,)` boolean array records only the QUOKKA temperature branch;
   table availability does not change it.
3. `self.despotic_reader.read_fields()` returns `despotic_fields`: temperature
   and CO `lumPerH` for all cells,
   then CII `lumPerH` and electron, ionized-H, and neutral-H number densities
   for cold cells with valid DESPOTIC temperature.
4. `self.cloudy_reader.read_fields()` returns `cloudy_fields` for every hot
   cell, independently of DESPOTIC. The named atomic-line values are
   emissivity_per_nH2 [erg cm³/s]. Failed Cloudy queries affect atomic lines
   in those hot cells; they do not remove DESPOTIC CO.
5. `self.calculate_intrinsic_lines()` returns one `IntrinsicLineEmission` per
   line. Each record keeps its emissivity [erg/s/cm³] and the temperature [K]
   of the adopted emitting state together, both `(N,)`. Follow these actions:

   - `calculate_cold_hydrogen_and_cii_lines()`: calculate cold CII and analytic
     Halpha/HI from DESPOTIC fields, keeping its returned temperature.
   - `initialize_ciii_and_civ_lines()`: set cold CIII/CIV emission to zero.
   - `fill_hot_cloudy_lines()`: fill all eight hot atomic lines from Cloudy,
     keeping the temperature used for the Cloudy query.
   - `calculate_co_lines()`: calculate both CO lines from DESPOTIC fields and
     temperature in every QUOKKA temperature branch.

   DESPOTIC `lumPerH` uses query nH; Cloudy uses physical `nH**2`. Each line
   keeps NaN where its own emissivity cannot be calculated.
6. `self.apply_foreground_dust()` returns named `LineEmission` records. It
   multiplies available intrinsic emissivities by `exp(-sigma * foreground_NH)`
   and retains the paired temperature unchanged for spectrum and phase products.
7. `self.build_batch_emission()` combines these already constructed line records
   with DESPOTIC temperature, the QUOKKA cold branch and lookup clipping counts.
   `emission.lines['halpha']` selects Halpha. Images and spectra are added later.

The calculator methods use `self.line_keys` and
`self.dust_cross_section_cm2_H`; these fixed values are not passed again at
each calculation step. Per-species functions in
[physics/line_emissivity.py](../src/quokka2s/physics/line_emissivity.py) implement the numerical
CO, CII, hydrogen, and hot atomic-line formulas using the queried arrays.

The `DespoticCellReader` and `CloudyCellReader` live in
[despotic/cell_fields.py](../src/quokka2s/despotic/cell_fields.py) and
[cloudy/cell_fields.py](../src/quokka2s/cloudy/cell_fields.py). Their `read_fields()` methods use
the lookup stored on that reader; callers supply the current cell inputs.
`despotic_fields` holds named temperature, luminosity-per-H, and number-density
arrays, each `(N,)`.
Cold-only fields are NaN elsewhere. `cloudy_fields.emissivity_per_nH2['halpha']`
is the Halpha `(N,)` array, with NaN outside successful selected hot cells.
The calculator's emissivity method passes these arrays to the per-species
formula functions; those functions do not receive lookup objects.

Inside `DespoticCellReader.read_fields()`, follow five steps:

1. `self.prepare_query_inputs()` prepares nH, shielding NH and dV/dr for
   all N cells. Coordinate bounds and tiny endpoint clipping use the same rules.
2. `self.lookup.temperature_and_co()` queries T_D, CO10 and CO21 together for all N
   cells. Keep this existing bundled interpolation; it is already one clear action.
3. Mark unavailable/nonpositive T_D and select eligible cold cells using the
   caller's T_QUOKKA branch. These two masks stay local to `read_fields()`.
4. `self.read_cii_and_hydrogen_densities()` queries CII lumPerH and e-/H+/H
   number densities only for those Q cold cells. With no eligible cold cells,
   it returns empty arrays without a table call.
5. `self.restore_batch_positions()` restores those four cold-only arrays
   to N-cell positions. T and CO are already full-batch arrays and pass through.
   The final `DespoticCellFields` fields and units remain unchanged.

Inside `CloudyCellReader.read_fields()`, follow this order:

1. `self.prepare_query_inputs()` selects T, nH and shielding NH using
   `selected_cells`, the caller's hot-cell mask. The three named arrays keep
   the selected cell order.
2. `self.lookup.interpolate_available()` queries the current 3D table. It
   prepares logarithmic coordinates, inspects nodes once, and interpolates
   successful cells. No separate model-depth coordinate is supplied.
3. `self.count_column_clipping()` counts NH clipping among successful queries.
4. `self.restore_batch_positions()` places `(8, Q)` queried values back into
   the original N-cell positions, then returns named `(N,)` row views and
   `temperature_K`, the temperature supplied for each queried cell.
   Unqueried or failed emissivities remain NaN; unqueried temperatures are NaN.
   `failed_cells`
   marks only queried failures; an unqueried cold cell is not a Cloudy failure.

The lookup uses only the three coordinates above. Jeans length is calculated
when building the table, with mu=1 and a 100 pc cap; it is not a lookup axis.

`CloudyLookup.sample()` is the strict entry point for diagnostics: it raises
if any requested cell has failed support. Both entry points share the same
coordinate preparation and corner inspection.

Important fields in the returned `BatchEmission`:

| Result | Shape | Unit or meaning |
|---|---|---|
| `despotic_temperature_K` | `(N,)` | K |
| `lines` | Dictionary of ten `LineEmission` records | Select a line by name |
| `lines['halpha'].intrinsic_emissivity_erg_s_cm3` | `(N,)` | Intrinsic Halpha ε, erg/s/cm³ |
| `lines['halpha'].temperature_K` | `(N,)` | Adopted Halpha emitting-state temperature, also used for broadening, K |
| `lines['halpha'].attenuated_emissivity_erg_s_cm3` | `(N,)` | Dust-attenuated Halpha ε, erg/s/cm³ |
| `lines['halpha'].emissivity_is_missing` | `(N,)` | Boolean: this cell has no Halpha emissivity; physical zero is not missing |
| `cold_cells` | `(N,)` | Boolean: `T_QUOKKA < 3000 K` |
| `despotic_coordinate_clipped_cells` | Three integer counts | Clipped nH, NH and dVdr queries with usable DESPOTIC temperature |

`cold_cells` records only `T_QUOKKA < 3000 K`; table availability does not
change it. There is no global line-validity mask. For example, a hot cell
with missing DESPOTIC results can have available Halpha and missing CO.
`emissivity_is_missing` is calculated from each line's intrinsic array;
images and spectra select that same line's available entries. Unexpected
invalid required fields with usable temperature still raise errors.
Low-level Cloudy `failed_queries` describes the Q selected queries, before
results are restored to N positions.

These lookup/emissivity arrays belong to the current batch. The complete
snapshot's per-cell emissivities or luminosities are never stored in memory.
`emission_calculator.dust_cross_section_cm2_H['halpha']` is a scalar [cm²/H]
shared by all batches. `CellEmissionCalculator` stores only these fixed inputs:

| Attribute | Stored value |
|---|---|
| `despotic_reader` | `DespoticCellReader`, holding the fixed DESPOTIC lookup |
| `cloudy_reader` | `CloudyCellReader`, holding the fixed Cloudy lookup |
| `line_keys` | Ten line names used by the products and saved file axes |
| `dust_cross_section_cm2_H` | Dictionary of scalar cross-sections [cm²/H], keyed by line name |

`calculate(cells=...)` returns the existing `BatchEmission` without storing it
on the calculator. Per-line records reference its result arrays without copying.

## 4. See how the batch contributes to each product

After calculating a `BatchEmission` once, `accumulate_batch()` passes that
same result to four explicit calls:

```python
emission = emission_calculator.calculate(cells=cells)
products.images.add_batch(cells, emission)
products.spectra.add_batch(cells, emission)
products.phases.add_batch(cells, emission)
products.record_batch(cells, emission)
```

Images read each named line and select its available cell positions. Spectra
group lines with identical selections, then pack that group's fields into
three `(Ngroup_lines, Navailable_cells)` numerical arrays. These arrays exist
only while adding that batch. Independent cell sums check image and spectrum
luminosities using each line's own selection. Saved image/profile axes and
line names are unchanged; `cell_counts_by_regime` has shape `(Nline, 2)`.

Read the three accumulators separately; they receive already calculated cell
emissivities, masks, temperatures, and grid positions. `record_batch()` keeps
independent luminosity sums, mass totals, and counts for checking the products.
`counts` records all processed cells and cells lacking mixed temperature for
gas-phase statistics. `missing_emissivity_cells` and
`missing_emissivity_mass_g` record separate totals for every line.

| Product | File and function | Calculation |
|---|---|---|
| Image | [products/line_luminosity_images.py](../src/quokka2s/products/line_luminosity_images.py), `LineLuminosityImageAccumulator.add_batch()` | Find each cell's native `(x,y)` pixel and add `epsilon * volume`; all z cells on that sightline add to the same pixel |
| Spectrum | [products/integrated_spectra.py](../src/quokka2s/products/integrated_spectra.py), `IntegratedSpectra.add_batch()` | Centre the line at cell `vz`, integrate its thermal Gaussian over velocity channels, and add luminosity to the selected region's integrated spectrum |
| Gas phases | [products/gas_phase_velocity.py](../src/quokka2s/products/gas_phase_velocity.py), `GasPhaseVelocityAccumulator.add_batch()` | Assign the mixed-temperature phase, add cell mass to its velocity channel, and accumulate velocity moments |

Image accumulation uses the native 256 × 256 sightlines. Spectrum accumulation
uses each line's available cells directly; it does not go through a rebinned image.
Intrinsic and attenuated spectra share the same thermal integration for a
cell. They differ only in the luminosity being distributed over channels.
The same luminosities, velocities and thermal widths also enter
[products/line_velocity_moments.py](../src/quokka2s/products/line_velocity_moments.py).
`LineVelocityMoments.from_cells()` calculates the batch's total light, mean
velocity and centered second moment, including each cell's full Gaussian.
`merged()` combines these small totals across batches, workers and cold/hot
branches. It also includes the separation between batch mean velocities.
No extra cell arrays remain in memory.

For one line, with cell luminosities L, velocities v and thermal widths s:

```text
mean_velocity = sum(L * v) / sum(L)
sigma_full² = sum(L * ((v - mean_velocity)² + s²)) / sum(L)
```

For example, two equal-light cells at -10 and +10 km/s, each with thermal
width 3 km/s, give mean velocity 0 and full sigma sqrt(100 + 9) km/s.
The full moments do not depend on the saved ±200 km/s channel window.
`IntegratedSpectra` owns one set of running arrays:

| Attribute | Shape | Stored value |
|---|---|---|
| `dL_dv` | `(2, 10, 2, 400)` | Spectrum [erg/s/(km/s)]; axes are dust state, line, cold/hot, channel |
| `input_luminosity` | `(2, 10, 2)` | Cell luminosity supplied to each spectrum [erg/s] |
| `cell_counts` | `(10, 2)` | Available cells for each line and cold/hot branch |
| `full_line_moments` | Dictionary by dust state, line, branch | Full Gaussian luminosity, centroid and centered second moment |

Dust index 0 is intrinsic; index 1 is attenuated. For example,
`dL_dv[1, line_keys.index('halpha'), 1, :]` is the hot, dust-attenuated
Halpha spectrum. One `add_batch(cells=..., emission=...)` updates both dust
states; the Gaussian channel probabilities are shared.

When the batch finishes, its temporary query and emissivity arrays can be
released. When the slab finishes, the six slab arrays are released. The lookup
tables and accumulated products remain for the next slab.

## 5. Then read concurrency and saving

Start with `process_snapshot()`: `select_slabs_to_process()` returns only index
windows (32 for this snapshot), and `count_cells_in_slabs()` counts their core
cells. With one shared worker pool, the main loop calls `process_one_slab()`
and then `report_snapshot_progress()` for each window.

For a first reading, follow the `chunk_workers == 1` branch inside
`process_one_slab()` to `process_serial_batches()`. It calls the same
`accumulate_batch()` used in parallel. This is a reading order, not a request
to change the supplied YAML. Each slab is read and finished before the next;
the list of windows does not hold any simulation field arrays.

With multiple chunk workers, `process_parallel_batches()` submits a bounded
number of batches from the already loaded slab. Workers read the same immutable
slab arrays; each owns its output accumulators. The main thread merges their
results in input order using `EmissionProducts.merge()`. It reads the next slab
only after the current one is finished. `spectral_workers` controls additional
spectral-integration threads inside each batch.

The end of `main()` has three separate actions:

```python
outputs = products.build_outputs()
image_conservation = products.check_outputs(outputs)
write_products_and_report(
    config=config,
    snapshot=snapshot,
    emission_calculator=emission_calculator,
    products=products,
    outputs=outputs,
    image_conservation=image_conservation,
    began=began,
)
```

`build_outputs()` asks each accumulator to build its numerical arrays and
summaries, returning `EmissionOutputs`. `check_outputs()` then compares each
line's arrays with its independent cell sums and checks gas-phase accounting. Its
returned image-conservation report is separate from the generated arrays.
A failed check raises before writing files. The writer saves:

| File | Main numerical array | Shape for this snapshot |
|---|---|---|
| `images.npz` | `line_luminosity_image_erg_s` | `(dust_state=2, line=10, x=256, y=256)` |
| `spectra.npz` | `dL_dv_erg_s_per_kms` | `(dust_state=2, line=10, regime=2, channel=400)` |
| `phase_velocity.npz` | `histogram_mass_g` | `(phase_plus_total=6, channel=400)` |

The files also contain their axes, line/dust-state names, and units in field
names. Spectral products retain input luminosity, captured luminosity, and
luminosity outside the ±200 km/s window. Total-line moments have shape
`(dust_state=2, line=10)`:

| Saved field | Meaning |
|---|---|
| `line_centroid_full_kms`, `line_sigma_full_kms` | Complete Gaussian line moments, including emission outside saved channels |
| `line_second_raw_moment_full_kms2` | Full luminosity-weighted mean of v², including thermal variance |
| `full_line_luminosity_erg_s` | Total light used for the full moments |
| `line_centroid_window_kms`, `line_sigma_window_kms` | Moments calculated from the saved channels |

Window moments are also saved by cold/hot branch with shape `(2, 10, 2)`.
Zero-light lines have NaN centroids and dispersions.

Process checks independent cell sums against image pixels, spectrum inputs
and full-moment light. It checks the channel integral against captured light,
and captured plus outside light against cell sums. These comparisons use
`rtol=1e-10, atol=0`: float64 summation-order differences are accepted; expected
zero light must remain zero. It also checks finite physical outputs, dust
attenuation, cold CIII/CIV zeros and gas-phase accounting. Plot does not repeat
these science checks or validate fixed array shapes/channel counts.

`emission_report.json` keeps the numerical settings and results in readable
form:

| Report field | Contents |
|---|---|
| `input_tables` | DESPOTIC, Cloudy, and dust table paths |
| `snapshot_grid` | Grid shape and cell widths in cm |
| `physical_settings` | `X_H`, column mean, and column directions |
| `execution` | Slab size, query-batch size, and worker counts |
| Counts, masses, and conservation checks | Per-line missing emissivities, unavailable gas temperatures, luminosity sums, and spectral-window losses |

`write_products_and_report()` receives the input settings, snapshot, calculator,
accumulated products, and generated numerical payloads and the check report. Its steps are
`add_output_metadata()`, `save_product_arrays()`, and
`build_processing_report()`, followed by writing the report and completion
status. `status.json` records progress or a failure; it is not a resume
checkpoint.

## 6. Follow `plot`

Open [plot_emission_results.py](../src/quokka2s/plot_emission_results.py) and start at `main()`.
The rendering functions are in [figures/emission_results.py](../src/quokka2s/figures/emission_results.py):

1. `load_plot_config()` reads the product and figure directories.
2. `load_plot_products()` reads the three saved NPZ files once into
   `SavedEmissionProducts(images, spectra, gas_phases)`. It loads the fields
   needed for drawing; process has already checked the scientific products.
3. `draw_emission_products()` calls `plot_line_images()`,
   `plot_line_spectra()`, and `plot_gas_phase_comparisons()` in that order.
   Inside image plotting, `combine_image_pixels_for_display()` optionally sums neighbouring
   pixels for display. Gas-phase comparisons use
   [figures/gas_phase_spectra.py](../src/quokka2s/figures/gas_phase_spectra.py).
4. Each plotting step saves PNG/PDF in the figure directory's `png/` and `pdf/`
   subdirectories. If `titled_output_dir` is set, the same loaded numerical
   products also produce copies with titles under its own `png/` and `pdf/`.

No step reopens yt, the snapshot, DESPOTIC, or Cloudy tables. Plotting changes
colour limits, labels, displayed velocity range, peak normalization for the
phase comparison, and optional image binning. It does not recalculate cell
emissivities. The normal spectrum plots use cold+hot totals. Phase-comparison
plots use total Halpha/HI/CII/CO and hot CIII/CIV; the saved spectra retain both
regimes. Curves use the saved velocity axis. Sigma labels use the full-profile
moments, including emission outside the saved channel window. Both full and
window moments remain available in the saved numerical products.

## Shared physical helpers and additional figures

[physics/settings.py](../src/quokka2s/physics/settings.py) defines X_H, the measured velocity-gradient floor, and the shielding-column
choice. [physics/gas_fields.py](../src/quokka2s/physics/gas_fields.py) calculates nH, NH, and
dV/dr from unit-aware snapshot fields. [physics/hydrogen_emissivity.py](../src/quokka2s/physics/hydrogen_emissivity.py)
contains the two analytic cold hydrogen emissivities. Physical constants are
centralized in [constants.py](../src/quokka2s/constants.py).

Figure 1 and the emission phase diagrams have explicit callers:

- `tools/figures/build_table_input_slice.py`: read the selected x slice, calculate its
  five fields, save arrays, then optionally draw Figure 1.
- `tools/figures/build_emission_phase_histograms.py`: read slabs, calculate each batch
  once, accumulate two-dimensional histograms, save arrays, then draw
  the ten-panel phase diagram.
- `tools/figures/build_gas_projection_maps.py`: accumulate gas projections using cells
  with an available mixed temperature, then save and draw the maps.

Their `--no-plot` option saves data only; `--plot-only` reads their saved data.
They reuse `Snapshot` and `CellEmissionCalculator` directly and do not load
a separate workflow framework.
