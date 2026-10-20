# Reading `process` → saved data → `plot`

Start at `main()` near the bottom of [process_snapshot.py](../src/quokka2s/process_snapshot.py).
Read what each call receives and returns before opening the function itself.
Process and plot have independent entry points:

```bash
uv run process
uv run plot
```

From the repository root, both read the default `configs/emission.yaml`.
Use `--config PATH` to select another file. Each runs directly;
there is no shared dispatcher.

First prepare the environment with `uv sync`. uv
automatically installs the checkout in editable mode and registers the two
commands from `[project.scripts]` in `pyproject.toml`. They call the existing
`process_snapshot.main()` and `plot_emission_results.main()` functions directly.
No activation, import-path export, or manual editable installation is needed.
Plot reports the saved-results directory, each line being drawn, output paths,
and the final file count and elapsed time; these messages do not alter products.
The equivalent `uv run python -m quokka2s.process_snapshot` and
`uv run python -m quokka2s.plot_emission_results` commands remain useful
for Python debugging.

The first three settings select one snapshot and its results:

```yaml
jobname: sigma13.v2
dataset: ../inputs/snapshots/plt0655228
results_dir: ../output
```

The supplied dataset path names snapshot `plt0655228`. Its results directory
is `output/sigma13.v2/plt0655228/`: process writes `processed/`, and plot reads
`processed/` and writes `figures/` plus the optional `figures_titled/`.
Changing `jobname` to `sigma50` selects a different job even when its snapshot
is also called `plt0655228`. Plot uses the snapshot name only; it does not
open the dataset path or emission tables.
If x/y ranges are supplied, both commands add a region folder under the snapshot:
`region_x64-66_y64-66/` for `[64, 66)` on both axes. An omitted or `null` axis
uses `all` in the name; both omitted or `null` keep the full-box directory.
These names come from the YAML alone, so plot can select a region without
the original snapshot.

The two main commands are in the [root README](../README.md). Detailed
settings and table-building commands are in the [usage guide](usage.md).
The scientific choices are in the [physics reference](emission_method.md).

The package has five purpose-specific folders:

| Folder | Responsibility |
|---|---|
| `physics/` | Derived gas fields, emissivity branches, and dust attenuation |
| `cloudy/` | Three-dimensional Cloudy table lookup and cell queries |
| `despotic/` | DESPOTIC building, filling unavailable nodes, and table lookup |
| `products/` | Numerical image, spectrum, gas-phase, and projection accumulation |
| `figures/` | Draw figures from already calculated numerical arrays |

## 1. Follow the stages in `main()`

| Stage | Function | What it produces |
|---|---|---|
| Read settings | `load_process_config()` | A `ProcessSettings` record with input/output paths and execution settings |
| Open inputs | `processing_inputs.load_processing_inputs(config)` | `snapshot`, `emission_calculator` |
| Create output arrays | `EmissionProducts(...)` | Initially empty image, spectrum, and gas-phase accumulators |
| Process cells | `process_snapshot()` | The accumulated numerical results |
| Build outputs | `products.build_outputs()` | Image, spectrum and gas-phase arrays and summaries |
| Add interpretation | `ResultMetadata.from_processing_inputs()` then `outputs.add_metadata()` | Selected pixel coordinates, region and ordered dust metadata |
| Check outputs | `products.check_outputs(outputs)` | Conservation/accounting checks; image-conservation diagnostics |
| Build report | `processing_report.build_processing_report()` | Readable settings, counts, luminosities and check results |
| Save | `result_files.write_emission_results()` then `write_status()` | Three NPZ files, the JSON report, then completion status |

The orchestration stays in [process_snapshot.py](../src/quokka2s/process_snapshot.py).
Input loading belongs to [processing_inputs.py](../src/quokka2s/processing_inputs.py), and output writing
belongs to [result_files.py](../src/quokka2s/result_files.py);
[result_metadata.py](../src/quokka2s/result_metadata.py) describes saved geometry
and dust values, and [processing_report.py](../src/quokka2s/processing_report.py)
builds the human-readable summary.
[products/emission_products.py](../src/quokka2s/products/emission_products.py) groups the accumulated products.
The objects group related variables; they are not extra copies of the snapshot:

- `snapshot` holds the yt dataset and an optional `xy_region` of native index
  bounds. `shape`, cell widths/volume, `cell_count`, and `projected_area_cm2`
  describe the original box. `processing_shape`, `processing_cell_count`,
  `processing_area_cm2`, and `processing_xy_origin` describe the selected region.
  `snapshot.native_reader` caches file locations and units from yt metadata.
  During processing, its scoped field cache shares converted native fields
  across up to four adjacent selected slabs. Outside that context it retains
  no numerical payloads or open maps. `snapshot.read_slab()` still copies the
  current core fields and derives the same six slab arrays.
- `emission_calculator` is a `CellEmissionCalculator`. It holds the DESPOTIC
  and Cloudy readers, ten line names, and named dust cross-sections. Each reader
  holds its fixed lookup table. These objects are created once and shared across
  batches; they do not store queried cell fields or emissivities.
- `slab` owns six field arrays, its global x position, local shape, and cell
  volume. `slab.iter_batches(batch_size=...)` yields consecutive zero-copy cell
  views; `slab.batch(start=..., stop=...)` selects one explicit range.
- `cells` carries those batch views and their position. Its first/last full-grid
  cell IDs are calculated properties, so callers cannot provide conflicting IDs.
- `products` holds the running sums: luminosity images, spectra, gas-phase
  statistics, total luminosities, cell counts, and mass totals. It saves the
  expected selected-cell count and projected area at construction, so `build_outputs()`
  needs no arguments and products retain no dataset/table references.

Slab reads, table queries, and spectrum accumulation remain explicit actions.
Hydrogen density is cached on the current batch. Constructing `LineEmission`
finishes its writable calculation: its three result arrays become read-only,
and `emissivity_is_missing` is calculated once. Images, spectra and accounting
reuse that line's mask; a physical zero remains available.

Inside `load_processing_inputs(config)`, follow three named steps:

1. `check_input_paths()` checks the configured input and output locations.
2. `open_snapshot()` opens yt, obtains the original grid geometry, and records
   any x/y selection from the YAML.
3. `load_emission_calculator()` loads the tables, creates their cell readers,
   and prepares the per-line dust cross-sections. It derives line names once
   from Cloudy plus `co10` and `co21`, then constructs `CellEmissionCalculator`.

`print_run_settings()` displays the selected run and paths before input loading.
After yt opens the snapshot, `print_snapshot_summary()` displays its native grid,
physical cell widths and selected index ranges without reading cell fields.
Stage messages precede table loading, each slab read/calculation, and final
construction/check/saving. They are terminal messages only; the numerical
products and the existing progress timing remain unchanged.

The function creates the output directory after loading succeeds and returns:

```python
snapshot, emission_calculator = load_processing_inputs(config)
```

Physical constants come from [constants.py](../src/quokka2s/constants.py), which converts
Astropy constants once to the units used by NumPy arrays. Functions import
these constants directly; they are not passed through the calculator or constructors.
The hydrogen mass is read directly from yt's `mh` and converted to grams in
[gas_fields.py](../src/quokka2s/physics/gas_fields.py), which owns the snapshot
density conversions. The shared constants module does not import yt, so plotting
saved products still works without yt. The old CIAOLoop constants in the table-reuse
check describe how that table was built and remain historical values.

Opening checks cover missing inputs and numerical compatibility, including the
DESPOTIC table's density/column settings and recorded coordinate bounds.
Its build-snapshot shape and cell count are historical metadata; they need not
match the current snapshot. Final checks verify cell accounting and luminosity
conservation. Follow the slab and batch calculations below before studying each check.

## 2. Read one slab: `snapshot_reader.read_slab()`

The reader accepts three-dimensional uniform level-0 grids. It reads `(Nx, Ny, Nz)`,
cell widths and volume from the current snapshot. A full slab has shape
`(slab_nx, Ny, Nz)`; the last slab may contain fewer x cells.

For the reference `(256, 256, 2048)` grid, one slab contains eight x cells and
all y,z cells: `(8, 256, 2048)`, or 4,194,304 cells. There are 32 such slabs. The reader
copies one extra cell on each x/y side, giving loaded fields of shape
`(10, 258, 2048)`. Only the `(8, 256, 2048)` core contributes to products.
x/y neighbours wrap across the original box faces: the first x core `0:8`
loads indices `[255, 0, ..., 7, 8]`. Both directions use centred differences;
z retains one-sided differences at its two outer faces. Full z columns stay
together for both column-density calculations.

Call `snapshot.read_slab(x_start=0, x_stop=8)` for the first eight x layers.
The caller supplies only the desired cells; the reader determines all neighbour
indices and removes those neighbours from its returned arrays.

`slab_windows()` supplies these global, half-open core bounds without reading
any fields. For a full box:

```python
for x_start, x_stop in slab_windows(
    x_start=0,
    x_stop=256,
    slab_nx=8,
):
    slab = snapshot.read_slab(
        x_start=x_start,
        x_stop=x_stop,
    )
    # Finish this slab's batches before reading the next slab.
```

For a selected region, pass its global x bounds instead. The last window may be
shorter than `slab_nx`; neighbour selection still belongs to `read_slab()`.

Process opens `native_reader.reuse_slab_fields()` around the actual selected
slab windows, after applying `max_slabs`.
[slab_field_cache.py](../src/quokka2s/slab_field_cache.py) groups up to four
adjacent windows and reads their combined x range once, including the original
periodic x/y neighbours and full z. Each slab receives a read-only view into
five converted density, temperature and velocity arrays; batches receive only
the usual six copied core arrays. The cache belongs to the reading thread.
Its 3 GiB limit counts retained field values, not total process memory.
Groups shrink when needed; an unshared single window uses the direct reader.
The context releases retained values on completion or an exception.

Read `read_slab()` from top to bottom as these steps:

1. `snapshot.native_reader.read_slab_fields()` returns selected x/y cells plus
   one wrapped neighbour on each side, keeping full z. Within a shared group,
   these are read-only views; direct reads return owned field copies.
2. `select_core_simulation_fields()` copies core density, QUOKKA temperature
   and vz, removing the x/y neighbours.
3. `calculate_dust_foreground_column()` calculates the column to the observer.
4. `calculate_shielding_column()` calculates the +z/-z harmonic-mean column.
5. `calculate_core_velocity_gradient()` calculates dV/dr using neighbours.
6. Release this slab's halo views. The scoped cache can retain their group
   until its last read. Flatten the six copied core arrays into read-only
   `SlabArrays`.

All six numerical field arrays retain their three-dimensional core-slab shape
until the final flattening in `read_slab()`.

In [native_snapshot_reader.py](../src/quokka2s/native_snapshot_reader.py),
`NativeSnapshotReader` uses yt's native block bounds, payload offsets, field
order, dtype and units. FAB header reads use an 8 KiB buffer to avoid unnecessary
cell-payload read-ahead while locating each block's data start.
It copies selected density, temperature and three
momenta into compact arrays. The selected block geometry determines how it reads:

- Full block y/z coverage, including a narrow x selection: read a contiguous
  temporary buffer for each field. Such a slice touches rows throughout the
  field, so adjacent slab reuse avoids reading the same buffer repeatedly.
- Full block x coverage with partial y/z coverage: read only the required
  contiguous y bands in each z plane, sharing one read-only file handle across
  the block's five fields.
- Other partial selections: copy selected values from read-only NumPy memory
  maps.

Every path fills the same compact arrays in the same order. Maps, file handles
and temporary read buffers are released after copying. The scoped slab cache
retains only converted field arrays until their selected group has been read;
metadata alone retains no cell payloads or open maps. Momentum is divided by
native density, then velocities and density are normalized to yt's declared
gas units. Cell widths also come from yt. The code derives the
shielding column and velocity gradient using
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
global x ranges `64:72`, `72:80`, ..., `120:128`. The first read copies
x=63:73 and y=31:97 with full z, giving loaded shape `(10, 66, 2048)`.
Removing neighbours returns the `(8, 64, 2048)` core. Its `y_start=32` and
`native_y_size=256` retain the original positions. Images subtract the region's
origin `(64, 32)` to map these cells to local pixels. Original full-grid cell
IDs skip the unselected y rows between x layers; they are not a contiguous
range from first to last.

A one-cell x/y selection loads only `(3, 3, 2048)` fields, including
original-box neighbours on all four sides. Removing those neighbours returns
the `(1, 1, 2048)` core with complete z columns.

## 3. Calculate one batch: `emission_calculator.calculate()`

With `query_chunk: 1000000`, this slab is processed in four batches of one
million cells and one batch of 194,304 cells. The serial caller uses:

```python
for cells in slab.iter_batches(batch_size=config.query_chunk):
    accumulate_batch(
        cells=cells,
        emission_calculator=emission_calculator,
        products=products,
    )
    del cells
```

Each `CellBatch` carries six array views plus the original grid position, cell
volume, and first/last cell IDs. Selecting a batch does not copy its six arrays.
Consume and discard these views: retaining a batch also keeps the underlying
slab arrays alive. The caller drops the last batch before releasing its slab;
only accumulated products survive. Let `N` be the current batch's cell count.

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
2. Select cells with `cells.temperature_QUOKKA_K` less than
   `EMISSION_TEMPERATURE_BOUNDARY_K` from
   [physics/settings.py](../src/quokka2s/physics/settings.py), currently 3000 K.
   The resulting `(N,)` `cold_cells` array records only the QUOKKA temperature
   branch; table availability does not change it.
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

1. `self.prepare_query_inputs()` checks and clips nH, shielding NH and dV/dr for
   all N cells. `self.lookup.prepare_queries()` prepares their logarithmic
   coordinates once. Subsequent field queries reuse this batch-local record.
2. `self.lookup.temperature_and_co(queries=queries)` queries T_D, CO10 and CO21
   together for all N cells at those coordinates.
3. Mark unavailable/nonpositive T_D and select eligible cold cells using the
   caller's T_QUOKKA branch. These two masks stay local to `read_fields()`.
4. `self.read_cii_and_hydrogen_densities()` queries CII lumPerH and e-/H+/H
   number densities only for those Q cold cells. With no eligible cold cells,
   it returns empty arrays without a table call.
5. `self.restore_batch_positions()` restores those four cold-only arrays
   to N-cell positions. T and CO are already full-batch arrays and pass through.
   The final `DespoticCellFields` fields and units remain unchanged.

Inside `CloudyCellReader.read_fields()`, follow this order:

1. The reader selects T, nH and shielding NH directly from `cells` using
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
`emissivity_is_missing` is calculated once from each line's intrinsic array;
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
products.images.add_batch(cells=cells, emission=emission)
products.spectra.add_batch(cells=cells, emission=emission)
products.phases.add_batch(cells=cells, emission=emission)
products.record_batch(cells=cells, emission=emission)
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

`record_batch()` also passes the calculated batch to
[`ColdGasLuminosityByDespoticTemperature`](../src/quokka2s/products/cold_despotic_luminosity.py).
It selects `T_QUOKKA < 3000 K` cells whose finite DESPOTIC temperature reaches
10,000, 40,000, 100,000 or 1,000,000 K, then sums Halpha, HI and CII light
separately before and after dust. Only the small luminosity sums survive the
batch. Parallel batches merge these sums in the existing input order. Report
construction divides by the existing cold-plus-hot total for each line and
dust state, including light outside the saved velocity channels. No extra
snapshot reads or emission-table queries are needed.

| Product | File and function | Calculation |
|---|---|---|
| Image | [products/line_luminosity_images.py](../src/quokka2s/products/line_luminosity_images.py), `LineLuminosityImageAccumulator.add_batch()` | Find each cell's native `(x,y)` pixel and add `epsilon * volume`; all z cells on that sightline add to the same pixel |
| Spectrum | [products/integrated_spectra.py](../src/quokka2s/products/integrated_spectra.py), `IntegratedSpectra.add_batch()` | Centre the line at cell `vz`, integrate its thermal Gaussian over velocity channels, and add luminosity to the selected region's integrated spectrum |
| Gas phases | [products/gas_phase_velocity.py](../src/quokka2s/products/gas_phase_velocity.py), `GasPhaseVelocityAccumulator.add_batch()` | Assign the mixed-temperature phase, add cell mass to its velocity channel, and accumulate velocity moments |

Image accumulation uses the selected native x/y sightlines: 256 × 256 for
the full reference snapshot, or 512 × 512 for a full 512 × 512 × 4096 grid.
Spectrum accumulation uses each line's available cells directly; it does not
go through a rebinned image.
Intrinsic and attenuated spectra share the same thermal integration for a
cell. They differ only in the luminosity being distributed over channels.
Each integration thread owns its temporary channel-by-cell arrays. It reuses
their storage for division and half scaling; the Gaussian arithmetic and
input-order summation remain unchanged. These arrays are released after the task.
The same luminosities, velocities and thermal widths also enter
[products/line_velocity_moments.py](../src/quokka2s/products/line_velocity_moments.py).
`LineVelocityMoments.from_cells()` calculates the batch's total light, mean
velocity and centered second moment, including each cell's full Gaussian.
`merged()` combines these small totals across batches, workers and cold/hot
branches. It also includes the separation between batch mean velocities.
No extra cell arrays remain in memory.
The channel integrator and full moments use the same
`MINIMUM_GAUSSIAN_WIDTH_KMS` from `line_velocity_moments.py`: 1e-5 km/s.

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

The end of `main()` keeps construction, checking, reporting and writing separate:

```python
outputs = products.build_outputs()
metadata = ResultMetadata.from_processing_inputs(
    snapshot=snapshot,
    emission_calculator=emission_calculator,
)
outputs.add_metadata(metadata=metadata)
image_conservation = products.check_outputs(outputs)
report = build_processing_report(
    config=config,
    snapshot=snapshot,
    emission_calculator=emission_calculator,
    products=products,
    outputs=outputs,
    metadata=metadata,
    image_conservation=image_conservation,
    began=began,
)
write_emission_results(
    directory=config.output_dir,
    outputs=outputs,
    report=report,
)
```

`build_outputs()` asks each accumulator to build its numerical arrays and
summaries, returning `EmissionOutputs`. It also prepares the numerical fields
used in figures: per-area spectra, peak-normalized line/gas profiles, gas
channel centres, comparison sigma and image colour limits. Raw arrays remain
in the same products. `check_outputs()` then compares each
line's arrays with its independent cell sums and checks gas-phase accounting. Its
returned image-conservation report is separate from the generated arrays.
A failed check raises before writing files. The writer saves:

| File | Main numerical array | Shape for the reference snapshot |
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
| `cold_gas_luminosity_by_despotic_temperature` | Cold-branch Halpha/HI/CII light above four TD thresholds, corresponding total light, and intrinsic/attenuated fractions |
| `elapsed_seconds` | Time to build the report, before saving; final `status.json` includes file-writing time |

`write_emission_results()` receives only the output directory, finished payloads
and report. It writes the three NPZ files and `emission_report.json`; it does not
open yt, query tables, add metadata or perform checks. `main()` marks completion
only after writing succeeds. `status.json` records progress or a failure,
including the failed stage; it is not a resume checkpoint.

## 6. Follow `plot`

Open [plot_emission_results.py](../src/quokka2s/plot_emission_results.py) and start at `main()`.

1. `load_plot_config()` reads the shared YAML and returns `PlotSettings` with
   product and figure directories under `results_dir/jobname/snapshot_name/`.
   Supplied x/y index ranges add the same region folder used by process.
   It obtains the snapshot name from the supplied `dataset` path without
   opening that path or checking emission-table files.
2. `read_emission_results()` in the root
   [emission_results.py](../src/quokka2s/emission_results.py) reads the three
   saved NPZ files once into `EmissionResults(images, spectra, gas_phases, ...)`.
   It loads the fields
   needed for drawing; process has already checked the scientific products.
3. `draw_emission_products()` selects each line's saved attenuated total spectrum
   by name. Images, normalized gas curves, surface-luminosity spectra and colour
   limits are already saved; no numerical preparation occurs here. An optional
   resolution setting selects `images_factor_2.npz`, prepared separately from
   native pixels by `prepare_emission_results.py`.
4. It draws images with
   [figures/line_luminosity_images.py](../src/quokka2s/figures/line_luminosity_images.py),
   spectra with [figures/line_spectra.py](../src/quokka2s/figures/line_spectra.py),
   and phase comparisons with
   [figures/gas_phase_spectra.py](../src/quokka2s/figures/gas_phase_spectra.py).
5. Each plotting step saves PNG/PDF in the figure directory's `png/` and `pdf/`
   subdirectories. With the default `titled_figures: true`, the returned
   `titled_output_dir` points to `figures_titled/`; the same loaded numerical
   products also produce copies with titles under its own `png/` and `pdf/`.

No step reopens yt, the snapshot, DESPOTIC, or Cloudy tables. Plotting selects
saved fields and applies labels, colours, line styles, axis windows and layout.
It does not calculate normalization, moments, physical conversions or image
pixel sums. The normal spectrum plots use cold+hot totals. Phase-comparison
plots use total spectra for every line; CIII/CIV have zero cold emission, so
their totals equal their hot profiles. The saved spectra retain both regimes.
Curves use the saved velocity axis. Sigma labels use the full-profile
moments, including emission outside the saved channel window. Both full and
window moments remain available in the saved numerical products.

For an optional new image resolution, run
`uv run python -m quokka2s.prepare_emission_results --image-downsample-factor 2` once.
It sums neighbouring native pixels, saves their own edges and colour limits
in a separate file, and leaves native image values unchanged. The same command
without a factor prepares older saved products from their accumulated arrays;
it does not read any simulation cells. New snapshot runs save all required
native-resolution figure fields directly.

The results object selects arrays by name rather than requiring callers to
remember packed-axis indices:

```python
image = results.images.for_line(line="halpha", dust_state="attenuated")
line = results.spectra.for_line(line="halpha", dust_state="attenuated")
cnm = results.gas_phases.for_phase(phase="CNM")
```

The image is a `(Nx, Ny)` view in erg/s per pixel. `line` carries the saved
velocity/profile arrays and full/window moments. `cnm` uses the gas-phase
file's own channel edges. A branch query uses its saved regime name, for example
`T_QUOKKA_ge_3000K`, and returns `None` for full moments because these are saved
only for the total line.

[products/gas_phase_velocity.py](../src/quokka2s/products/gas_phase_velocity.py)
owns the phase names and temperature bounds and saves them as `phase_keys` and
`phase_boundaries_K`. `GasPhaseResults` carries these values; the comparison
figure's phase footer uses the saved definitions.

## Shared physical helpers and additional figures

### Follow one DESPOTIC table point

`despotic/cell_solver.py` creates a fresh cloud at the adopted initial state,
adds its emitters and calls native chemical/thermal equilibrium with `TableGOW`.
That class inherits GOW's initialization, abundances and ordinary `dxdt()`.

The checked `chemEvol` fixed-temperature loop calls
`network.prepare_fixed_temperature_rhs()` before each integration attempt.
It returns a `FixedTemperatureChemistry` object from
`despotic/fixed_temperature_chemistry.py`. The object holds that segment's
temperature, effective density, column, radiation and dust values. Its
`__call__(trial_abundances, time_s)` returns twelve abundance derivatives in
`s^-1`. Three small objects make its work explicit:

- `AbundanceWorkspace` recomputes all eighteen species from each trial state,
  reusing the name/index mappings and a private scratch array.
- `PreparedTwoBodyRates` prepares temperature-only coefficients once; it
  recomputes all composition-dependent coefficients for every trial.
- `PreparedReactionTerms` prepares fixed density/radiation factors; shielding,
  gas reactions and grain reactions still use the current trial abundances.

These helpers live in `abundance_workspace.py`, `two_body_rates.py` and
`reaction_terms.py`. Returned derivatives are fresh arrays; later scratch
updates cannot overwrite a previously returned derivative.

The clumping factor uses the saved network composition, as in native GOW.
The thermal velocity, shielding and composition-dependent rates use the current
trial composition. Each new segment or local-time retry constructs a new
callable. No cache survives a temperature/state update, and the native
initial guesses, solver precision and acceptance checks are retained.

Table metadata records the actual reaction implementation source. Raw-table
reuse accepts only reviewed equivalent model settings; checkpoint reuse also
requires the exact implementation identity.

### Reusable table-input survey

`uv run python -m quokka2s.check_table_input_ranges` reads the same emission YAML.
Follow its two distinct actions:

1. `table_input_survey.load_or_measure_input_ranges()` reuses completed statistics or calls
   `scan_snapshot_input_ranges()`. `NativeBlockRangeReader` reads complete native
   fields and shares them between XY columns at the same z, keeping complete
   z-column physics;
   `TableInputStatistics.record_batch()` records physical nH/NH/dVdr for all
   cells and nH/NH/T_QUOKKA for hot cells. Only extrema/counts survive each batch.
2. `build_table_range_report()` compares those stored extrema with the current
   axes. It performs no cell reads and no table interpolation. Cloudy NH
   clipping is allowed; all other axes use the same coverage tolerance as process.

`TableInputFiles` stores `input_ranges.json` separately from the freshly built
`table_range_report.json`. Source file state and plain physical definitions
guard reuse; tables are not a dependency of raw coordinate statistics.
Region/partial scans retain explicit completeness information. This survey
checks ranges only, so it does not claim interpolation holes have been checked.

The range reader in [table_input_reader.py](../src/quokka2s/table_input_reader.py)
validates a Cartesian native-block layout and selects XY columns intersecting
the requested region. For a `(512, 512, 4096)` snapshot with
`(256, 256, 128)` native blocks, a full scan visits four XY columns, each with
32 z layers. A float64 native field occupies 64 MiB per block. Region selection
crops the retained core; gradient neighbours still come from the original box,
wrapping across its x/y faces.

`prepare_column_states(columns=...)` first reads each selected column's density
in ascending z order and saves its cumulative column at every block face.
Each `NativeColumnState` holds these lower-boundary values and its own upper
column and velocity-face carries.

`iter_layers(columns=...)` then visits z layers from top to bottom. At each z,
it processes all selected XY columns before moving down. `NativeZLayerFields`
reads each needed native field once at that z; another core or periodic
neighbour reuses the same read-only array. The cache is cleared before the next
z layer, on an error, or when the iterator is closed. Unit conversion and
lower-z plane reads still use the original native reader directly.

`calculate_layer(state=..., z_index=...)` returns four owned `TableInputLayer`
arrays: density, QUOKKA temperature, shielding NH and velocity gradient.
Incoming column carries precede the new cells in each cumulative sum,
preserving the original addition order rather than adding block totals.
Velocity neighbours preserve central differences at internal block faces and
one-sided differences only at physical z boundaries. Only the current z
layer's native fields, returned arrays and small column carries are retained.

Each layer's `iter_batches()` yields views for `TableInputStatistics`; batches
do not trigger disk reads. Emission processing still uses `read_slab()` and
its six fields in x-slab order, preserving its existing accumulation order.
For the single-snapshot command, `--max-columns 1` limits a diagnostic to one
native XY column with complete z physics. Omitted selected columns make the
scan partial. The emission YAML's `slab_nx` and `max_slabs` control emission
processing, not this range scan.

For several jobs, `uv run python -m quokka2s.check_job_table_input_ranges` reads
`configs/table_input_survey.yaml`. `source_root` contains the configured job
directories, and each job contains `plt` plus digits snapshot directories.
Follow its actions in this order:

1. `read_job_survey_config()` reads the job locations, scan sizes and invocation
   limits (`snapshot_workers`, `max_new_snapshots`, `max_runtime_minutes`),
   plus the scan's `query_chunk` and optional `max_columns`.
2. `discover_job_snapshots()` lists direct snapshot directories in numeric order;
   `check_batch_output_paths()` protects the entire source root from output writes.
3. `scan_job_snapshots()` reuses completed statistics and dispatches unfinished
   snapshots to separate workers. Each worker reads one snapshot and saves its
   own statistics. The coordinator compares ranges with the current tables and
   records results in inventory order.
4. `build_job_survey_summary()` combines extrema from full successful scans only;
   the entry point saves `snapshots.csv`, `jobs.csv` and `summary.json`.

These actions belong to [table_input_jobs.py](../src/quokka2s/table_input_jobs.py).
Both commands pass a small `SnapshotInputSettings` record to the shared scan;
emission, plotting and dust settings are not required.

`snapshot_workers` defaults to one; setting it to two scans two snapshots at
once. Each worker uses one calculation thread and owns its snapshot reader and
layer arrays. A layer retains its four copied cell arrays until its batches are
reduced; column carries survive until that column finishes. Afterward only
scalar extrema and counts remain.
There are no image/spectrum accumulators and no emission-table interpolation.
Completed snapshots survive interruption. With multiple workers, Ctrl-C stops
new dispatch and lets active snapshots finish and save. In single-worker mode,
Ctrl-C interrupts the current snapshot; that snapshot is restarted next time.
Source-file state and physical rules must match before completed statistics are
reused. Failed, partial and not-selected snapshots remain visible in summaries.
`max_new_snapshots` counts newly dispatched scans, not reused statistics;
`--max-new-snapshots` overrides this limit. `max_runtime_minutes` stops new
dispatches after the budget, while active workers finish and save their results.
Neither limit truncates an active snapshot. `--max-snapshots 1` instead selects
the first inventory entry, including an already completed entry. Unscanned
snapshots keep the survey incomplete. `max_columns` limits each scan; a partial
scan cannot establish full input ranges.

[physics/settings.py](../src/quokka2s/physics/settings.py) defines X_H, the
QUOKKA temperature branch boundary, the measured velocity-gradient floor, and
the shielding-column choice. Snapshot processing and table-domain checks use
these same settings. Paths and worker counts belong to the YAML configuration.
[physics/gas_fields.py](../src/quokka2s/physics/gas_fields.py) calculates nH, NH,
and dV/dr from unit-aware snapshot fields. `hydrogen_number_density_cm3()` is
the shared density conversion for cell batches, dust columns and the unit-aware
shielding calculation. Its `mixed_gas_temperature_K()`
selects DESPOTIC temperature for cold cells and QUOKKA temperature for hot cells,
preserving missing cold temperatures. Gas-phase statistics, projection maps,
and phase histograms share this rule. Each line's thermal broadening still uses
its own emitting-state temperature.
[physics/hydrogen_emissivity.py](../src/quokka2s/physics/hydrogen_emissivity.py)
contains the two analytic cold hydrogen emissivities. Physical constants are
centralized in [constants.py](../src/quokka2s/constants.py).

Table tools also have specific shared owners:

- [cloudy/table_definition.py](../src/quokka2s/cloudy/table_definition.py)
  supplies the ordered Cloudy line labels, grid and radiation recipe to the
  builder and packager. [cloudy/incident_spectrum.py](../src/quokka2s/cloudy/incident_spectrum.py)
  reads incident-continuum exports for both the SED generator and radiation
  figures, keeping photon energy in Ryd and intensity in erg/cm²/s.
- [despotic/solver_settings.py](../src/quokka2s/despotic/solver_settings.py)
  supplies adopted solver defaults and reviewed source identities.
  [despotic/table_data.py](../src/quokka2s/despotic/table_data.py) owns table
  records and line-result field names; [despotic/snapshot_domain.py](../src/quokka2s/despotic/snapshot_domain.py)
  checks physical settings and table-axis bounds. The reference snapshot's
  grid dimensions and cell count remain build metadata.
- [file_provenance.py](../src/quokka2s/file_provenance.py) calculates file hashes
  for table-build and diagnostic file identities. Process-time table
  compatibility uses the physical-domain checks above, not file hashes.

Figure 1 and the emission phase diagrams have explicit callers:

- `tools/figures/build_table_input_slice.py`: read the selected x slice, query
  DESPOTIC temperature, save its five fields and prepared panels, then optionally
  draw Figure 1.
- `tools/figures/build_emission_phase_histograms.py`: read slabs, calculate each batch
  once, accumulate two-dimensional histograms, save arrays, then draw
  the ten-panel phase diagram.
- `tools/figures/build_gas_projection_maps.py`: query DESPOTIC temperature and
  accumulate gas projections using cells with an available mixed temperature,
  then save the maps and their display ranges before drawing.

Their `--no-plot` option saves data only; `--plot-only` reads their saved data.
They reuse `Snapshot` directly. Slice and projection preparation use the loaded
`DespoticCellReader`; line-luminosity histograms use `CellEmissionCalculator`.
They do not load a separate workflow framework.
