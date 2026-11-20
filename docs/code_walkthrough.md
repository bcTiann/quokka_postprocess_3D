# Reading process → saved data → plot

Start with one explicitly selected snapshot and a small x/y region:

```bash
uv run python process_snapshot.py --snapshot /absolute/path/plt0008703 --x 64 66 --y 64 66
uv run python plot_emission_results.py --snapshot /absolute/path/plt0008703 --x 64 66 --y 64 66
```

These are independent scripts. Process computes and saves; plot reads saved
arrays. A parent run directory is rejected. There is one production Gaussian
method: each original cell's full luminosity goes to one nearest stored node,
then completed node rows convolve into the final 400-bin spectrum. Sigma nodes
are logarithmically placed; select the closest sigma by linear distance, then
the closest actual phase knot on that layer. Exact ties choose the lower knot.

```text
process_snapshot.py
  main: arguments + YAML -> one ProcessSettings
  run_one_snapshot: prepare Gaussian templates before process timing
    load_processing_inputs: metadata + physical tables
    prepare_processing_snapshot: own an original-FAB column reader
    process_snapshot -> process_native_columns:
      read selected full-z XY tile into RAM
      derive_native_column_tile -> six physical cell arrays
      process_template_batches -> bounded physical/Gaussian worker waves
        accumulate_batch -> emission + images + Gaussian inputs + gas phases
        prepare_corner_increments -> one nearest-node contribution per cell
        scatter_corner_wave -> luminosities in shared H arrays
        merge -> small products in input order
    build_outputs -> convolve H rows, prepare all numerical display fields
    check_outputs -> write_emission_results -> completed status

plot_emission_results.py
  main: arguments + YAML -> one PlotSettings
  plot_one_snapshot -> saved numerical arrays -> figures
```

`config` holds paths/settings; `snapshot` holds geometry, units and reader
metadata; `emission_calculator` holds immutable physical lookup tables.
None is a whole-snapshot field array. A tile owns its raw/derived arrays and
releases them before the next tile. Original-cell Gaussian inputs live for
one bounded worker wave. `EmissionProducts` owns snapshot result accumulators,
including H, which retains selected-node luminosity across every batch and tile
until final convolution. Sigma layers are log-spaced, but this Gaussian route
does not interpolate in sigma or phase.

Within a batch, lines using the same gas state share their completed read-only
temperature array. Table queries prepare flat corner indices once for masks and
values, and share adjacent corner-weight factors. Near-uniform axes use a
predicted bracket only after checking the actual nodes; other coordinates use
`searchsorted`. Gaussian preparation reuses the same positive-light mask and
equal full/clipped window probabilities. Node positions, nearest-node selection,
missing-value support and original-cell conservation accounts are unchanged.

Open one function at a time and identify its input, action and output before
following another call. The root scripts only launch package `main()`; all
reusable calculations live under `src/quokka2s/`. `pyproject.toml` declares
package/dependencies without process/plot command aliases. No custom Cython
build is needed. `uv sync` installs the source checkout in editable mode.

Both scripts read `configs/emission.yaml` unless `--config` selects another.
`dataset: null` requires an explicit absolute snapshot path. Alternatively set
one absolute dataset path in YAML. `--jobname` optionally replaces the output
group. CLI x/y arguments replace only their own YAML axis. Plot uses the dataset
folder name without accessing the original path. Repeat CLI selection arguments
when plotting; they do not persist. Process always leaves original files read-only.

For array notation below, N counts selected Gaussian rows, P counts luminosity
columns sharing a width and K=400 counts channels. `CornerTerms.source_cells`
indexes local N rows; `destination_slots` indexes the shared node store. Neither
is a global simulation-cell ID. Some internal names still use `corner`, but
the current Gaussian route emits only one unit-weight node contribution per
covered active cell. Velocities and sigma use km/s; L uses erg/s.

## 1. Follow the stages in `main()`

| Input/action | Function | Output |
|---|---|---|
| CLI + YAML | `load_process_config()` | One resolved `ProcessSettings` |
| Existing selected result directory | `reuse_completed_results()` | Reuse intact matching products, or preserve/report a conflict |
| Gaussian grid settings | `prepare_template_engine()` | Shared immutable templates; outside process timing |
| Snapshot path + region + physical table paths | `load_processing_inputs()` | Snapshot metadata and emission calculator |
| Geometry + line names + templates | `EmissionProducts(...)` | Empty image, H, spectrum and gas-phase sums |
| Snapshot metadata | `prepare_processing_snapshot()` | Owned original-file reader |
| Reader + calculator + products | `process_snapshot()` | Accumulated selected-region products and stage timings |
| Completed accumulators | `build_outputs()` | Three numerical payloads and summaries |
| Geometry + ordered dust values | `outputs.add_metadata(...)` | Saved interpretation fields |
| Products + independent cell totals | `check_outputs()` | Conservation checks |
| Validated products | `write_emission_results()` | Three NPZs and a JSON report |
| Saved result completion | `write_status()` | Final status including saving time |

`main()` processes exactly one selected snapshot. It does not loop over a run
or invoke plot. `run_one_snapshot()` owns that numerical calculation and its
field-reader lifetime. `write_status()` records progress and completion; it
neither calculates emissivities nor reads snapshot fields.

`load_processing_inputs()` checks paths, opens yt metadata with the requested
native x/y selection, then loads DESPOTIC, Cloudy and dust tables. It creates the
output directory only after inputs load successfully. The returned `snapshot`
contains original `shape`, cell widths/volume and native reader metadata;
`processing_shape`, `processing_cell_count` and `processing_area_cm2` describe
the requested region. The calculator holds line definitions and immutable
lookup readers; it does not retain queried cell fields.

Completed-result reuse checks status/report completion, exact configuration,
input/source/runtime identity, NPZ CRCs and required fields. Original payloads
are stat'ed rather than hashed. Table/source hashes detect numerical changes.
Identity is captured before calculation and rechecked before saving. Old or
conflicting results remain untouched and drawable by the separate plot script.

## 2. Read selected full-z tiles from the original plt

`process_snapshot()` calls `native_column_process.process_native_columns()`.

1. `NativeColumnReader` checks level-zero FAB coverage and intersects XY tiles
   with `snapshot.xy_region`. Short edge FABs and non-divisible grid dimensions
   use their actual metadata bounds.
2. `_validate_plan()` checks selected tiles cover every requested cell exactly
   once; it does not require query batches to divide Ny or Nz.
3. `_memory_plan()` reserves current raw/derived arrays, batch workers,
   neighbouring faces and headroom, with a 30 GB RSS guard.
4. `read_tile()` reads intersecting original FAB fields through `np.fromfile`.
   NumPy copies selected values into owned z-contiguous RAM arrays. It also
   obtains original global x/y neighbour faces for gradients. A narrow region
   can still read a whole intersecting FAB field, but no disk copy is created.
5. `derive_native_column_tile()` converts disposable x/y momenta into cgs
   velocities once, waits for every conversion, then reuses these values for
   both gradient neighbours. It releases the x/y storage after gradients,
   converts density/z momentum in place, reuses temperature and
   allocates foreground/shielding columns. Scalar unit factors come from
   `snapshot_units.scalar_unit_stages()` in the original conversion order.
   Each derivation task uses bounded NumPy blocks targeting at most 131,072
   cells, rounded down to complete z columns. At least one complete column is
   retained even when Nz exceeds that target. These blocks do not change the
   physical/Gaussian batch size. `z_reader_threads` controls this derivation
   pool; `z_read_workers` controls a separate pool of original FAB-field reads.
   Any positive reader count is allowed, including more than the five field
   names, because separate z FABs also provide independent read tasks.
6. `slab.iter_batches(batch_size=config.query_chunk)` yields selected cell views,
   including a shorter final batch. `process_template_batches()` finishes each
   bounded wave, merges products in input order and releases worker inputs.
7. Release the tile before reading the next. Completed H luminosities remain
   until final convolution. Status updates after completed tiles.

Grid size comes from metadata, including Nz=2028 or a larger XY box. A selected
1x1 XY region is valid. The existing derivative implementation requires native
Nx>=2 and Nz>=2. A multilevel AMR layout is separate from arbitrary level-zero
grid dimensions and remains unsupported.

Density/temperature/velocity, shielding columns, foreground columns and gradient
formulas retain their scientific definitions. z columns always remain complete,
and a regional boundary never substitutes for a global gradient boundary.
Tile order and within-tile batch order remain deterministic. Original aligned
full-box settings preserve their previous boundaries; newly allowed irregular
sizes/regions can have shorter batches. All reads, RAM copies and derivation
remain inside process time. Counters report requested bytes, not physical disk
traffic.

## 3. Calculate one batch: `emission_calculator.calculate()`

For example, a slab of 4,194,304 cells uses 8 equal batches with the default
`query_chunk: 524288`. Other slab sizes can end with a shorter batch. The serial
caller uses:

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
4. `self.cloudy_reader.read_fields()` queries hot cells independently of
   DESPOTIC and returns successful queries compactly. The named atomic-line
   values are emissivity_per_nH2 [erg cm³/s], accompanied by original cell
   indices, query temperature and physical nH. Failed Cloudy queries leave
   atomic lines missing at those hot cells; they do not remove DESPOTIC CO.
5. `self.calculate_intrinsic_lines()` returns one `IntrinsicLineEmission` per
   line. Each record keeps its emissivity [erg/s/cm³] and the temperature [K]
   of the adopted emitting state together, both `(N,)`. Follow these actions:

   - `calculate_cold_hydrogen_and_cii_lines()`: calculate cold CII and analytic
     Halpha/HI from DESPOTIC fields, keeping its returned temperature.
   - `initialize_ciii_and_civ_lines()`: set cold CIII/CIV emission to zero.
   - `fill_hot_cloudy_lines()`: multiply compact Cloudy coefficients by their
     matching physical `nH**2`, then fill final atomic ε/T arrays at the saved
     cell indices. It retains the exact temperature used for each query.
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
is a compact `(R,)` array for the R successful hot queries. Its entry 7 belongs
to original cell `cloudy_fields.cell_indices[7]`; its matching T and physical nH
have the same compact order. Only the final per-line ε/T arrays have shape `(N,)`
and retain NaN at unavailable positions.
The calculator's emissivity method passes these arrays to the per-species
formula functions; those functions do not receive lookup objects.

Inside `DespoticCellReader.read_fields()`, follow five steps:

1. `self.prepare_query_inputs()` checks and clips nH, shielding NH and dV/dr for
   all N cells. `self.lookup.prepare_queries()` prepares their logarithmic
   coordinates once. Subsequent field queries reuse this batch-local record.
2. `self.lookup.temperature_and_co(queries=queries)` queries T_D, CO10 and CO21
   together for all N cells at those coordinates. The shared
   `table_interpolation.py` applies their original linear corner weights with
   standard SciPy sparse multiplication, retaining explicit zero weights.
   NumPy forms all eight adjacent-node weights together as a tensor product;
   this changes their preparation, not the logarithmic query coordinates or
   per-field missing-value rules.
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
   prepares logarithmic coordinates and reuses the same eight corner weights
   for support checks and linear/log interpolation of successful cells.
   [table_interpolation.py](../src/quokka2s/table_interpolation.py) forms a CSR
   row with those eight weights and multiplies all needed field columns using
   standard SciPy. Explicit zero weights remain to preserve missing-node NaN
   propagation. The returned matrix is line-first. No separate model-depth coordinate is supplied.
3. `self.count_column_clipping()` counts NH clipping among successful queries.
4. Retain successful query rows and their original `cell_indices`. When every
   query succeeds, the eight named coefficient arrays are direct row views of
   the lookup matrix. Otherwise, filter the compact matrix, T and physical nH
   together once. The returned `CloudyCellFields` holds `(R,)` arrays, not eight
   restored N-cell fields. `fill_hot_cloudy_lines()` writes their converted
   emissivities and matching T directly into final per-line arrays; failed hot
   positions remain NaN and the already calculated cold entries are retained.

The lookup uses only the three coordinates above. Jeans length is calculated
when building the table, with mu=1 and a 100 pc cap; it is not a lookup axis.

`CloudyLookup.sample()` is the strict entry point for diagnostics: it raises
if any requested cell has failed support. Both entry points share coordinate
preparation, corner order and support rules. The processing path reinspects
the current zero/failure masks for every query; it does not cache cell emission.

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

Images read each named line and use its intrinsic missing mask for both dust
states. [`CompactImageAccumulator`](../src/quokka2s/products/compact_images.py)
first multiplies each cell's emissivity by volume. Complete z columns reshape
that luminosity into `[column,z]` and sum with the available-cell mask. Partial
first/last columns retain `np.bincount` with per-cell column IDs. Only touched
columns are retained. Their unique pixel IDs allow ordinary NumPy indexed
addition into the main image, without a custom compiled image kernel.
Spectra group lines with identical selections and thermal widths. Default
template preparation writes original-cell mu, sigma and luminosity directly
into bounded batch queues. Workers assign each luminosity to one nearest existing
template node, rather than constructing 400 probabilities for every original
cell. Explicit node references retain their temporary branch
matrices only while adding a batch. Independent cell sums check image and spectrum
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
| Spectrum | [template_spectra.py](../src/quokka2s/products/gaussian/template_spectra.py), `TemplateIntegratedSpectra` with shared preparation | Assign full luminosity to one existing nearest template node, retain H across all tiles, then expand occupied rows once into 400 channels |
| Gas phases | [products/gas_phase_velocity.py](../src/quokka2s/products/gas_phase_velocity.py), `GasPhaseVelocityAccumulator.add_batch()` | Assign the mixed-temperature phase, add cell mass to its velocity channel, and accumulate velocity moments |

Image accumulation uses the selected native x/y sightlines: 256 × 256 for
the full reference snapshot, or 512 × 512 for a full 512 × 512 × 4096 grid.
Spectrum accumulation uses each line's available cells directly and does not
go through a rebinned image. The sole Gaussian route is:

1. Keep the existing cold/hot branches, availability masks, Doppler centers and
   physical thermal widths. Group only lines with identical widths.
2. `template_table.py` constructs sigma layers from 1e-5 to 1024 km/s at natural-log
   spacing 1/32. Each layer samples the mean's fractional channel position from
   0 to 1. Transition spacing is at most `min(0.1,0.1*sigma/delta_v)`; redundant
   exactly flat interior knots may be omitted for narrow profiles. A template
   stores the union of its layer's phase-node five-sigma supports plus guards,
   so it can be shifted before cropping to the observed 400 channels.
3. `template_spectra.py` retains one bounded worker wave's original mu/sigma/L
   vectors while shared preparation computes images, full moments and counts.
   `template_batches.py` then handles those groups in their original order.
4. [template_corners.py](../src/quokka2s/products/gaussian/template_corners.py)
   splits each mean into integer shift and fractional phase. Select the closest
   actual sigma by linear distance, then the closest phase knot on that layer.
   Exact ties choose the lower knot. A selected phase-one knot becomes phase
   zero with integer shift plus one. This is a two-stage rule, not a global
   two-dimensional distance. Each covered active cell emits one unit-weight
   contribution. Workers retain unsorted cell indices and slots, without
   per-worker grouped luminosity sums.
   The final 40 accounts remain separate: two dust states, ten lines and two
   temperature branches. Each line keeps its own availability mask and light.
5. [corner_scatter.py](../src/quokka2s/products/gaussian/corner_scatter.py) runs
   one task per active account in the same worker pool. Each task processes the
   wave's batch/group increments in their original order and adds
   the full cell luminosity (node weight 1) to its own profile plane through the
   installed `scipy.sparse._sparsetools.coo_matvec`. No two tasks write the same
   plane. [corner_store.py](../src/quokka2s/products/gaussian/corner_store.py)
   owns these float64 arrays: a B-bin stored template has K+B-1 integer shifts
   that can overlap the K-bin observation, and its accounts cover exactly those
   shifts. Coordinates are no longer needed after preparing the terms, but
   their read-only luminosity owners remain alive until all profile writers
   join. Main input-light, count, full-moment, direct-fallback and analytic
   endpoint accounts then merge in batch order. After all cells,
   `np.convolve(..., mode="valid")` produces exactly K observed channels from
   each occupied stored row. This sums the selected nearest-node templates.
   Nearest-node assignment introduces a measured parameter approximation;
   it is distinct from floating-point reduction differences.
6. Outside sigma coverage, `template_engine.py` calculates stable original-cell
   bin probabilities in bounded outside-only blocks and accumulates them
   directly. The original-cell full/window and five-sigma-clipped endpoint
   accounts are always calculated from the original mu/sigma/L, not from the
   selected-node sums. Invalid coordinates still raise; valid uncovered sigmas use
   direct +/-5sigma integrals without a luminosity omission threshold. To reduce
   direct integration for broader known widths,
   set `gaussian_grid: {sigma_max: 8192.0}` before preparation.

The COO operation is a private API of the pinned SciPy 1.16.3 installation,
not a project-compiled extension or a public compatibility promise. Gaussian
preparation checks the exact version, availability and a probe with repeated
indices and nonzero initial output. Before calling it, the wrapper validates
signed native int32/int64 indices, equal lengths, bounds, aligned contiguous
float64 buffers, matching template-store identity and nonoverlapping outputs.
Owned term arrays and borrowed luminosity stay read-only for the wave; callers
must not reopen their write flags or mutate the owners. On submission or worker
failure, pending tasks are cancelled and every running writer is joined before
releasing borrowed inputs. A partly modified failed store cannot be finalized
or saved. This buffer contract matters because the private kernel performs no
index-bounds validation itself.

`template_coverage.py` separately merges counts, min/max ranges and per-line,
dust-state and cold/hot fallback luminosity references. Final logs summarize
these fields; `spectral_report.*.template_coverage` retains the same account data.
`fallback_probability_seconds` sums worker time generating outside-only bin
probabilities; `integration_seconds` includes all lookup/direct work and is also
a worker sum. Actual Gaussian activity/wave elapsed times remain under
`processing_timings.gaussian`; these counters are not substitutes for wall time.

Each selected template node has its own five-sigma support without
renormalization. Independent original-cell analytic endpoint references describe
full and clipped window light. The signed residual includes the nearest-node
parameter approximation, shifted selected-node support and floating reduction. The identity
`input = captured + outside + truncated_inside - signed_lookup_residual`
closes strict independent budgets. Complete cell moments
remain untruncated, and per-line missingness remains independent.

The same luminosities, velocities and thermal widths also enter
[products/line_velocity_moments.py](../src/quokka2s/products/line_velocity_moments.py).
The 2D `shared_width_moments()` path preserves each line's positive mask,
velocity anchor and NumPy reductions while reusing identical velocity/thermal
arrays. Its `LineVelocityMoments` records contain total light, mean velocity
and centered second moment, including every complete cell Gaussian.
`merged()` combines these totals across batches, workers and cold/hot branches. It also includes the separation between batch mean velocities.
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
`TemplateIntegratedSpectra` retains the common spectrum arrays below, small
endpoint references and profile-major template-row/integer-shift accounts. Direct
out-of-table contributions merge before the final node expansion; output
construction completes the 400-channel spectrum:

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

When nearest-node preparation finishes, the batch's coordinates and emissivity arrays
can be released. Its packed luminosity remains borrowed until every profile
writer for that wave joins, then the raw terms and luminosity are released.
When the slab finishes, the six slab arrays are released. The lookup tables and
accumulated products remain for the next slab.

## 5. Then read concurrency and saving

Start with `process_snapshot()`: the default dispatches into
`native_column_process.process_native_columns()`. Each tile finishes its worker
deposits/direct fallbacks and merges finished products before releasing its
fields. Tile visit order is deterministic; no delayed products or original-cell
Gaussian inputs survive a tile. Accumulated selected-node luminosities
remain until the final expansion.

`template_batches.process_template_batches()`. `chunk_workers` limits the wave
size, including when it is 1. Workers first call the same `accumulate_batch()`
used for every batch. The defaults are 16 batches of at most
524288 cells, keeping at most 8388608 cells in a full wave. The direct-template
branch fills its owned column-contiguous `[cell, profile]` luminosity queues
from selected emissivity fields. Each profile therefore reads a contiguous
cell-luminosity vector. Stable NumPy full moments reuse those queues in two
passes, avoiding branch-wide luminosity/temperature matrices and repeated temporary vectors.
Gaussian inputs survive only this bounded worker wave. The next phase prepares
the original unsorted nearest-node terms and calculates original-cell endpoint
accounts/direct fallbacks. After these batch tasks finish, the same pool
scatters all wave contributions into disjoint profile planes, with one task per
active account. Luminosity stays alive through that joined scattering phase.
The main thread then merges small products and accounts in batch order via
`EmissionProducts.merge()`. The main process merges each completed wave immediately. Final stored-row
convolution runs once after the last wave.

`processing_timings.gaussian` separates query-plus-addition elapsed union from
whole Gaussian-wave wall time including independent analytic accounts and
scheduling. The sum of overlapping worker call times is a diagnostic, not wall
time. Timing lists are summarized and cleared after each completed wave.
Node-store initialization, in-place profile scattering, ordered small-product
merge and final convolution have separate clocks. In-place scattering already
belongs to the Gaussian activity union and broader Gaussian-wave wall time;
adding its separate counter again would double-count it.
`gaussian_total_seconds_including_corner_store` includes the Gaussian waves,
store initialization and finalization; per-worker term preparation alone is
not the complete Gaussian cost. `final_corner_expansion_pending` remains true
until finalization, and the completion report is refreshed after that final
expansion.

The end of `run_one_snapshot()` keeps construction, checking, reporting and writing separate:

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
and full-moment light. It checks the channel integral against captured light
and the cutoff-aware four-account identities against cell sums. These comparisons use
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
| `execution` | Original-file source, query-batch size, and worker counts |
| Counts, masses, and conservation checks | Per-line missing emissivities, unavailable gas temperatures, luminosity sums, and spectral-window losses |
| `cold_gas_luminosity_by_despotic_temperature` | Cold-branch Halpha/HI/CII light above four TD thresholds, corresponding total light, and intrinsic/attenuated fractions |
| `elapsed_seconds` | Process time through report construction, excluding Gaussian build. Final `status.json` includes saving and cleanup |

`write_emission_results()` receives only the output directory, finished payloads
and report. It writes three uncompressed NPZ files and `emission_report.json`; it does not
open yt, query tables, add metadata or perform checks. `main()` marks completion
only after writing succeeds. `status.json` records progress or a failure,
including the failed stage; it is not a resume checkpoint.

## 6. Follow `plot`

Open [plot_emission_results.py](../src/quokka2s/plot_emission_results.py) and start at `main()`.

1. `load_plot_config()` reads shared YAML/CLI choices into one `PlotSettings`.
   It selects saved product/figure paths from the dataset folder name and optional
   region without accessing the original snapshot or physical tables.
2. `main()` calls `plot_one_snapshot()` once. `read_emission_results()` loads the
   saved NPZs and report. Incomplete diagnostic products require explicit
   `allow_partial: true`. Process never calls this entry point.
3. `draw_emission_products()` selects each line's saved attenuated total spectrum
   by name for the gas-phase comparison. Images, normalized gas/line curves,
   surface-luminosity spectra and colour limits are already saved; no numerical
   preparation occurs here. An optional resolution setting selects
   `images_factor_2.npz`, prepared separately from native pixels by
   `prepare_emission_results.py`.
4. It draws images with
   [figures/line_luminosity_images.py](../src/quokka2s/figures/line_luminosity_images.py),
   spectra with [figures/line_spectra.py](../src/quokka2s/figures/line_spectra.py),
   phase comparisons with
   [figures/gas_phase_spectra.py](../src/quokka2s/figures/gas_phase_spectra.py),
   and all-lines comparisons with
   [figures/all_line_spectra.py](../src/quokka2s/figures/all_line_spectra.py).
   Each selected style draws its images once and its spectral figures with ±50
   and ±200 km/s limits. The limits affect axes only, and filenames distinguish
   the views with `_v50` and `_v200`.
   With multiple workers, image jobs receive the saved planes for one line and
   spectrum jobs are grouped by renderer and velocity window. The parent
   loads the shared font cache before the workers start with Matplotlib's Agg
   backend. `plot_workers` caps the number of rendering processes, subject to
   available CPUs and jobs. A value of 1 draws serially with the same renderers.
5. Each plotting step saves only the selected formats. Titled figures go in
   `figures_titled/`, and untitled figures in `figures/`, using `png/` and/or
   `pdf/` subdirectories. `--png --pdf --titled --untitled` writes both formats
   and both styles from the same loaded products. Previously saved files in
   unselected formats or styles remain in place.

No step reopens yt, the snapshot, DESPOTIC, or Cloudy tables. Plotting selects
saved fields and applies labels, colours, line styles, axis windows and layout.
It does not calculate normalization, moments, physical conversions or image
pixel sums. The normal spectrum and all-lines plots use cold+hot totals. The
all-lines renderer draws separate intrinsic and attenuated figures, with ten
saved peak-normalized curves per figure. The peak is taken across the complete
saved ±200 km/s window during numerical preparation and is shared by both
display views. Selecting the ±50 view therefore does not renormalize a curve.
Phase-comparison
plots use total spectra for every line; CIII/CIV have zero cold emission, so
their totals equal their hot profiles. The saved spectra retain both regimes.
Curves use the saved velocity axis. Sigma labels use the full-profile
moments, including emission outside the saved channel window. Both full and
window moments remain available in the saved numerical products.

For an optional new image resolution, run
`uv run python -m quokka2s.prepare_emission_results --products /absolute/path/to/processed --image-downsample-factor 2` once.
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
