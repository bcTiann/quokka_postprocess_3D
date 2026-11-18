# Current emission physics and products

The current entry points are `uv run python process_snapshot.py` and
`uv run python plot_emission_results.py`, after automatic editable installation with `uv sync`.
Each uses its default YAML in `configs/`; `--config PATH` selects another file.
Use the [root README](../README.md) for the main commands, the
[usage guide](usage.md) for settings and table building, and the
[code walkthrough](code_walkthrough.md) to follow the code.
This reference describes the adopted scientific choices for that route.

## Cell state and table queries

The density conversion is `nH = X_H * rho / m_H`, with
`X_H = 0.7157683773530885`. The shielding `NH` is the harmonic mean of the +z
and −z hydrogen columns. DESPOTIC is queried at `(nH, NH, dVdr)`; the hot
Cloudy branch is queried at `(nH, NH, T_QUOKKA)`.

Cloudy uses the unscaled Cloudy 17.02 default composition. DESPOTIC retains
GOW defaults: He/H = 0.1, C/H = 1.6e-4, O/H = 3.2e-4, Si/H = 1.7e-6.
The density conversion does not redefine either table's elemental abundances.
The Jeans estimate uses actual mass density, fixed `mu=1`, and the inherited
approximately 100 pc cap. This fixed mu belongs only to the length estimate.

The original ten Cloudy density planes retain their historical `X_H=0.76`
Jeans density conversion, node values and metadata. Two additional lower-density
planes use the corrected `X_H=0.7157683773530885` explicitly. At every added
temperature, both conversions give an uncapped Jeans length above 100 pc, so
the applied model depth is identical. The new density minimum is
`8.030857221391619e-8 cm^-3`; the seven attenuation columns and 21 temperatures
are unchanged. Extension provenance describes only the new planes; historical
metadata are not rewritten to claim a new build. The previous table remains
in Git history (commit `72c7f4884`, path
`inputs/tables/cloudy/emission_old.npz`); it is not a runtime input.

## Line emissivities

The branch selector is always `T_QUOKKA`, with 3000 K assigned to the hot
branch:

| Lines | `T_QUOKKA < 3000 K` | `T_QUOKKA >= 3000 K` |
|---|---|---|
| Halpha, HI 21 cm | Analytic expressions using DESPOTIC state | Cloudy |
| CII | DESPOTIC | Cloudy |
| CIII 977/1907/1909, CIV 1548/1551 | Omitted, zero by prescription | Cloudy |
| CO(1–0), CO(2–1) | DESPOTIC | DESPOTIC |

DESPOTIC line `lumPerH` values are multiplied by the query `nH` to obtain
volume emissivity. Cloudy coefficients are multiplied by `nH**2`.
Cell luminosity is `epsilon * cell_volume`, in erg/s.

Each line carries the temperature of its adopted emitting state alongside its
emissivity. DESPOTIC returns its equilibrium temperature with its fields; Cloudy
uses the QUOKKA temperature supplied to its query. These same temperatures are
used for thermal widths. Both CO lines use DESPOTIC temperature for every cell
with available CO/DESPOTIC results. Cold CIII/CIV have zero light, so their stored
QUOKKA temperature has no spectral effect. A DESPOTIC temperature above 3000 K
does not move a cell into the hot QUOKKA branch.

DESPOTIC interpolates stored field values directly on logarithmic coordinates,
with the adopted SciPy `RegularGridInterpolator` semantics. The process loader
reads its seven required fields while full table tools retain their complete
loader. Standard SciPy sparse multiplication applies the same eight linear
corner weights to the fields together. NumPy prepares those eight weights as
one tensor product of the three adjacent-node weight pairs. This is physical
table interpolation, distinct from Gaussian nearest-node assignment. Explicit
zero weights retain original per-field NaN propagation. Singleton axes and special interpolation/error
settings retain the existing regular-grid path. The shared implementation
is `table_interpolation.py`; no project compilation is required.
Cloudy coefficients are interpolated in logarithm
unless a valid zero corner has non-negligible interpolation weight, when
that line/query uses the coefficient itself. Its attenuation lookup coordinate
is bounded by the table; this does not replace the cell's physical column.

The interpolated DESPOTIC table fills only unavailable values inside the
finite-support convex hull using SciPy `griddata` with linear interpolation.
Finite high-temperature solver solutions are retained. Original solver-failure
masks and remaining numerical-unavailability masks have different meanings.

Each line skips only its own missing emissivities. A missing DESPOTIC result
does not remove a hot cell's Cloudy atomic emission; CO requires DESPOTIC at
all temperatures. Missing values remain NaN, while a physical zero remains
available. Gas-phase statistics use the mixed temperature independently:
cold cells require DESPOTIC temperature and hot cells use QUOKKA temperature.
Per-line missing-cell counts and mass fractions are recorded in
`emission_report.json`.

## Foreground dust

For each cell, the foreground dust column extends from its centre to the
outer −z boundary face along the same `(x,y)` sightline. All cells in front
contribute a full width, and the emitting cell contributes half its width.
This is a separate array from the harmonic-mean shielding column used for
emissivity-table queries.

The pinned Draine Milky Way `R_V=3.1` table supplies total-extinction
cross-section per H nucleus. Log-log interpolation at each rest wavelength
gives `sigma_ext`; attenuation multiplies the intrinsic emissivity by
`exp(-sigma_ext * foreground_NH)`. It includes absorption and scattering out
of the sightline, without scattered-in light. H I 21 cm lies beyond the table
range and is left unchanged by the adopted approximation.

## Saved products and figures

Processing saves intrinsic and attenuated native-resolution LOS-z luminosity
images, integrated spectra, and gas-phase velocity distributions. By default
these cover the full box. Optional native x/y index ranges select a region with
the original full z depth; gradients still use original-box neighbours, and
column densities still use complete z columns. Original snapshots are read-only.
Process reads intersecting original level-zero FAB fields into bounded
z-contiguous RAM tiles. It writes no converted snapshot. Metadata supplies grid
sizes and selected x/y bounds; shorter edge blocks and final partial batches
are supported. Only selected core cells enter products. Gradients retain
global periodic x/y neighbours and one-sided outer-z differences; shielding
and foreground columns retain the complete native z depth.
The six derived arrays preserve the density, temperature, column, gradient and
velocity definitions above. Surface-luminosity spectra use the selected x/y area.
Plot reads only saved products.

Spectra have
400 channels over −200 to +200 km/s. Gaussian thermal profiles are integrated
over channel boundaries. Luminosity outside that window is recorded rather
than moved into edge channels. Each line and dust state has one integrated
spectrum. Cold/hot contributions are retained separately.

The sole process Gaussian method evaluates original
cells through preintegrated fractional-mean templates on logarithmically spaced
sigma nodes. With channel width `delta_v=1 km/s`, write
`mu/delta_v=floor(mu/delta_v)+phase`. Integer shifts
move the stored template; the fractional phase lies in [0,1]. Natural-log sigma
nodes have step 0.03125 and default coverage [1e-5,1024] km/s. The physical width
floor remains 1e-5 km/s. Valid widths outside template coverage are integrated
directly from their original mu/sigma, with stable bin probabilities clipped
at +/-5 sigma and no renormalization or luminosity omission threshold. Invalid
coordinates, unsafe integer shifts and nonpositive/nonfinite widths still raise.

Each sigma layer independently samples phase with transition spacing at most
`min(0.1,0.1*sigma/delta_v)`. When 10 sigma is narrower than a channel, exactly
flat phase interiors can omit redundant knots while retaining fine boundary
transitions. All templates store their phase-union five-sigma support with
zero guard bins; they are not restricted to the observed 400 bins before shifting.
Default coverage can be extended explicitly, for example
`gaussian_grid: {sigma_max: 8192.0}` for a large snapshot with broader emitting
profiles. Coverage counts, observed ranges and luminosity-weighted fallback
fractions are saved per line, dust state and temperature branch. They describe
whether extending the finite table would reduce direct integration work.

For each cell, select the closest actual sigma node by linear distance, then
select the closest actual fractional-phase knot on that sigma layer. Exact ties
choose the lower sigma or phase. A selected phase-one node is represented as
phase zero plus one integer shift. Each covered active cell contributes its
full luminosity to this one node with weight 1. The grid has logarithmic sigma
spacing, but there are no sigma or phase interpolation weights in this step.
This is a two-stage nearest rule, not a global two-dimensional Euclidean
distance and not four-node interpolation.

For node q and account p, accumulate H[q,p] = sum_i L[i,p] over cells selecting
that node and shift. H persists across all batches and tiles of the snapshot;
original-cell arrays do not. The final channel luminosity is
sum_q probability[q,j] * H[q,p]. Intrinsic/attenuated, line and cold/hot accounts remain separate. There
is no cell-brightness cut. Replacing an original Gaussian by its nearest node
introduces a parameter approximation, which was compared against the previous
four-node scheme before adoption. It is distinct from floating-point summation
error; luminosity accounting alone does not measure spectral shape error.

The immutable table is prepared outside process time. Worker batches use NumPy
coordinates to prepare original nearest-node contributions. Separate workers then own
separate profile planes, accumulating original cell luminosity directly into
NumPy arrays. The prebuilt private SciPy 1.16.3 `coo_matvec` kernel performs that
arithmetic; no project code is compiled. Types, lengths, bounds and ownership are
checked before the call. Writers join before their wave-local inputs are freed.
The accumulation preserves each selected node
and its luminosity; rounding can differ with batch reduction order. Template
coverage and original-cell reference integrals are unchanged. Every stored
template row with B bins has K+B-1 possible integer shifts overlapping K observed
bins; each account stores just that row-specific interval. Direct NumPy
convolution in valid mode maps its accumulated shift luminosities to K channels.
All raw-term preparation, in-place accumulation, convolution and
luminosity multiplication remain inside process timing. Original-cell arrays
survive only one bounded wave. Stored probabilities are zero outside each
selected node's own support; no original-cell support mask replaces that
selected template. Each stored node Gaussian truncates at its own
`mu_node +/-5sigma_node`
without renormalization. Original batch contents and width groups are unchanged.
Finished products merge in deterministic native-tile order, with ascending
batches within each tile; floating-point reductions can round differently from
a full-grid batch walk. Full moments use an anchored centroid followed by a
centered second pass, never subtraction of nearly equal raw second moments.
Installed NumPy/SciPy versions and CPU dispatch are recorded with the results;
library reductions can change the last floating-point bits.

The existing complete moments retain original-cell Gaussians and full support.
Independent original-cell analytic endpoint references separately provide the
untruncated and five-sigma-clipped window light. Captured spectrum light can
differ from the clipped reference because of nearest-node parameter
approximation, shifted selected-node supports and floating reduction. Four
accounts therefore distinguish:

```text
input = captured + outside + truncated_inside - signed_lookup_residual
signed_lookup_residual = captured - exact_original_cell_clipped_window
```

These identities close strict independent luminosity budgets without treating
approximation as missing light. The tail bound applies to the original-cell
analytic references; actual queried spectra use the selected-node supports.
Images, gas phases, physical temperatures/widths, per-line missingness and full
moments retain their definitions. Compare spectra and finite-window moments
against a named reference when assessing grid accuracy.

The original-file reader uses a configurable pool of FAB-field read tasks.
Any positive `z_read_workers` is valid; independent fields from different z FABs
can keep more than five readers active. A separate derivation pool uses
`z_reader_threads` tasks on disjoint complete z columns. Each task uses NumPy
blocks targeting at most 131,072 cells, rounded down to whole columns, with
at least one column per block. For example, 16 derivation threads can be selected
in YAML; the useful count depends on the snapshot and machine.

`chunk_workers` separately bounds physical/Gaussian batches in a wave. Those
workers prepare unchanged emission/products, one nearest-node contribution per
covered Gaussian and direct integration for uncovered widths. The same batch
pool adds full cell luminosities into disjoint NumPy profile planes, preserving
original batch/group order within each profile. All writers join before wave-local
luminosity is released. One main thread merges the remaining small products.
Final template convolution occurs after every original cell has been processed;
it remains timed and is reported separately. There is no nested spectral pool.
Project calculations use NumPy/SciPy without custom compiled extensions.

Gaussian template preparation/validation is separately measured outside process
elapsed time. Input identity, metadata construction, physical table loading, every
payload/derived/emission/product operation, checks, save and cleanup remain
inside final `status.json.elapsed_seconds`. The report is assembled before its
own save, so its elapsed time ends earlier. Plotting is a later separate step.

Two dispersions are saved: `line_sigma_window_kms` uses the saved channels;
`line_sigma_full_kms` includes each available cell's complete Gaussian. The
full value is calculated from cell luminosity L, LOS velocity v and the same
thermal width s used for channel integration:

```text
mean_velocity = sum(L * v) / sum(L)
sigma_full² = sum(L * ((v - mean_velocity)² + s²)) / sum(L)
```

These totals are accumulated during the same batch calculation. No wider
channel grid or second table query is needed. Intrinsic and attenuated light
have separate luminosity weights and separate moments. Figure legends use
the saved full-profile dispersion; curves retain the saved velocity channels.

Gas phases use `T_DESPOTIC` when `T_QUOKKA < 3000 K`, otherwise `T_QUOKKA`.
The boundaries 200, 3000, 10^4, and 10^5.5 K define CNM, UNM, WNM, WIM, and
HIM, with equality assigned to the hotter phase. Phase moments retain the
full velocity range of gas with available mixed temperature, including each phase's dispersion about the
common mass-weighted gas mean and about its own mean.

Plotting draws saved native images unless a prepared coarse image is selected.
Numerical preparation sums neighbouring pixel luminosities into a separate
image product, retaining the native data. Normal spectrum comparisons
show cold+hot totals. The peak-normalized gas-phase comparison uses the
attenuated total Halpha/HI/CII/CO and hot CIII/CIV profiles. Its gas
histogram bin centres are connected without additional smoothing. CO's WIM/HIM
curves are faint. Individual-line spectra, gas-phase comparisons and the
all-lines comparisons each have separate ±50 and ±200 km/s display views.
These limits do not change saved products or moment calculations.

The all-lines comparison uses cold+hot total spectra for all ten transitions,
with separate intrinsic and dust-attenuated figures. Each profile is divided
by its own peak among the 400 saved channels over −200 to +200 km/s. Both
display views select this same saved normalized profile; the ±50 view does
not use a second peak normalization. Curves therefore retain the same heights
at the same velocities when the view changes.

Per-area spectra, peak-normalized gas/line profiles and numerical colour limits
are prepared and saved before plotting. Renderers select these fields directly.
