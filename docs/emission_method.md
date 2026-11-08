# Current emission physics and products

The current entry points are `uv run process` and
`uv run plot`, after automatic editable installation with `uv sync`.
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
metadata are not rewritten to claim a new build. The previous table is kept
as `inputs/tables/cloudy/emission_old.npz`.

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
loader. Common finite/uniform queries use the equivalent native trilinear
kernels; retained reference paths handle the original special policies.
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
With null `prepared_z_fields`, process prepares five temporary C-order XYZ
float64 copies of raw density, temperature and momentum densities, preserving
native coordinates/units and buffering one complete field at a time. An explicit
absolute cache path is validated and only read; process never creates,
overwrites or deletes a file there. Its optional metadata sidecar contains
geometry, units and native block descriptors only. Metadata is restored inside
input loading, or yt opens the original snapshot if no sidecar is present.
Only an owned temporary cache is removed when processing finishes or fails.

The bounded eager reader reads current core x cells and periodic x neighbours
with native full y and complete z, then closes its five field streams. A fused
native pass derives selected complete columns with original unit-conversion
order and momentum/density division, fills six core arrays and releases raw
slab fields. It preserves original periodic x/y gradient neighbours and
one-sided outer-z differences. This route uses no persistent mmap or grouped
multi-slab field cache. Plot reads only saved products.

The six copied core-slab arrays retain the density,
temperature, column, gradient and velocity definitions and units listed in the
[code walkthrough](code_walkthrough.md). All products accumulate only selected
cells, and surface-luminosity spectra use the selected projected area.
Spectra have
400 channels over −200 to +200 km/s. Gaussian thermal profiles are integrated
over channel boundaries. Luminosity outside that window is recorded rather
than moved into edge channels. Each line and dust state has one integrated
spectrum. Cold/hot contributions are retained separately.

The process default `spectral_method: histogram-2d` represents each emitting
cell on a bounded grid in mean velocity `mu` and `ln(sigma)`. Bilinear corner
weights distribute that cell's own luminosity to four neighboring nodes.
Narrow/broad sigma bands retain different mean-velocity resolutions. All
emitting Gaussians must fit the configured grid; values outside it raise rather
than clip, disappear or use a 1D fallback. Intrinsic/attenuated and cold/hot
accounts remain separate. Only identical physical widths share geometry.

Sparse batch increments merge in the original input batch/group order into
40 node accounts (two dust states, ten lines, two temperature branches).
A single standard-normal prefix table is prepared before process timing.
Its dimensionless grid is [-5,5] at spacing 1/128. Stable analytic fine-cell
areas are cumulatively summed from zero; online channel queries use only
linear interpolation and differences. Each node uses `z=(v-mu)/sigma`;
adjacent channels reuse their shared cumulative boundary. The same table
handles different finite mu and positive sigma without rebuilding a mu/sigma
integral template. The preceding two-dimensional luminosity histogram remains
bounded and still requires emitting cells to fit its configured grid. Node
Gaussians truncate outside `mu +/- 5*sigma` without renormalization. Final
expansion reuses a node's probabilities across all its luminosity accounts. Independent
channel rows may run concurrently while each keeps ascending node/profile and
original block/band accumulation order.

The runtime arithmetic gate compares native FMA/separate-multiply-add choices
against an independent NumPy contraction of the same prefix-table
coefficients using exact bytes, including partial and full original blocks.
Its signature binds the actual loaded prefix binary, NumPy dispatch, table,
channel edges, histogram geometry and observed team. If no arithmetic mode
matches, preparation raises rather than silently selecting another method.
The bounded two-dimensional physical approximation remains unchanged.

Four luminosity accounts distinguish actual captured channel light, exact
outside-window light, exact within-window light removed by the five-sigma node
cutoff, and the signed lookup-window residual. Thus:

```text
input = captured + outside + truncated_inside - signed_lookup_residual
signed_lookup_residual = captured - exact_clipped_window
```

The tail accounts apply to histogram-node Gaussians, not an independent
cell-selection cut. No luminosity threshold drops cells. Full cell Gaussian
moments remain untruncated. Grid interpolation changes channel values; the
report records the adopted grid, cutoff, arithmetic gate, accounts and actual
native teams, preserving per-line missingness and physical widths/masses.

Explicit `spectral_method: exact` uses SciPy's double-precision erf, and
`linear-lut` retains the one-dimensional erf table on [0,8] at spacing 1/4096
with odd symmetry and its original handling of boundaries/error policies.
These methods are references, never the 2D default or automatic fallback.
The [usage settings](usage.md#spectral-integration) describe selection and reuse.

The bounded reader uses five I/O streams followed by up to 32 complete-column
derivation tasks in the same pool. Up to 16 batch workers then calculate
emission/products. Main node merging uses four verified OpenMP workers and
final channel expansion eight when supported, otherwise serial one; those
phases do not multiply into a nested per-cell worker count. Thirteen extensions
build normally; Linux verifies compiler/runtime OpenMP support and macOS uses
serial kernels. Explicit unsupported native teams raise.

Gaussian table/geometry/synthetic-gate preparation and actual z-copy conversion
are separately measured outside process elapsed time. Input identity, metadata
construction/restoration, physical table loading, cache validation, every
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
