# Current emission physics and products

The current entry points are `python -m quokka2s.process_snapshot --config ...` and
`python -m quokka2s.plot_emission_results --config ...`. Use the [root README](../README.md) for
commands and [code walkthrough](code_walkthrough.md) to follow the code.
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

The current three-dimensional Cloudy table was built with the older
`X_H=0.76` Jeans density conversion. The hot-cell adapter permits its reuse
only where each queried cell and every contributing table node remain capped
under the checked conversions. It does not assume equivalence below the cap.
The current eight-line builder explicitly supplies the corrected `X_H`.
Historical table metadata are not rewritten to claim a new build.

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

DESPOTIC uses SciPy `RegularGridInterpolator` on logarithmic coordinates and
interpolates the stored field values directly. Cloudy also uses
`RegularGridInterpolator`: its coefficient is interpolated in logarithm
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
column densities still use complete z columns. All products accumulate only
selected cells, and surface-luminosity spectra use the selected projected area.
Spectra have
400 channels over −200 to +200 km/s. Gaussian thermal profiles are integrated
over channel boundaries. Luminosity outside that window is recorded rather
than moved into edge channels. Each line and dust state has one integrated
spectrum. Cold/hot contributions are retained separately.

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

Plotting draws images at native resolution unless a pixel-binning factor is
selected. It sums pixel luminosities when binning. Normal spectrum comparisons
show cold+hot totals. The peak-normalized gas-phase comparison uses the
attenuated total Halpha/HI/CII/CO and hot CIII/CIV profiles. Its gas
histogram bin centres are connected without additional smoothing. CO's WIM/HIM
curves are faint. Display limits of ±50 km/s do not change saved products or
moment calculations.
