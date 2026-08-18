# Cloudy-native HM2012 + foreground-filtered Black (1987) ISRF

## 1. Desired incident radiation

The radiation incident on the illuminated face of the Cloudy cell is

$$
J_\nu^{\rm surface}
=J_\nu^{\rm HM12,Cloudy}
+T_\nu(N_{\rm H,fg}=10^{21}\,{\rm cm^{-2}},\mathrm{leak}=0)
J_\nu^{\rm ISM}
+J_\nu^{\rm CMB}(z=0).
$$

The production workflow now uses Cloudy 17.02's native

```text
table HM12 redshift 0
```

for the first term.  It no longer uses Grackle's
`HM12_UVB/z_0.0000e+00.out` file as a production input.

The production parameter files read the first two terms from one custom SED
and add the CMB with Cloudy's native `CMB redshift 0` command.  The CMB is not
included automatically by either `table HM12` or `table ISM`.

### Why the HM2012 input was changed

The blue curve below was obtained from the Grackle HM2012 command file and the
dashed orange curve from Cloudy's native `table HM12 redshift 0`.  Both were
previously exported with Cloudy `save incident continuum`; the plot script now
simply reads the two existing files and overlays their first two columns.

![Grackle HM2012 versus Cloudy native HM12](output/radiation_fields/hm12_grackle_vs_cloudy_simple.png)

The continua overlap almost exactly above roughly 0.009 Ryd, including the
UV, hydrogen-ionizing, and X-ray ranges used by the line calculation.  They
differ in the very-low-energy tail because the Grackle file explicitly sets
that tail to a negligible value, whereas Cloudy's native HM12 supplies its own
low-energy continuation.  A matched nine-temperature test previously found
identical C II, H-alpha, and H I emissivities to CIAOLoop's 0.0001 dex output
precision.  We nevertheless use the native Cloudy command from now on so that
the radiation definition and Cloudy version are explicit and self-contained.

The simple comparison is reproduced with:

```bash
MPLCONFIGDIR=/private/tmp/quokka_mpl_cache \
  conda run -n quokka python \
  scripts/compare_grackle_hm12_with_cloudy_hm12.py
```

That script reads:

```text
work/cloudy_cooling_tools_history/examples/grackle/HM12_ISM_NH21/export_hm12_external.inc
work/cloudy_cooling_tools_history/examples/grackle/HM12_ISM_NH21/export_hm12_native.inc
```

The second file, `export_hm12_native.inc`, is obtained directly from Cloudy.
The input file `export_hm12_native.in` contains:

```text
title export Cloudy native table HM12 redshift 0
table HM12 redshift 0
hden -10
constant temperature 1e4 K
stop zone 1
set dr 0
save incident continuum "export_hm12_native.inc"
```

It is run with:

```bash
/Users/tianbaochen/cloudy/c17.02/source/cloudy.exe \
  -r export_hm12_native
```

Here, `table HM12 redshift 0` asks Cloudy for its built-in HM12 spectrum, and
`save incident continuum "export_hm12_native.inc"` writes that spectrum to the
`.inc` file.  The first column is photon energy in Rydbergs and the second is
$\nu\,4\pi J_\nu$ in $\mathrm{erg\,cm^{-2}\,s^{-1}}$.  Only after Cloudy has
created the file does the plotting script read it:

```python
cloudy = np.loadtxt(data_dir / "export_hm12_native.inc")
energy_ryd = cloudy[:, 0]
nu_4pi_Jnu = cloudy[:, 1]
```

This export is automated in `scripts/build_hm12_filtered_ism_sed.py`, which
writes the input above and invokes `cloudy.exe -r export_hm12_native`.

Only this diagnostic comparison retains a reference to the former Grackle
HM2012 export.  The new production builder and tables do not read it.

## 2. Black/ISM component and foreground attenuation

`table ISM` is Cloudy's built-in Black (1987) local interstellar radiation
field.  The green curve is the field without additional foreground
attenuation.  The dashed curve is produced with

```text
table ISM
extinguish column=21 leak=0
```

and is the second term used in the desired radiation sum.

![Black/ISM before and after the foreground column](output/radiation_fields/cloudy_BlackISM_NH0_vs_NH21.png)

In this figure, $N_{\rm H,fg}=0$ means that no additional `extinguish`
command was applied.  The foreground operation leaves the non-ionizing field
below 1 Ryd unchanged, removes almost all of the interpolated 1--4 Ryd
ionizing continuum, and permits the most penetrating hard photons to reappear
gradually at high energies.  `leak=0` means that absorbed radiation is not
added back as an unattenuated geometrical leak.  It does not mean zero
transmission at every energy.

Both radiation figures show Cloudy's `save incident continuum` quantity

$$
\nu\,4\pi J_\nu\quad[\mathrm{erg\,cm^{-2}\,s^{-1}}]
$$

against photon energy in Rydbergs.

## 3. Why the ISM component is filtered separately

Cloudy applies `extinguish` to the combined incident continuum, not only to
the most recently entered radiation component.  Therefore this input is not
the desired model:

```text
table HM12 redshift 0
table ISM
extinguish column=21 leak=0
```

It would attenuate both HM2012 and Black/ISM.  Instead, the workflow:

1. exports Cloudy's native `table HM12 redshift 0`;
2. exports `table ISM` after applying the foreground `extinguish` operation;
3. adds the two continua in linear `nu * 4 pi * J_nu` units;
4. writes the sum as one custom Cloudy SED;
5. solves the gas once under that combined incident radiation.

Separate HM2012-only and ISM-only line emissivities must not be added because
the ion fractions, electron density, level populations, and radiative transfer
respond nonlinearly:

$$
\epsilon_{\rm HM12+ISM}
\ne\epsilon_{\rm HM12}+\epsilon_{\rm ISM}.
$$

## 4. Build the HM2012 plus filtered-ISM Cloudy SED

The previous sections produced the two tabulated quantities needed by the gas
model:

```text
export_hm12_native.inc     Cloudy's native HM12 spectrum
export_ism_filtered.inc    Black/ISM after foreground NH=1e21 attenuation
```

The purpose of `scripts/build_hm12_filtered_ism_sed.py` is to combine them.
It takes the second column, $\nu\,4\pi J_\nu$, from both files on the same
Cloudy energy grid and adds it point by point:

$$
(\nu 4\pi J_\nu)_{\rm combined}
=(\nu 4\pi J_\nu)_{\rm HM12}
+(\nu 4\pi J_\nu)_{\rm ISM,filtered}.
$$

The sum is written as:

```text
work/cloudy_cooling_tools_history/examples/grackle/HM12_NATIVE_ISM_NH21/z_0.0000e+00.sed
```

This is the single incident spectrum that the production line calculations
read.  The production `.par` files then add `CMB redshift 0` as a separate
Cloudy continuum component.  Keeping the CMB separate is equivalent to adding
it to the custom SED in linear intensity, but makes its physical origin and
redshift explicit in every production input.

Cloudy's `table SED` command reads the spectral shape, but it also needs one
reference point that fixes the absolute intensity of the entire curve.  This
is not a normalization to unity and does not rescale HM12 to an arbitrary
strength.

The reference point is chosen at 1 Ryd.  From the exported native HM12 field,

$$
(\nu F_\nu)_{1\,\mathrm{Ryd}}=3.28517\times10^{-7}
\ \mathrm{erg\,cm^{-2}\,s^{-1}}.
$$

Using $\nu_{1\,\mathrm{Ryd}}=3.28984\times10^{15}\,\mathrm{Hz}$ gives

$$
\log_{10}F_\nu
=\log_{10}\left(\frac{\nu F_\nu}{\nu}\right)
=-22.0006176315.
$$

The filtered ISM contribution at 1 Ryd is effectively zero, so the HM12 value
is also the value of the combined spectrum at this reference energy.  The
corresponding Cloudy absolute-intensity command is:

```text
f(nu) = -22.0006176315 at 1 Ryd
```

Run the complete export-and-combine step from the repository root:

```bash
cd /Users/tianbaochen/quokka_postprocess_3D

MPLCONFIGDIR=/private/tmp/quokka_mpl_cache \
  conda run -n quokka python \
  scripts/build_hm12_filtered_ism_sed.py
```

Finally, the script asks Cloudy to read the new `.sed` and export it again.
Comparing this round-trip export with the numerical sum checks that Cloudy
interprets the custom spectrum correctly.  The maximum difference is
$2.43\times10^{-5}$ dex.

The builder also has an `--include-cmb` diagnostic mode that writes a custom
SED containing all three terms.  The production tables documented below use
the two-component SED plus a separate `CMB redshift 0` command instead.

## 5. Current CIAOLoop six-line production grids

The native-HM12 radiation-only baseline `.par` files are:

```text
vendor/cloudy_cooling_tools/examples/grackle/hm2012_native_plus_filtered_ism_defaultabund_sixline_column_10x10x21.par
vendor/cloudy_cooling_tools/examples/grackle/hm2012_native_plus_filtered_ism_defaultabund_sixline_jeans_10x21.par
```

Those two files are the original radiation-only baseline.  The latest adopted
configuration, used for the current LOS-y and LOS-z spectra, is:

```text
vendor/cloudy_cooling_tools/examples/grackle/hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_column_10x10x21.par
vendor/cloudy_cooling_tools/examples/grackle/hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_jeans_10x21.par
```

They enter the new SED directly:

```text
command table SED "HM12_NATIVE_ISM_NH21/z_0.0000e+00.sed"
command f(nu) = -22.0006176315 at 1 Ryd
command CMB redshift 0
command cosmic rays rate -16.698970
```

There is no `loop [init "HM12_UVB/z_*.out"]` in the new production files.

Both grids calculate the same six lines in one Cloudy solution:

```text
C  2 157.636m
H  1 6562.81A
H  1 21.1207c
C  3 977.020A
C  3 1906.68A
C  3 1908.73A
```

The current physical settings are:

```text
Cloudy 17.02 default abundances
molecular chemistry enabled (no `no H2 molecule` command)
charge transfer enabled (no `no charge transfer` command)
hydrogen cosmic-ray ionization rate = 2e-17 s^-1
CMB redshift = 0
no grains
T = 3.6 K to 1e9 K, 21 logarithmic samples
nH = 10 samples from log10(nH)=-4.7142857 to 6
NH = 10 samples from log10(NH)=15 to 24 for the column table
maximum Jeans length = 100 pc for the Jeans table
```

The one-point smoke test at
$(n_H,N_H,T)=(1\,\mathrm{cm^{-3}},10^{20}\,\mathrm{cm^{-2}},10^4\,\mathrm{K})$
completed successfully and returned all six line emissivities.

Run the current full column grid followed automatically by the Jeans grid with
11 workers and macOS sleep prevention:

```bash
scripts/run_hm12_filtered_ism_cmb_cr_mol_ct_sixline_tables.sh 11
```

The new output directories are:

```text
work/cloudy_cooling_tools_history/examples/grackle/hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_column_10x10x21_output/
work/cloudy_cooling_tools_history/examples/grackle/hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_jeans_10x21_output/
```

The molecular-plus-charge-transfer run produced all 100 column maps and all 10
Jeans maps, with 21 temperature rows per map.  Its raw outputs contain 8
failed column-grid nodes and 23 failed Jeans-grid nodes.  These failures are
retained explicitly rather than silently filled.

## 6. Downstream rebuild after Cloudy completes

Build the raw line-table bundles without filling failed Cloudy nodes:

```bash
conda run -n quokka python \
  scripts/build_hm12_filtered_ism_sixline_bundles.py \
  --stem hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline \
  --charge-transfer-enabled \
  --cosmic-ray-rate-s 2e-17 \
  --cmb-redshift 0 \
  --molecular-network-enabled
```

The new products use explicit native-HM12 names:

```text
data/cloudy_hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_column_10x10x21.npz
data/cloudy_hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_jeans_10x21.npz
data/cloudy_hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_failure_nodes.json
```

Then audit whether the actual simulation interpolation stencils touch any raw
failure node:

```bash
conda run -n quokka python \
  scripts/sample_hm12_filtered_ism_sixline_failures.py \
  --column-table data/cloudy_hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_column_10x10x21.npz \
  --jeans-table data/cloudy_hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_jeans_10x21.npz \
  --output output/plt0655228_down1_Lext15kpc/hm12_native_filtered_ism_cmb_cr_mol_ct_sixline_failure_sampling.json \
  --force
```

The temperature policy remains:

```text
split cells with T_QUOKKA at 3000 K
use T_DESPOTIC as the Cloudy lookup and thermal temperature below 3000 K
use T_QUOKKA at or above 3000 K
```

If no simulation cell touches a raw failure node, no interpolation or repair is
performed.  For `plt0655228`, all 134,217,728 cells are inside both table
domains and no interpolation stencil touches a failed node.

Generate the LOS-y, `R=infinity` spectra with:

```bash
MPLCONFIGDIR=/private/tmp/quokka_mpl_cache \
  caffeinate -dimsu \
  conda run -n quokka python \
  scripts/plot_hm12_filtered_ism_sixline_spectra.py \
  --column-table data/cloudy_hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_column_10x10x21.npz \
  --jeans-table data/cloudy_hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_jeans_10x21.npz \
  --output-dir output/plt0655228_down1_Lext15kpc/native_hm12_filtered_black_ism_cmb_cr_mol_ct_sixline \
  --state-key cloudy_native_hm2012_filtered_ism_cmb_cr_mol_ct \
  --cloudy-label "Cloudy HM2012 + filtered ISM + CMB + CR + molecular + charge transfer" \
  --filename-tag nativeHM2012_filteredISM_CMB_CR_Mol_CT \
  --los y --workers 11 --force
```

The LOS-z calculation must use `velocity_z`, the x-y projected area, and a
wider velocity range.  It is generated independently with:

```bash
MPLCONFIGDIR=/private/tmp/quokka_mpl_cache \
  caffeinate -dimsu \
  conda run -n quokka python \
  scripts/plot_hm12_filtered_ism_sixline_spectra.py \
  --column-table data/cloudy_hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_column_10x10x21.npz \
  --jeans-table data/cloudy_hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_jeans_10x21.npz \
  --output-dir output/plt0655228_down1_Lext15kpc/native_hm12_filtered_black_ism_cmb_cr_mol_ct_sixline_LOSz \
  --state-key cloudy_native_hm2012_filtered_ism_cmb_cr_mol_ct \
  --cloudy-label "Cloudy HM2012 + filtered ISM + CMB + CR + molecular + charge transfer" \
  --filename-tag nativeHM2012_filteredISM_CMB_CR_Mol_CT \
  --los z --velocity-range-kms 200 --workers 11 --force
```

The new spectrum directory is:

```text
output/plt0655228_down1_Lext15kpc/native_hm12_filtered_black_ism_cmb_cr_mol_ct_sixline/
output/plt0655228_down1_Lext15kpc/native_hm12_filtered_black_ism_cmb_cr_mol_ct_sixline_LOSz/
```

The failure audit is recorded in:

```text
output/plt0655228_down1_Lext15kpc/hm12_native_filtered_ism_cmb_cr_mol_ct_sixline_failure_sampling.json
```

The completed LOS-z report records `velocity_z`, the x-y projected area
$9.108324\times10^{42}\,\mathrm{cm^2}$, 300 channels over
$-200\leq v_z\leq200\,\mathrm{km\,s^{-1}}$, and conservation of the integrated
line luminosity within the selected velocity domain.

New C II figures retain DESPOTIC as the independent reference; new H-alpha and
H I figures retain the analytic pipeline reference.  Former Cloudy curves
built from the external Grackle HM2012 input are deliberately not copied into
the native-HM12 figures.

## 7. Historical-output boundary

Files containing `HM12_ISM_NH21`, without `NATIVE` in the radiation-directory
name, and tables named `hm2012_plus_filtered_ism...`, without `native`, are the
older external-Grackle-HM2012 results.  The no-CMB/no-CR/no-H2/no-charge-transfer
tables named `hm2012_native_plus_filtered_ism_defaultabund...` are the native
HM12 radiation-only baseline.  Both are retained for controlled comparison;
neither is the current molecular-plus-charge-transfer production state.
