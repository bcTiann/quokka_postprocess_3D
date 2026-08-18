# Reproducing the Cloudy six-line tables

This document describes the portable workflow used to build the current
Cloudy lookup tables. A fresh clone does not need the author's directory
layout and does not read any file under an old `work/` checkout.

## 1. Required software

- Cloudy 17.02, compiled on the machine that will build the tables;
- Python 3.10 or newer;
- NumPy and Matplotlib;
- Perl, used by the vendored `CIAOLoop_lines` driver.

Install the repository's Python environment with:

```bash
git clone https://github.com/bcTiann/quokka_postprocess_3D.git
cd quokka_postprocess_3D
python -m pip install -r requirements.txt
python -m pip install -e .
```

Cloudy itself is not bundled. Pass its executable explicitly or set
`CLOUDY_EXE`:

```bash
export CLOUDY_EXE=/path/to/cloudy/c17.02/source/cloudy.exe
test -x "$CLOUDY_EXE"
```

## 2. One-command reproduction

First run the one-point smoke test. It builds the incident radiation field,
renders a portable CIAOLoop parameter file, and evaluates all six lines at
one `(n_H,N_H,T)` point:

```bash
python scripts/reproduce_cloudy_sixline_tables.py \
  --cloudy-exe "$CLOUDY_EXE" \
  --smoke-only
```

If the smoke test succeeds, build both production tables:

```bash
python scripts/reproduce_cloudy_sixline_tables.py \
  --cloudy-exe "$CLOUDY_EXE" \
  --workers 11 \
  --force
```

`--workers` controls the maximum number of simultaneous local Cloudy
processes. Start with a value appropriate for the machine. `--force`
replaces generated runtime outputs and final table products; it does not
modify source templates.

The final products are:

```text
data/cloudy_hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_column_10x10x21.npz
data/cloudy_hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_jeans_10x21.npz
data/cloudy_hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_failure_nodes.json
```

Generated SEDs, rendered parameter files, raw CIAOLoop maps, and logs are
kept under:

```text
runtime/cloudy_sixline/
```

Both `runtime/` and `data/` are ignored by Git because they are generated
products, not source files.

## 3. Physical model

The radiation incident on the illuminated face of each Cloudy calculation is

$$
J_\nu^{\rm surface}
=J_\nu^{\rm HM12,Cloudy}(z=0)
+T_\nu(N_{\rm H,fg}=10^{21}\,{\rm cm^{-2}},\mathrm{leak}=0)
J_\nu^{\rm ISM}
+J_\nu^{\rm CMB}(z=0).
$$

The components are:

- Cloudy's native `table HM12 redshift 0` spectrum;
- Cloudy's Black (1987) `table ISM` spectrum after foreground filtering with
  `extinguish column=21 leak=0`;
- `CMB redshift 0`.

The foreground extinction must be applied to the ISM component before it is
combined with HM2012. Placing `extinguish` after both spectra in one Cloudy
input would attenuate HM2012 as well, which is not the adopted model.

The other production settings are:

```text
Cloudy version                 17.02
element abundances             Cloudy defaults
H cosmic-ray ionization rate   2e-17 s^-1
molecular chemistry            enabled
charge transfer                enabled
grains                         not added
turbulence                     not added
temperature grid               3.6 K to 1e9 K, 21 log-spaced samples
density grid                   log10(n_H/cm^-3) = -4.7142857 to 6, 10 samples
column grid                    log10(N_H/cm^-2) = 15 to 24, 10 samples
maximum Jeans length           100 pc
```

Molecular chemistry and charge transfer are enabled by retaining Cloudy's
defaults: the production inputs do not issue `no H2 molecule` or
`no charge transfer`.

## 4. Lines and table axes

All six lines are calculated in each Cloudy solution:

```text
C  2 157.636m
H  1 6562.81A
H  1 21.1207c
C  3 977.020A
C  3 1906.68A
C  3 1908.73A
```

The column-density table has axes:

```text
(line, log_nH, log_NH, log_T) = (6, 10, 10, 21)
```

The Jeans-length table has axes:

```text
(line, log_nH, log_T) = (6, 10, 21)
```

For the Jeans table, CIAOLoop computes the attenuation length from the local
density and fixed temperature and caps it at 100 pc. For the column table,
Cloudy is stopped at each explicitly tabulated `N_H`.

## 5. What the entry script does

`scripts/reproduce_cloudy_sixline_tables.py` performs these steps in order:

1. validates the supplied Cloudy executable and vendored CIAOLoop driver;
2. calls `scripts/build_hm12_filtered_ism_sed.py`;
3. exports native HM12 and the unfiltered and filtered Black/ISM continua;
4. adds native HM12 and filtered ISM in linear
   `nu * 4 pi * J_nu` units;
5. writes and round-trip checks the custom Cloudy SED;
6. renders machine-specific `.par` files from the tracked `.par.in` templates;
7. runs the one-point smoke test;
8. runs the 100-map column grid and the 10-map Jeans grid;
9. calls `scripts/build_hm12_filtered_ism_sixline_bundles.py` to create the
   compact NPZ tables and failure manifest.

The tracked templates are:

```text
vendor/cloudy_cooling_tools/examples/grackle/hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_smoke.par.in
vendor/cloudy_cooling_tools/examples/grackle/hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_column_10x10x21.par.in
vendor/cloudy_cooling_tools/examples/grackle/hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline_jeans_10x21.par.in
```

Only two values are substituted into the templates:

```text
@CLOUDY_EXE@   absolute path supplied by the reproducer
@OUTPUT_DIR@   generated runtime output directory
```

The physical commands and grid axes are not rewritten by the reproducer.

## 6. Incident SED construction

The builder exports the two adopted non-CMB components with Cloudy 17.02
`save incident continuum`. On Cloudy's common energy grid it computes

$$
(\nu 4\pi J_\nu)_{\rm combined}
=(\nu 4\pi J_\nu)_{\rm HM12}
+(\nu 4\pi J_\nu)_{\rm ISM,filtered}.
$$

The result is written to:

```text
runtime/cloudy_sixline/examples/grackle/HM12_NATIVE_ISM_NH21/z_0.0000e+00.sed
```

The production parameter files read it with:

```text
command table SED "HM12_NATIVE_ISM_NH21/z_0.0000e+00.sed"
command f(nu) = -22.0006176315 at 1 Ryd
command CMB redshift 0
```

`f(nu)` fixes the absolute intensity of the tabulated SED at 1 Ryd. It is
derived from the native Cloudy HM12 export; it is not a normalization to unity
and does not arbitrarily rescale the field. The CMB is added as a separate
Cloudy continuum component.

The builder then asks Cloudy to export the custom SED again. This round-trip
comparison verifies that Cloudy reads the written spectrum with the intended
shape and absolute scale.

## 7. Failure handling and validation

A CIAOLoop value of `-99` means that Cloudy returned zero emissivity and is
converted to an exact physical zero. A missing or crashed grid row remains
`NaN` and is marked in `failure_mask`. The bundle builder never silently
fills a failed node.

For the reference production run, the union masks contained:

```text
column table   8 failed grid nodes
Jeans table   23 failed grid nodes
```

The failure manifest records every failed coordinate. These counts are a
reference, not a reason to overwrite a different result: a reproducer should
inspect the generated JSON and investigate if the failure pattern changes.

The table files also store:

- Cloudy version and radiation-field description;
- abundance, CMB, cosmic-ray, molecular, and charge-transfer metadata;
- portable parameter-template name and SHA-256 hash;
- raw logarithmic emissivity, linear emissivity, zero mask, and failure mask.

## 8. Simulation sampling is a separate validation step

Generating the tables does not require a QUOKKA snapshot or a DESPOTIC table.
Those inputs are needed only to determine whether a particular simulation
interpolation stencil touches a failed Cloudy node and to generate spectra.

For the reference `plt0655228` snapshot, all 134,217,728 cells were inside both
table domains and no interpolation stencil touched a failed node. The lookup
temperature policy used for that downstream analysis was:

```text
split by T_QUOKKA at 3000 K
below 3000 K: use T_DESPOTIC for Cloudy lookup and thermal width
at/above 3000 K: use T_QUOKKA
```

This downstream result validates the tables for that snapshot; it is not part
of the table-generation command above.
