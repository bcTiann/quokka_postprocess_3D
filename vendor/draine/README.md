# Draine Milky Way dust extinction table

`kext_albedo_WD_MW_3.1_60_D03.all` is the original numerical table from
[B. T. Draine's dust model archive](https://www.astro.princeton.edu/~draine/dust/dustmix.html).
Direct source:
`https://www.astro.princeton.edu/~draine/dust/extcurvs/kext_albedo_WD_MW_3.1_60_D03.all`.
SHA-256: `b56680cc38b85f051f20c4405303e8c480cc9bec714fd5ba722a257a40ae840c`.

The pipeline uses the `C_ext/H` column (cm² per H nucleus), which includes
absorption and scattering out of the line of sight. It interpolates in
log wavelength and log `C_ext/H` at each line's rest wavelength. The file's
maximum wavelength is 1 cm, so H I 21 cm is explicitly left unattenuated.
The dust abundance per H is held spatially fixed; there is no model of dust
destruction or scattered-in light.

The original table cites Weingartner & Draine (2001), Li & Draine (2001),
and Draine (2003) for grain sizes, optical properties, and normalization.
