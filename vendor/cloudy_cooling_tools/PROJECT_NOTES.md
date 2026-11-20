# Cloudy cooling tools snapshot

This directory is a source-only snapshot used by the Cloudy table workflow in
this repository.  It was copied from `brittonsmith/cloudy_cooling_tools` at
commit `3e842e5d03de7fb3e9696b1c69d8b37cef1d018e`.

Local changes included in this snapshot:

- `CIAOLoop_lines` adds line-emissivity map output compatible with Cloudy 17.
- Its Jeans density conversion uses `coolingMapHydrogenMassFraction`
  (default `0.7157683773530885`) to match QUOKKA's rho-to-nH conversion.
  This changes no elemental abundances. An explicit value of `0.76`
  reproduces the historical conversion.
- `scripts/subtract_cooling_lite.pl` converts Cloudy 17 component fractions
  back to physical heating/cooling rates using the total rate.

The [Cloudy table builder](../../tools/cloudy/build_cloudy_emission_table.py)
generates its eight-line parameter files at runtime. The source reference
files, including the original `CIAOLoop` and Grackle examples, are available
in this repository at commit `3f8a63b1d`.

The main project commands are in the [repository-root README](../../README.md);
table-building commands and paths are in the [usage guide](../../docs/usage.md#build-tables).

Generated Cloudy outputs, UVB data files, logs, nested Git metadata, simulation
snapshots, and pipeline intermediates are intentionally excluded from Git.
