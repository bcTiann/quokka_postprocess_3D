# Cloudy cooling tools snapshot

This directory is a source-only snapshot used by the Cloudy table workflow in
this repository.  It was copied from `brittonsmith/cloudy_cooling_tools` at
commit `3e842e5d03de7fb3e9696b1c69d8b37cef1d018e`.

Local changes included in this snapshot:

- `CIAOLoop_lines` adds line-emissivity map output compatible with Cloudy 17.
- Its Jeans density conversion uses `coolingMapHydrogenMassFraction`
  (default `0.7157683773530885`) to match QUOKKA's rho-to-nH conversion.
  This changes no elemental abundances. An explicit value of `0.76`
  reproduces the historical conversion; the upstream `CIAOLoop` is retained
  unchanged as a source reference.
- `scripts/subtract_cooling_lite.pl` converts Cloudy 17 component fractions
  back to physical heating/cooling rates using the total rate.
- Eight-line parameters are generated at runtime by
  `tools/cloudy/build_cloudy_emission_table.py`. Ten upstream-style parameter
  examples are included in `examples/grackle/` as source references.

The upstream README is preserved as upstream documentation. This is a selected
source snapshot, so some data/example paths described there are not included.
Current project commands and paths are documented in the repository-root README.

Generated Cloudy outputs, UVB data files, logs, nested Git metadata, simulation
snapshots, and pipeline intermediates are intentionally excluded from Git.
