# Cloudy cooling tools snapshot

This directory is a source-only snapshot used by the Cloudy table workflow in
this repository.  It was copied from `brittonsmith/cloudy_cooling_tools` at
commit `3e842e5d03de7fb3e9696b1c69d8b37cef1d018e`.

Local changes included in this snapshot:

- `CIAOLoop_lines` adds line-emissivity map output compatible with Cloudy 17.
- `scripts/subtract_cooling_lite.pl` converts Cloudy 17 component fractions
  back to physical heating/cooling rates using the total rate.
- `examples/grackle/*.par` contains the C II, H-alpha, HM2012, and diagnostic
  parameter files used in this project.

Generated Cloudy outputs, UVB data files, logs, nested Git metadata, simulation
snapshots, and pipeline intermediates are intentionally excluded from Git.
