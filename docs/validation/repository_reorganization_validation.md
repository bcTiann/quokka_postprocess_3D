# Repository reorganization: 2026-10-06

The reorganization goal is complete. The baseline is the dirty working tree immediately before this cleanup, not
Git HEAD. Source, tests, documentation, completed numerical products and both
figure sets were frozen before any files were moved.

## Scope

- Configuration files moved to `configs/`; current instructions moved to `docs/`.
- Thirteen active tools moved to `tools/cloudy/`, `tools/despotic/` and
  `tools/figures/`, with their actual callers and default paths updated.
- Seven package modules and eleven test files received more specific names.
- Shared DESPOTIC field names now live in `despotic/table_data.py`; display
  labels live in `figures/line_labels.py`.
- Eighty-seven unused source/test/parameter files were packed into a verified
  archive, including 34 project-specific Cloudy parameter experiments.
- Historical notes and older archives were packed separately. Large old local
  data and results were moved to `archive/local/`, not deleted.
- The Cloudy table is now `inputs/tables/cloudy/emission.npz`, with identical
  bytes. The five molecular-data files moved to `vendor/molecular_data/LAMDA/`
  without changing their contents.
- Current processing products are in `output/plt0655228/processed/`; previous
  products remain recoverable in the local archive.

No manuscript was edited. No table was rebuilt. Current physics and numerical
calculation rules were preserved. No commit or push was made.

## Completed checks

| Check | Result |
|---|---|
| Current unittest suite | 224 passed |
| Standalone function tests | 30 passed |
| Test inventory | 28 tests belonged exclusively to retired features; 28 existing cases changed class names; one measured-domain requirement test added; no unexplained loss |
| Current module imports | 107 project imports resolve across 49 package files and 13 tools |
| Current local documentation links | All resolve |
| Retired-file archive | All 87 file hashes verified after reading them back |
| Historical document archive | All 17 file members read successfully |
| Real cells at x=0,128,255 | 1,572,864 cells; 83 input/emission/temperature/product arrays exactly equal, including NaNs |
| Full snapshot processing | 134,217,728 cells; completed in 558.20 seconds |
| Full numerical products | All 53 saved arrays exactly equal |
| Scientific report | Equal except timing and the verified Cloudy path rename |
| Retained dust/radiation paper tools | Three PNGs and 18 saved arrays exactly equal; three PDFs equal except generation timestamps |
| Preserved radiation exports | All 14 `.inc` files exactly equal to their originals |
| Full emission figure comparison | All 78 PNGs have identical pixels; all 78 PDFs have identical bytes after normalizing only generation timestamps |

The known DESPOTIC checkpoint consumer-interruption race was excluded from
both unittest suites. It is not counted as a pass and was not repaired in
this cleanup. Completed emission processing does not use that table-building
checkpoint path.

Table building now requires `--snapshot-domain`: the existing measured
35 × 35 × 53 grid calculation is retained; the unused fixed-axis fallback and
old extension tool were retired. Source-file identity is still checked when
resuming a table build. A historical checkpoint may require its original
source; it is not silently accepted under renamed current source files.

## Evidence

Local evidence is in `output/repo_reorganization_20261006/`:

- `baseline/`, `pre_reorganization_source.tar.gz`: frozen source and products.
- `current_test_summary.json`, `function_tests_summary.json`,
  `test_comparison.json`: test results and retirement/rename accounting.
- `real_cell_comparison.json`: exact real-cell and accumulated-product checks.
- `complete_process.log`, `complete_plot.log`: complete command logs.
- `complete_run_comparison.json`: completed numerical and rendered comparisons.
- `layout_audit.json`, `archive_verification.json`,
  `paper_assets_verification.json`: paths, imports, archive and paper-asset checks.
- `execution_configs/`: exact copies of the validation configuration files;
  their relative paths assume the original `configs/` location.

The portable retirement/move inventory is [archive/manifest.json](../../archive/manifest.json).
Current usage is in the [root README](../../README.md), with the
[directory guide](../repository_layout.md) and
[code walkthrough](../code_walkthrough.md).
