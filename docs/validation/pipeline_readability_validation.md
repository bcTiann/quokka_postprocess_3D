# Pipeline readability refactor: verification

The full-profile moment addition and process/plot check cleanup were verified
on 2026-10-06. The current saved axes use `dust_state`, and line moments have
explicit `full` and `window` names. See the dated entry in the
[current plan](../../archive/README.md) and
`archive/local/output/full_line_moments_20261006/`. The older evidence below is preserved
with its original interfaces and file names.

This round implements the [readability plan](../../archive/README.md).
The numerical baseline is the working tree frozen before this round, including
its existing uncommitted changes. It is not Git HEAD.

This records the 2026-10-04 interfaces and their validation. The current
interfaces are `DespoticCellReader.read_fields()`, `CloudyCellReader.read_fields()`
and `CellEmissionCalculator.calculate(cells=...)`; see the
[package reading guide](../code_walkthrough.md). The evidence below remains
the record of that earlier round.

## What changed

- Lookup names now match their content: `DespoticLookup`, `DespoticCellFields`,
  `CloudyLookup`, `CloudyCellFields`, and `query_cloudy_fields()`.
- `BatchEmission` retains two independent masks: `valid_cells` and `cold_cells`.
  Query-stage failures remain local. DESPOTIC coordinate adjustments leave the
  batch as integer counts rather than three additional cell-sized arrays.
- DESPOTIC coordinate preparation and interpolation share one explicit path.
  Cloudy separates physical coordinates, logarithmic coordinates, node
  inspection, interpolation, and restoration to original cell positions.
- Images separate cell luminosities from pixel mapping. Spectra separate
  retained inputs, cell luminosities, Gaussian integration, and accumulated sums.
  Single and paired dust spectra share the same thermal/integration calculation.
- Gas-phase moments, phase-panel values and projection inputs use named objects.
- Plot loads `SavedEmissionProducts` once. Gas/line comparison uses
  `LineSpectraForComparison` and `GasPhaseVelocityProfiles`, without rebuilding
  nested report dictionaries or redundant total profiles.
- English comments/docstrings describe types, shapes, units, upstream sources
  and concrete indices. Public calls use expanded named arguments.

No old-name aliases were added. Existing NPZ/report field names are the saved
data format, independent of internal names: for example, `variant_keys` still
records intrinsic/attenuated order. `raw_luminosity` still controls spectral
area normalization, not dust selection. The commands below record the earlier interface; current commands are in
the root README:

```bash
python -m quokka2s process --config emission_process.yaml
python -m quokka2s plot --config emission_plot.yaml
```

The processing command requires a new output directory; this round's validation
uses its own directories and does not overwrite existing scientific products.

## Numerical and figure checks

All evidence is under `archive/local/output/pipeline_readability_20261004/`.

| Check | Result | Evidence |
|---|---|---|
| Frozen/live low and high lookup stages, including empty/broadcast queries and 3D/4D readers | 552 exact comparisons, including NaNs and masks | `lookup_equivalence.json` |
| Synthetic image/spectrum/phase/projection products | Exact numerical agreement | `product_equivalence.json` and `product_validation/` |
| Real x=0, 128, 255 slices | 1,572,864 cells; 12 canonical fields per slice exactly equal | `real_batches_comparison.json` |
| Full snapshot process | 134,217,728 cells; all 47 fields in the three NPZ files exactly equal | `full_snapshot_comparison.json` |
| Full processing report | Same physics, counts, masses, luminosities, sigma, clipping and checks; only timestamps/duration differ | `before_process/emission_report.json`, `after_process/emission_report.json` |
| Full plot | 78 PNG files byte-for-byte equal; 78 PDF contents equal after removing creation/modification timestamps | `full_plot_comparison.json` |
| Independent rendering from identical input products | Paper/titled figures and temperature-component figure exactly equal | `plot_only_comparison.json` |
| Complete test inventory, including module functions | 360 before/after; 358 pass and the same 2 pre-existing failures; no added/missing test IDs | `complete_suite_comparison.json` |

The full run excludes 19,827 cells in both versions. Their mass fraction is
`6.346420748075392e-7`. No emissivity, dust geometry, temperature branch,
interpolation rule, physical zero, cell-selection rule, Gaussian convention,
native image resolution or velocity-channel count was changed.

`source_changes.json` identifies changes relative to the frozen working tree.
Source hashes are verification artifacts only; they are not runtime dependencies.

## Known issues, separate from this refactor

The complete test suite is **not entirely passing**. Both versions fail:

1. `test_despotic_checkpoint.CheckpointTests.test_parallel_consumer_interruption_stops_workers_and_resumes`:
   workers may write additional checkpoints after the interruption returns.
   The exact extra file list varies; the comparison therefore also records
   changed failure details. The baseline test child wrote all outcomes but
   stalled during process shutdown. Only its verified Loky workers were stopped
   to let the harness finish. DESPOTIC builder/checkpoint code was not changed.
2. `test_gas_phase_velocity.test_invalid_edges_rejected`:
   one invalid-edge case does not raise the expected error in either version.

There is also an independent diagnostic-cache follow-up:
`cloudy.cell_coverage.validated_coverage_lookup()` copies a lookup, changes
its grids/masks, and retains the original cached RGI objects. Coverage counts
inspect masks directly and are unaffected. A four-dimensional diagnostic query
with allowed tiny failed-node weights can still use the original cache. This
already exists in the baseline, is outside the active three-dimensional
process path, and was not changed in this numerical-behavior-preserving round.

No commit or push was performed in this round.

## Follow-up, 2026-10-05

The independent model-depth workflow, including the copied-lookup cache path,
has now been removed. The fixed three-dimensional lookup remains. The invalid
velocity-edge test was also retired under the fixed valid-channel contract;
no extra runtime check was added. The interruption/worker-cancellation issue
is still unresolved. See [the cleanup verification](cloudy_3d_cleanup_validation.md)
for the new test and two-slab product comparisons. The original results above
remain the record of the earlier readability refactor.
