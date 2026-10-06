# Process/plot simplification: 2026-10-06

The goal is complete. Independent processing and plotting commands use their
own YAML settings; the shared dispatcher and old API aliases are removed.
`IntegratedSpectra` directly owns both dust states' profiles and full moments,
with one production `add_batch()` path. Images retain native resolution.

Start reading at [process_snapshot.main](../../src/quokka2s/process_snapshot.py),
then use the [code guide](../code_walkthrough.md) to follow each semantic stage.
The physical rules, table interpolation, per-line missingness, 400 channels,
dust geometry, full sigma, gas phases and bounded concurrency remain unchanged.

## Verification

| Check | Result |
|---|---|
| Frozen working-tree suite | 249 passed |
| Current suite | 251 passed; two new entry-point import-isolation tests |
| Gas-phase plot function tests | 3 passed |
| Identical real-cell inputs | 1,572,864 cells; 83 result fields exactly equal, including NaNs |
| Random spectral streaming scenarios | 184 output comparisons exactly equal |
| Complete processing | 134,217,728 cells; 577 seconds |
| Three saved NPZ products | 53 fields exactly equal |
| Scientific JSON report | Exactly equal except completion/elapsed time |
| Regenerated PNGs | 78 pixel-identical |
| Regenerated PDFs | 78 byte-identical after normalizing only creation/modification dates |

The existing DESPOTIC interruption/checkpoint race test was excluded from both
suite runs, retained in the repository, and not reported as passing. That
checkpoint implementation is unchanged. Three tests were renamed to exercise
the production accumulator and independent Gaussian kernel instead of removed APIs.

Evidence and the frozen source/products/figures are in
`archive/local/output/pipeline_simplification_20261006/`. See `test_comparison.json`,
`cell_emission_parity.json`, `real_product_parity.json`,
`complete_run_comparison.json`, `process.log` and `plot.log`.

## Commands and results

```bash
python -m quokka2s.process_snapshot --config configs/emission_process.yaml
python -m quokka2s.plot_emission_results --config configs/emission_plot.yaml
```

The supplied configs now select
`archive/local/output/plt0655228/processed_simplified_20261006/`. A future process run needs
a new output directory; the completed one is preserved. Plot can be rerun from
the existing products. The two figure directories remain
`archive/local/output/plt0655228/figures/` and `archive/local/output/plt0655228/figures_titled/`.

No manuscript edits, commits or pushes were made in this stage.
