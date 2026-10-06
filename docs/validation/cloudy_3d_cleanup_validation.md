# Three-dimensional Cloudy lookup cleanup, 2026-10-05

The active Cloudy lookup accepts only shielding NH, hydrogen density and
temperature. Model depth is set during table generation, using CIAOLoop's
Jeans option, mu=1, and the 100 pc cap. No independent length coordinate is
accepted by the lookup.

Removed the unused explicit-depth reader branches and their exclusive
building, coverage and diagnostic callers: three package modules, ten scripts
and ten test modules. The copied-lookup interpolation-cache issue belonged to
this removed workflow. Snapshots, tables and existing numerical results were
not changed or deleted.

The invalid-channel-edge test was removed because it asserted an input check
that the fixed production-channel contract does not require. The production
entry point still constructs 401 increasing edges for 400 channels, using
`np.linspace(-200, 200, 401)`. No extra constructor validation was added.

## Verification

- Ran processing on the same two real x slabs, x=0:16, before and after:
  8,388,608 cells. All 47 fields in the three saved NPZ products match exactly,
  including intrinsic and attenuated images, spectra and gas-phase data.
- Ran the remaining test suite, including module-level test functions:
  241 passed. One previously failing DESPOTIC interruption/worker-cancellation
  test was explicitly skipped; that independent issue remains unresolved.
- Verified the actual three-dimensional table-build, Jeans/X_H/cap and
  packaging behavior with 14 focused tests.
- Compared the immediately precleanup and current lookup: all 428 comparisons
  over 6,193 query points match exactly, including actual table nodes,
  clipping, zero/failure support, scalar/broadcast/empty queries and endpoint
  tolerances. All 32 focused lookup tests pass.
- The plot code and stored table values were not changed. This cleanup used
  two-slab verification; the earlier full-snapshot comparison remains recorded
  separately in `pipeline_readability_validation.md`.

Evidence is under `archive/local/output/cloudy_3d_cleanup_20261005/`:
`removed_explicit_depth_workflow.json`, `two_slab_comparison.json`,
`current_suite.json`, `exclusive_depth_cleanup_validation.json`, and
`lookup_3d_equivalence.json`.
An initial validation harness lacked a multiprocessing main guard; its
unsuccessful harness logs were retained under `harness_without_spawn_guard_*`.
The final guarded harness is `check_current_suite.py`.

No commit or push was performed.
