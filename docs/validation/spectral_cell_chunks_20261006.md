# Spectral cell chunk benchmark — 2026-10-06

The default Gaussian-integration cell chunk is **8,192 emitting cells**.
This is a measured performance setting for the current Mac and snapshot,
not a physical limit. Every emitting cell is still integrated.

## Fixed inputs and settings

- Snapshot: `inputs/snapshots/plt0655228`, 134,217,728 cells.
- Tables: the current fixed DESPOTIC and Cloudy emission tables.
- Slabs: eight x layers; table-query batches: 1,000,000 cells.
- Batch workers: two; spectral workers per batch: three.
- Velocity channels: 400 over −200 to 200 km/s.
- Channel chunk: 150; all interpolation, emissivity, dust and thermal rules unchanged.

Candidates ran sequentially in fresh processes, so each peak RSS is independent.
Screening times below include batch calculation, product allocation and ordered
merging, but exclude snapshot reading, loading tables and saving products.

## Screening: the same real x = 0:8 slab

| Emitting cells per spectral chunk | Compute and merge (s) | Peak process RSS (GiB) |
|---:|---:|---:|
| 4,096 | 15.015 | 3.651 |
| **8,192** | **14.299** | 4.072 |
| 16,384 | 14.912 | 3.821 |
| 32,768 | 15.262 | 4.574 |
| 65,536 | 15.571 | 4.787 |
| 131,072 | 18.108 | 6.734 |

The three fastest candidates were repeated on x = 0:8, 128:136 and 248:256.
Their combined compute/merge times were 47.720 s for 4,096, **45.001 s for
8,192**, and 46.600 s for 16,384. Peak RSS was respectively 3.964, **3.773**
and 3.992 GiB. RSS includes the whole process, not just Gaussian arrays.

## Complete-process comparison

Both full runs include loading inputs, reading all slabs, calculation, merging,
conservation checks and saving products. The selected run used the actual
production default without a benchmark override.

| Spectral cell chunk | Complete elapsed time (s) | Peak RSS (GiB) |
|---:|---:|---:|
| 16,384 | 567.275 | 4.234 |
| **8,192** | **561.774** | **4.115** |

The observed full-run time reduction was **0.97%**, and peak RSS fell **2.79%**.
There was one full run per configuration; these figures describe this comparison,
not a precise or universal speedup. The full-process gain is small despite the
consistent advantage in the shorter compute comparisons.

## Numerical verification

- All 53 saved numerical/label fields were compared; 45 were exactly equal.
- Every image and gas-phase field was exactly equal.
- Per-line input luminosity, complete luminosity, complete centroid, complete
  sigma, complete raw second moment and cell counts were exactly equal.
- Channel spectra differed by at most 2.98e-14 relative to nonzero channel values,
  or 8.56e-15 as a fraction of the corresponding profile peak.
- Other differences were floating-point summation roundoff. Window-outside light
  is a subtraction residual; its difference was measured against input luminosity.
- Both full runs passed the existing luminosity checks. All 53 related unit tests passed.

The shared default is defined once as `DEFAULT_SPECTRAL_CELL_CHUNK` in
`src/quokka2s/products/integrated_spectra.py`. `EmissionProducts` uses this default.
Explicit small chunks in scientific tests are retained.

Measurements, full products, per-field comparisons and the benchmark script are
saved under `output/spectral_chunk_audit_20261006/`; `summary.json` contains the
combined results. The existing primary processed products were preserved.
