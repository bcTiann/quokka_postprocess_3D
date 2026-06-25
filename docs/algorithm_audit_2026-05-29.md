# Algorithm audit + bug fix log — 2026-05-29

End-to-end audit of the `quokka2s` DESPOTIC table builder and downstream pipeline,
triggered by the user's question "are (nH, NH, dVdr) really three independent
inputs to DESPOTIC?".  The audit uncovered **one cosmetic mis-naming** (the
two-stage broadcast), **one catastrophic build-time bug** (missing emitters in
`setTempEq`), and a handful of smaller cache-invalidation / pipeline issues.
This document lists everything that changed and why.

## tl;dr

- **Catastrophic**: every 3D GOW LVG table built before today had a
  missing-emitter bug that made CNM cells **1–2.5 dex too hot** (e.g.
  Tg=9379 K at (nH=10, NH=10²⁰) instead of the correct 74 K).
  Cause: `solver.py` never called `cell.addEmitter` before `setChemEq`, so
  `setTempEq` inside `iterateDust` saw `cell.emitters = []` and excluded
  C+/CO/HCO+/O line cooling from the thermal balance.
- The DESPOTIC `setChemEq → chemEvol → applyAbundances(addEmitters=True)`
  auto-add path was supposed to populate emitters at solve time but
  empirically does *not* — `cell.emitters` stays empty after `setChemEq`
  returns.  Fix: explicitly `cell.addEmitter(sp, 0.0)` for each desired
  species before `setChemEq`.
- The "two-stage dVdr broadcast" optimization in the builder was *exactly*
  correct under the buggy solver (because line cooling was zero, dVdr had no
  effect through dEdt).  With the bug fixed, line cooling is back and dVdr
  genuinely matters — so the broadcast had to come out anyway.
- All three 3D tables (GOW, NL99, NL99_GC) rebuilt with the fix.  The 4D
  builder code is updated but no 4D table rebuild is planned this round.

---

## Bug catalogue

### 1. (critical) Missing emitters in `setChemEq`

**File**: `src/quokka2s/tables/solver.py`
**Severity**: catastrophic — wrong Tg by 1–2.5 dex in CNM regime.

**Symptom**: At (nH=10 cm⁻³, NH=10²⁰ cm⁻², dVdr=10⁻¹⁵ s⁻¹), the old
solver returned `Tg = 9379 K` (WNM-like).  The correct CNM equilibrium is
~74 K (C+ 158 μm cooling dominant).  127× temperature error.

**Verification table** (10 representative cells, GOW network):

| cell | Tg (buggy) | Tg (fixed) | error factor | log₁₀ ratio |
|---|---|---|---|---|
| WIM-like (1e-4, 1e15) | 43508 K | 43497 K | 1.00× | 0.000 |
| WNM (0.09, 1e18) | 10884 K | 10805 K | 1.01× | +0.003 |
| CNM thin (1.3, 1e20) | 9802 K | 8131 K | 1.21× | +0.081 |
| **CNM (10, 1e20)** | **9379 K** | **74 K** | **127×** | **+2.102** |
| **CNM dense (76, 1e21)** | **7673 K** | **27 K** | **281×** | **+2.449** |
| **cold molec (1e3, 8e21)** | **122 K** | **5.7 K** | **21×** | **+1.329** |
| molec cloud (9e3, 9e22) | 8.4 K | 5.5 K | 1.52× | +0.181 |
| dense core (1e5, 9e22) | 3.7 K | 3.7 K | 1.02× | +0.007 |
| very dense (1e6, 1e24) | 3.4 K | 3.4 K | 1.00× | 0.000 |
| **thin high NH (1.3, 9e22)** | **6553 K** | **19 K** | **337×** | **+2.528** |

**Root cause**: DESPOTIC's `cloud.dEdt()` computes line cooling by iterating
over `self.emitters` (cloud.py:979).  `setTempEq` inside `iterateDust` uses
this for the gas thermal balance.  If `self.emitters` is empty, line cooling
is zero — the only cooling channels are LyA (Boltzmann-suppressed below 10⁴ K),
dust thermal, and gas-dust coupling.  At CNM densities, dust coupling is
negligible (low n) and LyA is suppressed → no cooling → Tg saturates near the
PE+CR-vs-LyA equilibrium around 10⁴ K.

The GOW chemistry network's `applyAbundances(addEmitters=True)` (GOW.py:1137)
*does* try to add emitters via `cloud.addEmitter('co', abd['CO'])` etc., but
only via `try: ... except despoticError: print('Warning: ...')`.  In our
solver context this auto-add empirically does not persist: `cell.emitters = []`
after `setChemEq` returns — even with `DESPOTIC_HOME` correctly pointing at
the bundled LAMDA data directory.  Whether this is silent LAMDA failure, a
cell-reference mismatch, or a DESPOTIC version-specific behaviour, the
empirical answer is the same: the auto-add path cannot be relied upon.

**Fix** (solver.py ~line 220):

```python
# CRITICAL: pre-add line-cooling emitters so setTempEq inside iterateDust
# actually sees CO/C+/C/HCO+/O contributions.  Abundance 0.0 is a placeholder;
# applyAbundances updates it to the network's equilibrium value during the
# chemistry solve.
for sp in species_order:
    cell.addEmitter(sp, 0.0)
```

Added BEFORE `cell.comp.computeDerived(cell.nH)` and BEFORE `cell.setChemEq`.

**Validation hit**: the `representative_cold_dense.py` validation script
called the SAME buggy solver to compare "table vs real DESPOTIC" — both
sides used the same broken code path, so they agreed to 0.001–0.003 dex.
Future runs of that validation against the fixed table+solver should also
agree, but at a different (correct) absolute Tg.

### 2. (deprecated) Two-stage dVdr broadcast in builder

**File**: `src/quokka2s/tables/{solver,builder}.py`
**Severity**: cosmetic under bug #1 (zero practical effect); becomes wrong
once bug #1 is fixed.

**Symptom**: The builder called `setChemEq` ONCE per (nH, colDen) pair at
`canonical_dvdr = dvdr_grid[17]`, then broadcast Tg / μ / cv / Eint /
abundances across all 35 dVdr storage indices.  Only `lineLum` was actually
recomputed per dVdr.

**Empirical check (under buggy solver)**: 10 representative cells, 35 dVdr
points per cell, full `setChemEq` at each → `max(|log₁₀(full / broadcast)|)
= 0.0000 dex` for every cell.  The broadcast was exact (because line cooling
was zero, dVdr had no path into dEdt).

**Why removed anyway**: with bug #1 fixed, line cooling re-enters dEdt and
the LVG escape probability β(τ) ∝ 1/(1+τ_Sobolev) with τ ∝ 1/dVdr genuinely
affects thermal balance.  The broadcast would no longer be exact.

**Fix**: `calculate_single_despotic_point` now takes a single `dvdr_val`
(not a grid).  The builder loops over (nH, colDen, dVdr) and calls the
solver 35³ times instead of 35²; each call is an independent setChemEq.

Cost: ~35× build time at fixed parallelism.  Real wall-clock: ~80 min per
table at 10-core parallelism, vs. ~2.5 min for the broadcast build.

### 3. (Hα silent zero) `val[T > 1e5 K] = 0.0` legacy gating

**File**: `src/quokka2s/pipeline/prep/physics_fields.py:384`
**Severity**: silent — wrong Hα flux in HIM gas, irrespective of HIGH_T_4D_BLEND.

`_make_number_density_field` zeroed **all** species number densities
(including `e-` and `H+`) above 10⁵ K with a single hard line.  The
`_Halpha_luminosity` field then computed `α_B(T) · n_e · n_H+ = 0` for every
hot ionized cell, so no Hα emission was ever recorded from HIM.  The comment
called this "cold-path legacy gating (kept for backward-compat)".

**Fix**: remove the line.  Hot-gas behaviour now flows through:
- `HIGH_T_4D_BLEND = True` → 4D table lookup at `T_gamma_mu` (real chemistry)
- `HIGH_T_4D_BLEND = False` → saturated `tg_final` interpolation (~5×10⁴ K
  equilibrium; H mostly ionized; e-, H+ ≈ nH)

### 4. (cache stale) `compute_cache_key` missing several invalidation inputs

**File**: `src/quokka2s/pipeline/cache.py:77`
**Severity**: silent — toggling config didn't always invalidate cached
intermediates, so plots could show pre-toggle values.

**Old key components**:
- snapshot path + mtime
- 3D table path + mtime
- downsample_factor
- L_ext_kpc
- COLUMN_DENSITY_MEAN
- schema_version

**Added**:
- `HIGH_T_4D_BLEND` (bool toggle)
- `T_QK_HIGH_K` (high-T branch threshold)
- `T_CUTOFF` (per-species lumPerH cutoff dict, serialised)
- `T_CUTOFF_DEFAULT`
- `DESPOTIC_TABLE_4D_PATH` + mtime
- `DVDR_FLOOR` (from physics_fields)

Schema version bumped 3 → 4 to force re-evaluate every cache file.

### 5. `LambdaLine.{species}` per-emitter cooling rates now stored

**File**: `src/quokka2s/tables/{solver,builder}.py`

The old `cell.dEdt()` output had `LambdaLine = {}` because emitters were
empty (bug #1).  With the fix, dEdt returns
`LambdaLine = {'CO': float, 'C+': float, ...}`.  The solver's new
`_flatten_energy_terms` function recursively flattens this into the energy
namespace as `LambdaLine.CO`, `LambdaLine.C+`, etc.  The builder writes one
per-cell array per emitter, so the npz now carries:

```
energy::LambdaLine.CO       (35, 35, 35) array
energy::LambdaLine.C+       ...
energy::LambdaLine.C        ...
energy::LambdaLine.HCO+     ...
energy::LambdaLine.O        ...
```

This lets downstream analysis answer "which species dominates cooling in
this regime?" without re-solving DESPOTIC.

### 6. `_temperature_gamma_mu` should pass the cell's real dVdr to the 4D lookup

**Files**: `src/quokka2s/tables/lookup.py:298`,
          `src/quokka2s/pipeline/prep/physics_fields.py:_temperature_gamma_mu`

Pre-fix, `TableLookup4D.temperature_gamma_mu(nH, NH, e_specific)` defaulted
`dVdr_cgs` to the table's median dVdr value, with the justification "μ/cv
are dVdr-independent so any slice works".  That was true under the bug-#1
solver (and broadcast builder) but is no longer true: the new 4D table —
once rebuilt — has genuinely dVdr-dependent μ/cv.  The pipeline now passes
the cell's actual `dVdr_lvg` field through.

(The current 4D table on disk is still the old broadcast build; it'll be
overwritten when build_table_4d is re-run.  Until then, μ/cv along the dVdr
axis is constant in that table, so passing real dVdr is harmless but doesn't
exploit any new information.)

### 7. Debug `print` left in `_Halpha_luminosity`

**File**: `src/quokka2s/pipeline/prep/physics_fields.py:415-416`

Two `print(...)` statements that fired on every cell-batch evaluation of
the field.  Removed.

### 8. (housekeeping) `AttemptRecord` extended

**File**: `src/quokka2s/tables/models.py`

Added `dvdr_idx` and `dvdr` fields (both default `None` for backward
compat) so per-dVdr failures are traceable.

---

## What didn't change (but was audited)

- **`_column_density_H`** — 6-direction sum / harmonic / arithmetic / max /
  min combinations look correct; `along_sight_cumulation` semantics verified
  (np.flip+cumsum+flip for + direction, plain cumsum for −).  Lateral
  extension `L_ext × <n_H>(z)` added only to ±x, ±y rays; ±z gets no
  extension (rationale: stratified box already spans the disk).
- **Solver hardcoded `sigmaNT = 2.0 km/s`, `chi = 1.0`, `ionRate =
  2.0e-17`, `Zd = 1.0`, etc.** — the table only represents MW-like dust,
  ISRF, and CR rate.  Cells in the simulation with very different sigmaNT
  (e.g. shock fronts) will get wrong line widths, but those parameters
  cannot become extra table axes without exploding the grid size.  Logged
  here as a known limitation, not a bug.
- **`DVDR_FLOOR = 1e-18` vs table dVdr_min = 1e-19** — pipeline cells with
  ∇·v/3 < 10⁻¹⁸ are pinned at 10⁻¹⁸ before lookup.  Bottom decade of the
  table is therefore unreachable but harmless (linear-interp can still
  evaluate at the floor).

---

## Pre-fix vs post-fix snapshot dirs

- `output_tables_3D_GOW_LVG_v1/` — original build, pre-`computeDerived`,
  pre-emitter-fix.  KEEP for archaeology.
- `output_tables_3D_GOW_LVG/` (the one called "v2" earlier today) — has
  `computeDerived` fix but still missing-emitter bug.  WRONG in CNM regime.
- `output_tables_3D_GOW_LVG_v3/` — `computeDerived` + emitter fix + true
  3-input solve.  This is the new canonical.

(Names with explicit `_v1/v2/v3` are temporary while the rebuild is in
progress.  After validation, v3 will be renamed to the canonical name and
v2 archived.)

## Re-validation TODO (after rebuild)

1. Re-run `representative_cold_dense.py` against the rebuilt table to
   confirm table ≈ real-time real DESPOTIC (both now using the fixed
   solver).  Expected: still 0.001 dex agreement, but at much lower
   absolute Tg in CNM.
2. Re-run `multi_field_slices` to see the corrected Tg map.  CNM
   regions should drop from ~10⁴ K to ~10²–10³ K.
3. Re-do the L_ext 0-vs-9-vs-15 kpc comparison — the magnitude of the
   "DSP hotter than QK" effect in cold+dense gas may shrink substantially
   (some of it was the emitter bug, not real physics).


---

## Addendum 2026-05-30: bug #9 — sphere thermal balance under LVG line emission

Discovered while validating that v4 (post-emitter-fix) Tg was *still*
dVdr-invariant cell-by-cell.  Root cause: `setChemEq`'s internal
`setTempEq()` defaults `escapeProbGeom='sphere'`, and the sphere escape
probability formula (emitter.py:669-678) does NOT read `cell.dVdr`.  So
thermal balance was sphere-derived; the post-solve `lineLum(LVG)` used
LVG — two different geometries, inconsistent.

DESPOTIC code audit confirmed: the only place `cell.dVdr` enters any
physics formula is `emitter.py:684` inside the `escapeProbGeom == "LVG"`
branch.  There is no auto-detection that switches to LVG when `cell.dVdr`
is set; the user must pass `escapeProbGeom='LVG'` explicitly.

**Fix**: in `solver.py`, pass `tempEqParam={'escapeProbGeom': escape_geom}`
to `setChemEq`.  This propagates the caller's geometry all the way to
`em.setLevPopEscapeProb` inside `_gdTempResid → dEdt` so thermal balance
uses the same LVG geometry as the post-solve `lineLum`.

### Verified behaviour (test at nH=10, NH=1e20):

```
dVdr=1e-19  →  Tg = 1670 K   (photons trapped → line cooling suppressed)
dVdr=1e-17  →  Tg =  146 K
dVdr=1e-15  →  Tg =   75 K
dVdr=1e-13  →  Tg =   74 K   (high dVdr plateau — optically thin)
dVdr=1e-12  →  Tg =   74 K
```

23× variation across the dVdr range — the physically expected
photon-trapping behaviour, recoverable now that the geometry is consistent
between thermal balance and reported line luminosity.

## Final table layout (2026-05-30)

| dir | purpose |
|---|---|
| `output_tables_3D_GOW_LVG/` | **CANONICAL** — v4 cleaned (LVG + emitter + true 3D solve, NaN-filled) |
| `output_tables_3D_GOW_LVG_v4/` | v4 raw build (pre-cleanup, has 2 garbage cells + 3086 NaN) |
| `output_tables_3D_GOW_LVG_v3_sphere/` | v3 — emitter fix but still sphere thermal balance (inconsistent) |
| `output_tables_3D_GOW_LVG_v2_buggy_emitter/` | v2 — was canonical until 2026-05-30; missing-emitter bug; CNM 100× too hot |
| `output_tables_3D_GOW_LVG_v1/` | v1 — original, also missing-emitter bug + no `computeDerived` |
| `output_tables_3D_NL99_GC_LVG/` | **CANONICAL** — v2 cleaned (same fixes as GOW v4) |
| `output_tables_3D_NL99_GC_LVG_v2/` | v2 raw build (pre-cleanup, 2485 garbage cells in WIM regime) |
| `output_tables_3D_NL99_GC_LVG_v1_buggy_emitter/` | v1 — missing-emitter bug |

### v2 → v4 GOW table differences (full-table statistics)

```
log10(v2 / v4):
  mean    = +0.65 dex
  median  = +0.02 dex
  p10     = +0.00            (v2 never cooler than v4 — adding emitters can only cool)
  p90     = +2.23 dex        (90th percentile: v2 was 170× too hot)
  max     = +2.59 dex

fraction of cells where |v2 − v4| > 0.5 dex (>3.2×): 36.4%
                                  > 1.0 dex (>10×): 29.8%
                                  > 2.0 dex (>100×): 16.0%
```

16% of the table was wrong by >100× T in v2.  The error was concentrated
in the CNM / cold-molecular regime — exactly the cells most relevant for
CO/C+ analyses.

### NaN-fill post-process

Both new canonical tables underwent identical post-processing:

1. Cells with `Tg > 10⁶ K` (numerical garbage — LVG iterator landing in
   spurious 100 MK attractor for some WIM cells) → set to NaN.
2. NaN cells in `tg_final`, `mu`, `cv`, `Eint`, every `*_abundance`,
   every line-emission field, and every `energy::*` field → filled via
   3D log-space linear interpolation from non-NaN neighbours.  Cells
   outside the convex hull of valid data → filled by nearest-neighbour.
3. `failure_mask` left untouched (so cells that were originally
   problematic are still flagged for analytics, even if Tg is now finite).

Both tables now have 100% finite cells.  GOW failure_mask: 5032
(originally failed, now filled).  NL99_GC failure_mask: 6584.
