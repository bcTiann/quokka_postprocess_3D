# Cold+dense DESPOTIC validation — the *proportional* (representative) sample

This document explains, step by step, exactly how the **proportional-sample**
panel (the **bottom row** of
`output/despotic_validation/plots/despotic_validation_cold_dense_d1_sharedcbar_combined.png`)
is produced.

- Compute script: `scripts/validate_despotic_cold_dense.py`
- Combined-figure script: `scripts/replot_validation_shared_cbar.py`
- Run with the env python directly (never `conda run`):
  `/opt/homebrew/Caskroom/miniconda/base/envs/yt-env/bin/python`

---

## 0. What question this figure answers

The pipeline's `temperature_despotic` is a **table interpolation** over
`(n_H, N_H, dVdr)` of DESPOTIC's equilibrium temperature `tg_final`. Two things
could make it untrustworthy: (a) bad interpolation, or (b) the surprising result
that in cold+dense gas DESPOTIC comes out *hotter* than QUOKKA.

The validation re-runs the **real** DESPOTIC solver on a set of real simulation
cells and, per cell, compares three temperatures:

```
T_QUOKKA   vs   T_DSP(table interp)   vs   T_DSP(real setChemEq)
```

The **proportional** sample answers specifically: *for a **typical** cold+dense
cell — i.e. where most of the gas actually sits — how well does the table match
real DESPOTIC, and is DESPOTIC really hotter than QUOKKA?*

This is the complement of the `even` (stratified) sample, which instead spreads
cells evenly across the whole `(ρ, T_QK)` plane to probe the rare corners too.

---

## 1. Load the simulation cubes  (`_load_cubes`)

1. `yt.load(cfg.YT_DATASET_PATH)` — the QUOKKA snapshot `plt0655228`.
2. `physics_fields.add_all_fields(ds)` registers the derived fields.
3. A `YTDataProvider` is built with the **disk cache key** for the current
   `(snapshot, table, downsample, L_ext)`. This lets the expensive derived
   fields be **read from cache** instead of recomputed.
4. Pull five fields on the full covering grid and flatten each to a 1-D array in
   cgs units:

   | variable | field | meaning |
   |---|---|---|
   | `rho`    | `('gas','density')`            | mass density [g cm⁻³] |
   | `n_H`    | `('gas','number_density_H')`   | H-nucleus number density [cm⁻³] |
   | `T_qk`   | `('gas','temperature_quokka')` | QUOKKA's own temperature [K] |
   | `colden` | `('gas','column_density_H')`   | 6-direction harmonic-mean N_H (incl. L_ext) [cm⁻²] |
   | `dvdr`   | `('gas','dVdr_lvg')`           | LVG velocity gradient [s⁻¹] |

   `colden` is the **same column density the table is fed** (the L_ext + 6-dir
   harmonic mean), so each L_ext run loads a different `colden` cube; this is why
   the validation is repeated per L_ext.

> The L_ext value comes from `cfg.COLUMN_EXTENSION_LATERAL_KPC`, optionally
> overridden by the `VALIDATE_LEXT` environment variable so the 0/9/99 runs need
> no config edits.

---

## 2. The cold+dense mask

Compute `log_rho = log10(rho)` and `log_Tqk = log10(T_qk)`, then keep cells that
are cold **and** dense **and** have valid table inputs:

```python
mask = (log_rho > LOG_RHO_MIN) & (log_Tqk < LOG_TQK_MAX)
       & finite(log_rho, log_Tqk) & (n_H > 0) & (colden > 0) & (dvdr > 0)
```

with `LOG_RHO_MIN = -23.0` and `LOG_TQK_MAX = 4.0` (i.e. ρ > 10⁻²³ g cm⁻³ and
T_QUOKKA < 10⁴ K). For L_ext = 0 this leaves **118 475** cells — the population
we will sample from.

---

## 3. The proportional sample  (`_proportional_sample`) — **the key step**

```python
def _proportional_sample(mask, rng):
    idx = np.flatnonzero(mask)              # flat indices of all masked cells
    k   = min(N_TARGET, idx.size)           # N_TARGET = 400
    return rng.choice(idx, size=k, replace=False)
```

That is the entire method: **draw 400 cells uniformly at random from the masked
set, without replacement, and without any binning.**

### Why "uniform over cells" = "proportional to population" = representative

Partition the masked cells into regions (e.g. `(ρ, T_QK)` bins) with cell counts
`N_1, N_2, …`. A uniform draw over individual cells picks each cell with equal
probability `1/N_tot`. So the expected number of sampled cells landing in region
`b` is

```
E[n_b] = N_TARGET · (N_b / N_tot)     ⟹     n_b ∝ N_b
```

The sample therefore reproduces the **true frequency distribution** of the
cold+dense gas: the densest, most-populated regions get the most points, the
sparse corners get few. It is an unbiased miniature of the population.

### Contrast with the `even` (stratified) sample

`_stratified_sample` first buckets cells into 0.5-dex `(log ρ, log T_QK)` bins
and takes an **equal quota per bin** (≈ `N_TARGET / B`). That deliberately
**flattens** the distribution — rare bins are over-represented relative to how
many cells they actually contain. Good for "is the table OK everywhere,
including the extremes?"; bad for "what does a typical cell look like?".

Empirically the two samples barely overlap (their 400+400 union is ~795 unique
cells), confirming they probe different cells.

---

## 4. Table-interpolated temperature  (`T_DSP_table`)

For the 400 sampled cells at once:

```python
lookup = phys.ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)   # 3D (nH,NH,dVdr) table
T_tab  = lookup.temperature(n_H[sample], colden[sample], dvdr[sample])
```

`lookup.temperature` is a `RegularGridInterpolator` (linear in log-space) over
the table's `tg_final` grid — exactly what the pipeline's `temperature_despotic`
field uses. This is the quantity whose accuracy we are checking.

---

## 5. Real DESPOTIC temperature  (`T_DSP_real`)

For each sampled cell we run the **actual** solver with the same chemistry
network and dust/radiation parameters that *built* the table:

```python
out = calculate_single_despotic_point(
        nH, colDen, [dVdr], chem_network=GOW, log_failures=False)
T_real = out[6]      # final_Tg from setChemEq(evolveTemp="iterateDust")
```

This solves chemistry **and** thermal balance from scratch (no interpolation), so
`T_real` is the ground truth the table is approximating.

**Optimization (union-solve):** the script runs both the `even` and `prop`
samples in one invocation. It takes the **union** of their cell indices,
de-duplicates, and solves real DESPOTIC **once per unique cell** (~795 instead of
800), then splits the results back to each method. Cells shared by both methods
get an identical `T_real`. At ~1.3 s/cell this is ~17 min per L_ext.

---

## 6. Per-cell numbers and the CSV

For the sampled cells the script writes
`despotic_validation_cold_dense_d1_Lext{L}kpc_prop.csv` with columns:

```
flat_idx, log_rho, n_H, colDen, dVdr,
T_QK, T_DSP_table, T_DSP_real,
dex_table_over_real = log10(T_table / T_real),   # table interpolation error
dex_real_over_QK    = log10(T_real / T_QK),       # DESPOTIC vs QUOKKA
failed
```

The combined-figure script reads these CSVs (no DESPOTIC re-solve).

---

## 7. The scatter panel

Each panel plots, for the `ok` cells (not failed, all three T > 0):

- **x-axis:** `log10 T_QUOKKA`  **y-axis:** `log10 T_DESPOTIC`
- **filled circle** = `T_DSP_real`, **coloured by `log10 N_H`** (viridis).
- **hollow black ring** = `T_DSP_table`, drawn on top of the same x, so the two
  estimates for one cell sit on the same vertical line — any gap between ring and
  dot is the **interpolation error**.
- **dashed 1:1 line** `T_DESPOTIC = T_QUOKKA`. Points **above** the line ⇒
  DESPOTIC hotter than QUOKKA.
- **text box** reports the table-vs-real error for that panel: `median` and
  `max` of `|log10(T_table / T_real)|`.

In the **combined** figure (`replot_validation_shared_cbar.py`):

- **rows = sampling method** — top `even`, **bottom `prop`**.
- **columns = L_ext** — 0 / 9 / 99 kpc.
- the `log10 N_H` colour scale **and** the x/y axis ranges are **pooled across
  all 6 panels**, so colours and positions are directly comparable.

---

## 8. How to reproduce

```bash
PY=/opt/homebrew/Caskroom/miniconda/base/envs/yt-env/bin/python
cd /Users/baochen/quokka_postprocessing

# 1) solve real DESPOTIC for both samples at each L_ext (writes *_even.csv / *_prop.csv)
for L in 0 9 99; do
  VALIDATE_LEXT=$L $PY scripts/validate_despotic_cold_dense.py
done

# 2) draw the 2-row (even/prop) x 3-col (L_ext) combined figure from the CSVs
$PY scripts/replot_validation_shared_cbar.py
```

Knobs in `validate_despotic_cold_dense.py`: `N_TARGET` (400), `LOG_RHO_MIN`
(−23), `LOG_TQK_MAX` (4), `RNG_SEED` (42). Config must be at
`DOWNSAMPLE_FACTOR = 1`; the per-L_ext `column_density_H` field cache must exist
(it does after a normal d1 pipeline / multi_field run at that L_ext) so the cube
load is a cache hit rather than a 90-s recompute.

---

## 9. How to read the result (what the prop row shows)

- The prop points form a **tight clump at `log10 T_QK ≈ 1.2–1.5`** (T_QK ≈
  15–30 K): that is where the overwhelming majority of cold+dense cells live.
- **Interpolation error is negligible** — median 0.000–0.002 dex (max ≤0.13 dex),
  so the table reproduces real DESPOTIC for typical cells essentially exactly.
- **`T_real > T_QK` in 100 % of prop cells** at all three L_ext. So for the
  representative cold+dense cell, DESPOTIC's equilibrium temperature is **always
  hotter** than QUOKKA's — the surprising result is real physics
  (ISRF photoelectric heating), not an interpolation artifact.

(The `even` row dips to ~93–97 % only because its deliberately-flattened sample
includes rare corner cells that occasionally flip.)
