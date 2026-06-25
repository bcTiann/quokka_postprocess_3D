# Line Emission Calculations in quokka2s

Reference for how Hα, HI 21 cm, CO J=1-0, and [C II] 158 μm volumetric
emissivities are computed per cell in the post-processing pipeline.

All formulas yield ε [erg s⁻¹ cm⁻³]; cell luminosity is L_cell = ε · dV.

## Source code map

All implementations live in
[`src/quokka2s/pipeline/prep/physics_fields.py`](../src/quokka2s/pipeline/prep/physics_fields.py).
Direct jump-to-source links:

| Symbol | Where | What it is |
|---|---|---|
| `T_QK_TWO_REGIME_K = 3000 K` | [physics_fields.py:34](../src/quokka2s/pipeline/prep/physics_fields.py#L34) | cold/hot regime threshold |
| `_build_h_atomic_constants()` | [physics_fields.py:51](../src/quokka2s/pipeline/prep/physics_fields.py#L51) | astropy-derived K_SAHA_PREF, T_H_ION, ν_Hα + NIST-pinned ν₂₁ and A₂₁ |
| `A_C_TOTAL = 1.6e-4` | [physics_fields.py:101](../src/quokka2s/pipeline/prep/physics_fields.py#L101) | total C per H (GOW xC) |
| `_build_cii_lte_atomic_tables()` | [physics_fields.py:104](../src/quokka2s/pipeline/prep/physics_fields.py#L104) | CHIANTI atomic data import |
| `_x_H_ion_saha(T, n_H)` | [physics_fields.py:180](../src/quokka2s/pipeline/prep/physics_fields.py#L180) | Saha → x_H+ helper (shared by Hα, HI, C+) |
| `_temperature_two_regime(field, data)` | [physics_fields.py:237](../src/quokka2s/pipeline/prep/physics_fields.py#L237) | `('gas','temperature_two_regime')` field |
| `_make_luminosity_field('CO')` | [physics_fields.py:547](../src/quokka2s/pipeline/prep/physics_fields.py#L547) | CO single-regime DESPOTIC closure |
| `_make_number_density_field(species, lamda_token)` | [physics_fields.py:582](../src/quokka2s/pipeline/prep/physics_fields.py#L582) | `('gas','HI')`, `('gas','H+')`, `('gas','e-')`, etc. |
| **`_Halpha_luminosity(field, data)`** | [physics_fields.py:614](../src/quokka2s/pipeline/prep/physics_fields.py#L614) | **Hα** — §1 below |
| **`_HI_luminosity(field, data)`** | [physics_fields.py:670](../src/quokka2s/pipeline/prep/physics_fields.py#L670) | **HI 21 cm** — §2 below |
| **CO via `_make_luminosity_field('CO')`** | [physics_fields.py:547](../src/quokka2s/pipeline/prep/physics_fields.py#L547) | **CO** — §3 below |
| **`_Cplus_luminosity_two_regime(field, data)`** | [physics_fields.py:754](../src/quokka2s/pipeline/prep/physics_fields.py#L754) | **C+ 158 μm** — §4 below |
| `add_all_fields(ds)` | [physics_fields.py:932](../src/quokka2s/pipeline/prep/physics_fields.py#L932) | field registration for yt |

Editor tip: in VS Code / Cursor, ⌘-click any link above to jump straight
to the line.  GitHub web renders them too.

---

## 0. Two-regime split (T_QK = 3000 K)

Three of the four species (Hα, HI, C+) use a **two-regime** treatment that
switches at `T_QK_TWO_REGIME_K = 3000 K`:

```
hot  = T_QUOKKA ≥ 3000 K     →  closed-form analytic physics
cold = T_QUOKKA <  3000 K    →  3D DESPOTIC LAMDA table lookup
```

`T_QUOKKA` is the simulation's own temperature.  Why this split:
- DESPOTIC's 3D table saturates at T ≈ 5×10⁴ K (its solver's upper bound).
  For hot ionised gas the table is meaningless.
- For cold cells DESPOTIC is the right answer (its chemistry network
  includes UV / CR photoionisation, dust shielding, molecular cooling).
- 3000 K is well above the WIM thermal floor and below DESPOTIC's
  saturation, so cells either fall safely below it (use table) or above
  it (use analytic).

CO is **single-regime** (3D DESPOTIC LAMDA only) — see §3.

A derived field `('gas', 'temperature_two_regime')` returns
`T_DSP` for cold cells and `T_QK` for hot cells; it's the canonical
"cell temperature" used by all thermal-width and phase-classifier code.

---

## 1. Hα (656.3 nm — recombination line)

### Physics

H+ + e⁻ → H* (any n) → cascade → photon at 656.3 nm if the cascade passes
n=3 → n=2 (case B).  Emissivity:

```
ε_Hα = 0.45 · E_Hα · α_B(T) · n_e · n_H+         [erg/s/cm³]
```

- E_Hα = h·ν_Hα = 3.03×10⁻¹² erg (656.3 nm photon)
- 0.45 = case-B branching fraction (Draine 2011 §14)
- α_B(T) = effective case-B recombination rate, fit:
  ```
  α_B(T) = 2.54×10⁻¹³ · T₄^(−0.8163 − 0.0208·ln T₄)    cm³/s
  T₄ = T / 10⁴ K
  ```
- n_e, n_H+ = electron and proton number densities

### Cold branch (T_QK < 3000 K)

```
n_e   = data[('gas', 'e-')]                  # from DESPOTIC 3D table at (n_H, N_H, dVdr)
n_H+  = data[('gas', 'H+')]                  # same
α_B   = α_B(T_DESPOTIC)                       # T from DESPOTIC's saturated-but-realistic
```

DESPOTIC's chemistry network gives realistic n_e, n_H+ in CNM/WNM/WIM
because it includes UV photoionisation.

### Hot branch (T_QK ≥ 3000 K)

```
x_H+  = _x_H_ion_saha(T_QK, n_H_tot)          # Draine 2011 Eq 3.17 / Sparke 3.17
n_e   = x_H+ · n_H_tot                        # H-only charge neutrality
n_H+  = x_H+ · n_H_tot                        # = n_e
α_B   = α_B(T_QK)                              # real hot-gas T
```

`_x_H_ion_saha` solves the textbook Saha equation:

```
                              ⎛ 2π m_e k_B T ⎞^(3/2)
n_e · n_H+ / n_HI  =          ⎜──────────────⎟         · exp(−I_H / k_B T)
                              ⎝       h²      ⎠
                       ≡  K(T)

x² / (1 − x) = K(T) / n_H_tot ≡ R       (charge neutrality)
x = 2 / (√(1 + 4/R) + 1)                  (numerically stable)
```

- K_SAHA_PREF = 2π m_e k_B / h² ≈ 1.7999×10¹⁰ cm⁻² K⁻¹ — **computed at import from `astropy.constants`**
- T_H_ION    = Ryd·h·c / k_B  ≈ 1.5789×10⁵ K  — **computed at import from `astropy.constants`** (= 13.6 eV / k_B)

### Why hot branch matters

α_B drops as T⁻⁰·⁸.  In HIM (T_QK ≈ 10⁶ K) the saturated T_DSP ≈ 5×10⁴ K
would **overestimate α_B by ~7×**.  Using T_QK gives the correct (small)
HIM contribution.  The total Hα change vs single-regime is small (~10 %)
but the HIM piece is now physically right.

### Implementation
- 📍 Function: [`_Halpha_luminosity`](../src/quokka2s/pipeline/prep/physics_fields.py#L614) in `physics_fields.py:614`
- 📍 Saha helper: [`_x_H_ion_saha`](../src/quokka2s/pipeline/prep/physics_fields.py#L180) in `physics_fields.py:180`
- 📍 Constants builder: [`_build_h_atomic_constants`](../src/quokka2s/pipeline/prep/physics_fields.py#L51) in `physics_fields.py:51`
- Cache: `('gas', 'H_alpha_luminosity')` is in CACHED_FIELDS
  ([cache.py CACHED_FIELDS set](../src/quokka2s/pipeline/cache.py))

---

## 2. HI 21 cm (1.42 GHz — hyperfine transition)

### Physics

²S_{1/2} hyperfine ²P_{3/2} → ²P_{1/2} of the ground 1s electronic state.
Only neutral H atoms can emit (H+ has no electron, free e⁻ has no nucleus
to couple with).  In the high-T limit (k T ≫ h ν₂₁/k = 0.07 K — always
true in ISM), 3/4 of the H atoms are in the upper hyperfine state:

```
ε_21 = (3/4) · n_HI · A_10 · h · ν_21           [erg/s/cm³]
```

- A_10 = 2.85×10⁻¹⁵ s⁻¹ (spontaneous emission rate; **Furlanetto, Oh & Briggs 2006 Phys.Rep. line after Eq.14; Pritchard & Loeb 2012 Rep.Prog.Phys. just below Eq.6 — both standard 21 cm reviews quote the same value**). Not in CHIANTI/LAMDA → pinned in code with citation.
- ν_21 = 1.420405751768×10⁹ Hz (NIST / NRAO; 12 significant figures)
- n_HI = neutral H number density [cm⁻³]

### Cold branch (T_QK < 3000 K)

```
n_HI = data[('gas', 'HI')]              # DESPOTIC 3D table → neutral atomic H number density
```

📝 **Naming convention**: the yt field is `('gas', 'HI')` (neutral atomic
H = n_HI [cm⁻³]).  The total H nuclei density is `('gas', 'number_density_H')`
= ρ·X_H/m_H — a different field.  The underlying LAMDA species token in
`H.dat` is still `'H'`, but the yt-side name was renamed to `'HI'` on
2026-06-20 to remove that ambiguity.

### Hot branch (T_QK ≥ 3000 K)

```
x_H+  = _x_H_ion_saha(T_QK, n_H_tot)
n_HI  = (1 − x_H+) · n_H_tot
```

Same Saha helper as Hα.

### ⚠️ Numerical zero at T > 10⁴ K — by design, not a bug

At T ≥ 10⁴ K, Saha gives x_H+ → 1 to all 16 float64 significant digits.
`1 − x_H+` rounds to **bit-exact 0**.  Therefore n_HI = 0, ε_HI = 0
for hot cells.

→ phase_combined HI panel is empty above T = 10⁴ K.  Not a clip, not
a truncation — Saha LTE physics + float64 precision.

### ⚠️ "QUOKKA-self-consistent" vs "MW-realistic" — known trade-off

Real WIM (T ≈ 10⁴ K) is only ~1 % ionised because **UV photoionisation +
recombination kinetics** balance it (CIE, not Saha).  Real WNM/WIM
contributes ~50-60 % of MW HI 21 cm emission.

Saha LTE assumes thermal collisional equilibrium — it puts x_H+ → 1
above 5000 K because exp(−13.6 eV / k_B T) becomes large.  This is
**internally consistent with QUOKKA's no-UV thermodynamics** but **NOT
with the real ISM**.  Total L_HI from our setup is ~10³¹ erg/s = ~5 dex
below MW for a 1×1×8 kpc patch.

This is a deliberate choice (recorded in
`[[two-regime-per-species]]` memory).  If we wanted MW-realistic HI we
would have to switch the hot branch to CHIANTI CIE (`fiasco.Element('H',
T).equilibrium_ionization`) — but then the hot-gas H ionisation would
no longer be self-consistent with QUOKKA's energy balance.

### L_HI phase decomposition (verified 2026-06-20)

| Phase | T_QK range [K] | L_HI [erg/s] | fraction |
|---|---|---|---|
| CNM | 0 – 200 | 8.3×10³⁰ | 39.0 % |
| UNM | 200 – 3000 | 1.2×10³¹ | 58.2 % |
| WNM | 3000 – 10000 | 5.9×10²⁹ | 2.8 % |
| WIM | 10000 – 3×10⁵ | 1.7×10¹⁵ | ~10⁻¹⁵ % |
| HIM | > 3×10⁵ | 0 (bit-exact) | 0 % |

### Implementation
- 📍 Function: [`_HI_luminosity`](../src/quokka2s/pipeline/prep/physics_fields.py#L670) in `physics_fields.py:670`
- 📍 Saha helper: [`_x_H_ion_saha`](../src/quokka2s/pipeline/prep/physics_fields.py#L180) in `physics_fields.py:180` (same as Hα)
- 📍 `('gas','HI')` species (n_HI from DESPOTIC for cold branch): [`_make_number_density_field('HI', lamda_token='H')`](../src/quokka2s/pipeline/prep/physics_fields.py#L582) in `physics_fields.py:582`
- 📍 Constants `A_HI_21`, `NU_HI_21`: [`_build_h_atomic_constants`](../src/quokka2s/pipeline/prep/physics_fields.py#L51) in `physics_fields.py:51`
- Cache: `('gas', 'HI_luminosity')` is in CACHED_FIELDS

---

## 3. CO J=1-0 (115.3 GHz — rotational line)

### Physics

J=1 → J=0 rotational transition of the CO molecule.  Excitation comes from
collisions with H₂ (cold dense gas) and electrons (warm gas, less
important here).  Below ~3000 K CO exists; above ~3000 K it's thermally
dissociated.

### Single regime: DESPOTIC LAMDA table

```
lumPerH = 3D-table lookup of CO line cooling rate at (n_H, N_H, dVdr)
                 [erg / s per H nucleus]
ε_CO    = n_H · lumPerH                            [erg/s/cm³]
```

DESPOTIC pre-computes lumPerH on a 3D grid by solving the level
populations + radiative transfer (LVG, escape probability) for every
(n_H, N_H, dVdr) point.

### Why no hot branch

At T_QK ≥ 3000 K the DESPOTIC table returns lumPerH ≈ 0 anyway because
CO is thermally dissociated — chemistry sets n_CO → 0 in hot gas.  No
analytic hot branch needed.  (Setting it explicitly to 0 vs trusting the
table to give 0 was discussed; the latter is simpler and works.)

### Implementation
- 📍 Field factory: [`_make_luminosity_field`](../src/quokka2s/pipeline/prep/physics_fields.py#L547) in `physics_fields.py:547` (creates a closure parameterised by species name)
- 📍 Registered as `('gas','CO_luminosity')` inside [`add_all_fields`](../src/quokka2s/pipeline/prep/physics_fields.py#L932) (`physics_fields.py:932`)
- Cache: `('gas', 'CO_luminosity')` is in CACHED_FIELDS
- Atomic data: DESPOTIC's bundled `co.dat` LAMDA file
- DESPOTIC table builder (offline, separate from runtime): `src/quokka2s/tables/solver.py`

---

## 4. [C II] 158 μm (1.9 THz — fine-structure transition)

### Physics

²P_{3/2} → ²P_{1/2} fine-structure transition of the C+ ion (one valence
electron, 2p¹ ground configuration).  Dominant cooling line of the
cold neutral medium AND a major coolant of the WIM.

### Cold branch (T_QK < 3000 K) — same as CO

```
lumPerH = 3D DESPOTIC table lookup, species 'C+'
ε_C+    = n_H · lumPerH                            [erg/s/cm³]
```

### Hot branch (T_QK ≥ 3000 K) — closed-form LTE

Five steps; constants from CHIANTI 10.1 via the `fiasco` Python package
(hard-coded at module import).

#### Step 1: H Saha → n_e

Same helper as Hα and HI:

```
x_H+ = _x_H_ion_saha(T_QK, n_H_tot)
n_e  = x_H+ · n_H_tot
```

(Negligible carbon contribution to electrons: n_C+/n_e ~ A_C ~ 10⁻⁴.)

#### Step 2: Two-stage C Saha + 3-state conservation

The carbon ionisation in HIM can reach C++, so a single-stage Saha
isn't enough.

```
S_C1 = 2 · (2π m_e k_B T / h²)^(3/2) · (U_C+/U_C0)  · exp(−I_C  / k_B T)
S_C2 = 2 · (2π m_e k_B T / h²)^(3/2) · (U_C++/U_C+) · exp(−I_C2 / k_B T)

r1 = S_C1 / n_e             # n_C+  / n_C0
r2 = S_C2 / n_e             # n_C++ / n_C+

#                                  C0 : C+ : C++  =  1 : r1 : r1·r2
x_Cp = r1 / (1 + r1 + r1·r2)       # ionisation fraction of C+
```

⚠️ **Partition functions are computed from CHIANTI's full level list**:

```
U(T) = Σ_i g_i · exp(−E_i / k_B T)
```

NOT from the single-ground-state degeneracy (`g_0 = 2` for C+, `g_0 = 1`
for C0).  Using `g_ground` ratios gives S_C off by a factor of 3 at WIM
temperatures (empirically verified 2026-06-17).

#### Step 3: n_C+

```
n_C+ = x_Cp · A_C · n_H_tot
```

- A_C = 1.6×10⁻⁴ = GOW chemistry network's total carbon abundance
  per H (xC default).  This is the value baked into the v4 DESPOTIC
  table; using it in the hot branch keeps cold/hot self-consistent.
  Do NOT use Wolfire's "gas-phase 1.4×10⁻⁴" — match the simulation,
  not the literature.

#### Step 4: Boltzmann LTE level population

```
r   = (g_u / g_l) · exp(−T* / T_QK)         # T* = ΔE_158/k_B = 91.21 K
n_u = n_C+ · r / (1 + r)                    # fraction in upper level
```

- g_u/g_l = 4/2 = 2  (²P_{3/2} has 4 sublevels, ²P_{1/2} has 2)

#### Step 5: Emissivity

```
ε_C+ = n_u · A_ul · h · ν                       [erg/s/cm³]
```

- A_ul = 2.290×10⁻⁶ s⁻¹  (CHIANTI 10.1)
- ν    = 1.9006×10¹² Hz (158.74 μm)

### Atomic constants (all from CHIANTI 10.1 / fiasco at module import)

| symbol | value | meaning |
|---|---|---|
| `_CII_A_UL`     | 2.290×10⁻⁶ s⁻¹  | 158 μm spontaneous-emission rate |
| `_CII_NU_HZ`    | 1.9006×10¹² Hz  | line frequency |
| `_CII_T_STAR`   | 91.214 K        | ΔE/k_B |
| `_CII_G_L, _CII_G_U` | 2, 4       | level degeneracies |
| `_CII_I_C_EV`   | 11.2603 eV      | C → C+ ionisation potential |
| `_CII_I_C2_EV`  | 24.3833 eV      | C+ → C++ ionisation potential |
| `_CII_U_C0`, `_CII_U_CP`, `_CII_U_CPP` | (T-grid arrays) | partition functions |
| `A_C_TOTAL`     | 1.6×10⁻⁴        | total C abundance per H (GOW xC) |

### ⚠️ Empirical result: hot branch ε_C+ ≈ 0 at low n_e

The 2-stage Saha at hot-gas low-n_e conditions collapses x_C+ → ~0 above
T ≈ 5000 K because:
- r1 saturates (C0 → C+ is easy at hot T)
- r2 grows rapidly with T (C+ → C++ takes over)
- x_Cp = r1/(1+r1+r1·r2) → 1/r2 → ~0

So C+ HIM contribution is small (visible as the yellow band in
phase_combined Row 2 col 2).  Total L_C+ is **dominated by the cold
branch** (DESPOTIC CNM/UNM emission).

This is consistent with Saha LTE physics but **NOT what CIE would say**
(CIE keeps x_C+ ≈ 1 across the entire WIM/HIM up to T ~ 10⁵ K).  Same
trade-off as HI 21 cm — internally consistent with QUOKKA, not
MW-realistic.

### Implementation
- 📍 Function: [`_Cplus_luminosity_two_regime`](../src/quokka2s/pipeline/prep/physics_fields.py#L754) in `physics_fields.py:754`
- 📍 CHIANTI atomic data builder: [`_build_cii_lte_atomic_tables`](../src/quokka2s/pipeline/prep/physics_fields.py#L104) in `physics_fields.py:104` (runs once at module import)
- 📍 Saha helper (Step 1, H-Saha): [`_x_H_ion_saha`](../src/quokka2s/pipeline/prep/physics_fields.py#L180) in `physics_fields.py:180`
- 📍 `A_C_TOTAL = 1.6e-4`: [physics_fields.py:101](../src/quokka2s/pipeline/prep/physics_fields.py#L101)
- Cache: `('gas', 'C+_luminosity')` is in CACHED_FIELDS
- CHIANTI install info: see memory `[[fiasco-chianti-installed]]`

---

## Summary table

| Species | Line | Regime | Cold path | Hot path | A_ul / coeff |
|---|---|---|---|---|---|
| Hα | 656.3 nm | Two | n_e, n_H+ from DESPOTIC + α_B(T_DSP) | Saha → n_e, n_H+ + α_B(T_QK) | α_B Draine fit |
| HI | 21 cm | Two | n_HI from DESPOTIC | n_HI = (1−x_H+)·n_H, Saha | A_10 = 2.85×10⁻¹⁵ s⁻¹ (Furlanetto+2006) |
| CO | 115 GHz J=1-0 | Single | DESPOTIC table lumPerH | (same; ε ≈ 0 above 3000 K) | atomic data in DESPOTIC c.dat |
| C+ | 158 μm | Two | DESPOTIC table lumPerH | 3-state Saha + LTE Boltzmann + CHIANTI | A_ul = 2.29×10⁻⁶ s⁻¹ |

## Quick reference for common questions

| Question | Answer |
|---|---|
| Why is the HI panel empty at T > 10⁴ K in phase_combined? | Saha LTE gives x_H+ = 1 in float64 → n_HI = 0 bit-exactly.  By design, not a bug. |
| Why is C+ ε so small in HIM cells? | Saha 2-stage gives x_C+ → 1/r2 ≈ 0 at low n_e high T. |
| Why does CO have non-zero ε at T = 10⁶ K? | DESPOTIC table interpolation returns tiny but non-zero lumPerH at HIM-like (n_H, N_H, dVdr).  Numerical artefact, ε ~ 10⁻⁵⁰ erg/s/cm³. |
| Why is total L_HI 5 dex below MW? | Saha LTE kills HI in WIM/HIM (real WIM is photo-balanced, not collisional). |
| Is `('gas', 'HI')` total H or neutral H? | **Neutral atomic H = n_HI** (from DESPOTIC LAMDA `H.dat`).  Total H nuclei = `('gas', 'number_density_H')`.  (Renamed `'H'` → `'HI'` on 2026-06-20 to eliminate this ambiguity.) |
| What threshold separates cold from hot? | `T_QK_TWO_REGIME_K = 3000.0` K.  Cold = `T_QUOKKA < 3000`, hot = `T_QUOKKA ≥ 3000`. |

## See also

In `docs/`:
- `colden_lateral_extension.md` — N_H lateral extension for L_ext
- `algorithm_audit_2026-05-29.md` — DESPOTIC table audit
- `despotic_validation_prop_sample.md` — validation methodology
