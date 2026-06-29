# What is CIE, and how is its table computed?

*A beginner's guide for the grad student staring at `_build_cie_ion_fractions()` for the first time.*

---

## 1. The one-sentence idea

**Collisional Ionization Equilibrium (CIE)** is the rule that, in hot diffuse gas, the fraction of an element sitting in each ionization stage (how many of its electrons have been stripped off) depends **only on the gas temperature** — not on its density — and we look those fractions up from a precomputed temperature-only table built by the CHIANTI atomic database.

---

## 2. What "ionization" and "ionization equilibrium" mean

Let's build this up slowly.

**An atom** is a nucleus surrounded by bound electrons. **Ionization** is the act of knocking one of those electrons off, leaving behind a positively charged **ion**. We label the stages by how many electrons have been removed:

- Carbon with all its electrons = neutral carbon, written **C0** (or C I).
- Strip one electron → **C+** (C II).
- Strip another → **C++** (C III), and so on up to fully bare **C6+**.

(The two notations line up exactly: the superscript-charge form **C0 / C+ / C++** equals the spectroscopic form **C I / C II / C III** — "I" is the neutral atom, "II" is singly ionized, and so on. They're the same thing written two ways.)

Think of these stages as **rungs on a ladder**. The hotter and more violent the environment, the higher up the ladder an element tends to climb.

Now, why "equilibrium"? In a real gas, atoms are constantly being pushed **up** the ladder (ionized) and falling back **down** (recombining — an ion captures a free electron). Two opposing processes are happening all the time:

- **Ionization** pushes a stage up: `z → z+1`.
- **Recombination** pulls it back down: `z+1 → z`.

If you leave the gas alone long enough, these two flows balance out for every rung. The number of atoms on each rung **stops changing** — not because nothing is happening, but because the up-flow and down-flow are equal. That steady state is **ionization equilibrium**.

> **Analogy — a tug-of-war that's standing still.** Ionization is one team pulling the rope "up the ladder"; recombination is the other team pulling "down." Equilibrium is the rope sitting still. It's not still because the teams quit — it's still because they pull equally hard. Where the rope rests (which ion dominates) tells you how hard each team is pulling.

**Collisional** Ionization Equilibrium specifies *what does the pushing*: fast-moving **free electrons** slamming into ions. This is the regime of hot gas — the solar corona, the hot phase of the interstellar medium, the gas between galaxies in a cluster, and (for us) the hot cells in a QUOKKA simulation.

---

## 3. CIE: the collisional-vs-recombination balance

Here are the only two processes that matter in CIE, and their rates **per unit volume**.

**(a) Collisional ionization** — a free electron hits ion stage `z` and knocks an electron off, making `z+1`:

```
R_ion = n_e * n_z * S_z(T)
```

**(b) Recombination** — a free electron is captured back onto stage `z+1`, making `z`:

```
R_rec = n_e * n_{z+1} * alpha_{z+1}(T)
```

Reading the symbols:
- `n_e` = **number density** of free electrons — that just means the count of free electrons per cm³. (You need an electron for **either** process.)
- `n_z`, `n_{z+1}` = number densities of the two ion stages.
- `S_z(T)` = the **ionization rate coefficient**, and `alpha_{z+1}(T)` = the **recombination rate coefficient**. These encode all the atomic physics — how fast the thermal electrons move and how big the relevant cross-sections are. **They depend only on temperature.**

Note that both rates above are **two-body**: each carries exactly one `n_e` (one electron does the colliding, one electron gets captured). That two-body form is the heart of CIE, and it's what makes the next step work.

**Equilibrium** means rate-up = rate-down for that pair of stages:

```
n_e * n_z * S_z(T)  =  n_e * n_{z+1} * alpha_{z+1}(T)
```

### The key insight: the density cancels

Look at that equation. **`n_e` appears on both sides.** Cancel it:

```
n_z * S_z(T)  =  n_{z+1} * alpha_{z+1}(T)
```

Rearrange to get the ratio of adjacent stages:

```
n_{z+1} / n_z  =  S_z(T) / alpha_{z+1}(T)        ← only temperature, no density!
```

The free-electron density vanished completely. What's left is a pure ratio of two temperature-only rate coefficients. So in CIE, **the relative ion fractions are a function of temperature alone**. Double the gas density, halve it — the *fractions* don't move. (CHIANTI literally calls these the **"zero-density ion fractions."**)

> **Why does `n_e` cancel?** In CIE the down-process is **two-body radiative recombination** (rate ~ `n_e · n_ion`), which matches the **two-body collisional ionization** (rate ~ `n_e · n_z`). Both directions carry exactly one `n_e`, so adding more electrons speeds up **both** by the same factor — like a discount applied to buyer and seller alike. The price (the equilibrium balance point) doesn't budge. **This symmetry holds only because recombination is two-body.** At high density a *three-body* recombination channel turns on (rate ~ `n_e² · n_ion`); then the down-rate scales as `n_e²`, the `n_e` no longer cancels, and you're in the density-dependent **Saha** regime instead (see §5, caveat 4). Density changes **how fast** CIE equilibrium is reached, not **where** it lands.

### From one ratio to a full curve

You have a ratio for every adjacent pair on the ladder. In the standard CIE approximation — where single ionization and single recombination dominate — each stage couples only to its immediate neighbours, so you can **chain** the ratios and express every stage relative to, say, the neutral one. Then you fix the absolute numbers with **normalization** — all stages of an element must add up to the whole element:

```
sum over all stages z of  f_z(T) = 1,   where f_z = n_z / n_element
```

(e.g. `f(O I) + f(O II) + ... + f(O IX) = 1`.)

Solve the chain + normalization at each temperature and you get **the ion fraction of every stage as a function of T**. Each ion "lights up" in a characteristic narrow temperature window (its peak-formation temperature) and is essentially absent above and below it.

> **A caveat on "neighbours only."** The neighbour-only chain is an *approximation*, not an exact law. Real CIE balance can include processes that couple **non-adjacent** stages — double (multiple) ionization and autoionization, and charge transfer in some treatments — so the true rate matrix is not strictly tridiagonal. Importantly, the **density-cancellation argument does not depend on tridiagonality**: every term in the full balance still carries one `n_e`, so `n_e` cancels regardless of which stages couple to which. Neighbour-only coupling is just what makes the *bookkeeping* a simple chain; it is not what makes the density drop out.

> **Analogy — a thermometer strip.** Each ion glows only in its own temperature band. O VI shines around ~3e5 K; Fe XVII–XXV peak at 1e6–1e8 K. Spot one of these ions in a spectrum and, *if CIE holds*, you've pinned the gas temperature. That's why such ions are prized temperature diagnostics for hot gas.

This same CIE physics is what underlies the classic **Sutherland & Dopita (1993)** radiative cooling function — the emission (and hence cooling) at each temperature is computed assuming CIE ion fractions, from ~1e4 to ~1e8.5 K.

---

## 4. How the table is actually computed (CHIANTI + fiasco)

CIE tells us the *shape* of the calculation. **CHIANTI** supplies the actual numbers. Here's the assembly line.

### Step 1 — Look up the rate coefficients vs T

For every ion of every element (H through Zn), CHIANTI has two temperature-dependent coefficients pre-baked into its database (units cm³/s):

- **Total ionization rate** `alpha_I(T) = alpha_DI + alpha_EA`
  - *Direct ionization* (DI): an electron knocks out a bound electron.
  - *Excitation–autoionization* (EA): a collision excites an inner electron into an unstable state that then spits out an electron on its own.
- **Total recombination rate** `alpha_R(T) = alpha_RR + alpha_DR`
  - *Radiative recombination* (RR): an electron is captured and a photon carries off the energy (and escapes — that's the "low density / optically thin" part).
  - *Dielectronic recombination* (DR): an electron is captured while simultaneously bumping a bound electron up; the energy is radiated away later.

These are **not derived from scratch each run** — they come from lab measurements (Dere 2007 fits to laboratory cross-sections for low stages) and quantum-mechanical atomic-physics codes (FAC distorted-wave for higher stages; AUTOSTRUCTURE for dielectronic rates).

> **Analogy — a pre-measured lookup book.** The rate coefficients are like a reference book of "pump strengths vs temperature," assembled by experimentalists and theorists. You don't re-measure them — you look them up. **fiasco** is the Python librarian that reads the book (CHIANTI 10.1) and does the arithmetic for you.

### Step 2 — Assemble and solve the balance

For each element you have a chain of stages (carbon: C0, C+, C++, …, C6+). In equilibrium every stage is steady, `dn_i/dt = 0`: the rate **leaving** stage `i` equals the rate **arriving**. Written out for all stages, this is a coupled linear system. In the standard CIE approximation (single ionization + recombination dominate) each stage couples only to its neighbours, so the system is effectively **tridiagonal** and reduces, for any adjacent pair, to the textbook relation:

```
n_{i+1} / n_i  =  alpha_I(i, T) / alpha_R(i+1, T)
```

The key structural fact is that **every term in the system carries one `n_e`**, so it cancels — that's what confirms density-independence, and (as noted in §3) it would still cancel even if double-ionization or charge-transfer terms made the matrix not strictly tridiagonal.

> **Analogy — buckets and water.** Picture the stages as a row of buckets, electrons as water. Ionization pumps water UP a bucket; recombination pumps it DOWN. At equilibrium every bucket's level is steady. Both pumps are **two-body** (each driven by one electron), so the pump *strengths* depend on temperature, while raising `n_e` is like turning up a shared pressure dial that drives both pumps equally: it reaches equilibrium faster but doesn't change the final levels. (If a *three-body* recombination pump switched on at high density — driven by two electrons — that shared-dial symmetry would break, and you'd be in the Saha regime.)

### Step 3 — The result is a fraction-vs-T table

Solve that system at every temperature on a grid and you get, for element X, the **fraction in each stage as a function of T**. By construction the fractions sum to 1 at every temperature. There is **no density axis** — that's the defining feature of CIE and exactly why our code can interpolate on temperature alone. (CHIANTI's native ionization-equilibrium tables extend to ~1e9 K, but our builder requests a grid from **1e4 to 10^8.5 K** — see Step 4 and §6.)

### Step 4 — The fiasco API we call

In code, this is a one-liner per element:

```python
fiasco.Element('carbon', T_grid).equilibrium_ionization
```

`.equilibrium_ionization` returns a 2-D array of shape `(n_T, n_stages)`. **Column index = charge state**: column 0 = neutral, column 1 = singly ionized, column 2 = doubly ionized, etc. We're pinned to **CHIANTI 10.1** (installed at `~/.fiasco`), which refit ionization rates to new lab data and revised recombination rates.

---

## 5. CIE vs Saha vs PIE — when does each apply?

CIE is one of *three* ionization-balance regimes. They differ in **who does the ionizing** and **whether density matters**. This trips people up constantly, so here's the cheat sheet:

| Regime | What ionizes? | What recombines? | Density-dependent? | Where it applies |
|---|---|---|---|---|
| **Saha (LTE)** | Collisions (thermal electrons) | All inverse processes in detailed balance — collisional/three-body **and** radiative; populations set by LTE statistical mechanics at temperature T | **Yes** — ratio carries an explicit `1/n_e` | Very high density: stellar interiors (`n_e ~ 1e15–1e16 cm⁻³`+) |
| **CIE ("coronal")** | Collisions (thermal electrons) | **Radiative** recombination (photon escapes; two-body, ~`n_e·n_ion`) | **No** — `n_e` cancels | Hot, low-density, optically-thin gas: corona, hot ISM, intracluster medium |
| **PIE (photoionization)** | **External UV/X-ray photons** (e.g. O/B-star starlight) | Radiative recombination | Set by the **ratio** `U` (see below) | Photoionized gas: HII regions, the Warm Ionized Medium (WIM), much of the circumgalactic medium |

The math behind the table:

- **Saha:** `n_{z+1}/n_z = (1/n_e) * 2 * (g_{z+1}/g_z) * (2π m_e k T / h²)^(3/2) * exp(-chi_z / kT)`. The explicit **`1/n_e`** comes from the phase-space / partition-function ratio for the reaction `e⁻ + ion ⇌ atom`, **not** from any single down-process like three-body recombination. Saha is an LTE statement: it holds whenever the ionization/level populations are collisionally controlled and the electrons are Maxwellian at temperature T. (It does **not** require a Planckian radiation field — stellar-atmosphere LTE uses Saha even where `J_ν ≠ B_ν`. Strict thermodynamic equilibrium with a Planckian field is a *stronger, separate* condition.)
- **PIE:** `n_ion * Gamma = n_e * n_{ion+1} * alpha(T)`, where `Gamma` (the photoionization rate per ion) ∝ photon density. Solving, `n_{ion+1}/n_ion = Gamma / (n_e * alpha)`. The ionizing side has **no** `n_e` (a photon, not an electron, does the ionizing), but the recombination side does. Crucially, the ionization **state** is set by the **dimensionless ionization parameter** — schematically `U ~ n_gamma/n_H` (more standard: `U = Q / (4π r² n_H c)`, ionizing-photon flux over `n_H c`). At **fixed U the fractions are fixed**: scale `n_gamma` and `n_H` together and nothing changes. So PIE is "density-dependent" only in the sense that, for a *fixed external radiation field*, raising `n_H` lowers `U` and lowers the ionization — it is **not** density-dependent in the trivial way Saha is.

> **Analogy — who throws the punches.** **CIE** = a mosh pit where fast electrons (set by temperature) do the knocking, and energy escapes as light (low density). **Saha** = a pit so dense that collisions also do the *catching* — ionization balance is fixed by LTE statistical mechanics at temperature T, and the `1/n_e` reflects how much phase space a freed electron has (stellar interiors). **PIE** = ionization done by an external floodlight (O/B-star UV) rather than the crowd's own motion; what matters is the *ratio* of floodlight brightness to crowd size (`U`), not the crowd size alone (HII regions, the WIM).

---

## 6. How WE use it in this codebase

File: `<repo-root>/src/quokka2s/pipeline/prep/physics_fields.py`.
*(Line numbers below are a **2026-06-29 snapshot**; regrep if they have drifted. The package is at the repo-root `src/quokka2s/`.)*

### The 3-regime split

Our line-emission fields (Hα, HI 21 cm, [C II] 158 μm) pick an ionization model based on **QUOKKA's simulation temperature** `T_QK`, using two thresholds:

```
T < 3000 K        → DESPOTIC 3D LAMDA table        (cold molecular chemistry)
3000 K ≤ T < 1e4 K → analytic Saha (LTE, density-dependent bridge)
T ≥ 1e4 K          → CHIANTI CIE ion fractions      (density-INDEPENDENT; the new regime)
```

The constants are `T_QK_TWO_REGIME_K = 3000.0` and `T_CIE_K = 1.0e4` (lines 36 & 43). The 3000 K floor is below the WIM thermal floor but above where DESPOTIC's cold-chemistry network is useful; DESPOTIC's solver also saturates around ~5e4 K, so hot gas *must* be handled analytically. 1e4 K is the floor of CIE validity.

> **Naming caution.** The code's variables and comments call the middle regime "Saha CIE" (e.g. the `_x_H_ion_saha` docstring near line 213, and the comment near line 683). That naming is **loose**: the middle band is plain **LTE Saha, which is density-*dependent*** — and that is exactly why it is **not** CIE in the §5 sense. We reserve "CIE" for the CHIANTI density-*independent* regime (T ≥ 1e4 K). When you see "Saha CIE" in the source, read it as "LTE Saha."

> **Analogy — pick the right expert.** Below 3000 K you ask the cold-chemistry specialist (DESPOTIC); in the lukewarm band you do a quick hand calculation (LTE Saha); once it's genuinely hot you consult the standard hot-plasma reference table (CHIANTI CIE). Each expert only answers "what fraction is in this ion stage?" — the rest of the emission recipe is identical regardless of who you asked.

### The builder and the cached arrays

`_build_cie_ion_fractions()` (defined line 191) runs **once at import**. It builds a grid `np.logspace(4.0, 8.5, 600)` K (so **1e4 to 10^8.5 K ≈ 3e8 K**, consistent with Step 3 above), calls `equilibrium_ionization` for hydrogen and carbon, scrubs NaNs, and slices columns into five module-level 1-D arrays (unpacked at line 205):

- `_CIE_T_GRID` — the 600-point temperature grid
- `_CIE_X_H0` — neutral-H fraction (H column 0)
- `_CIE_X_HP` — ionized-H fraction (H column 1)
- `_CIE_X_CP` — singly-ionized carbon C+ (C column 1)
- `_CIE_X_CPP` — doubly-ionized carbon C++ (C column 2) — *built but not yet consumed; reserved for future use*

Per cell, evaluation is just a cheap `np.interp(T_qk, _CIE_T_GRID, ...)`. The module is **fail-loud**: both fiasco-backed builders (the [C II] LTE tables at line 179, *then* the CIE fractions at line 205) import fiasco with no `try/except`, so a missing CHIANTI install crashes **at import** rather than silently degrading — consistent with the project's no-silent-simplification rule. (If you're tracing exactly *where* it dies, the [C II] builder at line 179 runs first and would be the earliest to fail.)

### Which fraction each line takes

CIE only supplies **the ionization fraction**. The emissivity formula for each line is **unchanged** — only the *source* of the fraction swaps in the hot regime:

- **Hα** uses the **ionized** fraction `x_H+` → `n_e = n_H+ = x_H+(T)·n_H`, then the unchanged Draine 2011 Eq 14.6 emissivity `eps = 0.45 · E_Hα · alpha_B(T) · n_e · n_H+`. Note `alpha_B` is **still the Draine 2011 fit evaluated at T_QK** (lines 698–703) — CIE supplies *only* `x_H+`, not the recombination coefficient. (This mirrors the C+ point in caveat 9(d): the hot branch swaps the ionization stage and nothing else.)
- **HI 21 cm** uses the **neutral** fraction `x_H0` directly → `n_HI = x_H0(T)·n_H`, then `eps = 0.75 · n_HI · A_10 · h · nu_21`.
- **[C II] 158 μm** uses **C+** → `n_Cp = x_C+(T)·A_C·n_H` (A_C = 1.6e-4), then the same LTE Boltzmann level population and `eps = n_u · A_ul · h · nu` as the Saha branch.

The branches are merged with a nested `np.where(T_qk >= 1e4, eps_CIE, np.where(T_qk >= 3000, eps_Saha, eps_DESPOTIC))`. (`np.select` is deliberately avoided — astropy's helper mishandles an array `default` during yt's field-dependency detection.)

### Why we switched Saha → CIE: the C+ collapse (and why it's *not* a clean win for hydrogen)

This is the punchline for *why CIE was added at all* — but it cuts **both ways**, and that subtlety matters.

**For carbon, CIE genuinely rescues the line.** LTE Saha is density-dependent, and in hot **diffuse** gas the density is tiny, so Saha drives the free-electron count down so far that recombination is *starved* — the balance tips toward ever-higher ionization. For carbon this **over-ionizes**: Saha drove `x_C+` toward **~0** at 3e4 K (carbon shoved up to C++ and beyond), which effectively **killed the [C II] hot-branch emission**. CHIANTI CIE — density-independent, with radiative recombination folded in at coronal equilibrium — gives `x_C+ ≈ 0.93` at 3e4 K instead of ~0. There, CIE is unambiguously the better model.

**For hydrogen, the same density-independence cuts the other way.** CIE under-ionizes H at warm temperatures (this is just caveat 1 below, made concrete). Reading the actual CHIANTI table the code uses:

| T | `x_H+` (CIE) | `x_H0` (CIE) |
|---|---|---|
| 1e4 K | **0.0018** | 0.998 |
| 1.5e4 K | 0.36 | 0.64 |
| 2e4 K | 0.92 | 0.08 |
| 3e4 K | 0.996 | 0.004 |

So at the seam temperature (1e4 K) the gas is **essentially neutral** in CIE, and `x_H+` doesn't climb past ~0.9 until ~2e4 K. Because the Hα emissivity goes as `alpha_B · n_e · n_H+` with `n_e = n_H+ = x_H+ · n_H`, the **hot-CIE branch produces near-zero Hα for all 1e4–1.5e4 K gas**. And there is a real, possibly large **discontinuity at the 1e4 K seam**: just below it, low-density Saha *over*-ionizes H (pushing `x_H+` up); just above it, CIE *under*-ionizes H (`x_H+ ~ 0.002`). The two formulas can disagree by orders of magnitude — and the jump can go in **either** direction — so this is not the smooth, strictly-better upgrade the carbon story alone might suggest.

The reason is structural: the code's threshold is on **`T_QK`**, which does not know whether a cell is actually **photoionized**. In the WIM/warm regime the simulation cares about, the real ionization is set by stellar UV (a PIE problem), and *neither* low-density Saha nor CIE captures that. So treat "we switched to CIE above 1e4 K" as **"CIE is the right hot-gas ionization model in general, and it fixes the C+ collapse"** — *not* as "Hα got more realistic in the warm regime." For Hα at 1e4–1.5e4 K the switch arguably makes the number physically *worse*, and the 1e4 K seam deserves a careful eyeball (caveat 9b). The `1e4 K` threshold is **load-bearing**, not cosmetic.

---

## 7. Caveats (read before trusting a CIE number)

1. **The WIM is photoionized, NOT collisionally ionized — CIE is wrong there.** The Galaxy's Warm Ionized Medium (T ~ 1e4 K, n ~ 0.1 cm⁻³) is ionized by **O/B-star UV photons**, i.e. it's a **PIE** environment governed by the ionization parameter `U`. At its temperature, collisions alone barely ionize hydrogen — recall `x_H+(CIE) ≈ 0.0018` at 1e4 K — so **CIE badly under-ionizes H** there, and (per §6) that directly suppresses the hot-branch Hα. The photons do the work CIE doesn't account for. For such gas you'd use a photoionization code (e.g. CLOUDY), not CIE. Our `T ≥ 1e4 K` gate is keyed on the *simulation* temperature and does **not** know whether a cell is actually photoionized — keep that in mind for warm, irradiated gas.

2. **CIE assumes NO strong external radiation field and optically-thin gas.** Add a UV/X-ray background and the gas can be *over-ionized* relative to CIE at the same temperature (the PIE/hybrid regime — common in the circumgalactic medium, where authors must decide PIE vs CIE vs hybrid for absorbers like O VI).

3. **CIE assumes the gas reached STEADY STATE.** If the gas heats or cools faster than it can ionize/recombine — rapidly cooling supernova ejecta, fast shocks, cooling galactic-fountain gas — the ion fractions *lag* the temperature. That's **non-equilibrium ionization**, and CIE tables give wrong answers. Always check that `t_recomb` and `t_ion` are short compared to the dynamical/cooling time.

4. **Density-independence is an idealization** (the "zero-density" limit). It rests on recombination being **two-body** (radiative, ~`n_e·n_ion`), matching two-body collisional ionization, so the shared `n_e` cancels. At high enough density, **three-body recombination** (~`n_e²·n_ion`) and metastable-level effects creep in; the down-rate then scales as `n_e²`, the cancellation breaks, and you slide toward the density-dependent Saha regime. Negligible for typical hot diffuse gas — which is why CIE is so useful — but not exact.

5. **Different CIE tables disagree by tens of percent** because they use different rate coefficients (Sutherland & Dopita vs CHIANTI vs others). Each is internally consistent; **don't mix** ion fractions from one with rates from another. Cite which one you used — we use **CHIANTI 10.1** specifically. Re-deriving with a different CHIANTI version (v9, v10, v11) would shift the numbers; v11 even adds density-dependent and charge-transfer models that v10.1 — and therefore our code — does **not** use.

6. **CIE assumes a Maxwellian (thermal) electron distribution.** Some space/solar plasmas have non-Maxwellian (e.g. kappa) electrons, which shift the fractions; specialized tables exist (Dzifcakova & Dudik 2013).

7. **"T ≥ 1e4 K" is a rough floor for *hydrogen*, not "everything is ionized."** As the table in §6 shows, even hydrogen is only ~0.2% ionized at exactly 1e4 K in CIE; heavy elements' high stages need far hotter gas (Fe XVII–XXV peak at 1e6–1e8 K). "CIE applies" ≠ "fully ionized at 1e4 K."

8. **Density-independence is about the *fractions* only.** The actual number densities still scale with density: `n_C+ = x_C+(T) · A_C · n_H`. Only the fractional split among stages is density-free.

9. **Implementation gotchas in our code.** (a) `np.nan_to_num` turns any NaN fraction into 0 — a safety net that would *silently mask* a whole element going NaN, so watch for anomalously-zero hot-branch fractions. (b) The 3000 K and 1e4 K seams are **hard, unblended splits** — a discontinuity at the boundaries is possible (for Hα at 1e4 K it can be large; see §6), so eyeball it in slices. (c) HI reads `_CIE_X_H0` *directly* rather than `1 - x_H+` — and these are **exactly** equal, not just approximately: hydrogen's fiasco array is literally shape `(n_T, 2)` (columns H0 and H+), so after `nan_to_num` the two columns sum to 1 by construction. The direct-read is a deliberate, equivalent choice. (d) For C+ the LTE upper-level population is still a Boltzmann population at `T_QK` in *both* hot branches, and Hα's `alpha_B` is likewise evaluated at `T_QK` — **CIE fixes only the ionization stage, not the excitation or the recombination coefficient.** (e) Adding the CIE branch changed Build-task output *without* changing init args, so stale cached intermediates may need a full recompute (`--mode all`/compute) rather than a cache-hit plot.

---

## Key references

- **Kaastra**, *Collisional Ionisation Equilibrium (CIE)*, NED Level 5 — the balance equation, `n_e` cancellation, validity conditions.
- **Sutherland & Dopita (1993)** — the canonical CIE radiative cooling function (1e4–1e8.5 K).
- **Dere et al. (2009)**, A&A 498, 915 — CHIANTI ionization/recombination rates and "zero-density ion fractions."
- **Dere et al. (2023)** — CHIANTI Version 10.1 (the release this codebase is pinned to).
- **fiasco docs** — `Element.equilibrium_ionization` returns an `(n_T, n_stages)` array; `Ion.ionization_rate = alpha_DI + alpha_EA`, `recombination_rate = radiative + dielectronic`.
- **Haffner et al.**, NED Level 5 — the WIM as a photoionization-equilibrium environment.
- Code: `<repo-root>/src/quokka2s/pipeline/prep/physics_fields.py` — lines 36 & 43 (the 3-regime constants), 191–205 (`_build_cie_ion_fractions` + cached arrays), 650–709 (Hα), 712–767 (HI), 793–904 ([C II]). *(Line numbers are a 2026-06-29 snapshot; regrep if drifted.)*
