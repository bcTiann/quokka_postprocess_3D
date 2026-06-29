# Velocity dispersions & $\sigma$ in this pipeline: step-by-step derivations

This document explains **every** velocity dispersion ($\sigma$) the post-processing code computes, derives each formula from scratch, and gives small numbers you can check by hand. The goal is that a beginner can both *understand* and *verify* each $\sigma$.

There are three conceptually different families of $\sigma$ here, and it is essential to keep them apart:

1. **Gas kinematic dispersions** — spreads of the *gas velocity field* itself, weighted by mass. These subtract a mean velocity (they are *dispersions* / standard deviations). Functions: `mass_weighted_sigma`, `mass_weighted_sigma_3d`, `mass_weighted_sigma_by_phase`, `spaxel_moments_along_axis`.
2. **Intrinsic line widths** — per-cell *broadening kernels* that have nothing to do with bulk flow: the thermal Doppler width $\sqrt{k_BT/m}$ and the instrumental LSF width $(c/R)/2.355$. These subtract **no** mean; they are pure widths.
3. **Observed spectral width** — the second moment of an already-built emission-line spectrum, weighted by intensity (luminosity). Function: `_moment_sigma`.

A one-line summary table is at the very end (§8).

---

## 0. Setup & notation

### Cells and the velocity field

The simulation gives us a 3D Cartesian grid of gas cells. For cell $i$ we have a mass density $\rho_i$ and velocity components $(v_{x,i}, v_{y,i}, v_{z,i})$ in km/s. We will work on a **uniform covering grid**, where every cell has the *same volume* $V_{\rm cell}$. This fact is what lets us use density as the weight (see below).

### The mass weight $w_i$

Almost every $\sigma$ in the gas-kinematics family is a **mass-weighted** statistic. The natural weight is the cell's *mass* $m_i = \rho_i V_i$. Define the normalised weight

$$
w_i = \frac{m_i}{\sum_j m_j}, \qquad \sum_i w_i = 1, \qquad w_i \ge 0 .
$$

The set $\{w_i\}$ is a discrete probability distribution: it tells us "what fraction of the total mass lives in cell $i$." With it, the **expectation operator** is

$$
\mathbb{E}[g] \equiv \sum_i w_i\, g_i .
$$

### Why density $\rho_i$ may be used directly as the weight

On a uniform grid every cell has the same volume, $V_i = V_{\rm cell} = \text{const}$. Then the constant cancels:

$$
w_i = \frac{m_i}{\sum_j m_j} = \frac{\rho_i V_{\rm cell}}{V_{\rm cell}\sum_j \rho_j} = \frac{\rho_i}{\sum_j \rho_j}.
$$

So we never need the cell volume — **density is the weight**. (If cells had *unequal* volumes this cancellation would fail and you would have to weight by $\rho_i V_i$.) Every function below builds `w = rho / rho.sum()`, exactly this expression.

### Two algebraic forms of a variance

For any weighted distribution, the variance can be written two equivalent ways:

$$
\underbrace{\sum_i w_i (v_i - \langle v\rangle)^2}_{\text{centered (two-pass) form}}
\;=\;
\underbrace{\sum_i w_i v_i^2 - \langle v\rangle^2}_{\text{raw-moment (one-pass) form}}
\;=\; \mathbb{E}[v^2] - \mathbb{E}[v]^2 .
$$

**Proof — every step spelled out.** Throughout, $\langle v\rangle$ is a *single fixed number* (the mean), identical for every cell $i$, so it can be pulled out of any sum over $i$.

**Step 1 — expand the square.** Use the binomial $(a-b)^2 = a^2 - 2ab + b^2$ with $a=v_i$ and $b=\langle v\rangle$:

$$
(v_i - \langle v\rangle)^2 \;=\; v_i^2 \;-\; 2\,v_i\langle v\rangle \;+\; \langle v\rangle^2 .
$$

That `2` is simply the cross term of $(a-b)^2$ — it is **not** a typo. Watch where it goes.

**Step 2 — multiply every term by $w_i$ and sum over $i$.** A sum splits over a $+/-$ of terms, so we get three separate sums:

$$
\sum_i w_i (v_i - \langle v\rangle)^2
= \underbrace{\sum_i w_i v_i^2}_{\text{(A)}}
\;-\; \underbrace{\sum_i w_i\,(2\,v_i\langle v\rangle)}_{\text{(B)}}
\;+\; \underbrace{\sum_i w_i\,\langle v\rangle^2}_{\text{(C)}} .
$$

**Step 3 — pull the constants ($2$, $\langle v\rangle$, $\langle v\rangle^2$) outside each sum:**

$$
\text{(A)} = \mathbb{E}[v^2], \qquad
\text{(B)} = 2\langle v\rangle\underbrace{\sum_i w_i v_i}_{=\,\langle v\rangle}, \qquad
\text{(C)} = \langle v\rangle^2\underbrace{\sum_i w_i}_{=\,1}.
$$

**Step 4 — substitute the two facts** $\displaystyle\sum_i w_i v_i = \langle v\rangle$ (definition of the mean) and $\displaystyle\sum_i w_i = 1$ (weights sum to one). Then (B) $= 2\langle v\rangle\cdot\langle v\rangle = 2\langle v\rangle^2$ and (C) $= \langle v\rangle^2\cdot 1 = \langle v\rangle^2$, giving

$$
\sum_i w_i (v_i - \langle v\rangle)^2
\;=\; \mathbb{E}[v^2] \;-\; 2\langle v\rangle^2 \;+\; \langle v\rangle^2 .
$$

**Step 5 — combine the last two like-terms** (this is where the `2` "disappears"). Both $-2\langle v\rangle^2$ and $+\langle v\rangle^2$ are multiples of $\langle v\rangle^2$, so add their coefficients: $-2+1=-1$:

$$
-2\langle v\rangle^2 + \langle v\rangle^2 = (-2+1)\,\langle v\rangle^2 = -\langle v\rangle^2
\;\;\Longrightarrow\;\;
\boxed{\;\sum_i w_i (v_i - \langle v\rangle)^2 = \mathbb{E}[v^2] - \langle v\rangle^2\;}\qquad\blacksquare
$$

(The original one-line proof just compressed Steps 4–5: it wrote $-2\langle v\rangle^2 + \langle v\rangle^2$ and collapsed it to $-\langle v\rangle^2$ in the same breath — nothing was wrong, the step was just hidden.)

**Numeric check of the identity** — cells $v=[10,20,30]$, $\rho=[1,1,2]$, so $w=[0.25,0.25,0.5]$ and $\langle v\rangle=22.5$:

- centered form: $0.25(10-22.5)^2 + 0.25(20-22.5)^2 + 0.5(30-22.5)^2 = 39.0625 + 1.5625 + 28.125 = 68.75$
- raw-moment form: $\mathbb{E}[v^2]-\langle v\rangle^2 = (0.25\cdot 100 + 0.25\cdot 400 + 0.5\cdot 900) - 22.5^2 = 575 - 506.25 = 68.75$

Both give $68.75$ ✓.

Both forms give the *same* number in exact arithmetic. They differ **numerically**: the raw-moment form subtracts two large nearly-equal numbers (catastrophic cancellation) and can drift slightly negative; the centered form cannot. The codebase deliberately uses the **centered** form in the mass-weighted family and the **raw-moment** form (with a guard) in the per-spaxel collapse — for performance reasons explained in §4.

> Throughout, these are **population** standard deviations: we divide by the total weight (which is 1 after normalisation), with **no** Bessel $n-1$ correction. And $\sigma$ is invariant to a uniform velocity shift $v \to v + c$ (a dispersion does not care about the zero point) — *provided* the reference velocity it is measured about is itself a mean of the data, so it co-shifts by the same $c$. This holds for §1/§3/§4/§7 (each subtracts the data's own mean) and, slightly less obviously, for §2 (see the note there).

---

## 1. The building block — 1D mass-weighted mean & dispersion

**Function:** `mass_weighted_sigma` — `src/quokka2s/pipeline/utils.py:74-82`

This computes the mass-weighted **mean** and **standard deviation** of *one* velocity component over a set of cells.

### Derivation

With weights $w_i = \rho_i / \sum_j \rho_j$, the mass-weighted mean (first moment) is

$$
\langle v\rangle = \sum_i w_i v_i = \frac{\sum_i \rho_i v_i}{\sum_i \rho_i},
$$

and the dispersion is the square root of the mass-weighted second *central* moment,

$$
\boxed{\;\sigma = \sqrt{\sum_i w_i\,(v_i - \langle v\rangle)^2}\;}
$$

This is just the textbook definition of standard deviation, with "probability of outcome $i$" replaced by "fraction of mass in cell $i$." It answers: *how spread out are the velocities, counting each cell in proportion to how much mass it holds.*

### Code walk-through (`utils.py:74-82`)

```
L76  total  = rho.sum()                                # Σ_j ρ_j
L77  if total <= 0: return (nan, nan)                  # empty / zero-mass guard
L79  w      = rho / total                              # w_i = ρ_i/Σρ ,  Σw_i = 1
L80  v_mean = sum(vel_kms * w)                          # ⟨v⟩  (1st moment)
L81  sigma  = sqrt(sum((vel_kms - v_mean)**2 * w))      # √(2nd central moment)
L82  return v_mean, sigma
```

Note line 81 uses the **centered** form `(vel - v_mean)**2`, the numerically stable variant.

### Worked numeric check (do it by hand)

Take three cells: $v = [10, 20, 30]$ km/s, $\rho = [1, 1, 2]$.

- Weights: $w = [0.25, 0.25, 0.5]$ (sum $= 1$ ✓).
- Mean: $\langle v\rangle = 0.25\cdot10 + 0.25\cdot20 + 0.5\cdot30 = 22.5$.
- Variance: $0.25(10-22.5)^2 + 0.25(20-22.5)^2 + 0.5(30-22.5)^2 = 39.0625 + 1.5625 + 28.125 = 68.75$.
- $\sigma = \sqrt{68.75} = 8.29156$ km/s.

(Confirmed numerically: `mean, sigma = 22.5, 8.2915619758885`.)

### Where it's used

This is the canonical 1D dispersion. `Build_VelocityPhase` calls it for `total_x/y/z` (`velocity_phase.py:81-83`), and `mass_weighted_sigma_by_phase` calls it internally (`utils.py:153`) to get the **global** mean $v_{\rm global}$ used in §2.

> **Caveat (a private duplicate):** `integrated_spectrum.py:61-65` has its own `_mass_weighted_sigma` that is identical *except* it returns only $\sigma$ and **has no `total <= 0` guard** — it will divide by zero if `rho.sum() == 0`. Keep an eye on it if you ever feed it an empty selection.

---

## 2. Per-phase $\sigma$ about the GLOBAL center of mass

**Function:** `mass_weighted_sigma_by_phase` — `src/quokka2s/pipeline/utils.py:133-170`

The gas is split into 5 disjoint ISM temperature phases by `classify_temperature_phase` (`utils.py:52-59`), with cuts (`utils.py:45-48`):

| Phase | Temperature range |
|-------|-------------------|
| CNM | $T < 200$ K |
| UNM | $200 \le T < 3000$ K |
| WNM | $3000 \le T < 10^4$ K |
| WIM | $10^4 \le T < 10^{5.5}$ K |
| HIM | $T \ge 10^{5.5}\,(\approx 3.16\times10^5)$ K |

For each phase $P$ (say, the WIM) we want one number $\sigma_P$: *how spread out are the velocities of the gas in this phase?* There are **two different things** that could mean — and the 2026-06-29 change switched from the first to the second:

- **(old) spread about the phase's OWN average** — how much the WIM gas scatters around the WIM's *own* bulk velocity. This is the phase's purely **internal** spread.
- **(new) spread about the WHOLE gas's average** — how far the WIM velocities are from the bulk velocity of *all* gas together. This *also* picks up any systematic **streaming** of the WIM relative to the rest of the gas.

> **Analogy.** Measuring how "spread out" a marching band's trumpet section is. *Own-average:* how scattered the trumpets are around the trumpets' **own** center — their internal tightness. *Global-average:* how scattered they are around the **whole band's** center — which is large if the trumpet section has drifted off as a group, even when the trumpets are tight among themselves. We switched to the global-average version.

**Three velocities — keep them straight (this is the usual point of confusion):**

| symbol | what it is | averaged over which cells |
|---|---|---|
| $v_i$ | one cell's velocity component | the single cell $i$ |
| $\langle v\rangle_P$ | the **phase's own** mass-weighted mean | cells in phase $P$ **only** |
| $v_{\rm global}$ | the **all-gas** mass-weighted mean | **every** cell, all phases |

$\sigma_P$ asks: *how far do the phase-$P$ cells ($v_i$) sit from $v_{\rm global}$?* — using $v_{\rm global}$ (not $\langle v\rangle_P$) as the zero point.

### The two steps

**Step A — global reference velocity (the shared frame):**

$$
v_{\rm global} = \frac{\sum_{i\in{\rm all}} \rho_i v_i}{\sum_{i\in{\rm all}}\rho_i}.
$$

Computed once at `utils.py:153` by calling `mass_weighted_sigma(vel, rho)` over *all* cells and keeping the mean.

**Step B — per-phase quantities.** With *local* (within-phase) weights $w_i^P = \rho_i / \sum_{j\in P}\rho_j$ (so $\sum_{i\in P} w_i^P = 1$):

$$
\langle v\rangle_P = \sum_{i\in P} w_i^P v_i \quad\text{(phase's own bulk velocity, reported)},
$$
$$
\boxed{\;\sigma_P = \sqrt{\sum_{i\in P} w_i^P\,(v_i - v_{\rm global})^2}\;}\quad\text{(about the GLOBAL mean)} .
$$

### The decomposition theorem (parallel-axis / König–Huygens)

Why is subtracting $v_{\rm global}$ (a non-mean point) meaningful? Because the variance about *any* fixed reference $a$ splits cleanly.

**Theorem.** For any fixed $a$,
$$
\sum_{i\in P} w_i^P (v_i - a)^2 = \sigma_{P,\rm own}^2 + (\langle v\rangle_P - a)^2,
\qquad \sigma_{P,\rm own}^2 \equiv \sum_{i\in P} w_i^P (v_i - \langle v\rangle_P)^2 .
$$

**Proof — every step.** The trick: split each deviation *through the phase's own mean*. For each cell,

$$
v_i - a \;=\; \underbrace{(v_i - \langle v\rangle_P)}_{\text{deviation from own mean}\;\equiv\,d_i}
\;+\; \underbrace{(\langle v\rangle_P - a)}_{\text{a constant, same for every }i\;\equiv\,b}.
$$

So $v_i - a = d_i + b$, where $b$ does **not** depend on $i$ (it's the gap between the phase's own mean and the reference $a$). Square with $(d_i+b)^2 = d_i^2 + 2bd_i + b^2$, multiply by $w_i^P$, and sum over the phase's cells:

$$
\sum_i w_i^P (v_i-a)^2
= \underbrace{\sum_i w_i^P d_i^2}_{\text{(I)}}
\;+\; \underbrace{2b\sum_i w_i^P d_i}_{\text{(II)}}
\;+\; \underbrace{b^2\sum_i w_i^P}_{\text{(III)}} .
$$

Now each piece on its own:

- **(I)** $= \displaystyle\sum_i w_i^P (v_i-\langle v\rangle_P)^2 = \sigma_{P,\rm own}^2$ — the phase's **internal** variance, by definition.
- **(II)** $= 0$. The mass-weighted deviations from the mean always cancel:
  $\displaystyle\sum_i w_i^P d_i = \sum_i w_i^P v_i - \langle v\rangle_P\!\sum_i w_i^P = \langle v\rangle_P - \langle v\rangle_P\cdot 1 = 0$.
  (That cancellation is *exactly* what makes $\langle v\rangle_P$ "the mean" — and it's why the cross term drops, just like the §0 proof.)
- **(III)** $= b^2\displaystyle\sum_i w_i^P = b^2\cdot 1 = (\langle v\rangle_P - a)^2$.

Add the three: $\;\sigma_{P,\rm own}^2 + 0 + (\langle v\rangle_P - a)^2$. $\blacksquare$

Setting $a = v_{\rm global}$ gives the relation stated in the docstring (`utils.py:142`):

$$
\boxed{\;\sigma_P^2 = \sigma_{P,\rm own}^2 + (\langle v\rangle_P - v_{\rm global})^2\;}
$$

**Interpretation.** A phase that *streams* relative to the bulk (large $|\langle v\rangle_P - v_{\rm global}|$) gets an inflated $\sigma_P$ that bundles its internal turbulent spread $\sigma_{P,\rm own}$ together with its bulk-offset, added in quadrature. Always $\sigma_P \ge \sigma_{P,\rm own}$, with equality iff the phase is at rest in the global frame. This is the physically meaningful line-of-sight broadening for a *single observed frame*, where you cannot separate a streaming phase's bulk motion from its internal turbulence.

> **Shift-invariance note (this is the one case where the reference is NOT the local mean).** $\sigma_P$ is still invariant to a uniform shift $v\to v+c$ — but *precisely because* $v_{\rm global}$ is itself a mass-weighted mean of all cells, so it co-shifts to $v_{\rm global}+c$. The deviation $(v_i - v_{\rm global})$ is then unchanged, and $\sigma_P$ is preserved (verified numerically: $9.0139$ before and after a $+100$ km/s shift). It would *not* be shift-invariant if measured about a *fixed constant* reference instead of a co-shifting mean.

### The 2026-06-29 change (this is the "recently changed" function)

- **Before:** each phase's $\sigma$ was about its *own* COM ($a = \langle v\rangle_P$), so the offset term was dropped and $\sigma_P = \sigma_{P,\rm own}$. That reported only internal dispersion and was blind to inter-phase streaming.
- **Now:** it subtracts the single global mean, *adding* the $(\langle v\rangle_P - v_{\rm global})^2$ contribution. Note the deliberate code asymmetry: line 160 centers `v_mean` on the phase's own mean (for *reporting*), while line 161 subtracts `v_global` (for the $\sigma$).

### Code walk-through (`utils.py:133-170`)

```
L148 masks       = classify_temperature_phase(T_K)     # 5 disjoint T-only masks
L149 total_mass  = rho.sum()
L150 total_cells = T_K.size
L153 v_global, _ = mass_weighted_sigma(vel_kms, rho)   # Step A — over ALL cells
L155 for phase, mask in masks.items():
L156   m_p = rho[mask];  tot = m_p.sum()                # phase mass
L158   if tot > 0 and v_global finite:
L159     w      = m_p / tot                             # LOCAL within-phase weights
L160     v_mean = sum(vel[mask] * w)                     # phase's OWN mean (reported)
L161     sigma  = sqrt(sum((vel[mask] - v_global)**2*w)) # about GLOBAL mean ← note!
L164   out[phase] = {v_mean, sigma,
                     mass_frac = tot/total_mass,
                     cell_frac = mask.sum()/total_cells}
```

Because masks are disjoint and exhaustive: $\sum_P \text{mass\_frac}_P = 1$ and $\sum_P \text{cell\_frac}_P = 1$.

### Worked numeric check

Reuse $v = [10,20,30]$, $\rho = [1,1,2]$, so $v_{\rm global} = 22.5$ (from §1). Take phase $A = \{\text{cell }0, \text{cell }1\}$:

- Local weights $w^A = [0.5, 0.5]$; own mean $\langle v\rangle_A = 15.0$.
- Own variance: $0.5(10-15)^2 + 0.5(20-15)^2 = 25.0$.
- Variance about global: $0.5(10-22.5)^2 + 0.5(20-22.5)^2 = 0.5(156.25)+0.5(6.25) = 81.25$.
- Decomposition check: $\sigma_{A,\rm own}^2 + (\langle v\rangle_A - v_{\rm global})^2 = 25.0 + (15.0-22.5)^2 = 25.0 + 56.25 = 81.25$ ✓.

(Confirmed numerically: `own_var, glob_var, decomp = 25.0, 81.25, 81.25`.) The *old* per-phase-COM form would have reported only $25.0$.

### Where it's used

Drives `Build_VelocityPhase` `phase_x/y/z` (`velocity_phase.py:76-78`) and the per-phase `sigma` stored into `pdf_fixed` (`velocity_phase.py:159`), which `Plot_PhaseSpectrumOverlay` then shows as $\sigma_x/\sigma_y/\sigma_z$ (`phase_spectrum_overlay.py:126,132`).

---

## 3. Total 3D dispersion

**Function:** `mass_weighted_sigma_3d` — `src/quokka2s/pipeline/utils.py:85-108`

This is the full 3D mass-weighted velocity dispersion of *all* gas: the RMS of the complete velocity-deviation **vector**, with each Cartesian component centered on its **own** global mean.

### Derivation

Per-component means: $\langle v_c\rangle = \sum_i w_i v_{c,i}$ for $c \in \{x,y,z\}$. Then

$$
\sigma_{\rm 3D}^2 = \sum_i w_i \Big[(v_{x,i}-\langle v_x\rangle)^2 + (v_{y,i}-\langle v_y\rangle)^2 + (v_{z,i}-\langle v_z\rangle)^2\Big].
$$

### Proof that $\sigma_{\rm 3D} = \sqrt{\sigma_x^2 + \sigma_y^2 + \sigma_z^2}$

The **same** weight $w_i$ multiplies all three bracketed terms, so the single sum over cells distributes over the three terms:

$$
\sigma_{\rm 3D}^2
= \sum_i w_i (v_{x,i}-\langle v_x\rangle)^2
+ \sum_i w_i (v_{y,i}-\langle v_y\rangle)^2
+ \sum_i w_i (v_{z,i}-\langle v_z\rangle)^2
= \sigma_x^2 + \sigma_y^2 + \sigma_z^2 ,
$$

where each $\sigma_c$ is exactly the 1D `mass_weighted_sigma` of §1. So

$$
\boxed{\;\sigma_{\rm 3D} = \sqrt{\sigma_x^2 + \sigma_y^2 + \sigma_z^2}\;}
$$

For **isotropic** turbulence $\sigma_x = \sigma_y = \sigma_z = \sigma_{\rm 1D}$, giving the familiar $\sigma_{\rm 3D} = \sqrt{3}\,\sigma_{\rm 1D}$.

### Why "average per-cell speed first, then take $\sigma$" is WRONG

A tempting shortcut: collapse each cell to a scalar speed $s_i = \sqrt{v_{x,i}^2 + v_{y,i}^2 + v_{z,i}^2}$, then take the 1D dispersion of $\{s_i\}$. This is wrong for two compounding reasons (the docstring at `utils.py:88-96` explicitly warns against it):

1. **Wrong centering.** The dispersion must subtract each component's *own* mean *before* squaring. Forming a per-cell speed first and subtracting $\langle s\rangle$ subtracts the wrong (post-norm, scalar) center — the bulk-flow removal that makes $\sigma$ a *dispersion* (not a mean kinetic energy) is destroyed. In general $\langle s\rangle \ne \sqrt{\langle v_x\rangle^2 + \langle v_y\rangle^2 + \langle v_z\rangle^2}$ (Jensen's inequality on the nonlinear norm).
2. **$\sqrt{\cdot}$ is nonlinear** and does not commute with the weighted sum: $\sum_i w_i \sqrt{(\cdots)} \ne \sqrt{\sum_i w_i (\cdots)}$. Norming each cell first and averaging the norms yields the *mean speed*, biased high by the bulk motion. Only the component **variances** add linearly (they are energy-like); the speeds do not.

### Code walk-through (`utils.py:85-108`)

```
L98  total = rho.sum()
L99  if total <= 0: return ((nan,nan,nan), nan)
L101 w   = rho / total
L102 vmx = sum(vx * w);  vmy = sum(vy * w);  vmz = sum(vz * w)    # 3 means
L105 var = sum((vx-vmx)**2*w) + sum((vy-vmy)**2*w) + sum((vz-vmz)**2*w)
L108 return (vmx,vmy,vmz), sqrt(var)                              # = σ_x²+σ_y²+σ_z²
```

The code computes one combined sum of three variances (each centered on its own component mean) rather than three separate $\sigma$'s — algebraically identical by the proof above.

### Worked numeric check

$\rho = [1,1,2] \Rightarrow w = [0.25,0.25,0.5]$.
- $v_x = [10,20,30] \Rightarrow \sigma_x = 8.29156$ (from §1).
- $v_y = [0,5,10]$: $\langle v_y\rangle = 6.25$; $\sigma_y^2 = 0.25(39.0625)+0.25(1.5625)+0.5(14.0625) = 17.1875 \Rightarrow \sigma_y = 4.14578$.
- $v_z = [-4,0,4]$: $\langle v_z\rangle = 1.0$; $\sigma_z^2 = 0.25(25)+0.25(1)+0.5(9) = 11.0 \Rightarrow \sigma_z = 3.31662$.
- Quadrature: $\sigma_{\rm 3D} = \sqrt{68.75 + 17.1875 + 11.0} = \sqrt{96.9375} = 9.84568$.

(Confirmed numerically: `sx,sy,sz = 8.29156, 4.14578, 3.31662; s3d = 9.845684`.)

### Where it's used

Called once in `Build_VelocityPhase` (`velocity_phase.py:89`), stored as `result['total_3d']['sigma_3d']` (`velocity_phase.py:90,178`), printed at `velocity_phase.py:102-103`. Not shown in the overlay plots.

---

## 4. Per-spaxel LOS moments — the stable one-pass form

**Function:** `spaxel_moments_along_axis` — `src/quokka2s/pipeline/utils.py:111-130`

This collapses a 3D cube along one line-of-sight axis to make **2D maps**: for each spaxel (pixel on the plane perpendicular to `los_axis`) it returns the weighted mean LOS velocity $V$, the dispersion $\sigma$, and the total weight $W$. The weight is whatever the caller passes (mass $\rho$ or a luminosity).

### Derivation — the raw-moment form

For one spaxel with weights $w_k > 0$ and LOS velocities $v_k$ along the collapsed axis, accumulate three raw moments:

$$
W = \sum_k w_k, \qquad WV = \sum_k w_k v_k, \qquad WV2 = \sum_k w_k v_k^2 ,
$$

then

$$
V = \frac{WV}{W} = \langle v\rangle, \qquad
\text{var} = \frac{WV2}{W} - V^2 = \langle v^2\rangle - \langle v\rangle^2, \qquad
\boxed{\;\sigma = \sqrt{\max(\text{var}, 0)}\;}
$$

This $\langle v^2\rangle - \langle v\rangle^2$ equals the centered second moment $\sum_k w_k (v_k - \langle v\rangle)^2 / W$ — exactly the identity proved in §0.

### Two numerical subtleties

**Why the raw-moment (one-pass) form here?** The centered form needs $\langle v\rangle$ first, so it requires **two passes** over the LOS axis (one for $V$, one for the deviations). The raw-moment form accumulates $W$, $WV$, $WV2$ in a **single** vectorized pass and then combines. These are full 3D cubes (e.g. $256\times2048\times\cdots$ cells) collapsed to a 2D map; a second pass would double the memory traffic. So the codebase trades a little numerical robustness for speed *here*, and pays for it with a clamp (next point). Contrast the mass-weighted family (§1–3), which uses the centered two-pass form and never needs a clamp.

**Why the `max(var, 0)` clamp?** $\langle v^2\rangle - \langle v\rangle^2$ is the difference of two large nearly-equal positive numbers (catastrophic cancellation). For a cold/coherent spaxel where $v$ is nearly constant, the true variance $\approx 0$, and float64 rounding (ULP drift) can make `WV2/W - V*V` come out slightly negative (e.g. $-10^{-12}$). Then $\sqrt{\text{negative}} \to$ NaN. `np.maximum(var, 0.0)` (`utils.py:129`) pins that ULP-level drift to exactly 0, so $\sigma = 0$ — the physically correct width for a single-velocity spaxel.

### Code walk-through (`utils.py:111-130`)

```
L123 W   = weight.sum(axis=los_axis)                 # 0th moment per spaxel
L124 WV  = (weight*vel).sum(axis=los_axis)           # 1st raw moment
L125 WV2 = (weight*vel*vel).sum(axis=los_axis)       # 2nd raw moment
L126 with np.errstate(invalid='ignore', divide='ignore'):  # W==0 spaxels → NaN, no warn
L127   V     = WV / W
L128   var   = WV2/W - V*V                            # ⟨v²⟩ - ⟨v⟩²
L129   sigma = sqrt(maximum(var, 0.0))                # clamp ULP drift
L130 return V, sigma, W
```

$V$ and $\sigma$ are NaN exactly where $W = 0$ (empty sightlines).

### Frame caveat

This $\sigma$ is a pure **width about each spaxel's own** weighted centroid $V$ — it subtracts the *local* mean implicitly via $\langle v^2\rangle - \langle v\rangle^2$. It does **not** subtract a global/COM velocity, and each spaxel is independent. So mass-weighted-collapsing the whole $\sigma$ map does **not** recover the global `mass_weighted_sigma`: the per-spaxel version excludes spaxel-to-spaxel bulk variation (it is intrinsic per-sightline dispersion).

### Status

Defined in `utils.py` but currently has **no active (non-archive) caller** — only archived tasks (`archive/spaxel_sigma.py`, `archive/sigma_sfr_overlay.py`) use it. It is the only LOS-collapse moment method in the codebase; kept for reference.

---

## 5. Thermal line width $\sigma_{\rm th} = \sqrt{k_B T / m}$

**Functions:** `_H_atom_thermal_width` — `physics_fields.py:780-790` (H lines, $m = 1.00794$ amu); `_make_thermal_width_field` — `physics_fields.py:924-945` (factory; live only for CO and C$^+$ — see scope note below).

This is a **completely different kind of $\sigma$** from §1–4: it is not a spread of the gas velocity field, but the per-cell **intrinsic broadening kernel** of a spectral line. It subtracts no mean.

### Derivation from the Maxwellian

The 1D Maxwell–Boltzmann distribution for one Cartesian (line-of-sight) velocity component is

$$
f(v)\,dv = \left(\frac{m}{2\pi k_B T}\right)^{1/2} \exp\!\left(-\frac{m v^2}{2 k_B T}\right) dv .
$$

This is a zero-mean Gaussian $f(v) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp(-v^2/2\sigma^2)$. Matching the exponents,

$$
\frac{m v^2}{2 k_B T} \equiv \frac{v^2}{2\sigma^2}
\;\Longrightarrow\;
\sigma^2 = \frac{k_B T}{m}
\;\Longrightarrow\;
\boxed{\;\sigma_{\rm th} = \sqrt{\frac{k_B T}{m}}\;}
$$

This is the **one-component** (line-of-sight) Doppler width — the relevant one for a spectral line along a single axis. (The full 3D *speed* distribution has $\langle |v|^2\rangle = 3k_BT/m$; per component it is $k_BT/m$.) There is **no** $\sqrt{2}$ microturbulence term and no extra non-thermal width added here.

### Conventions — what this $\sigma$ is NOT

This is the **1-$\sigma$ Gaussian width**, *not* the FWHM and *not* the $b$-parameter:

$$
\text{FWHM} = 2\sqrt{2\ln 2}\;\sigma_{\rm th} \approx 2.3548\,\sigma_{\rm th},
\qquad
b = \sqrt{2}\,\sigma_{\rm th}.
$$

### Temperature and mass

- The temperature is `temperature_two_regime` (`physics_fields.py:788, 940`), *not* `T_DESPOTIC`. This was switched on 2026-06-18 so hot-gas widths aren't underestimated: DESPOTIC saturates at $\sim5\times10^4$ K, which would shrink $\sigma_{\rm th}$ by up to $\sqrt{20}$.
- Per-emitter masses: H $= 1.00794$ amu (used for **both** H$\alpha$ and HI 21 cm), CO $= 28.01$, C$^+ = 12.01$ amu.
- An HCO$^+ = 29.02$ amu entry survives in the `species_masses` dict (`physics_fields.py:930`) but is **dead code**: HCO$^+$ was dropped from the analysed emitters on 2026-06-23 (`EMITTERS_FREQ_WIDTH = ['CO', 'C+']`, `physics_fields.py:1043`), so no `HCO+_thermal_width` field is ever registered. The factory is live only for CO and C$^+$.
- Heavier emitter $\Rightarrow$ narrower width, since $\sigma_{\rm th} \propto m^{-1/2}$.

### Code (`physics_fields.py:788-789, 940-941`)

```
T       = data[('gas','temperature_two_regime')].to('K')   # per-cell unified T
sigma_v = sqrt((kb * T) / m).to('cm/s')                     # kb, m carry units
```

`kb` and `m` (an amu quantity) carry units via astropy/yt; the unit algebra $(\text{erg/K}\cdot\text{K}/\text{g})^{1/2} = \text{cm/s}$, and the `.to('cm/s')` both converts and **asserts** the unit (guarding against a $k_B/m$ unit slip — the no-hardcoded-constants rule).

### Verifiable checks

- **Mass scaling:** at fixed $T$, $\sigma_{\rm th}(\text{H})/\sigma_{\rm th}(\text{CO}) = \sqrt{28.01/1.00794} \approx 5.27$ — H lines are $\sim5\times$ broader thermally than CO. (Confirmed: `5.2716`.) This is *why* CO/C$^+$ spectra show a narrow spike.
- $\sigma_{\rm th}(\text{C}^+)/\sigma_{\rm th}(\text{CO}) = \sqrt{28.01/12.01} \approx 1.527$. (Confirmed: `1.5272`.)
- The H field is registered for both lines (`physics_fields.py:1108-1110` H$\alpha$, where `function=_H_atom_thermal_width` is at `:1109`; `:1118-1120` HI, function at `:1119`), differing only by $T$ (same mass).

### How it enters the spectrum — `build_spectral_cube`

**Function:** `build_spectral_cube` — `physics_fields.py:519-591` (V3).

The per-cell $\sigma_{\rm th}$ (here `thermal_val`, in cm/s) is converted to a *frequency* width and integrated over each channel:

$$
\sigma_\nu = \nu_{\rm gas}\,\frac{\sigma_v}{c},
\qquad
x = \frac{\nu_{\rm edge} - \nu_{\rm gas}}{\sqrt{2}\,\sigma_\nu},
\qquad
\text{bin\_frac}_k = \tfrac12\big[\mathrm{erf}(x_{\rm hi}) - \mathrm{erf}(x_{\rm lo})\big].
$$

```
L578 sigma_v   = maximum(thermal_val[i], 1.0)        # floor at 1 cm/s (no div-by-0)
L579 sigma_nu  = nu_gas * (sigma_v / c_cms)           # velocity width → freq width
L580 sqrt2_sigma = sqrt(2) * sigma_nu
L585 edges       = freq_edges_hz[ch0:ch1+1]           # channel-chunk edge slice
L586 x_edges     = (edges - nu_gas)/sqrt2_sigma        # at each bin edge
L587 erf_at_edges = erf(x_edges)                      # each edge erf'd ONCE (V3)
L588 bin_frac     = 0.5*(erf_at_edges[1:] - erf_at_edges[:-1])
L589 spec_cube[ch0:ch1] += lum_gas * bin_frac / delta_nu_bin   # erg/s/Hz
```

Key properties:
- **Flux conservation:** $\sum_k \text{bin\_frac}_k = 1$ per cell (a Gaussian integrates to 1 over $(-\infty,\infty)$ and the channel grid is wide).
- **Analytic erf integration**, not center-sampling, so even under-resolved lines conserve flux.
- The $\sigma_v$ floor at 1 cm/s (`L578`) guards `thermal_val == 0`; negligible vs real ($\sim$km/s) widths.
- V3 is documented **bitwise-identical** to the legacy `_build_spectral_cube_v0_legacy` (`physics_fields.py:477-516`); `sqrt2_sigma` is computed in the legacy multiply order to preserve ULP equality, and V3 just avoids double-`erf`-ing interior edges ($\sim1.8\times$ faster).

> **Important:** there is **no** explicit per-cell turbulent/non-thermal width added in the cube — only this thermal $\sqrt{k_BT/m}$ width plus the bulk Doppler shift (`Bulk_Doppler_factor_*`). The cube widths read by `SpectrumStore._ensure_primitives` (`spectrum_service.py:177-178`) are exactly these `*_thermal_width` fields.

---

## 6. Instrumental LSF width $\sigma_{\rm LSF} = (c/R)/2.355$

**Function:** `apply_spectral_lsf` — `src/quokka2s/pipeline/utils.py:13-23` (with `C_KMS` at `utils.py:10`).

This convolves the spectrum with a Gaussian instrumental line-spread function of resolving power $R$, producing the "observed" spectrum. Again a **pure width**, no mean subtracted.

### Derivation

**Resolving power.** $R \equiv \lambda/\Delta\lambda$, where $\Delta\lambda$ is the FWHM of the instrument's response at wavelength $\lambda$. Via the non-relativistic Doppler relation $\Delta\lambda/\lambda = \Delta v/c$ (valid for ISM velocities $\ll c$),

$$
\Delta v_{\rm FWHM} = c\,\frac{\Delta\lambda}{\lambda} = \frac{c}{R}.
$$

**FWHM $\leftrightarrow \sigma$.** For $g(x) \propto \exp(-x^2/2\sigma^2)$, the half-maximum is where $\exp(-x^2/2\sigma^2) = \tfrac12$, i.e. $x = \sigma\sqrt{2\ln 2}$; the full width is $\text{FWHM} = 2\sigma\sqrt{2\ln 2}$, so

$$
\text{FWHM} = 2\sqrt{2\ln 2}\;\sigma \approx 2.3548\,\sigma .
$$

**Combine:**

$$
\boxed{\;\sigma_{\rm LSF} = \frac{\Delta v_{\rm FWHM}}{2\sqrt{2\ln 2}} = \frac{c/R}{2.355}\ \text{[km/s]}\;}
\qquad
\sigma_{\rm channels} = \frac{\sigma_{\rm LSF}}{\Delta v_{\rm chan}} .
$$

### Code (`utils.py:21-23`)

```
L21 sigma_kms      = (C_KMS / R) / 2.355              # FWHM=c/R, then /2.3548 → 1σ
L22 sigma_channels = sigma_kms / dv_per_channel_kms   # km/s → channel units for scipy
L23 return gaussian_filter1d(spec, sigma=sigma_channels, axis=axis, mode='nearest')
```

### Checks and notes

- $2\sqrt{2\ln 2} = 2.354820045\ldots \approx 2.355$ (the hard-coded constant; sub-0.01% rounding). (Confirmed: `2.3548200450`.)
- `C_KMS = 299792.458` km/s is a literal at `utils.py:10` (not derived from astropy — a minor deviation from the no-hardcoded-constants rule, though it is the *exact defined* value of $c$).
- `scipy.gaussian_filter1d` normalises its kernel ($\int\text{kernel} = 1$), so the convolution **conserves total flux/luminosity** — it only redistributes.
- Larger $R \Rightarrow$ smaller $\sigma_{\rm LSF} \Rightarrow$ less smoothing. $R = \infty$ is bypassed by callers (`get_spectrum` only convolves if $R$ finite and $>0$). At $R = 10^4$, with the current grid (`integrated_spectrum.py:75-76`: `N_CHANNELS = 300`, `V_RANGE_KMS = ±50` → a $100$ km/s window, $\Delta v_{\rm chan} = 100/300 = 0.3333$ km/s/channel), $\sigma_{\rm LSF} = (c/R)/2.355 = (299792.458/10^4)/2.355 = 12.73$ km/s $= 12.73/0.3333 \approx 38$ channels (of $300$). (Confirmed: `38.19` channels.) Even at $\sim38/300$, this heavily smears all fine structure. (A stale source comment previously read "≈127ch"; that figure corresponds to a $\Delta v_{\rm chan}\approx0.100$ km/s/channel grid, i.e. a $\pm15$ km/s window over 300 channels — not the current $\pm50$ km/s config. The comment at `integrated_spectrum.py:75` has been corrected to "≈38ch".)
- **Convolution-of-Gaussians variance-add rule:** because the LSF is applied *after* the cube is built, the observed line variance is $\sigma_{\rm obs}^2 = \sigma_{\rm intrinsic}^2 + \sigma_{\rm LSF}^2$. This is the only instrumental $\sigma$, and it is independent of the per-cell thermal width.

Called from `SpectrumStore.get_spectrum` (`spectrum_service.py:105`) and `Build_SpeciesSpectrum` (`species_spectrum.py:145`) to produce the LSF-broadened `dsigma_dv_obs` from the intrinsic spectrum.

---

## 7. Observed spectral 2nd-moment $\sigma_{\rm obs}$ (and the Gaussian-fit width)

**Function:** `_moment_sigma` — three functionally identical copies: `integrated_spectrum.py:50-58`, `species_spectrum.py:57-64`, `phase_spectrum_overlay.py:42-49`. (They are *mathematically* identical and give the same number for the same input; they are not *byte*-identical — `phase_spectrum_overlay.py` returns `float('nan'), float('nan')` in its zero-total guard where the other two return `np.nan, np.nan`, and only `integrated_spectrum.py` carries a docstring.)

This is the velocity dispersion of an **already-built 1D spectrum** $S = d\Sigma/dv$ via its **intensity- (luminosity-) weighted** second moment. It is the "observed" line width $\sigma_{\rm obs}$ annotated on the spectrum figures.

### Derivation

The abscissa is the discretized velocity axis (channels) $v_k$; the weights are the spectrum amplitude per channel $S_k = (d\Sigma/dv)_k \ge 0$:

$$
w_k = \frac{S_k}{\sum_j S_j},
\qquad
\langle v\rangle = \sum_k w_k v_k,
\qquad
\boxed{\;\sigma_{\rm obs} = \sqrt{\sum_k w_k\,(v_k - \langle v\rangle)^2}\;}
$$

This is exactly the same centered second-moment formula as `mass_weighted_sigma` (§1) — but with two substitutions: mass $\to$ **luminosity**, and 3D cells $\to$ **velocity channels**. It is the spectral analogue of the gas dispersion.

### Code (`phase_spectrum_overlay.py:42-49`, identical in the other two)

```
L43 total  = spec.sum()                                  # total line intensity
L44 if total <= 0: return (nan, nan)
L46 w      = spec / total                                # luminosity-fraction per channel
L47 v_mean = sum(v * w)                                  # intensity-weighted centroid
L48 sigma  = sqrt(sum((v - v_mean)**2 * w))              # centered 2nd moment
L49 return v_mean, sigma
```

### Frame and interpretation

- **COM-relative width.** It subtracts the *luminosity-weighted centroid* $\langle v\rangle$ of the line profile, so it is a width about the line's own emission centroid.
- **Simulation rest frame.** The velocity axis is built from `shifted_freq = ν₀·(1 − v_los/c)` (`Bulk_Doppler_factor_*`, `physics_fields.py:947-974`) and `v_axis = c(ν₀−ν)/ν₀` — i.e. $v=0 \Leftrightarrow$ rest frequency, with **no** COM subtraction on the axis. The centroid $\langle v\rangle$ may sit slightly off 0 if the emission-weighted bulk velocity is nonzero.
- **Not a pure gas kinematic dispersion.** It is computed on the *observed* (LSF-convolved) spectrum where available (`phase_spectrum_overlay.py:113` prefers `dsigma_dv_obs`), so $\sigma_{\rm obs}$ includes **thermal + bulk-Doppler + instrumental** broadening. Contrast: the $\sigma_x/\sigma_y/\sigma_z$ "gas" markers on the same plots come from `mass_weighted_sigma_by_phase` / `mass_weighted_sigma` (mass-weighted, intrinsic). So $\sigma_{\rm obs}$ and $\sigma_{\rm gas}$ deliberately differ.

### The companion: Gaussian-fit width

**Function:** `_fit_gaussian` — `integrated_spectrum.py:23-47`, `species_spectrum.py:37-54` (two identical copies).

A single Gaussian $S(v) \approx A\exp\!\big[-\tfrac12((v-v_0)/\sigma)^2\big]$ is fitted to the peak-normalised spectrum via `scipy.curve_fit` (bounds: $\sigma \in [0.1, 100]$ km/s, $v_0 \in [-50, 50]$ km/s, `maxfev=10000`), returning $|\sigma|$ (absolute value, since a Gaussian is symmetric in the sign of $\sigma$). This is a **model-fit** width, not a moment.

**Cross-checks between the three width measures (all shown on the same panel):**
- For a clean single-Gaussian line, $\sigma_{\rm obs} \approx \sigma_{\rm fit} \approx \sigma_{\rm gas}$.
- $\sigma_{\rm obs} > \sigma_{\rm fit}$ when the profile has heavy wings (the moment captures wings; the fit captures the core).
- A fit pinned at a bound flags a non-Gaussian / clipped profile.
- The three functionally identical `_moment_sigma` copies (and two `_fit_gaussian` copies) must give the *same* result for the same input — a built-in consistency check.

Displayed: `phase_spectrum_overlay.py:118-119,151` ("X emission $\sigma_v=$"); `species_spectrum.py:187-194` (Gaussian + Moment); `integrated_spectrum.py:168-177`.

---

## 8. Summary table

| # | $\sigma$ name | File:line | Formula | Mean subtracted | Weight | What it measures |
|---|---------------|-----------|---------|-----------------|--------|------------------|
| 1 | `mass_weighted_sigma` (1D) | `utils.py:74-82` | $\sqrt{\sum_i w_i(v_i-\langle v\rangle)^2}$ | **own** global mass-weighted mean $\langle v\rangle$ | mass $\rho$ | spread of one velocity component over all gas |
| 2 | `mass_weighted_sigma_by_phase` | `utils.py:133-170` | $\sqrt{\sum_{i\in P} w_i^P(v_i-v_{\rm global})^2}$ | the **GLOBAL** all-phase mean $v_{\rm global}$ | mass $\rho$ (local per phase) | per-phase spread incl. bulk streaming: $\sigma_P^2=\sigma_{\rm own}^2+(\langle v\rangle_P-v_{\rm global})^2$ |
| 3 | `mass_weighted_sigma_3d` | `utils.py:85-108` | $\sqrt{\sigma_x^2+\sigma_y^2+\sigma_z^2}$ | each component's **own** mean | mass $\rho$ | total 3D dispersion of all gas |
| 4 | `spaxel_moments_along_axis` | `utils.py:111-130` | $\sqrt{\max(\langle v^2\rangle-\langle v\rangle^2,0)}$ | each spaxel's **own local** mean $V$ | caller's (mass or lum.) | per-sightline LOS width map (one-pass, ULP-clamped) |
| 5 | thermal width $\sigma_{\rm th}$ | `physics_fields.py:780-790`, `924-945` | $\sqrt{k_BT/m}$ | **none** (pure width) | — (per cell) | intrinsic 1D thermal Doppler line broadening |
| 6 | LSF width $\sigma_{\rm LSF}$ | `utils.py:13-23` | $(c/R)/2.355$ | **none** (pure width) | — | instrumental Gaussian smearing |
| 7 | `_moment_sigma` $\sigma_{\rm obs}$ | `integrated_spectrum.py:50-58` (+2 copies) | $\sqrt{\sum_k w_k(v_k-\langle v\rangle)^2}$ | line's own luminosity-weighted centroid | luminosity $S_k$ | observed line width (thermal + bulk + LSF) |
| — | `_fit_gaussian` $\sigma_{\rm fit}$ | `integrated_spectrum.py:23-47` (+1 copy) | single-Gaussian fit | fitted centroid $v_0$ | (fit to $S_k$) | model-fit core line width |

*Note: the three `_moment_sigma` copies are functionally identical (same number for the same input) but not byte-identical — their zero-total guard return type and docstrings differ (see §7).*

### Mental model in one paragraph

The mass-weighted family (1–3) measures the **gas velocity field**: how fast different parcels move relative to a common mean — bulk + turbulent kinematics. #4 does the same but per sightline, about each pixel's own centroid, with a fast one-pass numerical route. The thermal (5) and LSF (6) widths are **broadening kernels** — they have no mean to subtract; they just smear each line. Finally the observed spectral $\sigma$ (7) is what an instrument records: the gas kinematics convolved with thermal and instrumental broadening, measured as the second moment of the *emitted* line. That is why $\sigma_{\rm obs}$ (luminosity-weighted, line-of-sight, LSF-broadened) and $\sigma_{\rm gas}$ (mass-weighted, intrinsic) deliberately differ on the plots — they answer different questions.

---

*All formulas above were checked against the source at the cited file:line, and every worked numeric example was confirmed in NumPy. Line numbers are a 2026-06-29 snapshot; re-grep if the source drifts. Key relevant files: `/Users/baochen/quokka_postprocessing/src/quokka2s/pipeline/utils.py`, `/Users/baochen/quokka_postprocessing/src/quokka2s/pipeline/prep/physics_fields.py`, and the spectrum tasks under `/Users/baochen/quokka_postprocessing/src/quokka2s/pipeline/tasks/`.*