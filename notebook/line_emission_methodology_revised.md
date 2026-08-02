# Species-dependent Line-emission Calculations

The simulation provides the hydrodynamic gas temperature
$T_{\rm QUOKKA}$, while the pre-computed DESPOTIC table provides a
thermal-equilibrium temperature $T_{\rm DESPOTIC}$. These temperatures are not
combined into a single temperature field for every emitting species. Instead,
we adopt a species-dependent prescription for the line-emission calculations.

We define two boundaries using the QUOKKA temperature,

$$
T_0 = 3000\,{\rm K},
\qquad
T_1 = 1.307\times10^4\,{\rm K}.
$$

The first boundary, $T_0$, separates the DESPOTIC and analytic treatments for
hydrogen and C$^+$. The second boundary, $T_1$, applies only to the carbon
ionization calculation. It does not define an additional hydrogen regime.
We adopt $T_1=0.1\,\chi_{\rm C^0}/k_{\rm B}$, where
$\chi_{\rm C^0}=11.2603\,{\rm eV}$ is the first ionization energy of carbon.

For the hydrogen lines, the adopted temperature is

$$
T_{\rm H} =
\begin{cases}
T_{\rm DESPOTIC}, & T_{\rm QUOKKA}<T_0,\\[3pt]
T_{\rm QUOKKA},   & T_{\rm QUOKKA}\geq T_0.
\end{cases}
$$

CO uses $T_{\rm DESPOTIC}$ throughout. All explicit per-cell temperature
consumers for C$^+$, including the analytic ionization and excitation
calculations, thermal broadening, and phase diagrams, use $T_{\rm QUOKKA}$.
For $T_{\rm QUOKKA}<T_0$, however, the [C II] emissivity itself is read directly
from the DESPOTIC table and therefore does not require a separate runtime
temperature input.

## Hydrogen Ionization State

### Low-temperature cells

For cells with $T_{\rm QUOKKA}<T_0$, the electron, proton, and neutral-hydrogen
number densities are obtained from the pre-computed DESPOTIC lookup table:

$$
n_e=n_e^{\rm DESPOTIC},
\qquad
n_{\rm H^+}=n_{\rm H^+}^{\rm DESPOTIC},
\qquad
n_{\rm HI}=n_{\rm HI}^{\rm DESPOTIC}.
$$

The table is interpolated in the local hydrogen-nuclei number density
$n_{\rm H}$, hydrogen column density $N_{\rm H}$, and velocity gradient $dV/dr$.
The gas temperature used for the hydrogen-line calculation is
$T_{\rm DESPOTIC}$.

### Cells at and above 3000 K

For cells with $T_{\rm QUOKKA}\geq T_0$, the total electron fraction is inferred
from the QUOKKA thermodynamic state. The internal-energy density satisfies

$$
e_{\rm int}
=
\frac{1}{\gamma-1}
\frac{\rho}{\mu m_{\rm H}}
k_{\rm B}T_{\rm QUOKKA},
$$

and hence

$$
\frac{1}{\mu}
=
\frac{(\gamma-1)m_{\rm H}e_{\rm int}}
{\rho k_{\rm B}T_{\rm QUOKKA}}.
$$

The total hydrogen-nuclei number density is

$$
n_{\rm H}=\frac{X\rho}{m_{\rm H}},
$$

where $X=1/(1+0.1\times3.971)$ is the hydrogen mass fraction. We define the total free-electron
fraction as

$$
x_e\equiv\frac{n_e}{n_{\rm H}}.
$$

The working QUOKKA EOS inversion counts hydrogen and helium nuclei together
with all free electrons as

$$
\frac{1}{\mu}
=
X+\frac{Y}{4}+Xx_e,
$$

where the composition still uses $Z=0.02$ and $Y=1-X-Z$, but no separate
metal-nucleus term is included in this inversion. Therefore,

$$
x_e
=
\frac{\mu^{-1}-X-Y/4}{X}.
$$

The inferred electron fraction is restricted to

$$
0\leq x_e\leq 1+\frac{Y}{2X}.
$$

The upper limit corresponds to fully ionized hydrogen and helium for the
adopted composition. Importantly, $x_e$ describes all free electrons and is not
in general identical to the hydrogen ionization fraction.

For $T_{\rm QUOKKA}\geq T_0$, we assume that hydrogen is partitioned between
H I and H$^+$ and that its molecular fraction is negligible. We set

$$
x_{\rm H^+}
\equiv
\frac{n_{\rm H^+}}{n_{\rm H}}
=
\min(x_e,1).
$$

The electron, proton, and neutral-hydrogen number densities are then

$$
n_e=x_en_{\rm H},
$$

$$
n_{\rm H^+}=\min(x_e,1)n_{\rm H},
$$

and

$$
\begin{aligned}
n_{\rm HI}
&=n_{\rm H}-n_{\rm H^+}\\
&=\left[1-\min(x_e,1)\right]n_{\rm H}\\
&=\max(1-x_e,0)n_{\rm H}.
\end{aligned}
$$

When $x_e\leq1$, the free electrons are attributed to hydrogen ionization and
$x_{\rm H^+}=x_e$. When $x_e>1$, hydrogen is assumed to be fully ionized and
the additional electrons are attributed to helium and metals. We do not use a
hydrogen Saha calculation or a CHIANTI collisional-ionization-equilibrium
fraction in either case.

Combining the two temperature ranges, the adopted hydrogen densities can be
written compactly as

$$
n_e=
\begin{cases}
n_e^{\rm DESPOTIC},
& T_{\rm QUOKKA}<T_0,\\[3pt]
x_en_{\rm H},
& T_{\rm QUOKKA}\geq T_0,
\end{cases}
$$

$$
n_{\rm H^+}=
\begin{cases}
n_{\rm H^+}^{\rm DESPOTIC},
& T_{\rm QUOKKA}<T_0,\\[3pt]
\min(x_e,1)n_{\rm H},
& T_{\rm QUOKKA}\geq T_0,
\end{cases}
$$

and


$$
n_{\rm HI}=
\begin{cases}
n_{\rm HI}^{\rm DESPOTIC},
& T_{\rm QUOKKA}<T_0,\\[3pt]
\max(1-x_e,0)n_{\rm H},
& T_{\rm QUOKKA}\geq T_0.
\end{cases}
$$

## H$\alpha$ Emission

H$\alpha$ is produced by the $n=3\rightarrow2$ transition of hydrogen and
traces ionized gas. We assume Case-B recombination, under which photons
produced by recombinations directly to the ground state are immediately
reabsorbed.

The hydrogen recombination rate in a simulation cell is

$$
\dot N_{\rm rec}
=
\alpha_B(T_{\rm H})
n_e n_{\rm H^+}V_{\rm cell},
$$

where $V_{\rm cell}$ is the cell volume. We use the Case-B recombination
coefficient

$$
\alpha_B(T)
=
2.54\times10^{-13}
T_4^{-0.8163-0.0208\ln T_4}
\ {\rm cm^3\,s^{-1}},
$$

where

$$
T_4\equiv\frac{T}{10^4\,{\rm K}}.
$$

Approximately $45\%$ of Case-B recombinations produce an H$\alpha$ photon.
The volumetric emissivity and cell luminosity are therefore

$$
\epsilon_{\rm H\alpha}
=
0.45\,h\nu_{\rm H\alpha}
\alpha_B(T_{\rm H})
n_e n_{\rm H^+},
$$

and

$$
L_{\rm H\alpha,cell}
=
\epsilon_{\rm H\alpha}V_{\rm cell}.
$$

For $T_{\rm QUOKKA}\geq T_0$, this becomes

$$
\epsilon_{\rm H\alpha}
=
0.45\,h\nu_{\rm H\alpha}
\alpha_B(T_{\rm QUOKKA})
x_e\min(x_e,1)n_{\rm H}^2.
$$

Thus, the H$\alpha$ emissivity is proportional to $x_e^2n_{\rm H}^2$ when
$x_e\leq1$, and to $x_en_{\rm H}^2$ when $x_e>1$. The latter case includes the
contribution of non-hydrogen electrons to the recombination rate while limiting
the proton abundance to the total hydrogen abundance.

## H I 21-cm Emission

The H I 21-cm line is produced by the hyperfine transition from the $F=1$
upper level to the $F=0$ lower level of neutral atomic hydrogen. The adopted
transition frequency and spontaneous-emission coefficient are

$$
\nu_{21}
=
1.420405751768\times10^9\,{\rm Hz},
$$

and

$$
A_{10}=2.85\times10^{-15}\,{\rm s^{-1}}.
$$

The statistical weights of the upper and lower levels are $g_1=3$ and $g_0=1$.
The energy separation expressed as a temperature is

$$
T_\ast
\equiv
\frac{h\nu_{21}}{k_{\rm B}}
\simeq0.0682\,{\rm K}.
$$

Under the Boltzmann distribution, the upper-level fraction is

$$
\frac{n_1}{n_{\rm HI}}
=
\frac{g_1\exp(-T_\ast/T_{\rm spin})}
{g_0+g_1\exp(-T_\ast/T_{\rm spin})}.
$$

For $T_{\rm spin}\gg T_\ast$, which is satisfied for the gas considered here,

$$
\frac{n_1}{n_{\rm HI}}
\simeq
\frac{g_1}{g_0+g_1}
=
\frac{3}{4}.
$$

Assuming optically thin emission, the volumetric emissivity is

$$
\epsilon_{21}
=
n_1A_{10}h\nu_{21}
=
\frac{3}{4}n_{\rm HI}A_{10}h\nu_{21},
$$

and the luminosity of one simulation cell is

$$
L_{21,\rm cell}
=
\epsilon_{21}V_{\rm cell}.
$$

The neutral-hydrogen density in this expression follows the piecewise
prescription above. In particular, for $T_{\rm QUOKKA}\geq T_0$,

$$
\epsilon_{21}
=
\frac{3}{4}
\max(1-x_e,0)n_{\rm H}
A_{10}h\nu_{21}.
$$

Consequently, cells with $x_e>1$ have $n_{\rm HI}=0$ and produce no H I 21-cm
emission in this model.

## Carbon Ionization State

The [C II] calculation uses three temperature ranges defined by
$T_{\rm QUOKKA}$. For $T_{\rm QUOKKA}<T_0$, the emissivity is obtained directly
from the DESPOTIC table, so an explicit analytic C$^+$ fraction is not required.
For $T_{\rm QUOKKA}\geq T_0$, the electron density is

$$
n_e=x_en_{\rm H},
$$

where $x_e$ is inferred from the mean molecular weight as described above.

We define the Saha constants for the two adjacent carbon ionization stages as

$$
S_1(T)
=
2\left(\frac{2\pi m_e k_{\rm B}T}{h^2}\right)^{3/2}
\frac{U_{\rm C^+}(T)}{U_{\rm C^0}(T)}
\exp\left(-\frac{\chi_{\rm C^0}}{k_{\rm B}T}\right),
$$

and

$$
S_2(T)
=
2\left(\frac{2\pi m_e k_{\rm B}T}{h^2}\right)^{3/2}
\frac{U_{\rm C^{++}}(T)}{U_{\rm C^+}(T)}
\exp\left(-\frac{\chi_{\rm C^+}}{k_{\rm B}T}\right).
$$

Here $U_i(T)$ is the partition function of ionization stage $i$, and

$$
\chi_{\rm C^0}=11.2603\,{\rm eV},
\qquad
\chi_{\rm C^+}=24.3833\,{\rm eV}.
$$

The partition functions and ionization energies are obtained from CHIANTI
atomic data. CHIANTI supplies these atomic quantities, but it does not supply
the carbon ion fraction used in the current calculation.

### Intermediate-temperature carbon

For $T_0\leq T_{\rm QUOKKA}<T_1$, we retain the three ionization stages

$$
{\rm C^0}\rightleftharpoons{\rm C^+}
\rightleftharpoons{\rm C^{++}}.
$$

Defining

$$
r_1
\equiv
\frac{S_1(T_{\rm QUOKKA})}{n_e}
=
\frac{n_{\rm C^+}}{n_{\rm C^0}},
$$

and

$$
r_2
\equiv
\frac{S_2(T_{\rm QUOKKA})}{n_e}
=
\frac{n_{\rm C^{++}}}{n_{\rm C^+}},
$$

the carbon conservation equation is

$$
n_{\rm C}
=
n_{\rm C^0}+n_{\rm C^+}+n_{\rm C^{++}}.
$$

The C$^+$ fraction is therefore

$$
x_{\rm C^+}
\equiv
\frac{n_{\rm C^+}}{n_{\rm C}}
=
\frac{r_1}{1+r_1+r_1r_2}.
$$

### High-temperature carbon

For $T_{\rm QUOKKA}\geq T_1$, we neglect neutral carbon and assume

$$
n_{\rm C}=n_{\rm C^+}+n_{\rm C^{++}}.
$$

The second Saha equilibrium gives

$$
\frac{n_{\rm C^{++}}}{n_{\rm C^+}}
=
\frac{S_2(T_{\rm QUOKKA})}{n_e}.
$$

Combining this relation with carbon conservation gives

$$
\begin{aligned}
x_{\rm C^+}
&=\frac{n_{\rm C^+}}{n_{\rm C}}\\
&=\frac{1}{1+S_2(T_{\rm QUOKKA})/n_e}\\
&=\frac{n_e}{n_e+S_2(T_{\rm QUOKKA})}.
\end{aligned}
$$

Thus, the high-temperature C$^+$ fraction depends on both temperature and the
cell electron density. It is not a density-independent CHIANTI CIE fraction.

## [C II] 158-$\mu$m Emission

The [C II] $158\,\mu{\rm m}$ line is produced by the fine-structure transition
from the upper level $u={}^{2}P_{3/2}$ to the lower level
$l={}^{2}P_{1/2}$. The adopted atomic constants are

$$
A_{ul}=2.290\times10^{-6}\,{\rm s^{-1}},
$$

$$
\nu_{ul}=1.900594\times10^{12}\,{\rm Hz},
$$

and

$$
\frac{\Delta E_{ul}}{k_{\rm B}}=91.2141\,{\rm K},
\qquad
g_l=2,
\qquad
g_u=4.
$$

For $T_{\rm QUOKKA}<T_0$, the [C II] emissivity is obtained directly from the
pre-computed DESPOTIC lookup table:

$$
\epsilon_{\rm CII}^{\rm DESPOTIC}
=
n_{\rm H}\,
\ell_{\rm CII}^{\rm DESPOTIC}
(n_{\rm H},N_{\rm H},dV/dr),
$$

where $\ell_{\rm CII}^{\rm DESPOTIC}$ is the tabulated luminosity per hydrogen
nucleus. This branch retains the chemical, level-population, and LVG escape
probability treatment used when the table was constructed.

For $T_{\rm QUOKKA}\geq T_0$, we adopt the gas-phase carbon abundance

$$
{\rm Ab}({\rm C})
\equiv
\frac{n_{\rm C}}{n_{\rm H}}
=
1.6\times10^{-4}.
$$

The C$^+$ number density is

$$
n_{\rm C^+}
=
n_{\rm H}\,{\rm Ab}({\rm C})x_{\rm C^+}.
$$

For the two-level LTE model, the ratio of the upper- and lower-level
populations is

$$
\frac{n_u}{n_l}
=
\frac{g_u}{g_l}
\exp\left(-\frac{\Delta E_{ul}}{k_{\rm B}T_{\rm QUOKKA}}\right),
$$

and the upper-level fraction is

$$
N_u^{\rm LTE}
\equiv
\frac{n_u}{n_{\rm C^+}}
=
\frac{g_u\exp(-\Delta E_{ul}/k_{\rm B}T_{\rm QUOKKA})}
{g_l+g_u\exp(-\Delta E_{ul}/k_{\rm B}T_{\rm QUOKKA})}.
$$

The analytic [C II] emissivity is then

$$
\epsilon_{\rm CII}
=
n_{\rm H}\,{\rm Ab}({\rm C})x_{\rm C^+}
N_u A_{ul}h\nu_{ul}.
$$

Combining the three carbon regimes,

$$
\epsilon_{\rm CII}
=
\begin{cases}
\epsilon_{\rm CII}^{\rm DESPOTIC},
& T_{\rm QUOKKA}<T_0,\\[5pt]
n_{\rm H}{\rm Ab}({\rm C})
x_{\rm C^+}^{(3)}N_u^{\rm LTE}A_{ul}h\nu_{ul},
& T_0\leq T_{\rm QUOKKA}<T_1,\\[5pt]
n_{\rm H}{\rm Ab}({\rm C})
x_{\rm C^+}^{(2)}N_u^{\rm high}A_{ul}h\nu_{ul},
& T_{\rm QUOKKA}\geq T_1,
\end{cases}
$$

where

$$
x_{\rm C^+}^{(3)}
=
\frac{r_1}{1+r_1+r_1r_2},
$$

and

$$
x_{\rm C^+}^{(2)}
=
\frac{n_e}{n_e+S_2(T_{\rm QUOKKA})}.
$$

In the fiducial calculation,

$$
N_u^{\rm high}=N_u^{\rm LTE}.
$$

An optional comparison model replaces only the high-temperature upper-level
fraction with a pre-computed CHIANTI statistical-equilibrium result,

$$
N_u^{\rm high}
=
N_u^{\rm CHIANTI}(T_{\rm QUOKKA},n_{\rm H}).
$$

The CHIANTI lookup uses H and He CIE collider densities when solving the
level-population equations. It changes only the excitation fraction $N_u$; the
C$^+$ ionization fraction remains the two-stage Saha result
$x_{\rm C^+}^{(2)}$.

The cell luminosity is

$$
L_{\rm CII,cell}
=
\epsilon_{\rm CII}V_{\rm cell}.
$$

The analytic intermediate- and high-temperature branches assume optically thin
spontaneous escape. This differs from the low-temperature DESPOTIC branch,
which retains the LVG escape-probability treatment contained in the lookup
table.

## CO $J=1\rightarrow0$ Emission

CO emission primarily originates from cold molecular gas whose thermal and
chemical conditions are covered by the DESPOTIC table. We therefore do not
apply the QUOKKA-temperature regime boundaries to the CO emissivity.

For every simulation cell, the CO $J=1\rightarrow0$ emissivity is

$$
\epsilon_{\rm CO}
=
n_{\rm H}\,
\ell_{\rm CO}^{\rm DESPOTIC}
(n_{\rm H},N_{\rm H},dV/dr),
$$

where $\ell_{\rm CO}^{\rm DESPOTIC}$ is the tabulated luminosity per hydrogen
nucleus. The table includes the DESPOTIC chemical, thermal-balance,
level-population, and LVG escape-probability calculations.

The CO thermal width and CO-weighted phase diagrams use
$T_{\rm DESPOTIC}$. No two-regime temperature or C$^+$ temperature prescription
is applied to CO.

## Temperature Fields Used in Spectral and Phase Products

For clarity, the temperatures used outside the local emissivity calculation are
also species dependent:

- H$\alpha$ and H I thermal widths and luminosity-weighted phase diagrams use
  $T_{\rm H}$.
- C$^+$ thermal widths and [C II]-weighted phase diagrams use
  $T_{\rm QUOKKA}$.
- CO thermal widths and CO-weighted phase diagrams use $T_{\rm DESPOTIC}$.

These choices ensure that the temperature used to characterize each emitting
species is consistent across its luminosity, spectral, and phase-space
products.
