# CHIANTI/fiasco 如何计算 `[C II] 158 μm` emissivity

本文专门解释 CHIANTI/fiasco 的方法，不描述当前 pipeline 的 DESPOTIC cold branch。目标是从最基础的 number-density bookkeeping 开始，逐层看懂：

```text
Element.equilibrium_ionization
    -> Ion.level_populations
    -> Ion.contribution_function
    -> Ion.emissivity
```

最终物理量是 optically thin volumetric emissivity：

$$
\epsilon_{ul}
\quad[\mathrm{erg\,s^{-1}\,cm^{-3}}].
$$

本文依据：

- 项目环境中的 `fiasco 0.6.2` source；
- [`Element.equilibrium_ionization`](https://fiasco.readthedocs.io/en/stable/api/fiasco.Element.html#fiasco.Element.equilibrium_ionization)；
- [`Ion.level_populations`](https://fiasco.readthedocs.io/en/stable/api/fiasco.Ion.html#fiasco.Ion.level_populations)；
- [`Ion.contribution_function`](https://fiasco.readthedocs.io/en/stable/api/fiasco.Ion.html#fiasco.Ion.contribution_function)；
- [`Ion.emissivity`](https://fiasco.readthedocs.io/en/stable/api/fiasco.Ion.html#fiasco.Ion.emissivity)。

---

## 1. 一条 line 的 emissivity 到底是什么

设 upper level 为 $u$、lower level 为 $l$。每个处于 upper level 的 ion：

- 以 $A_{ul}$ 的 probability per unit time 发生 spontaneous emission；
- 每次 transition 放出一个 energy 为

$$
\Delta E_{ul}=h\nu_{ul}
$$

的 photon。

如果 upper-level ion number density 是 $n_u$，单位为 $\mathrm{cm^{-3}}$，那么：

$$
\boxed{
\epsilon_{ul}=n_uA_{ul}\Delta E_{ul}
=n_uA_{ul}h\nu_{ul}
}
$$

单位检查：

$$
\mathrm{cm^{-3}}
\times\mathrm{s^{-1}}
\times\mathrm{erg}
=\mathrm{erg\,s^{-1}\,cm^{-3}}.
$$

因此，无论是当前 pipeline 的 two-level LTE，还是 CHIANTI/fiasco 的 collisional-radiative model，最后一步都相同。真正的区别是如何求 $n_u$。

---

## 2. 从 total hydrogen 到 upper-level C+ 的完整 bookkeeping

CHIANTI 文档把一条 transition 的 emissivity 写成：

$$
\epsilon_{ij}
=n_{\rm H}\,
\mathrm{Ab}(X)\,
f_{X,k}\,
N_j\,
A_{ij}\Delta E_{ij}.
$$

这不是一条新的 emission law，而是把 $n_j$ 拆成了三个 fraction。

### 2.1 Element abundance

定义：

$$
\mathrm{Ab}(X)=\frac{n_X}{n_{\rm H}}.
$$

因此：

$$
\boxed{
n_X=n_{\rm H}\mathrm{Ab}(X)
}
$$

$n_X$ 是 element $X$ 的 total number density，包含这个 element 的所有 ionization stages。

对 carbon：

$$
n_{\rm C}
=n_{\rm C^0}+n_{\rm C^+}+n_{\rm C^{++}}+\cdots.
$$

如果使用项目的 gas-phase carbon abundance：

$$
\mathrm{Ab}(\mathrm C)=A_{\rm C}=1.6\times10^{-4},
$$

那么：

$$
n_{\rm C}=1.6\times10^{-4}n_{\rm H}.
$$

### 2.2 Ionization-stage fraction

定义指定 ionization stage $k$ 的 fraction：

$$
f_{X,k}=\frac{n_{X,k}}{n_X}.
$$

因此：

$$
\boxed{
n_{X,k}=n_Xf_{X,k}
=n_{\rm H}\mathrm{Ab}(X)f_{X,k}
}
$$

这里的 $f_{X,k}$ 不是泛称“有多少 element 被 ionized”，而是精确指定一个 charge state。

Carbon spectroscopic notation 与 charge state 的对应关系：

| Spectroscopic notation | fiasco name | Charge state |
|---|---|---|
| C I | `C 1` | $\mathrm{C^0}$ |
| C II | `C 2` | $\mathrm{C^+}$ |
| C III | `C 3` | $\mathrm{C^{++}}$ |

所以 `[C II] 158 μm` 使用：

$$
f_{\rm C+}=\frac{n_{\rm C+}}{n_{\rm C}},
$$

$$
\boxed{
n_{\rm C+}
=n_{\rm H}\mathrm{Ab}(\mathrm C)f_{\rm C+}
}
$$

### 2.3 Fractional level population

一个确定的 ionization stage 内部还有许多 energy levels。定义 level $j$ 的 fractional population：

$$
N_j=\frac{n_j}{n_{X,k}},
\qquad
\sum_jN_j=1.
$$

因此：

$$
\boxed{
n_j=n_{X,k}N_j
}
$$

对于 `[C II] 158 μm` upper level：

$$
n_u=n_{\rm C+}N_u,
$$

所以：

$$
\boxed{
n_u
=n_{\rm H}\mathrm{Ab}(\mathrm C)f_{\rm C+}N_u
}
$$

整个 number-density chain 是：

$$
n_{\rm H}
\xrightarrow{\times\mathrm{Ab}(\mathrm C)}
n_{\rm C}
\xrightarrow{\times f_{\rm C+}}
n_{\rm C+}
\xrightarrow{\times N_u}
n_u.
$$

### 2.4 代回 spontaneous emission

$$
\epsilon_{ul}=n_uA_{ul}h\nu_{ul},
$$

因此：

$$
\boxed{
\epsilon_{ul}
=n_{\rm H}
\mathrm{Ab}(\mathrm C)
f_{\rm C+}
N_u
A_{ul}h\nu_{ul}
}
$$

上式中各因子的角色完全不同：

| 因子 | 含义 | 决定什么 |
|---|---|---|
| $n_{\rm H}$ | hydrogen-nuclei number density | 总的物质量尺度 |
| $\mathrm{Ab}(\mathrm C)$ | carbon abundance per H nucleus | 有多少 carbon |
| $f_{\rm C+}$ | C+ ionization-stage fraction | carbon 中有多少是 C+ |
| $N_u$ | C+ upper-level fractional population | C+ 中有多少处于 upper level |
| $A_{ul}$ | spontaneous transition probability | 每个 upper ion 每秒 decay 几次 |
| $h\nu_{ul}$ | photon energy | 每次 decay 放出多少能量 |

CHIANTI/fiasco 的主要工作，就是计算 $f_{\rm C+}(T)$ 和 $N_u(T,n_e)$，并从 atomic database 提供 $A_{ul}$ 与 $\Delta E_{ul}$。

---

## 3. 第一步：`Element.equilibrium_ionization` 求 $f_{\rm C+}(T)$

### 3.1 Ionization/recombination balance

对每个 ionization stage $z$，CHIANTI 提供 temperature-dependent：

- ionization rate coefficient $I_z(T)$；
- recombination rate coefficient $R_z(T)$。

在 collisional ionization equilibrium，假设：

- ionization 与 recombination 达到 steady state；
- ion fractions 只依赖 $T$；
- 不显式依赖 density，因为所有相邻-stage rate 都带相同的 $n_e$，在 equilibrium equations 中约掉；
- 不包含外部 photoionization field 或 time-dependent non-equilibrium ionization。

对中间 stage $z$，steady-state equation 的结构是：

$$
0=
f_{z-1}I_{z-1}
+f_{z+1}R_{z+1}
-f_z(I_z+R_z).
$$

同时满足 normalization：

$$
\sum_{z=0}^{Z}f_z=1.
$$

### 3.2 fiasco source 如何构造 matrix

`fiasco/elements.py::_rate_matrix`：

```python
rate_matrix[:, i, i] = -(
    self[i].ionization_rate
    + self[i].recombination_rate
)

rate_matrix[:, i, i-1] = self[i-1].ionization_rate
rate_matrix[:, i, i+1] = self[i+1].recombination_rate
```

这就是相邻 ionization stages 之间的 tridiagonal rate matrix。

### 3.3 用 SVD 找 equilibrium null vector

```python
_, _, V = np.linalg.svd(self._rate_matrix.value)
ionization_fraction = np.fabs(V[:, -1, :])
ionization_fraction /= ionization_fraction.sum(axis=1)[:, None]
```

rate matrix 的 null-space vector 满足：

$$
\mathbf M(T)\mathbf f(T)=0.
$$

SVD 返回最小 singular value 对应的 vector；取绝对值解决 vector 整体正负号不定的问题，再做 normalization。

### 3.4 Carbon 调用和 column 含义

```python
carbon = fiasco.Element('carbon', temperature)
xC = carbon.equilibrium_ionization
```

对 atomic number $Z=6$ 的 carbon，结果 shape 为：

```text
(N_temperature, 7)
```

columns 是：

```text
C0, C+, C++, C3+, C4+, C5+, C6+
```

因此：

```python
f_Cplus = xC[:, 1]
```

即：

$$
f_{\rm C+}(T)=\texttt{xC[:,1]}.
$$

---

## 4. 第二步：`Ion.level_populations` 求 $N_u(T,n_e)$

这是 CHIANTI/fiasco 与当前 two-level LTE 算法差异最大的部分。

### 4.1 Statistical equilibrium

对一个 ion 的每个 level $i$，steady state 要求 total population-in rate 等于 total population-out rate：

$$
\sum_{j\ne i}N_jR_{ji}
-N_i\sum_{j\ne i}R_{ij}=0.
$$

并满足：

$$
\sum_iN_i=1.
$$

$R_{ij}$ 可以包含：

- spontaneous radiative decay；
- electron collisional excitation/de-excitation；
- proton collisional excitation/de-excitation；
- 可用时的 level-resolved ionization/recombination correction；
- CHIANTI v9+ 可用时的 two-ion model coupling。

### 4.2 Electron effective collision strength

CHIANTI 保存 Maxwellian-averaged effective collision strength：

$$
\Upsilon_{ul}(T)
=\int_0^\infty
\Omega_{ul}(E)
\exp\left(-\frac{E}{k_{\rm B}T}\right)
d\left(\frac{E}{k_{\rm B}T}\right).
$$

fiasco 用 `burgess_tully_descale` 将 CHIANTI 保存的 scaled spline 数据转换到指定 temperature：

```python
upsilon = burgess_tully_descale(
    self._scups['bt_t'],
    self._scups['bt_upsilon'],
    kBTE.T,
    self._scups['bt_c'],
    self._scups['bt_type'],
)
```

### 4.3 Electron collisional de-excitation

fiasco source 实现：

$$
q_{ul}^{(e)}(T)
=C_0\frac{\Upsilon_{ul}(T)}{g_u\sqrt{k_{\rm B}T}},
$$

其中 source 中：

$$
C_0=\frac{h^2}{(2\pi m_e)^{3/2}}.
$$

代码：

```python
c = const.h**2 / (2.0 * np.pi * const.m_e)**1.5
q_ul = c * upsilon / np.sqrt(self.thermal_energy) / omega_upper
```

结果单位为 $\mathrm{cm^3\,s^{-1}}$。

### 4.4 Electron collisional excitation

通过 detailed balance：

$$
\boxed{
q_{lu}^{(e)}(T)
=\frac{g_u}{g_l}
q_{ul}^{(e)}(T)
\exp\left(-\frac{\Delta E_{ul}}{k_{\rm B}T}\right)
}
$$

代码：

```python
kBTE = np.outer(1.0/self.thermal_energy, delta_energy)
q_lu = (
    omega_upper / omega_lower
    * q_ul
    * np.exp(-kBTE)
)
```

注意 `kBTE` 在这里实际是 $\Delta E/(k_{\rm B}T)$。

### 4.5 Proton collisions

默认：

```python
include_protons=True
```

fiasco 从 CHIANTI proton scaled-rate data 得到 $q_{ij}^{(p)}(T)$，并在 rate matrix 中乘 proton density：

$$
n_p
=\left(\frac{n_p}{n_e}\right)n_e.
$$

```python
d_p = self.proton_electron_ratio * d_e
rate_matrix += d_p * proton_collision_rate_matrix
```

如果该 ion 没有 proton data，fiasco 会 warning 并跳过 proton excitation/de-excitation。

### 4.6 Radiative 与 collisional rates 进入同一 matrix

```python
rate_matrix = (
    self._rate_matrix_radiative_decay
    + n_e * self._rate_matrix_collisional_electron
)

rate_matrix += (
    n_p * self._rate_matrix_collisional_proton
)
```

因此 matrix 中：

$$
R_{ul}
=A_{ul}
+n_eq_{ul}^{(e)}
+n_pq_{ul}^{(p)}+cdots,
$$

$$
R_{lu}
=n_eq_{lu}^{(e)}
+n_pq_{lu}^{(p)}+cdots.
$$

### 4.7 加 normalization 并解 linear system

fiasco 把 matrix 最后一行替换成全 1：

```python
c_matrix[:, -1, :] = 1
b = np.zeros(c_matrix.shape[2:])
b[-1] = 1
pop = np.linalg.solve(c_matrix.value, b)
```

最后一行代表：

$$
N_1+N_2+\cdots+N_{N_{\rm level}}=1.
$$

解得：

$$
\mathbf N(T,n_e)
=\left[N_1,N_2,\ldots,N_{N_{\rm level}}\right].
$$

对 `[C II] 158 μm`：

$$
N_u=N_{\,{}^2P_{3/2}}.
$$

---

## 5. 用 two-level system 理解 `level_populations`

完整 matrix 很抽象，可以先从 two-level atom 理解。

设：

- lower level population：$n_l$；
- upper level population：$n_u$；
- electron excitation coefficient：$q_{lu}$；
- electron de-excitation coefficient：$q_{ul}$。

steady state：

$$
n_ln_eq_{lu}
=n_u\left(A_{ul}+n_eq_{ul}\right).
$$

再加：

$$
n_l+n_u=n_{\rm C+}.
$$

解得 upper-level fractional population：

$$
\boxed{
N_u(T,n_e)
=\frac{n_eq_{lu}}
{A_{ul}+n_e(q_{ul}+q_{lu})}
}
$$

### 5.1 Low-density limit

若：

$$
n_eq_{ul}\ll A_{ul},
$$

则：

$$
N_u\approx\frac{n_eq_{lu}}{A_{ul}},
$$

$$
\epsilon_{ul}
\approx n_{\rm C+}n_eq_{lu}h\nu_{ul}.
$$

所以：

$$
\epsilon_{ul}\propto n_{\rm C+}n_e.
$$

物理图像是：一次 electron collision 把 C+ 激发到 upper level，随后几乎一定通过 spontaneous emission 放出 photon。

### 5.2 High-density/LTE limit

若 collisions 远快于 spontaneous decay，则：

$$
\frac{n_u}{n_l}
\rightarrow\frac{q_{lu}}{q_{ul}}.
$$

利用 detailed balance：

$$
\frac{q_{lu}}{q_{ul}}
=\frac{g_u}{g_l}
\exp\left(-\frac{\Delta E}{k_{\rm B}T}\right),
$$

得到 Boltzmann LTE：

$$
\frac{n_u}{n_l}
\rightarrow
\frac{g_u}{g_l}e^{-\Delta E/k_{\rm B}T}.
$$

因此当前 pipeline 的 two-level LTE 不是另一套 emission law，而是 CHIANTI collisional-radiative solution 的 high-density limit。

### 5.3 Critical density

忽略其他 colliders 时：

$$
n_{\rm crit,e}(T)
\approx\frac{A_{ul}}{q_{ul}^{(e)}(T)}.
$$

- $n_e\ll n_{\rm crit,e}$：subthermal；
- $n_e\gg n_{\rm crit,e}$：趋近 LTE。

---

## 6. 第三步：`Ion.contribution_function`

fiasco 定义：

$$
\boxed{
G_{ij}(n_e,T)
=\mathrm{Ab}(X)
f_{X,k}(T)
N_j(n_e,T)
A_{ij}\Delta E_{ij}
\frac{1}{n_e}
}
$$

单位：

$$
[G]=\mathrm{erg\,cm^3\,s^{-1}}.
$$

source 的逐行对应关系：

```python
populations = self.level_populations(density, **kwargs)
```

对应：

$$
N_j(n_e,T).
$$

```python
term = np.outer(self.ionization_fraction, 1./density)
term *= self.abundance
```

对应：

$$
\mathrm{Ab}(X)f_{X,k}(T)\frac{1}{n_e}.
$$

```python
A = self.transitions.A[bound_bound]
energy = const.h * const.c / wavelength
```

对应：

$$
A_{ij},qquad \Delta E_{ij}=\frac{hc}{\lambda_{ij}}.
$$

```python
g = term * populations[..., i_upper] * (A * energy)
```

对应完整的 $G_{ij}$。

### 6.1 为什么定义中有 $1/n_e$

这是 contribution function 的 normalization convention。它让 emissivity 可以写成：

$$
\epsilon_{ij}=G_{ij}n_{\rm H}n_e.
$$

虽然外面的 $n_e$ 与 $G$ 中显式的 $1/n_e$ 会相消，但 $G$ 仍然依赖 density，因为：

$$
N_j=N_j(T,n_e).
$$

因此绝不能根据代数消去 $n_e$ 就判断 emissivity 与 electron density 无关。

---

## 7. 第四步：`Ion.emissivity`

官方定义：

$$
\boxed{
\epsilon_{ij}(n_e,T)
=G_{ij}(n_e,T)n_{\rm H}n_e
}
$$

但 API 只要求输入 electron density：

```python
ion.emissivity(density=n_e)
```

所以 fiasco 内部通过：

```python
pe_ratio = proton_electron_ratio(self.temperature, ...)
```

计算：

$$
\mathrm{pe\_ratio}(T)=\frac{n_{\rm H}}{n_e}.
$$

然后 source：

```python
return g * pe_ratio * density**2
```

即：

$$
G_{ij}
\left(\frac{n_{\rm H}}{n_e}\right)
n_e^2
=G_{ij}n_{\rm H}n_e.
$$

把 $G$ 展开：

$$
\epsilon_{ij}
=\left[
\mathrm{Ab}(X)f_{X,k}N_jA_{ij}\Delta E_{ij}
\frac{1}{n_e}
\right]
n_{\rm H}n_e,
$$

得到：

$$
\boxed{
\epsilon_{ij}
=n_{\rm H}
\mathrm{Ab}(X)
f_{X,k}
N_j
A_{ij}\Delta E_{ij}
}
$$

再用：

$$
n_j
=n_{\rm H}\mathrm{Ab}(X)f_{X,k}N_j,
$$

最终回到：

$$
\boxed{
\epsilon_{ij}=n_jA_{ij}h\nu_{ij}
}
$$

---

## 8. `[C II] 158 μm` 的实际调用

### 8.1 先计算 C+ CIE fraction

```python
import astropy.units as u
import fiasco

T = T_cells * u.K

carbon = fiasco.Element('carbon', T)
f_cplus = carbon.equilibrium_ionization[:, 1]
```

### 8.2 构造 C II Ion

```python
cii = fiasco.Ion(
    'C 2',
    T,
    abundance=1.6e-4,
    ionization_fraction=f_cplus,
)
```

这两个 keyword 很重要：

- `abundance=1.6e-4`：使用项目自己的 gas-phase carbon abundance；否则 fiasco 默认是 `sun_coronal_1992_feldman_ext`；
- `ionization_fraction=f_cplus`：使用刚才 `Element.equilibrium_ionization` 的 on-the-fly CIE result；否则 `Ion` 默认读取名为 `chianti` 的 tabulated ionization-fraction dataset。

### 8.3 输入 electron density

```python
ne = ne_cells / u.cm**3

eps_all = cii.emissivity(
    ne,
    couple_density_to_temperature=True,
)
```

`couple_density_to_temperature=True` 表示：

```text
T[0] paired with ne[0]
T[1] paired with ne[1]
...
```

而不是构造所有 $T_i\times n_{e,j}$ combinations。

输出 shape：

```text
(N_cell, 1, N_bound_bound_transition)
```

### 8.4 选择 158 μm transition

`emissivity()` 的最后一维只包含 `is_bound_bound=True` 的 transitions，因此必须在同一个 filtered transition list 上找 index：

```python
tr = cii.transitions
bb = tr.is_bound_bound

lower = tr.lower_level[bb]
upper = tr.upper_level[bb]

k158 = np.where((lower == 1) & (upper == 2))[0]
eps_158 = eps_all[:, 0, k158[0]]
```

这里 CHIANTI 的 `lower_level=1, upper_level=2` 表示数据记录连接 level 1 和 level 2；实际 spontaneous emission 方向是 upper level 2 到 lower level 1：

$$
{}^2P_{3/2}\rightarrow{}^2P_{1/2}.
$$

---

## 9. 如果必须使用 simulation 自己的 $n_{\rm H}$

`Ion.emissivity(ne)` 内部通过 `proton_electron_ratio` 从 $n_e$ 推算 $n_{\rm H}$。这不一定严格等于 simulation 定义：

$$
n_{\rm H}^{\rm sim}=\frac{X_{\rm H}\rho}{m_{\rm H}}.
$$

若要严格保留 simulation 的 $n_{\rm H}$，应使用：

```python
G_all = cii.contribution_function(
    ne,
    couple_density_to_temperature=True,
)

eps_all = G_all * nH_cells[:, None, None] * ne[:, None, None]
```

数学上：

$$
\boxed{
\epsilon_{ij}^{\rm sim}
=G_{ij}(T,n_e)n_{\rm H}^{\rm sim}n_e
}
$$

这样：

- abundance、ion fraction、level population、$A_{ij}$、$\Delta E$ 来自 CII `Ion` object；
- $n_{\rm H}$ 明确使用 simulation field；
- 不使用 fiasco 内部的 $n_{\rm H}/n_e$ composition assumption。

---

## 10. 与当前 pipeline C+ analytic branch 的精确比较

两者都从：

$$
\epsilon_{ul}=n_uA_{ul}h\nu
$$

出发，但 $f_{\rm C+}$ 和 $N_u$ 的来源不同。

| 物理步骤 | 当前 pipeline | CHIANTI/fiasco 完整方法 |
|---|---|---|
| $n_{\rm C}$ | $A_{\rm C}n_{\rm H}$ | `Ion.abundance * n_H` |
| $f_{\rm C+}$ | high-$T$ 强制 C$^+$/C$^{++}$ 两态，并使用 $n_e/(n_e+S_2)$ | 可使用 `Element.equilibrium_ionization[:,1]`，或 `Ion` 默认 tabulated fraction |
| $n_{\rm C+}$ | $A_{\rm C}n_{\rm H}f_{\rm C+}$ | 相同 bookkeeping |
| $N_u$ | two-level Boltzmann LTE | multi-level statistical equilibrium |
| Electron collisions | 不显式计算 | CHIANTI $\Upsilon(T)$ → $q_{ij}(T)$ |
| Proton collisions | 无 | 有数据时默认包含 |
| Density dependence | LTE $N_u(T)$ | $N_u(T,n_e)$ |
| Radiative decay | $A_{ul}$ | $A_{ul}$ |
| Photon energy | $h\nu$ | $hc/\lambda=\Delta E$ |
| Optical depth | 无 | 无；同样是 optically thin |

最关键的关系是：

$$
\boxed{
N_u^{\rm pipeline}(T)
=N_u^{\rm LTE}(T)
}
$$

而：

$$
\boxed{
N_u^{\rm fiasco}(T,n_e)
=N_u^{\rm statistical\ equilibrium}(T,n_e)
}
$$

当 $n_e\gg n_{\rm crit}$ 时，两者趋近；当 $n_e\ll n_{\rm crit}$ 时，当前 LTE 方法会忽略 subthermal suppression。

---

## 11. 对大规模 simulation 的现实实现方式

> **当前项目实现（B 方案）**：simulation cell 提供
> \(n_{\rm H}^{\rm sim}=\rho X/m_{\rm H}\)。CHIANTI CIE table 只提供
> H、He 的各电离态比例；建表脚本再用 \(X=0.74\)、\(Y=0.26\)
> 和电荷守恒计算
>
> \[
> n_e=n_{\rm H}\left[
> \sum_q q f_{{\rm H},q}
> +\frac{Y}{4X}\sum_q q f_{{\rm He},q}
> \right],
> \qquad
> n_p=n_{\rm H}f_{{\rm H}^+}.
> \]
>
> 然后把明确的 \(n_e,n_p\) 交给 `Ion.level_populations`，保存
> \(N_u^{\rm CHIANTI}(T,n_{\rm H})\)。runtime 再用每个 cell 自己的
> \((T,n_{\rm H}^{\rm sim})\) 插值。所以下面以
> \(G(T,n_e)\) 为轴的段落是通用替代方案，不是本项目当前采用的接口。
> C 对自由电子密度的微小贡献在这里忽略；发射端使用当前高温两态
> Saha fraction：
> \(n_{\rm C+}=A_{\rm C}[n_e/(n_e+S_2)]n_{\rm H}\)。

不能对上亿个 cells 直接调用一次完整 `Ion.level_populations` matrix solve。更现实的流程是预建一个 CHIANTI emissivity lookup table：

$$
G_{\rm CII}(T,n_e)
$$

或：

$$
\frac{\epsilon_{\rm CII}}{n_{\rm H}}
=G_{\rm CII}(T,n_e)n_e.
$$

推荐 axes：

```text
log10(T/K)
log10(ne/cm^-3)
```

建表时：

1. 用 `Element.equilibrium_ionization` 得到 $f_{\rm C+}(T)$；
2. 构造 `Ion('C 2', ..., abundance=A_C, ionization_fraction=f_Cplus)`；
3. 用 `contribution_function(ne_grid)` 解 level populations；
4. 选出 level $2\rightarrow1$ transition；
5. 保存 $G_{158}(T,n_e)$；
6. runtime 对每个 cell interpolation；
7. 计算

$$
\epsilon_{158}
=G_{158}(T,n_e)n_{\rm H}^{\rm sim}n_e.
$$

这样既保留 CHIANTI collisional-radiative physics，也适合大规模 simulation。

---

## 12. 最短概念总结

```text
n_H
  |
  | * Ab(C)
  v
n_C                    total carbon
  |
  | * f_C+(T)
  |   from Element.equilibrium_ionization
  v
n_C+                   C+ ion number density
  |
  | * N_u(T, n_H)
  |   lookup built from explicit n_e(T,n_H), n_p(T,n_H)
  |   and Ion.level_populations
  v
n_u                    upper-level C+ number density
  |
  | * A_ul * h nu
  v
epsilon_CII            erg s^-1 cm^-3
```

最终仍然是：

$$
\boxed{
\epsilon_{\rm CII}=n_uA_{ul}h\nu
}
$$

CHIANTI/fiasco 的价值不是改变这条公式，而是用 atomic rates 和 statistical equilibrium 更完整地计算：

$$
f_{\rm C+}(T)
\quad\text{和}\quad
N_u[T,n_e(T,n_{\rm H}),n_p(T,n_{\rm H})].
$$
