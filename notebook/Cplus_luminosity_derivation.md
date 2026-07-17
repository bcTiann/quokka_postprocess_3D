# `[C II] 158 μm` 单个 cell 体积发射率的完整计算流程

本文档描述当前 `quokka2s` 代码如何为每个 simulation cell 计算

$$
\epsilon_{\mathrm{CII}}
\quad [\mathrm{erg\,s^{-1}\,cm^{-3}}].
$$

> 更新：高温 branch 现在可由 `CPLUS_HIGH_MODEL=lte|chianti` 选择。`chianti`
> 模式只替换 $T\ge1.307\times10^4$ K 的 upper-level population，并使用
> H+He CIE charge neutrality 建立显式 $n_e$、$n_p$ lookup；C 对 $n_e$
> 的微小贡献忽略，但仍用 $A_C f_{\rm C+}^{\rm CIE}$ 计算发射离子密度。本文后续的
> LTE 推导仍对应 `lte` comparison model。

这里的 yt field 名为 `('gas', 'C+_luminosity')`，但它的物理量实际是 **volumetric emissivity**，不是已经乘过 cell volume 的 luminosity。本文到 $\epsilon_{\mathrm{CII}}$ 为止，不包含 cell luminosity、thermal broadening、Doppler shift、spectral cube 或 integrated spectrum。

当前实现位于：

- field 注册：`src/quokka2s/pipeline/prep/physics_fields.py:1064-1071`
- 主计算函数：`_Cplus_luminosity_model()`，另由 LTE/CHIANTI wrappers 选择高温 excitation
- CHIANTI/fiasco atomic-data 初始化：同文件 `101-205`
- Hydrogen Saha helper：同文件 `212-267`
- DESPOTIC table interpolation：同文件 `430-455` 与 `src/quokka2s/tables/lookup.py:23-165`

---

## 1. 最终的 piecewise 定义

代码按 QUOKKA 原生温度 $T_{\rm QK}$ 做两个 hard boundaries、三个 regimes：

$$
\epsilon_{\mathrm{CII}} =
\begin{cases}
\epsilon_{\rm DSP},
& T_{\rm QK}<3000\ \mathrm{K},\\[4pt]
\epsilon_{\rm Saha+LTE},
& 3000\ \mathrm{K}\le T_{\rm QK}<10^4\ \mathrm{K},\\[4pt]
\epsilon_{\rm CHIANTI\ CIE+LTE},
& T_{\rm QK}\ge10^4\ \mathrm{K}.
\end{cases}
$$

对应代码：

```python
hot = T_qk >= T_QK_TWO_REGIME_K       # 3000 K

eps = np.where(
    T_qk >= T_CIE_K,                   # 10000 K
    eps_cie,
    np.where(hot, eps_hot, eps_cold),
)
```

因此：

- $T_{\rm QK}=3000\ \mathrm{K}$ 属于 `Saha+LTE` branch；
- $T_{\rm QK}=10^4\ \mathrm{K}$ 属于 `CHIANTI CIE+LTE` branch；
- boundaries 没有 interpolation 或 blending，允许出现数值 discontinuity；
- NumPy 会先计算三个 branch 的 array，再由 `np.where` 选择最终值，所以“未被选中”的 branch 仍会被执行，但不会进入最终输出。

---

## 2. 每个 cell 的基础输入

### 2.1 QUOKKA temperature

$$
T_{\rm QK}=T_{\rm boxlib}.
$$

代码只是为 QUOKKA 写入的 raw field 附加 Kelvin unit：

```python
def _temperature_quokka(field, data):
    return data[('boxlib', 'temperature')] * K
```

C+ 的三个 regimes 全部由 $T_{\rm QK}$ 决定；冷端并不是按 $T_{\rm DESPOTIC}$ 判断 branch。

### 2.2 Total hydrogen-nuclei number density

$$
n_{\rm H}=\frac{X_{\rm H}\rho}{m_{\rm H}},
$$

其中：

- $\rho$：simulation gas mass density，单位 $\mathrm{g\,cm^{-3}}$；
- $X_{\rm H}=0.74$：hydrogen mass fraction；
- $m_{\rm H}$：代码中的 `yt.units.mh`；
- $n_{\rm H}$ 表示 H nuclei 总数密度，即 $n_{\rm HI}+n_{\rm H^+}+2n_{\rm H_2}$，单位 $\mathrm{cm^{-3}}$。

对应代码：

```python
density_3d = data[('gas', 'density')].in_cgs()
n_H_3d = (density_3d * cfg.X_H) / m_H
```

### 2.3 冷端 table 的另外两个坐标

冷端还使用：

$$
N_{\rm H}\quad[\mathrm{cm^{-2}}],
\qquad
\left|\frac{dV}{dr}\right|_{\rm LVG}\quad[\mathrm{s^{-1}}].
$$

当前代码中：

$$
\left|\frac{dV}{dr}\right|_{\rm LVG}
=\max\!\left(\frac{|\nabla\cdot\mathbf v|}{3},10^{-18}\ \mathrm{s^{-1}}\right).
$$

$N_{\rm H}$ 是六个方向 $\pm x,\pm y,\pm z$ column 的组合；默认使用 harmonic mean，且 $x/y$ 方向可以加 lateral extension。本文把最终得到的 $N_{\rm H}$ 与 $dV/dr$ 当作预建 DESPOTIC table 的输入，不展开 table 的 build physics。

---

## 3. Regime I：$T_{\rm QK}<3000$ K，DESPOTIC table

### 3.1 读取和缓存 table

代码从 `cfg.DESPOTIC_TABLE_PATH` 读取预建 `.npz`，构造一个 process-level `TableLookup`：

```python
lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
```

`TableLookup` 的坐标轴是：

$$
\bigl(\log_{10}n_{\rm H},\ \log_{10}N_{\rm H},\
\log_{10}(dV/dr)\bigr).
$$

它使用 `scipy.interpolate.RegularGridInterpolator(method='linear')`。需要特别注意：

- independent variables 是 log-space coordinates；
- interpolated dependent variable `lumPerH` 本身没有先取 logarithm；
- 因此这是“在 log-coordinate grid 上对 linear `lumPerH` 做 trilinear interpolation”，不是 log-emissivity interpolation。

### 3.2 Clamp 到 table domain

每个输入先被分别 clamp 到 `.npz` 中实际保存的最小值和最大值：

$$
\begin{aligned}
n_{\rm H}^{\rm clip} &=
\operatorname{clip}(n_{\rm H},n_{\rm H,min},n_{\rm H,max}),\\
N_{\rm H}^{\rm clip} &=
\operatorname{clip}(N_{\rm H},N_{\rm H,min},N_{\rm H,max}),\\
(dV/dr)^{\rm clip} &=
\operatorname{clip}(dV/dr,(dV/dr)_{\min},(dV/dr)_{\max}).
\end{aligned}
$$

当前 canonical builder 使用的 nominal ranges 是：

$$
n_{\rm H}=10^{-4}\ldots10^6\ \mathrm{cm^{-3}},\quad
N_{\rm H}=10^{15}\ldots10^{24}\ \mathrm{cm^{-2}},\quad
dV/dr=10^{-19}\ldots10^{-12}\ \mathrm{s^{-1}},
$$

但 runtime 不硬编码这些范围，而是读取 table axes。

### 3.3 查 `C+_lumPerH`

代码调用：

```python
lumPerH = lookup.line_field(
    'C+', 'lumPerH',
    n_H_safe, col_safe, dV_safe,
)
```

table 中：

$$
\ell_{\rm CII}^{\rm DSP}
\equiv \mathrm{lumPerH}
\quad[\mathrm{erg\,s^{-1}\,H^{-1}}].
$$

若 interpolation 结果是 `NaN`，代码用 `np.nan_to_num(..., nan=0.0)` 将其置零。

### 3.4 转换成 volumetric emissivity

$$
\boxed{
\epsilon_{\rm DSP}
=n_{\rm H}^{\rm clip}\,
\ell_{\rm CII}^{\rm DSP}
}
$$

单位检查：

$$
\mathrm{cm^{-3}}
\times
\mathrm{erg\,s^{-1}\,H^{-1}}
=\mathrm{erg\,s^{-1}\,cm^{-3}}.
$$

一个容易忽略的实现细节是：当原始 $n_{\rm H}$ 超出 table range 时，代码不仅使用 $n_{\rm H}^{\rm clip}$ 查询 table，也使用 $n_{\rm H}^{\rm clip}$ 乘 `lumPerH`。因此冷端 emissivity 在 table density boundary 外会随 clamp 饱和，而不是继续按原始 $n_{\rm H}$ 线性缩放。

---

## 4. Import-time CHIANTI/fiasco atomic-data 初始化

以下内容在 `physics_fields.py` import 时只构建一次。当前项目安装的是 `fiasco 0.6.2`，README 指定使用 CHIANTI 10.1；但是代码本身没有显式传入或验证 CHIANTI database version，因此实际运行使用的是本机 fiasco HDF5 database 中的数据。

### 4.1 Temperature grids 和 Ion objects

用于 partition functions 与 atomic constants 的 grid：

$$
T_j=10^{2.5}\ldots10^{7.5}\ \mathrm{K},
\qquad N_T=600.
$$

```python
T_grid_q = np.logspace(2.5, 7.5, 600) * u.K
cii  = fiasco.Ion('C 2', T_grid_q)   # C+
ci   = fiasco.Ion('C 1', T_grid_q)   # C0
ciii = fiasco.Ion('C 3', T_grid_q)   # C++
```

`C 1`、`C 2`、`C 3` 是 spectroscopic ion stages，不是 electric charge 本身：分别对应 $\mathrm{C^0}$、$\mathrm{C^+}$、$\mathrm{C^{++}}$。

### 4.2 三个 partition functions

对 $q\in\{\mathrm{C^0,C^+,C^{++}}\}$，代码从 CHIANTI 的完整 model level list 计算：

$$
\boxed{
U_q(T)=\sum_i g_{q,i}
\exp\!\left(-\frac{E_{q,i}}{k_{\rm B}T}\right)
}
$$

其中：

$$
g_{q,i}=2J_{q,i}+1.
$$

对应调用：

```python
E_cii  = cii.levels.energy.to('erg').value[:, None]
E_ci   = ci.levels.energy.to('erg').value[:, None]
E_ciii = ciii.levels.energy.to('erg').value[:, None]

g_cii  = cii.levels.weight[:, None]
g_ci   = ci.levels.weight[:, None]
g_ciii = ciii.levels.weight[:, None]

U_Cp_grid  = (g_cii  * np.exp(-E_cii  / (k_B*T))).sum(axis=0)
U_C0_grid  = (g_ci   * np.exp(-E_ci   / (k_B*T))).sum(axis=0)
U_Cpp_grid = (g_ciii * np.exp(-E_ciii / (k_B*T))).sum(axis=0)
```

在 fiasco 中：

- `levels.energy` 优先使用 observed energy；若不存在则 fallback 到 theoretical energy；
- `levels.weight` 返回 $2J+1$。

per-cell 计算时用：

```python
U_Cp  = np.interp(T_safe, _CII_T_GRID, _CII_U_CP)
U_C0  = np.interp(T_safe, _CII_T_GRID, _CII_U_C0)
U_Cpp = np.interp(T_safe, _CII_T_GRID, _CII_U_CPP)
```

这里是对 $T$ 做 linear interpolation，不是对 $\log T$ 插值；`np.interp` 对 grid 外温度使用 endpoint value。

### 4.3 Ionization potentials

$$
I_{\rm C0}=11.2602969751\ \mathrm{eV},
\qquad
I_{\rm C+}=24.3833151896\ \mathrm{eV}.
$$

代码调用：

```python
I_C_eV = fiasco.Ion('C 1', [8000.0]*u.K).ionization_potential
I_C2_eV = fiasco.Ion('C 2', [8000.0]*u.K).ionization_potential
```

`8000 K` 只是构造 `Ion` object 所需的 temperature array；`ionization_potential` 是 temperature-independent atomic datum。随后用 Astropy 做

$$
\Theta_{\rm C0}=\frac{I_{\rm C0}}{k_{\rm B}},
\qquad
\Theta_{\rm C+}=\frac{I_{\rm C+}}{k_{\rm B}}
$$

的 eV-to-K conversion。

### 4.4 `[C II] 158 μm` transition constants

代码在 C II transitions 中选择：

```python
lo = cii.transitions.lower_level
up = cii.transitions.upper_level
sel = np.where((lo == 1) & (up == 2))[0]
i158 = int(sel[0])
```

即 CHIANTI level 1 和 level 2 之间的 ground-term fine-structure transition：

$$
{}^2P_{3/2}\rightarrow{}^2P_{1/2}.
$$

代码取得：

```python
A_ul = cii.transitions.A[i158]
Delta_E = cii.transitions.delta_energy[i158]
nu = Delta_E / const_ap.h
T_star = Delta_E / const_ap.k_B
g_l = cii.levels.weight[0]
g_u = cii.levels.weight[1]
```

本机当前数据给出：

$$
\begin{aligned}
A_{ul} &= 2.290\times10^{-6}\ \mathrm{s^{-1}},\\
\nu_{ul} &= 1.9005942459826\times10^{12}\ \mathrm{Hz},\\
T_*\equiv\Delta E/k_{\rm B} &= 91.2141377\ \mathrm{K},\\
g_l &=2,\\
g_u &=4.
\end{aligned}
$$

`transitions.A` 是 spontaneous radiative-decay probability；`transitions.delta_energy` 是由上下 level energies 得到的 $\Delta E$。

### 4.5 CHIANTI CIE ion fraction

另建一套 grid：

$$
T_j^{\rm CIE}=10^4\ldots10^{8.5}\ \mathrm{K},
\qquad N_T=600.
$$

```python
xC = np.asarray(
    fiasco.Element('carbon', T_grid_q).equilibrium_ionization
)
x_Cp = xC[:, 1]
```

columns 的顺序是：

$$
[\mathrm{C^0},\mathrm{C^+},\mathrm{C^{++}},\ldots],
$$

所以 `xC[:, 1]` 是 $x_{\rm C+}^{\rm CIE}(T)$。

需要精确区分：`equilibrium_ionization` 在本项目使用的 fiasco 0.6.2 中，不是简单返回一张预先 tabulated ion-fraction curve；它使用 CHIANTI 的 ionization/recombination rates 构造 rate matrix，并通过 SVD 在每个温度求 equilibrium population。结果仍然只依赖 $T$，不依赖 density。

per-cell 使用：

```python
x_Cp_cie = np.interp(T_safe, _CIE_T_GRID, _CIE_X_CP)
```

同样是 linear-$T$ interpolation，并在 grid 外使用 endpoint value。

---

## 5. Regime II：$3000\le T_{\rm QK}<10^4$ K，Saha ionization + two-level LTE

定义数值保护后的温度：

$$
T=\max(T_{\rm QK},1\ \mathrm{K}).
$$

在本 regime 中实际总有 $T\ge3000$ K，因此这个 floor 不改变结果。

### 5.1 Hydrogen Saha 给 electron density

代码先假设 electrons 只来自 hydrogen，并满足 charge neutrality：

$$
n_e=n_{\rm H^+}=x_{\rm H^+}n_{\rm H},
\qquad
n_{\rm HI}=(1-x_{\rm H^+})n_{\rm H}.
$$

Hydrogen Saha relation 写为：

$$
\frac{n_e n_{\rm H^+}}{n_{\rm HI}}
=K_{\rm H}(T),
$$

$$
K_{\rm H}(T)=
\left(\frac{2\pi m_e k_{\rm B}T}{h^2}\right)^{3/2}
\exp\!\left(-\frac{I_{\rm H}}{k_{\rm B}T}\right).
$$

令

$$
R=\frac{K_{\rm H}(T)}{n_{\rm H}},
$$

则

$$
\frac{x_{\rm H^+}^2}{1-x_{\rm H^+}}=R.
$$

代码使用避免 catastrophic cancellation 的正根：

$$
\boxed{
x_{\rm H^+}
=\frac{2}{\sqrt{1+4/R}+1}
}
$$

最终：

$$
n_e=x_{\rm H^+}n_{\rm H}.
$$

相关常数由 `astropy.constants` 计算：

$$
\frac{2\pi m_e k_{\rm B}}{h^2}
=1.7998656465\times10^{10}\ \mathrm{cm^{-2}\,K^{-1}},
$$

$$
\frac{I_{\rm H}}{k_{\rm B}}
=1.5788751240\times10^5\ \mathrm{K}.
$$

实现采用 `log10(R)` 并将 exponent clip 到 $[-290,290]$，以避免 float64 overflow/underflow。density 也有

$$
n_{\rm H}\leftarrow\max(n_{\rm H},10^{-30}\ \mathrm{cm^{-3}})
$$

的 numerical floor。

### 5.2 Carbon 两级 Saha ionization chain

代码只保留三种 carbon ion stages：

$$
\mathrm{C^0}\rightleftharpoons\mathrm{C^+}
\rightleftharpoons\mathrm{C^{++}}.
$$

两个 Saha constants：

$$
S_1(T)=
2\left(\frac{2\pi m_e k_{\rm B}T}{h^2}\right)^{3/2}
\frac{U_{\rm C+}(T)}{U_{\rm C0}(T)}
\exp\!\left(-\frac{I_{\rm C0}}{k_{\rm B}T}\right),
$$

$$
S_2(T)=
2\left(\frac{2\pi m_e k_{\rm B}T}{h^2}\right)^{3/2}
\frac{U_{\rm C++}(T)}{U_{\rm C+}(T)}
\exp\!\left(-\frac{I_{\rm C+}}{k_{\rm B}T}\right).
$$

前面的 factor 2 是 free-electron spin degeneracy。定义：

$$
r_1=\frac{S_1}{n_e}=\frac{n_{\rm C+}}{n_{\rm C0}},
\qquad
r_2=\frac{S_2}{n_e}=\frac{n_{\rm C++}}{n_{\rm C+}}.
$$

三态守恒给出：

$$
n_{\rm C0}:n_{\rm C+}:n_{\rm C++}
=1:r_1:r_1r_2,
$$

因此

$$
\boxed{
x_{\rm C+}^{\rm Saha}
=\frac{r_1}{1+r_1+r_1r_2}
}
$$

代码使用 $n_e^{\rm safe}=\max(n_e,10^{-30}\ \mathrm{cm^{-3}})$，并把最后的 $x_{\rm C+}$ clip 到 $[0,1]$。$S_1,S_2$ 也在 log-space 中计算并把 exponent clip 到 $[-290,290]$。

这里忽略 carbon 自身以及 helium 对 $n_e$ 的贡献，先用 H-only Saha 固定 $n_e$，再解 carbon ratios。

### 5.3 从 ion fraction 得到 C+ number density

固定 gas-phase carbon abundance：

$$
A_{\rm C}=\frac{n_{\rm C,tot}}{n_{\rm H}}=1.6\times10^{-4}.
$$

于是：

$$
n_{\rm C+}=x_{\rm C+}^{\rm Saha}A_{\rm C}n_{\rm H}.
$$

此处 $A_{\rm C}$ 来自 `config.py::A_C`，不是从 CHIANTI abundance table 或 DESPOTIC cold chemistry 读取。

### 5.4 two-level LTE upper-level population

定义：

$$
q(T)=\frac{g_u}{g_l}\exp\!\left(-\frac{T_*}{T}\right).
$$

在 two-level LTE approximation 下：

$$
\frac{n_u}{n_l}=q(T),
\qquad
n_u+n_l=n_{\rm C+},
$$

所以：

$$
\boxed{
n_u=n_{\rm C+}\frac{q(T)}{1+q(T)}
}
$$

对应代码：

```python
r = (_CII_G_U / _CII_G_L) * np.exp(-_CII_T_STAR / T_safe)
n_u = n_Cp * r / (1.0 + r)
```

这里没有调用 fiasco 的 collisional-radiative level-population 或 emissivity API。CHIANTI 的完整 $U_{\rm C+}$ 只用于前一节的 Saha ionization ratio；line upper-level fraction 本身仍使用 ground-term two-level approximation，而不是 $g_u e^{-E_u/kT}/U_{\rm C+}$。

### 5.5 spontaneous emission

每个 upper-level ion 的 photon emission rate 是 $A_{ul}$，每个 photon energy 是 $h\nu_{ul}$，因此：

$$
\boxed{
\epsilon_{\rm Saha+LTE}
=n_u A_{ul}h\nu_{ul}
}
$$

展开后：

$$
\boxed{
\epsilon_{\rm Saha+LTE}
=\left(A_{\rm C}n_{\rm H}x_{\rm C+}^{\rm Saha}\right)
\left[\frac{(g_u/g_l)e^{-T_*/T}}
{1+(g_u/g_l)e^{-T_*/T}}\right]
\left(A_{ul}h\nu_{ul}\right)
}
$$

单位：

$$
\mathrm{cm^{-3}}\times\mathrm{s^{-1}}\times
(\mathrm{erg\,s})\times\mathrm{s^{-1}}
=\mathrm{erg\,s^{-1}\,cm^{-3}}.
$$

---

## 6. Regime III：$T_{\rm QK}\ge10^4$ K，CHIANTI CIE ion fraction + two-level LTE

高温 branch 不再使用 Hydrogen Saha electron density，也不使用 carbon Saha partition-function ratios。它直接采用 fiasco 求得的 CHIANTI CIE fraction：

$$
x_{\rm C+}^{\rm CIE}=x_{\rm C+}^{\rm CIE}(T).
$$

然后：

$$
n_{\rm C+}^{\rm CIE}
=x_{\rm C+}^{\rm CIE}A_{\rm C}n_{\rm H},
$$

$$
n_u^{\rm CIE}
=n_{\rm C+}^{\rm CIE}
\frac{(g_u/g_l)e^{-T_*/T}}
{1+(g_u/g_l)e^{-T_*/T}},
$$

最终：

$$
\boxed{
\epsilon_{\rm CHIANTI\ CIE+LTE}
=n_u^{\rm CIE}A_{ul}h\nu_{ul}
}
$$

或完全展开：

$$
\boxed{
\epsilon_{\rm CHIANTI\ CIE+LTE}
=\left(A_{\rm C}n_{\rm H}x_{\rm C+}^{\rm CIE}(T)\right)
\left[\frac{(g_u/g_l)e^{-T_*/T}}
{1+(g_u/g_l)e^{-T_*/T}}\right]
\left(A_{ul}h\nu_{ul}\right)
}
$$

对应代码：

```python
x_Cp_cie = np.interp(T_safe, _CIE_T_GRID, _CIE_X_CP)
n_Cp_cie = x_Cp_cie * A_C_TOTAL * n_H_sim
n_u_cie  = n_Cp_cie * r / (1.0 + r)
eps_cie  = n_u_cie * _CII_A_UL * h_cgs * _CII_NU_HZ
```

这里 fiasco/CHIANTI 只决定 $x_{\rm C+}^{\rm CIE}(T)$ 和 atomic constants；line excitation 仍然是代码自己的 two-level LTE，不是 CHIANTI collisional-radiative emissivity。

---

## 7. 代码调用与物理量一览

| 目的 | 代码调用 | 得到的物理量 |
|---|---|---|
| C I object | `fiasco.Ion('C 1', T_grid_q)` | $\mathrm{C^0}$ levels、ionization potential |
| C II object | `fiasco.Ion('C 2', T_grid_q)` | $\mathrm{C^+}$ levels、transitions、ionization potential |
| C III object | `fiasco.Ion('C 3', T_grid_q)` | $\mathrm{C^{++}}$ levels |
| Level energy | `ion.levels.energy` | $E_i$；observed 优先、theoretical fallback |
| Statistical weight | `ion.levels.weight` | $g_i=2J_i+1$ |
| Partition function | 手动 `sum(g*exp(-E/kT))` | $U_{\rm C0},U_{\rm C+},U_{\rm C++}$ |
| Ionization potential | `ion.ionization_potential` | $I_{\rm C0},I_{\rm C+}$ |
| 158 μm transition | `(lower_level==1) & (upper_level==2)` | ground-term fine-structure pair |
| Einstein coefficient | `cii.transitions.A[i158]` | $A_{ul}$ |
| Transition energy | `cii.transitions.delta_energy[i158]` | $\Delta E$ |
| Frequency | `delta_energy / astropy.constants.h` | $\nu_{ul}$ |
| Equivalent temperature | `delta_energy / astropy.constants.k_B` | $T_*=\Delta E/k_B$ |
| CIE fractions | `fiasco.Element('carbon', T).equilibrium_ionization` | 全部 carbon ion stages 的 $x_q(T)$ |
| C+ CIE fraction | `xC[:, 1]` | $x_{\rm C+}^{\rm CIE}(T)$ |
| 冷端 DESPOTIC | `lookup.line_field('C+', 'lumPerH', ...)` | $\ell_{\rm CII}^{\rm DSP}$ |

---

## 8. 当前实现包含的关键 assumptions

1. **Cold branch 与 hot branches 的 radiative-transfer treatment 不同。** 预建 LVG DESPOTIC `lumPerH` 已包含 table build 时的 chemistry、level population 和 escape probability；两个 analytic branches 使用 $n_uA_{ul}h\nu$，没有 optical-depth 或 escape-probability factor，相当于 optically thin spontaneous escape。

2. **Warm branch 使用 Saha ionization equilibrium。** 它依赖 $T$ 和 $n_e$，并采用 H-only electron budget；carbon 与 helium 对 $n_e$ 的贡献被忽略。

3. **Hot branch 使用 CHIANTI CIE。** $x_{\rm C+}^{\rm CIE}$ 只依赖 $T$；它不是 Saha fraction，也不依赖 cell density。

4. **两个 analytic branches 都假设 two-level LTE excitation。** upper-level fraction 不依赖 collider density，因此没有 critical-density suppression；这与低密度下的 full collisional-radiative solution 不同。

5. **固定 carbon abundance。** analytic branches 使用 $A_{\rm C}=1.6\times10^{-4}$，不随 metallicity、depletion 或 cell chemistry 改变。

6. **Hard boundaries。** 3000 K 和 $10^4$ K 处没有 blending，也没有强制 continuity。

7. **CHIANTI version 未由代码强制。** 项目说明使用 CHIANTI 10.1，但 `fiasco.Ion(...)` 没有显式传入 HDF5 database path/version；不同本地 database 可能改变 constants、levels 或 CIE rates。

8. **Interpolation 均为 linear-$T$。** partition functions 和 CIE fractions 的 per-cell sampling 都使用 `np.interp(T, ...)`，不是 log-$T$ interpolation。

9. **Field cache。** `('gas', 'C+_luminosity')` 是 Level-1 cached field。修改任何上述公式或 atomic-data source 后，应增加 `pipeline/cache.py::CACHE_SCHEMA_VERSION`，否则旧 HDF5 field cache 可能继续被读取。

---

## 9. 最短端到端流程

```text
rho, T_QK, N_H, dV/dr
        |
        +-- n_H = X_H rho / m_H
        |
        +-- T_QK < 3000 K
        |      -> clamp(n_H, N_H, dV/dr)
        |      -> DESPOTIC table: C+ lumPerH
        |      -> epsilon = n_H_clip * lumPerH
        |
        +-- 3000 K <= T_QK < 1e4 K
        |      -> H Saha -> n_e
        |      -> U_C0, U_C+, U_C++ from CHIANTI levels
        |      -> two-stage carbon Saha -> x_C+
        |      -> n_C+ = A_C n_H x_C+
        |      -> two-level LTE -> n_u
        |      -> epsilon = n_u A_ul h nu
        |
        +-- T_QK >= 1e4 K
               -> CHIANTI rates + CIE solve -> x_C+(T)
               -> n_C+ = A_C n_H x_C+
               -> two-level LTE -> n_u
               -> epsilon = n_u A_ul h nu
```

---

## 10. 外部 API 参考

- [fiasco `Ion` API](https://fiasco.readthedocs.io/en/stable/api/fiasco.Ion.html)
- [fiasco `Element` API](https://fiasco.readthedocs.io/en/stable/api/fiasco.Element.html)
- [fiasco `Levels` API](https://fiasco.readthedocs.io/en/stable/api/fiasco.Levels.html)
- [fiasco `Transitions` API](https://fiasco.readthedocs.io/en/stable/api/fiasco.Transitions.html)
- [CHIANTI atomic database](https://www.chiantidatabase.org/)
- Hydrogen Saha relation：Draine, *Physics of the Interstellar and Intergalactic Medium* (2011), Eq. 3.17。
