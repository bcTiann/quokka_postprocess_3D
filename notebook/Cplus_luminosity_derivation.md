# `[C II] 158 μm` 单个 cell 体积发射率的完整计算流程

本文档描述当前 `quokka2s` 代码如何为每个 simulation cell 计算

$$
\epsilon_{\mathrm{CII}}
\quad [\mathrm{erg\,s^{-1}\,cm^{-3}}].
$$

> 更新：$T\ge1.307\times10^4$ K 时强制
> $n_{\rm C}=n_{\rm C+}+n_{\rm C++}$，并用 C$^+\leftrightarrow$C$^{++}$
> 的第二级 Saha 平衡计算 $x_{\rm C+}=n_e/(n_e+S_2)$。高温 excitation
> 固定使用 two-level LTE upper-level population。
> 所有显式 C$^+$ temperature consumers（regime selection、Saha、LTE 与
> thermal width）统一使用每个 cell 的 $T_{\rm QUOKKA}$。冷端 DESPOTIC
> `lumPerH` 是预建 thermal/chemical/LVG solution，没有独立 runtime temperature 轴。

这里的 yt field 名为 `('gas', 'C+_luminosity')`，但它的物理量实际是 **volumetric emissivity**，不是已经乘过 cell volume 的 luminosity。本文到 $\epsilon_{\mathrm{CII}}$ 为止，不包含 cell luminosity、thermal broadening、Doppler shift、spectral cube 或 integrated spectrum。

当前实现位于：

- field 注册：`src/quokka2s/pipeline/prep/physics_fields.py:1064-1071`
- 主计算函数：`_Cplus_luminosity()`
- CHIANTI/fiasco atomic-data 初始化：同文件 `101-205`
- mean-molecular-weight electron fraction：`src/quokka2s/line_regimes.py`
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
& 3000\ \mathrm{K}\le T_{\rm QK}<1.307\times10^4\ \mathrm{K},\\[4pt]
\epsilon_{\rm high},
& T_{\rm QK}\ge1.307\times10^4\ \mathrm{K}.
\end{cases}
$$

对应代码：

```python
hot = T_qk >= T_QK_TWO_REGIME_K       # 3000 K

eps = np.where(
    T_qk >= T_CIE_K,                   # 1.307e4 K
    eps_high,
    np.where(hot, eps_hot, eps_cold),
)
```

因此：

- $T_{\rm QK}=3000\ \mathrm{K}$ 属于 `Saha+LTE` branch；
- $T_{\rm QK}=1.307\times10^4\ \mathrm{K}$ 属于 high-temperature two-stage Saha branch；
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
- $X_{\rm H}=1/(1+0.1\times3.971)\simeq0.71577$：hydrogen mass fraction；
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

## 5. Regime II：$3000\le T_{\rm QK}<1.307\times10^4$ K，Saha ionization + two-level LTE

本 regime 直接使用 QUOKKA temperature：

$$
T=T_{\rm QK}
$$

### 5.1 QUOKKA mean molecular weight 给 electron density

代码由 simulation 的 internal-energy density、mass density 与温度反推

$$
\frac{1}{\mu}
=\frac{(\gamma-1)m_{\rm H}e_{\rm int}}
{\rho k_{\rm B}T}.
$$

把 $x_e$ 明确定义为所有来源的总 free-electron fraction，

$$
x_e=\frac{1/\mu-X-Y/4}{X},
\qquad
n_e=x_en_{\rm H},
$$

其中 $X=1/(1+0.1\times3.971)$、$Z=0.02$、$Y=1-X-Z$；组成仍包含 $Z$，
但这个工作 EOS 反演不单独加入 $Z/\bar A_Z$。代码直接使用这个公式得到的 $x_e$，不施加上下限。
这个 thermodynamic inversion 不区分电子来自 H、He 或 metals。

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

代码直接使用由 mean molecular weight 得到的 $n_e$ 计算 $S_1/n_e$ 和
$S_2/n_e$，不对 $n_e$ 或 $x_{\rm C+}$ 施加 floor 或 clip。$S_1$ 和 $S_2$
也直接按照上面的 Saha 公式计算。随后用这两个 carbon ratios 和三态守恒
求解 $x_{\rm C+}$。

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

## 6. Regime III：$T_{\rm QK}\ge1.307\times10^4$ K，两态 C$^+$/C$^{++}$ Saha

高温 branch 强制：

$$
n_{\rm C}=n_{\rm C+}+n_{\rm C++},
\qquad n_{\rm C0}=0.
$$

只保留第二级 Saha 平衡：

$$
\frac{n_e n_{\rm C++}}{n_{\rm C+}}=S_2(T),
\qquad
\frac{n_{\rm C++}}{n_{\rm C+}}=\frac{S_2(T)}{n_e}.
$$

与 carbon conservation 联立得到：

$$
\boxed{
x_{\rm C+}^{\rm high}
=\frac{n_{\rm C+}}{n_{\rm C}}
=\frac{1}{1+S_2(T)/n_e}
=\frac{n_e}{n_e+S_2(T)}
}
$$

以及

$$
x_{\rm C++}^{\rm high}=1-x_{\rm C+}^{\rm high}.
$$

这里的 $n_e=x_en_H$ 与 intermediate branch 相同，来自 QUOKKA mean molecular
weight；carbon 自身的电子贡献忽略。随后：

$$
n_{\rm C+}^{\rm high}=x_{\rm C+}^{\rm high}A_{\rm C}n_{\rm H}.
$$

高温 C$^+$ upper-level fraction 固定采用 two-level LTE：

$$
n_u^{\rm high}=n_{\rm C+}^{\rm high}
\frac{(g_u/g_l)e^{-T_*/T}}
{1+(g_u/g_l)e^{-T_*/T}},
$$

最终

$$
\boxed{
\epsilon_{\rm high}=n_u^{\rm high}A_{ul}h\nu_{ul}
}.
$$

对应代码：

```python
x_Cp_high = n_e / (n_e + S_C2)
n_Cp_high = x_Cp_high * A_C_TOTAL * n_H_sim
n_u_high  = n_Cp_high * upper_fraction_high
eps_high  = n_u_high * _CII_A_UL * h_cgs * _CII_NU_HZ
```

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
| High-T C+ fraction | `n_e / (n_e + S_C2)` | $n_e/(n_e+S_2)$ |
| 冷端 DESPOTIC | `lookup.line_field('C+', 'lumPerH', ...)` | $\ell_{\rm CII}^{\rm DSP}$ |

---

## 8. 当前实现包含的关键 assumptions

1. **Cold branch 与 hot branches 的 radiative-transfer treatment 不同。** 预建 LVG DESPOTIC `lumPerH` 已包含 table build 时的 chemistry、level population 和 escape probability；两个 analytic branches 使用 $n_uA_{ul}h\nu$，没有 optical-depth 或 escape-probability factor，相当于 optically thin spontaneous escape。

2. **Warm branch 使用 Saha ionization equilibrium。** 它依赖 $T$ 和 $n_e$；
   $n_e=x_en_H$ 来自 mean-molecular-weight inversion，$x_e$ 是所有来源的
   total electron fraction，并直接使用公式结果而不施加上下限。

3. **Hot branch 使用两态 Saha。** 强制 $n_C=n_{C+}+n_{C++}$，所以
   $x_{C+}=n_e/(n_e+S_2)$，同时依赖 $T$ 与由 QUOKKA mean molecular weight
   推得的 $n_e$。

4. **两个 analytic branches 都使用 two-level LTE excitation。**

5. **固定 carbon abundance。** analytic branches 使用 $A_{\rm C}=1.6\times10^{-4}$，不随 metallicity、depletion 或 cell chemistry 改变。

6. **Hard boundaries。** 3000 K 和 $1.307\times10^4$ K 处没有 blending，也没有强制 continuity。

7. **CHIANTI version 未由代码强制。** 项目说明使用 CHIANTI 10.1，但 `fiasco.Ion(...)` 没有显式传入 HDF5 database path/version；不同本地 database 可能改变 constants、levels 或 CIE rates。

8. **Partition-function interpolation 是 linear-$T$。** per-cell sampling 使用
   `np.interp(T, ...)`，不是 log-$T$ interpolation。

9. **Field cache。** `C+_luminosity` 是 Level-1 cached field。修改任何上述公式或 atomic-data source 后，应增加
   `pipeline/cache.py::CACHE_SCHEMA_VERSION`，否则旧 HDF5 field cache 可能继续被读取。

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
        +-- 3000 K <= T_QK < 1.307e4 K
        |      -> QUOKKA mean molecular weight -> n_e
        |      -> U_C0, U_C+, U_C++ from CHIANTI levels
        |      -> two-stage carbon Saha -> x_C+
        |      -> n_C+ = A_C n_H x_C+
        |      -> two-level LTE -> n_u
        |      -> epsilon = n_u A_ul h nu
        |
        +-- T_QK >= 1.307e4 K
               -> QUOKKA mean molecular weight -> n_e
               -> enforce n_C = n_C+ + n_C++
               -> second Saha equilibrium -> x_C+ = n_e/(n_e+S_2)
               -> n_C+ = A_C n_H x_C+
               -> two-level LTE excitation -> n_u
               -> epsilon = n_u A_ul h nu
```

---

## 10. 外部 API 参考

- [fiasco `Ion` API](https://fiasco.readthedocs.io/en/stable/api/fiasco.Ion.html)
- [fiasco `Levels` API](https://fiasco.readthedocs.io/en/stable/api/fiasco.Levels.html)
- [fiasco `Transitions` API](https://fiasco.readthedocs.io/en/stable/api/fiasco.Transitions.html)
- [CHIANTI atomic database](https://www.chiantidatabase.org/)
