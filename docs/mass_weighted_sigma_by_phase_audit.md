# `mass_weighted_sigma` / `mass_weighted_sigma_by_phase` 审计

这份文档专门解释两个函数：

- `src/quokka2s/pipeline/utils.py::mass_weighted_sigma`
- `src/quokka2s/pipeline/utils.py::mass_weighted_sigma_by_phase`

目标是把代码里的 $\sigma$ 和常见的方差公式对齐，并回答一个具体问题：

> 现在的 `mass_weighted_sigma_by_phase` 到底写得对不对？

结论先说：

1. `mass_weighted_sigma` 写得对。它算的是一个速度分量的 **mass-weighted population standard deviation**。
2. `mass_weighted_sigma_by_phase` 数学上也自洽，但它算的不是“每个 phase 自己内部的 velocity dispersion”。它算的是：**phase 内的 cell 相对于全体气体 global mean velocity 的 RMS 宽度**。
3. 所以它会把两个东西加在一起：phase 内部速度散布，以及这个 phase 的整体 bulk velocity 相对全体气体 bulk velocity 的偏移。
4. 如果你的科学目标是“每个 phase 内部的 turbulent/internal dispersion”，现在这行就应该改成减去 phase 自己的 mean。

运行口径备注：当前项目应该用 `L_ext = 15 kpc`。这两个 sigma 公式本身不直接读 `L_ext`，但 `Build_VelocityPhase` 用 `temperature_two_regime` 分 phase，而这个温度场可能通过 `temperature_despotic -> column_density_H -> L_ext` 间接受影响。所以要比较真实结果时，`Build_VelocityPhase` 的 cache/output 应该来自 `L_ext=15 kpc` 的运行。

---

## 1. 先对齐你熟悉的方差公式

你写的核心想法是对的：

$$
\sigma^2
= \mathbb{E}[v^2] - \mathbb{E}[v]^2 .
$$

但需要注意一个 bookkeeping：如果左边写普通求和，就要除以样本数或总权重。对于 unweighted population：

$$
\mu = \langle v\rangle = \frac{1}{N}\sum_i v_i ,
$$

$$
\sigma^2
= \frac{1}{N}\sum_i (v_i-\mu)^2
= \frac{1}{N}\sum_i v_i^2 - \mu^2
= \mathbb{E}[v^2] - \mathbb{E}[v]^2 .
$$

所以严格地说，不是

$$
\sum_i (v_i-\mu)^2
= \sum_i v_i^2 - \mu^2 ,
$$

而是

$$
\frac{1}{N}\sum_i (v_i-\mu)^2
= \frac{1}{N}\sum_i v_i^2 - \mu^2 .
$$

或者直接把 $\sum$ 理解成 expectation：

$$
\mathbb{E}[g] = \frac{1}{N}\sum_i g_i .
$$

代码里用的是 weighted expectation。它先把权重归一化到和为 1，然后所有求和都等价于 $\mathbb{E}_w[\cdot]$。

---

## 2. 什么样的 weighted variance 是合法的？

先给每个 cell 一个 raw weight $a_i$。weighted variance 合法需要：

$$
a_i \ge 0,\qquad \sum_i a_i > 0,
$$

并且权重是 finite 的。然后归一化：

$$
w_i = \frac{a_i}{\sum_j a_j},
\qquad
\sum_i w_i = 1.
$$

这样 $\{w_i\}$ 就是一个离散概率分布。weighted expectation 定义为：

$$
\mathbb{E}_w[g] = \sum_i w_i g_i .
$$

weighted mean 是：

$$
\mu_w = \mathbb{E}_w[v] = \sum_i w_i v_i .
$$

weighted variance 是：

$$
\sigma_w^2
= \sum_i w_i (v_i-\mu_w)^2
= \mathbb{E}_w[v^2] - \mathbb{E}_w[v]^2 .
$$

如果直接用没有归一化的 raw weight，也可以写成：

$$
\mu_w
= \frac{\sum_i a_i v_i}{\sum_i a_i},
$$

$$
\sigma_w^2
= \frac{\sum_i a_i (v_i-\mu_w)^2}{\sum_i a_i}.
$$

这说明几个重要点：

- 权重整体乘一个常数，不改变结果。
- 负权重不适合拿来定义 variance。
- 零权重可以，它只是表示这个 cell 不贡献。
- 这里算的是 population standard deviation，没有 Bessel correction，也就是没有除以 $N-1$。这是合理的，因为我们不是用随机样本估计未知母体，而是在描述这个 simulation cube 本身。

---

## 3. 为什么代码用 `rho` 当 mass weight？

物理上最自然的 mass weight 是 cell mass：

$$
a_i = m_i = \rho_i \Delta V_i .
$$

但当前 pipeline 用的是 full-domain uniform covering grid。每个 cell 体积相同：

$$
\Delta V_i = \Delta V_{\rm cell}.
$$

归一化时这个常数会抵消：

$$
w_i
= \frac{\rho_i \Delta V_{\rm cell}}
       {\sum_j \rho_j \Delta V_{\rm cell}}
= \frac{\rho_i}{\sum_j \rho_j}.
$$

所以在当前代码路径里，用 `rho` 直接当权重是合法的。这个结论也适用于现在的 block-mean downsampled dataset，因为 downsample 后仍然是 uniform grid。

如果以后换成 unequal-volume cell，比如真正的 AMR cell，那么只用 `rho` 就不对了，应该用：

$$
a_i = \rho_i \Delta V_i .
$$

---

## 4. `mass_weighted_sigma` 在算什么？

代码：

```python
def mass_weighted_sigma(vel_kms, rho):
    total = rho.sum()
    if total <= 0:
        return float('nan'), float('nan')
    w = rho / total
    v_mean = float(np.sum(vel_kms * w))
    sigma  = float(np.sqrt(np.sum((vel_kms - v_mean) ** 2 * w)))
    return v_mean, sigma
```

数学上就是：

$$
\mu
= \frac{\sum_i \rho_i v_i}{\sum_i \rho_i},
$$

$$
\sigma
= \sqrt{
    \frac{\sum_i \rho_i (v_i-\mu)^2}{\sum_i \rho_i}
  }.
$$

这就是一个速度分量的 mass-weighted population standard deviation。

代码用的是 centered form：

$$
\sum_i w_i(v_i-\mu)^2.
$$

这通常比 raw-moment form 更稳：

$$
\mathbb{E}_w[v^2] - \mathbb{E}_w[v]^2.
$$

因为 raw-moment form 可能是在两个很接近的大数之间相减，数值上容易有 cancellation。

### 手算例子

设：

$$
v = [10,20,30]\ {\rm km\,s^{-1}},
\qquad
\rho = [1,1,2].
$$

则：

$$
w = [0.25,0.25,0.5],
$$

$$
\mu
=0.25(10)+0.25(20)+0.5(30)
=22.5.
$$

方差：

$$
\sigma^2
=0.25(10-22.5)^2
 +0.25(20-22.5)^2
 +0.5(30-22.5)^2
=68.75.
$$

所以：

$$
\sigma = \sqrt{68.75}=8.29156\ {\rm km\,s^{-1}}.
$$

---

## 5. `mass_weighted_sigma_by_phase` 当前在算什么？

当前代码核心是：

```python
v_global, _ = mass_weighted_sigma(vel_kms, rho)

for phase, mask in masks.items():
    m_p = rho[mask]
    tot = float(m_p.sum())
    if tot > 0 and np.isfinite(v_global):
        w      = m_p / tot
        v_mean = float(np.sum(vel_kms[mask] * w))
        sigma  = float(np.sqrt(np.sum((vel_kms[mask] - v_global) ** 2 * w)))
```

先定义全体气体的 mass-weighted mean velocity：

$$
\mu_{\rm all}
= \frac{\sum_{i\in{\rm all}}\rho_i v_i}
       {\sum_{i\in{\rm all}}\rho_i}.
$$

然后对某个 phase $P$，定义 phase 内部归一化权重：

$$
w_i^P
= \frac{\rho_i}{\sum_{j\in P}\rho_j},
\qquad i\in P.
$$

代码会报告这个 phase 自己的 mean：

$$
\mu_P
= \sum_{i\in P} w_i^P v_i.
$$

但注意，当前 `sigma` 不是围绕 $\mu_P$ 算的，而是围绕 $\mu_{\rm all}$ 算的：

$$
\boxed{
\sigma_{P,{\rm global}}
= \sqrt{
    \sum_{i\in P} w_i^P (v_i-\mu_{\rm all})^2
  }
}.
$$

这是这份文档最关键的一行。

---

## 6. 它不是 phase 内部 dispersion

如果我们要算 phase 自己内部的 velocity dispersion，应该是：

$$
\sigma_{P,{\rm own}}
= \sqrt{
    \sum_{i\in P} w_i^P (v_i-\mu_P)^2
  }.
$$

当前实现算的是 $\sigma_{P,{\rm global}}$。这两个量的关系是：

$$
\boxed{
\sigma_{P,{\rm global}}^2
= \sigma_{P,{\rm own}}^2
 +(\mu_P-\mu_{\rm all})^2
}.
$$

推导如下。把每个 cell 相对 global mean 的偏差拆开：

$$
v_i-\mu_{\rm all}
= (v_i-\mu_P)+(\mu_P-\mu_{\rm all}).
$$

平方后乘以 $w_i^P$ 并对 phase 内 cell 求和：

$$
\sum_{i\in P} w_i^P(v_i-\mu_{\rm all})^2
=
\sum_{i\in P} w_i^P(v_i-\mu_P)^2
+2(\mu_P-\mu_{\rm all})\sum_{i\in P}w_i^P(v_i-\mu_P)
+(\mu_P-\mu_{\rm all})^2\sum_{i\in P}w_i^P.
$$

中间项为 0，因为围绕 phase mean 的加权偏差和为 0：

$$
\sum_{i\in P}w_i^P(v_i-\mu_P)=0.
$$

最后一项里的权重和为 1：

$$
\sum_{i\in P}w_i^P = 1.
$$

所以：

$$
\sigma_{P,{\rm global}}^2
= \sigma_{P,{\rm own}}^2
 +(\mu_P-\mu_{\rm all})^2.
$$

物理含义：

- $\sigma_{P,{\rm own}}$ 是这个 phase 内部的速度散布。
- $|\mu_P-\mu_{\rm all}|$ 是这个 phase 整体相对全体气体 bulk velocity 的偏移。
- 当前代码把这两者以平方和的方式合在一个 `sigma` 里。

所以如果某个 phase 内部很窄，但整体相对全体气体在 streaming，当前 `sigma` 仍然会变大。

---

## 7. 所以当前实现到底对不对？

这取决于你想让 `sigma` 表示什么。

### 如果你要的是 global-frame phase width，当前实现是对的

当前代码回答的问题是：

> 只看 phase $P$ 里的 gas，它们在全体气体 bulk rest frame 里的 mass-weighted RMS velocity width 是多少？

这个定义适合做“同一个 observer/global frame 下各 phase 的 apparent LOS width”。

它还有一个很好的 consistency check。设 phase mass fraction 为：

$$
f_P = \frac{M_P}{M_{\rm all}}.
$$

因为所有 phase 都围绕同一个 $\mu_{\rm all}$ 算宽度，所以：

$$
\boxed{
\sigma_{\rm all}^2
= \sum_P f_P\,\sigma_{P,{\rm global}}^2
}.
$$

这说明当前 phase sigma 的 mass-fraction 加权平方平均能还原 total gas variance。

### 如果你要的是 phase-internal turbulence，当前实现不对

如果你想回答的是：

> 每个 phase 减去自己的 bulk motion 后，内部速度散布有多大？

那么当前代码就不该减 `v_global`，而应该减 `v_mean`：

```python
sigma = float(np.sqrt(np.sum((vel_kms[mask] - v_mean) ** 2 * w)))
```

这种定义下，total variance 的分解是：

$$
\sigma_{\rm all}^2
= \sum_P f_P
   \left[
     \sigma_{P,{\rm own}}^2
     +(\mu_P-\mu_{\rm all})^2
   \right].
$$

也就是说，如果切换到 phase-internal sigma，bulk offset 项最好单独输出：

$$
\Delta v_P = \mu_P-\mu_{\rm all}.
$$

这样才能同时看见“内部散布”和“phase 整体 streaming”。

---

## 8. 当前代码最容易误读的地方

现在输出 dict 里：

```python
v_mean = phase's own mean
sigma  = RMS around global mean
```

也就是说，`v_mean` 不是 `sigma` 实际减掉的中心。

这在数学上没问题，但命名和图注上很容易让人误会。更清楚的名字会是：

- `sigma_about_global_mean`
- `sigma_global_frame`
- `phase RMS about all-gas mean`

如果图上写的是 “phase internal dispersion”，那就和当前代码不一致。

---

## 9. 最小数值例子

仍然用：

$$
v=[10,20,30],
\qquad
\rho=[1,1,2].
$$

全体 mean 是：

$$
\mu_{\rm all}=22.5.
$$

设 phase `A` 包含前两个 cell：

$$
v_A=[10,20],
\qquad
\rho_A=[1,1],
\qquad
w_A=[0.5,0.5].
$$

phase 自己的 mean：

$$
\mu_A=15.
$$

phase-internal variance：

$$
\sigma_{A,{\rm own}}^2
=0.5(10-15)^2+0.5(20-15)^2
=25.
$$

当前代码的 global-frame phase variance：

$$
\sigma_{A,{\rm global}}^2
=0.5(10-22.5)^2+0.5(20-22.5)^2
=81.25.
$$

两者关系：

$$
25+(15-22.5)^2
=25+56.25
=81.25.
$$

所以两个公式都合法，但它们回答的是不同问题。

---

## 10. 最终判断

`mass_weighted_sigma`：

- 对当前 uniform-grid workflow 是正确的。
- 它就是一个速度分量的 mass-weighted population standard deviation。

`mass_weighted_sigma_by_phase`：

- 如果目标是 “phase 在全体气体 rest frame 里的 RMS width”，当前实现正确。
- 如果目标是 “phase 内部 turbulent/internal dispersion”，当前实现不符合这个定义。
- 当前实现应该在文档、变量名或图注里明确写成 global-frame sigma，否则很容易被误读。

如果要改成 phase-internal dispersion，核心改动只有一行：

```python
sigma = float(np.sqrt(np.sum((vel_kms[mask] - v_mean) ** 2 * w)))
```

如果还关心 phase 相对全体气体的 bulk motion，建议同时输出：

$$
\Delta v_P = \mu_P-\mu_{\rm all}.
$$
