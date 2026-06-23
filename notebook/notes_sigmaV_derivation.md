# σ_v 公式推导：从连续统计二阶矩到 cell 级质量加权

目标：解释 `mass_weighted_sigma` 和 `_moment_sigma` 这两种写法**本质上是同一个公式**，只是"离散化方式"不同。

---

## Part 1：起点 —— 连续概率分布的二阶矩

在概率论里，速度 v 的概率密度函数 P(v) 满足：

$$
\int P(v)\,dv = 1
$$

均值和方差的定义：

$$
\langle v\rangle = \int v\,P(v)\,dv
$$

$$
\sigma^2 = \int (v - \langle v\rangle)^2\,P(v)\,dv
$$

这两个就是 "**一阶矩**" 和 "**中心二阶矩**"。

---

## Part 2：非归一化分布函数 f(v)

实际物理里，我们手上的不是归一化的 P(v)，而是类似"强度分布"、"质量分布密度"之类的 f(v)，**不一定积分为 1**。

定义三个矩：

$$
m_0 = \int f(v)\,dv \qquad (\text{总量}, \text{比如总光度/总质量})
$$

$$
m_1 = \int v\,f(v)\,dv
$$

$$
m_2 = \int (v - \langle v\rangle)^2\,f(v)\,dv
$$

显然归一化 `P(v) = f(v) / m_0`，所以：

$$
\boxed{\langle v\rangle = \frac{m_1}{m_0}, \qquad \sigma^2 = \frac{m_2}{m_0}}
$$

这就是**"矩方法"最原始的定义**。后面两种离散化都是从这个公式出发。

---

## Part 3：离散化 A —— 观测谱上的 moment method

观测者拿到的是**一条已经在规则 v grid 上采样的光谱**：

- v grid：$v_0, v_1, \dots, v_{N-1}$，每个 channel 等距间隔 $\Delta v$
- 强度：$f_k \equiv f(v_k)$，是每个 channel 的光谱值（单位：比如 erg/s/Hz/cm² 或 dΣ/dv）

### 步骤 1：用矩形积分近似连续积分

$$
\int f(v)\,dv \;\approx\; \sum_{k=0}^{N-1} f_k \cdot \Delta v
$$

三个矩：

$$
m_0 \approx \sum_k f_k \Delta v
$$

$$
m_1 \approx \sum_k v_k f_k \Delta v
$$

$$
m_2 \approx \sum_k (v_k - \langle v\rangle)^2 f_k \Delta v
$$

### 步骤 2：Δv 消掉

因为 Δv 是常数，分子分母里都出现：

$$
\langle v\rangle = \frac{m_1}{m_0} = \frac{\sum_k v_k f_k \Delta v}{\sum_k f_k \Delta v} = \frac{\sum_k v_k f_k}{\sum_k f_k}
$$

$$
\sigma^2 = \frac{\sum_k (v_k - \langle v\rangle)^2 f_k}{\sum_k f_k}
$$

### 步骤 3：定义归一化权重 w_k

$$
w_k \equiv \frac{f_k}{\sum_j f_j} \quad \Longrightarrow \quad \sum_k w_k = 1
$$

带入：

$$
\boxed{\langle v\rangle = \sum_k w_k v_k, \qquad \sigma^2 = \sum_k w_k (v_k - \langle v\rangle)^2}
$$

### 代码对照

这就是 `integrated_spectrum.py:51-59` 的 `_moment_sigma`：

```python
def _moment_sigma(v, spec):
    total = spec.sum()          #  Σ f_k        （m_0 / Δv）
    w      = spec / total       #  w_k = f_k / Σf
    v_mean = np.sum(v * w)      #  <v> = Σ w_k v_k
    sigma  = np.sqrt(np.sum((v - v_mean)**2 * w))   #  σ² = Σ w_k (v_k - <v>)²
```

---

## Part 4：离散化 B —— cell 级质量加权

现在换另一种视角。我们不再把速度轴切成等距 channel，而是**把每个模拟 cell 看作一个离散 sample**：

- cell $i$ 的速度：$v_i$（一个具体值，比如 `velocity_x[i,j,k]`）
- cell $i$ 的质量：$m_i = \rho_i \cdot V_{\rm cell}$

### 步骤 1：把 f(v) 建模成冲激列

真实的"质量对速度的分布"是**离散**的——每个 cell 在自己速度处贡献一个 $\delta$ 函数：

$$
f(v) = \sum_i m_i \,\delta(v - v_i)
$$

这里 $\delta(v-v_i)$ 是 Dirac delta：只有 $v = v_i$ 时它是无穷大，其他地方 0，满足 $\int \delta(v-v_i)\,dv = 1$。

### 步骤 2：代入连续矩

用 delta 函数的筛选性质 $\int g(v)\,\delta(v-v_i)\,dv = g(v_i)$：

$$
m_0 = \int f(v)\,dv = \sum_i m_i \int \delta(v-v_i)\,dv = \sum_i m_i
$$

$$
m_1 = \int v\,f(v)\,dv = \sum_i m_i \int v\,\delta(v-v_i)\,dv = \sum_i m_i v_i
$$

$$
m_2 = \int (v-\langle v\rangle)^2 f(v)\,dv = \sum_i m_i (v_i-\langle v\rangle)^2
$$

### 步骤 3：Part 2 的 boxed 公式

$$
\langle v\rangle = \frac{m_1}{m_0} = \frac{\sum_i m_i v_i}{\sum_i m_i}
$$

$$
\sigma^2 = \frac{m_2}{m_0} = \frac{\sum_i m_i (v_i - \langle v\rangle)^2}{\sum_i m_i}
$$

### 步骤 4：用 ρ 代 m（均匀网格 V_cell 消掉）

因为模拟是均匀网格，$V_{\rm cell}$ 对所有 cell 都一样：

$$
m_i = \rho_i \cdot V_{\rm cell}
$$

$$
\frac{\sum_i m_i v_i}{\sum_i m_i} = \frac{V_{\rm cell}\sum_i \rho_i v_i}{V_{\rm cell}\sum_i \rho_i} = \frac{\sum_i \rho_i v_i}{\sum_i \rho_i}
$$

$V_{\rm cell}$ 完全消掉。定义权重：

$$
w_i \equiv \frac{\rho_i}{\sum_j \rho_j} \quad \Longrightarrow \quad \sum_i w_i = 1
$$

最终：

$$
\boxed{\langle v\rangle = \sum_i w_i v_i, \qquad \sigma^2 = \sum_i w_i (v_i - \langle v\rangle)^2}
$$

### 代码对照

这就是 `pipeline/utils.py` 的 `mass_weighted_sigma`：

```python
def mass_weighted_sigma(vel_kms, rho):
    total  = rho.sum()                     # Σ ρ_j
    w      = rho / total                   # w_i = ρ_i / Σρ
    v_mean = np.sum(vel_kms * w)           # <v> = Σ w_i v_i
    sigma  = np.sqrt(np.sum((vel_kms - v_mean)**2 * w))   # σ² = Σ w_i (v_i - <v>)²
```

---

## Part 5：两者对比——结构一样，含义不同

| | moment 法（谱） | 质量加权（cell） |
|---|----------------|--------------------|
| f(v) 的模型 | 连续光谱，在规则 v grid 上采样 | 离散 delta 列：每个 cell 贡献一个 δ(v−v_i) |
| 抽样点 | 固定的 $v_k$（等距 channel） | 每个 cell 自己的 $v_i$（不规则散布） |
| 权重 $w$ | 该 channel 的光度 $f_k$ | 该 cell 的质量 $m_i \propto \rho_i$ |
| 求和范围 | $k = 0 \dots N_{\rm chan}-1$ | $i$ = 所有 cell |
| 含热展宽？ | ✅ 含（谱里已 spread 过 thermal σ） | ❌ 不含（cell v 是 bulk v） |
| 物理含义 | 观测者从线剖面提取的 σ | 模拟底层气体的真实湍流/bulk σ |

**公式结构完全一样**：`σ² = Σ w (v − <v>)²`。只是 `v` 和 `w` 来自不同的东西。

---

## Part 6：一个小数字例子

假设只有 4 个 cell：

| cell | v [km/s] | ρ |
|------|---------|---|
| 0 | −10 | 1 |
| 1 |  0 | 4 |
| 2 | +10 | 4 |
| 3 | +20 | 1 |

### 用 cell 法（`mass_weighted_sigma`）

$$
\sum \rho = 1+4+4+1 = 10
$$

$$
w = [0.1,\,0.4,\,0.4,\,0.1]
$$

$$
\langle v\rangle = 0.1\cdot(-10) + 0.4\cdot 0 + 0.4\cdot 10 + 0.1\cdot 20 = -1 + 0 + 4 + 2 = 5\text{ km/s}
$$

$$
\sigma^2 = 0.1(-15)^2 + 0.4(-5)^2 + 0.4(5)^2 + 0.1(15)^2 = 22.5 + 10 + 10 + 22.5 = 65
$$

$$
\sigma = \sqrt{65} \approx 8.06\text{ km/s}
$$

### 用 moment 法（`_moment_sigma`）

假设我把同样的数据做成一个"谱"：在 v = [−10, 0, +10, +20] 这 4 个 channel 上，强度 spec = [1, 4, 4, 1]（用 ρ 当 spec 值，相当于模拟的 mass-weighted 分布正好和谱一样）。

代入公式——和上面一模一样：$\sigma \approx 8.06$ km/s。

**两种方法对同一个数据得到完全相同的答案**——因为它们本质是同一个二阶矩公式。

---

## Part 7：什么时候选哪个

- **你要"观测者测到的线宽"** → 用 moment 法，作用在 species 光度加权的谱上（已经含热展宽和仪器响应）
- **你要"模拟真实动力学的 σ"** → 用 cell 法，作用在 velocity 场上，用 ρ 做权重

`PhaseSigmaVTask` 要的是后者，所以用 `mass_weighted_sigma`。
