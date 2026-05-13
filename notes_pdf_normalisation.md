# 从直方图到 PDF：为什么要除以 Δv

## 写这个 note 的原因

在 `PhaseSigmaV_hist.png` 里，目前每个 panel 的纵轴是

```
ax.hist(v_rel, bins=bins, weights=rho[m], ...)
```

也就是「每个 bin 里所有 cell 的密度之和」`Σ ρ_i`。这个值**依赖于 bin 的宽度** Δv：你把 120 个 bin 改成 240 个，每根柱子的高度大概会减半。我们想要的是一个**和 bin 数无关、形状有物理意义的曲线**，叫 PDF（probability density function，概率密度函数）。

下面从最基础讲起。

---

## 1. 最朴素的直方图：计数

假设我有 $N$ 个 cell 的速度 $\{v_1, v_2, \ldots, v_N\}$。把速度轴切成宽度 $\Delta v$ 的小区间 $[v_k, v_k + \Delta v)$，记第 $k$ 个 bin 里有 $N_k$ 个 cell。

这就是 `plt.hist(v)` 默认画的东西：

$$
\text{柱子高度} \;=\; N_k.
$$

**问题**：如果我把 bin 加倍（$\Delta v \to \Delta v / 2$），每个 bin 大概只装一半的点，所以每根柱子高度也大概减半。**柱子的高度本身没有物理意义**，它只是「在这个特定 bin 数下，落进来多少个点」。换不同 bin 数的图就不一样高。

---

## 2. 加权直方图：每个点不再贡献「1」，而贡献一个权重 $w_i$

现在我要的不是「有多少 cell 速度在这个 bin」，而是**「这个 bin 里有多少质量（或多少密度）」**。所以我给每个 cell 一个权重 $w_i$。在我们的代码里：

$$
w_i \;=\; \rho_i.
$$

（严格来说，这单元体积都一样，所以 $\rho_i$ 正比于 $m_i$ —— 用 $\rho$ 还是 $m$ 出来的形状是一样的。）

加权直方图的柱高是：

$$
W_k \;=\; \sum_{i \in \text{bin } k} w_i \;=\; \sum_{i \in \text{bin } k} \rho_i.
$$

这就是当前代码 `ax.hist(..., weights=rho[m], ...)` 画出来的东西，单位是密度的单位（g cm⁻³，因为是密度的求和）。

**还是有 bin-dependence 问题**：bin 加倍 → 每个 bin 装的 cell 减半 → 柱子高度减半。

---

## 3. 什么叫 PDF？

PDF $f(v)$ 是一个函数，物理意义是「**单位速度区间内的概率密度**」。它的核心定义性质是：

$$
\boxed{\;\int_{-\infty}^{+\infty} f(v)\, dv \;=\; 1.\;}
$$

也就是说，把整条曲线下面的面积加起来等于 1。注意是**面积**，不是柱子高度之和。

「面积」是 (柱高) × (柱宽 Δv) 的总和。所以柱高的单位必然是 **「1 / 速度单位」**，比如 $(\mathrm{km/s})^{-1}$，这样乘上 $\Delta v$（单位 km/s）才能变成无量纲的概率。

---

## 4. 从加权直方图到 PDF

我有两个量：

- $W_k = \sum_{i \in k} w_i$（这个 bin 的总权重）
- $W_{\rm tot} = \sum_i w_i$（所有 cell 的总权重）

我想构造一个估计 $\hat f(v_k) \approx f(v_k)$，让 $\sum_k \hat f(v_k) \cdot \Delta v = 1$。

**第一步：归一化总权重**

$$
\frac{W_k}{W_{\rm tot}}
$$

这是「这个 bin 占总质量的比例」。所有 bin 加起来 = 1（无量纲）。但这还不是 PDF，因为它对 bin 宽度敏感（bin 加倍 → 每个比例减半）。

**第二步：除以 bin 宽度 Δv**

$$
\hat f(v_k) \;=\; \frac{1}{\Delta v}\,\frac{W_k}{W_{\rm tot}} \;=\; \frac{W_k}{W_{\rm tot}\,\Delta v}.
$$

这个量的单位是 $(\mathrm{km/s})^{-1}$，正是 PDF 该有的单位。验证一下面积：

$$
\sum_k \hat f(v_k) \cdot \Delta v
\;=\; \sum_k \frac{W_k}{W_{\rm tot}\,\Delta v} \cdot \Delta v
\;=\; \frac{1}{W_{\rm tot}}\sum_k W_k
\;=\; \frac{W_{\rm tot}}{W_{\rm tot}} \;=\; 1. \;\checkmark
$$

**而且现在它和 bin 宽度无关了**：bin 加倍 → $W_k$ 减半 + $\Delta v$ 减半 → 比值不变。这正是我们想要的「形状」。

> 直观理解：除以 $W_{\rm tot}$ 把绝对值变成「占比」；除以 $\Delta v$ 把「这个 bin 的占比」变成「每单位 $v$ 的占比密度」。两步都不能少。

---

## 5. 「密度加权 PDF」的物理意义

这里我们用的权重是 $\rho$，所以得到的是 **density-weighted PDF**：

$$
f_\rho(v)\, dv \;=\; \frac{\sum_{i: v_i \in [v, v+dv]} \rho_i}{\sum_i \rho_i}.
$$

它告诉你「**在速度 $v$ 附近的那一小段区间里，集中了多少比例的质量**」。

对比一下：

| 权重 $w_i$ | 得到的 PDF 含义 |
|---|---|
| $w_i = 1$（不加权） | cell 数密度 PDF：「速度 $v$ 附近有多少**格子**」 |
| $w_i = \rho_i$ | 质量/密度 PDF：「速度 $v$ 附近有多少**质量**」 |

在天体物理里我们关心的几乎总是后者 —— 一片冷致密 cloud 可能只占很少几个 cell 但承载了大部分质量；不加权的 PDF 会把它压扁。

---

## 6. matplotlib 怎么帮我们做这件事

`plt.hist` 有一个参数 `density`：

```python
ax.hist(v_rel, bins=bins, weights=rho[m], density=True, ...)
```

当 `density=True` 时，matplotlib 自动把每个柱子做这样的换算：

$$
\text{柱高} \;=\; \frac{W_k}{W_{\rm tot}\,\Delta v},
$$

也就是把上面手算的两步归一化一次性做完。`weights=...` 决定 $W_k$ 怎么算；`density=True` 决定要不要除以 $W_{\rm tot}\,\Delta v$。

**所以代码改动只是加一个参数 `density=True`，外加把纵轴标签改成 `PDF [(km/s)⁻¹]`**。

---

## 7. Sanity check：新图的纵轴量级应该多大？

如果某个 phase 的速度分布近似一个 σ 大约几十 km/s 的 Gaussian，PDF 的峰值是

$$
f_{\max} \;=\; \frac{1}{\sigma\sqrt{2\pi}} \;\approx\; \frac{0.4}{\sigma}.
$$

例如 σ ≈ 40 km/s 的 hot 相，峰值 ≈ 0.01 (km/s)⁻¹。σ ≈ 5 km/s 的 cool 相，峰值 ≈ 0.08 (km/s)⁻¹。新图的 y 轴应该在这个量级，而不像现在 sum-of-ρ 那样的 $10^{-22}$ g cm⁻³。

另一个检查：把 bin 数从 120 改成 240，柱子高度**应该几乎不变**（除了变得更细更抖）。这是 PDF 的一个判别特征。

---

## 8. 注意事项

**(a) 每个 panel 各自归一化**
我们的 `_plot_hist` 是 3 行（phase）× 2 列（LOS）共 6 个 panel，每个 panel 单独 `ax.hist(...)`。开了 `density=True` 之后，**每个 panel 自己的积分 = 1**，而不是六个 panel 加起来 = 1。

后果：你**不能**通过比较两个 panel 的峰高来判断哪个 phase 占的质量多 —— 那是「形状对比」，不是「质量份额对比」。质量份额信息在每个 panel 标题里的 `m_frac=...` 里，bar plot 里也有。

**(b) Percentile 截断**
代码里 `bins = np.linspace(np.percentile(v_rel, [0.5, 99.5]), 120)`，意思是只在 99% 主体范围里画 bin。这就丢掉了两端 ≈ 1% 的质量，所以显示出来的积分严格说是 ≈ 0.99 而不是 1。这是常规做法（不让长尾把图压扁），不影响形状结论。

**(c) σ 值不变**
panel 标题里的 σ 来自 `mass_weighted_sigma_by_phase`，是用 ρ 直接做加权 moment 算的，跟画图的 binning 完全无关。改 `density=True` 不会动它。

---

## 总结

- 当前 `Σ ρ` 形式的纵轴 = 「这个 bin 里所有 cell 密度之和」，单位 g cm⁻³，**和 bin 数有关**。
- 想要 PDF，需要除以 **总权重 $W_{\rm tot}$ × bin 宽度 $\Delta v$**。
- matplotlib 用 `density=True` 一行搞定。
- 新纵轴单位 $(\mathrm{km/s})^{-1}$，**和 bin 数无关**，整条曲线下面积 = 1。
- 物理含义：「速度 $v$ 附近承载了多少比例的质量」。
