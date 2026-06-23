# HI 21 cm 体积发射率公式 — 从头推导

> 目标读者：物理本科生（学过 QM、EM、热统）
> 目的：让你看完之后能自己写出 `ε_21 = (3/4) · n_HI · A_10 · h · ν_21` 这个公式
> 阅读时间：30 分钟

---

## §0 我们到底在算什么？

天体物理里的 ISM 里**一立方厘米气体每秒发多少能量到 21 cm 这条谱线**？

这个量叫"**体积发射率**"（volumetric emissivity），符号 ε，单位是 erg/(s·cm³)。它是 pipeline 里每个 cell 拿到的最基本物理量，后面所有图（投影图、谱线图）都是它沿 LOS 累加 / FFT / binning 的衍生品。

最终答案：

$$\boxed{\varepsilon_{21} = \frac{3}{4} \, n_{\rm HI} \, A_{10} \, h \, \nu_{21}}$$

每个符号都有物理来历。这份笔记就是讲这条公式怎么"长出来的"。

---

## §1 第一性原理：光从哪来？

任何谱线辐射的最底层物理是：

> **某个原子/分子处在激发态，它"自发"跳到低能级，发出一个光子。**

数学上这个过程的速率由 **Einstein A 系数** 描述：

$$\frac{dN_{\rm upper}}{dt}\bigg|_{\rm spontaneous} = -A_{ul} \cdot N_{\rm upper}$$

含义：每个上能级原子，**每秒以概率 A_ul 跳下来发一个光子**。

如果有 $N_{\rm upper}$ 个原子在上能级，**单位时间发出的光子数**就是：

$$\dot{N}_\gamma = N_{\rm upper} \cdot A_{ul}$$

每个光子能量 $E_\gamma = h\nu_{ul}$，所以**单位时间发出的能量**是：

$$L = N_{\rm upper} \cdot A_{ul} \cdot h\nu_{ul}$$

如果换成单位体积，把 $N_{\rm upper}$ 换成数密度 $n_{\rm upper}$：

$$\varepsilon = n_{\rm upper} \cdot A_{ul} \cdot h\nu_{ul} \quad [{\rm erg/(s \cdot cm^3)}]$$

**这就是任何谱线发射率公式的骨架**。要算 21 cm，我们只需要确定三件事：

1. **$\nu_{ul}$**：21 cm 是哪两个能级之间的跃迁？频率多少？
2. **$A_{ul}$**：这个跃迁的 Einstein A 是多少？
3. **$n_{\rm upper}$**：上能级粒子数密度是多少？

---

## §2 21 cm 是哪个跃迁？氢原子的 hyperfine 结构

### §2.1 氢原子能级图速记

普通量子力学里你学的 H 原子能级是 **n = 1, 2, 3, ...**（主量子数）：

```
   n=∞ -------- ionization (13.6 eV above ground)
   ...
   n=4 -------- 12.75 eV
   n=3 -------- 12.09 eV     ← Hα 是 n=3→2  (656 nm, 4.6×10¹⁴ Hz)
   n=2 -------- 10.20 eV     ← Lyα 是 n=2→1  (122 nm)
   n=1 -------- ground       
```

但是！如果你**放大** ground state n=1 看，它其实**不是一个能级，是两个**：

```
n=1 ground state 放大:
                    ┌── F=1   ←─┐
   n=1   ──────────|            │  ΔE = 5.87 μeV
                    └── F=0   ←─┘  hν = 5.87 μeV
                                   ν = 1.4204 GHz
                                   λ = 21.106 cm  ←── "21 cm 线"!
```

这种细微分裂叫 **超精细结构 (hyperfine structure)**。

### §2.2 为什么有这个分裂？

H 原子有 **两个粒子**：

- 电子，自旋 $s_e = \frac{1}{2}$，磁矩 $\mu_e$
- 质子，自旋 $s_p = \frac{1}{2}$，磁矩 $\mu_p$

它们各自的磁矩**会相互作用**（磁偶极-磁偶极耦合）。两个 1/2 自旋可以耦合成两种总自旋 F：

$$F = \vec{s_e} + \vec{s_p}$$

- $F = 0$：电子和质子自旋**反平行**（净磁矩 = 0）
- $F = 1$：电子和质子自旋**平行**（净磁矩 ≠ 0）

平行的总磁矩自相耦合能量稍高（同性磁极互斥），所以 **F=1 比 F=0 高一点点**。

能量差 $\Delta E$ 来自具体计算（hyperfine Hamiltonian），结果：

$$\Delta E = h \nu_{21} = 5.87 \times 10^{-6} \text{ eV} \approx \frac{1}{10000} \text{ Hα 光子能量}$$

转成频率和波长：
- $\nu_{21} = \Delta E / h = 1.4204 \times 10^9$ Hz = **1.42 GHz**
- $\lambda_{21} = c/\nu = 21.106$ cm

⚠️ 注意尺度：21 cm 跃迁的能量是 H 原子 ionization 能（13.6 eV）的 **百万分之一**。它跟 Hα/Lyα 这些 eV 量级的电子跃迁是**两个不同尺度的物理**。

### §2.3 简单图示

把这些放在一起：

```
Energy
 ↑
 │   n=2  ─────────────  (10.20 eV above ground state)
 │
 │
 │
 │       ┌─── F=1  ──┐
 │   n=1 │            │ ← 21 cm photon emitted on F=1→0 transition
 │       └─── F=0  ──┘
 │
 0   ground state (F=0)
```

**21 cm 跃迁就是 F=1 → F=0**，发出一个 1.42 GHz 的射电光子。

---

## §3 上能级 (F=1) 有多少 H 原子？

ε 公式里 $n_{\rm upper} = n_{F=1}$。我们需要知道这个数。

### §3.1 总数：$n_{\rm HI}$

宇宙中性 H 原子总数密度，常用符号 $n_{\rm HI}$。它包括 F=0 和 F=1 两个超精细态：

$$n_{\rm HI} = n_{F=0} + n_{F=1}$$

具体在 ISM 里 $n_{\rm HI}$ 跟温度、密度、电离率有关，DESPOTIC chemistry 解出来给我们。Pipeline 里它就是 `('gas', 'H')` 这个字段。

### §3.2 怎么把 $n_{\rm HI}$ 分到 F=0 和 F=1？

热平衡下用 **Boltzmann 分布**。每个能级被占据的概率正比于：

$$N_i \propto g_i \exp\left(-\frac{E_i}{k_B T}\right)$$

其中 $g_i$ 是**统计权重 (statistical weight)**，等于该能级的简并度。

### §3.3 F=0 和 F=1 的简并度是多少？

总自旋量子数 F 对应的状态有 $g_F = 2F + 1$ 个：

| 能级 | F | $m_F$ 取值 | 简并度 $g$ |
|---|---|---|---|
| 下 | 0 | 0 | $g_0 = 1$ |
| 上 | 1 | -1, 0, +1 | $g_1 = 3$ |

总共 4 个 hyperfine 子能级（1+3）。

⚠️ 注意：如果你不分超精细结构，整个 H 原子 ground state 的简并度是 4（因为 $g = 2(2L+1)(2I+1) = 2 \cdot 1 \cdot 2 = 4$），这 4 个对应这里的 1+3 hyperfine 子能级。

### §3.4 应用 Boltzmann

$$\frac{n_{F=1}}{n_{F=0}} = \frac{g_1}{g_0} \exp\left(-\frac{\Delta E}{k_B T}\right) = 3 \exp\left(-\frac{h\nu_{21}}{k_B T}\right)$$

定义"hν/k 对应的温度" $T_* \equiv h\nu_{21} / k_B$：

$$T_* = \frac{6.626 \times 10^{-27} \text{ erg·s} \cdot 1.4204 \times 10^9 \text{ s}^{-1}}{1.381 \times 10^{-16} \text{ erg/K}} = 0.0681 \text{ K}$$

所以：

$$\frac{n_{F=1}}{n_{F=0}} = 3 \exp\left(-\frac{0.0681}{T}\right)$$

### §3.5 高温极限：3/4

ISM 任何温度都满足 $T \gg T_* = 0.07$ K。**最低的 ISM 温度是 CNM ~ 50 K**，所以 $T/T_* \geq 700$，指数因子 $\exp(-T_*/T)$ 非常接近 1。

具体看数：

| T (K) | $T_*/T$ | $\exp(-T_*/T)$ | $n_1/n_0$ | $n_1 / n_{\rm HI}$ |
|---|---|---|---|---|
| 1 | 0.068 | 0.934 | 2.80 | 0.737 |
| 10 | 0.0068 | 0.993 | 2.98 | 0.748 |
| 100 | 0.00068 | 0.9993 | 3.00 | **0.750** |
| 10000 | tiny | 1.000 | 3.00 | 0.750 |

⚠️ 重要观察：**T > 10 K 时，$n_{F=1}/n_{\rm HI}$ 已经精确等于 3/4**（误差 < 10⁻³）。

所以在所有 ISM 条件下：

$$\boxed{n_{F=1} = \frac{3}{4} n_{\rm HI}}$$

这就是公式里 3/4 的物理来源 —— **统计权重比** 在高温极限。

### §3.6 物理直觉

为什么是 3:1 而不是 1:1？

- F=0 是**单态**：只有 1 种 $m_F$ 取值
- F=1 是**三重态**：有 3 种 $m_F$ 取值（-1, 0, +1）

热平衡下每个独立子状态被占据的概率相同（都是 $\exp(-E/kT)/Z$）。F=1 的"格子"是 F=0 的 3 倍，所以 F=1 总占据数是 F=0 的 3 倍 → 上能级占总数的 3/4。

---

## §4 Einstein A：跃迁概率

### §4.1 物理是什么

$A_{10}$ = 一个 F=1 上能级原子，**每秒**自发跃迁到 F=0 发出一个光子的概率。

定量值（QED 计算 + 实验）：

$$A_{10} = 2.876 \times 10^{-15} \text{ s}^{-1}$$

### §4.2 这个数有多小

$A_{10}$ 的倒数是平均寿命：

$$\tau = \frac{1}{A_{10}} = \frac{1}{2.876 \times 10^{-15}} \approx 3.5 \times 10^{14} \text{ s} \approx \boxed{1100 \text{ 万年}}$$

⚠️ 一个 F=1 的 H 原子平均要等 **一千万年** 才会自发发出 21 cm 光子。这是为什么 21 cm 是 "**forbidden line**"（禁戒跃迁）。

### §4.3 为什么这么慢？

这是 **磁偶极跃迁 (magnetic dipole transition)**。Einstein A 的量级估计：

- 普通"电偶极允许"跃迁（如 Hα）：$A \sim 10^7$ s⁻¹（寿命 ~ 100 ns）
- 磁偶极跃迁：$A \sim \alpha^2 (\nu / c)^3 a_0^2$（α = fine-structure constant ~ 1/137）
- 21 cm 还多了一个 $(\nu/\nu_{\rm electronic})^3$ 抑制因子，因为 21cm 频率比电子跃迁低 ~10⁶ 倍

所以 $A_{21\rm cm} \sim A_{\rm electric} \times \alpha^2 \times 10^{-18} \sim 10^{-15}$ ✓

### §4.4 为什么还能观测到？

这么慢的发射也能被观测到，因为：

- ISM 里中性 H 数密度大（盘面 ~1 cm⁻³）
- LOS 上累积的 H 原子数 enormous（柱密度 $N_{\rm HI} \sim 10^{20}$ cm⁻²）
- 单 cell 发射弱，但**沿 LOS 几 kpc 加起来，每秒能逃出大量光子**

---

## §5 完整公式拼装

### §5.1 把三块拼起来

带回 §1 的骨架公式 $\varepsilon = n_{\rm upper} \cdot A \cdot h\nu$：

$$\varepsilon_{21} = n_{F=1} \cdot A_{10} \cdot h \nu_{21}$$

代入 §3.5 的 $n_{F=1} = \frac{3}{4} n_{\rm HI}$：

$$\boxed{\varepsilon_{21} = \frac{3}{4} \, n_{\rm HI} \, A_{10} \, h \, \nu_{21}}$$

✅ **公式推完了**。

### §5.2 数值代入

```
A_{10} = 2.876e-15 s⁻¹
ν_{21} = 1.4204e9 Hz
h      = 6.626e-27 erg·s
```

把常数乘起来：

$$\frac{3}{4} \cdot A_{10} \cdot h \cdot \nu_{21} = 0.75 \cdot 2.876 \times 10^{-15} \cdot 6.626 \times 10^{-27} \cdot 1.4204 \times 10^9$$
$$= 2.04 \times 10^{-32} \quad [\text{erg/s per cm}^3 \text{ per (cm}^{-3} \text{ HI)}]$$

所以：

$$\varepsilon_{21} = n_{\rm HI} \times 2.04 \times 10^{-32} \quad [{\rm erg/s/cm^3}]$$

只要给我 $n_{\rm HI}$（cm⁻³），就能直接乘出体积发射率。

### §5.3 数值校验

CNM 典型条件 $n_{\rm HI} \approx 30$ cm⁻³：

$$\varepsilon_{21} = 30 \times 2.04 \times 10^{-32} = 6.1 \times 10^{-31} \text{ erg/s/cm}^3$$

跟 pipeline 实测对比（前面 grid 输出）：
```
HI_luminosity median = 3.6e-34, max = 1.2e-30
```
最大值 1.2e-30 对应 cell 里 $n_{\rm HI} \approx 60$ cm⁻³（盘内最致密区），跟我们的估算一致 ✓

---

## §6 隐含假设：什么时候这个公式有效？

我们的公式有 **两个关键假设**，要明确：

### §6.1 假设 1：高温极限（3/4 因子）

要求 $T \gg T_* = 0.07$ K。

ISM 温度都远超过 0.07 K（最低也 50 K），**这个假设永远成立**，误差 < 10⁻³。

### §6.2 假设 2：光学薄

我们假设**所有发出的光子都直接逃出去，不被附近的 H 原子重新吸收**。

什么情况下这会失败？光学厚度 $\tau$ 大的时候。21 cm 光学厚度公式：

$$\tau_{21} = \frac{3 c^3}{32 \pi \nu_{21}^3} \cdot \frac{A_{10}}{T_{\rm spin}} \cdot \frac{N_{\rm HI}}{\Delta v}$$

代入数字：

$$\tau_{21} \approx 5.5 \times 10^{-19} \cdot \frac{N_{\rm HI} \, [\rm cm^{-2}]}{T_{\rm spin} \, [\rm K] \cdot \Delta v \, [\rm km/s]}$$

典型 ISM cell：
- $N_{\rm HI} \sim 10^{20}$ cm⁻²，$T_{\rm spin} \sim 100$ K，$\Delta v \sim 5$ km/s
- $\tau_{21} \sim 5.5 \times 10^{-19} \cdot 10^{20} / 500 \approx 10^{-4}$
- ✓ **完全光学薄**

光学厚的情况：
- $N_{\rm HI} > 10^{21}$ + $T_{\rm spin} < 50$ K + $\Delta v < 1$ km/s（很冷很密的 CNM filament）
- $\tau_{21} \sim 1$
- 这种 cell 在 plt263168 里**几乎没有**（盘面也基本不到 $10^{21}$）

所以你 snapshot 里 99.9%+ cell 满足光学薄假设 → 公式 effectively exact。

---

## §7 跟 H-alpha 的对比

为什么 H-alpha 不用同样的方法？

| 比较项 | HI 21 cm | H-alpha |
|---|---|---|
| 跃迁 | F=1 → F=0 (hyperfine) | n=3 → n=2 (electronic) |
| Einstein A | $2.88 \times 10^{-15}$ s⁻¹ | $4.41 \times 10^{7}$ s⁻¹ |
| 上能级 fraction | 3/4（热平衡 + 高温极限） | **不能**用 Boltzmann |
| 主要激发机制 | 碰撞 (collisional in CNM) | **复合 cascade** (e⁻ + H⁺) |
| 在 ISM 主区 | 中性气主导 (CNM, WNM) | 电离气主导 (HII, WIM) |

**关键区别**：H-alpha 的 n=3 上能级在 HII region 里**不是热平衡布居的**。它由复合事件产生：

```
HII region 里:
  自由 e⁻ + H⁺ → 复合 → H 原子在某个 n* 高激发态
                       → cascade 下来
                       → 经过 n=3 → n=2 时发出 Hα

n=3 的"产生率" 由 e⁻·H⁺ 复合决定 (二体过程)
而不是由 Boltzmann (n=3 跟 n=1 的热平衡比例)
```

如果错误用 Boltzmann 算 H-alpha：

$$n_{n=3} \approx g_3 \exp(-E_3/kT) \cdot n_{\rm HI}$$

在 ionized gas 里 $n_{\rm HI}$ 几乎为 0（H 都电离了），就算 Boltzmann 给的"上能级 fraction"再大，乘 0 还是 0。这样会**严重低估** H-alpha。

正确方法是 Case-B 复合公式：$\varepsilon_{H\alpha} \propto \alpha_B(T) \cdot n_e \cdot n_{\rm HII}$。**完全不同的物理框架**。

⚠️ HI 21 cm 是简单的，因为 ground state F=0/F=1 的占据比例**几乎只受 T 控制**（碰撞快速 thermalize，Boltzmann 适用）。Hα 复杂因为 n=3 的占据由复合速率决定，不是 Boltzmann。

---

## §8 总结：如果让你从零算 HI 21 cm 你怎么做

```
Step 1 ─── 找跃迁
            21 cm = H atom F=1 → F=0 hyperfine
            ν = 1.4204 GHz, λ = 21.1 cm

Step 2 ─── 找 Einstein A
            QED 算 / 表查
            A_10 = 2.876e-15 s⁻¹

Step 3 ─── 算上能级粒子数
            统计权重: g(F=1) = 3, g(F=0) = 1
            Boltzmann: n_1/n_0 = 3 · exp(-T_*/T)
            T_* = hν/k = 0.07 K
            ISM T >> T_* → exp ≈ 1
            → n_1/n_total = 3/4
            → n_upper = (3/4) · n_HI

Step 4 ─── 拼公式
            ε = n_upper · A · hν
              = (3/4) · n_HI · A_10 · h · ν_21
              [erg/s/cm³]

Step 5 ─── 校验假设
            T > 10 K  ✓ (ISM 永远满足)
            τ < 1     ✓ (光学薄 except dense cold filament)
```

要算其他 forbidden line（[CII] 158 μm, [OI] 63μm 之类的 fine-structure line），步骤几乎一样：

- 换 ν 和 A
- 换简并度（看 J 量子数）
- 看温度极限是否还成立（fine-structure $T_* \sim 100$ K，CNM 不太满足，需要解 detailed balance）

---

## §9 进一步阅读

- **Draine 2011**，"Physics of the ISM"：Chapter 8 (HI 21 cm) 讨论得极详细
- **Field 1958**：Wouthuysen-Field effect（在 Lyα 强的环境，T_spin 跟 Lyα radiation field 耦合，不再等于 T_kinetic）
- **Furlanetto 2006**："Cosmology at low frequencies"：21cm 在 reionization 时代的应用

---

## 附录：常用数值（CGS）

```
h        = 6.626 × 10⁻²⁷ erg·s
k_B      = 1.381 × 10⁻¹⁶ erg/K
c        = 2.998 × 10¹⁰ cm/s
ν_21     = 1.4204 × 10⁹ Hz
λ_21     = 21.106 cm
hν_21    = 9.41 × 10⁻¹⁸ erg = 5.87 × 10⁻⁶ eV
T_*      = 0.0681 K
A_10     = 2.876 × 10⁻¹⁵ s⁻¹
3hν·A/4  = 2.04 × 10⁻³² erg/s per cm⁻³ HI
```
