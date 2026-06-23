# DESPOTIC Table Diagnostics — Physics Notes

*Created 2026-04-07 from conversation with Claude Code*

---

## 1. 为什么 CO / HCO+ 的 lumPerH 随 N_H 上升？

### 结论
这是正确的物理行为，不是 bug。原因是 **UV 屏蔽使 CO/HCO+ 分子能够存在**（chemistry turn-on），而不是通常误以为的"自屏蔽降低发射"。

### 两种效应的方向相反

| 效应 | 发生机制 | 对 lumPerH 的方向 |
|---|---|---|
| **化学自屏蔽 (chemistry)** | N_H 小时 UV 穿透 → CO / HCO+ 被光解，丰度 X(CO) ≈ 0。N_H 增大后 UV 被 dust + H₂ + CO self-shielding 吸收，X(CO) 从 ~0 上升到 canonical ~10⁻⁴ | **↑（变亮）** |
| **辐射转移光厚 (radiative trapping)** | N_CO 很大时 J=1-0 光深 τ ≫ 1，escape probability β ∝ 1/τ → 每个分子发出的净光子被邻居吸收 | ↓（变暗）|

### 在 DESPOTIC 表的 N_H 范围内发生了什么

- **N_H ≲ 10²⁰–10²¹ cm⁻²**：CO 还没"点亮"（被 UV 打掉），lumPerH 极低。这是**化学零点**，不是光厚效应。
- **N_H ~ 10²¹–10²² cm⁻²**：UV 屏蔽 kick in，CO 丰度迅速 turn-on，lumPerH 上升约 2–3 个量级（图里的陡升段）。
- **N_H ≳ 10²²–10²³ cm⁻²**：丰度已饱和，J=1-0 进入光厚区，但高阶跃迁（J=2-1, 3-2 等）仍然光薄，总线发射饱和成**高位平台**而非下降。
- DESPOTIC 的 `lumPerH` 已是 escape-probability 修正后 + 多条 J 线加总，所以不会像单一光薄近似那样出现明显 turnover。

### HCO+ 与 CO 的异同
HCO+ 的化学网络在 CO 上游，行为类似但 turn-on N_H 略高；曲线形状与 CO 相近。

### C+ 的不同
C+ 是 UV **电离**产生（不是被 UV 破坏），不受 CO 式的自屏蔽控制，lumPerH vs N_H 几乎平坦，只在 N_H > 10²³ 才略有下降。这与图里 C+ 子图"最平缓"完全吻合。

---

## 2. col_mid 的问题与解决方案

### 问题
DESPOTIC 表的 `col_values` 数组的索引中点（`col_mid ≈ 3×10¹⁹ cm⁻²`）**落在 CO/HCO+ 的化学死区**，此处 CO 丰度 ≈ 0，lumPerH 极低。用这个值固定 N_H 然后 sweep T，等于在"CO 几乎不发光"的区域讨论温度依赖，结论会被严重拉偏。

### 解决方案：luminosity-weighted median N_H
对每个 species，在其 T_peak（自身最亮的温度）下扫 col_vals，用 lumPerH 作为权重，计算**对数空间的加权中位数**：

```python
N_H_ref[sp] = weighted_percentile(col_vals, lumPerH_vs_col, pct=50)
```

这个值代表"表内 N_H 轴上，亮度集中的位置"，物理意义：**在 T_peak 下，哪段 N_H 对 lumPerH 贡献最多**。

典型结果预期：
- **CO**：N_H_ref ≈ 几 × 10²¹ – 10²² cm⁻²（典型分子云柱密度）
- **HCO+**：类似或略高
- **C+**：与 col_mid 差距不大（曲线平坦，加权不敏感）

### 与 SigmaNTCheckTask 的对比
`SigmaNTCheckTask` 也用 `weighted_percentile`，但权重是**真实 sim cell 的光度**，数据是**每 cell 的 Δv_x**——回答"真实光子从什么速度梯度区来"。TableDiagnostics 的权重是**表格点的 lumPerH**，数据是**表的 N_H 网格**——回答"表内哪段 N_H 最亮"。两者统计函数相同，语义不同：前者是"真实分布 × 亮度"加权，后者是"纯表内部"加权。

---

## 3. GOW 网络的有效温度范围（重要！）

### 关键事实：GOW 只追踪单次电离态
GOW 化学网络（Gong, Ostriker & Wolfire 2017）的 state vector（`GOW.py:45-47`）：
```python
specList = ['H2','H+','H2+','H3+','He+','O+','C+','CO','HCO+','Si+','CHx','OHx']
```
**碳的守恒只闭合在 {C+, C, CO, CHx, HCO+} 中，没有 C²⁺/C³⁺/C⁴⁺ 反应通道，也没有 He²⁺、O²⁺。**

### 后果：T > ~5×10⁴ K 时网络产生虚假结果
在碰撞电离平衡（CIE）下，T > 3×10⁴ K 时碳开始向 C²⁺ 过渡，T ~ 10⁵ K 时 C+ 丰度已显著下降。但 GOW 没有相应反应通道，**强制把所有碳保持为 C+/C/CO**，导致：
- lumPerH vs T 图在 10⁵–10⁷ K 出现**虚假平台**（图里 C+ 子图的平坦段）
- 这不是物理信号，是 GOW 把 C 硬锁在 C+ 后，C II 158μm 的 Boltzmann 饱和产生的 artefact

### 速率拟合的额外限制
- Janev H 碰撞电离多项式（`GOW.py:449-458`）：拟合范围 ~10²–10⁵ K，超出为外推
- H₂ 碰撞解离（`GOW.py:429-446`）：拟合到 T ≲ 几×10⁴ K
- Arrhenius 速率（`GOW.py:373, 379, 419, 442, 476`）：标定于 PDR 温度

### `setChemEq` 不检查 T 范围
`setChemEq.py:73-89`：fixed 模式原样信任输入的 `cloud.Tg`，不 clamp、不警告。给它 T=10⁶ K 会静默返回收敛但物理完全错误的结果。

### 结论：T > 5×10⁴ K 的表数据都是外推 artefact，后处理必须截断
**GOW 网络物理上限 ≈ 5×10⁴ K**。表在该温度以上的所有 lumPerH 值不可信，必须在后处理中置零。

---

## 4. T_CUTOFF 历史与当前选择

| Species | 旧值 (SpeciesCutoff run) | 当前值 (GOWnetworkLimit run) | 说明 |
|---|---|---|---|
| **CO** | 100 kK | **50 kK** | CO 在 T~几千 K 就碰撞解离，100k 只是保险；改为网络上限 |
| **C+** | 200 kK | **50 kK** | 原来偏宽松（误判为"保守"）；实际是 GOW 外推区，必须截断 |
| **HCO+** | 100 kK | **50 kK** | 同 CO |

T_CUTOFF 现为 `config.py` 中的单一真源，`physics_fields.py` 和 `table_diagnostics.py` 均从此处 import，避免两处不一致。

对比目录：
- `plots_3DVersion_SpeciesCutoff_GOW/` — 旧 per-species cutoff，基线对比
- `plots_3DVersion_GOWnetworkLimit_5e4_GOW/` — 当前，统一 5×10⁴ K

---

## 5. 阶梯 (staircase) 形态

图 1 各子图里曲线呈阶梯而非光滑——这是**表格 T 方向格点分辨率有限**的痕迹（nearest / 低阶插值），不是 bug。说明 spectrum 计算里 T 方向插值的数值噪声主要来源于表的格点密度，而非代码问题。

---

## 6. 代码位置

| 文件 | 内容 |
|---|---|
| `pipeline/prep/config.py` | `T_CUTOFF` dict + `T_CUTOFF_DEFAULT` — 单一真源 |
| `pipeline/prep/physics_fields.py` | `_make_luminosity_field` — 实际对 lumPerH 置零 |
| `pipeline/utils.py` | `weighted_percentile(data, weights, pct)` — 共享工具函数 |
| `pipeline/tasks/table_diagnostics.py` | `TableDiagnosticsTask` — 产生两张诊断图 |
| `pipeline/tasks/sigmaNT_check.py` | `SigmaNTCheckTask` — 复用同一 `weighted_percentile` |
| `pipeline/prep/physics_fields.py` | `_temperature` — log-T 二分法反解 Eint→T |
| `check_eint_monotonicity.py` | 单调性诊断脚本（项目根，不属于 pipeline）|

---

## 7. Eint → T 二分法审计（`_temperature`）

### 7.1 公式自洽性

DESPOTIC 的 `computeEint(T)` (`composition.py:237-300`) 返回**无量纲 ε**，定义为：

$$\varepsilon = E_\text{int per H} / (k_B T)$$

包含平动 + H₂ 振动 + H₂ 转动（完整配分函数）。`lookup.Eint(nH, NH, T)` 直接插值此表。

二分方程（`physics_fields.py:_temperature`）：
$$T \cdot \varepsilon(T) = \frac{E_\text{int,quokka}}{n_H k_B}$$

两边单位均为 K，与 DESPOTIC 定义**完全一致**。**不使用** `(γ-1)μm_H/ρ` 理想气体近似 —— DESPOTIC 里没有 γ、μ 这两个量；`Eint` 按真实配分函数计算，包含 H₂ vib/rot 非线性贡献。

### 7.2 n_H 约定

`_number_density_H = ρ·X_H/m_H`（`X_H=0.74` 质量分数）。

这给出**总 H 核数密度** `n_HI + n_H+ + 2·n_H2`：H₂ 分子质量 ≈ 2·m_H，正好贡献 2 个 H 核，与 DESPOTIC "per H nucleus" 归一化约定一致。**无系数错误**。

### 7.3 单调性审计（`check_eint_monotonicity.py`）

| 表 | 非单调点 | 占比 | 主要位置 |
|---|---|---|---|
| T2e7 (T_max=2×10⁷ K) | 143 / 41414 | **0.35%** | GOW 失效区 T~2×10⁴ K + 极低 T (1–3 K) |
| T2e6 (T_max=2×10⁶ K) | 184 / 41457 | **0.44%** | GOW 失效区 T~6×10⁴–10⁵ K |

两类违反均**不影响谱线**：
- **GOW 失效区违反**：H 电离转变处 ε 非连续跳变（artefact）。对应 cell 的 lumPerH 已被 `T_CUTOFF=5×10⁴ K` 置零，不进入谱线计算。
- **极低 T 违反** (T ≤ 3 K)：H₂ 转动配分函数 `T²/Z·dZ/dT` 在 `T << θ_rot=85.3 K` 时数值不稳。quokka 模拟 cell 几乎不会命中这段。

结论：**不修表，不修二分法的 clamp 逻辑**；两类违反区域物理上已由 `T_CUTOFF` 和 sim 本身的温度范围隔离。

### 7.4 log-T 二分（2025-04 修复）

**原版**（25 次线性二分，区间 `[T_min, T_max]`）：在 T=10 K 时绝对精度 ~0.6 K → 相对误差 ~6%，超过 `tolerance=5%`，造成**假阳性 bad point 报告**。

**修复后**（40 次 log-T 二分）：`log_T_low/high = log10([T_min, T_max])`，每步取几何中点。整个温度范围的**相对精度均匀 ≈ 机器精度**（`10^(log_range / 2^40)`）。开销：多 15 次向量化表插值，可忽略。

### 7.5 GOW 失效区诊断计数器

`_temperature` 在 `return` 前新增：
```
[INFO] N/M cells have T > 5e4 K (GOW network invalid region; lumPerH already zeroed ...)
```
每次 pipeline 运行均可看到有多少 cell 的解落入 GOW 失效区，作为 sanity check。

