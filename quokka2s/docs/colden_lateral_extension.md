# Column-density 横向扩展 (Lateral colDen extension)

> 适用文件: `pipeline/prep/physics_fields.py::_column_density_H`
> 控制开关: `pipeline/prep/config.py::COLUMN_EXTENSION_LATERAL_KPC`
> Cache key 钩子: `pipeline/cache.py::compute_cache_key`

这份文档解释了我们怎么、为什么、以及在代码里如何在每个 cell 的柱密度上额外加一段 "横向 (x/y) 盒外的等效柱". 它影响的是 DESPOTIC chemistry 看到的屏蔽量, 也就是平衡温度 `temperature_despotic` 跟所有依赖它的派生场 (`CO_luminosity`, `C+_luminosity`, 等等).

---

## 1. 问题背景: shearing-box 的边界假象

我们后处理的 QUOKKA 数据是一个 **stratified shearing box**:

```
                    ┌─────────────────┐
                    │                 │  <- domain_top  (z = +Lz/2,  open BC)
                    │     1 kpc       │
                    │    × 1 kpc      │
            ─ ─ ─ ─ │                 │ ─ ─ ─ ─        <- midplane (z = 0)
                    │                 │
                    │      ~8 kpc 高  │
                    │                 │
                    └─────────────────┘  <- domain_bot  (z = -Lz/2,  open BC)

     ←──── 周期 ────→
            (shearing)
```

domain 大小 = **1 kpc × 1 kpc × 几 kpc 高**, 坐落在银河系中心约 **R ≈ 8 kpc** 的位置.

它的边界条件:
- **±x 方向**: shearing periodic — 物理上是"无穷大圆盘的一小块, 旁边还有更多盘面气体". 不是物理边界, 而是数值上的周期回绕.
- **±y 方向**: 同 ±x, periodic — "更多盘面气体" 在外面.
- **±z 方向**: 真正的物理 open BC — 上面/下面就是 halo, 真的没有更多 disk gas 了 (盒子高度 ~ 8 kpc, scale-height ~ 数百 pc, 远大于盘厚, 已经把 disk 全装下了).

### colDen 是什么 + 为什么要算它

`column_density_H` 表示每个 cell 沿某个方向到 box 边界为止累加的 H 柱密度 `N_H = ∫ n_H dl` [cm⁻²]. 

DESPOTIC 用 `N_H` (以及局部 `n_H`, `dV/dr`) 在 lookup table 里查出该 cell 的:
- 平衡温度 `tg_final`
- 化学物种丰度 (CO, C+, e-, H+, ...)
- 谱线发射率 (`lumPerH`)

`N_H` 之所以重要, 是因为它决定了 **A_V** (dust 消光) 以及 self-shielding: dust + H₂ + CO 越多, 紫外光被屏蔽得越厉害, gas 越冷, CO 越能存活, 化学进入"分子云核" regime.

### 问题: 原来的 colDen 只对 box 之内积分

我们之前的实现 (`L_ext = 0` 时), 对每个 cell 从它自己往 6 个方向 (±x, ±y, ±z) 累加到 box 边界为止, 然后取 **谐波平均** (因为 6 个方向的 `N` 不一样, 化学只见 "光最容易钻进来" 的方向, 谐波平均近似这个 "最薄边" 加权):

```
N_eff = 6 / Σ_k (1 / N_k),   k ∈ {+x, -x, +y, -y, +z, -z}
```

对 ±z 这没问题 — z 方向真的就是 disk 边缘, 外面是 halo.

对 **±x, ±y 就有问题**了 — 物理上 box 边界不是 disk 边界! 旁边还有大约 **±10 kpc** 的盘面 gas 一直延伸下去 (banco 系实际 disk 半径 ~10-15 kpc, 盒子只取了中间 1 kpc 一片). 但我们的 6 面 colDen 只算到 box 边界 (~0.5 kpc) 就停了, **低估了 ±x/±y 方向的真实柱密度**.

这导致 DESPOTIC 看到的 `N_H` 偏低 → A_V 偏低 → 屏蔽偏弱 → 平衡 T 偏高, CO 偏少. 对盘内气体, 这个偏差会很大 (盘面 ~10 kpc 气体的柱密度跟 0.5 kpc 比, 通常多 1-2 dex).

---

## 2. 数学公式

我们的 fix: 把 box 外那段盘面 gas 的贡献按 **简化模型** 加回到 ±x/±y 的 `N_k` 上.

简化假设是: box 之外 ±x/±y 方向的 gas 跟 box 内 **同一高度 z** 的 (x,y)-平均密度 `⟨n_H⟩(z)` 是接近的. 在 box ~1 kpc 这种"shearing-box 尺寸 ≪ disk radial scale"的极限下, 这个假设是合理的 — 旁边 ±10 kpc 内的 disk gas 都坐在差不多的 z 处, 平均密度差异很小.

设 `L_ext` = 我们想模拟的横向延伸长度 (默认 9 kpc, 即假装 box 旁边还有 9 kpc 的盘面气体). 那么每个 cell, 对每个横向方向 `k ∈ {+x, -x, +y, -y}`, 我们把它的柱密度修正为:

$$
N_k^{\text{corrected}}(x, y, z) = N_k^{\text{box}}(x, y, z) + L_{\text{ext}} \times \overline{n_H}(z)
$$

其中:
- `N_k^box (x,y,z)` = 原本对 box 内积分得到的柱密度 (沿方向 k)
- `n̄_H(z)` = 对所有 (i, j) 网格做 (x,y)-平均: `n̄_H(z) = ⟨n_H(x_i, y_j, z)⟩_{i,j}`
- `L_ext × n̄_H(z)` = 假想 box 外那段 (长度 L_ext, 密度 n̄_H(z)) 贡献的柱密度

**Z 方向不加**: 

$$
N_{\pm z}^{\text{corrected}} = N_{\pm z}^{\text{box}}
$$

因为盒子在 z 方向已经覆盖了 ±4 kpc, 远大于 disk 的 scale height (~300 pc), 物理上 ±z box 边界外就是真的没什么气体了.

最后还是 6 方向谐波平均:

$$
N_{\text{eff}}(x, y, z) = \frac{6}{\sum_k \frac{1}{N_k^{\text{corrected}}(x, y, z)}}
$$

### 关键性质

1. **L_ext = 0 → 恢复原版**: 公式回到 6 面纯 box 积分, 不加任何东西. (我们用 L_ext = 0 跟 L_ext = 9 两个版本对照, 看 chemistry / temperature 怎么变.)

2. **只增不减**: `N_k^corrected ≥ N_k^box`, 加 contribution 是非负的. 谐波平均也不会小于原版 (虽然不是单调放大, 因为谐波平均偏向最小的那个 N).

3. **每个 z-layer 用同一个 n̄_H(z)**: 也就是说, **同一 z 高度的所有 cell, 它们 ±x/±y 加的延伸 colDen 是一样的** (因为 n̄ 只是 z 的函数). 这是 "shearing-box 各处都看到同样的盘面环境" 的假设的直接体现.

---

## 3. 代码实现: `_column_density_H`

文件: `quokka2s/src/quokka2s/pipeline/prep/physics_fields.py`, 行 55-113.

```python
def _column_density_H(field, data):
    density_3d = data[('gas', 'density')].in_cgs()

    # 1D fallback (用于 PhasePlot 场景, 此时没有 3D 空间结构) ─ 跳过
    if density_3d.ndim == 1:
        ...

    dx_3d = data[("boxlib", "dx")].in_cgs()
    dy_3d = data[("boxlib", "dy")].in_cgs()
    dz_3d = data[("boxlib", "dz")].in_cgs()

    n_H_3d = (density_3d * cfg.X_H) / m_H

    # ── 横向延伸贡献 (单独算一次, 对 ±x/±y 通用) ────────────────────
    L_ext_kpc = float(cfg.COLUMN_EXTENSION_LATERAL_KPC)
    if L_ext_kpc > 0.0:
        L_ext_qty    = (L_ext_kpc * kpc).in_units('cm')   # unyt 标量, cm
        n_bar_z      = n_H_3d.mean(axis=(0, 1))           # 形状 (nz,), cm^-3
        N_ext_lat_3d = (L_ext_qty * n_bar_z)[None, None, :]
        # 形状: (1, 1, nz), broadcast 时跟 (nx, ny, nz) 对齐
    else:
        N_ext_lat_3d = None

    # ── 6 方向累加 + 谐波平均 (streaming, 节省内存) ────────────────
    inv_sum = None
    for axis, sign, dxyz, lateral in (
        ("x", "+", dx_3d, True),
        ("x", "-", dx_3d, True),
        ("y", "+", dy_3d, True),
        ("y", "-", dy_3d, True),
        ("z", "+", dz_3d, False),
        ("z", "-", dz_3d, False),
    ):
        N = along_sight_cumulation(n_H_3d * dxyz, axis=axis, sign=sign)
        if lateral and N_ext_lat_3d is not None:
            N = N + N_ext_lat_3d           # ← 关键加法在这里
        inc = 1.0 / N
        del N                              # 立刻释放, 内存节约
        if inv_sum is None:
            inv_sum = inc
        else:
            inv_sum = inv_sum + inc
        del inc
    return (6.0 / inv_sum).to('cm**-2')
```

逐段解释:

### 3.1 算 `n̄_H(z)`

```python
n_bar_z = n_H_3d.mean(axis=(0, 1))   # 对 axis 0 (x) 和 axis 1 (y) 同时做 mean
```

`n_H_3d` 的形状是 `(nx, ny, nz)`. `mean(axis=(0, 1))` 把前两个轴塌掉, 留下形状 `(nz,)` 的一维数组 — 每个 z-layer 的 (x,y)-平均 H 数密度.

### 3.2 算横向延伸的 `N_ext`

```python
L_ext_qty = (L_ext_kpc * kpc).in_units('cm')
N_ext_lat_3d = (L_ext_qty * n_bar_z)[None, None, :]
```

`L_ext_qty` 是 unyt scalar (例如 9 kpc → 2.78 × 10²² cm). `n_bar_z` 单位是 cm⁻³. 乘起来 `L_ext_qty * n_bar_z` 单位是 **cm⁻²** ✓ (柱密度).

`[None, None, :]` 把它从 (nz,) 升维到 **(1, 1, nz)**, 这样在第一个 axis (x) 跟第二个 axis (y) 上 broadcast 会沿着这两个方向 (而 z 上还是原样).

为什么这样设计? 因为 cell `(i, j, k)` 应该加的"盒外柱"等于 `L_ext × n̄_H(z_k)`, 跟 `i, j` 都无关 — 加的是 "这个 z 高度上, 盒子之外那段盘面的平均柱". 所以 `N_ext_lat_3d[i, j, k] = L_ext × n_bar_z[k]`, 跟 `i, j` 无关.

### 3.3 单位检查 (重要 — 这里之前出过 bug)

```python
L_ext_qty   = (L_ext_kpc * kpc).in_units('cm')   # unyt, cm
n_bar_z     = n_H_3d.mean(axis=(0, 1))           # unyt, cm^-3
N_ext_lat_3d = (L_ext_qty * n_bar_z)[None, None, :]  # unyt, cm * cm^-3 = cm^-2 ✓
```

第一次实现的时候我把 `L_ext` 当成纯 float (`L_ext_cm = 9 * 3.086e21`), 然后跟 unyt 的 `n_H` 乘起来, **unyt 不知道这个 float 是 cm**, 结果单位算错变成 cm⁻³, 后续加法 `N = N + N_ext_lat_3d` 报单位不一致.

修复就是上面这样: 把 `L_ext` 也保留成 unyt quantity (`L_ext_kpc * kpc`), 显式 `.in_units('cm')`. 这样 unyt 自动追踪单位, 乘法/加法都是 type-safe 的.

### 3.4 6 方向 streaming 累加 + 谐波平均

`along_sight_cumulation(arr, axis, sign)` 在 `analysis.py` 里:

```python
def along_sight_cumulation(data, axis, sign):
    """沿指定 axis 跟 direction 做累加求和."""
    if sign == "+":   # 从 cell 往 +axis 方向到 box 边缘累加
        return np.flip(np.cumsum(np.flip(data, axis=axis), axis=axis), axis=axis)
    if sign == "-":   # 从 cell 往 -axis 方向到 box 边缘累加
        return np.cumsum(data, axis=axis)
```

`+` 方向用 `flip → cumsum → flip` 的小把戏实现 "反向" cumsum (从尾巴往头累加).

6 个方向的循环:

```python
inv_sum = None
for axis, sign, dxyz, lateral in (
    ("x", "+", dx_3d, True),
    ("x", "-", dx_3d, True),
    ("y", "+", dy_3d, True),
    ("y", "-", dy_3d, True),
    ("z", "+", dz_3d, False),
    ("z", "-", dz_3d, False),
):
    N = along_sight_cumulation(n_H_3d * dxyz, axis=axis, sign=sign)
    if lateral and N_ext_lat_3d is not None:
        N = N + N_ext_lat_3d
    inc = 1.0 / N
    del N
    inv_sum = inc if inv_sum is None else inv_sum + inc
    del inc
return (6.0 / inv_sum).to('cm**-2')
```

`lateral` 标记 ±x, ±y 是 True (加延伸), ±z 是 False (不加).

**每次循环只持有一个 cube 在内存里** (`N`, 然后 `inc`), 累加完释放. 这是 streaming 写法, 内存峰值 ~5 GB 而不是同时保留 6 个累加 cube ~22 GB. 对 down=1 数据 (256×256×2048) 很关键, 不然 24 GB Mac 装不下 (历史上, 这条 path 之前是 stack-then-aggregate 版, 直到改成 streaming 才搞定 down=1).

---

## 4. 为什么只加 ±x, ±y, 不加 ±z

物理上:
- **±x, ±y 是 shearing periodic** — 盒子边界不是 disk 边界. 实际 disk 半径 ~10-15 kpc, 盒子只切了中间 1 kpc. 边界外有大量盘面 gas, **必须补**.
- **±z 是 open BC** — 盒子高度 ~8 kpc, 远超过 disk scale height. 边界外是真正的 halo gas (n_H ~ 10⁻⁴ cm⁻³). 即使再加 9 kpc 的 halo, `L_ext × n_halo ~ 9×3e21 cm × 10⁻⁴ cm⁻³ ~ 3×10¹⁸ cm⁻²`, 远小于盘内典型的 ~10²¹ cm⁻², **不补也无所谓**.

代码里这一行就是这个区分:

```python
for axis, sign, dxyz, lateral in (
    ("x", "+", dx_3d, True),    # ← 横向, 加延伸
    ("x", "-", dx_3d, True),
    ("y", "+", dy_3d, True),
    ("y", "-", dy_3d, True),
    ("z", "+", dz_3d, False),   # ← 纵向, 不加
    ("z", "-", dz_3d, False),
):
```

---

## 5. 配置开关跟 cache key 隔离

### 5.1 config.py

```python
# pipeline/prep/config.py
COLUMN_EXTENSION_LATERAL_KPC = 9.0     # 改这一个数字切换 L_ext

_LEXT_TAG = f"_Lext{COLUMN_EXTENSION_LATERAL_KPC:g}kpc"
OUTPUT_DIR = f"{_OUTPUT_ROOT}/{_DATASET_BASENAME}_down{DOWNSAMPLE_FACTOR}{_LEXT_TAG}/"
```

`L_ext = 9.0` 时 `OUTPUT_DIR` 会带 `_Lext9kpc` 后缀, `L_ext = 0.0` 时带 `_Lext0kpc`. 所以两个版本的输出图自动分到不同目录, **不会互相覆盖**:

```
output/plt0655228_down2_Lext0kpc/   ← 不加延伸
output/plt0655228_down2_Lext9kpc/   ← 加 9 kpc 延伸
output/plt0655228_down1_Lext0kpc/   ← (full-res) 不加
output/plt0655228_down1_Lext9kpc/   ← (full-res) 加
```

### 5.2 cache.py — 让 cache 也分开

```python
# pipeline/cache.py
CACHE_SCHEMA_VERSION = 3   # 因为加了 L_ext 维度, 旧 cache 全失效

def compute_cache_key(
    dataset_path, despotic_table_path, downsample_factor,
    column_extension_lateral_kpc: float = 0.0,
) -> str:
    h = hashlib.sha1()
    for component in (
        str(Path(dataset_path).resolve()),
        f'{_file_mtime(dataset_path):.0f}',
        str(Path(despotic_table_path).resolve()),
        f'{_file_mtime(despotic_table_path):.0f}',
        f'downsample={int(downsample_factor)}',
        f'L_ext_kpc={float(column_extension_lateral_kpc):g}',   # ← 折入 hash
        f'schema={CACHE_SCHEMA_VERSION}',
    ):
        h.update(component.encode())
        h.update(b'\x00')
    return h.hexdigest()
```

把 `L_ext` 折进 sha1 hash 的好处: 同一个 field cache 文件 (例如 `field_gas_column_density_H.h5`) 在 L=0 状态写下的内容跟在 L=9 状态写下的内容, cache_key 不同 → load 时会检测 mismatch → 自动 invalidate → 重算. 不会把 L=0 的 cache 错当成 L=9 的来用.

**陷阱**: cache 文件名只按 field 名 (`field_gas_column_density_H.h5`) 命名, 不带 L_ext suffix. 所以 L=0 跟 L=9 的 cache **会互相覆盖** (谁后写, 谁就在). 想保留两个版本的 cache, 需要手动备份 (或者改 filename 加 L_ext suffix, 我们没做, 因为磁盘空间是更稀缺的资源).

### 5.3 PipelineConfig 串联

```python
# pipeline/base.py
@dataclass
class PipelineConfig:
    ...
    column_extension_lateral_kpc: float = 0.0   # 默认 0, 由 run_pipeline 注入

# tasks/run_pipeline.py
pipeline_config = PipelineConfig(
    ...
    column_extension_lateral_kpc = cfg.COLUMN_EXTENSION_LATERAL_KPC,
)
```

然后 cache key 计算的时候从 `pipeline_config.column_extension_lateral_kpc` 取值, 跟 `physics_fields.py::_column_density_H` 从 `cfg.COLUMN_EXTENSION_LATERAL_KPC` 取值是同一个全局值, **保证一致**.

---

## 6. Sanity checks

### 6.1 L_ext=0 必须等价于原版 (没加延伸前的行为)

代码里通过 `if L_ext_kpc > 0.0: ... else: N_ext_lat_3d = None` 短路实现. L_ext=0 时, 横向 N 不加任何东西, 跟原 6 面 harmonic mean 完全一致. ✓

### 6.2 T_QUOKKA 在 L_ext 变化下应该完全不变

`temperature_quokka = boxlib/temperature * K` 跟 colDen 完全无关 (是 QUOKKA sim 直接输出的 T). 所以 L_ext=0 跟 L_ext=9 跑出来的 T_QK 应该 **bit-for-bit identical**.

我们的 `TemperatureLextDiffTask` 跑出来确认: `log10(T_QK(L=9) / T_QK(L=0))` 的最大绝对值 = **0.00e+00 dex**. ✓ 这证明 (a) L_ext 改动确实没污染 T_QK, (b) 数据 pipeline 整体可复现.

### 6.3 T_DSP 应该是 "加延伸后温度低一点"

加 L_ext → N_H 更大 → A_V 更大 → 屏蔽更好 → photoheating 弱 → DESPOTIC 平衡 T 应该更低.

实测 `log10(T_DSP(L=9) / T_DSP(L=0))` 的 p99 ≈ -0.04 dex (差 ~10%), 方向对, 量级小. 量级小是因为致密 gas 早就 A_V > 5 完全屏蔽了 (再加多余的 colDen 没用), 稀薄 gas 即便加了 9 kpc 也还是 A_V << 1 不够屏蔽 (DESPOTIC 平衡 T 还是 photoheating 主导, 没多大变化). 中间 A_V ~ 0.5-3 的过渡区影响最大. ✓

### 6.4 CO 发射应该变化更剧烈

虽然 T_DSP 变化小, CO 化学跟 N_H 是高度非线性的 (`shielding` function 在 A_V ~ 1-2 那段斜率最陡). 实测 `EmitterLextDiffTask` 显示 `log10(L_CO(L=9) / L_CO(L=0))` 在过渡 cell 可以差 **1-3 dex** (CO 自蔽门槛附近, 几倍 N_H 差异就让 CO/总碳比变 10-100×). C+ 反向变化 (CO 多了, C+ 少了). 总碳数守恒. ✓

---

## 7. Limitations / what's NOT modeled

1. **横向 gas 用的是当下盒内 `(x, y)`-平均, 不是真实邻近盘面 gas**. 真实银河系盘在 R ~ 8 kpc 附近有 spiral arm 结构, 局部密度可能差几倍. 我们假设 box 周围 ±9 kpc 范围内 disk 还是 "差不多的". 对 first-order shielding 这够了, 对要 1% 精度的化学定量就不一定.

2. **没考虑 dust 跟 gas 比例的空间变化**. 我们假设 A_V/N_H = 4×10⁻²² cm²/mag 是个常数. 真实 MW 不同 R 跟不同 phase (CNM/WNM) 这个比有 ~30% 变化, 没建模.

3. **没考虑 self-shielding 的几何方向性**. 6 面 harmonic mean 是个简化 — 真实 CO/H₂ self-shielding 跟 line-of-sight 方向相关, 我们丢掉了角度信息. (做 PDR-style ray tracing 才能恢复这一点, 不在 quokka2s 范围内.)

4. **L_ext = 9 kpc 是个磋商选择**. 物理上 disk 实际半径 ~10-15 kpc, 我们的 9 kpc 是个保守估计 — 大致代表 "到 disk 外缘那一边". 也可以试 L_ext = 5 kpc (更窄) 或 12 kpc (更宽) 看 sensitivity.

5. **不影响 ±z 方向 colDen** — 假设盒子高度足够覆盖 disk 厚度. 对盒子高 8 kpc 而言成立, 对窄盒子 (例如高 < 1 kpc) 不成立.

---

## 8. 怎么验证 / 切换 L_ext

切换跑 L_ext = 9:

```python
# pipeline/prep/config.py
COLUMN_EXTENSION_LATERAL_KPC = 9.0
```

然后:

```bash
cd quokka2s/src
/opt/homebrew/Caskroom/miniconda/base/envs/yt-env/bin/python \
    -m quokka2s.pipeline.tasks.run_pipeline --mode all
```

注意:
- 输出会到 `output/{dataset}_down{N}_Lext9kpc/`
- 切回 `L_ext=0.0` 再跑一次, 输出到 `output/{dataset}_down{N}_Lext0kpc/`
- `TemperatureLextDiffTask` 和 `EmitterLextDiffTask` 会自动 cross-read 两个目录的 task intermediate 做差值图

直接看效果:
- `phase_colden.png` (两个目录对比) — `⟨log N_H⟩_M` 在每个 (ρ, T) bin 上的差
- `temperature_lext_diff.png` — 空间 slice 上的 T_DSP 比值
- `emitter_lext_diff.png` — 各发射线的 surface brightness 比值

---

## 9. 历史变更

- **2026-05-19**: 首版实现 — `L_ext × n_bar(z)` 加在 ±x/±y 的 N 上, 跟 6 面 harmonic mean 结合. 当时 `make_downsampled_dataset` 的 `force_periodicity()` 也是这个 PR 引入 (避免 covering_grid 边缘 ULP rounding 报错).
- **2026-05-19**: Cache schema 从 v2 → v3, `compute_cache_key` 加入 `column_extension_lateral_kpc` 维度. 旧 cache 全失效, 强制重算.
- **2026-05-20**: 出了一次 unit bug — `L_ext_cm` 当 float 跟 unyt 相乘单位算错. Fix: 改用 `(L_ext_kpc * kpc).in_units('cm')`.
- **2026-05-22**: 文档化 (本文件).
