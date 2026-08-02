# Cloudy HM2012 [C II] 158 μm 流程：从 Cloudy 到 QUOKKA 光谱

这份说明记录当前实际使用的流程，目标是在另一个目录或另一台机器上复现同一张 Cloudy [C II] 表，以及 QUOKKA 的低温/高温对比光谱。这里的“生产表”是本项目自己用 Cloudy 17.02 计算的线发射率表，不是 Grackle 官方发布的 cooling/heating HDF5 文件。

## 1. 最终在算什么

Cloudy 表的输入坐标是：

- `T`：QUOKKA cell 的温度，单位 K；
- `n_H`：氢核数密度，单位 cm⁻³；
- `N_H`：从模拟计算出的氢柱密度，单位 cm⁻²。

Cloudy 对每个 `(n_H, N_H, T)` 点输出 C II 157.636 μm 线在 slab 最深一个 zone 的局部体积发射率 `epsilon_CII`，单位 erg s⁻¹ cm⁻³。表里保存的是

```text
q_CII = epsilon_CII / n_H^2       [erg s^-1 cm^3]
```

运行时恢复为

```text
epsilon_CII = q_CII(T, n_H, N_H) * n_H^2.
```

这是一个局部 emissivity lookup，不是把整块 Cloudy slab 的所有 zone 积分成总 luminosity。`CIAOLoop_lines` 使用 Cloudy 输出的 `.lines` 最后一行，因此取的是最深 zone。

最终混合模型是：

```text
T_QUOKKA < 3000 K   -> DESPOTIC/GOW/LVG 表，输入 (n_H, N_H, dv/dr)
T_QUOKKA >= 3000 K  -> Cloudy HM2012 表，输入 (T_QUOKKA, n_H, N_H)
```

边界 `T_QUOKKA = 3000 K` 属于 Cloudy。Cloudy 分支没有 `dv/dr` 输入，也没有 LVG 或 turbulence 命令；速度只在后面的光谱合成阶段用于 Doppler shift。低温 Cloudy 表只是和 DESPOTIC 对比用，不改变生产混合模型。

## 2. 物理设定，不要暗中更改

生产参数文件是：

```text
work/cloudy_cooling_tools_history/examples/grackle/hm_2012_cii_cloudy_full.par
```

固定设定如下：

| 设定 | 当前值 |
|---|---|
| Cloudy | 17.02 |
| 辐射场 | `HM12_UVB/z_0.0000e+00.out`，项目中称 HM2012 z=0 |
| 谱线 | `C  2 157.636m` |
| 温度 | `constant temperature`；不是让 Cloudy 求热平衡温度 |
| 几何/柱密度 | 每个模型显式 `stop column density N_H` |
| Jeans length | 关闭，`coolingMapUseJeansLength = 0` |
| C/H | `1.6e-4`，即 `element carbon abundance -3.795880` |
| 其他金属 | `metals 0 log`，Cloudy solar-scale 默认值 |
| H2 network | `no H2 molecule` |
| charge transfer | `no charge transfer` |
| iteration | `iterate to convergence` |
| dust/grains | 没有加入命令 |
| cosmic rays | 没有加入命令 |
| microturbulence | 没有加入命令，即使用 Cloudy 默认值 |
| LVG / `dv/dr` | 不输入 Cloudy |

注意：本地的 `HM12_UVB/z_0.0000e+00.out` 第一行写的是 `Haardt & Madau (2011)`，但历史 Grackle/Cloudy 工具和本项目沿用 HM2012 命名。复现时应复制这一个确切文件或整个 `HM12_UVB` 目录，不能换成另一份同名 UV background。

QUOKKA 的 `N_H` 也不是 Jeans-length 近似。当前 pipeline 对每个 cell 分别从 ±x、±y、±z 积分 `n_H dl`，默认取六个方向柱密度的 harmonic mean；±x、±y 还加入 `L_ext=15 kpc` 的外部盘柱密度，±z 不加。必须保持：

```text
LEXT_KPC=15
COLDEN_MEAN=harmonic
```

否则 Cloudy 表本身虽不变，但每个 simulation cell 查询到的 `N_H` 会变，最终光谱也会变。

## 3. 必须带走的文件

至少复制这些文件；不要用原始 `CIAOLoop` 替代 `CIAOLoop_lines`：

```text
work/cloudy_cooling_tools_history/CIAOLoop_lines
work/cloudy_cooling_tools_history/examples/grackle/HM12_UVB/
work/cloudy_cooling_tools_history/examples/grackle/hm_2012_cii_cloudy_full.par
work/cloudy_cooling_tools_history/examples/grackle/hm_2012_cii_zero_sparse.par
work/cloudy_cooling_tools_history/examples/grackle/hm_2012_cii_lowT_sparse.par
scripts/analyze_cloudy_cii_zero_scan.py
scripts/sample_cloudy_cii_raw_failures.py
scripts/build_cloudy_cii_full_table.py
scripts/build_cloudy_cii_lowT_sparse_table.py
src/quokka2s/cloudy_cii_lookup.py
src/quokka2s/pipeline/prep/physics_fields.py
src/quokka2s/pipeline/tasks/cplus_high_model_comparison.py
src/quokka2s/pipeline/tasks/cplus_low_cloudy_comparison.py
scripts/plot_cplus_high_full31.py
```

`CIAOLoop_lines` 相对历史原版 `CIAOLoop` 的关键修正是：Cloudy 17.02 的 `save last lines, emissivity` 返回线性 emissivity；代码现在先除以实际 `hden^2`，再取 `log10`。原版把返回值当成已经取过 log 的数再减 `2 log10(n_H)`，对 Cloudy 17 是错的。确切计算是：

```perl
my $normalizedEmissivity = $line_emissivities[$q] / ($hden * $hden);

if ($normalizedEmissivity > 0.) {
    $line_emissivities[$q] = log($normalizedEmissivity) / log(10.);
}
else {
    $line_emissivities[$q] = -99.0;
}
```

这里 `-99` 只是“Cloudy 成功返回精确 0”的 sentinel。若 `.dat` 某一行只有温度、没有第二列，那是 Cloudy 失败，不是 0。

## 4. 新机器/新目录准备

以下命令都假设项目完整复制到了新位置：

```bash
export PROJECT=/path/to/quokka_postprocess_3D
export CLOUDY_SOURCE=/path/to/cloudy/c17.02/source

cd "$PROJECT"
conda activate quokka
python -m pip install -e .
```

当前参考环境是 Python 3.11.15、NumPy 2.2.6、SciPy 1.16.3、h5py 3.16.0、Matplotlib 3.10.6、yt 4.5.dev0。项目的 `requirements.txt` 记录了 Python 依赖，但 Cloudy 17.02 需要单独编译。

先确认 Cloudy 自己能正常运行：

```bash
cd "$CLOUDY_SOURCE"
./cloudy.exe < /path/to/a/known/test.in > /tmp/cloudy_reproduction_test.out
tail -n 5 /tmp/cloudy_reproduction_test.out
```

然后编辑以下三个 `.par` 文件中的绝对路径：

```text
cloudyExe = /path/to/cloudy/c17.02/source/cloudy.exe
outputDir = /new/project/.../对应的_output目录
```

要改的文件是 `hm_2012_cii_cloudy_full.par`、`hm_2012_cii_zero_sparse.par` 和 `hm_2012_cii_lowT_sparse.par`。检查：

```bash
cd "$PROJECT/work/cloudy_cooling_tools_history/examples/grackle"
grep -nE 'cloudyExe|outputDir' \
  hm_2012_cii_cloudy_full.par \
  hm_2012_cii_zero_sparse.par \
  hm_2012_cii_lowT_sparse.par
```

## 5. 可选：重新验证最高温度 `Tmax`

这一步用 5 个 `n_H` × 5 个 `N_H` × 29 个温度，从 3000 K 扫到 1e9 K。它只是决定生产表的高温边界；若目的是复现现有结果，可以直接采用已确认的 `Tmax=2.7289777828080403e6 K`，跳到下一节。

macOS、11 核：

```bash
cd "$PROJECT/work/cloudy_cooling_tools_history/examples/grackle"
caffeinate -dimsu ../../CIAOLoop_lines -np 11 hm_2012_cii_zero_sparse.par \
  2>&1 | tee hm_2012_cii_zero_sparse_11core.log

cd "$PROJECT"
conda run -n quokka python scripts/analyze_cloudy_cii_zero_scan.py
```

Linux 去掉 `caffeinate -dimsu`。已有扫描的 25 条曲线在 `log10(T/K)=6.436` 及更高温的可用节点全部为精确 0，直到 1e9 K 都没有重新出现正发射率；因此生产表把 `10^6.436 = 2.7289778e6 K` 留作最后一个零节点，超过它运行时返回 0。历史扫描在三个较低温节点有 Cloudy failure，所以分析器会严格报告 incomplete；这些 failure 不位于用于证明“6.436 以后一直为零”的高温区。若要做完全独立的审计，应把这三个失败节点补算后再要求分析器退出码为 0。

这不是随意截断：边界来自 Cloudy 的精确零输出。但它仍只是在上述 5×5 稀疏 `(n_H,N_H)` 上验证的物理假设，若改变辐射场、丰度或 Cloudy 版本，必须重新扫描。

## 6. 生成高温生产网格

生产网格是：

```text
16 n_H × 15 N_H × 31 T = 240 个 slab/run = 7440 次 Cloudy 调用
log10 n_H: -4.7142857 ... 6
log10 N_H: 15 ... 24
log10 T:   log10(3000) ... 6.436
Delta log10 T = 0.09862929 dex
```

运行：

```bash
cd "$PROJECT/work/cloudy_cooling_tools_history/examples/grackle"
caffeinate -dimsu ../../CIAOLoop_lines -np 11 hm_2012_cii_cloudy_full.par \
  2>&1 | tee hm_2012_cii_cloudy_full_11core_31T.log
```

这一步在本机运行了数小时；实际时间受 Cloudy 编译、CPU 和失败点等待时间影响很大。完成时必须看到：

```text
Run completed successfully
```

快速检查：

```bash
OUT="$PROJECT/work/cloudy_cooling_tools_history/examples/grackle/hm_2012_cii_cloudy_full_output"

find "$OUT" -name 'hm_2012_cii_cloudy_full_run*.dat' | wc -l
grep -c 'Cloudy crashed for T' \
  "$PROJECT/work/cloudy_cooling_tools_history/examples/grackle/hm_2012_cii_cloudy_full_11core_31T.log"

for i in $(seq 1 240); do
  f="$OUT/hm_2012_cii_cloudy_full_run${i}.dat"
  n=$(awk '!/^#/ {n++} END {print n+0}' "$f")
  if [ "$n" -ne 31 ]; then echo "incomplete run $i: $n/31 rows"; fi
done
```

参考结果是 240 个 `.dat`、每个 31 个温度位置；日志中有 56 个 Cloudy crash/missing 节点。注意：失败温度仍会写一行只有 `logT` 的记录，所以总行数 31 不代表 31 个 emissivity 都成功。

如果并行计算被中断，不要直接相信 `-r`：`.run` 文件记录的是已发出的 run，未必是已完成的 run。先找最早一个少于 31 个非注释行的 `.dat`；安全做法是从那里及其后的 run 重跑。最简单、最不容易污染结果的方法是在 `.par` 中换一个新的 `outputDir` 从头跑。若手动使用 `-r`，必须先备份并把 `.run` 的数字记录截到“最早不完整 run”的那一行；`-r` 会重跑该 run。不要对未检查的并行 `.run` 盲目恢复。

## 7. 先审计 failure，再允许构建表

当前 56 个失败节点没有被插值填补。是否能保留它们，必须先用目标 simulation 的全部 cell 检查其三线性插值 stencil 是否碰到 failure。

先确保 `column_density_H` cache 已由同一个 snapshot、`LEXT_KPC=15` 和 `COLDEN_MEAN=harmonic` 生成。然后：

```bash
cd "$PROJECT"
export YT_DATASET=/path/to/plt0655228
export LEXT_KPC=15
export COLDEN_MEAN=harmonic

MPLCONFIGDIR=/tmp/quokka-mpl conda run -n quokka \
python scripts/sample_cloudy_cii_raw_failures.py \
  --dataset "$YT_DATASET" \
  --column-cache "$PROJECT/intermediates/plt0655228/fields/field_gas_column_density_H.h5" \
  --output "$PROJECT/output/plt0655228_down1_Lext15kpc/cloudy_cii_full31_raw_failure_sampling.json"
```

参考 simulation 的结果：

```text
all cells                         134217728
T_QUOKKA >= 3000 K               133738699
inside Cloudy T range            111974042
T > 2.7289778e6 K                21764657
touches a failed Cloudy node     0
n_H outside table                0
N_H outside table                0
```

只有 `touches_failure_node = 0` 时，才能在下一步使用 `--allow-unused-failures`。若大于 0，必须补算报告列出的 touched nodes，不能把它们当作 0，也不能无条件插值。

## 8. 把原始 `.dat` 构建成 runtime NPZ

```bash
cd "$PROJECT"
conda run -n quokka python scripts/build_cloudy_cii_full_table.py \
  --allow-unused-failures
```

输出：

```text
data/cloudy_cii_hm2012_z0_full.npz
work/cloudy_cooling_tools_history/examples/grackle/hm_2012_cii_cloudy_full_failures.json
```

参考表应为：

```text
shape                 (16, 15, 31)
true Cloudy zero      356 nodes
unavailable failure    56 nodes
file size             about 78 KiB
```

表中 failed node 的 coefficient 数组位置只能用 0 作存储占位，但 `failure_mask=True`，它与 `zero_mask` 明确分离；运行时只要 failure 的三线性权重大于 `1e-12` 就抛错。因此 failure 没有被当成物理零。

插值规则：

- 八个角都为正：对 `log10(q_CII)` 做三线性插值；
- stencil 中包含 Cloudy 的真实零：对非负的线性 `q_CII` 做三线性插值，避免对零取 log；
- stencil 中包含 failure 且权重 `>1e-12`：报错；
- `n_H` 或 `N_H` 越界：报错，不 clamp；
- `T<3000 K`：生产 Cloudy lookup 报错，由 DESPOTIC 分支负责；
- `T>2.7289778e6 K`：根据高温零扫描返回 0。

验证表：

```bash
cd "$PROJECT"
conda run -n quokka python -c \
"import numpy as n; d=n.load('data/cloudy_cii_hm2012_z0_full.npz'); print(d['emissivity_per_nH2'].shape, int(d['zero_mask'].sum()), int(d['failure_mask'].sum()), d['out_of_bounds_policy'].item())"

conda run -n quokka python -m unittest tests.test_cloudy_cii_lookup
```

预期第一条命令打印 `(16, 15, 31) 356 56 temperature_above_max_zero; other_axes_raise`，测试全部通过。

## 9. 接入 QUOKKA 并生成高温光谱

设置所有输入，避免新机器继续读旧绝对路径或旧表：

```bash
cd "$PROJECT"
export QUOKKA_ROOT="$PROJECT"
export YT_DATASET=/path/to/plt0655228
export DESPOTIC_TABLE="$PROJECT/output_tables_3D_GOW_LVG/despotic_table_co10_co21_clean.npz"
export CLOUDY_CII_TABLE="$PROJECT/data/cloudy_cii_hm2012_z0_full.npz"
export LEXT_KPC=15
export COLDEN_MEAN=harmonic
```

计算 `T_QUOKKA >= 3000 K` 的模型比较 cache：

```bash
MPLCONFIGDIR=/tmp/quokka-mpl \
caffeinate -dimsu conda run -n quokka \
python -m quokka2s.pipeline.tasks.run_pipeline \
  --mode compute --force --task Build_CplusHighModelComparison \
  2>&1 | tee "$PROJECT/output/rebuild_cplus_high.log"
```

这个 task 会算 Saha、DESPOTIC 和 Cloudy 三条高温曲线；最终两模型图只读 DESPOTIC 和 Cloudy。当前 full-resolution `256×256×2048` snapshot 参考耗时约 19 分钟。

光谱的具体合成方法是：每个 cell 用 `epsilon × cell volume` 得到 luminosity；用 LOS=`y` 的 cell bulk velocity 做 Doppler shift；用 `T_QUOKKA` 和 C⁺ 质量 `12.01 amu` 算 thermal Gaussian width；在 ±50 km s⁻¹ 的 300 个 channel 上求和；最后除以 LOS 垂直平面的总投影面积。结果单位为 `Lsun pc^-2 (km s^-1)^-1`。`R=infinity`，不做仪器 LSF convolution。

## 10. 可选：低温 Cloudy 诊断表与低温光谱

这张表只用于画 `T_QUOKKA<3000 K` 的 Cloudy/DESPOTIC 对比。生产模型低温仍用 DESPOTIC。

```bash
cd "$PROJECT/work/cloudy_cooling_tools_history/examples/grackle"
caffeinate -dimsu ../../CIAOLoop_lines -np 11 hm_2012_cii_lowT_sparse.par \
  2>&1 | tee hm_2012_cii_lowT_sparse_11core.log

cd "$PROJECT"
conda run -n quokka python scripts/build_cloudy_cii_lowT_sparse_table.py
```

低温诊断表是 5×5×21，温度 10–3000 K；参考结果 failure=0、zero=0。然后：

```bash
export CLOUDY_CII_LOWT_DIAGNOSTIC_TABLE="$PROJECT/data/cloudy_cii_hm2012_z0_lowT_sparse_diagnostic.npz"

MPLCONFIGDIR=/tmp/quokka-mpl \
caffeinate -dimsu conda run -n quokka \
python -m quokka2s.pipeline.tasks.run_pipeline \
  --mode compute --force --task Build_CplusLowCloudyComparison \
  2>&1 | tee "$PROJECT/output/rebuild_cplus_low.log"
```

## 11. 画两张统一 y 轴的最终图

确认高温和低温 cache 都已经重新生成后运行：

```bash
cd "$PROJECT"
MPLCONFIGDIR=/tmp/quokka-mpl conda run -n quokka \
python scripts/plot_cplus_high_full31.py \
  --output-dir "$PROJECT/output/plt0655228_down1_Lext15kpc"
```

脚本会自动选择 `task_intermediates` 中最新的高温、低温 cache；若目录里有多组配置，最好显式指定，避免误读旧 cache：

```bash
MPLCONFIGDIR=/tmp/quokka-mpl conda run -n quokka \
python scripts/plot_cplus_high_full31.py \
  --output-dir "$PROJECT/output/plt0655228_down1_Lext15kpc" \
  --high-cache /exact/path/Build_CplusHighModelComparison_xxxxxxxx.h5 \
  --low-cache /exact/path/Build_CplusLowCloudyComparison_xxxxxxxx.h5
```

输出的两张统一线性 y 轴图是：

```text
Cplus_TQK_lt3000_DESPOTIC_Cloudy_Rinf_shared_ylim_simpletitle.png
Cplus_TQK_ge3000_DESPOTIC_Cloudy_full31_Rinf_shared_ylim_simpletitle.png
```

## 12. 当前参考文件校验值

在源目录运行：

```bash
shasum -a 256 \
  work/cloudy_cooling_tools_history/CIAOLoop_lines \
  work/cloudy_cooling_tools_history/examples/grackle/hm_2012_cii_cloudy_full.par \
  work/cloudy_cooling_tools_history/examples/grackle/HM12_UVB/z_0.0000e+00.out \
  data/cloudy_cii_hm2012_z0_full.npz \
  data/cloudy_cii_hm2012_z0_lowT_sparse_diagnostic.npz
```

本次记录的参考 SHA-256：

```text
825c53045d259e1a88825c07e69cba9581a9ad160dd959f1e01d67c1a87f5146  CIAOLoop_lines
7b6e43cd0094e7460a76463eaca7b5ee146939224fa285e4a94a19f71a68af09  hm_2012_cii_cloudy_full.par
3dfb92da2c9b331ebd103ee4ecdcb88d0a44ec3f218c5daff4624bcd456b43fb  z_0.0000e+00.out
6f204771905034d143dd8112276b217f073c12f8db87da78dca30edf64ae9c3a  cloudy_cii_hm2012_z0_full.npz
07130c5905113560c8aa86ad237d4a060dd215e6584ef1688826a3d9cbbd8f57  cloudy_cii_hm2012_z0_lowT_sparse_diagnostic.npz
```

`.par` 包含绝对路径，所以换机器后它的 checksum 必然改变；这不表示物理参数改变。判断结果是否一致，应同时检查表的 shape、zero/failure 数、轴数组、元数据和实际谱线结果。

## 13. 最短复现顺序

如果已有 Cloudy 17.02、同一个 UV 文件和同一个 simulation，实际顺序就是：

```text
改 .par 的 cloudyExe/outputDir
→ CIAOLoop_lines -np 11 生成 240 个高温 .dat
→ sample_cloudy_cii_raw_failures.py 检查目标 simulation 不碰 failure
→ build_cloudy_cii_full_table.py --allow-unused-failures 生成 NPZ
→ Build_CplusHighModelComparison 生成高温 spectrum cache
→ 运行低温诊断表及 Build_CplusLowCloudyComparison（仅在需要低温对比图时）
→ plot_cplus_high_full31.py 画统一 y 轴的两张图
```

最容易造成“看起来跑成功、结果却不同”的四个问题是：用了原始 `CIAOLoop`、换了 HM12 UV 文件、让 `N_H` 的 `LEXT_KPC/COLDEN_MEAN` 改变、或 plotting 读到了旧 task cache。复现时应优先检查这四项。
